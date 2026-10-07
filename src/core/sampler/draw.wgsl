// The draw (sampler/sampler.h): the next token from the candidates top-k
// selection kept, sorted largest first. Invocation i works candidate i; the
// last also runs Philox; invocation 0 alone takes the sums and running sums,
// in candidate order.
//
// Bindings: 1 the constants, 2 the candidates, 3 the sampled record.

struct Draw {
    count: u32,   // the candidates, kCandidates
}

override workgroup_size: u32;              // 64, one invocation a candidate
override last_token: bool = false;         // the draw covers the last token alone

@group(0) @binding(1) var<uniform> draw: Draw;
@group(0) @binding(2) var<storage, read> candidates: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> sampled: array<u32, 4>;

var<workgroup> weight: array<f32, 64>;        // top-p's, then its probabilities
var<workgroup> tempered: array<f32, 64>;      // at the turn's temperature
var<workgroup> above_min_p: array<u32, 64>;
var<workgroup> running: array<f32, 64>;       // tempered's running sums
var<workgroup> total: f32;
var<workgroup> u: f32;

// The high word of a × b, from 16-bit halves: WGSL has no 64-bit integer.
fn mulhi(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}

// Philox4x32-10 (Salmon et al., SC '11; Random123's constants).
fn philox(counter: vec4<u32>, key_in: vec2<u32>) -> vec4<u32> {
    var c = counter;
    var key = key_in;
    for (var r = 0u; r < 10u; r++) {
        let hi0 = mulhi(0xD2511F53u, c.x);
        let lo0 = 0xD2511F53u * c.x;
        let hi1 = mulhi(0xCD9E8D57u, c.z);
        let lo1 = 0xCD9E8D57u * c.z;
        c = vec4<u32>(hi1 ^ c.y ^ key.x, lo1, hi0 ^ c.w ^ key.y, lo0);
        key = key + vec2<u32>(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

// The top 23 bits scaled by 2⁻²³, plus 2⁻²⁴: in [2⁻²⁴, 1 − 2⁻²⁴], exact.
fn unit_interval(word: u32) -> f32 {
    return f32(word >> 9u) * (1.0 / 8388608.0) + (1.0 / 16777216.0);
}

// Whether f32 bits are a finite number, by their exponent: WGSL may assume
// a float is never infinite or NaN, so no float comparison can tell.
fn finite(bits: u32) -> bool {
    return (bits & 0x7F800000u) != 0x7F800000u;
}

fn negative_infinity(bits: u32) -> bool {
    return bits == 0xFF800000u;
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) t: u32) {
    let k = min(step.top_k, draw.count);
    // At a top_p of 1 top-p keeps every candidate, and none of its work is
    // done: no weight, total, division or running sum. The step's settings
    // are uniform, so every branch on them keeps the barriers uniform.
    let top_p = step.top_p < 1.0;
    let first_bits = candidates[0].x;
    let failed = !finite(first_bits);
    weight[t] = 0.0;
    tempered[t] = 0.0;
    above_min_p[t] = 0u;
    // A candidate past top_k is never read, so its invocation works none of
    // it. −∞ weighs nothing; a non-finite first fails the draw before any of
    // it is used.
    if (t < k && !failed) {
        let mine_bits = candidates[t].x;
        if (!negative_infinity(mine_bits)) {
            let first = bitcast<f32>(first_bits);
            let mine = bitcast<f32>(mine_bits);
            if (top_p) {
                weight[t] = exp(mine - first);
            }
            if (step.temperature > 0.0) {
                tempered[t] = exp((mine - first) / step.temperature);
            }
            above_min_p[t] = select(0u, 1u, step.min_p == 0.0 || mine >= first + step.log_min_p);
        }
    }
    if (t == workgroup_size - 1u) {
        // Counted by the position the drawn token takes, however the
        // tokens before it were stepped (sampler.h).
        u = unit_interval(philox(vec4<u32>(step.position + step.tokens, 0u, 0u, 0u), step.seed).x);
    }
    workgroupBarrier();
    if (top_p) {
        if (t == 0u) {
            var sum = 0.0;
            for (var i = 0u; i < k; i++) {
                sum = sum + weight[i];
            }
            total = sum;
        }
        workgroupBarrier();
        if (t < k && !failed) {
            weight[t] = weight[t] / total;   // the softmax at temperature 1
        }
        workgroupBarrier();
    }
    if (t != 0u) {
        return;
    }
    // top-p: below 1, the shortest prefix reaching top_p, one at least;
    // tempered's running sums carried beside it.
    var kept = k;
    var cumulative = 0.0;
    var tempered_sum = 0.0;
    for (var i = 0u; i < k; i++) {
        tempered_sum = tempered_sum + tempered[i];
        running[i] = tempered_sum;
        if (top_p) {
            cumulative = cumulative + weight[i];
            if (cumulative >= step.top_p) {
                kept = i + 1u;
                break;
            }
        }
    }
    // min-p: a prefix too, the candidates being sorted.
    var survivors = 1u;
    while (survivors < kept && above_min_p[survivors] == 1u) {
        survivors = survivors + 1u;
    }
    var chosen = 0u;
    if (step.temperature > 0.0) {
        let goal = u * running[survivors - 1u];
        chosen = survivors - 1u;   // should rounding leave none, the last survivor
        for (var i = 0u; i < survivors; i++) {
            if (running[i] > goal) {
                chosen = i;
                break;
            }
        }
    }
    sampled[0] = select(candidates[chosen].y, 0u, failed);
    sampled[1] = select(0u, 1u, failed);
    sampled[2] = bitcast<u32>(u);
    sampled[3] = 0u;
}
