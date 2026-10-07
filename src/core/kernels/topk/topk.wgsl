// Top-k selection (kernels/topk/topk.h): each workgroup keeps the top 64 of
// its tile of 1,024 entries by bitonic select, and writes them, sorted, as
// (logit bits, token) pairs. `first` reads logits, a token's id its index;
// `merge` reads the pairs a pass before it wrote.
//
// Bindings: 1 the constants, 2 the input, 3 the output.

struct TopK {
    count: u32,   // the entries this pass reads
}

override workgroup_size: u32;              // 256, the launcher's
override last_token: bool = false;         // every pass covers the last token alone

const kTile = 1024u;
const kRun = 64u;

@group(0) @binding(1) var<uniform> topk: TopK;
@group(0) @binding(2) var<storage, read> input: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<u32>>;

// The tile, as order-preserving keys and token ids: 8 KiB.
var<workgroup> keys: array<u32, 1024>;
var<workgroup> ids: array<u32, 1024>;

// A logit's f32 bits as a key whose unsigned order is the logits' order:
// −0 made +0, so the two zeros tie; every NaN the largest key, above +∞.
fn key_of(bits: u32) -> u32 {
    var b = bits;
    if (b == 0x80000000u) {
        b = 0u;
    }
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) {
        return 0xFFFFFFFFu;
    }
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

// The logit's bits back from its key: a NaN's as the canonical quiet NaN.
fn bits_of(key: u32) -> u32 {
    if (key == 0xFFFFFFFFu) {
        return 0x7FC00000u;
    }
    if ((key & 0x80000000u) != 0u) {
        return key & 0x7FFFFFFFu;
    }
    return ~key;
}

// Whether entry i comes before entry j: the larger logit, or of equal ones
// the lower token.
fn before(i: u32, j: u32) -> bool {
    return keys[i] > keys[j] || (keys[i] == keys[j] && ids[i] < ids[j]);
}

fn exchange(i: u32, j: u32) {
    let k = keys[i];
    let d = ids[i];
    keys[i] = keys[j];
    ids[i] = ids[j];
    keys[j] = k;
    ids[j] = d;
}

// Puts entries i and i + j in order: descending, or ascending when `up`.
fn order(i: u32, j: u32, up: bool) {
    if (before(i + j, i) != up) {
        exchange(i, i + j);
    }
}

// Every run of 64 sorted by a bitonic network, run r descending when r is
// even and ascending when odd, so each pair of runs is bitonic when joined.
fn sort_runs(t: u32) {
    for (var k = 2u; k <= kRun; k = k * 2u) {
        for (var j = k / 2u; j > 0u; j = j / 2u) {
            for (var p = 0u; p < 2u; p++) {
                let q = t + p * 256u;              // one of 512 pairs
                let i = (q / j) * 2u * j + q % j;
                order(i, j, (i & k) != 0u);
            }
            workgroupBarrier();
        }
    }
}

// Runs merged in pairs, 16 to 8 to 4 to 2 to 1. A descending run and the
// ascending one after it give, elementwise, their larger halves: a bitonic
// sequence holding the pair's top 64, sorted by 6 stages into the slot of
// the first, descending for an even output run and ascending for an odd.
fn merge_runs(t: u32) {
    for (var round = 1u; round <= 4u; round++) {
        let runs = 16u >> round;
        let stride = kRun << round;                // between output runs
        let half = kRun << (round - 1u);           // between a pair's runs
        for (var p = 0u; p < 2u; p++) {
            let q = t + p * 256u;
            if (q < runs * kRun) {
                let a = (q / kRun) * stride + q % kRun;
                if (before(a + half, a)) {
                    keys[a] = keys[a + half];
                    ids[a] = ids[a + half];
                }
            }
        }
        workgroupBarrier();
        for (var j = kRun / 2u; j > 0u; j = j / 2u) {
            if (t < runs * kRun / 2u) {
                let u = t / (kRun / 2u);           // the output run
                let q = t % (kRun / 2u);
                let i = u * stride + (q / j) * 2u * j + q % j;
                order(i, j, (u & 1u) != 0u);
            }
            workgroupBarrier();
        }
    }
}

fn select_and_write(wg: u32, t: u32) {
    sort_runs(t);
    merge_runs(t);
    if (t < kRun) {
        output[wg * kRun + t] = vec2<u32>(bits_of(keys[t]), ids[t]);
    }
}

@compute @workgroup_size(workgroup_size)
fn first(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    _ = step.tokens;   // binding 0 is every launch's (step.wgsl)
    for (var p = 0u; p < 4u; p++) {
        let s = t + p * 256u;
        let e = wg.x * kTile + s;
        // Past the row: below every logit, the last of any order.
        keys[s] = 0u;
        ids[s] = 0xFFFFFFFFu;
        if (e < topk.count) {
            keys[s] = key_of(input[e]);
            ids[s] = e;
        }
    }
    workgroupBarrier();
    select_and_write(wg.x, t);
}

@compute @workgroup_size(workgroup_size)
fn merge(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    _ = step.tokens;   // binding 0 is every launch's (step.wgsl)
    for (var p = 0u; p < 4u; p++) {
        let s = t + p * 256u;
        let e = wg.x * kTile + s;
        keys[s] = 0u;
        ids[s] = 0xFFFFFFFFu;
        // A slot the last pass padded — its tile held fewer than 64 — carries
        // token 0xFFFFFFFF and stays last: its logit bits are no logit's.
        if (e < topk.count && input[2u * e + 1u] != 0xFFFFFFFFu) {
            keys[s] = key_of(input[2u * e]);
            ids[s] = input[2u * e + 1u];
        }
    }
    workgroupBarrier();
    select_and_write(wg.x, t);
}
