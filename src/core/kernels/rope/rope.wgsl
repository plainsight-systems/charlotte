// QK-norm, RoPE and the cache append (kernels/rope/rope.h). For each token at
// position p and each head of d values:
//
//   QK-norm   x = (x × r) × g,  r = inverseSqrt(mean of x² + epsilon)
//   RoPE      pair k turns by p × θ_k:  x' = x cos − y sin,  y' = x sin + y cos
//
// Queries are written back in place; keys and values are packed into the
// layer's cache at slot p mod slots. Invocation i of a head holds its vec4s
// i and i + d / 8; a workgroup is 512 / d heads of one token.
//
// inverseSqrt, dot, round, cos, sin and workgroupBarrier are WGSL built-ins;
// pack is the cache format's (format.h); reduce_head is below.

struct Step {
    position: u32,
    tokens: u32,
    ids: array<vec4<u32>, 128>,
}

struct Rope {
    slots: u32,
    epsilon: f32,
    // Each pair's θ_k / 2π, turns a position, four to a vec4: d / 2 used.
    turns: array<vec4<f32>, 32>,
}

// The program sets workgroup_size from the launcher (rope.cpp, 64), and the
// launcher every other override; the defaults are Qwen3's, so a tool that
// reads this file alone can size what uses them.
override workgroup_size: u32 = 64;
override head_dimension: u32 = 128;
override query_heads: u32 = 16;
override key_value_heads: u32 = 8;
override halves: bool = true;
override qk_norm: bool = true;
override factors: bool = false;

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> rope: Rope;
@group(0) @binding(2) var<storage, read> query_gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> key_gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> factor: array<f32>;
@group(0) @binding(5) var<storage, read_write> query: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> key_cache: array<u32>;
@group(0) @binding(9) var<storage, read_write> value_cache: array<u32>;

const kTau = 6.283185307179586;
// Pairs a head has at most: d / 2 for d up to 256.
const kMaxPairs = 128u;

var<workgroup> partial: array<f32, workgroup_size>;
var<workgroup> cosines: array<f32, kMaxPairs>;
var<workgroup> sines: array<f32, kMaxPairs>;

// The sum of `value` over the `lanes` invocations of this invocation's head,
// in a fixed order: halving the head at each level. Every invocation of the
// head returns the same total. One barrier before the tree and one after
// each level; the first also publishes whatever the workgroup wrote before
// the call.
fn reduce_head(local: u32, lane: u32, lanes: u32, value: f32) -> f32 {
    partial[local] = value;
    workgroupBarrier();
    for (var stride = lanes / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
    }
    return partial[local - lane];
}

fn turn(k: u32) -> f32 {
    return rope.turns[k / 4u][k % 4u];
}

// Pair k's cosine and sine, from the workgroup's table.
fn cs(k: u32) -> vec2<f32> {
    return vec2<f32>(cosines[k], sines[k]);
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let lanes = head_dimension / 8u;
    let quarter = head_dimension / 4u;              // vec4s in half a head
    let rotating_heads = query_heads + key_value_heads;
    let per_row = (rotating_heads + key_value_heads) * lanes;
    // A row is a whole number of workgroups (rope.h), so the workgroup's
    // token and first head are the same for every invocation in it.
    let first = group.x * workgroup_size;
    let row = first / per_row;
    if (row >= step.tokens) {
        return;
    }
    let p = step.position + row;
    let rotates = (first % per_row) / lanes < rotating_heads;

    // Step 1, where any head of the workgroup rotates: the token's cos and
    // sin for each pair, once, shared out among the workgroup.
    if (rotates) {
        for (var k = local; k < head_dimension / 2u; k += workgroup_size) {
            var t = f32(p) * turn(k);
            if (factors) {
                t = t / factor[k];
            }
            let angle = kTau * (t - round(t));      // in [−π, π]
            cosines[k] = cos(angle);
            sines[k] = sin(angle);
        }
    }

    // Step 2: this invocation's two vec4s of its head.
    let at = first + local;
    let head = (at % per_row) / lanes;
    let lane = at % lanes;
    let is_query = head < query_heads;
    let is_value = head >= rotating_heads;
    var kv_head = head - query_heads;
    if (is_value) {
        kv_head = head - rotating_heads;
    }
    var base: u32;
    var a: vec4<f32>;
    var b: vec4<f32>;
    if (is_query) {
        base = (row * query_heads + head) * quarter;
        a = query[base + lane];
        b = query[base + lane + lanes];
    } else {
        base = (row * key_value_heads + kv_head) * quarter;
        if (is_value) {
            a = value[base + lane];
            b = value[base + lane + lanes];
        } else {
            a = key[base + lane];
            b = key[base + lane + lanes];
        }
    }

    // Step 3, with QK-norm, in a workgroup that rotates: r = 1 / sqrt(mean
    // of the head's squares + epsilon), then (x × r) × g. Value heads sharing
    // such a workgroup reach the barriers, as WGSL requires, and keep their
    // values; a workgroup of value heads alone reduces nothing. The tree's
    // first barrier publishes Step 1's table; without QK-norm a barrier of
    // its own does.
    if (qk_norm && rotates) {
        let total = reduce_head(local, lane, lanes, dot(a, a) + dot(b, b));
        if (!is_value) {
            let r = inverseSqrt(total / f32(head_dimension) + rope.epsilon);
            if (is_query) {
                a = (a * r) * query_gain[lane];
                b = (b * r) * query_gain[lane + lanes];
            } else {
                a = (a * r) * key_gain[lane];
                b = (b * r) * key_gain[lane + lanes];
            }
        }
    } else if (rotates) {
        workgroupBarrier();
    }

    // Step 4, query and key heads: each pair turned by its angle. Under
    // Halves, a and b hold four whole pairs (4i + c, 4i + c + d / 2); under
    // Adjacent, each holds two (2k, 2k + 1).
    if (!is_value) {
        if (halves) {
            let k = 4u * lane;
            let c = vec4<f32>(cosines[k], cosines[k + 1u], cosines[k + 2u], cosines[k + 3u]);
            let s = vec4<f32>(sines[k], sines[k + 1u], sines[k + 2u], sines[k + 3u]);
            let x = a;
            a = x * c - b * s;
            b = x * s + b * c;
        } else {
            let ka = 2u * lane;
            let kb = ka + head_dimension / 4u;
            let a0 = cs(ka);
            let a1 = cs(ka + 1u);
            let b0 = cs(kb);
            let b1 = cs(kb + 1u);
            a = a * vec4<f32>(a0.x, a0.x, a1.x, a1.x) + vec4<f32>(-a.y, a.x, -a.w, a.z) * vec4<f32>(a0.y, a0.y, a1.y, a1.y);
            b = b * vec4<f32>(b0.x, b0.x, b1.x, b1.x) + vec4<f32>(-b.y, b.x, -b.w, b.z) * vec4<f32>(b0.y, b0.y, b1.y, b1.y);
        }
    }

    // Step 5: queries back in place, f32; keys and values packed into the
    // cache at slot p mod slots — four values to two words.
    if (is_query) {
        query[base + lane] = a;
        query[base + lane + lanes] = b;
    } else {
        let word = ((p % rope.slots) * key_value_heads + kv_head) * (head_dimension / 2u) + 2u * lane;
        let pa = pack(a);
        let pb = pack(b);
        if (is_value) {
            value_cache[word] = pa.x;
            value_cache[word + 1u] = pa.y;
            value_cache[word + quarter] = pb.x;
            value_cache[word + quarter + 1u] = pb.y;
        } else {
            key_cache[word] = pa.x;
            key_cache[word + 1u] = pa.y;
            key_cache[word + quarter] = pb.x;
            key_cache[word + quarter + 1u] = pb.y;
        }
    }
}
