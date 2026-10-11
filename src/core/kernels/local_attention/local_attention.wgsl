// An audio encoder's local attention (kernels/local_attention/
// local_attention.h): each row's query against the keys of its window, its
// own and the w − 1 rows before it, with a relative position term for each
// distance, the scores soft-capped, then a softmax and the values' weighted
// sum. For row r, head h, distance j:
//
//   q̂   = (q[r] × q_scale) × s          s the layer's per_dim_scale
//   k̂   = k[r − j] × k_scale
//   s_j = tanh((q̂ · k̂ + q̂ · P[w − j]) × (1 / c)) × c
//   p_j = exp(s_j − max s) / Σ exp(s − max s)
//   out = Σ_j p_j v[r − j]
//
// over the j with r − j ≥ 0: llama.cpp's gemma4a blocked form, its
// ggml_scale, ggml_tanh, ggml_scale and its mask, which leaves exactly
// these.
//
// fma, tanh, clamp, exp, max, min and workgroupBarrier are WGSL built-ins.

struct LocalAttention {
    q_scale: f32,
    k_scale: f32,
    softcap: f32,
    window: u32,
}

// The program sets workgroup_size from the launcher (local_attention.cpp,
// 128); the default only lets a tool that reads this file alone size what
// uses it.
override workgroup_size: u32 = 128;

@group(0) @binding(1) var<uniform> attention: LocalAttention;
@group(0) @binding(2) var<storage, read> per_dim_scale: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> relative: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

const kHeads = 8u;
const kHeadVec4s = 32u;    // d = 128
const kRowVec4s = 256u;    // H × d = 1,024
const kLanes = 16u;        // a head's invocations; the window is at most this

// Each head's window of scores, then of weights, lane j's at h × 16 + j.
var<workgroup> scores: array<f32, kHeads * kLanes>;

// The sum of a · b over a head's 128 dimensions, i in order, an fma a term.
fn dot_in_order(sum: f32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, sum))));
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) n: u32) {
    let r = group.x;
    // The whole workgroup leaves here or none of it does: its barriers below
    // are reached by all of it.
    if (r >= step.tokens) {
        return;
    }
    let h = n / kLanes;
    let lane = n % kLanes;
    let head = r * kRowVec4s + h * kHeadVec4s;
    // The distances this row has: its own key and those of the rows before
    // it, at most the window.
    let count = min(attention.window, r + 1u);

    // Step 1: lane j's score, distance j: q̂ · k̂ and q̂ · P[w − j], taken
    // apart and added, as llama.cpp adds its two score matrices; then the cap.
    if (lane < count) {
        let j = lane;
        let keys = (r - j) * kRowVec4s + h * kHeadVec4s;
        let positions = (attention.window - j) * kRowVec4s + h * kHeadVec4s;
        var content = 0.0;
        var position = 0.0;
        for (var i = 0u; i < kHeadVec4s; i++) {
            let q = (query[head + i] * attention.q_scale) * per_dim_scale[i];
            content = dot_in_order(content, q, key[keys + i] * attention.k_scale);
            position = dot_in_order(position, q, relative[positions + i]);
        }
        // WGSL states tanh's accuracy as that of sinh(x) / cosh(x), a form
        // that is NaN past |x| of about 89; the argument is held to ±10,
        // past which tanh is ±1 in f32 already, so the cap is exact there.
        let capped = clamp((content + position) * (1.0 / attention.softcap), -10.0, 10.0);
        scores[h * kLanes + j] = tanh(capped) * attention.softcap;
    }
    workgroupBarrier();

    // Step 2: lane 0 of each head turns its scores into weights: the max,
    // the exponentials and their sum in j order, once a head.
    if (lane == 0u) {
        let base = h * kLanes;
        var largest = scores[base];
        for (var j = 1u; j < count; j++) {
            largest = max(largest, scores[base + j]);
        }
        var total = 0.0;
        for (var j = 0u; j < count; j++) {
            let e = exp(scores[base + j] - largest);
            scores[base + j] = e;
            total = total + e;
        }
        let inverse = 1.0 / total;
        for (var j = 0u; j < count; j++) {
            scores[base + j] = scores[base + j] * inverse;
        }
    }
    workgroupBarrier();

    // Step 3: lane l writes dimensions 8l to 8l + 7 of its head: Σ_j p_j
    // v[r − j], j in order.
    let at = h * kHeadVec4s + 2u * lane;
    var low = vec4<f32>(0.0);
    var high = vec4<f32>(0.0);
    for (var j = 0u; j < count; j++) {
        let p = scores[h * kLanes + j];
        let values = (r - j) * kRowVec4s + at;
        low = fma(vec4<f32>(p), value[values], low);
        high = fma(vec4<f32>(p), value[values + 1u], high);
    }
    output[r * kRowVec4s + at] = low;
    output[r * kRowVec4s + at + 1u] = high;
}
