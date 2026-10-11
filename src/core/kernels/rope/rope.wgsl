// QK-norm and RoPE, what both of rope's forms share (kernels/rope/rope.h):
// the launch's constants, the gains and factors, the queries, the angle
// tables and the steps. Each form's entry point is its own source, composed
// after this one by its launcher (rope.cpp): rope_cache.wgsl, which packs
// keys and values into the cache, and rope_encode.wgsl, which writes them
// back in place. For each token at position p and each head of d values:
//
//   QK-norm   x = (x × r) × g,  r = inverseSqrt(mean of x² + epsilon)
//   RoPE      pair k turns by p × θ_k:  x' = x cos − y sin,  y' = x sin + y cos
//
// Invocation i of a head holds its vec4s i and i + d / 8; a workgroup is
// workgroup_size / (d / 8) heads of one token.
//
// inverseSqrt, dot, round, cos, sin and workgroupBarrier are WGSL built-ins;
// reduce_head, angles and rotate are below.

struct Rope {
    slots: u32,
    epsilon: f32,
    // Each pair's θ_k / 2π, turns a position, four to a vec4: d / 2 used.
    turns: array<vec4<f32>, 64>,
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
override value_norm: bool = false;
// Gemma 4's vision encoder: each half of a head NEOX-paired within itself,
// the first turned by the patch's column and the second by its row (rope.h).
// The encode form's alone.
override axial: bool = false;
// d / 2, the pairs a head has, which size the workgroup's angle tables. Set
// on every launch, as WebKit requires of an override sizing a workgroup
// array (docs/research/2026-10-09-webkit-shader-compiler.md).
override pairs_per_head: u32 = 64u;

@group(0) @binding(1) var<uniform> rope: Rope;
@group(0) @binding(2) var<storage, read> query_gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> key_gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> factor: array<f32>;
@group(0) @binding(5) var<storage, read_write> query: array<vec4<f32>>;
const kTau = 6.283185307179586;

var<workgroup> partial: array<f32, workgroup_size>;
var<workgroup> cosines: array<f32, pairs_per_head>;
var<workgroup> sines: array<f32, pairs_per_head>;

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

// The workgroup's token: its row and position, and whether its heads
// rotate. A row is a whole number of workgroups (rope.h), so these are the
// same for every invocation of a workgroup — uniform, as the barriers after
// them require: from the workgroup's index alone.
struct Token {
    row: u32,
    p: u32,
    rotates: bool,
}

fn token_of(group: u32) -> Token {
    let lanes = head_dimension / 8u;
    let rotating_heads = query_heads + key_value_heads;
    let per_row = (rotating_heads + key_value_heads) * lanes;
    let first = group * workgroup_size;
    let row = first / per_row;
    return Token(row, step.position + row, (first % per_row) / lanes < rotating_heads);
}

// An invocation's head and lane, and the first vec4 of its head's row in
// its buffer.
struct Head {
    head: u32,
    lane: u32,
    is_query: bool,
    is_value: bool,
    kv_head: u32,
    base: u32,
}

fn head_of(group: u32, local: u32, row: u32) -> Head {
    let lanes = head_dimension / 8u;
    let quarter = head_dimension / 4u;              // vec4s in half a head
    let rotating_heads = query_heads + key_value_heads;
    let per_row = (rotating_heads + key_value_heads) * lanes;
    let index = group * workgroup_size + local;
    var h: Head;
    h.head = (index % per_row) / lanes;
    h.lane = index % lanes;
    h.is_query = h.head < query_heads;
    h.is_value = h.head >= rotating_heads;
    h.kv_head = h.head - query_heads;
    if (h.is_value) {
        h.kv_head = h.head - rotating_heads;
    }
    if (h.is_query) {
        h.base = (row * query_heads + h.head) * quarter;
    } else {
        h.base = (row * key_value_heads + h.kv_head) * quarter;
    }
    return h;
}

// The token's cos and sin for each pair, once, shared out among the
// workgroup; published by the next barrier.
fn angles(p: u32, local: u32) {
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

// The vec4s of its head invocation `lane` holds: lane and lane + d / 8 — a
// pair's two dimensions under Halves — or, axial, within half h = lane /
// (d / 16), vec4 j = lane mod (d / 16) of the half and the vec4 a quarter of
// the head after it: the half's NEOX pairs.
fn holds(lane: u32) -> vec2<u32> {
    let lanes = head_dimension / 8u;
    if (axial) {
        let per_half = lanes / 2u;
        let first = (lane / per_half) * lanes + lane % per_half;
        return vec2<u32>(first, first + per_half);
    }
    return vec2<u32>(lane, lane + lanes);
}

// Axial: the token's angles by its patch's column, for the first half's
// d / 4 pairs, then by its row, for the second's — x = row mod columns, y =
// row / columns — each half's frequencies the same d / 4 turns.
fn angles_axial(row: u32, local: u32) {
    let quarter = head_dimension / 4u;
    let x = row % step.columns;
    let y = row / step.columns;
    for (var k = local; k < head_dimension / 2u; k += workgroup_size) {
        let p = select(x, y, k >= quarter);
        let t = f32(p) * turn(k % quarter);
        let angle = kTau * (t - round(t));      // in [−π, π]
        cosines[k] = cos(angle);
        sines[k] = sin(angle);
    }
}

// Each pair of a query or key head turned by its angle. Under Halves, a and
// b hold four whole pairs (4i + c, 4i + c + d / 2); axial, four of a half's
// pairs, angles from the half's quarter of the table; under Adjacent, each
// holds two (2k, 2k + 1).
fn rotate(a: vec4<f32>, b: vec4<f32>, lane: u32) -> array<vec4<f32>, 2> {
    if (halves || axial) {
        let per_half = head_dimension / 16u;
        let k = select(4u * lane, (lane / per_half) * (head_dimension / 4u) + 4u * (lane % per_half), axial);
        let c = vec4<f32>(cosines[k], cosines[k + 1u], cosines[k + 2u], cosines[k + 3u]);
        let s = vec4<f32>(sines[k], sines[k + 1u], sines[k + 2u], sines[k + 3u]);
        return array<vec4<f32>, 2>(a * c - b * s, a * s + b * c);
    }
    let ka = 2u * lane;
    let kb = ka + head_dimension / 4u;
    let a0 = cs(ka);
    let a1 = cs(ka + 1u);
    let b0 = cs(kb);
    let b1 = cs(kb + 1u);
    return array<vec4<f32>, 2>(
        a * vec4<f32>(a0.x, a0.x, a1.x, a1.x) + vec4<f32>(-a.y, a.x, -a.w, a.z) * vec4<f32>(a0.y, a0.y, a1.y, a1.y),
        b * vec4<f32>(b0.x, b0.x, b1.x, b1.x) + vec4<f32>(-b.y, b.x, -b.w, b.z) * vec4<f32>(b0.y, b0.y, b1.y, b1.y));
}
