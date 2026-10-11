// Block attention (kernels/block_attention/block_attention.h): every token
// of a step attending to every token of the same step within the layer's
// window either side. Two entry points:
//
//   scores  S = Q Kᵀ × scale × log2(e), written whole, and each 64-key
//           chunk's maximum for each query row;
//   weigh   out = Σ_j 2^(s_j − m) v_j / Σ_j 2^(s_j − m), m the row's maximum.
//
// A query vector is one query head of one row: a tile's 32 vectors are
// 32 / G rows × the G query heads sharing one key-value head, vector v of
// the tile row v / G and head kv × G + v % G. A key j is live for row i when
// j < N and |i − j| <= half_window; liveness is recomputed from positions
// wherever it is needed, never carried in a value (block_attention.h).
//
// exp2, max, dot and workgroupBarrier are WGSL built-ins.

struct BlockAttention {
    half_window: u32,
    scale_log2e: f32,
}

// The program sets workgroup_size from the launcher (block_attention.cpp,
// 128) and the launcher the shape, each a plain override set on every
// launch, as WebKit requires; the defaults are EmbeddingGemma 2's sliding
// layers', so a tool that reads this file alone can size what uses them.
override workgroup_size: u32 = 128;
override head_dimension: u32 = 256;
override query_heads: u32 = 4;
override key_value_heads: u32 = 2;
override group: u32 = 2;

const kVectors = 32u;      // query vectors a tile
const kChunk = 64u;        // keys a scores workgroup, and a chunk
const kStep = 32u;         // d a scores step; keys a weigh step
const kDims = 64u;         // output dimensions a weigh workgroup
const kLanes = 16u;        // invocations sharing 4 vectors
const kPad = 9u;           // a staged key's vec4s: 8 and one of padding

@group(0) @binding(1) var<uniform> attention: BlockAttention;

// scores' bindings.
@group(0) @binding(2) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;
@group(0) @binding(5) var<storage, read_write> score_maxima: array<f32>;

// weigh's: its own declarations, as two entry points may share binding
// numbers.
@group(0) @binding(2) var<storage, read> w_scores: array<f32>;
@group(0) @binding(3) var<storage, read> w_maxima: array<f32>;
@group(0) @binding(4) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> attended: array<vec4<f32>>;

// scores: a step's 32 query slabs of 32 floats, 8 vec4s a vector; then,
// once the products are done, each invocation's chunk maxima and whether
// each is live, two vec4s an invocation. 4 KiB.
var<workgroup> q_tile: array<vec4<f32>, 256>;
// scores: the chunk's 64 key slabs, 9 vec4s a key, the ninth padding so the
// 16 keys a read takes fall in distinct banks. 9 KiB.
var<workgroup> k_tile: array<vec4<f32>, 576>;

// weigh: a step's weights, 32 vectors × 32 keys, 8 vec4s a vector. 4 KiB.
var<workgroup> p_tile: array<vec4<f32>, 256>;
// weigh: the step's 32 values' 64 dimensions, 16 vec4s a key. 8 KiB.
var<workgroup> v_tile: array<vec4<f32>, 512>;
// weigh: each vector's row maximum.
var<workgroup> row_max: array<f32, kVectors>;

// Whether key j is live for row i. A step is at position 0 (block_attention.h),
// so rows are positions, and only how far apart two are matters.
fn is_live(i: u32, j: u32) -> bool {
    let n = step.tokens;
    let apart = select(j - i, i - j, i >= j);
    return i < n && j < n && apart <= attention.half_window;
}

// The query vector v of tile `tile` for key-value head kv: its row, and its
// query head.
fn row_of(tile: u32, v: u32) -> u32 {
    return (tile * kVectors + v) / group;
}

fn head_of(kv: u32, tile: u32, v: u32) -> u32 {
    return kv * group + (tile * kVectors + v) % group;
}

fn tiles() -> u32 {
    return (step.tokens * group + kVectors - 1u) / kVectors;
}

fn chunks() -> u32 {
    return (step.tokens + kChunk - 1u) / kChunk;
}

@compute @workgroup_size(workgroup_size)
fn scores_of(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let quads = head_dimension / 4u;
    // Workgroup: key-value head, then vector tile, then chunk. All uniform.
    let kv = wg.x % key_value_heads;
    let tile = (wg.x / key_value_heads) % tiles();
    let c = wg.x / (key_value_heads * tiles());
    let vq = t / kLanes;          // vectors 4 vq .. 4 vq + 3
    let lane = t % kLanes;        // keys lane, lane + 16, lane + 32, lane + 48 of the chunk

    var acc0 = vec4<f32>(0.0);    // vector 4 vq + 0, its 4 keys
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var d0 = 0u; d0 < head_dimension; d0 += kStep) {
        // Stage the tile's query slabs and the chunk's key slabs, 32 of d.
        for (var i = t; i < kVectors * 8u + kChunk * 8u; i += workgroup_size) {
            if (i < kVectors * 8u) {
                let v = i / 8u;
                let kq = i % 8u;
                let row = row_of(tile, v);
                var q = vec4<f32>(0.0);
                if (row < step.tokens) {
                    q = query[(row * query_heads + head_of(kv, tile, v)) * quads + d0 / 4u + kq];
                }
                q_tile[v * 8u + kq] = q;
            } else {
                let b = (i - kVectors * 8u) / 8u;
                let kq = (i - kVectors * 8u) % 8u;
                let j = c * kChunk + b;
                var k = vec4<f32>(0.0);
                if (j < step.tokens) {
                    k = key[(j * key_value_heads + kv) * quads + d0 / 4u + kq];
                }
                k_tile[b * kPad + kq] = k;
            }
        }
        workgroupBarrier();
        // 16 dot products of 32, an invocation's 4 vectors by its 4 keys,
        // each one chain along d in order.
        for (var kq = 0u; kq < 8u; kq++) {
            let k0 = k_tile[lane * kPad + kq];
            let k1 = k_tile[(lane + 16u) * kPad + kq];
            let k2 = k_tile[(lane + 32u) * kPad + kq];
            let k3 = k_tile[(lane + 48u) * kPad + kq];
            let q0 = q_tile[(4u * vq) * 8u + kq];
            let q1 = q_tile[(4u * vq + 1u) * 8u + kq];
            let q2 = q_tile[(4u * vq + 2u) * 8u + kq];
            let q3 = q_tile[(4u * vq + 3u) * 8u + kq];
            acc0 += vec4<f32>(dot(q0, k0), dot(q0, k1), dot(q0, k2), dot(q0, k3));
            acc1 += vec4<f32>(dot(q1, k0), dot(q1, k1), dot(q1, k2), dot(q1, k3));
            acc2 += vec4<f32>(dot(q2, k0), dot(q2, k1), dot(q2, k2), dot(q2, k3));
            acc3 += vec4<f32>(dot(q3, k0), dot(q3, k1), dot(q3, k2), dot(q3, k3));
        }
        workgroupBarrier();
    }

    // Write each live score, scaled into log2 units; a masked one 0.0, never
    // read as a score. Each vector's maximum over its live keys of the four.
    let n = step.tokens;
    var maxima = vec4<f32>(0.0);
    var live = vec4<f32>(0.0);
    for (var a = 0u; a < 4u; a++) {
        var acc = acc0;
        if (a == 1u) {
            acc = acc1;
        } else if (a == 2u) {
            acc = acc2;
        } else if (a == 3u) {
            acc = acc3;
        }
        let v = 4u * vq + a;
        let row = row_of(tile, v);
        let head = head_of(kv, tile, v);
        var top = 0.0;
        var seen = false;
        for (var b = 0u; b < 4u; b++) {
            let j = c * kChunk + lane + 16u * b;
            let s = acc[b] * attention.scale_log2e;
            let alive = is_live(row, j);
            if (row < n && j < n) {
                scores[(head * n + row) * n + j] = select(0.0, s, alive);
            }
            if (alive) {
                top = select(s, max(top, s), seen);
                seen = true;
            }
        }
        maxima[a] = top;
        live[a] = select(0.0, 1.0, seen);
    }
    // The products are done, so the query slabs are free: each invocation's
    // maxima and liveness, two whole vec4s, then one invocation a vector
    // folds its 16 lanes'. Max is exact, so the order does not matter.
    q_tile[2u * t] = vec4<f32>(maxima.x, live.x, maxima.y, live.y);
    q_tile[2u * t + 1u] = vec4<f32>(maxima.z, live.z, maxima.w, live.w);
    workgroupBarrier();
    if (t < kVectors) {
        let v = t;
        let row = row_of(tile, v);
        if (row < n) {
            var top = 0.0;
            var seen = false;
            for (var l = 0u; l < kLanes; l++) {
                let entry = q_tile[2u * ((v / 4u) * kLanes + l) + (v % 4u) / 2u];
                var m = entry.x;
                var alive = entry.y;
                if ((v % 2u) == 1u) {
                    m = entry.z;
                    alive = entry.w;
                }
                if (alive > 0.5) {
                    top = select(m, max(top, m), seen);
                    seen = true;
                }
            }
            score_maxima[(head_of(kv, tile, v) * n + row) * chunks() + c] = top;
        }
    }
}

// Whether chunk c holds a live key for row i: positions alone decide.
fn chunk_live(i: u32, c: u32) -> bool {
    let n = step.tokens;
    let first = c * kChunk;
    let last = min(first + kChunk, n) - 1u;
    var lo = 0u;
    if (i > attention.half_window) {
        lo = i - attention.half_window;
    }
    let hi = i + attention.half_window;
    return max(first, lo) <= min(last, hi);
}

@compute @workgroup_size(workgroup_size)
fn weigh(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let quads = head_dimension / 4u;
    let n = step.tokens;
    // Workgroup: key-value head, then vector tile, then 64 of d. All uniform.
    let kv = wg.x % key_value_heads;
    let tile = (wg.x / key_value_heads) % tiles();
    let dt = wg.x / (key_value_heads * tiles());
    let vq = t / kLanes;          // vectors 4 vq .. 4 vq + 3
    let lane = t % kLanes;        // dimensions 4 lane .. 4 lane + 3 of the 64

    // Each vector's row maximum, folded from its live chunks'.
    if (t < kVectors) {
        let row = row_of(tile, t);
        var top = 0.0;
        if (row < n) {
            var seen = false;
            for (var c = 0u; c < chunks(); c++) {
                if (chunk_live(row, c)) {
                    let m = w_maxima[(head_of(kv, tile, t) * n + row) * chunks() + c];
                    top = select(m, max(top, m), seen);
                    seen = true;
                }
            }
        }
        row_max[t] = top;
    }

    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    var l = vec4<f32>(0.0);       // each of the 4 vectors' sum of weights
    for (var j0 = 0u; j0 < n; j0 += kStep) {
        workgroupBarrier();       // row_max written; the last step's tiles read
        // Stage the step's weights, 2^(s − m), 0 for a key not live, and its
        // values' 64 dimensions.
        for (var i = t; i < kVectors * 8u + kStep * kLanes; i += workgroup_size) {
            if (i < kVectors * 8u) {
                let v = i / 8u;
                let kq = i % 8u;
                let row = row_of(tile, v);
                let head = head_of(kv, tile, v);
                var p = vec4<f32>(0.0);
                for (var e = 0u; e < 4u; e++) {
                    let j = j0 + 4u * kq + e;
                    if (is_live(row, j)) {
                        p[e] = exp2(w_scores[(head * n + row) * n + j] - row_max[v]);
                    }
                }
                p_tile[v * 8u + kq] = p;
            } else {
                let b = (i - kVectors * 8u) / kLanes;
                let q = (i - kVectors * 8u) % kLanes;
                let j = j0 + b;
                var x = vec4<f32>(0.0);
                if (j < n) {
                    x = value[(j * key_value_heads + kv) * quads + dt * (kDims / 4u) + q];
                }
                v_tile[b * kLanes + q] = x;
            }
        }
        workgroupBarrier();
        // O += p × v and l += p for each key in order, 4 vectors.
        for (var kq = 0u; kq < 8u; kq++) {
            let p0 = p_tile[(4u * vq) * 8u + kq];
            let p1 = p_tile[(4u * vq + 1u) * 8u + kq];
            let p2 = p_tile[(4u * vq + 2u) * 8u + kq];
            let p3 = p_tile[(4u * vq + 3u) * 8u + kq];
            for (var e = 0u; e < 4u; e++) {
                let x = v_tile[(4u * kq + e) * kLanes + lane];
                o0 = fma(vec4<f32>(p0[e]), x, o0);
                o1 = fma(vec4<f32>(p1[e]), x, o1);
                o2 = fma(vec4<f32>(p2[e]), x, o2);
                o3 = fma(vec4<f32>(p3[e]), x, o3);
                l = l + vec4<f32>(p0[e], p1[e], p2[e], p3[e]);
            }
        }
    }

    // O / l, at each vector's head's columns of its row. Every row below N
    // has a live key, itself, so l is never 0 there.
    for (var a = 0u; a < 4u; a++) {
        let v = 4u * vq + a;
        let row = row_of(tile, v);
        if (row < n) {
            var o = o0;
            if (a == 1u) {
                o = o1;
            } else if (a == 2u) {
                o = o2;
            } else if (a == 3u) {
                o = o3;
            }
            attended[(row * query_heads + head_of(kv, tile, v)) * quads + dt * (kDims / 4u) + lane] = o / l[a];
        }
    }
}
