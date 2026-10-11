// Block attention's streaming form (kernels/block_attention/block_attention.h),
// for a head of 64: every query vector of a step against every key of the
// same step within the layer's window either side, the scores never
// written — the online softmax, a running maximum m, sum l and output o
// carried across steps of 16 keys:
//
//   s_j = q · k_j × scale × log2(e)
//   out = Σ_j 2^(s_j − m) v_j / Σ_j 2^(s_j − m)
//
// A workgroup is one key-value head and a tile of 32 query vectors — 32 / G
// rows × the G query heads sharing that key-value head, vector v of the tile
// row v / G and head kv × G + v % G, as the written form's — its 128
// invocations 4 a vector: invocation t serves vector t / 4, lane c = t mod 4,
// scoring keys 4c .. 4c + 3 of each step and owning output dimensions
// 16c .. 16c + 15. A key j is live for row i when j < N and |i − j| <=
// half_window, recomputed from positions, never carried in a value; a
// vector whose step holds no live key changes nothing that step, its
// maximum kept with a flag that it has seen one, so no infinity is ever
// compared.
//
// exp2, max and workgroupBarrier are WGSL built-ins.

struct BlockAttention {
    half_window: u32,
    scale_log2e: f32,
}

// The program sets workgroup_size from the launcher (block_attention.cpp,
// 128) and the launcher the shape, each a plain override set on every
// launch, as WebKit requires; the defaults are EmbeddingGemma 2's vision
// encoder's.
override workgroup_size: u32 = 128;
override head_dimension: u32 = 64;
override query_heads: u32 = 12;
override key_value_heads: u32 = 12;
override group: u32 = 1;

const kVectors = 32u;      // query vectors a tile
const kKeys = 16u;         // keys a step
const kLanes = 4u;         // invocations a vector
const kQuads = 16u;        // vec4s of a head of 64
const kPad = 17u;          // a staged vector's vec4s: 16 and one of padding

@group(0) @binding(1) var<uniform> attention: BlockAttention;
@group(0) @binding(2) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> attended: array<vec4<f32>>;

// The tile's 32 query vectors, staged once. 8.5 KiB.
var<workgroup> q_tile: array<vec4<f32>, 544>;
// A step's 16 keys, then its 16 values, in one slot. 4.25 KiB.
var<workgroup> kv_tile: array<vec4<f32>, 272>;
// A step's scores, then their weights, 32 vectors × 16 keys. 2 KiB.
var<workgroup> w_tile: array<f32, 512>;

// Whether key j is live for row i. A step is at position 0
// (block_attention.h), so rows are positions, and only how far apart two
// are matters.
fn is_live(i: u32, j: u32) -> bool {
    let n = step.tokens;
    let apart = select(j - i, i - j, i >= j);
    return i < n && j < n && apart <= attention.half_window;
}

@compute @workgroup_size(workgroup_size)
fn stream(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = step.tokens;
    // Workgroup: key-value head, then vector tile. Uniform.
    let kv = wg.x % key_value_heads;
    let tile = wg.x / key_value_heads;
    let v = t / kLanes;
    let c = t % kLanes;
    let vector = tile * kVectors + v;
    let row = vector / group;
    let head = kv * group + vector % group;

    // Stage the tile's query vectors, four vec4s an invocation; a vector past
    // the step's rows is zeros, and is never written out.
    for (var i = t; i < kVectors * kQuads; i += workgroup_size) {
        let tv = tile * kVectors + i / kQuads;
        let r = tv / group;
        let h = kv * group + tv % group;
        var x = vec4<f32>(0.0);
        if (r < n) {
            x = query[(r * query_heads + h) * kQuads + i % kQuads];
        }
        q_tile[(i / kQuads) * kPad + i % kQuads] = x;
    }

    var m = 0.0;
    var seen = false;
    var l = 0.0;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    for (var j0 = 0u; j0 < n; j0 += kKeys) {
        // The last step's weights and values are read; the slot is free.
        workgroupBarrier();
        for (var i = t; i < kKeys * kQuads; i += workgroup_size) {
            let j = j0 + i / kQuads;
            var x = vec4<f32>(0.0);
            if (j < n) {
                x = key[(j * key_value_heads + kv) * kQuads + i % kQuads];
            }
            kv_tile[(i / kQuads) * kPad + i % kQuads] = x;
        }
        workgroupBarrier();

        // Scores: four keys of this vector, each a chain along d in order.
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var q4 = 0u; q4 < kQuads; q4++) {
            let qv = q_tile[v * kPad + q4];
            a0 = a0 + qv * kv_tile[(4u * c + 0u) * kPad + q4];
            a1 = a1 + qv * kv_tile[(4u * c + 1u) * kPad + q4];
            a2 = a2 + qv * kv_tile[(4u * c + 2u) * kPad + q4];
            a3 = a3 + qv * kv_tile[(4u * c + 3u) * kPad + q4];
        }
        let scale = attention.scale_log2e;
        w_tile[v * kKeys + 4u * c + 0u] = ((a0.x + a0.y) + (a0.z + a0.w)) * scale;
        w_tile[v * kKeys + 4u * c + 1u] = ((a1.x + a1.y) + (a1.z + a1.w)) * scale;
        w_tile[v * kKeys + 4u * c + 2u] = ((a2.x + a2.y) + (a2.z + a2.w)) * scale;
        w_tile[v * kKeys + 4u * c + 3u] = ((a3.x + a3.y) + (a3.z + a3.w)) * scale;
        workgroupBarrier();

        // The vector's maximum over the step's live keys, read in key order —
        // the same bits in each of its four lanes — and the running maximum.
        var top = 0.0;
        var live = false;
        for (var k = 0u; k < kKeys; k++) {
            if (is_live(row, j0 + k)) {
                let s = w_tile[v * kKeys + k];
                top = select(s, max(top, s), live);
                live = true;
            }
        }
        let next = select(m, select(top, max(m, top), seen), live);
        // Its own four weights, 0 for a key not live; read past the barrier.
        var p = vec4<f32>(0.0);
        if (is_live(row, j0 + 4u * c + 0u)) { p.x = exp2(w_tile[v * kKeys + 4u * c + 0u] - next); }
        if (is_live(row, j0 + 4u * c + 1u)) { p.y = exp2(w_tile[v * kKeys + 4u * c + 1u] - next); }
        if (is_live(row, j0 + 4u * c + 2u)) { p.z = exp2(w_tile[v * kKeys + 4u * c + 2u] - next); }
        if (is_live(row, j0 + 4u * c + 3u)) { p.w = exp2(w_tile[v * kKeys + 4u * c + 3u] - next); }
        // Every key read: the slot takes the step's values.
        for (var i = t; i < kKeys * kQuads; i += workgroup_size) {
            let j = j0 + i / kQuads;
            var x = vec4<f32>(0.0);
            if (j < n) {
                x = value[(j * key_value_heads + kv) * kQuads + i % kQuads];
            }
            kv_tile[(i / kQuads) * kPad + i % kQuads] = x;
        }
        // Every score read: the weights take their places.
        workgroupBarrier();
        w_tile[v * kKeys + 4u * c + 0u] = p.x;
        w_tile[v * kKeys + 4u * c + 1u] = p.y;
        w_tile[v * kKeys + 4u * c + 2u] = p.z;
        w_tile[v * kKeys + 4u * c + 3u] = p.w;
        workgroupBarrier();

        if (live) {
            // Rescaled where the maximum rose; 2^0 is exactly 1 where it did not.
            if (seen) {
                let alpha = exp2(m - next);
                l = l * alpha;
                o0 = o0 * alpha;
                o1 = o1 * alpha;
                o2 = o2 * alpha;
                o3 = o3 * alpha;
            }
            m = next;
            seen = true;
            // Every key in order: each lane of the vector the same l.
            for (var k = 0u; k < kKeys; k++) {
                let w = w_tile[v * kKeys + k];
                l = l + w;
                let base = k * kPad + 4u * c;
                o0 = o0 + w * kv_tile[base + 0u];
                o1 = o1 + w * kv_tile[base + 1u];
                o2 = o2 + w * kv_tile[base + 2u];
                o3 = o3 + w * kv_tile[base + 3u];
            }
        }
    }

    // Every row below N has a live key, itself, so l > 0.
    if (row < n) {
        let at = (row * query_heads + head) * kQuads + 4u * c;
        attended[at + 0u] = o0 / l;
        attended[at + 1u] = o1 / l;
        attended[at + 2u] = o2 / l;
        attended[at + 3u] = o3 / l;
    }
}
