// Attention (kernels/attention/attention.h). Two entry points share this
// module: main, FlashAttention-2's online softmax over the layer's KV cache
// in 64-key chunks fixed by position; and combine, which folds a split
// step's chunks. Both fold chunks with the same functions, fold_factors,
// fold and finish, in chunk order, so a query's output is the same bits
// however its step was shaped.
//
// For a query q at position p:   s_j = (q · k_j) × scale,  j in p − W + 1 .. p
//                                out = Σ_j softmax(s)_j × v_j
// computed in log2 units — queries staged × scale × log2(e), exp2 for exp.
//
// unpack4 is the KV cache format's (format.h); exp2, max, select, dot and
// workgroupBarrier are WGSL built-ins.

struct Attention {
    slots: u32,
    window: u32,
    scale_log2e: f32,
}

// The program sets workgroup_size from the launcher (attention.cpp, 64) and
// the launcher the shape; the defaults are Qwen3's, so a tool that reads
// this file alone can size what uses them.
override workgroup_size: u32 = 64;
override head_dimension: u32 = 128;
override query_heads: u32 = 16;
override key_value_heads: u32 = 8;

// Derived from the shape when the pipeline is built; never set.
override queries_per_tile: u32 = 1024u / head_dimension;               // M
override keys_per_tile: u32 = 2048u / head_dimension;                  // B
override group: u32 = query_heads / key_value_heads;                   // G
override rows_per_tile: u32 = queries_per_tile / group;                // R
override lanes_per_query: u32 = workgroup_size / queries_per_tile;     // 64 / M

override partial_rows: u32 = 512u;   // the partial buffers' query rows

const kChunkKeys = 64u;

@group(0) @binding(1) var<uniform> attention: Attention;

// main's bindings.
@group(0) @binding(2) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> key_cache: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> value_cache: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> attended: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> partial_stats: array<vec2<f32>>;

// combine's: the same buffers, partials read-only — two entry points may
// share binding numbers.
@group(0) @binding(2) var<storage, read> c_partials: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> c_stats: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> c_output: array<vec4<f32>>;

// main's workgroup memory: the tile of queries, f32, M × d = 1,024; the
// tile of keys as stored, each row padded by a word against bank
// conflicts; the tile of values as stored, B × d / 2 = 1,024 words; the
// tile's scores, then weights; and each query's chunk statistics.
var<workgroup> q_tile: array<f32, 1024>;
var<workgroup> k_tile: array<u32, keys_per_tile * (head_dimension / 2u + 1u)>;
var<workgroup> v_tile: array<u32, 1024>;
var<workgroup> p_tile: array<f32, queries_per_tile * keys_per_tile>;
var<workgroup> chunk_live: array<u32, queries_per_tile>;
var<workgroup> chunk_m: array<f32, queries_per_tile>;
var<workgroup> chunk_l: array<f32, queries_per_tile>;
var<workgroup> alpha: array<f32, queries_per_tile>;

// The step's chunks for this layer, as kernels/interface.h's key_chunks:
// x the first, y the count, z the splits.
fn key_chunks() -> vec3<u32> {
    var earliest = 0u;
    if (step.position + 1u > attention.window) {
        earliest = step.position + 1u - attention.window;
    }
    let first = earliest / kChunkKeys;
    let count = (step.position + step.tokens - 1u) / kChunkKeys - first + 1u;
    var splits = 1u;
    if (step.tokens * count <= partial_rows) {
        splits = count;
    }
    return vec3<u32>(first, count, splits);
}

// Whether key position j is live for row r of the workgroup's tile, whose
// first row is at p_first: within the step's rows and the tile's keys,
// causal, and within the window. Recomputed wherever needed, so no score
// carries it: WGSL may assume no infinities, and no finite sentinel is
// below every score (GDSA.16).
fn is_live(r: u32, rows: u32, j: u32, k_end: u32, p_first: u32) -> bool {
    let p = p_first + r;
    return r < rows && j < k_end && j <= p && j + attention.window > p;
}

// Folding a chunk's state into a query's: the result is a × state + b ×
// chunk, for the sum and each output alike, where x holds a, y b, z the new
// maximum. An empty side contributes nothing: (1, 0) or (0, 1). A factor
// whose exponent is 0 is exactly 1.
fn fold_factors(live: bool, m: f32, chunk_is_live: bool, chunk_max: f32) -> vec3<f32> {
    if (!chunk_is_live) {
        return vec3<f32>(1.0, 0.0, m);
    }
    if (!live) {
        return vec3<f32>(0.0, 1.0, chunk_max);
    }
    let top = max(m, chunk_max);
    let a = select(exp2(m - top), 1.0, m == top);
    let b = select(exp2(chunk_max - top), 1.0, chunk_max == top);
    return vec3<f32>(a, b, top);
}

fn fold_sum(l: f32, a: f32, chunk_sum: f32, b: f32) -> f32 {
    return l * a + chunk_sum * b;
}

fn fold(o: vec4<f32>, a: f32, chunk_o: vec4<f32>, b: f32) -> vec4<f32> {
    return o * a + chunk_o * b;
}

fn finish(o: vec4<f32>, l: f32) -> vec4<f32> {
    return o / l;
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let d = head_dimension;
    let words = d / 2u;           // a key or value row, as stored
    let quads = d / 4u;           // vec4s a row; word pairs a row
    let chunks = key_chunks();
    let tiles = (step.tokens + rows_per_tile - 1u) / rows_per_tile;
    // Workgroup: key-value head, then row tile, then split. All uniform.
    let kv = wg.x % key_value_heads;
    let tile = (wg.x / key_value_heads) % tiles;
    let split = wg.x / (key_value_heads * tiles);
    let row0 = tile * rows_per_tile;
    let rows = min(rows_per_tile, step.tokens - row0);
    let p_first = step.position + row0;
    let p_last = p_first + rows - 1u;
    var earliest = 0u;
    if (p_first + 1u > attention.window) {
        earliest = p_first + 1u - attention.window;
    }
    // The chunks this workgroup computes: its split's, or every one its
    // rows reach — each of which then holds at least one tile.
    var c_begin = earliest / kChunkKeys;
    var c_end = p_last / kChunkKeys + 1u;
    if (chunks.z > 1u) {
        c_begin = chunks.x + split;
        c_end = c_begin + 1u;
    }

    // Stage the queries: query vector m is row m / G, head kv × G + m % G,
    // scaled into log2 units. Rows past the step are zeros, never live.
    for (var i = t; i < 256u; i += workgroup_size) {
        let qm = i / quads;
        let c = i % quads;
        let r = qm / group;
        var q = vec4<f32>(0.0);
        if (r < rows) {
            q = query[((row0 + r) * query_heads + kv * group + qm % group) * quads + c] * attention.scale_log2e;
        }
        let at = qm * d + 4u * c;
        q_tile[at] = q.x;
        q_tile[at + 1u] = q.y;
        q_tile[at + 2u] = q.z;
        q_tile[at + 3u] = q.w;
    }

    // This invocation's part of the output: query m, word pairs lane, lane
    // + 64 / M, … — four vec4s, 16 values. Query m's statistics belong to
    // its lane 0, which alone writes them: the softmax's, then the split's.
    let m = t / lanes_per_query;
    let lane = t % lanes_per_query;
    let owner = lane == 0u;
    let r = m / group;
    let head = kv * group + m % group;
    var o_run = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var o_chunk = o_run;
    // The fold so far, kept by every lane of the query alike, each folding
    // with the same functions the combine does — so no lane waits on another.
    var run_live = false;
    var run_m = 0.0;
    var run_l = 0.0;
    if (owner) {
        chunk_live[m] = 0u;
        chunk_m[m] = 0.0;
        chunk_l[m] = 0.0;
    }

    for (var c = c_begin; c < c_end; c++) {
        for (var k = 0u; k < 4u; k++) {
            o_chunk[k] = vec4<f32>(0.0);
        }
        // Tiles of B keys from the chunk's start, or the first tile holding
        // a live key, to the workgroup's last row.
        let k_begin = max(c * kChunkKeys, earliest / keys_per_tile * keys_per_tile);
        let k_end = min((c + 1u) * kChunkKeys, p_last + 1u);
        for (var j0 = k_begin; j0 < k_end; j0 += keys_per_tile) {
            workgroupBarrier();   // the last tile's phases, and the last chunk's fold, are done
            // A chunk's statistics start empty, after the barrier every lane
            // of the last chunk's fold has passed.
            if (owner && j0 == k_begin) {
                chunk_live[m] = 0u;
                chunk_m[m] = 0.0;
                chunk_l[m] = 0.0;
            }

            // Phase 1: the tile's keys and values from the KV cache, slot j
            // mod slots, a vec4 of words at a time.
            for (var i = t; i < keys_per_tile * (words / 4u); i += workgroup_size) {
                let b = i / (words / 4u);
                let w4 = i % (words / 4u);
                let j = j0 + b;
                var kw = vec4<u32>(0u);
                var vw = vec4<u32>(0u);
                if (j < k_end) {
                    let at = ((j % attention.slots) * key_value_heads + kv) * (words / 4u) + w4;
                    kw = key_cache[at];
                    vw = value_cache[at];
                }
                let ka = b * (words + 1u) + 4u * w4;
                k_tile[ka] = kw.x;
                k_tile[ka + 1u] = kw.y;
                k_tile[ka + 2u] = kw.z;
                k_tile[ka + 3u] = kw.w;
                let va = b * words + 4u * w4;
                v_tile[va] = vw.x;
                v_tile[va + 1u] = vw.y;
                v_tile[va + 2u] = vw.z;
                v_tile[va + 3u] = vw.w;
            }
            workgroupBarrier();

            // Phase 2: scores, invocation (m, b) a dot product of d; a key
            // not live for the query is never scored.
            for (var pair = t; pair < queries_per_tile * keys_per_tile; pair += workgroup_size) {
                let qm = pair / keys_per_tile;
                let b = pair % keys_per_tile;
                var s = 0.0;
                if (is_live(qm / group, rows, j0 + b, k_end, p_first)) {
                    for (var w = 0u; w < quads; w++) {
                        let ka = b * (words + 1u) + 2u * w;
                        let kk = unpack4(vec2<u32>(k_tile[ka], k_tile[ka + 1u]));
                        let qa = qm * d + 4u * w;
                        s += dot(vec4<f32>(q_tile[qa], q_tile[qa + 1u], q_tile[qa + 2u], q_tile[qa + 3u]), kk);
                    }
                }
                p_tile[pair] = s;
            }
            workgroupBarrier();

            // Phase 3: the online softmax, by each query's owner. Liveness is
            // the mask's, recomputed from positions, never read from a
            // score. A tile with no live key leaves the query's state as it
            // was.
            if (owner) {
                var top_tile = 0.0;
                var seen = false;
                for (var b = 0u; b < keys_per_tile; b++) {
                    if (is_live(r, rows, j0 + b, k_end, p_first)) {
                        let s = p_tile[m * keys_per_tile + b];
                        top_tile = select(s, max(top_tile, s), seen);
                        seen = true;
                    }
                }
                var a = 1.0;
                if (seen) {
                    var old = chunk_m[m];
                    if (chunk_live[m] == 0u) {
                        old = top_tile;
                        chunk_live[m] = 1u;
                    }
                    let top = max(old, top_tile);
                    a = select(exp2(old - top), 1.0, old == top);
                    var sum = 0.0;
                    for (var b = 0u; b < keys_per_tile; b++) {
                        var e = 0.0;
                        if (is_live(r, rows, j0 + b, k_end, p_first)) {
                            e = exp2(p_tile[m * keys_per_tile + b] - top);
                        }
                        p_tile[m * keys_per_tile + b] = e;
                        sum += e;
                    }
                    chunk_l[m] = chunk_l[m] * a + sum;
                    chunk_m[m] = top;
                } else {
                    for (var b = 0u; b < keys_per_tile; b++) {
                        p_tile[m * keys_per_tile + b] = 0.0;
                    }
                }
                alpha[m] = a;
            }
            workgroupBarrier();

            // Phase 4: O = O × α + P · V, this invocation's 16 outputs.
            let a = alpha[m];
            for (var k = 0u; k < 4u; k++) {
                let pair_at = 2u * (lane + k * lanes_per_query);
                var acc = o_chunk[k] * a;
                for (var b = 0u; b < keys_per_tile; b++) {
                    let va = b * words + pair_at;
                    acc += p_tile[m * keys_per_tile + b] * unpack4(vec2<u32>(v_tile[va], v_tile[va + 1u]));
                }
                o_chunk[k] = acc;
            }
        }
        if (chunks.z > 1u) {
            // Split: the chunk's unnormalized state, for the combine. Only
            // the owner reads the statistics, its own writes.
            if (r < rows) {
                let at = (split * step.tokens + row0 + r) * query_heads + head;
                for (var k = 0u; k < 4u; k++) {
                    partials[at * quads + lane + k * lanes_per_query] = o_chunk[k];
                }
                if (owner) {
                    partial_stats[at] = vec2<f32>(chunk_m[m], select(0.0, chunk_l[m], chunk_live[m] == 1u));
                }
            }
        } else {
            // Unsplit: folded here, in chunk order, as the combine folds.
            // An unsplit chunk has a tile, so its statistics were written
            // before that tile's third barrier and every lane reads them
            // final, with no barrier of their own.
            let live = chunk_live[m] == 1u;
            let f = fold_factors(run_live, run_m, live, chunk_m[m]);
            run_l = fold_sum(run_l, f.x, chunk_l[m], f.y);
            for (var k = 0u; k < 4u; k++) {
                o_run[k] = fold(o_run[k], f.x, o_chunk[k], f.y);
            }
            run_m = f.z;
            run_live = run_live || live;
        }
    }

    if (chunks.z == 1u && r < rows) {
        let at = (r + row0) * query_heads + head;
        for (var k = 0u; k < 4u; k++) {
            var out = vec4<f32>(0.0);
            if (run_live) {
                out = finish(o_run[k], run_l);
            }
            attended[at * quads + lane + k * lanes_per_query] = out;
        }
    }
}

// The combine: one invocation a vec4 of a query's output, folding the
// query's chunks in order and dividing.
@compute @workgroup_size(workgroup_size)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    let quads = head_dimension / 4u;
    let per_row = query_heads * quads;
    if (id.x >= step.tokens * per_row) {
        return;
    }
    let chunks = key_chunks();
    let row = id.x / per_row;
    let head = (id.x % per_row) / quads;
    let quad = id.x % quads;
    var live = false;
    var m = 0.0;
    var l = 0.0;
    var o = vec4<f32>(0.0);
    for (var s = 0u; s < chunks.z; s++) {
        let at = (s * step.tokens + row) * query_heads + head;
        let stats = c_stats[at];
        let chunk_is_live = stats.y > 0.0;
        let f = fold_factors(live, m, chunk_is_live, stats.x);
        l = fold_sum(l, f.x, stats.y, f.y);
        o = fold(o, f.x, c_partials[at * quads + quad], f.y);
        m = f.z;
        live = live || chunk_is_live;
    }
    var out = vec4<f32>(0.0);
    if (live) {
        out = finish(o, l);
    }
    c_output[(row * query_heads + head) * quads + quad] = out;
}
