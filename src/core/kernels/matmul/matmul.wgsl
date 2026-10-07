// The matrix products (kernels/matmul/matmul.h): y[t][o] = Σ_k W[o][k] ×
// x[t][k], the weight decoded in the load path by its format's unpack.
//
// Both forms add an output's products in one order, fixed by K alone: a
// row's G = K / 32 groups cut into 32 ranges at fixed boundaries, each
// range summed from zero in order with mac, and the ranges added to a total
// that starts at zero, in order. Decode gives each range to an invocation;
// prefill walks them in turn, keeping each output's range sum and total.
//
// Entry points, one a form and epilogue, each binding only what it writes:
//   decode_write, decode_qkv, decode_gated, prefill_write, prefill_qkv,
//   prefill_gated.
//
// A product's rows are its members' rows in order — a weight piece, or a
// fused group's Q, K, V or gate, up — each member read from its own first
// word in the binding.
//
// unpack is the weight format's (format.h); exp, tanh-free activations and
// workgroupBarrier as below are WGSL built-ins.

struct Step {
    position: u32,
    tokens: u32,
    ids: array<vec4<u32>, 128>,
}

// Each member: x its first word in the binding, y its blocks, z its first
// output row, w its rows.
struct Matmul {
    members: array<vec4<u32>, 3>,
}

// The program sets workgroup_size and last_token; the launcher the rest.
// The defaults are Qwen3's QKV, so a tool reading this file alone can size
// what uses them.
override workgroup_size: u32 = 64;
override last_token: bool = false;
override columns: u32 = 1024;        // K
override out_width: u32 = 1024;      // a write's output row width
override activation: u32 = 0;       // 0 SiLU, 1 GELU in its tanh form

const kRanges = 32u;

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> matmul: Matmul;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

// Decode: each of 8 row slots' 32 range sums.
var<workgroup> partials: array<f32, 256>;
// Prefill: the step's 32 × 32 input floats and the tile's 64 rows' 32
// decoded weights, each row padded a word against bank conflicts.
var<workgroup> x_tile: array<f32, 32u * 33u>;
var<workgroup> w_tile: array<f32, 64u * 33u>;

// The one multiply-add both forms use: an explicit fma, which the target's
// compiler fuses in every pipeline alike, where acc + w × x was contracted
// in one pipeline and not the other.
fn mac(acc: f32, w: f32, x: f32) -> f32 {
    return fma(w, x, acc);
}

fn groups() -> u32 {
    return columns / 32u;
}

// Range r's first group; range r is groups range_lo(r) .. range_lo(r + 1).
fn range_lo(r: u32) -> u32 {
    return r * groups() / kRanges;
}

// The member holding the product's row `r`, and the row within it: x the
// member, y its row, z 1 when r is a row of the product.
fn locate(r: u32) -> vec3<u32> {
    var rest = r;
    for (var m = 0u; m < 3u; m++) {
        let rows = matmul.members[m].w;
        if (rest < rows) {
            return vec3<u32>(m, rest, 1u);
        }
        rest -= rows;
    }
    return vec3<u32>(0u, 0u, 0u);
}

// The 32 weights of group g of member m's row `row`.
fn weights_of(m: u32, row: u32, g: u32) -> array<vec4<f32>, 8> {
    let member = matmul.members[m];
    return unpack(member.x, member.y, row * groups() + g);
}

fn token_of_decode() -> u32 {
    return select(0u, step.tokens - 1u, last_token);
}

// e^x for x <= 0, as explicit arithmetic rather than the built-in exp,
// which the target's compiler evaluated differently in the decode and
// prefill pipelines: x = k ln 2 + r, ln 2 in two parts (Cody and Waite), so
// r is near exact; e^r by its Taylor series to r⁶ in Horner's form, within
// 2⁻²² for |r| <= ln 2 / 2; then 2^k exactly by ldexp. Every multiply-add
// is an fma.
fn exp_nonpositive(x: f32) -> f32 {
    let xc = max(x, -100.0);
    let k = floor(fma(xc, 1.4426950408889634, 0.5));
    var r = fma(k, -0.693145751953125, xc);
    r = fma(k, -1.4286068203094172e-06, r);
    var p = 1.0 / 720.0;
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    return ldexp(p, i32(k));
}

// SiLU, x × sigmoid(x), and GELU's tanh form, each from e^x of a
// non-positive x, so no intermediate overflows: WGSL may assume no
// infinities. Every multiply-add is an explicit fma and each quotient one
// reciprocal, leaving the compiler nothing to contract differently in the
// two pipelines.
fn activate(g: f32) -> f32 {
    if (activation == 0u) {
        let e = exp_nonpositive(-abs(g));
        let r = 1.0 / (1.0 + e);
        return g * select(e * r, r, g >= 0.0);
    }
    let z = 0.7978845608028654 * fma(0.044715 * g * g, g, g);
    let e = exp_nonpositive(-2.0 * abs(z));
    let t = sign(z) * ((1.0 - e) * (1.0 / (1.0 + e)));
    let half_g = 0.5 * g;
    return fma(half_g, t, half_g);
}

// ---- Decode ----------------------------------------------------------------

// Invocation `lane`'s range of K for four rows, against the step's one
// input row: the input's floats loaded once a group, used for all four.
// `rows` holds each slot's member and row, `live` whether it is a row.
fn decode_ranges(lane: u32, members: vec4<u32>, rows: vec4<u32>, live: vec4<bool>) -> vec4<f32> {
    let token = token_of_decode();
    var acc = vec4<f32>(0.0);
    for (var g = range_lo(lane); g < range_lo(lane + 1u); g++) {
        var x: array<vec4<f32>, 8>;
        for (var v = 0u; v < 8u; v++) {
            x[v] = input[token * (columns / 4u) + g * 8u + v];
        }
        for (var j = 0u; j < 4u; j++) {
            if (live[j]) {
                let w = weights_of(members[j], rows[j], g);
                var a = acc[j];
                for (var v = 0u; v < 8u; v++) {
                    for (var c = 0u; c < 4u; c++) {
                        a = mac(a, w[v][c], x[v][c]);
                    }
                }
                acc[j] = a;
            }
        }
    }
    return acc;
}

// Slot s's total: its 32 range sums added in order to a total from zero.
fn slot_total(s: u32) -> f32 {
    var total = 0.0;
    for (var r = 0u; r < kRanges; r++) {
        total = total + partials[s * kRanges + r];
    }
    return total;
}

// A workgroup's 8 rows of the product, slots team × 4 + j: invocation lane
// of a team sums its range of the team's 4 rows; after one barrier the caller
// adds each slot's ranges.
fn decode_rows(wg: u32, t: u32) {
    let team = t / 32u;
    let lane = t % 32u;
    var members = vec4<u32>(0u);
    var rows = vec4<u32>(0u);
    var live = vec4<bool>(false);
    for (var j = 0u; j < 4u; j++) {
        let at = locate(wg * 8u + team * 4u + j);
        members[j] = at.x;
        rows[j] = at.y;
        live[j] = at.z == 1u;
    }
    let sums = decode_ranges(lane, members, rows, live);
    for (var j = 0u; j < 4u; j++) {
        partials[(team * 4u + j) * kRanges + lane] = sums[j];
    }
    workgroupBarrier();
}

@compute @workgroup_size(workgroup_size)
fn decode_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    decode_rows(wg.x, t);
    if (t < 8u) {
        let at = locate(wg.x * 8u + t);
        if (at.z == 1u) {
            out0[matmul.members[at.x].z + at.y] = slot_total(t);
        }
    }
}

@compute @workgroup_size(workgroup_size)
fn decode_qkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    decode_rows(wg.x, t);
    if (t < 8u) {
        let at = locate(wg.x * 8u + t);
        if (at.z == 1u) {
            let total = slot_total(t);
            if (at.x == 0u) {
                out0[at.y] = total;
            } else if (at.x == 1u) {
                out1[at.y] = total;
            } else {
                out2[at.y] = total;
            }
        }
    }
}

// Gated: a team's 4 slots are gate rows p, p + 1 and up rows p, p + 1, so
// invocations 0 to 3 each hold a gate row and its up row.
@compute @workgroup_size(workgroup_size)
fn decode_gated(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let team = t / 32u;
    let lane = t % 32u;
    let p = wg.x * 4u + team * 2u;
    let width = matmul.members[0].w;
    let live = vec4<bool>(p < width, p + 1u < width, p < width, p + 1u < width);
    let sums = decode_ranges(lane, vec4<u32>(0u, 0u, 1u, 1u), vec4<u32>(p, p + 1u, p, p + 1u), live);
    for (var j = 0u; j < 4u; j++) {
        partials[(team * 4u + j) * kRanges + lane] = sums[j];
    }
    workgroupBarrier();
    if (t < 4u) {
        let s = t / 2u;
        let k = t % 2u;
        let row = wg.x * 4u + s * 2u + k;
        if (row < width) {
            let gate = slot_total(s * 4u + k);
            let up = slot_total(s * 4u + 2u + k);
            out0[row] = activate(gate) * up;
        }
    }
}

// ---- Prefill ---------------------------------------------------------------

// Tile row i's member and row, as decode's slots: a gated tile's rows 0 to
// 31 are gate rows and 32 to 63 the same up rows.
fn tile_row(out_tile: u32, i: u32, gated: bool) -> vec3<u32> {
    if (gated) {
        let row = out_tile * 32u + i % 32u;
        return vec3<u32>(i / 32u, row, select(0u, 1u, row < matmul.members[0].w));
    }
    return locate(out_tile * 64u + i);
}

// An invocation's micro-tile: tokens 4 × (t % 8) .. + 3 and 8 outputs; for
// a gated tile, gate outputs 4 × (t / 8) .. + 3 and the same up outputs.
fn micro_row(t: u32, j: u32, gated: bool) -> u32 {
    let c = t / 8u;
    if (gated) {
        return select(32u + c * 4u + j - 4u, c * 4u + j, j < 4u);
    }
    return c * 8u + j;
}

// The tile's totals, each output's ranges summed from zero and added to its
// total in order, exactly as decode adds them.
fn prefill_tile(wg: u32, t: u32, gated: bool) -> array<f32, 32> {
    let token_tiles = (step.tokens + 31u) / 32u;
    let token_tile = wg % token_tiles;
    let out_tile = wg / token_tiles;
    let tr = t % 8u;
    let mine = tile_row(out_tile, t, gated);
    var range_sum: array<f32, 32>;
    var total: array<f32, 32>;
    for (var i = 0u; i < 32u; i++) {
        range_sum[i] = 0.0;
        total[i] = 0.0;
    }
    for (var r = 0u; r < kRanges; r++) {
        for (var g = range_lo(r); g < range_lo(r + 1u); g++) {
            workgroupBarrier();   // the last step's tiles are read
            // The step's input floats, 4 vec4s an invocation.
            for (var q = 0u; q < 4u; q++) {
                let i = t + q * 64u;
                let tok = i / 8u;
                let v = i % 8u;
                let token = token_tile * 32u + tok;
                var x = vec4<f32>(0.0);
                if (token < step.tokens) {
                    x = input[token * (columns / 4u) + g * 8u + v];
                }
                let at = tok * 33u + 4u * v;
                x_tile[at] = x.x;
                x_tile[at + 1u] = x.y;
                x_tile[at + 2u] = x.z;
                x_tile[at + 3u] = x.w;
            }
            // Row t of the tile's weights, decoded.
            var w = array<vec4<f32>, 8>();
            if (mine.z == 1u) {
                w = weights_of(mine.x, mine.y, g);
            }
            for (var v = 0u; v < 8u; v++) {
                for (var c = 0u; c < 4u; c++) {
                    w_tile[t * 33u + 4u * v + c] = w[v][c];
                }
            }
            workgroupBarrier();
            // Each k: the micro-tile's 4 inputs and 8 weights read from
            // workgroup memory once, into registers, then its 32 multiply-
            // adds — 12 reads, not one a product.
            for (var k = 0u; k < 32u; k++) {
                var xs: array<f32, 4>;
                var ws: array<f32, 8>;
                for (var i = 0u; i < 4u; i++) {
                    xs[i] = x_tile[(4u * tr + i) * 33u + k];
                }
                for (var j = 0u; j < 8u; j++) {
                    ws[j] = w_tile[micro_row(t, j, gated) * 33u + k];
                }
                for (var i = 0u; i < 4u; i++) {
                    for (var j = 0u; j < 8u; j++) {
                        range_sum[i * 8u + j] = mac(range_sum[i * 8u + j], ws[j], xs[i]);
                    }
                }
            }
        }
        // Range r done: into the total, in order, and the next from zero.
        for (var o = 0u; o < 32u; o++) {
            total[o] = total[o] + range_sum[o];
            range_sum[o] = 0.0;
        }
    }
    return total;
}

@compute @workgroup_size(workgroup_size)
fn prefill_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, false);
    let token_tiles = (step.tokens + 31u) / 32u;
    let out_tile = wg.x / token_tiles;
    for (var i = 0u; i < 4u; i++) {
        let token = (wg.x % token_tiles) * 32u + 4u * (t % 8u) + i;
        for (var j = 0u; j < 8u; j++) {
            let at = tile_row(out_tile, micro_row(t, j, false), false);
            if (token < step.tokens && at.z == 1u) {
                out0[token * out_width + matmul.members[at.x].z + at.y] = totals[i * 8u + j];
            }
        }
    }
}

@compute @workgroup_size(workgroup_size)
fn prefill_qkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, false);
    let token_tiles = (step.tokens + 31u) / 32u;
    let out_tile = wg.x / token_tiles;
    for (var i = 0u; i < 4u; i++) {
        let token = (wg.x % token_tiles) * 32u + 4u * (t % 8u) + i;
        for (var j = 0u; j < 8u; j++) {
            let at = tile_row(out_tile, micro_row(t, j, false), false);
            if (token < step.tokens && at.z == 1u) {
                let width = matmul.members[at.x].w;
                let value = totals[i * 8u + j];
                if (at.x == 0u) {
                    out0[token * width + at.y] = value;
                } else if (at.x == 1u) {
                    out1[token * width + at.y] = value;
                } else {
                    out2[token * width + at.y] = value;
                }
            }
        }
    }
}

@compute @workgroup_size(workgroup_size)
fn prefill_gated(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, true);
    let token_tiles = (step.tokens + 31u) / 32u;
    let out_tile = wg.x / token_tiles;
    let width = matmul.members[0].w;
    for (var i = 0u; i < 4u; i++) {
        let token = (wg.x % token_tiles) * 32u + 4u * (t % 8u) + i;
        for (var j = 0u; j < 4u; j++) {
            let row = out_tile * 32u + (t / 8u) * 4u + j;
            if (token < step.tokens && row < width) {
                out0[token * width + row] = activate(totals[i * 8u + j]) * totals[i * 8u + 4u + j];
            }
        }
    }
}
