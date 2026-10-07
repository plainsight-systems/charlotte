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
override tile_tokens: u32 = 32;     // a prefill tile's tokens: 8, 16 or 32
override set_rows: u32 = 4;         // a decode set's rows, Write and QKV: 1 to 4

const kRanges = 32u;

@group(0) @binding(1) var<uniform> matmul: Matmul;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

// Decode: each of 8 row slots' 32 range sums.
var<workgroup> partials: array<f32, 256>;
// Prefill: the step's input floats and the tile's 64 row slots' decoded
// weights, each written as whole vec4s along K — a write to one component
// of a vector in workgroup memory may write all four, so invocations never
// share a vector: token i's k-quad v at x_tile[8i + v], row slot s's at
// w_tile[9s + v], each slot's row padded a vec4 so the 16 slots a read
// takes, s = oq + 16j, fall in distinct banks.
var<workgroup> x_tile: array<vec4<f32>, 8u * tile_tokens>;
var<workgroup> w_tile: array<vec4<f32>, 576>;

const kTileRows = 64u;       // a prefill tile's outputs
const kOutputLanes = 16u;    // invocations a token quad; an invocation's outputs are slots lane + 16j
const kMicro = 4u;           // an invocation's tokens, and its outputs
const kSlotStride = 9u;      // a slot's 8 k-quads and a vec4 of padding

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

// A workgroup's 2 × set_rows rows of the product, slots team × 4 + j for j
// below set_rows: invocation lane of a team sums its range of the team's
// rows; after one barrier the caller adds each slot's ranges.
fn decode_rows(wg: u32, t: u32) {
    let team = t / 32u;
    let lane = t % 32u;
    var members = vec4<u32>(0u);
    var rows = vec4<u32>(0u);
    var live = vec4<bool>(false);
    for (var j = 0u; j < set_rows; j++) {
        let at = locate(wg * 2u * set_rows + team * set_rows + j);
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

// The slot holding the workgroup's row t, for t below 2 × set_rows.
fn row_slot(t: u32) -> u32 {
    return (t / set_rows) * 4u + t % set_rows;
}

@compute @workgroup_size(workgroup_size)
fn decode_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    decode_rows(wg.x, t);
    if (t < 2u * set_rows) {
        let at = locate(wg.x * 2u * set_rows + t);
        if (at.z == 1u) {
            out0[matmul.members[at.x].z + at.y] = slot_total(row_slot(t));
        }
    }
}

@compute @workgroup_size(workgroup_size)
fn decode_qkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    decode_rows(wg.x, t);
    if (t < 2u * set_rows) {
        let at = locate(wg.x * 2u * set_rows + t);
        if (at.z == 1u) {
            let total = slot_total(row_slot(t));
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

// ---- Prefill ----------------------------------------------------------------

// Row slot `slot`'s member and row, as decode's slots: an invocation's 4
// outputs are slots lane + 16j. Write and QKV slots are the tile's rows in
// order; a gated tile's slot 16j + c is gate row 2c + j for j 0 and 1, and
// the same up row, 2c + j − 2, for j 2 and 3, so one invocation holds both.
fn slot_row(out_tile: u32, slot: u32, gated: bool) -> vec3<u32> {
    if (gated) {
        let j = slot / kOutputLanes;
        let row = out_tile * (kTileRows / 2u) + (slot % kOutputLanes) * 2u + j % 2u;
        return vec3<u32>(j / 2u, row, select(0u, 1u, row < matmul.members[0].w));
    }
    return locate(out_tile * kTileRows + slot);
}

// The tile's totals for invocation t's 4 tokens × 4 outputs, a vec4 a token
// over its outputs: each output's ranges summed from zero, along K in order,
// and added to its total in order, exactly as decode adds them; fma on a
// vec4 is the scalar fma a component.
fn prefill_tile(wg: u32, t: u32, gated: bool) -> array<vec4<f32>, 4> {
    let token_tiles = (step.tokens + tile_tokens - 1u) / tile_tokens;
    let token_tile = wg % token_tiles;
    let out_tile = wg / token_tiles;
    let tq = t / kOutputLanes;
    let lane = t % kOutputLanes;
    // A quad wholly past the step stages and passes the barriers, and skips
    // the products.
    let live = token_tile * tile_tokens + tq * kMicro < step.tokens;
    // The row slots this invocation decodes: t, and t + workgroup_size when
    // the workgroup is smaller than the tile's rows.
    let slot0 = t;
    let slot1 = t + workgroup_size;
    let mine0 = slot_row(out_tile, slot0, gated);
    let mine1 = slot_row(out_tile, min(slot1, kTileRows - 1u), gated);
    let x_at = tq * kMicro * 8u;
    var s0 = vec4<f32>(0.0);
    var s1 = vec4<f32>(0.0);
    var s2 = vec4<f32>(0.0);
    var s3 = vec4<f32>(0.0);
    var y0 = vec4<f32>(0.0);
    var y1 = vec4<f32>(0.0);
    var y2 = vec4<f32>(0.0);
    var y3 = vec4<f32>(0.0);
    for (var r = 0u; r < kRanges; r++) {
        for (var g = range_lo(r); g < range_lo(r + 1u); g++) {
            workgroupBarrier();   // the last step's tiles are read
            // The tile's rows' groups, decoded.
            if (slot0 < kTileRows) {
                stage_row(slot0, mine0, g);
            }
            if (slot1 < kTileRows) {
                stage_row(slot1, mine1, g);
            }
            // The step's input floats, a vec4 at a time along K.
            for (var q = t; q < 8u * tile_tokens; q += workgroup_size) {
                let token = token_tile * tile_tokens + q / 8u;
                var x = vec4<f32>(0.0);
                if (token < step.tokens) {
                    x = input[token * (columns / 4u) + g * 8u + q % 8u];
                }
                x_tile[q] = x;
            }
            workgroupBarrier();
            // Each k-quad: the 4 outputs' weights, transposed in registers so
            // w[c] is their 4 weights at k = 4v + c, and the 4 tokens' inputs;
            // 8 vec4 reads, then 64 multiply-adds, k in order.
            if (live) {
                for (var v = 0u; v < 8u; v++) {
                    let w = transpose(mat4x4<f32>(w_tile[lane * kSlotStride + v],
                                                  w_tile[(lane + 16u) * kSlotStride + v],
                                                  w_tile[(lane + 32u) * kSlotStride + v],
                                                  w_tile[(lane + 48u) * kSlotStride + v]));
                    let x0 = x_tile[x_at + v];
                    let x1 = x_tile[x_at + 8u + v];
                    let x2 = x_tile[x_at + 16u + v];
                    let x3 = x_tile[x_at + 24u + v];
                    s0 = fma(w[0], vec4<f32>(x0.x), s0);
                    s1 = fma(w[0], vec4<f32>(x1.x), s1);
                    s2 = fma(w[0], vec4<f32>(x2.x), s2);
                    s3 = fma(w[0], vec4<f32>(x3.x), s3);
                    s0 = fma(w[1], vec4<f32>(x0.y), s0);
                    s1 = fma(w[1], vec4<f32>(x1.y), s1);
                    s2 = fma(w[1], vec4<f32>(x2.y), s2);
                    s3 = fma(w[1], vec4<f32>(x3.y), s3);
                    s0 = fma(w[2], vec4<f32>(x0.z), s0);
                    s1 = fma(w[2], vec4<f32>(x1.z), s1);
                    s2 = fma(w[2], vec4<f32>(x2.z), s2);
                    s3 = fma(w[2], vec4<f32>(x3.z), s3);
                    s0 = fma(w[3], vec4<f32>(x0.w), s0);
                    s1 = fma(w[3], vec4<f32>(x1.w), s1);
                    s2 = fma(w[3], vec4<f32>(x2.w), s2);
                    s3 = fma(w[3], vec4<f32>(x3.w), s3);
                }
            }
        }
        // Range r done: into the totals, in order, and the next from zero.
        y0 = y0 + s0;
        y1 = y1 + s1;
        y2 = y2 + s2;
        y3 = y3 + s3;
        s0 = vec4<f32>(0.0);
        s1 = vec4<f32>(0.0);
        s2 = vec4<f32>(0.0);
        s3 = vec4<f32>(0.0);
    }
    return array<vec4<f32>, 4>(y0, y1, y2, y3);
}

// Row slot `slot`'s group g, decoded into w_tile as whole vec4s, or zeros
// for a slot past the product's rows.
fn stage_row(slot: u32, at: vec3<u32>, g: u32) {
    var w = array<vec4<f32>, 8>();
    if (at.z == 1u) {
        w = weights_of(at.x, at.y, g);
    }
    for (var v = 0u; v < 8u; v++) {
        w_tile[slot * kSlotStride + v] = w[v];
    }
}

// The first of invocation t's 4 tokens in the step.
fn first_token(wg: u32, t: u32) -> u32 {
    let token_tiles = (step.tokens + tile_tokens - 1u) / tile_tokens;
    return (wg % token_tiles) * tile_tokens + (t / kOutputLanes) * kMicro;
}

fn out_tile_of(wg: u32) -> u32 {
    return wg / ((step.tokens + tile_tokens - 1u) / tile_tokens);
}

@compute @workgroup_size(workgroup_size)
fn prefill_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, false);
    let first = first_token(wg.x, t);
    for (var j = 0u; j < kMicro; j++) {
        let at = slot_row(out_tile_of(wg.x), t % kOutputLanes + kOutputLanes * j, false);
        for (var i = 0u; i < kMicro; i++) {
            let token = first + i;
            if (token < step.tokens && at.z == 1u) {
                out0[token * out_width + matmul.members[at.x].z + at.y] = totals[i][j];
            }
        }
    }
}

@compute @workgroup_size(workgroup_size)
fn prefill_qkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, false);
    let first = first_token(wg.x, t);
    for (var j = 0u; j < kMicro; j++) {
        let at = slot_row(out_tile_of(wg.x), t % kOutputLanes + kOutputLanes * j, false);
        let width = matmul.members[at.x].w;
        for (var i = 0u; i < kMicro; i++) {
            let token = first + i;
            if (token < step.tokens && at.z == 1u) {
                let value = totals[i][j];
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

// Gated: an invocation's outputs are gate rows 2c and 2c + 1 and the same up
// rows, components 0, 1 and 2, 3, c its lane.
@compute @workgroup_size(workgroup_size)
fn prefill_gated(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, true);
    let first = first_token(wg.x, t);
    let width = matmul.members[0].w;
    for (var j = 0u; j < 2u; j++) {
        let row = out_tile_of(wg.x) * (kTileRows / 2u) + (t % kOutputLanes) * 2u + j;
        for (var i = 0u; i < kMicro; i++) {
            let token = first + i;
            if (token < step.tokens && row < width) {
                out0[token * width + row] = activate(totals[i][j]) * totals[i][j + 2u];
            }
        }
    }
}
