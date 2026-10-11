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
//   decode_write, decode_qkv, decode_gated, decode_gated_input,
//   prefill_write, prefill_qkv, prefill_gated, prefill_gated_input;
//   split_ranges, the split form's for every epilogue it takes, and its
//   combine's: combine_write, combine_qkv, combine_gated_input.
//
// A product's rows are its members' rows in order — a weight piece, or a
// fused group's Q, K, V or gate, up — each member read from its own first
// word in the binding.
//
// unpack is the weight format's (format.h); exp, tanh-free activations and
// workgroupBarrier as below are WGSL built-ins.

// Each member: x its first word in the binding, y its blocks, z its first
// output row, w its rows. For a gated product, z is instead the member's
// first row within its weight: 0 for a gate and an up that are weights of
// their own, the second half's first row for a GLU's gate over one weight's
// halves (matmul.h).
struct Matmul {
    members: array<vec4<u32>, 3>,
    // GatedByInput's input u: x its row stride, y its first column, in
    // floats; 0 for every other epilogue.
    gate_input: vec4<u32>,
    // A clipped product's bounds, each one F32 word of the weights binding:
    // x the input's least, y its most, z the output's least, w its most.
    clip: vec4<u32>,
    // A biased Write's bias: x its first word in the weights binding.
    bias: vec4<u32>,
}

// The program sets workgroup_size and last_token; the launcher the rest.
// The defaults are Qwen3's QKV, so a tool reading this file alone can size
// what uses them.
override workgroup_size: u32 = 64;
override last_token: bool = false;
override columns: u32 = 1024;        // K
override out_width: u32 = 1024;      // a write's output row width
override activation: u32 = 0;       // 0 SiLU, 1 GELU in its tanh form, 2 the sigmoid
// A product's options (matmul.h): its input and outputs clamped to the
// bounds `clip` names; a Write's bias added; a Write's outputs through the
// activation. None splits along K, so the combine, which binds no weights,
// never needs the bounds or the bias.
override clipped: bool = false;
override biased: bool = false;
override activated: bool = false;
override tile_tokens: u32 = 32;     // a prefill tile's tokens: 8, 16 or 32
override set_rows: u32 = 4;         // a decode set's rows, Write and QKV: 1 to 4

const kRanges = 32u;

@group(0) @binding(1) var<uniform> matmul: Matmul;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;
// GatedByInput's u, read-only: its own declaration of binding 5, which the
// QKV entry points write as out1, as two entry points may share a binding.
@group(0) @binding(5) var<storage, read> gate_input: array<f32>;

// Decode: each of 8 row slots' 32 range sums.
var<workgroup> partials: array<f32, 256>;
// Prefill: the step's input floats and the tile's 64 row slots' decoded
// weights, each written as whole vec4s along K — a write to one component
// of a vector in workgroup memory may write all four, so invocations never
// share a vector: token i's k-quad v at x_tile[8i + v], row slot s's at
// w_tile[9s + v], each slot's row padded a vec4 so the 16 slots a read
// takes, s = oq + 16j, fall in distinct banks.
// Workaround (browser): WebKit fails a pipeline whose workgroup array is
// sized by an expression of overrides, or by an override left at its
// default, so x_tile's length, 8 × tile_tokens, is its own override, set by
// the launcher (matmul.cpp) on every launch: with tile_tokens for prefill,
// and the largest tile's for decode, which never touches x_tile.
override x_tile_vec4s: u32 = 256u;
// The program's: the launch's rows are the step's tokens over this — 1 but
// for a product after a vision encoder's pooling, whose rows are its
// patches over k² (kernels/interface.h, Geometry).
override row_divisor: u32 = 1u;

// The rows this launch covers: the step's tokens, divided.
fn step_rows() -> u32 {
    return step.tokens / row_divisor;
}
var<workgroup> x_tile: array<vec4<f32>, x_tile_vec4s>;
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

// An F32 word of the weights binding: a bound or a bias.
fn word_f32(w: u32) -> f32 {
    return bitcast<f32>(weights[w]);
}

// The input as the product reads it: clamped to its bounds where clipped.
// A clamp is exact, so it adds no rounding.
fn clamp_input(x: vec4<f32>) -> vec4<f32> {
    if (!clipped) {
        return x;
    }
    return clamp(x, vec4<f32>(word_f32(matmul.clip.x)), vec4<f32>(word_f32(matmul.clip.y)));
}

// A total clamped to the output's bounds where clipped.
fn clamp_output(v: f32) -> f32 {
    if (!clipped) {
        return v;
    }
    return clamp(v, word_f32(matmul.clip.z), word_f32(matmul.clip.w));
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
    return select(0u, step_rows() - 1u, last_token);
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
    if (activation == 2u) {
        let e = exp_nonpositive(-abs(g));
        let r = 1.0 / (1.0 + e);
        return select(e * r, r, g >= 0.0);
    }
    let z = 0.7978845608028654 * fma(0.044715 * g * g, g, g);
    let e = exp_nonpositive(-2.0 * abs(z));
    let t = sign(z) * ((1.0 - e) * (1.0 / (1.0 + e)));
    let half_g = 0.5 * g;
    return fma(half_g, t, half_g);
}

// ---- Decode ----------------------------------------------------------------

// Invocation `lane`'s range of K for its set's `slots` rows, 1 to 4, against
// the step's one input row: the input's floats loaded once a group, used for
// each. `rows` holds each slot's member and row, `live` whether it is a row.
fn decode_ranges(lane: u32, slots: u32, members: vec4<u32>, rows: vec4<u32>, live: vec4<bool>) -> vec4<f32> {
    let token = token_of_decode();
    var acc = vec4<f32>(0.0);
    for (var g = range_lo(lane); g < range_lo(lane + 1u); g++) {
        var x: array<vec4<f32>, 8>;
        for (var v = 0u; v < 8u; v++) {
            x[v] = clamp_input(input[token * (columns / 4u) + g * 8u + v]);
        }
        for (var j = 0u; j < slots; j++) {
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
    let sums = decode_ranges(lane, set_rows, members, rows, live);
    for (var j = 0u; j < set_rows; j++) {
        partials[(team * 4u + j) * kRanges + lane] = sums[j];
    }
    workgroupBarrier();
}

// The slot holding the workgroup's row t, for t below 2 × set_rows.
fn row_slot(t: u32) -> u32 {
    return (t / set_rows) * 4u + t % set_rows;
}

// A Write's output o from its total: clamped, biased and activated as the
// product's options ask, in that order — llama.cpp's clamp after the
// product, its bias added after, the feed-forward's activation last.
fn finish(total: f32, o: u32) -> f32 {
    var y = clamp_output(total);
    if (biased) {
        y = y + word_f32(matmul.bias.x + o);
    }
    if (activated) {
        y = activate(y);
    }
    return y;
}

@compute @workgroup_size(workgroup_size)
fn decode_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    decode_rows(wg.x, t);
    if (t < 2u * set_rows) {
        let at = locate(wg.x * 2u * set_rows + t);
        if (at.z == 1u) {
            let o = matmul.members[at.x].z + at.y;
            out0[o] = finish(slot_total(row_slot(t)), o);
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
    let gate_at = matmul.members[0].z;
    let up_at = matmul.members[1].z;
    let rows = vec4<u32>(gate_at + p, gate_at + p + 1u, up_at + p, up_at + p + 1u);
    let sums = decode_ranges(lane, 4u, vec4<u32>(0u, 0u, 1u, 1u), rows, live);
    for (var j = 0u; j < 4u; j++) {
        partials[(team * 4u + j) * kRanges + lane] = sums[j];
    }
    workgroupBarrier();
    if (t < 4u) {
        let s = t / 2u;
        let k = t % 2u;
        let row = wg.x * 4u + s * 2u + k;
        if (row < width) {
            let gate = clamp_output(slot_total(s * 4u + k));
            let up = clamp_output(slot_total(s * 4u + 2u + k));
            out0[row] = activate(gate) * up;
        }
    }
}

// GatedByInput: an output row's total through the activation, times the
// input u at the token's row and the output's column.
fn gated_by_input(total: f32, token: u32, o: u32) -> f32 {
    return activate(total) * gate_input[token * matmul.gate_input.x + matmul.gate_input.y + o];
}

@compute @workgroup_size(workgroup_size)
fn decode_gated_input(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    decode_rows(wg.x, t);
    if (t < 2u * set_rows) {
        let at = locate(wg.x * 2u * set_rows + t);
        if (at.z == 1u) {
            let o = matmul.members[at.x].z + at.y;
            out0[o] = gated_by_input(slot_total(row_slot(t)), token_of_decode(), o);
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
        let member = j / 2u;
        return vec3<u32>(member, matmul.members[member].z + row, select(0u, 1u, row < matmul.members[0].w));
    }
    return locate(out_tile * kTileRows + slot);
}

// Invocation t's place in the tile at (token_tile, out_tile), worked out once
// a tile rather than once a range: its lane, whether its token quad holds any
// of the step, the row slots it decodes — t, and t + workgroup_size when the
// workgroup is smaller than the tile's rows — with each slot's member and
// row, and its quad's first input in x_tile.
struct TilePlace {
    token_tile: u32,
    lane: u32,
    live: bool,
    slot0: u32,
    slot1: u32,
    mine0: vec3<u32>,
    mine1: vec3<u32>,
    x_at: u32,
}

fn tile_place(token_tile: u32, out_tile: u32, t: u32, gated: bool) -> TilePlace {
    let tq = t / kOutputLanes;
    let slot1 = t + workgroup_size;
    // A quad wholly past the step stages and passes the barriers, and skips
    // the products.
    return TilePlace(token_tile, t % kOutputLanes, token_tile * tile_tokens + tq * kMicro < step_rows(), t, slot1,
                     slot_row(out_tile, t, gated), slot_row(out_tile, min(slot1, kTileRows - 1u), gated),
                     tq * kMicro * 8u);
}

// Invocation t's sums of range r for its 4 tokens × 4 outputs of its tile, a
// vec4 a token over its outputs: the range's groups along K in order, each
// from zero, exactly as decode sums a range; fma on a vec4 is the scalar fma
// a component. Every invocation calls it alike, so its barriers are reached
// in uniform control flow.
fn tile_range(at: TilePlace, t: u32, r: u32) -> array<vec4<f32>, 4> {
    let token_tile = at.token_tile;
    let lane = at.lane;
    let live = at.live;
    let slot0 = at.slot0;
    let slot1 = at.slot1;
    let mine0 = at.mine0;
    let mine1 = at.mine1;
    let x_at = at.x_at;
    var s0 = vec4<f32>(0.0);
    var s1 = vec4<f32>(0.0);
    var s2 = vec4<f32>(0.0);
    var s3 = vec4<f32>(0.0);
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
            if (token < step_rows()) {
                x = clamp_input(input[token * (columns / 4u) + g * 8u + q % 8u]);
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
    return array<vec4<f32>, 4>(s0, s1, s2, s3);
}

// The tile's totals for invocation t's 4 tokens × 4 outputs: each range's
// sums added to its total in order, a total from zero, exactly as decode adds
// them.
fn prefill_tile(wg: u32, t: u32, gated: bool) -> array<vec4<f32>, 4> {
    let token_tiles = (step_rows() + tile_tokens - 1u) / tile_tokens;
    let at = tile_place(wg % token_tiles, wg / token_tiles, t, gated);
    var y0 = vec4<f32>(0.0);
    var y1 = vec4<f32>(0.0);
    var y2 = vec4<f32>(0.0);
    var y3 = vec4<f32>(0.0);
    for (var r = 0u; r < kRanges; r++) {
        let s = tile_range(at, t, r);
        y0 = y0 + s[0];
        y1 = y1 + s[1];
        y2 = y2 + s[2];
        y3 = y3 + s[3];
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
    let token_tiles = (step_rows() + tile_tokens - 1u) / tile_tokens;
    return (wg % token_tiles) * tile_tokens + (t / kOutputLanes) * kMicro;
}

fn out_tile_of(wg: u32) -> u32 {
    return wg / ((step_rows() + tile_tokens - 1u) / tile_tokens);
}

@compute @workgroup_size(workgroup_size)
fn prefill_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, false);
    let first = first_token(wg.x, t);
    for (var j = 0u; j < kMicro; j++) {
        let at = slot_row(out_tile_of(wg.x), t % kOutputLanes + kOutputLanes * j, false);
        for (var i = 0u; i < kMicro; i++) {
            let token = first + i;
            if (token < step_rows() && at.z == 1u) {
                let o = matmul.members[at.x].z + at.y;
                out0[token * out_width + o] = finish(totals[i][j], o);
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
            if (token < step_rows() && at.z == 1u) {
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
            if (token < step_rows() && row < width) {
                out0[token * width + row] = activate(clamp_output(totals[i][j])) * clamp_output(totals[i][j + 2u]);
            }
        }
    }
}

@compute @workgroup_size(workgroup_size)
fn prefill_gated_input(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let totals = prefill_tile(wg.x, t, false);
    let first = first_token(wg.x, t);
    for (var j = 0u; j < kMicro; j++) {
        let at = slot_row(out_tile_of(wg.x), t % kOutputLanes + kOutputLanes * j, false);
        for (var i = 0u; i < kMicro; i++) {
            let token = first + i;
            if (token < step_rows() && at.z == 1u) {
                let o = matmul.members[at.x].z + at.y;
                out0[token * out_width + o] = gated_by_input(totals[i][j], token, o);
            }
        }
    }
}

// ---- Prefill split along K ---------------------------------------------------

// The product's rows: its members' together.
fn product_rows() -> u32 {
    return matmul.members[0].w + matmul.members[1].w + matmul.members[2].w;
}

// Where range r's sum of the product's row p for `token` lies in
// `range_sums`: range first, so a range's sums for adjacent rows are adjacent.
fn range_sum_at(r: u32, token: u32, p: u32) -> u32 {
    return (r * step_rows() + token) * product_rows() + p;
}

// The tiled form's workgroup (token tile, output tile) for range r alone,
// numbered token tile first, then output tile, then range: the range sums of
// its 4 tokens × 4 outputs a lane written to `range_sums`, bound as out0.
@compute @workgroup_size(workgroup_size)
fn split_ranges(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let rows = product_rows();
    let token_tiles = (step_rows() + tile_tokens - 1u) / tile_tokens;
    let out_tiles = (rows + kTileRows - 1u) / kTileRows;
    let token_tile = wg.x % token_tiles;
    let out_tile = (wg.x / token_tiles) % out_tiles;
    let r = wg.x / (token_tiles * out_tiles);
    let sums = tile_range(tile_place(token_tile, out_tile, t, false), t, r);
    let first = token_tile * tile_tokens + (t / kOutputLanes) * kMicro;
    for (var j = 0u; j < kMicro; j++) {
        let p = out_tile * kTileRows + t % kOutputLanes + kOutputLanes * j;
        for (var i = 0u; i < kMicro; i++) {
            let token = first + i;
            if (token < step_rows() && p < rows) {
                out0[range_sum_at(r, token, p)] = sums[i][j];
            }
        }
    }
}

// The combine's bindings, its own declarations, since it reads no weights:
// 2 `range_sums`, read-only; 3 its first output; 4 its second, or
// GatedByInput's u, read-only; 5 its third. No entry point uses both of two
// declarations of one binding.
@group(0) @binding(2) var<storage, read> range_sums: array<f32>;
@group(0) @binding(3) var<storage, read_write> combined0: array<f32>;
@group(0) @binding(4) var<storage, read_write> combined1: array<f32>;
@group(0) @binding(4) var<storage, read> combined_u: array<f32>;
@group(0) @binding(5) var<storage, read_write> combined2: array<f32>;

// The total of the product's row p for `token`: its 32 range sums added to a
// total from zero, in order, as decode's slot_total and the tiled form add
// them.
fn combined(token: u32, p: u32) -> f32 {
    var total = 0.0;
    for (var r = 0u; r < kRanges; r++) {
        total = total + range_sums[range_sum_at(r, token, p)];
    }
    return total;
}

@compute @workgroup_size(workgroup_size)
fn combine_write(@builtin(global_invocation_id) id: vec3<u32>) {
    let rows = product_rows();
    if (id.x >= step_rows() * rows) {
        return;
    }
    let token = id.x / rows;
    let p = id.x % rows;
    let at = locate(p);
    combined0[token * out_width + matmul.members[at.x].z + at.y] = combined(token, p);
}

@compute @workgroup_size(workgroup_size)
fn combine_qkv(@builtin(global_invocation_id) id: vec3<u32>) {
    let rows = product_rows();
    if (id.x >= step_rows() * rows) {
        return;
    }
    let token = id.x / rows;
    let p = id.x % rows;
    let at = locate(p);
    let width = matmul.members[at.x].w;
    let value = combined(token, p);
    if (at.x == 0u) {
        combined0[token * width + at.y] = value;
    } else if (at.x == 1u) {
        combined1[token * width + at.y] = value;
    } else {
        combined2[token * width + at.y] = value;
    }
}

// As gated_by_input, with u read through the combine's own binding.
@compute @workgroup_size(workgroup_size)
fn combine_gated_input(@builtin(global_invocation_id) id: vec3<u32>) {
    let rows = product_rows();
    if (id.x >= step_rows() * rows) {
        return;
    }
    let token = id.x / rows;
    let p = id.x % rows;
    let at = locate(p);
    let o = matmul.members[at.x].z + at.y;
    let u = combined_u[token * matmul.gate_input.x + matmul.gate_input.y + o];
    combined0[token * out_width + o] = activate(combined(token, p)) * u;
}
