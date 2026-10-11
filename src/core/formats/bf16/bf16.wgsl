// BF16 unpack (formats/bf16/bf16.h). A "block" is one bfloat16 value, two
// bytes: the top half of an f32. Group g of a piece is its values 32g ..
// 32g + 31, sixteen words, each two values, the lower half first, as F16's
// are. Each value is its word's half shifted into the top of an f32 and
// reinterpreted, so nothing is computed and every value comes back exact,
// as ggml's ggml_bf16_to_fp32 gives it. The caller asks only for groups
// within the piece.
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        let at = group * 16u + v * 2u;
        let a = weights[base + at];
        let b = weights[base + at + 1u];
        out[v] = bitcast<vec4<f32>>(vec4<u32>(a << 16u, a & 0xFFFF0000u, b << 16u, b & 0xFFFF0000u));
    }
    return out;
}
