// F16 unpack (format.h). A "block" is one half-precision value, two bytes, so
// group g of a piece is its values 32g .. 32g + 31: sixteen words, each two
// values, the lower half first, decoded with unpack2x16float. Every f16 is
// an f32, so values come back exact, within the numeric contract format.h
// states. The caller asks only for groups within the piece.
fn unpack(blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        let at = group * 16u + v * 2u;
        out[v] = vec4<f32>(unpack2x16float(weights[at]), unpack2x16float(weights[at + 1u]));
    }
    return out;
}
