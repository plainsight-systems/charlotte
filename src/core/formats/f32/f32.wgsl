// F32 unpack (format.h). A "block" is one float, stored whole, so group g of
// a piece is its floats 32g .. 32g + 31, read as words and reinterpreted, so
// nothing is computed; values come back as stored, within the numeric
// contract format.h states. The caller asks only for groups within the piece.
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        let at = group * 32u + v * 4u;
        out[v] = bitcast<vec4<f32>>(vec4<u32>(weights[base + at], weights[base + at + 1u], weights[base + at + 2u], weights[base + at + 3u]));
    }
    return out;
}
