// Q8_0 unpack (format.h). A block is 32 weights: an fp16 scale d and 32
// signed 8-bit codes; weight j is qs[j] * d, as ggml's dequantize_row_q8_0.
// On the device a piece holds every block's 32 code bytes, then every
// block's scale (device_layout.h): eight words of codes per block, then two
// scales a word.
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    let d = unpack2x16float(weights[base + blocks_in_piece * 8u + group / 2u])[group % 2u];
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        // Codes 4v .. 4v + 3, low byte first, each sign-extended.
        let word = weights[base + group * 8u + v];
        let q = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u),
                          bitcast<i32>(word)) >> vec4<u32>(24u);
        out[v] = vec4<f32>(q) * d;
    }
    return out;
}
