// Q4_0 unpack (format.h). A block is 32 weights: an fp16 scale d and 16
// bytes of 4-bit codes; weight j is ((qs[j] & 0xF) - 8) * d and weight j + 16
// is ((qs[j] >> 4) - 8) * d, as ggml's dequantize_row_q4_0. On the device a
// piece holds every block's 16 code bytes, then every block's scale
// (device_layout.h): four words of codes per block, then two scales a word.
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    let scale_byte = blocks_in_piece * 16u + group * 2u;
    let d = unpack2x16float(weights[base + scale_byte / 4u])[(scale_byte / 2u) % 2u];
    var out: array<vec4<f32>, 8>;
    for (var w = 0u; w < 4u; w++) {
        // Code bytes 4w .. 4w + 3 of the block, low byte first.
        let word = weights[base + group * 4u + w];
        let bytes = vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFFu);
        out[w] = (vec4<f32>(bytes & vec4<u32>(0xFu)) - 8.0) * d;
        out[w + 4u] = (vec4<f32>(bytes >> vec4<u32>(4u)) - 8.0) * d;
    }
    return out;
}
