// Q4_1 unpack (format.h). A block is 32 weights: an fp16 scale d, an fp16
// minimum m and 16 bytes of 4-bit codes; weight j is (qs[j] & 0xF) * d + m
// and weight j + 16 is (qs[j] >> 4) * d + m, as ggml's dequantize_row_q4_1.
// On the device a piece holds every block's 16 code bytes, then every
// block's d and m together, one word a block (device_layout.h).
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    let dm = unpack2x16float(weights[base + blocks_in_piece * 4u + group]);
    var out: array<vec4<f32>, 8>;
    for (var w = 0u; w < 4u; w++) {
        // Code bytes 4w .. 4w + 3 of the block, low byte first.
        let word = weights[base + group * 4u + w];
        let bytes = vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFFu);
        out[w] = vec4<f32>(bytes & vec4<u32>(0xFu)) * dm.x + dm.y;
        out[w + 4u] = vec4<f32>(bytes >> vec4<u32>(4u)) * dm.x + dm.y;
    }
    return out;
}
