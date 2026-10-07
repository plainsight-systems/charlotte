// Q6_K unpack (format.h). A super-block is 256 weights, eight groups of 32,
// each weight six bits: q = low nibble | high pair << 4. Group k of a block
// takes scales 2k and 2k + 1 of its sixteen signed 8-bit scales, the first
// for weights 0 .. 15, the second for 16 .. 31, and an fp16 d; weight l is
// (d * scale) * (q - 32), in that order, as ggml's dequantize_row_q6_K.
//
// On the device a piece holds blocks as upload repacked them (q6_k.h): every
// block's low nibbles, 16 bytes a group, byte j holding weight j's then
// weight j + 16's; then every block's high pairs, 8 bytes a group, byte m
// holding the pairs of weights m, m + 8, m + 16 and m + 24 from bit 0 up;
// then every block's scales, then every block's d (device_layout.h). A group
// reads 8 words: four of nibbles, two of pairs, one of scales, one of d.
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    let n = blocks_in_piece;
    let b = group / 8u;
    let k = group % 8u;
    let d = unpack2x16float(weights[base + n * 52u + b / 2u])[b % 2u];
    // The group's two scales, adjacent bytes at an even offset in their word.
    let scale_byte = n * 192u + b * 16u + k * 2u;
    let scales = weights[base + scale_byte / 4u] >> ((scale_byte % 4u) * 8u);
    let first = d * f32(bitcast<i32>(scales << 24u) >> 24u);
    let second = d * f32(bitcast<i32>((scales >> 8u) << 24u) >> 24u);
    let pairs_at = n * 32u + b * 16u + k * 2u;
    let pairs = vec2<u32>(weights[base + pairs_at], weights[base + pairs_at + 1u]);
    var out: array<vec4<f32>, 8>;
    for (var w = 0u; w < 4u; w++) {
        // Weights 4w .. 4w + 3 and 16 + 4w .. 16 + 4w + 3: nibble bytes
        // 4w .. 4w + 3; pair bytes 4(w % 2) .. 4(w % 2) + 3, at bits 2(w / 2)
        // and 2(w / 2) + 4.
        let word = weights[base + b * 32u + k * 4u + w];
        let nibbles = vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u);
        let pair_word = select(pairs.x, pairs.y, w % 2u == 1u);
        let pair_bytes = vec4<u32>(pair_word, pair_word >> 8u, pair_word >> 16u, pair_word >> 24u);
        let shift = (w / 2u) * 2u;
        let low = (nibbles & vec4<u32>(0xFu)) | (((pair_bytes >> vec4<u32>(shift)) & vec4<u32>(3u)) << vec4<u32>(4u));
        let high = ((nibbles >> vec4<u32>(4u)) & vec4<u32>(0xFu)) |
                   (((pair_bytes >> vec4<u32>(shift + 4u)) & vec4<u32>(3u)) << vec4<u32>(4u));
        out[w] = first * vec4<f32>(vec4<i32>(low) - vec4<i32>(32));
        out[w + 4u] = second * vec4<f32>(vec4<i32>(high) - vec4<i32>(32));
    }
    return out;
}
