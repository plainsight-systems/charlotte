// An audio encoder's two subsampling convolutions (kernels/subsample/
// subsample.h), each 3 × 3, stride 2, padding 1, then a layer norm over the
// output channels — mean-centred, a gain, no bias — and ReLU. Two entry
// points: `first`, the spectrogram's one channel into 128; `second`, 128
// channels into 32.
//
// For a position's channels x₁ … x_C:
//
//   μ   = (x₁ + … + x_C) / C
//   σ²  = ((x₁ − μ)² + … + (x_C − μ)²) / C
//   y_c = max((x_c − μ) × inverseSqrt(σ² + ε) × g_c, 0)
//
// llama.cpp's ggml_norm, ggml_mul by the gain and ggml_relu.
//
// inverseSqrt, max and select are WGSL built-ins.

struct Subsample {
    epsilon: f32,
}

// The program sets workgroup_size from the launcher (subsample.cpp): 64 for
// `first`, 32 for `second`; the default only lets a tool that reads this
// file alone size what uses it.
override workgroup_size: u32 = 64;

@group(0) @binding(1) var<uniform> subsample: Subsample;
@group(0) @binding(2) var<storage, read> conv: array<f32>;
@group(0) @binding(3) var<storage, read> gain: array<f32>;
@group(0) @binding(4) var<storage, read> input: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

const kMelBins = 128u;
const kFirstChannels = 128u;
const kFirstWidth = 64u;      // frequencies after the first, 128 / 2
const kSecondChannels = 32u;
const kSecondWidth = 32u;     // after the second, 64 / 2

// The first convolution's 9 taps for output (t, f) and channel c, kh then
// kw: w[c][0][kh][kw] × mel[2t − 1 + kh][2f − 1 + kw], a padded tap 0.
fn first_sum(taps: array<f32, 9>, c: u32) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < 9u; k++) {
        sum = sum + conv[c * 9u + k] * taps[k];
    }
    return sum;
}

// One workgroup a step row r: positions 2r and 2r + 1 in time, those under
// T₂ = ⌈T / 2⌉; invocation f frequency f of each. Each position's 128
// channels are computed three times, for μ, for σ² and to write, rather
// than held: 128 floats an invocation would spill (subsample.h).
@compute @workgroup_size(workgroup_size)
fn first(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) f: u32) {
    let r = group.x;
    if (r >= step.tokens || f >= kFirstWidth) {
        return;
    }
    let frames = step.columns;
    let positions = (frames + 1u) / 2u;
    for (var p = 0u; p < 2u; p++) {
        let t = 2u * r + p;
        if (t >= positions) {
            break;
        }
        // The 9 mel values the position reads, kh along time, kw along
        // frequency; a tap before 0 or past the last frame or bin reads 0.
        var taps: array<f32, 9>;
        for (var kh = 0u; kh < 3u; kh++) {
            for (var kw = 0u; kw < 3u; kw++) {
                let time = i32(2u * t + kh) - 1;
                let bin = i32(2u * f + kw) - 1;
                var x = 0.0;
                if (time >= 0 && u32(time) < frames && bin >= 0 && u32(bin) < kMelBins) {
                    x = input[u32(time) * kMelBins + u32(bin)];
                }
                taps[kh * 3u + kw] = x;
            }
        }
        var total = 0.0;
        for (var c = 0u; c < kFirstChannels; c++) {
            total = total + first_sum(taps, c);
        }
        let mean = total / f32(kFirstChannels);
        var squares = 0.0;
        for (var c = 0u; c < kFirstChannels; c++) {
            let d = first_sum(taps, c) - mean;
            squares = squares + d * d;
        }
        let scale = inverseSqrt(squares / f32(kFirstChannels) + subsample.epsilon);
        // [t][f][c]: the position's channels contiguous, 32 vec4s.
        let base = (t * kFirstWidth + f) * (kFirstChannels / 4u);
        for (var j = 0u; j < kFirstChannels / 4u; j++) {
            let c = 4u * j;
            let x = vec4<f32>(first_sum(taps, c), first_sum(taps, c + 1u), first_sum(taps, c + 2u),
                              first_sum(taps, c + 3u));
            let g = vec4<f32>(gain[c], gain[c + 1u], gain[c + 2u], gain[c + 3u]);
            output[base + j] = max((x - mean) * scale * g, vec4<f32>(0.0));
        }
    }
}

// Channels c to c + 3 of the second convolution's weights at input channel
// i and tap (kh, kw): w[c][i][kh][kw], 1,152 apart from one channel to the
// next.
fn second_weights(c: u32, i: u32, tap: u32) -> vec4<f32> {
    let stride = kFirstChannels * 9u;
    let at = (c * kFirstChannels + i) * 9u + tap;
    return vec4<f32>(conv[at], conv[at + stride], conv[at + 2u * stride], conv[at + 3u * stride]);
}

// One workgroup of 32 a step row r; invocation f its frequency's 32
// channels, held as 8 named vec4 accumulators, each channel summed over kh,
// then kw, then the 128 input channels.
@compute @workgroup_size(workgroup_size)
fn second(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) f: u32) {
    let r = group.x;
    if (r >= step.tokens || f >= kSecondWidth) {
        return;
    }
    let positions = (step.columns + 1u) / 2u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);
    for (var kh = 0u; kh < 3u; kh++) {
        let time = i32(2u * r + kh) - 1;
        if (time < 0 || u32(time) >= positions) {
            continue;
        }
        for (var kw = 0u; kw < 3u; kw++) {
            let frequency = i32(2u * f + kw) - 1;
            if (frequency < 0 || u32(frequency) >= kFirstWidth) {
                continue;
            }
            let tap = kh * 3u + kw;
            let at = (u32(time) * kFirstWidth + u32(frequency)) * kFirstChannels;
            for (var i = 0u; i < kFirstChannels; i++) {
                let x = input[at + i];
                a0 = a0 + second_weights(0u, i, tap) * x;
                a1 = a1 + second_weights(4u, i, tap) * x;
                a2 = a2 + second_weights(8u, i, tap) * x;
                a3 = a3 + second_weights(12u, i, tap) * x;
                a4 = a4 + second_weights(16u, i, tap) * x;
                a5 = a5 + second_weights(20u, i, tap) * x;
                a6 = a6 + second_weights(24u, i, tap) * x;
                a7 = a7 + second_weights(28u, i, tap) * x;
            }
        }
    }
    // The 32 channels' statistics in channel order.
    var total = 0.0;
    total = total + a0.x + a0.y + a0.z + a0.w;
    total = total + a1.x + a1.y + a1.z + a1.w;
    total = total + a2.x + a2.y + a2.z + a2.w;
    total = total + a3.x + a3.y + a3.z + a3.w;
    total = total + a4.x + a4.y + a4.z + a4.w;
    total = total + a5.x + a5.y + a5.z + a5.w;
    total = total + a6.x + a6.y + a6.z + a6.w;
    total = total + a7.x + a7.y + a7.z + a7.w;
    let mean = total / f32(kSecondChannels);
    let d0 = a0 - mean;
    let d1 = a1 - mean;
    let d2 = a2 - mean;
    let d3 = a3 - mean;
    let d4 = a4 - mean;
    let d5 = a5 - mean;
    let d6 = a6 - mean;
    let d7 = a7 - mean;
    var squares = 0.0;
    squares = squares + d0.x * d0.x + d0.y * d0.y + d0.z * d0.z + d0.w * d0.w;
    squares = squares + d1.x * d1.x + d1.y * d1.y + d1.z * d1.z + d1.w * d1.w;
    squares = squares + d2.x * d2.x + d2.y * d2.y + d2.z * d2.z + d2.w * d2.w;
    squares = squares + d3.x * d3.x + d3.y * d3.y + d3.z * d3.z + d3.w * d3.w;
    squares = squares + d4.x * d4.x + d4.y * d4.y + d4.z * d4.z + d4.w * d4.w;
    squares = squares + d5.x * d5.x + d5.y * d5.y + d5.z * d5.z + d5.w * d5.w;
    squares = squares + d6.x * d6.x + d6.y * d6.y + d6.z * d6.z + d6.w * d6.w;
    squares = squares + d7.x * d7.x + d7.y * d7.y + d7.z * d7.z + d7.w * d7.w;
    let scale = inverseSqrt(squares / f32(kSecondChannels) + subsample.epsilon);
    // Row r's floats 32f to 32f + 31: frequency then channel, as llama.cpp
    // flattens them.
    let base = r * (kSecondWidth * kSecondChannels / 4u) + f * (kSecondChannels / 4u);
    let zero = vec4<f32>(0.0);
    output[base + 0u] = max(d0 * scale * vec4<f32>(gain[0], gain[1], gain[2], gain[3]), zero);
    output[base + 1u] = max(d1 * scale * vec4<f32>(gain[4], gain[5], gain[6], gain[7]), zero);
    output[base + 2u] = max(d2 * scale * vec4<f32>(gain[8], gain[9], gain[10], gain[11]), zero);
    output[base + 3u] = max(d3 * scale * vec4<f32>(gain[12], gain[13], gain[14], gain[15]), zero);
    output[base + 4u] = max(d4 * scale * vec4<f32>(gain[16], gain[17], gain[18], gain[19]), zero);
    output[base + 5u] = max(d5 * scale * vec4<f32>(gain[20], gain[21], gain[22], gain[23]), zero);
    output[base + 6u] = max(d6 * scale * vec4<f32>(gain[24], gain[25], gain[26], gain[27]), zero);
    output[base + 7u] = max(d7 * scale * vec4<f32>(gain[28], gain[29], gain[30], gain[31]), zero);
}
