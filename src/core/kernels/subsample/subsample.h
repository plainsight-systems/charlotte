#pragma once

#include <cstdint>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// An audio encoder's front end (graph/audio.h): the spectrogram, an image
// of frames × mel bins, shrunk 4 times along each axis by two convolutions,
// each 3 × 3, stride 2, padding 1, F32 and without bias, each followed by a
// layer norm over its output channels — mean-centred, a gain, no bias —
// and ReLU. llama.cpp's gemma4a graph: ggml_conv_2d, ggml_norm over the
// channels, ggml_mul by the gain, ggml_relu. Two entry points of one module.
//
// Layouts, f32, frame-major as media/audio.h writes the spectrogram:
//   - `mel`: T frames of M = 128 bins, [t][m].
//   - `subsampled`: T₂ = ⌈T / 2⌉ positions in time × M / 2 = 64 in
//     frequency × C₁ = 128 channels, [t][f][c]: a position's channels
//     contiguous, as the second convolution reads them.
//   - the second's output: R = ⌈T₂ / 2⌉ rows of M / 4 = 32 positions in
//     frequency × C₂ = 32 channels, [r][f][c], so row r is the 1,024 floats
//     llama.cpp's flatten gives — frequency then channel — and the input
//     projection reads it as a row.
// Weights as the file stores them, (3, 3, in, out): w[o][i][kh][kw] at ((o ×
// in + i) × 3 + kh) × 3 + kw; kh along time, kw along frequency. A tap
// outside the input reads 0, the padding.
//
//   - `first`: output (t, f, c) = Σ over kh, kw of w[c][0][kh][kw] ×
//     mel[2t − 1 + kh][2f − 1 + kw], 9 terms in that order. Then over the
//     position's 128 channels: μ the mean, σ² the mean of (x − μ)², y = (x −
//     μ) / √(σ² + ε) × g[c], ReLU. One workgroup of 64 a step row r — the
//     launch's rows are the step's R — covering positions 2r and 2r + 1 in
//     time, those under T₂; invocation f takes frequency f of each. It
//     computes its position's 128 channels three times over, 9 multiply-adds
//     each: once for μ, once for σ², once to write — 3,456 multiply-adds for
//     128 outputs rather than holding 128 floats an invocation, which would
//     spill, or reducing across invocations, which would cost barriers.
//     The 5 mel frames a row reads, 4r − 1 to 4r + 3, come from the GPU's
//     caches; the 1,152 weights and 128 gains are uniform across
//     the workgroup, each read by every invocation at once (GPU.2). Writes:
//     each invocation its 128 channels as 32 vec4s.
//   - `second`: output (r, f, c) = Σ over kh, kw, i of w[c][i][kh][kw] ×
//     sub[2r − 1 + kh][2f − 1 + kw][i] — kh, then kw, then the 128 input
//     channels, in that order, 1,152 terms; then the layer norm over the
//     32 channels, held in registers, and ReLU. One workgroup of 32 a row r,
//     invocation f its frequency's 32 channels: 36,864 multiply-adds an
//     invocation, its 32 accumulators named in registers, the input's 1,152
//     floats read once each and the 36,864 weights — 147 KB, uniform across
//     the workgroup — from the GPU's caches. Writes: 8 vec4s an invocation,
//     row r's floats 32f to 32f + 31.
//   - ε the file's, 10⁻⁶, in the constants; the gain bound beside the
//     weights.
//   - Constants (binding 1): struct Subsample { epsilon: f32, frames: u32 },
//     T₂ for `second`'s bound, T for `first`'s — the step's `columns` is T,
//     and each form takes its own bound from it.
//   - Bindings: 2 the convolution's weights, 3 its gain, F32, read-only; 4
//     the input, read-only; 5 the output, written.
//
// Determinism (GDSA.2): run to run — every sum in the order above, each
// position's statistics in channel order.
//
// Accuracy, against f64 from the same inputs: each convolution sum within
// γ(n) × Σ|w x| of its terms, n = 9 or 1,152; each normalized output within
// a few units of f32 rounding times its row's spread, which the GPU test
// measures; ReLU exact. A margin checked on the target, not a proof.
//
// What it costs, at T = 2,000 frames, R = 500:
//   - `first`: 1,000 positions in time × 64 × 128 channels × 27
//     multiply-adds, 2.2 × 10⁸; writes 33 MB once.
//   - `second`: 500 × 32 × 32 × 1,152, 5.9 × 10⁸; reads 33 MB, each float
//     about 2.25 times through the caches; writes 2 MB.
//   - Launches: 2 a clip.
// Optimization not taken: the two convolutions fused (graph/audio.h).
// Optimization not taken: the first's statistics shared across invocations
// through workgroup memory: it would save two of every three multiply-adds,
// 1.5 × 10⁸ of the step's 1.5 × 10¹¹, for a reduction of 128 floats a
// position pair with its barriers.
//
// Verification the implementation is held to, on the GPU against f64:
// each form at T of 1, 2, 7 and 2,032 — the first and last positions'
// padding, an odd T, the most frames — written where it should be and
// nowhere past its last row; a position of equal channels, normalized to 0;
// negative outputs cut to 0. On the CPU: each launch's geometry and
// bindings.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.20   Prefer return values to out parameters — the launch is returned.
//     I.4    Make interfaces precisely and strongly typed.
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level — run
//            to run.
//     GPU.2  Shape data for coalesced lane access — channels contiguous;
//            weights uniform across a workgroup.
//     GPU.3  Keep hot values in registers — `second`'s 32 accumulators named.

// The first convolution: `mel`, T frames of 128 bins, into `subsampled`.
// One workgroup of 64 a step row, the step's R rows.
[[nodiscard]] Launch subsample_first_launch(const residency::WeightView& weights, const residency::WeightView& gain,
                                            const residency::BufferRange& mel,
                                            const residency::BufferRange& subsampled, float epsilon);

// The second: `subsampled` into `rows`, R rows of 1,024 floats.
[[nodiscard]] Launch subsample_second_launch(const residency::WeightView& weights, const residency::WeightView& gain,
                                             const residency::BufferRange& subsampled,
                                             const residency::BufferRange& rows, float epsilon);

}  // namespace bllm::kernels
