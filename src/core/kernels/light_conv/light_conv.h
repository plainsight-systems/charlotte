#pragma once

#include <cstdint>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// A Conformer's light convolution, between its two pointwise projections
// (graph/audio.h), in one launch: a causal depthwise convolution over time,
// an RMS norm with a gain, and SiLU. The GLU before it is the first
// pointwise projection's gated epilogue (kernels/matmul/matmul.h), so its
// rows arrive gated, W wide. llama.cpp's gemma4a graph builds the same from
// ggml_sigmoid, ggml_mul, ggml_pad_ext, ggml_ssm_conv, ggml_rms_norm,
// ggml_mul and ggml_silu, five passes over the rows.
//
// For row r and channel c of the width W, its input g the GLU's R rows of
// W:
//   y[r][c] = Σ_{i=0..K−1} w[c][i] × g[r − K + 1 + i][c], g of a row before 0
//     being 0 — llama.cpp's left padding of K − 1 rows, K = 5, the taps in
//     i order; w is conv_dw, (K, W), F32, w[c][i] at c × K + i;
//   z[r][c] = y[r][c] × ρ × gain[c], ρ = 1 / √(mean over c of y² + ε);
//   out[r][c] = z × σ(z), computed as z / (1 + e^−z), ggml_silu's form.
//
//   - One workgroup of 256 a row; the launch's rows are the step's R.
//     Invocation n holds the vec4s n, n + 256, … of the row — channels 4n
//     to 4n + 3 for W = 1,024 — as norm.h's invocations hold theirs. For
//     each it reads g at its K rows, r − 4 to r, and the K weights, then y
//     in registers. Its squares summed, then
//     norm.h's tree in workgroup memory, 8 levels, 9 barriers; ρ once; each
//     invocation writes its vec4s of out.
//   - SiLU's e^−x from WGSL's exp.
//   - Constants (binding 1): struct LightConv { epsilon: f32, kernel: u32 }.
//     The width an override, as norm.h's is.
//   - Bindings: 2 conv_dw and 3 the norm's gain, F32, read-only; 4 the
//     GLU's rows, read-only; 5 the output, written.
//
// Determinism (GDSA.2): run to run — the taps in order, the reduction a
// fixed tree, as norm.h's.
//
// Accuracy, against f64 from the same inputs: y within γ(K) × Σ|w g| of its
// taps; the norm within norm.h's margin for a row of W; the GPU test
// measures it.
//
// What it costs, at R = 500 and W = 1,024 a layer:
//   - Reads: R × W floats, K times each, 2 MB from memory and the rest
//     from the GPU's caches, since a row's K inputs are its neighbours' too;
//     writes R × W, 2 MB. Unfused, the convolution, the norm and SiLU would
//     each write and read 2 MB, in 3 launches.
//   - Multiply-adds: 5 a channel and the norm's: 2.6 × 10⁶ a layer; one
//     exponential a channel, SiLU's.
//   - 12 launches a step.
// Optimization (practice): the GLU in the product that makes its inputs,
// each output's sigmoid once and only W of the 2W outputs written — half
// the bytes the product would write and this launch read (GDSA.6).
// Optimization (practice): the convolution, its norm and SiLU in one
// launch, where llama.cpp makes three passes (GDSA.6, GPU.6).
//
// Verification the implementation is held to, on the GPU against f64: at R
// of 1, 4, 5 and 508 — rows with fewer than 4 predecessors, the first with
// all, the most; a row of zeros, written as zeros. On the CPU: the launch's geometry and bindings.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.20   Prefer return values to out parameters — the launch is returned.
//     I.4    Make interfaces precisely and strongly typed.
//   C++ performance guidelines
//     GDSA.6 Count passes over global memory — one pass where there were
//            three, the GLU in the product.
//     GDSA.2 Declare each floating-point reduction's determinism level — run
//            to run.
//     GPU.6  Batch tiny GPU work — one launch a layer.

// One workgroup of 256 a row, the step's rows. Preconditions: `depthwise`
// is (kernel, width) F32 and `gain` width F32; width at most 4,096 and a
// multiple of 4; `input` rows of width, the GLU's.
[[nodiscard]] Launch light_conv_launch(const residency::WeightView& depthwise, const residency::WeightView& gain,
                                       const residency::BufferRange& input, const residency::BufferRange& output,
                                       std::uint32_t width, std::uint32_t kernel, float epsilon);

}  // namespace bllm::kernels
