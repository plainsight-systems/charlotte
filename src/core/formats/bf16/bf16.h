#pragma once

#include "bllm/shaders_generated.h"
#include "core/formats/device_layout.h"
#include "core/formats/format.h"

namespace bllm::formats {

// Axis B: changes with a new weight format.
//
// BF16: one bfloat16 value a block, two bytes — f32's sign, its 8 exponent
// bits and the top 7 of its mantissa. The weights a file stores at bfloat16:
// EmbeddingGemma 2's per-layer projection, per_layer_model_proj, 512 ×
// 12,288, as Google ships it and ggml-org's Q8_0 file keeps it.
//
//   - Unpack, bf16.wgsl: group g of a piece is its values 32g .. 32g + 31,
//     sixteen words, each two values, the lower half first, as F16's are
//     (f16.wgsl). A value is the top half of an f32: the lower half of word
//     w is bitcast<f32>(w << 16), the upper bitcast<f32>(w & 0xffff0000).
//     Every bfloat16 is an f32, so the values come back exact, bit for bit
//     with ggml's ggml_bf16_to_fp32 — which shifts the same way — with no
//     exception format.h allows: there is no arithmetic to round, and the
//     shift keeps a subnormal or a zero's sign as stored.
//   - Layout: one stream of the stored values, kBF16Layout {BF16, 2,
//     {{0, 2}}}, as F16's (device_layout.h): upload writes the file's bytes
//     as they lie.
//   - No pack. A bfloat16 KV cache — load policy's CachePrecision::BF16 —
//     needs each value rounded to 7 mantissa bits in WGSL, a pack this
//     format does not supply, so the graph refuses that precision by name
//     (format.h), and the cache stays F16 (logical-overview.md, cache
//     precision).
//
// What it costs: one word read a pair of weights, half F32's bytes — the
// projection's 12.6 MB against 25.2 MB at F32 — and a shift and a mask a
// pair, where F16's unpack2x16float converts. The products read it as any
// format (kernels/matmul/matmul.h); nothing else does.
// Optimization (practice): the weight decoded in the load path, never
// expanded to f32 in memory (GDSA.18) — what ggml's BF16 matrix kernels do.
//
// What it asks of the other contracts:
//   - formats/device_layout.h: kBF16Layout, beside kF16Layout.
//   - formats/format.h: its note that BF16 has no format becomes that it has
//     no pack; the rule it states — a cache precision whose format has no
//     pack is refused — is unchanged.
//   - capability/capability.h: one row, BF16.
//
// Verification the implementation is held to, on the GPU against a CPU
// reference mirroring ggml_bf16_to_fp32: every weight bit for bit, over
// groups of edge values — zeros of both signs, subnormals, the largest
// finite, values at each exponent — and random finite ones, in a piece at a
// nonzero base within its binding.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking to run-time checking — the Format
//            takes its type from its layout, as every format's does.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path —
//            decoded into registers, never written out expanded.
inline constexpr Format kBF16{kBF16Layout, shaders::bf16};

}  // namespace bllm::formats
