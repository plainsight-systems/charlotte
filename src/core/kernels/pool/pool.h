#pragma once

#include <cstdint>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// Pooling: a step's rows reduced to the one vector an encoder returns
// (graph/encoder.h). Two entry points of one module, each one launch:
//
//   - `mean`: the mean of rows 0 .. N − 1 of `normed`, W floats each, written
//     at row N − 1 of `pooled` — where a launch over the last token reads its
//     input, so the output projection after it runs as a decoder's head does
//     (kernels/interface.h, Rows::LastToken). Mean pooling over every token,
//     the text's ends included, as sentence-transformers' Pooling with
//     include_prompt and llama.cpp's LLAMA_POOLING_TYPE_MEAN take it.
//     One workgroup of 256 invocations, WebGPU's default
//     maxComputeInvocationsPerWorkgroup, which the harness keeps
//     (gpu/device_requirements.h): invocation c owns the row's vec4s c, c +
//     256, …, at most 4 for W at most 4,096 — one for EmbeddingGemma 2's 512,
//     half the invocations idle. Each adds its vec4s down the rows in row
//     order, one chain of N adds from zero a vec4, then multiplies by 1 /
//     N, computed once in f32. No reduction crosses invocations, so it needs
//     no workgroup memory and no barrier.
//   - `unit`: row 0 of `projected`, D floats, divided by its Euclidean
//     length into `embedding` — llama.cpp's common_embd_normalize with its
//     default, the L2 norm, and sentence-transformers' Normalize. A row of
//     length 0 is written as zeros, as llama.cpp writes it, rather than
//     divided by zero. One workgroup of 256 invocations: each its row's
//     vec4s c, c + 256, …, squares summed in order, then a tree in
//     workgroup memory, 8 levels, as kernels/norm/norm.h reduces a row; the
//     inverse square root once; each invocation scales the vec4s it read.
//
//   - Constants (binding 1): struct Pool { vec4s: u32, pooling: u32,
//     epsilon: f32 }, the row's width over 4 — a loop bound, not a
//     workgroup array's size, so a uniform serves and one pipeline each for
//     any width — then grid's k and ε, which mean and unit leave unread.
//   - `grid`, a vision encoder's (graph/vision.h): soft token r, at column
//     u and row v of the pooled grid of C / k columns, the step's `columns`
//     C, is the mean of its k × k patches of the input, dy then dx in order,
//     each vec4 summed then divided by k², as ggml's average pool; then the
//     row RMS-normalized without a gain — x × inverseSqrt(mean of x² + ε),
//     unit's tree for the sum — at ε the file's over W, llama.cpp's √W scale
//     before that norm folded in, so no pass multiplies by it. One workgroup
//     of 256 a pooled row, its rows the step's patches over k² (row_divisor,
//     kernels/interface.h): a launch over every row, not the last. Bindings 2
//     the patches, read-only, and 3 the pooled rows, written. For W = 768,
//     192 working invocations, 9 reads of a vec4 and one write each.
//   - Both are launches over the last token (interface.h): one row's
//     workgroups, one, in a step that asks for logits, as an encoder's
//     one step does.
//   - Bindings: `mean` 2 `normed`, read-only, and 3 `pooled`, written;
//     `unit` 2 `projected`, read-only, and 3 `embedding`, written.
//
// Determinism (GDSA.2): run to run — each mean one chain in row order, the
// length one fixed tree.
//
// Accuracy, against f64 from the same rows: each mean within γ(N + 1) ×
// mean_i |x_i| of the row's components, γ(n) = n 2⁻²⁴ / (1 − n 2⁻²⁴); each
// unit component within norm.h's margin for a reduction of D squares. A
// margin the GPU test checks on the target, not a proof.
//
// What it costs, counted, at N tokens, for W = 512 and D = 768:
//   - `mean`: reads N × W × 4 bytes — 1 MiB at N = 512 — writes 2 KiB; 128
//     working invocations each a chain of N dependent adds, 512 at N = 512, the
//     launch's latency: about the time of 512 dependent loads and adds, a
//     few microseconds, against an encode of tens of milliseconds
//     (runtime/embedder.h).
//   - `unit`: reads and writes 3 KiB; 9 barriers.
//   - Launches: 2 an encode, about 3 µs at 1.5 µs each.
// The mean is one workgroup, not split across rows with partial sums and a
// second launch: a split would spread 1 MiB of reads over more of the GPU,
// saving at most those few microseconds, for a launch and a fixed fold of
// its own.
//
// Where WebGPU limits it: subgroup operations are optional, for
// cross-browser compatibility, so `unit`'s length is the tree above, 9
// barriers, where a subgroup sum would need 1, as norm.h states for its
// own.
//
// Verification the implementation is held to, on the GPU against f64: `mean`
// at N of 1, 2, 33 and 512, written at row N − 1 and nowhere else; `unit` on
// a row of large values, of tiny ones, and of zeros, written as zeros. On
// the CPU: each launch's geometry and bindings.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.20   Prefer return values to out parameters — the launches are
//            returned.
//     I.4    Make interfaces precisely and strongly typed.
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level — run
//            to run, as above.
//     GDSA.5 Aggregate within the workgroup before touching global memory —
//            no atomics; one write a component.
//     GPU.6  Batch tiny GPU work — the mean as one launch, counted above.

// The pooling launches, `mean` then `unit`. Preconditions: `normed` and
// `pooled` hold kPrefillBlock rows of `width` floats, `projected` and
// `embedding` one row of `dimensions`; both at most 4,096 and multiples of 4.
// Each launch is one workgroup of 256 invocations, within WebGPU's default
// limit at every width allowed.
[[nodiscard]] Launch mean_launch(const residency::BufferRange& normed, const residency::BufferRange& pooled,
                                 std::uint32_t width);
// A vision step's soft tokens' rows: each of its patches' pooled grid's
// rows, the mean of its k × k patches of `patches`, W floats each,
// RMS-normalized without a gain at `epsilon` — the file's ε over W, the √W
// scale llama.cpp applies before that norm folded in (graph/vision.h) —
// into `pooled`. One workgroup of 256 a pooled row, the step's patches
// over k² rows (row_divisor k²). Preconditions: W at most 4,096 and a
// multiple of 4; the step's `columns` a multiple of k, and its patches of
// k² × the pooled rows.
[[nodiscard]] Launch grid_launch(const residency::BufferRange& patches, const residency::BufferRange& pooled,
                                 std::uint32_t width, std::uint32_t pooling, float epsilon);

[[nodiscard]] Launch unit_launch(const residency::BufferRange& projected, const residency::BufferRange& embedding,
                                 std::uint32_t dimensions);

}  // namespace bllm::kernels
