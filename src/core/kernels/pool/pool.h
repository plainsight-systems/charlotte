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
//     One workgroup of W / 4 invocations, at most 1,024 — W at most 4,096 —
//     invocation c owning vec4 c of the row: it adds the rows' vec4s at c in
//     row order, one chain of N adds from zero, then multiplies by 1 / N,
//     computed once in f32. No reduction crosses invocations, so it needs no
//     workgroup memory and no barrier.
//   - `unit`: row 0 of `projected`, D floats, divided by its Euclidean
//     length into `embedding` — llama.cpp's common_embd_normalize with its
//     default, the L2 norm, and sentence-transformers' Normalize. A row of
//     length 0 is written as zeros, as llama.cpp writes it, rather than
//     divided by zero. One workgroup of 256 invocations: each its row's
//     vec4s c, c + 256, …, squares summed in order, then a tree in
//     workgroup memory, 8 levels, as kernels/norm/norm.h reduces a row; the
//     inverse square root once; each invocation scales the vec4s it read.
//
//   - Variants are override constants — `width`, W or D — set on every
//     launch, as WebKit requires
//     (docs/research/2026-10-09-webkit-shader-compiler.md).
//   - Constants (binding 1): none read; a launch binds its slot as every
//     launch does (interface.h).
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
//     invocations each a chain of N dependent adds, 512 at N = 512, the
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
[[nodiscard]] Launch mean_launch(const residency::BufferRange& normed, const residency::BufferRange& pooled,
                                 std::uint32_t width);
[[nodiscard]] Launch unit_launch(const residency::BufferRange& projected, const residency::BufferRange& embedding,
                                 std::uint32_t dimensions);

}  // namespace bllm::kernels
