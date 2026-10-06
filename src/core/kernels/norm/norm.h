#pragma once

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// RMSNorm, with the residual add before it fused in (docs/architecture/
// kernel-fusions.md). One kernel for the attention, feed-forward and final
// norms of every architecture here, and for both regimes: a row's norm needs
// only that row.
//
// For each row it covers:
//   1. Where the launch adds: y, the last block's output, is added into X,
//      the hidden row, and X written back. Where the architecture normalizes
//      a block's output before adding it — Gemma 3's post-attention and
//      post-feed-forward norms — y is first normalized with its own gain.
//   2. normed = (X × r) × g, where r = 1 / sqrt(mean(X²) + ε) and g is the
//      norm's gain: llama.cpp's RMS_NORM then MUL, in that order. ε is the
//      file's (model description). Gemma 3's gains need no special case:
//      llama.cpp's converter stored them as 1 + w (arch/architecture.h).
// The first layer's attention norm does not add — the embedding wrote X —
// and the final norm covers only the step's last token, row tokens - 1,
// which is all the output head reads (interface.h).
//
//   - One workgroup of 256 invocations per row. Invocation i takes the row's
//     vec4s i, i + 256, …, holding them in registers, so X is read from
//     memory once; rows up to 4,096 wide, at most 4 vec4s an invocation,
//     cover every listed model (Llama 3.2 1B's 2,048 is the widest). The
//     width is an override constant, so each invocation's loops run
//     ceil(W / 1,024) times, fixed when the pipeline is built — once for
//     Qwen3, twice for Llama 3.2 and Gemma 3 — and the bounds check folds
//     away where 1,024 divides the width. A model has one width, so this
//     adds no pipelines.
//   - The sum of squares is reduced in workgroup memory, in a fixed tree:
//     each invocation's own vec4s in order, then halving across the
//     workgroup, 8 levels. No atomics (GDSA.5). A barrier where one
//     invocation reads what another wrote, and nowhere else: one before the
//     tree and one after each level, 9 a reduction; one more only between
//     the post-norm's reduction and the next, which rewrites the partial sum
//     the first was read from (GPU.8).
//   - Determinism (GDSA.2): run to run, and independent of the step's other
//     rows and its token count, since one workgroup reduces one row in one
//     fixed order — so a token's normed row, and the key and value computed
//     from it, are the same whether the token was prefilled or decoded. Not
//     bit for bit with llama.cpp or across GPUs, which sum in other orders.
//   - Accuracy, for finite rows whose squares stay finite in f32 — every
//     activation the listed models produce: each output within 2⁻¹⁸ of the
//     same computation in f64, measured against the largest output of its
//     row. That is the margin a 4,096-wide row's sum leaves (16 sequential
//     adds an invocation, then 8 tree levels, each a rounding of at most
//     2⁻²⁴), with the inverse square root's 2 units and the multiplies; WGSL
//     lets the compiler reassociate and does not fix the rounding direction,
//     so it is a margin the GPU test checks on the target, not a proof.
//     Against the row, not the element: adding y into X can cancel, and an
//     element near zero carries its row's error. A row whose squares
//     overflow f32 — about 10¹⁹ — is outside it; no listed model's
//     activations come near.
//   - Variants are override constants, so one source serves all three:
//     `add`, and `post_norm` (which requires `add`); `width` is one too. Every binding is
//     declared in every variant, so a launch that does not add still binds
//     the output buffer and, without a post-norm, binds its gain twice: both
//     read-only, so neither aliases a buffer the kernel writes.
//   - Constants (binding 1), as WGSL lays them out:
//       struct Norm { epsilon: f32 }
//     Bindings: 2 the gain, 3 the post-norm's gain, 4 the block's output y,
//     read-only; 5 the hidden buffer X; 6 the normed buffer.
//   - Gains are F32 and one piece: a norm's weight is one row of the hidden
//     width, far under a binding (format.h's F32).
//   - It reads the plan's `output` working buffer: a block's last matmul
//     writes its result there, kPrefillBlock rows of the hidden width, and
//     the norm after it adds it into X (residency/plan.h).
//
// What it costs, counted. Launches: two a layer and the final norm — 57 a
// pass for Qwen3 0.6B, about 86 µs of dispatch at 1.5 µs (interface.h).
// Bytes, a row, for Qwen3's 1,024 width: X read and written, y read, the
// gain read, normed written — 4 KiB each, 20 KiB; Gemma 3's post-norm adds
// its gain, 4.5 KiB at its 1,152, 27 KiB a row in all; the first layer's
// attention norm, which does not add, 12 KiB. One norm launch over a 512-row
// prefill step moves about 10 MiB, about 25 µs at 400 GB/s. A pass's 57 —
// one without the add, 55 with it over every row, the final norm's one row —
// move W × (1,112 T + 20) bytes for T rows and width W: about 556 MiB, 1.46
// ms, at T = 512, of which the gains, read again by every row but 4 KiB
// each, are 112 MiB; at decode, 1.1 MiB, under 3 µs, far below the
// launches' 86 µs. A row's latency is its reductions', not its
// bytes: one tree of 8 levels, or for Gemma 3's post-norm two dependent ones
// — y's scale is needed before X can be added to and reduced — 16 levels and
// two inverse square roots.
// Optimization (practice): the residual add and the gain ride in the norm's
// launch, as vLLM's fused_add_rms_norm and llama.cpp's RMS_NORM + MUL + ADD
// do, rather than in launches of their own (GPU.6).
// Optimization (practice): the width fixed at pipeline creation, not read
// from a uniform, so the loops run as many times as the row needs, not 4
// (GDSA.6's whole-tile accounting: an inactive iteration still costs its
// instructions). Gemma 3's 288 vec4s still leave 224 invocations idle in
// the second.
//
// Where WebGPU limits it, and what each limit costs here:
//   - Subgroup operations (subgroupAdd) are the `subgroups` feature, which
//     WebGPU leaves optional and not every browser offers. For cross-browser
//     compatibility the harness requires only WebGPU's defaults and no
//     optional feature (gpu/device_requirements.h), though the target
//     itself offers subgroups. So the reduction is the tree above, 9 barriers, where the practice of
//     llama.cpp's Metal kernel (simd_sum) and vLLM's (CUB's block reduce,
//     over warp shuffles) — a subgroup sum, the partials through workgroup
//     memory, a subgroup sum of those — needs 1.
//   - WGSL zero-fills workgroup memory before a workgroup runs, so each one
//     also stores its 1 KiB of partial sums and waits at one more barrier
//     before Step 1, though the kernel writes every entry before reading it.
// Within those limits the reduction follows the practice for a tree without
// subgroups (Harris, "Optimizing Parallel Reduction in CUDA"): sequential
// addressing, so active invocations stay contiguous (his kernel 3); each
// invocation's own vec4s summed before the tree (4); several vec4s an
// invocation for rows wider than 1,024 (7). His kernel 5, the last levels
// run without barriers within a warp, is the subgroup sum above. And one
// workgroup a row, one vec4 an invocation for Qwen3, as llama.cpp and vLLM
// size theirs: a 4 KiB row split across workgroups would need a second
// launch or atomics to combine its sums, slower for a decode step's one
// row, not faster.
//
// Verification the implementation is held to, on the GPU against an f64
// reference: every variant, at the widths of the listed models; rows of
// large and of tiny magnitude; a step of many rows and of one, each row's
// result identical in both; and the final norm touching only the last row.
//
// Guidelines, by corpus:
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level — as
//            above.
//     GDSA.5 Aggregate within the workgroup before touching global memory —
//            no atomics; one write a row.
//     GDSA.6 Count passes over global memory — the bytes above; and no
//            loop iterations beyond the row's.
//     GPU.5  Use workgroup memory where reuse or reordering pays — the
//            reduction's partial sums, 1 KiB.
//     GPU.6  Batch tiny GPU work — the add and the gain fused in.
//     GPU.8  Make barriers describe real hazards — none after the last
//            reduction, where nothing writes the partial sums again.

// One norm of the graph.
struct NormLaunch {
    const residency::WeightView& gain;            // F32, one row of the hidden width
    const residency::WeightView* post_gain;       // Gemma 3's post-norm of y, or null
    bool add;                                     // add y into X first; required by a post-norm
    residency::BufferRange output;                // y: the block's output
    residency::BufferRange hidden;                // X
    residency::BufferRange normed;
    float epsilon;
    Rows rows;                                    // LastToken for the final norm
};

// The launch for one norm. Preconditions: the gains are F32 views of the
// hidden width, at most 4,096; a post-norm only where the launch adds.
[[nodiscard]] Launch norm_launch(const NormLaunch& norm);

}  // namespace bllm::kernels
