#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// The matrix products: each layer's Q, K and V projection, output
// projection, gate and up projection with the activation, and down
// projection, and the output head (docs/architecture/kernel-fusions.md).
// For a weight W of N rows of K — a GGUF matrix lists K first, so row o is
// output o's K weights, contiguous — and input rows x of K floats:
//
//   y[t][o] = Σ_k W[o][k] × x[t][k]
//
// The weight stays in its file's format and is decoded in the kernel's load
// path through the format's unpack, 32 weights at a time (format.h); no
// weight is written out expanded (GDSA.18).
//
// Two forms, one per regime, each its own launch; a step runs one (the
// launch contract's regime, below). The head is the exception: it covers
// the step's last token alone, in both regimes, so its one launch is the
// decode form, run in every step.
// Both forms add each output's products in one order, fixed by K alone
// (GDSA.2). A row's K / 32 groups are cut into 32 ranges at fixed
// boundaries — range r is groups floor(r × G / 32) up to floor((r + 1) × G /
// 32), G = K / 32 — and
//
//   y = ((0 + s_0) + s_1) + … + s_31,  s_r = (((0 + w_a x_a) + w_b x_b) + …)
//
// each range's sum s_r taken from zero over its weights in order, and the
// ranges added in order. Each multiply-add is written as acc + w × x in
// both forms, so the compiler is given the same expression to contract or
// not.
//
//   - Decode, one token: a matrix times a vector, bound by reading each
//     weight once. A workgroup of 64 invocations is two sets of 32, each
//     set taking 4 rows; invocation r of a set takes range r of its 4 rows.
//     For each group of its range it loads the input's 32 matching floats
//     into registers once and multiplies them by the 4 rows' unpacked
//     groups, so the input is read once a set, not once a row, as
//     llama.cpp's Metal kernel reuses it across rows. The 4 rows' 32 range
//     sums meet in workgroup memory, and each row's first invocation adds
//     them in order: one barrier. Qwen3's 1,024-wide rows are 32 groups,
//     one a range, so a set's invocations read 32 adjacent blocks of each
//     of the format's streams: for Q4_0's codes, 512 bytes in 512 (GPU.2).
//     A wider row's ranges are 2 to 8 groups, so invocation r reads block
//     r × n + i at its i-th step: 512 bytes spread over n × 512 — 3 times
//     for Qwen3's 3,072-wide down projection, 8 for Llama 3.2's 8,192 —
//     each line's rest read by the same invocation's next n − 1 steps, from
//     the GPU's caches. The bytes from device memory are the weights, once.
//   - Prefill, two to 512 tokens: a tiled matrix product. A workgroup of 64
//     takes a tile of 32 tokens × 64 outputs and steps along K 32 at a time,
//     a group a step: the step's 32 × 32 input floats into workgroup memory,
//     coalesced, and the 64 rows' groups unpacked there, one group an
//     invocation, 8 KiB of f32; a barrier; then each invocation adds a
//     4-token × 8-output micro-tile's 1,024 multiply-adds from 12 reads of
//     workgroup memory a step of k, into each output's range sum; at a
//     range's last group it adds the range sum into the output's total and
//     starts the next from zero — two accumulators an output, 64 an
//     invocation; a barrier. 12.3 KiB of workgroup memory, its weight rows
//     padded a word against bank conflicts (GPU.5). Workgroups are numbered
//     token tile first, so the token tiles reading one weight tile run
//     together and find it in the GPU's caches.
//
// Epilogues — what a product writes — are variants, override constants:
//   - write: y to one buffer: the output projection and down projection
//     into the plan's `output`, which the next norm adds (norm.h); the head
//     into `logits`, for the step's last token only.
//   - QKV: the layer's Q, K and V weights as one product of their stacked
//     rows, one binding spanning the three, each read from its own place in
//     it (residency, below): outputs 0 .. H_q × d − 1 into `query`, the next
//     H_kv × d into `key`, the rest into `value` — the rope kernel's inputs.
//   - gated activation: the layer's gate and up weights as one product,
//     likewise, rows gate then up; a decode set's 4 rows are 2 gate rows
//     and the same 2 of up, a prefill tile's 64 outputs 32 and the same 32,
//     so the invocation that finishes an output holds both values, and writes
//     activation(gate) × up — SiLU for Qwen3 and Llama 3.2, GELU's tanh
//     form for Gemma 3, as the model description states — never gate or up
//     themselves. The down projection reads it.
//
//   - Pieces. A weight larger than one binding is split by rows
//     (residency/plan.h): one launch a piece, each covering its rows. The
//     head of Llama 3.2 is 2 launches, Gemma 3's 3. Every listed Q, K, V,
//     gate and up weight is one piece.
//   - Formats. Each format a weight uses is a pipeline: Q4_0 for most,
//     Q4_1 for a few layers' down projections, Q6_K and Q8_0 for heads.
//   - Variants are override constants — `rows` (N), `columns` (K), the
//     epilogue, the QKV split rows, the activation — so indices fold when
//     the pipeline is built.
//   - Constants (binding 1): struct Matmul { members: array<vec4<u32>, 3> }
//     — for each member of the product, or its one piece: its first word
//     within the binding, its blocks, for unpack, and its first output row
//     and row count.
//   - Bindings: 2 the weights (`weights`, unpack's): one piece, or the range
//     spanning a fused group's members; 3 the input; then the outputs,
//     written — one, or query, key and value. Five at most.
//
// What it asks of the other contracts:
//   - kernels/interface.h: a launch may run in one regime only, Decode a
//     step of one token, Prefill one of more (regime_for), and is not
//     dispatched in the other; Geometry gains the regime.
//   - residency/plan.h: a layer's Q, K and V, and its gate and up, each
//     lie in one buffer, within one binding's span from the first to the
//     last — the plan opens a new buffer for a group that would not fit the
//     open one. Each stays the weight it is, in file order, with its own
//     view, and upload and its routes are unchanged: a fused launch binds the
//     span, the tensors the file puts between its members included and
//     never read. In the listed files a span is at most 18.9 MB, Llama
//     3.2's gate and up; its QKV span, 5.9 MB, holds its 2.4 MB output
//     projection unread between them; both are far under a binding's 128
//     MiB. A group whose members differ in format or width, or whose span
//     exceeds a binding, is launched as separate products, an epilogue each.
//   - formats/format.h: unpack takes its piece's first word within the
//     binding — fn unpack(base: u32, blocks_in_piece: u32, group: u32) —
//     0 for a binding of one piece, as every caller's is today.
//   - residency/plan.h: the gate working buffer holds the activation, and
//     the up buffer is gone: 6 MiB fewer for Qwen3.
//   - model/model_description.h: the feed-forward activation; and describe
//     refuses attention and final logit softcapping, which no kernel
//     applies, rather than run a file that declares them without it.
//
// Determinism (GDSA.2): run to run, and batch-invariant — a token's
// outputs are the same bits decoded alone or prefilled in a step of any
// size, since both forms add its products in the order above, fixed by K
// and never by the step. Norm, rope and attention are batch-invariant too,
// so a token's whole pass is. Not bit for bit with llama.cpp, whose
// matrix-vector and matrix kernels add in orders of their own.

// Accuracy: each output within γ(2K) × Σ_k |W[o][k] × x[k]| of the same
// sum in f64 over the decoded weights, γ(n) = n u / (1 − n u) and u = 2⁻²⁴:
// the standard bound for a sum of K products whose multiplies and adds
// round separately, which a fused multiply-add only tightens — WGSL leaves
// contraction to the compiler. K is at most 8,192, Llama 3.2's down
// projection, so γ(2K) is at most 2⁻¹⁰ / (1 − 2⁻¹⁰), about 2⁻¹⁰. The weights decode as format.h states. As
// the norm's, a bound the GPU test checks on the target.
//
// What it costs, counted, for Qwen3.
//   - Decode. Weights read once: a layer's QKV 2.36 MB, output 1.18 MB,
//     gate and up 3.54 MB, down 1.77 MB at Q4_0 or 1.97 MB at Q4_1 — about
//     8.9 MB, 248 MB across 28 layers, 3 of whose down projections are Q4_1
//     — and the head's 127.6 MB: 376 MB, 0.94 ms at 400 GB/s, the weights'
//     share of interface.h's 380 MB file. Arithmetic, 2 × 0.6 × 10⁹ operations, is far below
//     it. Workgroups: QKV 512, output 128, gate and up 768, down 128, head
//     18,992. Each set of 32 invocations reads its whole input once, from the
//     GPU's caches, two a workgroup: 15 MiB a layer — QKV 4, output 2, gate
//     and up 6, down 3 — and 148 MiB for the head, against device memory's
//     8.9 MB and 127.6 MB of weights, each input being at most 12 KiB for
//     Qwen3. Each workgroup's 256 range sums pass through 1 KiB of
//     workgroup memory. Launches: 4 a layer and
//     the head's one, 113 a step, about 170 µs at 1.5 µs.
//   - Prefill, 512 tokens. 15.7 × 10⁶ weights a layer, 440 × 10⁶ across
//     the layers, each multiplied by 512 tokens: 450 × 10⁹ operations,
//     about 32 ms at the M3 Max's roughly 14 f32 TFLOPS (third-party
//     figure) if the tile ran at peak; the head adds one token's. How near
//     peak it runs is the tile's: 2.7 multiply-adds a read of workgroup
//     memory, and 2 barriers a step of 32 along K — 64 for Qwen3's
//     1,024-wide inputs — and the registers an invocation holds: 64
//     accumulators, the range sum and total of its 32 outputs that batch
//     invariance asks, and 12 operands, about 76 values, where 32
//     accumulators would serve a kernel free to add in its own order; more
//     registers can mean fewer resident workgroups (GPU.3). Each weight is
//     decoded once a token tile, 16 times for 512 tokens: about 4 GB
//     through the GPU's caches, and from device
//     memory about once, a projection's weights, at most 3.5 MB, staying in
//     them while its token tiles run. Workgroups: QKV 1,024 of the tile.
// Optimization (practice): weights decoded in the load path, never
// expanded (GDSA.18) — llama.cpp's mul_mv and mul_mm do the same.
// Optimization (practice): Q, K and V as one product and gate and up as
// one, so a layer runs 2 products where it would run 5 — vLLM's
// QKVParallelLinear and MergedColumnParallelLinear, which concatenate the
// weights at load; here one binding spans them, so upload writes the file
// as it lies — 84 launches a step for Qwen3, about 126 µs (GPU.6).
// Optimization (practice): the activation is the gate-and-up product's
// epilogue, so gate and up are never written and read back, nor read by a
// launch of its own: 48 KiB a token and a layer saved of the 60 KiB
// separate kernels would move (GDSA.6).
// Optimization (practice): decode reuses each input value it loads for 4
// rows, and prefill each decoded weight for 32 tokens and each input for 64
// outputs (GPU.2, GPU.5).
// Optimization (practice): one order of addition for both forms, fixed by
// K, so a token's projections do not depend on its step — the fixed split
// Thinking Machines' batch-invariant kernels use (GDSA.2).
//
// Where WebGPU limits it, and what each limit costs here:
//   - WebGPU has no matrix-multiply units for f32 — Chrome's subgroup
//     matrices are an experiment, not cross-browser — so prefill's
//     multiply-adds run on the shader cores, one at a time an invocation.
//   - Workgroup memory is 16 KiB at the defaults the harness keeps for
//     cross-browser compatibility: a 32 × 64 tile, where native kernels
//     take 64 × 64 to 128 × 128 and reuse each loaded value two to four
//     times more.
//   - Subgroup operations are optional, for the same reason: decode's 32
//     range sums a row meet in workgroup memory, one barrier and 31 adds by
//     one invocation, where llama.cpp's Metal kernel sums a row across a
//     SIMD-group.
//
// Levers not taken:
//   - Attention's combine folded into the output projection's load: every
//     one of its 128 decode workgroups needs the whole input, so each would
//     fold every split itself — 16 MiB of partials read a layer at 4,096
//     tokens, against the combine's 128 KiB once and one 1.5 µs launch.
//   - An order of addition each form chooses for itself: prefill's 32
//     extra accumulators an invocation saved, decode's ranges free to
//     interleave, at the cost of a token's bits depending on its step.
//   - Larger tiles from a raised workgroup-memory limit: the harness asks
//     for WebGPU's defaults alone.
//
// Verification the implementation is held to, on the GPU against f64 over
// the format's CPU-decoded weights: each form, each epilogue, each listed
// format, at each listed model's widths; prefill steps of 2, 31, 32, 33 and
// 512 tokens; a weight split into pieces; QKV outputs landing in their
// buffers; the activation for SiLU and GELU; and a token's outputs the same
// bits decoded alone and prefilled in steps of 2, 33 and 512, at widths
// whose ranges are one group, several, and of unequal sizes (Gemma 3's
// 1,152 and 6,912). On the CPU: each launch's geometry, regime, bindings
// and variant, and the range boundaries.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.12   Don't make data members const or references in a copyable
//            type — MatmulLaunch holds values and a pointer.
//     F.20   Prefer return values to out parameters.
//     I.4    Make interfaces precisely and strongly typed — the epilogue is
//            an enumeration.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path.
//     GDSA.6 Count passes over global memory — the bytes above, and the
//            activation never written.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            batch-invariant: one order, fixed by K, never by the step.
//     GPU.2  Shape data for coalesced lane access — decode's rows.
//     GPU.5  Use workgroup memory where reuse pays — prefill's tiles.
//     GDSA.16 Stream through on-chip tiles — prefill steps along K through
//            workgroup memory, each output written once.
//     GPU.6  Batch tiny GPU work — fused weights, fewer launches.

// What a product writes.
enum class Epilogue {
    Write,
    QKV,
    GatedActivation,
};

// One matrix product of the graph.
struct MatmulLaunch {
    // The weight, or a fused group's members in output order: Q, K, V, or
    // gate, up. Never null; members are one piece each, in one buffer.
    std::array<const residency::WeightView*, 3> weights;
    std::uint32_t members;                           // 1, 2 or 3
    residency::BufferRange input;                    // rows of K floats
    // Write: outputs[0]. QKV: query, key, value. GatedActivation: the
    // activation, in outputs[0].
    std::array<residency::BufferRange, 3> outputs;
    Epilogue epilogue;
    std::uint32_t key_rows_from;                     // QKV: the first K row, H_q × d
    std::uint32_t value_rows_from;                   // QKV: the first V row
    model::FeedForwardActivation activation;         // GatedActivation's
    Rows rows;                                       // LastToken for the head
};

// The product's launches: for each piece of the weight, a decode form and a
// prefill form, each in its regime; or for the head, the decode form alone,
// in every regime. Preconditions: K is a whole number of 32-weight groups;
// a fused group's members lie in one buffer within a binding's span, and
// their rows are the epilogue's.
[[nodiscard]] std::vector<Launch> matmul_launches(const MatmulLaunch& matmul);

}  // namespace bllm::kernels
