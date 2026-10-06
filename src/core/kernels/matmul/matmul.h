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
//   - Decode, one token: a matrix times a vector, bound by reading each
//     weight once. Workgroups of 64 invocations take 8 rows, 8 invocations
//     a row; invocation l of a row takes the row's groups l, l + 8, …, each
//     an unpack of 32 weights and the input's 32 matching floats, summing
//     its dot products in order; the row's 8 partial sums meet in workgroup
//     memory and the first invocation adds them in order — one barrier.
//     Adjacent invocations take adjacent groups, so a row's eight read 8
//     consecutive blocks of each of the format's streams (GPU.2). The input
//     is read by every workgroup from the GPU's caches: at most 12 KiB.
//   - Prefill, two to 512 tokens: a tiled matrix product. A workgroup of 64
//     takes a tile of 32 tokens × 64 outputs and steps along K 32 at a time:
//     the step's 32 × 32 input floats into workgroup memory, coalesced, and
//     the 64 rows' groups unpacked there, one group an invocation, 8 KiB of
//     f32; a barrier; then each invocation adds a 4-token × 8-output
//     micro-tile's 1,024 multiply-adds from 12 reads of workgroup memory a
//     step of k; a barrier. 12.3 KiB of workgroup memory, its weight rows
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
//     likewise, rows gate then up; a decode workgroup's 8 rows are 4 gate rows and the same
//     4 of up, a prefill tile's 64 outputs 32 and the same 32, so each
//     invocation holds both values of the outputs it writes, and writes
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
// Determinism (GDSA.2): run to run. Each output is one invocation's sum in
// a fixed order — in prefill, along K; in decode, along each invocation's
// groups, then its row's 8 partials in order — so within a regime a token's
// outputs do not depend on the step's other tokens. Across regimes they
// differ in the low bits: a token's projections decoded and prefilled are
// not bit for bit equal, as in llama.cpp, whose matrix-vector and matrix
// kernels differ the same way. Equal bits would need the prefill tile to
// keep each output's 8 decode partials apart — 256 accumulators an
// invocation where it keeps 32 — or a decode that sums each row in one
// invocation, 512 invocations for Qwen3's QKV rows reading their rows'
// bytes far apart. So norm, rope and attention give a token the same bits
// however a prefill is chunked, and decode and prefill agree to rounding.
//
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
//     18,992. Every workgroup reads its whole input from the GPU's caches:
//     7.5 MiB a layer and 74 MiB for the head, against device memory's 9 MB
//     and 127.6 MB, each input being at most 12 KiB. Launches: 4 a layer and
//     the head's one, 113 a step, about 170 µs at 1.5 µs.
//   - Prefill, 512 tokens. 15.7 × 10⁶ weights a layer, 440 × 10⁶ across
//     the layers, each multiplied by 512 tokens: 450 × 10⁹ operations,
//     about 32 ms at the M3 Max's roughly 14 f32 TFLOPS (third-party
//     figure) if the tile ran at peak; the head adds one token's. How near
//     peak it runs is the tile's: 2.7 multiply-adds a read of workgroup
//     memory, and 2 barriers a step of 32 along K — 64 for Qwen3's
//     1,024-wide inputs. Each weight is decoded once a token tile, 16 times
//     for 512 tokens: about 4 GB through the GPU's caches, and from device
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
// Optimization (practice): decode reads each weight row with eight
// adjacent invocations, and prefill reuses each decoded weight for 32
// tokens and each input for 64 outputs (GPU.2, GPU.5).
//
// Where WebGPU limits it, and what each limit costs here:
//   - WebGPU has no matrix-multiply units for f32 — Chrome's subgroup
//     matrices are an experiment, not cross-browser — so prefill's
//     multiply-adds run on the shader cores, one at a time an invocation.
//   - Workgroup memory is 16 KiB at the defaults the harness keeps for
//     cross-browser compatibility: a 32 × 64 tile, where native kernels
//     take 64 × 64 to 128 × 128 and reuse each loaded value two to four
//     times more.
//   - Subgroup operations are optional, for the same reason: decode's 8
//     partials meet in workgroup memory, one barrier, where llama.cpp's
//     Metal kernel sums a row across a SIMD-group.
//
// Levers not taken:
//   - Attention's combine folded into the output projection's load: every
//     one of its 128 decode workgroups needs the whole input, so each would
//     fold every split itself — 16 MiB of partials read a layer at 4,096
//     tokens, against the combine's 128 KiB once and one 1.5 µs launch.
//   - Decoded and prefilled projections equal to the bit: the costs under
//     Determinism.
//   - Larger tiles from a raised workgroup-memory limit: the harness asks
//     for WebGPU's defaults alone.
//
// Verification the implementation is held to, on the GPU against f64 over
// the format's CPU-decoded weights: each form, each epilogue, each listed
// format, at each listed model's widths; prefill steps of 2, 31, 32, 33 and
// 512 tokens; a weight split into pieces; QKV outputs landing in their
// buffers; the activation for SiLU and GELU; and a token's prefill outputs
// the same in steps of 3 and of 512. On the CPU: each launch's geometry,
// regime, bindings and variant.
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
//     GDSA.2 Declare each floating-point reduction's determinism level.
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
