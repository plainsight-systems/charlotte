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
// launch contract's token range, below), prefill in one of three tile widths.
// The head is the exception: it covers the step's last token alone, in
// both regimes, so its one launch is the decode form, run in every step.
// Both forms add each output's products in one order, fixed by K alone
// (GDSA.2). A row's K / 32 groups are cut into 32 ranges at fixed
// boundaries — range r is groups floor(r × G / 32) up to floor((r + 1) × G /
// 32), G = K / 32 — and
//
//   y = ((0 + s_0) + s_1) + … + s_31,  s_r = (((0 + w_a x_a) + w_b x_b) + …)
//
// each range's sum s_r taken from zero over its weights in order, and the
// ranges added to a total that starts at zero, in order: 32 adds. Each
// multiply-add is an explicit fma in both forms. Written as acc + w × x,
// the target's compiler contracted it in one pipeline and not the other,
// and the decode and prefill bits differed; an explicit fma it fuses in
// both. The gated activation likewise takes its e^x from explicit
// arithmetic, every multiply-add an fma: the built-in exp gave different
// bits in the two pipelines. WGSL lets fma() round twice and lets each
// pipeline reassociate, so equal bits are what the target's compiler gives,
// which the GPU test checks, not what WGSL promises.
//
//   - Decode, one token: a matrix times a vector, bound by reading each
//     weight once. A workgroup of 64 invocations is two sets of 32, each
//     set taking 4 rows; invocation r of a set takes range r of its 4 rows.
//     For each group of its range it loads the input's 32 matching floats
//     into registers once and multiplies them by the 4 rows' unpacked
//     groups, so the input is read once a set, not once a row, as
//     llama.cpp's Metal kernel reuses it across rows. The workgroup's 8
//     rows' range sums meet in workgroup memory, and invocations 0 to 7
//     each add one row's in order — for the gated activation, invocations
//     0 to 3 one gate row and its up row each: one barrier, then 32 adds,
//     or 64, in parallel rather than one invocation's 128 a set. Qwen3's
//     1,024-wide rows are 32 groups, one a range, so a set's invocations
//     read 32 adjacent blocks of each
//     of the format's streams: for Q4_0's codes, 512 bytes in 512 (GPU.2).
//     A wider row's ranges are 2 to 8 groups, so invocation r reads block
//     r × n + i at its i-th step: 512 bytes spread over n × 512 — 3 times
//     for Qwen3's 3,072-wide down projection, 8 for Llama 3.2's 8,192 —
//     each line's rest read by the same invocation's next n − 1 steps, from
//     the GPU's caches. The bytes from device memory are the weights, once.
//     A product whose workgroups of 8 rows would number under 512 gives
//     each set fewer rows: the most, of 4, 3, 2 and 1, that still makes 512
//     workgroups of 2 sets, or 1 when none does — the Write and QKV
//     epilogues, by an override constant, `set_rows`. Qwen3's output and
//     down projections, 1,024 rows, take 1, 512 workgroups, not 128; Llama
//     3.2's QKV, 3,072 rows, takes 3, and its output and down, 2,048, take
//     2, 512 workgroups each; Gemma 3's QKV, 1,536, and output and down,
//     1,152, take 1, 768 and 576. Each row is still summed by 32
//     invocations, a range each, in the same order. The gated product keeps
//     its 4, which its epilogue needs in one set; the listed ones run 768
//     workgroups or more.
//   - Prefill, two to 512 tokens: a tiled matrix product. A workgroup takes
//     a tile of τ tokens × 64 outputs, τ 8, 16 or 32, and steps along K 32
//     at a time, a group a step. Its 4τ invocations — 32, 64 or 128, under
//     WebGPU's default limit of 256 — each own a micro-tile of 4 consecutive
//     tokens × 4 outputs: invocation t token quad t / 16 and outputs —
//     row slots — l + 16j for its lane l = t % 16. So every tile's
//     invocations run one inner loop, its bounds WGSL constants rather than
//     the tile's override: for each of the step's 8 k-quads, 4 vec4 reads of
//     its outputs' weights and 4 of its tokens' inputs, the weights
//     transposed in registers so each k's 4 weights are one vec4, then 16
//     vec4 fmas, k in order, into each token's 4 range sums — 64
//     multiply-adds from 8 reads of workgroup memory. Each
//     token's range sums and totals are named vec4s, 8 of them, 32
//     accumulators: the 4 × 4 micro-tile llama.cpp's and MLC's WebGPU
//     kernels hold, never an array indexed by loop counters, which the
//     compiler may keep in memory rather than registers (GPU.3). At a
//     range's last group each total adds its range sum, componentwise, and
//     the range sum restarts from zero, so each output's order of addition
//     is decode's.
//     Before each step's products the tile is staged in workgroup memory as
//     the decode and the input give it, along K, each invocation writing
//     whole vec4s: a write to one component of a vector in workgroup memory
//     may write all four, so two invocations never share one (SIMD.2). Row
//     slot s's group, decoded, is its 8 k-quads at vec4 9s, a vec4 of
//     padding a slot so the 16 slots a read takes spread over the banks;
//     token i's 32 inputs are its 8 at vec4 8i. The invocations stride over
//     both jobs together — 64 row decodes and 8τ input vec4s, each read
//     coalesced along its row — then a barrier, the products, a barrier.
//     Within a read of the products the 16 lanes of a token quad take 16
//     slots 9 vec4s apart: over 32 banks of a word, 8 vec4s a cycle, lanes
//     l and l + 8 share a bank phase, so the read takes two cycles, the
//     floor for 16 16-byte loads; a 32-lane SIMD-group's two token quads
//     read two vec4s, each broadcast (GPU.5). Workgroup memory:
//     9 KiB of weights and τ / 8 KiB of inputs, 10, 11 or 13 KiB.
//     Workgroups are numbered token tile first, so the token tiles reading
//     one weight tile are dispatched together and may find it in the GPU's
//     caches (costs, below).
//     Shorter steps take narrower tiles of the same 64 outputs, 16 tokens or
//     8, with fewer invocations of the same micro-tile (GDSA.6): a
//     step takes the narrowest tile that holds it whole, 8 for 2 to 8 tokens
//     and 16 for 9 to 16, and the 32-token tile from 17. Each token tile
//     reads and decodes the whole weight, so the rule first keeps a step to
//     the fewest weight passes — one up to 32 tokens — and among those takes
//     the narrowest. A workgroup with a tile of τ tokens, for each group of
//     32 along K, does 2,048τ multiply-adds and 256τ vec4 reads of
//     workgroup memory, loads 32τ input floats, decodes 64 weight blocks and
//     passes two barriers: a narrower tile holding the step does fewer
//     multiply-adds, reads and loads, and the same decodes and barriers.
//     A step's last token tile holds its remainder: an invocation whose token
//     quad lies wholly past the step stages and passes both barriers, as
//     every invocation must, but skips the products, so a tile runs the
//     products of its live quads alone. The target issues a 32-lane
//     SIMD-group's work together, two token quads, so a step's products run
//     to the next 8 tokens — at most 7 padded a step: at 33 tokens the
//     second tile's first SIMD-group of its four; at 2 tokens the 8-token
//     tile's one, the guard skipping nothing. Guarding each token's chain
//     of fmas instead would cut that to the next quad, in a step's last
//     tile alone, for a branch a token on every k-quad of every tile, the
//     full tiles' included; not taken. Past 32 tokens
//     the 32-token tile keeps a step to the fewest weight passes, decodes
//     and barriers, and the guard its multiply-adds to the step's. Tile
//     width and the guard change no output's order of addition, so every
//     tile's bits are the others' and decode's.
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
//     staged so lane c's slots hold gate rows 2c and 2c + 1 and the same up
//     rows, so the invocation that finishes an output holds both values, and writes
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
//   - Each form and epilogue is an entry point of one module, binding only
//     the outputs it writes; `columns` (K), a write's `out_width` and the
//     activation are override constants, so indices fold when the pipeline
//     is built.
//   - Constants (binding 1): struct Matmul { members: array<vec4<u32>, 3> }
//     — for each member of the product, or its one piece: its first word
//     within the binding, its blocks, for unpack, and its first output row
//     and row count.
//   - Bindings: 2 the weights (`weights`, unpack's): one piece, or the range
//     spanning a fused group's members; 3 the input; then the outputs,
//     written — one, or query, key and value. Five at most.
//
// What it asks of the other contracts:
//   - kernels/interface.h: a launch may run in a range of token counts
//     only — a regime's, Decode a step of one token and Prefill one of more
//     (regime_for, tokens_of), or part of one, as the three prefill tiles
//     take — and is not dispatched outside it; Geometry gains the range.
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
// Determinism (GDSA.2): run to run, and batch-invariant on the target — a
// token's outputs are the same bits decoded alone or prefilled in a step of
// any size, since both forms add its products in the order above, fixed by
// K and never by the step, and the GPU test holds the target's compiler to
// contracting the two alike; WGSL does not promise it, so another browser's
// compiler could differ in the low bits, which that test would show. Norm,
// rope and attention are batch-invariant too, so a token's whole pass is. Not bit for bit with llama.cpp, whose
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
//     it. Workgroups: QKV 512, output 512, gate and up 768, down 512, head
//     18,992. Each set of 32 invocations reads its whole input once, from the
//     GPU's caches, two a workgroup: 30 MiB a layer — QKV 4, output 8, gate
//     and up 6, down 12 — and 148 MiB for the head, against device memory's
//     8.9 MB and 127.6 MB of weights, each input being at most 12 KiB for
//     Qwen3. Each workgroup's range sums, 256 or 64, pass through at most
//     1 KiB of workgroup memory. Launches: 4 a layer and
//     the head's one, 113 a step, about 170 µs at 1.5 µs.
//   - Prefill, 512 tokens. 15.7 × 10⁶ weights a layer, 440 × 10⁶ across
//     the layers, each multiplied by 512 tokens: 450 × 10⁹ operations,
//     about 32 ms at the M3 Max's roughly 14 f32 TFLOPS (third-party
//     figure) if the tile ran at peak; the head adds one token's. How near
//     peak it runs is the tile's: 8 multiply-adds a vec4 read of workgroup
//     memory, and 2 barriers a step of 32 along K — 64 for Qwen3's
//     1,024-wide inputs — and the registers an invocation holds: 32
//     accumulators, the range sum and total of its 16 outputs that batch
//     invariance asks, and a k-quad's 32 operand floats, about 64 values,
//     all at constant indices; a kernel free
//     to add in its own order would keep the 16 totals alone, and more
//     registers can mean fewer resident workgroups (GPU.3). Each input
//     float is loaded once an output tile: MACs / 64 loads, 14 GB of f32
//     from the GPU's caches for a 512-token step, beside the weights'
//     3.97 GB below. Each weight is
//     decoded once a token tile, 16 times for 512 tokens: 3.97 GB of weight
//     loads. From device memory that is between 248 MB, if a weight tile
//     stays in the GPU's caches while its 16 token tiles run, and the whole
//     3.97 GB, about 10 ms at 400 GB/s, if none does. Workgroups are
//     numbered token tile first so that tiles reading one weight tile are
//     dispatched together; WebGPU promises neither the order they run in
//     nor what the caches keep, so where in that range a step falls is the
//     target's, to be measured. Workgroups: QKV 1,024 of the tile.
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
// rows, and prefill each decoded weight for its tile's 32 tokens, or 16 or
// 8, and each input for 64 outputs (GPU.2, GPU.5).
// Optimization (practice): a decode product too small to make 512
// workgroups of 8 rows takes fewer rows a set, the most that reaches 512,
// as llama.cpp's WebGPU mat-vec gives a row 64 invocations and ONNX
// Runtime's 16, where 4 rows a set gives it 8: in the profile
// (docs/research/2026-10-07-forward-pass-profile.md) Qwen3's products of
// 128 workgroups read their weights at 107 to 180 GB/s, those of 512 and
// 768 at 260 to 340, and the head's 18,992 at 349 to 383 (GPU.3). Its cost
// is each set reading its input for fewer rows, from the GPU's caches: for
// Qwen3, 15 MiB more a layer; no product past 512 is split further.
// Optimization (practice): a prefill invocation's micro-tile is 4 tokens ×
// 4 outputs, its accumulators named vec4s and its loops' bounds constants,
// as llama.cpp's, MLC's and ONNX Runtime's WebGPU kernels keep 16 to 32
// accumulators an invocation, unrolled or constant-indexed: the 32-token
// tile's 4 × 8 micro-tile, 64 accumulators in arrays indexed by loop
// counters, took each launch of a 32-token step 13.7 to 16 times as long
// as the 16-token tile's 2 × 8 took a 16-token step's, for twice the
// tokens (docs/research/2026-10-07-forward-pass-profile.md) (GPU.3).
// Optimization (practice): the staged tiles are read as whole vec4s along
// K, 8 for a k-quad's 64 multiply-adds, where 4 tokens and 4 outputs read
// as scalars would be 32 (SIMD.2, GPU.5).
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
//     range sums a row meet in workgroup memory, one barrier and 32 adds by
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
//   - 32-output tiles, llama.cpp's and MLC's 32 × 32: each input float is
//     loaded once 32 outputs, not 64 — 28 GB from the GPU's caches for a
//     512-token Qwen3 step against 14. They stage inputs at f16, halving
//     that; here that needs the optional shader-f16 feature, and inputs
//     rounded to f16 would no longer be the f32 decode reads, so a token's
//     bits would depend on its step.
//   - f16 accumulators, ONNX Runtime's default: the same feature, and
//     llama.cpp moved to f32 after f16 sums gave NaNs on Qwen models.
//
// Verification the implementation is held to, on the GPU against f64 over
// the format's CPU-decoded weights: each form, each epilogue, each listed
// format, at each listed model's widths; decode with 4, 3, 2 and 1 rows a
// set, at the row counts where each begins; prefill steps of 2 and 8 tokens,
// in the 8-token tile of 32 invocations, 9 and 16, in the 16-token of 64,
// and 17, 33 and 64, in the 32-token of 128 — every token quad and output
// quad of each; a weight
// split into pieces; QKV outputs landing in their buffers; the activation
// for SiLU and GELU; and a token's outputs the same bits decoded alone and
// prefilled in each of those steps, at tokens on each tile's edges, at widths
// whose ranges are one group, several, and of unequal sizes (Gemma 3's
// 1,152 and 6,912). On the CPU: each launch's geometry, token range, bindings
// and variant, and the range boundaries.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.12   Don't make data members const or references in a copyable
//            type — MatmulLaunch holds values and a pointer.
//     F.20   Prefer return values to out parameters.
//     I.4    Make interfaces precisely and strongly typed — the epilogue is
//            an enumeration.
//     ES.45  Avoid magic constants — the micro-tile's 4 × 4 and the tiles'
//            widths are named.
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
//     GPU.3  Treat occupancy as latency-hiding budget — 32 accumulators an
//            invocation, named, not 64 in arrays.
//     SIMD.2 Lay out data so vector loads are linear — whole vec4s along K.

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
    model::FeedForwardActivation activation;         // GatedActivation's
    Rows rows;                                       // LastToken for the head
};

// The product's launches: for each piece of the weight, a decode launch,
// and a prefill launch for each tile width, each in its token range; or for the
// head, the decode form alone, in every step. Preconditions: K is a whole number of 32-weight groups;
// a fused group's members lie in one buffer within a binding's span, and
// their rows are the epilogue's.
[[nodiscard]] std::vector<Launch> matmul_launches(const MatmulLaunch& matmul);

}  // namespace bllm::kernels
