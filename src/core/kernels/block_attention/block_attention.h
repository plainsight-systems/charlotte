#pragma once

#include <array>
#include <cstdint>

#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// Block attention: every token of a step attending to every token of the
// same step, either side, within the layer's window — an encoder's
// attention (graph/encoder.h). Between rope's encode form, which normalized
// and rotated the step's queries and keys in place and normalized its
// values, and the output projection. It reads no KV cache: the keys and
// values are the step's own working buffers, f32 as rope left them.
//
// For a query q at row i of a step of N tokens, of a head whose key-value
// head holds keys k_j and values v_j at rows j:
//
//   s_j = (q · k_j) × scale              for j < N with |i − j| <= h
//   out = Σ_j softmax(s)_j × v_j
//
// h half the layer's window: llama.cpp's symmetric window
// (LLAMA_SWA_TYPE_SYMMETRIC), keys at most n_swa / 2 either side — 512 for
// EmbeddingGemma 2's sliding layers, whose files declare 1,024. A full
// layer's window is its trained context (model_description.h), so h covers
// every row. scale is the model's, 1 for EmbeddingGemma 2, whose QK-norm
// takes its place.
//
// Not the decoder's kernel (kernels/attention/attention.h) with a flag. That
// kernel's mask is causal, its keys come from the cache by slot, its chunks
// and splits are fixed by position for decode, and its tile holds a whole
// group of query heads in 1,024 / d query vectors — at d = 512 two, which a
// global layer's group of 4 cannot divide. What differs here is the
// operation's kind — both directions, one step's own keys, no cache — so it
// is its own kernel (logical-overview.md, principle 6), and tuning one never
// touches the other (change-axes.md, the E test).
//
// Two forms, chosen by the head dimension, each its own entry point: the
// written form below, for heads of 256 and 512 — the text encoder's — and
// the streaming form after it, for heads of 64 — a vision encoder's
// (graph/vision.h), whose 2,520 patches' scores no binding holds.
//
// The written form's shape: two launches, the scores and then their
// weighted sum, with the scores written out between them.
//
//   1. `scores_of`: S = Q Kᵀ × scale × log2(e), a tiled product of two
//      activation matrices. A workgroup takes one key-value head, a tile of
//      32 query vectors — 32 / G rows × the G query heads sharing that
//      key-value head, so a key is read once for its whole group — and one
//      chunk of 64 keys, positions 64c .. 64c + 63. It steps along d 32 at a
//      time, staging in workgroup memory the tile's 32 query slabs and the
//      chunk's 64 key slabs, each 32 floats, the key slabs padded to 9 vec4s
//      a key, as matmul.h's weight slots are, so a read's 16 lanes spread
//      over the banks: 4 KiB and 9 KiB. Its 128 invocations each own 4 query
//      vectors × 4 keys — invocation t vectors 4(t / 16) .. + 3 and keys l,
//      l + 16, l + 32, l + 48 of the chunk, l = t % 16 — 16 accumulators in
//      named vec4s, each a chain of fmas along d in order from 0 (GPU.3,
//      matmul.h's micro-tile). At the end each live score is scaled and
//      written to `scores`, S[h][i][j] for query head h, row i and key j, N
//      floats a row, adjacent lanes adjacent keys; a masked one is written
//      0.0 and never read as a score. Then each vector's maximum over the
//      chunk's live keys: each invocation's over its 4, with whether any was
//      live, as two whole vec4s in the query slabs' workgroup memory, free
//      once the products are done, and one invocation a vector folding the
//      16 — written to `score_maxima`, one float a (head, row, chunk); 0.0
//      for a chunk with no live key, which liveness, below, skips.
//   2. `weigh`: out = Σ_j 2^(s_j − m) v_j / Σ_j 2^(s_j − m), m the row's
//      maximum. A workgroup takes one key-value head, a tile of 32 query
//      vectors, and 64 of the d output dimensions. First, one invocation a
//      vector folds its row's chunk maxima over the live chunks — max,
//      exactly associative, so any order gives the same m (GDSA.2) — into
//      workgroup memory, and a barrier. Then it steps along the keys 32 at
//      a time, staging the weights p = 2^(s − m), 0 for a key not live, 32
//      vectors × 32 keys, 4 KiB, computed as they are staged, 8 exp2 an
//      invocation; and the values' 64 dimensions of those 32 keys, 8 KiB,
//      16 vec4s a key, which a read's 16 lanes take whole and its 8 groups
//      of lanes share. Its 128 invocations each own 4 vectors × 4 dimensions — a
//      vec4 of the output a vector — and add p × v for each key, in key
//      order, into 4 vec4 accumulators, and each vector's p into its sum l,
//      in the same order: the 16 invocations sharing a vector each keep the
//      same l, the same bits, so no lane waits on another. At the end, O /
//      l, written to `attention` at query head h's columns of row i. Every
//      row below N has a live key — itself — so l is never 0.
//
//   - A step is at position 0, as an encoder's one step is, so a row is its
//     position: the scores launch's chunks are the step's own, ceil(N /
//     64), which the program counts as a split's — key_chunks at position 0,
//     with no window and no limit on the partial rows (kernels/interface.h).
//   - Liveness — j < N and |i − j| <= h — is recomputed from positions
//     wherever it is needed, never carried in a value: WGSL lets an
//     implementation assume no infinities, and no finite sentinel lies
//     below every score (attention.h). A chunk is live for a row when it
//     holds a live key, which positions alone decide.
//   - Exponentials are base 2, the scale folded with log2(e) when the
//     scores are written, as attention.h's are. exp2 is evaluated in
//     `weigh` alone, so a weight and the sum it enters are the same bits.
//   - Variants are override constants — `head_dimension`, `query_heads`,
//     `key_value_heads` and `group` — each a plain override the launcher
//     sets on every launch, as WebKit requires
//     (docs/research/2026-10-09-webkit-shader-compiler.md); the tiles are
//     the same shape at every d, so no workgroup array is sized by one.
//   - Constants (binding 1): struct BlockAttention { half_window: u32,
//     scale_log2e: f32 }, the layer's.
//   - Bindings: `scores_of` binds 2 the queries and 3 the keys, read-only, and
//     4 the scores and 5 the maxima, written; `weigh` binds 2 the scores and
//     3 the maxima and 4 the values, read-only, and 5 the attention buffer,
//     written — two entry points of one module, their own declarations
//     sharing binding numbers, as attention.h's combine does. Four storage
//     buffers each.
//
// Why the scores are written out, against FlashAttention's online softmax,
// which never writes them (GDSA.16) and which the decoder's kernel follows:
// at d = 512 the browser's 16 KiB of workgroup memory leaves an online tile
// two bad shapes. Holding a tile's outputs across the keys — 32 query
// vectors × 512 f32 — is 128 accumulators an invocation of 128, four times
// the 32 that matmul.h measured as the micro-tile's limit (GPU.3). Slicing
// the outputs by d instead, as `weigh` does, makes every slice recompute its
// rows' scores: 8 slices at d = 512, 8 times a global layer's 537 million
// score multiply-adds. Writing the scores once costs, at N = 512, 4 MiB
// written and read a layer — 192 MiB an encode, about 0.5 ms at 400 GB/s if
// none stays in the GPU's caches, which a layer's 4 MiB can, though WebGPU
// promises nothing of what they keep. GDSA.16's own caveat: an intermediate
// already the size the caches hold gains little from tiling, and its
// consumer here needs it whole, row by row.
//
// The streaming form, for a head of 64: one launch, `stream`, the scores
// never written — FlashAttention's online softmax (GDSA.16, CDSA.23). At N
// = 2,520 patches and 12 heads the written form's scores would be 305 MB a
// layer, over the default 128 MiB binding and 256 MiB buffer; and at d = 64
// the reason above against streaming is gone: a tile's outputs, 32 query
// vectors × 64, are 16 floats an invocation of 128, matmul.h's micro-tile.
//   - A workgroup takes one key-value head and a tile of 32 query vectors —
//     32 / G rows × the G query heads sharing that key-value head, as the
//     written form's tiles are — and walks every key of the step in order,
//     16 at a time. Its 128 invocations are 4 a query vector: invocation t
//     serves vector q = t / 4, lane c = t mod 4.
//   - Workgroup memory: the tile's queries, staged once, 32 vectors × 64
//     floats padded to 17 vec4s a vector, 8.5 KiB, so the 8 vectors a read's
//     lanes take spread over the banks; one slot for 16 keys, padded the same,
//     4.25 KiB, which holds the step's keys and then its values; and the
//     weights, 32 vectors × 16 keys, 2 KiB. 14.75 KiB, under the 16 KiB the
//     harness keeps.
//   - Each step of 16 keys, j0 .. j0 + 15:
//       1. Stage the 16 keys' 64 dimensions, 2 vec4s an invocation; barrier.
//       2. Scores: invocation (q, c) forms s for its vector and keys j0 +
//          4c .. j0 + 4c + 3, four chains of fmas along d in order from 0,
//          × scale × log2(e), and writes them to its four weight slots;
//          barrier.
//       3. Its vector's tile maximum over the tile's live keys, read from
//          the 16 slots in order — the same bits in each of the 4 lanes —
//          and the running maximum m' = max(m, that). Its own four weights
//          p = 2^(s − m'), 0 for a key not live, written over its scores.
//          Meanwhile the slot is free — every score read the keys before
//          the barrier — so each invocation stages 2 vec4s of the 16
//          values; barrier.
//       4. Rescale: o and l times 2^(m − m'), where m' rose. Then for each
//          key in order, l += p_j and o += p_j × v_j, o its vector's 16
//          dimensions 16c .. 16c + 15, 4 vec4 accumulators: the 4 lanes of a
//          vector keep the same l, the same bits; barrier, before the next
//          step stages over the slot and the weights.
//     At the end, o / l to `attention`, at query head h's columns of row i.
//   - Liveness as the written form's: from positions, j < N and |i − j| <=
//     h, never a value. A vector whose tile has no live key changes nothing
//     in that step: its maximum is carried with a flag that it has seen a
//     live key, so no infinity and no sentinel is ever compared (GDSA.16's
//     example). Every row has a live key, itself, so l > 0 at the end.
//   - Override constants `head_dimension`, `query_heads`, `key_value_heads`
//     and `group`, as the written form's; the tile is the same shape at
//     every group, so no workgroup array is sized by one.
//   - Constants (binding 1): the written form's struct. Bindings: 2 the
//     queries, 3 the keys, 4 the values, read-only; 5 the attention
//     buffer, written. Four storage buffers.
//   - The tile sizes are named, kStreamVectors 32 and kStreamKeys 16: a
//     measured parameter of the target, not of the operation (CDSA.21).
//
// What the streaming form costs, at N rows, for a vision layer of 12 heads,
// d = 64, G = 1, at N = 2,520:
//   - Multiply-adds: H_q × N² × d for the scores and as many for the sum,
//     1.56 × 10¹¹ over 16 layers (graph/vision.h).
//   - Workgroups: ceil(N G / 32) × H_kv, 79 × 12 = 948; steps a workgroup
//     ceil(N / 16), 158; barriers 4 a step, 632.
//   - exp2: 4 a step an invocation for the weights, and one rescale, 790 an
//     invocation at N = 2,520.
//   - Workgroup-memory loads a multiply-add: the scores read a query vec4
//     and 4 key vec4s for 16 fmas; the sum reads 4 weight vec4s and 64
//     value vec4s for 256: about 3.5 fmas a load.
//   - Bytes: each workgroup reads its head's keys and values, N × d × 8,
//     1.29 MB, from the GPU's caches — 1.22 GB a layer, over keys and values
//     of 7.7 MB each in device memory, read from it about once — and writes
//     its outputs once, 7.7 MB a layer. Against the written form's 305 MB of
//     scores written and read a layer, which no binding could hold.
//   - Workgroup memory zero-filled at launch, 14.75 KiB, about 29 stores an
//     invocation.
// Optimization (practice): FlashAttention's streaming softmax, the score
// matrix never leaving the workgroup (GDSA.16, CDSA.23).
// Optimization not taken, a lever to measure: the queries in registers, 16
// vec4s an invocation, which would free 8.5 KiB for a step of 32 keys and
// halve the barriers, at a register budget that could cost occupancy
// (GPU.3); the tile sizes are named so it can be tried by measurement.
//
// Accuracy, against f64 from the same queries, keys and values: the written
// form's margin, and the rescales': each step can multiply o and l by one
// rounded exp2, at most 158 at N = 2,520, each within 3 + 2|x| units in the
// last place, both by the same factor, so the ratio carries at most 2 units
// a rescale. A margin the GPU test checks on the target, not a proof.
//
// What it costs, counted, at N tokens; at N = 512 for EmbeddingGemma 2:
//   - Multiply-adds: H_q × N² × d for the scores and as many for the sum —
//     2,048 N² a sliding layer, 4,096 N² a global one; 57,344 N² a step,
//     1.5 × 10¹⁰ at N = 512, beside the products' 7.0 × 10¹⁰
//     (graph/encoder.h).
//   - Workgroups: `scores` ceil(N G / 32) × H_kv × ceil(N / 64), 512 a
//     layer of either kind; `weigh` ceil(N G / 32) × H_kv × d / 64, 256 a
//     sliding layer and 512 a global one.
//   - Bytes. `scores` writes the scores, H_q × N² × 4, 4 MiB, and the
//     maxima, H_q × N × ceil(N / 64) × 4, 64 KiB; each workgroup reads its
//     query tile, 32 × d floats, and its chunk's keys, 64 × d — 48 MiB a
//     sliding layer, 96 MiB a global one, from the GPU's caches, over
//     queries and keys of 2 MiB and 1 MiB, or 4 and 1, in device memory.
//     `weigh` reads each vector's row of scores, N floats, and its slice of
//     every value, N × 64 — 48 MiB a sliding layer and 96 MiB a global one,
//     from the GPU's caches — and writes the output once, H_q × N × d × 4,
//     2 or 4 MiB.
//   - Barriers: `scores` two a step along d and one for the maxima, 17 at d
//     = 256 and 33 at 512; `weigh` one after the maxima and two a step of
//     32 keys, 33 at N = 512.
//   - Workgroup memory: `scores_of` 13 KiB, `weigh` 12.1 KiB, under
//     WebGPU's default 16 KiB, which the harness keeps for cross-browser
//     compatibility; WGSL zero-fills it, a store an invocation of each word,
//     about 26 a launch's invocation.
//   - Launches: 2 a layer, 48 an encode, about 72 µs at 1.5 µs.
// Optimization (practice): a key-value head's keys and values read once for
// its whole group of query heads, its G heads' vectors tiled together — as
// attention.h's tile does, and grouped-query attention exists to allow
// (GPU.5).
// Optimization (practice): the products' 4 × 4 micro-tile, named vec4
// accumulators and padded slabs, the same as matmul.h's prefill tile and
// for its measured reasons (GPU.3, GPU.5, SIMD.2).
// Optimization (practice): the row maximum from chunk maxima the scores
// launch already holds in registers, 8 floats a row at N = 512, where
// rescanning the row in `weigh` would read N floats a vector for each of its
// d / 64 slices (GDSA.6).
//
// Where WebGPU limits it, and what each limit costs here:
//   - Workgroup memory, 16 KiB: the score matrix written between two
//     launches, counted above.
//   - Subgroup operations are optional, for the same reason: the chunk
//     maximum crosses invocations through workgroup memory and a barrier,
//     where a subgroup max would need neither.
//   - No matrix-multiply units for f32: both products run on the shader
//     cores, as matmul.h's do.
//
// Determinism (GDSA.2): run to run on the target. Every score is one chain
// of fmas along d in order; the maxima are exact; every output and its sum
// one chain over the keys in order. Nothing depends on which workgroup ran
// which tile. WGSL lets each pipeline contract or reassociate its
// multiply-adds as it chooses, so equal bits run to run are what the
// target's compiler gives, which the GPU test checks.
//
// Accuracy, against the same computation in f64 from the same queries, keys
// and values: each score within γ(d) × Σ_k |q_k k_k| × scale × log2(e), γ(n) =
// n 2⁻²⁴ / (1 − n 2⁻²⁴), and so each weight within that, relatively, through
// exp2, which WGSL bounds to 3 + 2|x| units in the last place; the sums of at
// most 512 weights and products add γ(512). So each output is within max|v|
// × (γ(d) × max_j Σ_k |q_k k_jk| × scale × log2(e) + γ(512) + 2⁻¹⁶), max|v|
// the largest value in the row's window: a margin the GPU test checks on
// the target, as attention.h's is, not a proof.
//
// Verification the implementation is held to, on the GPU against f64 from
// the same buffers: EmbeddingGemma 2's two layer shapes, d = 256 with G = 2
// and d = 512 with G = 4; N of 2, 31, 33, 64, 65 and 130, and 512 for the
// first; windows of 32 and 1,024 beside a full layer's, so keys either side
// are masked; every output within the margin; the same step run twice
// giving the same bits. Each launch's entry point, split and workgroups at
// 65 and 512 tokens. The streaming form the same way at the vision
// encoder's shape, d = 64 with G = 1 and 12 heads: N of 2, 15, 16, 17, 33,
// 630 and 2,520, a step of 16 keys partly live in each; a window of 32
// beside the whole, so a tile with no live key for a row is crossed; and
// against the written form at d = 64 and N = 130, each within its margin of
// f64. Its workgroups at 2,520 rows, 948.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.12   Don't make data members const or references in a copyable or
//            movable type — BlockAttentionLaunch holds values.
//     F.20   Prefer return values to out parameters — the launches are
//            returned.
//     I.4    Make interfaces precisely and strongly typed.
//   C++ performance guidelines
//     CDSA.23 Keep streaming state small and mergeable, with a fixed combine
//            order — the streaming form's maximum, sum and outputs, keys in
//            order.
//     CDSA.21 Measured, named thresholds — the streaming tile's sizes.
//     GDSA.16 Stream through on-chip tiles; never materialize the large
//            intermediate — weighed, and its caveat taken, above.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            run to run; the maxima exact.
//     GDSA.6 Count passes over global memory — the bytes above.
//     GPU.2  Shape data for coalesced lane access — adjacent lanes,
//            adjacent keys and dimensions.
//     GPU.3  Treat occupancy as latency-hiding budget — 16 accumulators an
//            invocation, named.
//     GPU.5  Use workgroup memory where reuse pays — every staged slab is
//            read by 16 or 32 invocations.
//     SIMD.2 Lay out data so vector loads are linear — whole vec4s along d.

// One layer's block attention.
struct BlockAttentionLaunch {
    model::LayerDescription layer;            // heads, head dimension, window
    float scale;                              // the model's attention scale
    residency::BufferRange query;             // normalized and rotated, rope's
    residency::BufferRange key;               // normalized and rotated, rope's
    residency::BufferRange value;             // normalized where the model says, rope's
    residency::BufferRange scores;            // H_q × N × N floats, written
    residency::BufferRange score_maxima;      // H_q × N × ceil(N / 64) floats, written
    residency::BufferRange attention;         // the output
};

// The layer's scores launch and its weigh launch, in that order.
// Preconditions: the head dimension is a multiple of 64, at most 512; the
// query heads a key-value head serves divide 32; the buffers are the plan's,
// sized for kPrefillBlock rows; every step is at position 0.
[[nodiscard]] std::array<Launch, 2> block_attention_launches(const BlockAttentionLaunch& attention);

// The streaming form's tile: query vectors a workgroup, keys a step.
inline constexpr std::uint32_t kStreamVectors = 32;
inline constexpr std::uint32_t kStreamKeys = 16;

// One layer's block attention in the streaming form: no scores bound.
struct StreamedAttentionLaunch {
    model::LayerDescription layer;            // heads, head dimension, window
    float scale;                              // the model's attention scale
    residency::BufferRange query;             // normalized and rotated, rope's
    residency::BufferRange key;
    residency::BufferRange value;
    residency::BufferRange attention;         // the output
};

// The layer's one streaming launch. Preconditions: the head dimension is
// 64; the query heads a key-value head serves divide 32; the buffers hold
// the step's rows, which may exceed kPrefillBlock; every step is at
// position 0.
[[nodiscard]] Launch streamed_attention_launch(const StreamedAttentionLaunch& attention);

}  // namespace bllm::kernels
