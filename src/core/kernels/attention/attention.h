#pragma once

#include <array>
#include <cstdint>

#include "core/formats/format.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// Attention: scores, the causal and window mask, softmax and the weighted
// sum of values, never writing the scores out (docs/architecture/
// kernel-fusions.md). Between rope, which wrote this step's keys and values
// into the KV cache and its queries in place, and the output projection.
//
// "KV cache" below is always the stored keys and values (cache/kv.h); the
// GPU's own caches, the hardware between its cores and device memory, are
// always "the GPU's caches".
//
// For a query q at position p, of a head whose key-value head holds keys
// k_j and values v_j at positions j:
//
//   s_j  = (q · k_j) × scale             for j in p − W + 1 .. p, W the window
//   out  = Σ_j softmax(s)_j × v_j
//
// scale is the model's (model/model_description.h): 1 / sqrt(d) for every
// listed model. Every key is read from the KV cache, the step's own included,
// rounded as stored (kernels/rope/rope.h).
//
// The shape of the computation: FlashAttention-2's online softmax. Keys and
// values stream through workgroup memory a tile at a time; each query keeps
// a running maximum m, sum l and weighted sum O, rescaled when its maximum
// rises; nothing larger than a tile of scores exists (GDSA.16).
//
//   - Chunks. Keys are taken in chunks of 64 consecutive positions, chunk c
//     holding positions 64c .. 64c + 63, fixed by position alone — ONNX
//     Runtime's fixed decode tile, so a decode step at 1,024 keys runs 128
//     workgroups where 256-key chunks ran 32 (costs, below). Each
//     chunk's (m, l, O) is computed from an empty state, then folded into the
//     query's result in chunk order by one merge function:
//       merge((m, l, O), (m_c, l_c, O_c)), m' = max(m, m_c):
//         l' = l × 2^(m − m') + l_c × 2^(m_c − m'),  O' likewise
//     where a factor whose exponent is 0 is exactly 1, and an empty side —
//     no live key — leaves the other as it was. The result is O / l. So a
//     query's output depends on its position, its query and the cached keys
//     and values in its window, and on nothing about the step: not its token
//     count, not which workgroup took which chunk (CDSA.23).
//   - Active chunks. A layer's step needs the chunks from the one holding
//     its first row's earliest key to the one holding its last row:
//       first = floor(max(0, position − W + 1) / 64)
//       last  = floor((position + tokens − 1) / 64)
//     W the layer's window — the trained context for a full-attention layer,
//     so first is 0 — giving count = last − first + 1: every chunk for a
//     full-attention layer, at most nine for one of Gemma 3's 512-key
//     window layers however long the context. Chunks outside are never
//     scheduled; leaving out a chunk with no live key changes no result.
//   - Splits. When a step has few rows, one workgroup a chunk: split s takes
//     chunk first + s, writes its unnormalized (m, l, O) for each query to
//     the partial buffers, and a combine launch folds the count of them in
//     chunk order and divides — Flash-Decoding's split, and vLLM's
//     PagedAttention V2 over 512-token partitions. When it has many, one
//     workgroup folds every active chunk itself, with the same merge. A
//     layer's step splits when tokens × count <= P, the partial buffers'
//     rows: 512, the attention buffer's own size, or the offered context's
//     chunks, ceil(C / 64), where that is more — 640 for Qwen3's 40,960 —
//     so a decode step splits at every position the context offers
//     (kernels/interface.h, key_chunks). Folding several chunks a split
//     instead would associate the merges differently from an unsplit step,
//     and change its bits. A chunk holding no live key for some of a
//     workgroup's rows writes empty partials for them, which the combine
//     skips.
//   - Tiles. A workgroup is 64 invocations and holds, in workgroup memory, a
//     tile of M = 1,024 / d query vectors — R = M / G consecutive rows of
//     the step, for the G query heads sharing one key-value head — at f32,
//     4 KiB; and a tile of B = 2,048 / d keys and their values as the KV cache
//     stores them, 4 KiB each:
//
//                      M query vectors   R rows × G heads   B keys a tile
//       Qwen3, d 128          8               4 × 2              16
//       Llama 3.2, d 64      16               4 × 4              32
//       Gemma 3, d 256        4               1 × 4               8
//
//     Each key and value is read from the KV cache once for the whole group:
//     grouped-query attention reads its shared keys once, not once a head.
//     A step's workgroups: ceil(tokens / R) × H_kv × splits. A decode
//     step's tile holds one row, its other query slots idle; its keys and
//     values are read once all the same, which is what decode is bound by.
//   - Per tile, four phases, a barrier before each: load the tile's keys and
//     values, slot j mod slots (cache/kv.h), coalesced; scores, invocation
//     (m, b) taking the dot product of query m and key b, d multiply-adds,
//     key rows padded by a word so the invocations reading different keys
//     at one column hit different banks (GPU.5); the online softmax, by
//     each query's owner, the first of its invocations; and O = O × α + P ·
//     V, each invocation 16 of the M × d outputs, adjacent invocations
//     adjacent words of each value row. At a chunk's end every invocation
//     folds its own outputs with the chunk's statistics, written before the
//     last tile's third barrier, and keeps the query's running maximum and
//     sum itself, so the fold needs no barrier.
//   - Exponentials are base 2: queries are multiplied by scale × log2(e)
//     as they are staged, so a score is already in log2 units and exp2
//     replaces exp, FlashAttention-2's form.
//   - Masks: a key is live for query p when j <= p and j >= p − W + 1. A
//     masked score is never formed: its weight is set to 0 and it is left
//     out of the maximum. Liveness is recomputed from positions wherever it
//     is needed, never carried in a score: WGSL lets an implementation
//     assume no infinities, and no finite sentinel lies below every finite
//     score. A tile with
//     no live key for a query leaves its state exactly as it was. Tiles
//     past a workgroup's last row are not loaded: causal tiles above the
//     diagonal are skipped.
//   - One source serves both roles, as two entry points: `main`, and
//     `combine`, which folds a query's splits and divides, d / 4
//     invocations a query, each a vec4 of the output. The merge is one WGSL
//     function both call, and the combine allocates none of main's
//     workgroup memory, which WGSL gives only the entry points that use it.
//     The other shape — the last workgroup of a split folding the rest —
//     would wait on other workgroups, which WebGPU gives no forward-progress
//     guarantee for (GDSA.4); the combine is a launch of its own.
//   - Variants are override constants — `head_dimension`, `query_heads`,
//     `key_value_heads` — so tile shapes and indices fold when the pipeline
//     is built; one pipeline per role for a model.
//   - Constants (binding 1): struct Attention { slots: u32, window: u32,
//     scale_log2e: f32 }, the layer's.
//   - Bindings: 2 the query buffer, 3 the layer's keys and 4 its values in
//     the KV cache, read-only; 5 the attention buffer, 6 the partial values and
//     7 the partial statistics, written. The combine binds 2 the partial
//     values and 3 the partial statistics, read-only, and 4 the attention
//     buffer, written: its own declarations, which share binding numbers
//     with main's as two entry points may. Six storage buffers at most,
//     under WebGPU's default 8.
//
// What it asks of the other contracts:
//   - kernels/interface.h and program.h: a launch may cover whole tiles of
//     rows, ceil(rows / rows_per_tile) of them; it may be split by key
//     chunks, naming its layer's window, so the program multiplies its
//     workgroups by key_chunks' split count for the step; and it may run
//     only when the step splits, which the combine does, so an unsplit step
//     pays no launch for it; and it names its entry point. The kernel computes the same first chunk and
//     split from the step's position and token count and its window, in
//     WGSL; the tests hold the two to agreement at every boundary.
//   - residency/plan.h: two working buffers — partial values, P rows of
//     H_q × d floats, and partial statistics, P rows of H_q × 2, P =
//     max(512, ceil(C / 64)) for the context C it offers — each a buffer of
//     its own: 5 MiB of values for Qwen3 at 40,960, 4 MiB at 512 rows.
//   - kernels/interface.h: kChunkKeys is 64, and key_chunks takes the
//     partial rows P, which the kernel receives as an override constant,
//     `partial_rows`; tokens × count stays below 2^32, at most 512 ×
//     262,145 (kMaxPositions / 64 + 1).
//   - formats/format.h: the KV cache format's WGSL gains pack's inverse, fn
//     unpack4(words: vec2<u32>) -> vec4<f32>, reading no binding; F16's is
//     two unpack2x16float.
//   - model/model_description.h: the model's attention scale.
//
// Determinism (GDSA.2): run to run, and batch-invariant on the target — a
// query's output is the same bits whatever the step's token count or
// splits, since chunks are fixed by position, each is computed by the same
// tile loop from an empty state, and they are folded in chunk order by one
// function, whether within a workgroup or by the combine. main and the
// combine are separate pipelines, and WGSL lets each fuse or reassociate
// that function's multiply-adds as it chooses, so equal bits are what the
// target's compiler gives, which the GPU test checks, not what WGSL
// promises. Not bit for bit with llama.cpp, whose tiles and order differ.
//
// Accuracy, against the same computation in f64 from the same queries and
// the KV cache's stored keys and values: the scores' rounding dominates. A
// d-term f32 dot product errs by up to d × 2⁻²⁴ × Σ_j |q_j k_j| × scale, in
// log2 units once scaled, and moves the key's weight by that much,
// relatively, through exp2; exp2's own 3 + 2|x| units and the f32 sums of at
// most 64 weights a chunk add far less. So each output is within max|v| ×
// (d × 2⁻²⁴ × max_j Σ_i |q_i k_ji| × scale × log2(e) + 2⁻¹⁶), max|v| the
// largest value in the query's window: a margin the GPU test checks on the
// target, as the norm's is, not a proof.
//
// What it costs, counted, for Qwen3; L the context after the step.
//   - Decode, one token. Each key and value is read once: L × H_kv × d × 2
//     bytes × 2, 4 KiB a token of context a layer, 112 KiB across 28
//     layers — 448 MiB at L = 4,096, about 1.2 ms at 400 GB/s, more than the
//     weights' 0.95 ms; attention is decode's second floor, and the larger
//     past about 3,300 tokens. The split's partials add H_q × (d + 2) × 4
//     bytes a chunk: 16 KiB of values written and read, and 128 bytes of
//     statistics written and read by each of the combine's d / 4
//     invocations a query, 4 KiB — about 20.6 KB against the chunk's 256 KiB
//     of keys and values, 7.9%. Workgroups: 8 × ceil(L / 64), 128
//     at 1,024 and 512 at 4,096. The combine's d / 4 invocations a query
//     each fold its count of partials in order, a chain of that many merges:
//     64 at 4,096, 128 at 8,192, each one exp2 — the side holding the
//     maximum keeps factor 1 — and a vec4 multiply and multiply-add.
//     Launches: one a layer, and the combine's a layer once L passes 64 —
//     56 a step, 84 µs at 1.5 µs (interface.h).
//   - Prefill, 512 tokens from position 0: query-key pairs under the causal
//     mask, T(T + 1) / 2 = 131,328 a head; 2 × d multiply-adds a pair, the
//     score's and the weighted sum's, 4 × d operations: 1.08 × 10⁹ a layer,
//     30 × 10⁹ a step, about 2 ms at the M3 Max's roughly 14 f32 TFLOPS
//     (third-party figure) if the arithmetic ran at peak. A row tile streams
//     the keys up to its last row, half a layer's 2 MiB of keys and values
//     on average: about 128 MiB of KV cache reads a layer. From device
//     memory that is between the layer's 2 MiB, if the GPU's caches keep it
//     while its row tiles run, as they can hold it, and the whole 128 MiB,
//     about 0.3 ms a layer at 400 GB/s, if they do not; WebGPU promises
//     neither the order workgroups run in nor what the caches keep, so where
//     in that range a step falls is the target's, to be measured.
//   - Barriers: four a tile, 16 a chunk for Qwen3, 8 for Llama 3.2, 32
//     for Gemma 3 — a decode workgroup's whole chain — and the merge at each
//     chunk's end needs none.
// Optimization (practice): the softmax is online and tiled, so the
// T × L scores are never written — FlashAttention-2 (GDSA.16).
// Optimization (practice): a decode step is split across the context in
// fixed chunks and combined, so 128 workgroups at 1,024 tokens and 512 at
// 4,096 read the KV cache where one a key-value head would be 8 —
// Flash-Decoding, vLLM's PagedAttention V2 (GDSA.8). The chunk is 64 keys,
// ONNX Runtime's: llama.cpp's and ONNX Runtime's WebGPU decode reach 256
// workgroups or more at 1,024 keys, and 256-key chunks ran 40 at the
// profile's position 1,024 — 1,025 keys, 5 chunks — which read the KV
// cache at about 32 GB/s (docs/research/2026-10-07-forward-pass-profile.md);
// 64-key chunks run 136 there (GPU.3).
// Optimization (practice): chunks fixed by position and folded in order,
// so a split does not change a result — the fixed split size Thinking
// Machines' batch-invariant attention uses (CDSA.23).
// Optimization (practice): a key-value head's keys are read once for its
// whole group of query heads, and the scale folded into the staged queries.
//
// Where WebGPU limits it, and what each limit costs here:
//   - Workgroup memory is 16 KiB at WebGPU's defaults, which the harness
//     keeps for cross-browser compatibility (gpu/device_requirements.h). A
//     tile is therefore M = 1,024 / d query vectors — 4 rows for Qwen3 —
//     where FlashAttention-2 takes 64 to 128 rows; prefill reads each
//     layer's KV cache 128 times rather than 4 to 8: 16 to 32 times the
//     reads, which the GPU's caches serve as far as they keep the layer's
//     keys and values (costs, above).
//   - Subgroup operations are optional, for the same reason: each score is
//     one invocation's whole dot product, and the softmax's per-tile maxima
//     and sums are one invocation's loop over B, where llama.cpp's Metal
//     kernels reduce across a SIMD-group.
//   - WebGPU has no matrix-multiply units: Chrome's subgroup matrices are
//     an experiment, not cross-browser. The arithmetic is f32 on the
//     shader cores.
//   - WGSL may assume no infinities: masks are flags, not −∞.
//   - WGSL zero-fills workgroup memory: 12.7 KiB for Qwen3, stored once a
//     workgroup — about 40% of the 32 KiB of keys and values a decode
//     workgroup reads from the KV cache for its chunk, though into on-chip
//     memory: about 51 stores an invocation, against about 2,000
//     multiply-adds an invocation a chunk.
//
// Levers not taken:
//   - Splits sized to fill the GPU rather than fixed by position: more
//     parallelism for a short context, and a result that would change with
//     the step's shape, which the batch-invariant fold rules out.
//   - The combine folded into the output projection's load: 28 launches a
//     decode step saved, 42 µs, for a matrix product that reads partials
//     and knows attention's merge.
//   - The combine's statistics folded once a head, one lane a head, its
//     factors shared through workgroup memory: 32 times fewer exp2s for
//     Qwen3, where each of a query's d / 4 invocations folds them itself.
//     The combine's 512 invocations a layer leave most of the GPU's lanes
//     idle, so the repeated exp2s run beside one another at no cost in time,
//     and each invocation's chain is the same count of merges either way;
//     a head's lane would make every output wait on its serial pass and a
//     barrier.
//   - Larger tiles from a raised workgroup-memory limit: Apple adapters
//     offer 32 KiB; the harness asks for none of the defaults' limits
//     raised.
//
// Verification the implementation is held to, on the GPU against f64 from
// the same KV cache contents: each listed shape, prefill steps of 1, 3, 4 and
// 512 rows and decode steps, at positions 0, near 64's boundaries and deep
// into the context; a decode step past 512 chunks, split; Gemma 3's window over a ring that has wrapped; a step
// split and unsplit, and the same query prefilled and decoded, giving the
// same bits; and a chunk wholly masked for some rows. On the CPU: key_chunks
// at its boundaries, and each launch's geometry, bindings and variant.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.12   Don't make data members const or references in a copyable
//            type — AttentionLaunch holds values and a pointer.
//     F.20   Prefer return values to out parameters — the launches are
//            returned.
//     I.4    Make interfaces precisely and strongly typed.
//   C++ performance guidelines
//     GDSA.16 Stream through on-chip tiles; never materialize the large
//            intermediate — the scores.
//     CDSA.23 Keep streaming state small and mergeable, with a fixed
//            combine order — (m, l, O), chunks fixed by position.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            batch-invariant, as above.
//     GDSA.6 Count passes over global memory — the bytes above.
//     GDSA.8 Partition irregular work by output position — splits.
//     GPU.2  Shape data for coalesced lane access — tile loads, writes.
//     GPU.5  Use workgroup memory where reuse pays — every tile is read by
//            every query of the workgroup; padded rows.
//     GPU.6  Batch tiny GPU work — the combine runs only when split.
//     GDSA.4 Never spin-wait on another workgroup without a
//            forward-progress guarantee — the combine is a launch.

// One layer's attention.
struct AttentionLaunch {
    model::LayerDescription layer;             // heads, head dimension, window
    float scale;                               // the model's attention scale
    residency::BufferRange query;              // rotated queries, rope's
    residency::PlannedCacheLayer cache;        // the layer's
    const formats::Format* cache_format;       // never null; its unpack4 reads the cache
    residency::BufferRange attention;          // the output
    residency::BufferRange partials;           // a split's unnormalized O
    residency::BufferRange partial_stats;      // and its m and l
    std::uint32_t partial_rows;                // the query rows they hold
};

// The layer's attention launch and its combine, in that order.
// Preconditions: the head dimension is 64, 128 or 256; the query heads a
// key-value head serves divide 1,024 / d; `cache_format` has a pack and is
// the format the plan sized `cache` in.
[[nodiscard]] std::array<Launch, 2> attention_launches(const AttentionLaunch& attention);

}  // namespace bllm::kernels
