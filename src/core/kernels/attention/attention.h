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
// into the cache and its queries in place, and the output projection.
//
// For a query q at position p, of a head whose key-value head holds keys
// k_j and values v_j at positions j:
//
//   s_j  = (q · k_j) × scale             for j in p − W + 1 .. p, W the window
//   out  = Σ_j softmax(s)_j × v_j
//
// scale is the model's (model/model_description.h): 1 / sqrt(d) for every
// listed model. Every key is read from the cache, the step's own included,
// rounded as stored (kernels/rope/rope.h).
//
// The shape of the computation: FlashAttention-2's online softmax. Keys and
// values stream through workgroup memory a tile at a time; each query keeps
// a running maximum m, sum l and weighted sum O, rescaled when its maximum
// rises; nothing larger than a tile of scores exists (GDSA.16).
//
//   - Chunks. Keys are taken in chunks of 256 consecutive positions, chunk c
//     holding positions 256c .. 256c + 255, fixed by position alone. Each
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
//       first = floor(max(0, position − W + 1) / 256)
//       last  = floor((position + tokens − 1) / 256)
//     W the layer's window — the trained context for a full-attention layer,
//     so first is 0 — giving count = last − first + 1: every chunk for a
//     full-attention layer, at most three for one of Gemma 3's 512-key
//     window layers however long the context. Chunks outside are never
//     scheduled; leaving out a chunk with no live key changes no result.
//   - Splits. When a step has few rows, one workgroup a chunk: split s takes
//     chunk first + s, writes its unnormalized (m, l, O) for each query to
//     the partial buffers, and a combine launch folds the count of them in
//     chunk order and divides — Flash-Decoding's split, and vLLM's
//     PagedAttention V2 over 512-token partitions. When it has many, one
//     workgroup folds every active chunk itself, with the same merge. A
//     layer's step splits when tokens × count <= 512, so the partial
//     buffers hold 512 query rows, the attention buffer's own size
//     (key_chunks below). A chunk holding no live key for some of a
//     workgroup's rows writes empty partials for them, which the combine
//     skips.
//   - Tiles. A workgroup is 64 invocations and holds, in workgroup memory, a
//     tile of M = 1,024 / d query vectors — R = M / G consecutive rows of
//     the step, for the G query heads sharing one key-value head — at f32,
//     4 KiB; and a tile of B = 2,048 / d keys and their values as the cache
//     stores them, 4 KiB each:
//
//                      M query vectors   R rows × G heads   B keys a tile
//       Qwen3, d 128          8               4 × 2              16
//       Llama 3.2, d 64      16               4 × 4              32
//       Gemma 3, d 256        4               1 × 4               8
//
//     Each key and value is read from the cache once for the whole group:
//     grouped-query attention reads its shared keys once, not once a head.
//     A step's workgroups: ceil(tokens / R) × H_kv × splits. A decode
//     step's tile holds one row, its other query slots idle; its keys and
//     values are read once all the same, which is what decode is bound by.
//   - Per tile, four phases, a barrier after each: load the tile's keys and
//     values, slot j mod slots (cache/kv.h), coalesced; scores, invocation
//     (m, b) taking the dot product of query m and key b, d multiply-adds,
//     key rows padded by a word so the invocations reading different keys
//     at one column hit different banks (GPU.5); the online softmax, one
//     invocation a query; and O = O × α + P · V, each invocation 16 of the
//     M × d outputs, adjacent invocations adjacent words of each value row.
//   - Exponentials are base 2: queries are multiplied by scale × log2(e)
//     as they are staged, so a score is already in log2 units and exp2
//     replaces exp, FlashAttention-2's form.
//   - Masks: a key is live for query p when j <= p and j >= p − W + 1. A
//     masked score is never formed: its weight is set to 0 and it is left
//     out of the maximum, with an explicit "live" flag in place of −∞,
//     since WGSL lets an implementation assume no infinities. A tile with
//     no live key for a query leaves its state exactly as it was. Tiles
//     past a workgroup's last row are not loaded: causal tiles above the
//     diagonal are skipped.
//   - One source serves both roles: the override `combine` selects the
//     combine, which folds a query's splits and divides, d / 4 invocations
//     a query, each a vec4 of the output. The merge is one WGSL function
//     both roles call.
//   - Variants are override constants — `head_dimension`, `query_heads`,
//     `key_value_heads`, `combine` — so tile shapes and indices fold when
//     the pipeline is built; one pipeline per role for a model.
//   - Constants (binding 1): struct Attention { slots: u32, window: u32,
//     scale_log2e: f32 }, the layer's.
//   - Bindings: 2 the query buffer, 3 the layer's keys and 4 its values in
//     the cache, read-only; 5 the attention buffer, 6 the partial values and
//     7 the partial statistics, written. The combine binds 6 and 7
//     read-only and writes 5. Six storage buffers, under WebGPU's default 8.
//
// What it asks of the other contracts:
//   - kernels/interface.h and program.h: a launch may cover whole tiles of
//     rows, ceil(rows / rows_per_tile) of them; it may be split by key
//     chunks, naming its layer's window, so the program multiplies its
//     workgroups by key_chunks' split count for the step; and it may run
//     only when the step splits, which the combine does, so an unsplit step
//     pays no launch for it. The kernel computes the same first chunk and
//     split from the step's position and token count and its window, in
//     WGSL; the tests hold the two to agreement at every boundary.
//   - residency/plan.h: two working buffers — partial values, 512 rows of
//     H_q × d floats, and partial statistics, 512 rows of H_q × 2 — each a
//     buffer of its own.
//   - formats/format.h: the cache format's WGSL gains pack's inverse, fn
//     unpack4(words: vec2<u32>) -> vec4<f32>, reading no binding; F16's is
//     two unpack2x16float.
//   - model/model_description.h: the model's attention scale.
//
// Determinism (GDSA.2): run to run, and batch-invariant — a query's output
// is the same bits whatever the step's token count or splits, since chunks
// are fixed by position, each is computed by the same tile loop from an
// empty state, and they are folded in chunk order by one function, whether
// within a workgroup or by the combine. Not bit for bit with llama.cpp,
// whose tiles and order differ.
//
// Accuracy, against the same computation in f64 from the same queries and
// the cache's stored keys and values: the scores' rounding dominates. A
// d-term f32 dot product errs by up to d × 2⁻²⁴ × Σ_j |q_j k_j| × scale, in
// log2 units once scaled, and moves the key's weight by that much,
// relatively, through exp2; exp2's own 3 + 2|x| units and the f32 sums of at
// most 256 weights a chunk add far less. So each output is within max|v| ×
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
//     bytes a chunk, written and read, 16 KiB against the chunk's 1 MiB of
//     keys and values: 1.6%. Workgroups: 8 × ceil(L / 256), 128 at 4,096.
//     Launches: one a layer, and the combine's a layer once L passes 256 —
//     56 a step, 84 µs at 1.5 µs (interface.h).
//   - Prefill, 512 tokens from position 0: query-key pairs under the causal
//     mask, T(T + 1) / 2 = 131,328 a head; 2 × d multiply-adds a pair, the
//     score's and the weighted sum's, 4 × d operations: 1.08 × 10⁹ a layer,
//     30 × 10⁹ a step, about 2 ms at the M3 Max's roughly 14 f32 TFLOPS
//     (third-party figure) if the arithmetic ran at peak. A row tile streams
//     the keys up to its last row, half a layer's 2 MiB of keys and values
//     on average, so about 128 MiB a layer passes through the cache
//     hierarchy; from device memory, about the 2 MiB once.
//   - Barriers: four a tile, 64 a chunk for Qwen3, 32 for Llama 3.2, 128
//     for Gemma 3; the merge at each chunk's end needs none.
// Optimization (practice): the softmax is online and tiled, so the
// T × L scores are never written — FlashAttention-2 (GDSA.16).
// Optimization (practice): a decode step is split across the context in
// fixed chunks and combined, so 128 workgroups at 4,096 tokens read the
// cache where one a key-value head would be 8 — Flash-Decoding, vLLM's
// PagedAttention V2 (GDSA.8).
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
//     where FlashAttention-2 takes 64 to 128 rows; prefill streams each
//     layer's keys and values through the cache 128 times rather than 4 to
//     8. Their device-memory traffic is about the same; the cache's is not.
//   - Subgroup operations are optional, for the same reason: each score is
//     one invocation's whole dot product, and the softmax's per-tile maxima
//     and sums are one invocation's loop over B, where llama.cpp's Metal
//     kernels reduce across a SIMD-group.
//   - WebGPU has no matrix-multiply units: Chrome's subgroup matrices are
//     an experiment, not cross-browser. The arithmetic is f32 on the
//     shader cores.
//   - WGSL may assume no infinities: masks are flags, not −∞.
//   - WGSL zero-fills workgroup memory: 12.7 KiB for Qwen3, stored once a
//     workgroup — about 10% of the 128 KiB of keys and values the workgroup
//     reads from the cache for a chunk, though into on-chip memory: about
//     51 stores an invocation, against about 8,000 multiply-adds an
//     invocation a chunk.
//
// Levers not taken:
//   - Splits sized to fill the GPU rather than fixed by position: more
//     parallelism for a short context, and a result that would change with
//     the step's shape, which the batch-invariant fold rules out.
//   - The combine folded into the output projection's load: 28 launches a
//     decode step saved, 42 µs, for a matrix product that reads partials
//     and knows attention's merge.
//   - Larger tiles from a raised workgroup-memory limit: Apple adapters
//     offer 32 KiB; the harness asks for none of the defaults' limits
//     raised.
//
// Verification the implementation is held to, on the GPU against f64 from
// the same cache contents: each listed shape, prefill steps of 1, 3, 4 and
// 512 rows and decode steps, at positions 0, near 256's boundaries and deep
// into the context; Gemma 3's window over a ring that has wrapped; a step
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
};

// Keys a chunk holds, and the query rows the partial buffers hold.
inline constexpr std::uint32_t kChunkKeys = 256;
inline constexpr std::uint32_t kPartialRows = residency::kPrefillBlock;

// A layer's chunks for a step, and how it splits them.
struct KeyChunks {
    std::uint32_t first;    // the chunk holding the first row's earliest key
    std::uint32_t count;    // through the chunk holding the last row
    std::uint32_t splits;   // count, if every row's partials fit, else 1
};

// Preconditions: tokens >= 1 and window >= 1.
[[nodiscard]] constexpr KeyChunks key_chunks(std::uint32_t position, std::uint32_t tokens,
                                             std::uint32_t window) noexcept {
    const std::uint32_t earliest = position + 1 > window ? position + 1 - window : 0;
    const std::uint32_t first = earliest / kChunkKeys;
    const std::uint32_t count = (position + tokens - 1) / kChunkKeys - first + 1;
    return {first, count, tokens * count <= kPartialRows ? count : 1};
}

// The layer's attention launch and its combine, in that order.
// Preconditions: the head dimension is 64, 128 or 256; the query heads a
// key-value head serves divide 1,024 / d; `cache_format` has a pack and is
// the format the plan sized `cache` in.
[[nodiscard]] std::array<Launch, 2> attention_launches(const AttentionLaunch& attention);

}  // namespace bllm::kernels
