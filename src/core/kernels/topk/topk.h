#pragma once

#include <cstdint>
#include <vector>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// Top-k selection: the step's logits, one row of the vocabulary, reduced to
// its kCandidates most likely tokens, sorted, for the sampler's draw
// (sampler/sampler.h). A selection, not a sort of the vocabulary (GDSA.7):
// k is 64, well inside an on-chip selection's reach, so each workgroup keeps
// the top 64 of its tile in workgroup memory and passes them on.
//
//   - Order: by logit, largest first, and between equal logits the lower
//     token first — a total order, so the 64 kept are exactly the first 64
//     of the whole row under it, whatever the tiling (GDSA.2). A logit's f32
//     bits become an order-preserving u32 key — −0 first made +0, so the two
//     zeros, equal as logits, tie and fall to the lower token; then the sign
//     bit flipped for a positive number, every bit for a negative — compared
//     as integers, so no
//     float comparison decides it and WGSL's freedom to assume no NaN or
//     infinity cannot reorder it. Every NaN, whatever its sign, is given the
//     largest key, above +∞, so any NaN in the row reaches the top, where the
//     draw refuses a candidate that is not finite (sampler.h): reported, never
//     sampled, and never summed into a softmax. −∞ keeps its place below
//     every finite logit, a weight of zero.
//   - A tile is 1,024 entries, a workgroup of 256 invocations, 4 each: keys
//     and token ids, 8 KiB of workgroup memory. Bitonic select (Shanbhag,
//     Pirk, Madden, SIGMOD 2018): each run of 64 sorted by a bitonic network,
//     21 stages; then runs merged in pairs, each pair's elementwise larger
//     halves forming a bitonic sequence sorted in 6 more stages, 16 runs to
//     8 to 4 to 2 to 1 — 49 stages, a barrier each, the active invocations
//     halving with the runs. The workgroup writes its 64, sorted.
//   - Passes: the first reads logits (f32, a token's id its index); each
//     later one reads the last's candidates (logit, token), 1,024 a
//     workgroup; the last runs one workgroup and writes `candidates`.
//     Between them two working buffers alternate, so no pass binds one
//     buffer both read-only and writable. A vocabulary of V takes passes
//     until one workgroup holds every survivor: V → 64·ceil(V / 1,024) → …
//     — 3 passes for every listed model: Qwen3's 151,936 to 9,536 to 640 to
//     64, Llama 3.2's 128,256 to 8,064 to 512 to 64, Gemma 3's 262,144 to
//     16,384 to 1,024 to 64.
//   - Every pass covers the last token alone, so runs only in a step that
//     asks for logits (kernels/interface.h).
//   - Constants (binding 1): struct TopK { count: u32 } — the entries the
//     pass reads. Bindings: 2 its input, 3 its output. Two entry points:
//     `first`, over logits, and `merge`, over candidates.
//
// What it costs, counted, for Qwen3: reads 608 KB of logits and writes
// 76 KB, then 76 KB to 5 KB, then 5 KB to 512 bytes: 771,072 bytes, 1.93 µs
// at 400 GB/s. Launches: 3, and the draw's, 6 µs at 1.5 µs (interface.h).
// Those 7.9 µs are the floor. Comparisons: a tile's 21 sorting stages are 16
// runs × 32 × 21 = 10,752 compare-exchanges, its merges 15 × (64 + 6 × 32) =
// 3,840: 14,592 a tile, 2.3 million over the 160 tiles of the three passes,
// run side by side within a pass. What a pass's latency adds is its 49
// stages one after another, each a barrier: at about 50 cycles a stage on
// the M3 Max's cores near 1.4 GHz — an estimate, not counted from the
// design, to be calibrated — 1.75 µs a pass, 5 µs for the three; and the
// draw's own work, estimated at 1.5 µs at a top_k of 64 (sampler.h). About
// 15 µs a sampled step, estimated: the selection and draw are measured
// together once they run, and that measure gives their share of a step.
// Optimization (practice): select, not sort — 64 kept from each tile of
// 1,024, never a sorted vocabulary, as FlashInfer's and Faiss's GPU
// selection do (GDSA.7).
//
// Levers not taken:
//   - Selection fused into the head's matrix product, so logits are never
//     written (GDSA.7): a decode workgroup produces 8 logits, so its top 64
//     keeps all 8, and carrying them on as (logit, token) pairs writes 8
//     bytes a token where the logits write 4 — 16V bytes in and out against
//     the 8V of writing the logits and selecting from them, 1.2 MB more for
//     Qwen3, and no launch saved. Fusion pays only with the head retiled so
//     a workgroup produces hundreds of logits; its decode form is shaped
//     for reading the weights once (matmul.h), and is not retiled for this.
//   - Radix select, which has no ceiling on k (GDSA.7): k is fixed at 64,
//     where bitonic select is the simpler of the two, and radix select's
//     histogram passes need atomics or a launch per digit.
//
// Verification the implementation is held to, on the GPU against a CPU
// sort under the same order: rows of each listed vocabulary's size and of
// sizes leaving the last tile part full; ties at the 64th place, broken by
// the lower token; rows of one repeated value; and −∞ entries.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.5    State preconditions — on topk_launches.
//   C++ performance guidelines
//     GDSA.7 Select top-k with a selection primitive, not a full sort.
//     GDSA.2 Declare each floating-point reduction's determinism level — a
//            total order on integer keys: the same 64 for any tiling.
//     GDSA.6 Count passes over global memory — the bytes above.
//     GPU.6  Batch tiny GPU work — 3 launches, counted.

// The candidates a selection keeps, and the draw reads; the plan sizes the
// buffers by it (residency/plan.h).
inline constexpr std::uint32_t kCandidates = residency::kCandidates;

// The launches that reduce `logits`, one row of `vocabulary` f32, to the
// top kCandidates in `candidates` — (logit, token) pairs, sorted — through
// `partials_a` and `partials_b` alternately. Preconditions: vocabulary >=
// kCandidates; `partials_a` holds n₁ = 64 · ceil(vocabulary / 1,024) pairs
// and `partials_b` 64 · ceil(n₁ / 1,024), as the plan sizes them; each later
// pass writes fewer than the one two before; `candidates` holds kCandidates.
[[nodiscard]] std::vector<Launch> topk_launches(const residency::BufferRange& logits, std::uint32_t vocabulary,
                                                const residency::BufferRange& partials_a,
                                                const residency::BufferRange& partials_b,
                                                const residency::BufferRange& candidates);

}  // namespace bllm::kernels
