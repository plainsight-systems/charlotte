#pragma once

#include <cstdint>
#include <string>

#include "core/kernels/interface.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace bllm::sampler {

// Axis F: changes with a new sampling method. The settings a model samples
// with are policy, and are passed in.
//
// Contract 10: the sampler. The next token is drawn on the GPU, from the
// top kCandidates the selection kept (kernels/topk/topk.h), and stays there:
// the next decode step embeds it from the GPU, and the CPU reads it back a
// step behind, while that step runs. A token read back each step before the
// next could start would cost a map's round trip — 0.5 ms median on the
// target (research/2026-08-31-gpu-readback-round-trip.md) — on every token:
// 24 to 38% added to a Qwen3 decode step of 1.3 to 2.1 ms (graph/graph.h). That
// research recorded reading back each step, when a step was thought to take
// 20 to 50 ms; counted, it does not, and the decision is reversed (GPU.1,
// GPU.7).
//
// The draw, one workgroup of 64 invocations, over the candidates sorted
// largest first, in llama.cpp's order (common.h's default chain, at the
// commit tools/make_reference_logits.sh pins). Invocation i below top_k
// works candidate i — its two exponentials, its min-p test, and its division
// by top-p's total — and a candidate past top_k is never read; the last
// invocation runs Philox, side by side; invocation 0 alone
// takes the sums and running sums, in candidate order, so the draw does not
// depend on how the others are scheduled:
//   1. top-k: the first top_k candidates.
//   2. top-p: below 1, their softmax at temperature 1 in f32 — each weight
//      exp(logit − first's) divided by their total, as llama.cpp's does —
//      and the shortest prefix whose probabilities, summed in order, reach
//      top_p, one candidate at least; at 1, nothing — as llama.cpp's top-p
//      returns at once for p of 1 or more — since an f32 softmax would round
//      a far tail to zero and drop it. Dividing first, not comparing a
//      running weight with top_p × the total, keeps llama.cpp's survivors:
//      the two round differently in f32 near the edge.
//   3. min-p: of those, each whose logit is at least the first's + ln
//      min_p — ln min_p taken once a turn, on the CPU in f32 as llama.cpp's
//      logf does, and carried in the step, not once a candidate.
//   4. temperature: zero draws the first candidate; otherwise each
//      survivor's weight is exp((logit − first's) / temperature) — a second
//      pass of exponentials, since top-p's were at temperature 1 — and the
//      draw is by inverse transform: the first survivor whose running sum of
//      weights exceeds u × their total, or, should rounding leave none, the
//      last survivor. At most 64 candidates, so at most 128 exponentials,
//      and one uniform draws where a random-key race spends one a candidate
//      (GDSA.21's caveat).
//   - u is Philox4x32-10's first word (Salmon et al., SC '11) keyed by the
//     turn's 64-bit seed with the drawn token's position as the counter —
//     the step's position plus its tokens, so a token's draw is the same
//     whether its prompt was prefilled in one step or resumed from the
//     cache: its top 23 bits scaled by 2⁻²³, plus 2⁻²⁴, so u lies in
//     [2⁻²⁴, 1 − 2⁻²⁴], every value exact in f32 and neither 0 nor 1. A pure
//     function of seed and position, so a run replays from its seed and any
//     step replays alone (GDSA.3). WGSL has no 64-bit integers; Philox's 32-bit high products
//     are formed from 16-bit halves. The same draw on the target's GPU and
//     compiler; WGSL's exp may round differently elsewhere, which moves a
//     draw only when u falls within that rounding of a boundary.
//   - Weights and sums are f32: each survivor's share is within a few units
//     in the last place of its exact value, the 64 summed in order.
//   - Failure is visible: when the first candidate's logit is not finite,
//     the draw writes token 0 and sets `failed`, so the next step embeds a
//     real token, and the runtime, reading the result a step later, stops the
//     turn with the step's logits named not finite (GDSA.21: reject NaN and
//     +∞).
//   - Settings are checked before a turn begins, by check below, never
//     clamped: 1 ≤ top_k ≤ kCandidates, temperature finite and ≥ 0, top_p
//     in (0, 1], min_p in [0, 1). A top_k of 0, llama.cpp's "every token",
//     is refused by name, as is one above 64: each listed model's card asks
//     for 64 or fewer — Gemma 3 64, Qwen3 20 — and llama.cpp's default is
//     40.
//   - Writes the sampled record to `sampled`, read by the next step's
//     gather and copied to the readback ring (below), and the 64 candidates
//     stay in `candidates` for a test, or a future display of alternatives,
//     to read.
//
// What it asks of the other contracts:
//   - kernels/interface.h: the step's head grows from 16 bytes to 48: after
//     position, tokens and logits, `fed` — 1 when the step's one token is
//     the last step's draw, read from `sampled` — then the seed's two words,
//     top_k, temperature, top_p and min_p, and two of padding; the
//     identifiers follow at 48. One declaration of it in WGSL,
//     kernels/step.wgsl, composed before every kernel by the program, as a
//     format's unpack is, in place of the five each kernel holds today. A
//     decode step writes 48 bytes, where it wrote 32.
//   - kernels/gather: with `fed`, row 0's identifier is `sampled`'s token,
//     bound read-only; a draw's token is below the vocabulary by
//     construction.
//   - residency/plan.h: working buffers `partials_a`, 64 · ceil(V / 1,024)
//     pairs of 8 bytes — 76 KB for Qwen3, 131 KB for Gemma 3 — and
//     `partials_b`, 64 for each 1,024 of those — 5 KB and 8 KB —
//     `candidates`, 512 bytes, and `sampled`, 16.
//   - graph/graph.h: output() appends the selection and the draw after the
//     head; all of them run only in a step that asks for logits.
//   - kernels/program.h: a step that asks for logits copies `sampled` into
//     one of two mappable readback slots, created at build, alternating,
//     after its pass, and maps it once submitted; its callback carries the
//     sampled record. Two steps may be outstanding — one running and the
//     next queued — so the next is submitted before the last's record maps;
//     WebGPU orders the step uniform's write after the submits before it, so
//     one uniform buffer serves both. WebGPU does not order mappings of two
//     buffers, so each slot carries its step's sequence number, and a record
//     that maps before an earlier step's is held until that one has been
//     delivered: callbacks reach the runtime in the order the steps were run.
//     A third run waits for the first's callback, so a slot is unmapped
//     before the copy that reuses it (GPU.7).
//   - runtime/runtime.h: decode steps are submitted back to back, each
//     `fed`; a step's token is known when its record maps, one step later.
//     So when a token is a stop, or the turn is cancelled, the step already
//     queued behind it runs anyway, about 1.7 ms, and its draw is
//     discarded. It wrote the keys and values of its input — the stop
//     token, or after a cancel a draw never accepted — at its position. The
//     runtime advances the cache for every step it runs and keeps each
//     position whose input token is accepted (runtime/turn.h): the stop
//     token's, which the next turn's prompt holds there when its template
//     closes the reply with it, and the diff truncates when it does not; and
//     not a draw the turn discarded. A position truncated lies past the
//     cache's length, and its high-water mark counts it (cache/kv.h):
//     overwritten by the next token there, never read before it, and a
//     sliding-window ring, whose slots exceed its window by a prefill block,
//     loses no entry a query still reads. The runtime never queues a step at
//     or past the context offered, where a full-attention layer, whose slots
//     are the context, would wrap onto position 0; nor past a turn's token
//     limit, which it knows ahead. This is the cache state the research
//     warned a pipeline puts at risk, stated so the
//     runtime's tests can hold it.
//
// What it costs, a sampled step: the draw is one launch, 1.5 µs. Each of the
// first top_k invocations loads its candidate's logit and the first's, and
// invocation 0 the chosen token: 8 × top_k + 4 bytes logically, of 4 ×
// top_k + 4 distinct, which the GPU's caches serve — 164 and 84 at Qwen3's
// top_k of 20, 516 and 260 at 64 — and it writes 16. Its work, in order:
//   - side by side: each of the first top_k invocations loads its candidate
//     and the first from device memory, computes its exponentials and its
//     min-p test, and writes them to workgroup memory; the others, past
//     top_k, do none of it, and wait at the barriers — a split by
//     invocation, so on a SIMD group straddling top_k its idle lanes cost
//     the group's issue slots, not its time; the last runs Philox's 10
//     rounds and writes u;
//   - a barrier; invocation 0 sums top-p's weights, at most top_k dependent
//     adds, and writes the total;
//   - a barrier; each of the first top_k invocations divides its own weight
//     by it;
//   - a barrier; invocation 0 runs the running sums, at most top_k
//     iterations — the tempered weights' and, beside them, top-p's
//     cumulative probability, stopping at top-p's cut — then the min-p scan
//     and the draw's comparisons with u × the survivors' total, at most
//     top_k iterations each, and writes the record.
// Three barriers, and on invocation 0, in sequence, four loops of at most
// top_k iterations — 256 at a top_k of 64, 80 at Qwen3's 20. At a top_p of
// 1, top-p's weights, total, division and cumulative sum are not computed:
// one barrier, and three loops.
// Optimization (practice): only the first top_k candidates are read and
// worked, and none of top-p's work is done where it would keep every
// candidate. The branch on top_p is uniform; the one on top_k splits
// invocations, which is cheaper than working candidates never read.
// An estimate of its latency at the worst, a top_k of 64 and a top_p below
// 1, on cores near 1.4 GHz from about 400 cycles for the loads, 50 a barrier
// and 8 an iteration with its read, add or compare, and loop control: 2,598
// cycles, 1.9 µs, leaving out the exponentials' and the divisions' latency,
// each on the path between two barriers, and the issue slots Philox takes.
// An estimate, partial, not a bound: the selection's stages and this are
// measured together once they run, and that measure, not this, says what
// share of a decode step they take.
// The readback copies 16 bytes and maps them while the next step runs, so a
// decode step's critical path no longer holds the map's round trip, 0.5 ms
// median and 0.8 ms at p95 on the target. On the CPU, a token costs one
// mapped read of 16 bytes and the runtime's work on it, a step behind.
// Optimization (practice): the token drawn and kept on the GPU, the readback
// a step behind, as vLLM samples on the device to avoid synchronizing with
// the CPU (GDSA.21, GPU.1, GPU.7).
//
// Levers not taken:
//   - The draw folded into the selection's last pass, whose one workgroup
//     already holds the 64 candidates: one launch fewer, about 1.5 µs, three
//     calls out of the module — its pipeline, bind group and dispatch — and
//     its candidate loads, 8 × top_k + 4 bytes logically, a sampled step —
//     0.09% of
//     a decode step — at the cost of the sampling method living inside the
//     selection kernel, so that a new method would change the selection
//     (docs/architecture/change-axes.md: axis F apart from E).
//
// Verification the implementation is held to:
//   - On the GPU against a CPU reference of the same steps in f64, given
//     the uniform the draw records: each of 1,000 draws over five settings —
//     llama.cpp's defaults, the listed models' cards, and others at the
//     truncations' edges — the reference's token, but for a draw within f32's
//     rounding of a decision, which may number 1 in 100 at most; temperature
//     zero, a top_k of 1, a top_p and a min_p that leave the first alone, each
//     drawing the first candidate; and, for fixed logits, the frequencies of
//     2,000 draws over consecutive positions within the 0.1% chi-squared
//     bound of the reference's probabilities, no truncated candidate ever
//     drawn (GDSA.21: test statistically, not by replay).
//   - Philox4x32-10 on the CPU against Random123's published known-answer
//     vectors, and the GPU's uniform bit for bit against it, across seeds and
//     positions.
//   - A replayed step draws the same token; a non-finite top candidate sets
//     `failed`; check refuses each setting outside its range, naming it.
//   - Through the forward pass: greedy decoding fed on the GPU gives the
//     tokens read back by the same draw run a step at a time.
//   - Through the runtime, the cache after a step is discarded. In each case
//     below, the turn that follows gives the same bits — every logit it
//     reads back, and the tokens it draws — as the same conversation run in
//     a fresh upload with no step discarded, so a stale entry the discarded
//     step left, or one it should have left and did not, shows as a
//     difference (graph/graph.h: logits do not depend on how a prompt is
//     stepped):
//       - a stop token drawn by the first decode step, the queued step behind
//         it discarded;
//       - a stop token drawn one position before the context offered, where
//         no step is queued behind it, and none is queued at the context's
//         end;
//       - a cancel while a step is queued behind the one running;
//       - the next turn's prompt sharing the whole accepted history, so the
//         cache is reused up to it and the discarded position is written
//         again;
//       - each of these on Gemma 3's sliding-window layers after the ring has
//         wrapped, so the discarded step's slot held a position still in an
//         earlier window.
//     These are where pipelined engines have failed: vLLM's async scheduling
//     in a request stopping with a step in flight while preempted, where one
//     count served both for positions holding keys and values and for tokens
//     to keep (vllm-project/vllm#58776); llama.cpp's speculative decoding,
//     the nearest thing it has, removes a rejected draft's entries by
//     position (llama_memory_seq_rm) and restores a checkpoint where it
//     cannot. Here the cache keeps the two counts apart, written and length
//     (cache/kv.h), and these cases hold them.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — the draw is a function of the
//            candidates, the settings, the seed and the position.
//     I.5    State preconditions — check, and the draw's.
//   C++ performance guidelines
//     GDSA.21 Draw categorical samples — by inverse transform here, its
//            caveat for weights already computed.
//     GDSA.3 Derive random numbers from (seed, stream, counter) — Philox,
//            keyed by the seed, counted by position.
//     GPU.1  Keep data on the device; every round trip needs a budget — the
//            token stays on the GPU.
//     GPU.7  Pipeline CPU and GPU work — the readback a step behind, in a
//            ring of two slots.

// What the draw writes, and the readback carries.
struct SampledRecord {
    std::uint32_t token;
    std::uint32_t failed;   // 1 when the top candidate's logit was not finite
    float u;                // the uniform the draw used: a step replayed shows the same
    std::uint32_t padding;
};
static_assert(sizeof(SampledRecord) == 16);

// What check refuses, naming the setting and its range.
struct SettingsResult {
    bool ok = true;
    std::string subject;
};

[[nodiscard]] SettingsResult check(const policy::SamplingSettings& settings);

// Writes the draw's fields of `step`: the seed, the settings, and ln min_p.
// Precondition: check(settings) passed.
void apply(const policy::SamplingSettings& settings, policy::Seed seed, kernels::Step& step);

// The draw's launch: over `candidates`, kernels::kCandidates (logit, token)
// pairs sorted largest first, into `sampled`, one SampledRecord; the
// settings and the seed are the step's (kernels/interface.h).
[[nodiscard]] kernels::Launch draw_launch(const residency::BufferRange& candidates,
                                          const residency::BufferRange& sampled);

}  // namespace bllm::sampler
