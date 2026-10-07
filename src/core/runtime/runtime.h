#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/cache/kv.h"
#include "core/kernels/program.h"
#include "core/policy/policy.h"
#include "core/runtime/stops.h"
#include "core/runtime/turn.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// The turn loop. A model loaded is a program (kernels/program.h), its cache
// (cache/kv.h) and its stop set (stops.h); a turn takes the whole rendered
// conversation as tokens and streams the reply's tokens back. Tokens in,
// tokens out: encoding the prompt and decoding the reply into text are the
// boundary's, with the tokenizer (tokenizer/tokenizer.h), so the loop is
// tested on identifiers alone.
//
// A turn, started with the prompt, the turn's policy and two callbacks:
//   1. Refused at once, without a callback, when it cannot run (StartError):
//      a turn still running; a lost device; an empty prompt; a prompt longer
//      than the context offered, naming both counts so the page can drop its
//      oldest messages and render again (logical-overview.md); settings
//      check refuses (sampler/sampler.h), or a max_tokens of 0.
//   2. The diff: the longest common prefix of the tokens the cache holds and
//      the prompt (cache/prefix.h), less one when it is the whole prompt —
//      the cache holds keys and values, not logits, and the first draw needs
//      the last prompt token's. The cache is truncated to it and keeps it, or
//      empties, for a ring rolled back past its reserve (cache/kv.h).
//   3. Steps, as the turn plans them (turn.h): the prompt's uncached tokens
//      in prefill blocks, then decode steps fed on the GPU. Each step run
//      advances the cache by its tokens, and copies a prefill block's
//      identifiers into the step; the turn's settings and seed are written
//      into the step once (sampler::apply).
//   4. As each step reports, in the order run: the turn decides its draw;
//      the next steps it plans are run; then an emitted draw's token is
//      passed to the token callback. The next step goes to the GPU before the
//      CPU's work on the token (GPU.7), and so a cancel from the token
//      callback finds that step queued — the case the cache must hold.
//   5. Finished: the cache is truncated to what the turn keeps, and the
//      held tokens to match; then the turn callback, once, with how it ended.
//
//   - Invariant, between turns: the tokens held are exactly the cache's
//     length, those whose keys and values it holds. During a turn they also
//     hold the prompt's uncached tokens and the accepted draws.
//   - Cancel ends the running turn at its next report; the steps already run
//     report first, and are discarded (turn.h). A cancel with no turn running
//     does nothing.
//   - Failure is visible, and leaves the cache consistent: a step's failure
//     empties it, so the next turn prefills from the first token; a
//     non-finite draw ends the turn NonFinite, the cache kept up to it; a lost
//     device refuses every later turn.
//   - Callbacks run inside the program's, on the thread that processes the
//     device's events. The token callback may cancel; the turn callback may
//     start the next turn or destroy the runtime.
//   - It owns the program, and so outlives none of the program's callbacks:
//     they carry the runtime's state, held by those in flight as the
//     program's is, so destroying a runtime with a turn running finishes that
//     turn Cancelled, its callback called. The upload the program borrows
//     outlives the runtime (I.11).
//
// Diagnostic builds only (core/diagnostics.h, TLM.6): observe_steps sets a
// step observer, or clears it, between turns. While one is set every step
// runs profiled — kernels/program.h's run_profiled, through all its
// launches — timed by the GPU at its one pass's beginning and end and
// reading back its record as run() does, so the turn's tokens are the same.
// As each step reports, before the turn's work on it, the observer is
// called with the step's kind, position and tokens and its two GPU times.
// The runtime reads no clock: a caller wanting the CPU's time of each
// report reads its own clock in the observer, and the two clocks are never
// subtracted (TLM.11). What it perturbs: each step also writes two
// timestamps and resolves and reads back two queries, within its one
// submit. The clean build has none of it, which
// tools/check_diagnostics_excluded.sh checks.
//
// What it costs, counted:
//   - A decoded token, on the CPU: one report from the program, the turn's
//     constant work, at most kMaxStops comparisons, one run of the program —
//     17 calls into WebGPU and 2 or 3 a launch, and 48 bytes written
//     (program.h) — 4 bytes
//     appended to the held tokens, and one token callback. Nothing waits on
//     the GPU: the token is read a step behind, while the next step runs.
//   - A turn, once: the diff, one comparison a held token, at most the
//     context offered — 40,960 for Qwen3, about 15 µs (cache/prefix.h); the
//     cache's truncation, a visit to each layer, twice; the prompt's uncached
//     identifiers copied once into the held tokens and once into the steps,
//     4 bytes each; and a prefill step for each kPrefillBlock of them.
//   - Its first token waits on the prefill steps and the last one's map, 0.5
//     ms median on the target (sampler/sampler.h); every later token arrives
//     a step after it was drawn. A stop is reported once any step queued
//     behind it has run — at most one; none at the limit or the context. A
//     cancel waits for what remains of every step already run, at most two,
//     since the next step is run before the token callback that may cancel:
//     at most two decode steps' time — each at position p reading its
//     weights and cache, 0.94 ms + 0.29 µs × p for Qwen3 from device memory
//     (graph/graph.h) — or two prefill blocks', less what they had already
//     done when the cancel came.
//   - Nothing is allocated after construction: the held tokens are reserved
//     at the context offered plus one, the step is the runtime's own, and the
//     turn is a value (MEM.9).
//
// What it asks of the other contracts:
//   - policy/policy.h: TurnPolicy carries max_tokens, the most tokens a
//     reply may draw, checked at least 1; the context offered bounds it
//     first. LoadPolicy carries the stop texts the model's generation config
//     lists beyond its file's (stops.h); none, for an unmeasured model.
//   - kernels/program.h: in diagnostic builds, its launch count, the
//     profiled step's bound.
//   - sampler/sampler.h: the cache after a step queued behind a turn's end
//     keeps each position whose input token is accepted (turn.h), the stop
//     token's among them, where it said the queued step's position is
//     truncated.
//
// Verification the implementation is held to:
//   - The turn, on the CPU (turn.h): the steps planned — the prefill blocks'
//     positions, sizes and logits; decode steps at prompt + j; never more
//     than two outstanding, nor one at the context or past the limit — and
//     what each way of ending emits and keeps: a stop at draw 0 and at a later
//     draw, with a step queued behind it and with none; a cancel in prefill,
//     with a decode step queued, and after the turn ended; a non-finite draw;
//     the limit at 1 and beyond; a prompt that fills the context, and one a
//     token short of it; each kind of step failure, and a cancellation by the
//     program; reports after the end discarded.
//   - The stop set, on the CPU, from each listed model's file and its
//     policy's texts, and each refusal by name.
//   - On the GPU, the cache cases sampler/sampler.h lists: each turn that
//     follows a discarded step gives the same bits as the same conversation
//     run as one prompt in a fresh upload. That turn ends at its limit, so
//     no step is discarded after it, and its drawn tokens and the 64
//     candidates its last step selected are compared bit for bit. The stop
//     set is the token a first run shows is drawn at the wanted step, and the
//     context a cache made that small; Qwen3's file for the full-attention
//     cases, and a synthetic Gemma 3 of forward_model's sizes, window 16 and a
//     rollback reserve of 16, for the ring past wrapping.
//   - On the GPU, the diff: a second turn reuses the common prefix and
//     prefills only the rest, reported as reused; a prompt wholly held
//     recomputes its last token; a turn's greedy tokens are those decoding
//     a step at a time gives; a cancel in prefill keeps the blocks run;
//     destroying a runtime mid-turn reports Cancelled; and each refusal.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — the turn's decisions are turn.h's; this
//            sequences the program and the cache around them.
//     E.27   Use error codes systematically — StartError, and TurnFailure.
//     I.11   Never transfer ownership by a raw pointer — the program is
//            owned, by unique_ptr; the upload borrowed, and stated so.
//     I.5    State preconditions — on construction and on start.
//   C++ performance guidelines
//     GPU.7  Pipeline CPU and GPU work — two steps in flight; the next run
//            before the CPU's work on a token.
//     TLM.6  Label measurement modes — the step observer is diagnostic
//            builds' alone.
//     TLM.11 Never align clocks without a calibrated pair — GPU times out,
//            the caller's CPU clock its own.
//     GPU.1  Keep data on the device — the token is fed on the GPU, and only
//            its 16-byte record is read back.
//     MEM.9  Allocate at init — every buffer the turn touches, at
//            construction.
//     WASM.2 Batch work across the JS boundary — the prompt crosses once a
//            turn; a token, once.

enum class StartError {
    Ok,
    Busy,             // a turn is running
    DeviceLost,       // a step reported the device lost; no turn can run
    EmptyPrompt,
    PromptTooLong,    // the subject names the prompt's tokens and the context offered
    Settings,         // the subject names the setting and its range
};

struct StartResult {
    StartError error = StartError::Ok;
    std::string subject;
    // For PromptTooLong, as numbers, so the page can shorten the
    // conversation by the excess: the prompt's tokens — or, for text refused
    // before it is encoded, the fewest it makes — and the context offered.
    std::uint64_t prompt_tokens = 0;
    std::uint32_t context = 0;
};

struct TurnResult {
    TurnFailure failure;
    TurnEnd end;                      // meaningful when failure is None
    std::string_view message;         // WebGPU's, for a step's failure
    std::uint32_t prompt;             // the prompt's tokens
    std::uint32_t reused;             // of those, the ones the cache already held
    std::uint32_t emitted;            // the reply's tokens passed to the token callback
    std::optional<tokenizer::TokenId> stop;   // the stop token drawn, for TurnEnd::Stop
};

// Each emitted token, in order, a step after it was drawn.
using TokenCallback = void (*)(tokenizer::TokenId token, void* userdata);
// Once a turn, when it has finished. `result.message` is valid only during
// the call.
using TurnCallback = void (*)(const TurnResult& result, void* userdata);

class Runtime {
public:
    // Preconditions: `program` was built for the model's graph, reading back
    // its draw's record (graph/graph.h); `cache` is empty and its capacity at
    // most the context the plan offers; `stops` is not empty.
    Runtime(std::unique_ptr<kernels::Program> program, cache::KvCache cache, StopSet stops);

    // Starts a turn over `prompt`, the whole rendered conversation. On Ok,
    // `on_token` is called for each token emitted and `on_turn` once, both
    // with `userdata`; otherwise neither is. Preconditions: every token below
    // the vocabulary; `prompt` is not read after the call.
    [[nodiscard]] StartResult start(std::span<const tokenizer::TokenId> prompt, const policy::TurnPolicy& policy,
                                    TokenCallback on_token, TurnCallback on_turn, void* userdata);

    // Ends the running turn at its next report; with none, nothing.
    void cancel() noexcept;

    // The context offered: the most tokens a prompt and its reply hold.
    [[nodiscard]] std::uint32_t capacity() const noexcept;

    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // What the runtime and its in-flight callbacks share; defined only where
    // the runtime is.
    struct State;

private:
    std::shared_ptr<State> state_;   // shared with in-flight callbacks
};

}  // namespace bllm::runtime
