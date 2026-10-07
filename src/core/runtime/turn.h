#pragma once

#include <cstdint>
#include <optional>

#include "core/sampler/sampler.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// One turn's steps and what becomes of each draw: which step runs next,
// whether a draw is emitted, ends the turn or is discarded, and how much of
// the cache the turn keeps. Pure — no GPU object, no cache, no callback — so
// every way a turn ends is tested on the CPU, the step queued behind a stop
// included, which a device cannot be made to produce on demand (F.8). The
// runtime (runtime.h) runs the steps it plans and hands it what they report.
//
// A turn, over a prompt of `prompt` tokens of which the cache holds the
// first `cached`, a context of `capacity` and a limit of `max_tokens`:
//   - Prefill: positions cached .. prompt − 1, in blocks of at most
//     kPrefillBlock tokens, in order; only the last asks for logits, and its
//     draw is draw 0.
//   - Decode: step j, for j = 0, 1, …, feeds draw j — `fed`, one token, at
//     position prompt + j — and asks for logits; its draw is draw j + 1. It
//     is planned only when prompt + j < capacity, so no step is queued at or
//     past the context, where a full-attention layer would wrap onto position
//     0; and when j + 1 < max_tokens, so none past the limit, which is known
//     ahead.
//   - At most kernels::Order::kOutstanding steps outstanding: the next is
//     planned as soon as one reports, so the step that feeds a draw is queued
//     before that draw is known on the CPU (sampler/sampler.h).
//   - Each draw, in order, when its step reports:
//       - failed — the top candidate not finite: the turn ends NonFinite,
//         the draw not accepted;
//       - a stop token: the turn ends Stop, the draw accepted, not emitted;
//       - otherwise accepted and emitted; the turn then ends Limit when it
//         was the max_tokens-th, or Context when no step can feed it.
//     A draw reported after the turn has ended — by a step already queued —
//     is discarded: neither accepted nor emitted.
//   - Cancel ends the turn Cancelled: nothing more is planned, and every
//     later draw is discarded.
//   - A step that fails ends the turn with the step's failure; one cancelled
//     with the program ends it Cancelled.
//   - The turn is finished once it has ended and every step it ran has
//     reported.
//
// What the cache keeps. A step writes the keys and values of its input
// tokens, at their positions. The accepted sequence is the prompt, then the
// accepted draws. The cache keeps each position whose input token is
// accepted and whose step ran — min(written, accepted) positions, where
// written counts every position a step ran at — and the runtime truncates to
// that. So, for a turn that ends:
//   - Stop at draw j: the step queued behind it, decode step j, fed the stop
//     token, and its entry is kept — the entry the next turn computes when
//     its template closes the reply with that token, and one the diff
//     truncates when it does not. That step's own draw is discarded. Where
//     no step was queued, at the context or the limit, there is no entry.
//   - Cancelled after draw j was emitted: the step feeding draw j + 1, queued
//     but unknown on the CPU, wrote one position past the accepted sequence,
//     and it is truncated. Cancelled in prefill: the blocks already run are
//     kept.
//   - NonFinite at draw j: the step that fed it wrote token 0 in its place,
//     and it is truncated.
//   - Limit or Context: no step ran past the accepted sequence; every
//     accepted draw has its entry but the last.
//   - A step failure: nothing is kept. A failed step may have written part
//     of its entries, and the next turn prefills from the first token.
// A rollback is never more than one position, within every ring's prefill
// block of spare slots (cache/kv.h), so a sliding-window cache keeps the rest.
//
//   - Invariants: reported <= run <= reported + kOutstanding; a step is
//     planned only before the end; the kept length is at most the accepted
//     length and at most the written length.
//
// What it costs: constant time and no allocation a step and a draw, on a
// class of a few counters made once a turn.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — the turn is decided apart from the GPU,
//            and its every end is tested without one.
//     C.2    Use class if the class has an invariant — the ones above.
//     I.5    State preconditions — on construction and on each report.
//   C++ performance guidelines
//     GPU.7  Pipeline CPU and GPU work — a step is queued before the draw
//            it feeds is known, and its draw is read a step behind.

// How a turn ended, when it did not fail.
enum class TurnEnd {
    Stop,        // a stop token was drawn
    Limit,       // the turn's max_tokens were emitted
    Context,     // the next step would be at the context offered
    Cancelled,   // cancel, or the program destroyed with steps in flight
};

// Why a turn failed.
enum class TurnFailure {
    None,
    NonFinite,    // the top candidate's logit was not finite
    Step,         // a step failed validation, or the device reported an internal error
    DeviceLost,
};

// Why a step did not complete, as the program reports it (kernels/program.h).
enum class StepFailure {
    Step,
    DeviceLost,
    Cancelled,   // the program was destroyed with the step in flight
};

// A step the turn plans: the step's head, and which of the prompt's tokens
// it carries, for a prefill block.
struct Planned {
    std::uint32_t position;
    std::uint32_t tokens;
    bool logits;
    bool fed;
    std::uint32_t first;   // the prompt index of its first token; unused when fed
};

// What becomes of a draw.
enum class Draw {
    Emit,       // accepted and emitted
    Stop,       // accepted, not emitted; the turn has ended
    Discard,    // neither: the turn ended before it
    Failed,     // not accepted; the turn has ended NonFinite
};

class Turn {
public:
    // Preconditions: cached < prompt <= capacity <= kernels::kMaxPositions,
    // max_tokens >= 1.
    Turn(std::uint32_t cached, std::uint32_t prompt, std::uint32_t capacity, std::uint32_t max_tokens) noexcept;

    // The next step to run, counted run; none when kOutstanding steps are
    // outstanding or nothing more is planned.
    [[nodiscard]] std::optional<Planned> next() noexcept;

    // The earliest outstanding step reported done: for one that asked for
    // logits, `record` is its draw, and the result says what becomes of it;
    // `stop` is whether its token is a stop token. Precondition: a step is
    // outstanding; `record` is given exactly when that step asked for logits.
    Draw report(const std::optional<sampler::SampledRecord>& record, bool stop) noexcept;

    // The earliest outstanding step did not complete: it failed, or was
    // cancelled with the program. A failure ends the turn with it, unless the
    // turn has already failed; a cancellation ends it Cancelled, unless it
    // has already ended. Either way the step counts as reported.
    // Precondition: a step is outstanding.
    void fail(StepFailure failure) noexcept;

    // Ends the turn Cancelled, unless it has already ended.
    void cancel() noexcept;

    [[nodiscard]] bool ended() const noexcept;
    // Ended, and every step run has reported.
    [[nodiscard]] bool finished() const noexcept;
    // Precondition for these three: ended().
    [[nodiscard]] TurnFailure failure() const noexcept;
    [[nodiscard]] TurnEnd end() const noexcept;    // meaningful when failure() is None
    // Positions the cache keeps: min(written, accepted), or 0 after a step
    // failure. Precondition: finished().
    [[nodiscard]] std::uint32_t kept() const noexcept;

    // Draws emitted so far.
    [[nodiscard]] std::uint32_t emitted() const noexcept;
    // Positions the steps run so far have written: the cached prompt
    // tokens, then one a token each step carried.
    [[nodiscard]] std::uint32_t written() const noexcept;

private:
    std::uint32_t cached_;
    std::uint32_t prompt_;
    std::uint32_t capacity_;
    std::uint32_t max_tokens_;
    std::uint32_t prefilled_ = 0;    // prompt tokens planned past `cached`
    std::uint32_t decodes_ = 0;      // decode steps planned
    std::uint32_t drawn_ = 0;        // draws reported, whatever became of them
    std::uint32_t accepted_ = 0;     // draws accepted
    std::uint32_t emitted_ = 0;
    std::uint64_t run_ = 0;          // steps planned
    std::uint64_t reported_ = 0;     // steps reported
    bool ended_ = false;
    TurnEnd end_ = TurnEnd::Stop;
    TurnFailure failure_ = TurnFailure::None;
};

}  // namespace bllm::runtime
