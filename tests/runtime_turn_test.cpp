// The turn (runtime/turn.h): the steps it plans and what each way of ending
// emits and keeps, driven by a script of draws in place of the GPU, in the
// order the runtime drives it — a report, then the steps it frees run, then
// the token's work, where a cancel may come.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <vector>

#include "core/runtime/turn.h"

using namespace bllm;
using runtime::Draw;
using runtime::Planned;
using runtime::Turn;
using runtime::TurnEnd;
using runtime::TurnFailure;

namespace {

constexpr std::uint32_t kNever = std::numeric_limits<std::uint32_t>::max();

// Draw i is token 100 + i; the script says which draw is a stop or not
// finite, and after which draw or step a cancel comes.
struct Script {
    std::uint32_t stop_at = kNever;
    std::uint32_t non_finite_at = kNever;
    std::uint32_t cancel_after_draw = kNever;
    std::uint32_t cancel_after_step = kNever;
};

struct Outcome {
    std::vector<Planned> steps;
    std::vector<Draw> draws;
    std::size_t most_outstanding = 0;
};

Outcome drive(Turn& turn, const Script& script) {
    Outcome out;
    std::deque<Planned> outstanding;
    const auto plan = [&] {
        while (const std::optional<Planned> p = turn.next()) {
            out.steps.push_back(*p);
            outstanding.push_back(*p);
            out.most_outstanding = std::max(out.most_outstanding, outstanding.size());
        }
    };
    plan();
    std::uint32_t draws = 0, steps = 0;
    while (!outstanding.empty()) {
        const Planned p = outstanding.front();
        outstanding.pop_front();
        ++steps;
        std::optional<std::uint32_t> drawn;
        if (p.logits) {
            const std::uint32_t i = draws++;
            const sampler::SampledRecord record{100 + i, i == script.non_finite_at ? 1u : 0u, 0.5f, 0};
            out.draws.push_back(turn.report(record, i == script.stop_at));
            drawn = i;
        } else {
            CHECK(turn.report(std::nullopt, false) == Draw::None);
        }
        plan();
        if (drawn == script.cancel_after_draw || steps == script.cancel_after_step) turn.cancel();
    }
    CHECK(turn.finished());
    return out;
}

std::uint32_t count(const std::vector<Draw>& draws, Draw d) {
    return static_cast<std::uint32_t>(std::count(draws.begin(), draws.end(), d));
}

}  // namespace

TEST_CASE("a turn prefills its uncached tokens in blocks, then decodes fed, two steps outstanding") {
    Turn turn{0, 1300, 4096, 3};
    const Outcome o = drive(turn, {});
    REQUIRE(o.steps.size() == 5);
    // Three prefill blocks, the last alone asking for logits.
    CHECK(o.steps[0].position == 0);
    CHECK(o.steps[0].tokens == 512);
    CHECK(!o.steps[0].logits);
    CHECK(o.steps[1].position == 512);
    CHECK(o.steps[1].first == 512);
    CHECK(!o.steps[1].logits);
    CHECK(o.steps[2].position == 1024);
    CHECK(o.steps[2].tokens == 276);
    CHECK(o.steps[2].logits);
    CHECK(!o.steps[2].fed);
    // Decode steps 0 and 1 feed draws 0 and 1; draw 2, the limit's, is fed by
    // none.
    for (std::uint32_t j = 0; j < 2; ++j) {
        CHECK(o.steps[3 + j].position == 1300 + j);
        CHECK(o.steps[3 + j].tokens == 1);
        CHECK(o.steps[3 + j].logits);
        CHECK(o.steps[3 + j].fed);
    }
    CHECK(o.most_outstanding == 2);
    CHECK(o.draws == std::vector<Draw>{Draw::Emit, Draw::Emit, Draw::Emit});
    CHECK(turn.failure() == TurnFailure::None);
    CHECK(turn.end() == TurnEnd::Limit);
    CHECK(turn.emitted() == 3);
    CHECK(turn.written() == 1302);
    CHECK(turn.kept() == 1302);   // every draw's entry but the last's
}

TEST_CASE("a turn prefills only what the cache does not hold") {
    Turn turn{700, 1300, 4096, 1};
    const Outcome o = drive(turn, {});
    REQUIRE(o.steps.size() == 2);
    CHECK(o.steps[0].position == 700);
    CHECK(o.steps[0].tokens == 512);
    CHECK(o.steps[1].position == 1212);
    CHECK(o.steps[1].tokens == 88);
    CHECK(o.steps[1].logits);
    // A limit of 1: draw 0 alone, and no decode step.
    CHECK(turn.end() == TurnEnd::Limit);
    CHECK(turn.emitted() == 1);
    CHECK(turn.kept() == 1300);
}

TEST_CASE("a stop at draw 0 keeps the stop token the queued step fed, and discards that step's draw") {
    Turn turn{4, 10, 4096, 100};
    const Outcome o = drive(turn, {.stop_at = 0});
    REQUIRE(o.steps.size() == 2);   // the prefill, and decode step 0 queued behind it
    CHECK(o.steps[1].fed);
    CHECK(o.draws == std::vector<Draw>{Draw::Stop, Draw::Discard});
    CHECK(turn.end() == TurnEnd::Stop);
    CHECK(turn.emitted() == 0);
    CHECK(turn.written() == 11);
    CHECK(turn.kept() == 11);   // the prompt and the stop token
}

TEST_CASE("a stop at a later draw keeps every accepted token's entry, the stop token's among them") {
    Turn turn{0, 10, 4096, 100};
    const Outcome o = drive(turn, {.stop_at = 3});
    CHECK(o.draws == std::vector<Draw>{Draw::Emit, Draw::Emit, Draw::Emit, Draw::Stop, Draw::Discard});
    CHECK(o.steps.size() == 5);   // the prefill, and decode steps 0 to 3
    CHECK(turn.emitted() == 3);
    CHECK(turn.kept() == 14);
}

TEST_CASE("no step is queued at the context: a stop one position before it has none behind it") {
    // A prompt a token short of the context: decode step 0 at position 9,
    // and none at 10.
    for (const std::uint32_t stop : {1u, kNever}) {
        Turn turn{0, 9, 10, 100};
        const Outcome o = drive(turn, {.stop_at = stop});
        REQUIRE(o.steps.size() == 2);
        CHECK(o.steps[1].position == 9);
        CHECK(o.draws.size() == 2);   // nothing discarded: no step behind draw 1
        CHECK(turn.end() == (stop == 1 ? TurnEnd::Stop : TurnEnd::Context));
        CHECK(turn.emitted() == (stop == 1 ? 1u : 2u));
        CHECK(turn.written() == 10);
        CHECK(turn.kept() == 10);
    }
    // A prompt that fills the context: draw 0 alone.
    Turn full{0, 10, 10, 100};
    const Outcome o = drive(full, {});
    CHECK(o.steps.size() == 1);
    CHECK(full.end() == TurnEnd::Context);
    CHECK(full.emitted() == 1);
    CHECK(full.kept() == 10);
}

TEST_CASE("a cancel with a decode step queued truncates the position that step wrote") {
    Turn turn{0, 10, 4096, 100};
    // Draw 2 is emitted, and its report runs decode step 3 before the cancel:
    // decode step 2 (feeding draw 2) and 3 (feeding the unknown draw 3) are
    // outstanding.
    const Outcome o = drive(turn, {.cancel_after_draw = 2});
    CHECK(o.draws == std::vector<Draw>{Draw::Emit, Draw::Emit, Draw::Emit, Draw::Discard, Draw::Discard});
    CHECK(turn.end() == TurnEnd::Cancelled);
    CHECK(turn.emitted() == 3);
    CHECK(turn.written() == 14);
    CHECK(turn.kept() == 13);   // the prompt and draws 0 to 2
}

TEST_CASE("a cancel in prefill keeps the blocks already run") {
    Turn turn{0, 2000, 4096, 100};
    const Outcome o = drive(turn, {.cancel_after_step = 1});
    // Blocks 0 and 1 ran at once; block 2 ran when block 0 reported, before
    // the cancel; block 3 never.
    CHECK(o.steps.size() == 3);
    CHECK(o.draws.empty());
    CHECK(turn.end() == TurnEnd::Cancelled);
    CHECK(turn.kept() == 1536);
}

TEST_CASE("a non-finite draw ends the turn and truncates the token 0 fed in its place") {
    Turn turn{0, 10, 4096, 100};
    const Outcome o = drive(turn, {.non_finite_at = 2});
    CHECK(o.draws == std::vector<Draw>{Draw::Emit, Draw::Emit, Draw::Failed, Draw::Discard});
    CHECK(turn.failure() == TurnFailure::NonFinite);
    CHECK(turn.emitted() == 2);
    CHECK(turn.written() == 13);
    CHECK(turn.kept() == 12);
}

TEST_CASE("a cancel after the turn has ended changes nothing") {
    Turn turn{0, 10, 4096, 100};
    const Outcome o = drive(turn, {.stop_at = 1, .cancel_after_draw = 1});
    CHECK(o.draws == std::vector<Draw>{Draw::Emit, Draw::Stop, Draw::Discard});
    CHECK(turn.end() == TurnEnd::Stop);
    CHECK(turn.kept() == 12);
}

TEST_CASE("a step's failure ends the turn with it and keeps nothing; a cancellation ends it Cancelled") {
    for (const auto [failure, want] : {std::pair{runtime::StepFailure::Step, TurnFailure::Step},
                                       std::pair{runtime::StepFailure::DeviceLost, TurnFailure::DeviceLost}}) {
        Turn turn{0, 10, 4096, 100};
        REQUIRE(turn.next());   // the prefill
        REQUIRE(turn.next());   // decode step 0
        CHECK(!turn.next());    // two outstanding
        CHECK(turn.report(sampler::SampledRecord{100, 0, 0.5f, 0}, false) == Draw::Emit);
        REQUIRE(turn.next());   // decode step 1
        turn.fail(failure);     // decode step 0 failed
        CHECK(turn.ended());
        CHECK(!turn.finished());
        CHECK(!turn.next());
        CHECK(turn.report(sampler::SampledRecord{101, 0, 0.5f, 0}, false) == Draw::Discard);
        CHECK(turn.finished());
        CHECK(turn.failure() == want);
        CHECK(turn.kept() == 0);
    }
    // A failure after the turn ended still keeps nothing: what the failed
    // step wrote is unknown.
    Turn stopped{0, 10, 4096, 100};
    REQUIRE(stopped.next());
    REQUIRE(stopped.next());
    CHECK(stopped.report(sampler::SampledRecord{100, 0, 0.5f, 0}, true) == Draw::Stop);
    stopped.fail(runtime::StepFailure::Step);
    CHECK(stopped.finished());
    CHECK(stopped.failure() == TurnFailure::Step);
    CHECK(stopped.kept() == 0);
    // The program destroyed with steps in flight: Cancelled, not failed.
    Turn destroyed{0, 10, 4096, 100};
    REQUIRE(destroyed.next());
    REQUIRE(destroyed.next());
    destroyed.fail(runtime::StepFailure::Cancelled);
    destroyed.fail(runtime::StepFailure::Cancelled);
    CHECK(destroyed.finished());
    CHECK(destroyed.failure() == TurnFailure::None);
    CHECK(destroyed.end() == TurnEnd::Cancelled);
}
