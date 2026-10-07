// The runtime on the GPU (runtime/runtime.h): turns over generated models,
// their tokens against decoding a step at a time, the diff's reuse, and the
// cache after a step is discarded — the turn that follows gives the same bits
// as the same conversation run as one prompt in a fresh upload.

#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/cache/kv.h"
#include "core/runtime/runtime.h"
#include "support/acquire.h"
#include "support/running.h"

using namespace bllm;
using namespace bllm::testing;
using runtime::StartError;
using runtime::TurnEnd;
using runtime::TurnFailure;
using tokenizer::TokenId;

namespace {

// A model loaded with its runtime, the cache at `capacity` tokens, or the
// context the plan offers.
struct Session {
    Running running;
    std::unique_ptr<runtime::Runtime> runtime;   // after `running`, whose upload it borrows: destroyed first
};

std::unique_ptr<Session> open(WGPUInstance instance, const gpu::Device& device, const std::string& fixture,
                              std::vector<std::uint32_t> stops, std::optional<std::uint32_t> capacity = {}) {
    policy::LoadPolicy load;
    load.rollback_reserve = 16;   // a ring of 16 + 512 + 16 slots for the Gemma 3 model's window layer
    auto s = std::make_unique<Session>();
    s->running = run_model(instance, device, fixture, load);
    const residency::ResidencyPlan& plan = s->running.upload->plan();
    cache::KvCache cache{s->running.description, plan, load.cache_precision, capacity.value_or(plan.context_offered)};
    std::vector<TokenId> ids;
    for (const std::uint32_t t : stops) ids.push_back(static_cast<TokenId>(t));
    s->runtime = std::make_unique<runtime::Runtime>(std::move(s->running.program), std::move(cache),
                                                    runtime::StopSet{ids});
    return s;
}

struct Turned {
    std::vector<std::uint32_t> tokens;   // emitted
    std::optional<runtime::TurnResult> result;
    std::string message;
    std::optional<std::uint32_t> cancel_after;   // tokens emitted before the token callback cancels
    runtime::Runtime* runtime = nullptr;
};

void on_token(TokenId token, void* userdata) {
    Turned& t = *static_cast<Turned*>(userdata);
    t.tokens.push_back(static_cast<std::uint32_t>(token));
    if (t.cancel_after && t.tokens.size() == *t.cancel_after) t.runtime->cancel();
}

void on_turn(const runtime::TurnResult& result, void* userdata) {
    Turned& t = *static_cast<Turned*>(userdata);
    t.result = result;
    t.message = std::string(result.message);
    t.result->message = {};
}

void pump_turn(WGPUInstance instance, const Turned& t) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{120};
    while (!t.result) {
        wgpuInstanceProcessEvents(instance);
        REQUIRE_MESSAGE(std::chrono::steady_clock::now() < deadline, "timed out waiting for the turn");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

// Hot enough that the generated models draw varied tokens: past 500 tokens
// the Gemma 3 model's top logit leads the next by about 11, so top-p, taken
// at temperature 1, would keep it alone.
const policy::TurnPolicy kSampled{{10.0f, 64, 1.0f, 0.0f}, policy::Seed{0x5EED}, UINT32_MAX};

std::unique_ptr<Turned> turn(WGPUInstance instance, Session& s, const std::vector<std::uint32_t>& prompt,
                             std::uint32_t max_tokens, std::optional<std::uint32_t> cancel_after = {}) {
    auto t = std::make_unique<Turned>();
    t->cancel_after = cancel_after;
    t->runtime = s.runtime.get();
    std::vector<TokenId> ids;
    for (const std::uint32_t id : prompt) ids.push_back(static_cast<TokenId>(id));
    policy::TurnPolicy policy = kSampled;
    policy.max_tokens = max_tokens;
    const runtime::StartResult started = s.runtime->start(ids, policy, on_token, on_turn, t.get());
    REQUIRE_MESSAGE(started.error == StartError::Ok, started.subject);
    pump_turn(instance, *t);
    REQUIRE_MESSAGE(t->result->failure == TurnFailure::None, t->message);
    return t;
}

std::vector<std::uint32_t> candidates(WGPUInstance instance, const gpu::Device& device, const Session& s) {
    const residency::BufferRange& c = s.running.candidates;
    const auto words = read_floats(instance, device, s.running.upload->buffer(c.buffer), c.offset, 128);
    std::vector<std::uint32_t> out;
    for (const float f : words) out.push_back(std::bit_cast<std::uint32_t>(f));
    return out;
}

std::vector<std::uint32_t> pseudo_random(std::size_t length, std::uint32_t seed) {
    std::vector<std::uint32_t> ids;
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < length; ++i) {
        state = state * 1664525 + 1013904223;
        ids.push_back((state >> 8) % 256);
    }
    return ids;
}

std::vector<std::uint32_t> joined(std::vector<std::uint32_t> a, const std::vector<std::uint32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// A token of the vocabulary that `draws` does not hold: a stop that never
// ends the turn.
std::uint32_t absent(const std::vector<std::uint32_t>& draws) {
    for (std::uint32_t t = 0;; ++t) {
        if (std::find(draws.begin(), draws.end(), t) == draws.end()) return t;
    }
}

// The draws a turn over `prompt` makes, the first `count`, in a session of
// their own: the turn's draws do not depend on its stop set, which only ends
// it, so a stop that is not drawn shows them all.
std::vector<std::uint32_t> draws(WGPUInstance instance, const gpu::Device& device, const std::string& fixture,
                                 const std::vector<std::uint32_t>& prompt, std::uint32_t count) {
    std::vector<std::uint32_t> stops{0};
    for (;;) {
        auto s = open(instance, device, fixture, stops);
        const auto t = turn(instance, *s, prompt, count);
        if (t->result->end == TurnEnd::Limit) return t->tokens;
        stops = {absent(joined(t->tokens, {static_cast<std::uint32_t>(*t->result->stop)}))};
    }
}

// The turn that follows a discarded step against the same conversation run
// as one prompt in a fresh upload: its tokens, and the 64 candidates its last
// step selected, the same bits. It ends at its limit, so no step is
// discarded after it.
void check_follows_as_fresh(WGPUInstance instance, const gpu::Device& device, Session& used,
                            const std::string& fixture, const std::vector<std::uint32_t>& stops,
                            const std::vector<std::uint32_t>& prompt, std::optional<std::uint32_t> capacity,
                            std::uint32_t max_tokens, std::uint32_t reused) {
    const auto next = turn(instance, used, prompt, max_tokens);
    CHECK(next->result->reused == reused);
    const auto after = candidates(instance, device, used);
    auto fresh = open(instance, device, fixture, stops, capacity);
    const auto alone = turn(instance, *fresh, prompt, max_tokens);
    CHECK(alone->result->reused == 0);
    CHECK(next->result->end == alone->result->end);
    CHECK(next->tokens == alone->tokens);
    CHECK(after == candidates(instance, device, *fresh));
}

}  // namespace

TEST_CASE("a turn draws the tokens decoding a step at a time does, and ends at its limit") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto prompt = pseudo_random(20, 1);
    // Step at a time: each step given the last's token by identifier.
    Running r = run_model(instance.get(), *device);
    std::vector<std::uint32_t> want;
    {
        kernels::Step step{};
        sampler::apply(kSampled.sampling, kSampled.seed, step);
        step.position = 0;
        step.tokens = 20;
        step.logits = 1;
        std::copy(prompt.begin(), prompt.end(), step.ids.begin());
        for (std::uint32_t i = 0; i < 30; ++i) {
            const StepOutcome ran = try_step(instance.get(), *r.program, step);
            REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
            sampler::SampledRecord record;
            std::memcpy(&record, ran.readback.data(), sizeof record);
            want.push_back(record.token);
            step.position = 20 + i;
            step.tokens = 1;
            step.ids[0] = record.token;
        }
    }
    auto s = open(instance.get(), *device, "forward_model", {absent(want)});
    const auto t = turn(instance.get(), *s, prompt, 30);
    CHECK(t->result->end == TurnEnd::Limit);
    CHECK(t->result->prompt == 20);
    CHECK(t->result->emitted == 30);
    CHECK(t->tokens == want);
}

TEST_CASE("a second turn reuses the common prefix, and a prompt wholly held recomputes its last token") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto first = pseudo_random(40, 2);
    const auto reply = draws(instance.get(), *device, "forward_model", first, 5);
    auto s = open(instance.get(), *device, "forward_model", {absent(reply)});
    const auto t = turn(instance.get(), *s, first, 5);
    CHECK(t->tokens == reply);
    // The cache holds the prompt and every draw but the last: 44.
    std::vector<std::uint32_t> diverging(first.begin(), first.begin() + 30);
    diverging.push_back((first[30] + 1) % 256);
    check_follows_as_fresh(instance.get(), *device, *s, "forward_model", {absent(reply)}, diverging, {}, 3, 30);
    // The whole of what is held, as the prompt: all but its last token reused.
    const auto held = pseudo_random(25, 3);
    auto again = open(instance.get(), *device, "forward_model", {absent(reply)});
    (void)turn(instance.get(), *again, held, 1);
    check_follows_as_fresh(instance.get(), *device, *again, "forward_model", {absent(reply)}, held, {}, 4, 24);
}

TEST_CASE("the cache after a discarded step serves the next turn as a fresh upload would") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    // Qwen3's shapes, every layer full attention; then Gemma 3's, its window
    // layer's ring of 544 slots wrapped by a 600-token prompt.
    for (const auto& [fixture, length] : {std::pair{std::string{"forward_model"}, std::size_t{40}},
                                          std::pair{std::string{"forward_gemma3"}, std::size_t{600}}}) {
        CAPTURE(fixture);
        const auto prompt = pseudo_random(length, 4);
        const auto n = static_cast<std::uint32_t>(length);
        const auto d = draws(instance.get(), *device, fixture, prompt, 24);
        // The first draw past draw 0 not drawn before it.
        std::uint32_t j = 1;
        while (j < d.size() && std::find(d.begin(), d.begin() + j, d[j]) != d.begin() + j) ++j;
        REQUIRE(j < d.size());
        const auto continuation = pseudo_random(7, 5);

        {
            INFO("a stop drawn by the first decode step, the queued step behind it discarded");
            auto s = open(instance.get(), *device, fixture, {d[0]});
            const auto t = turn(instance.get(), *s, prompt, 50);
            CHECK(t->result->end == TurnEnd::Stop);
            CHECK(t->tokens.empty());
            // The stop token's entry, written by the step queued behind it, is
            // kept: the next turn reuses it.
            check_follows_as_fresh(instance.get(), *device, *s, fixture, {d[0]},
                                   joined(joined(prompt, {d[0]}), continuation), {}, 4, n + 1);
        }
        {
            INFO("a stop drawn one position before the context, with no step queued behind it");
            // Draw j a stop, and a context of n + j,
            // so decode step j − 1 runs at its last position and none is
            // queued behind draw j.
            auto s = open(instance.get(), *device, fixture, {d[j]}, n + j);
            const auto t = turn(instance.get(), *s, prompt, 50);
            CHECK(t->result->end == TurnEnd::Stop);
            CHECK(t->tokens == std::vector<std::uint32_t>(d.begin(), d.begin() + j));
            // The whole of what is held, the prompt and draws 0 to j − 1,
            // fills the context.
            check_follows_as_fresh(instance.get(), *device, *s, fixture, {d[j]},
                                   joined(prompt, std::vector<std::uint32_t>(d.begin(), d.begin() + j)), n + j, 1,
                                   n + j - 1);
        }
        {
            INFO("a cancel while a step is queued behind the one running");
            const std::uint32_t stop = absent(d);
            auto s = open(instance.get(), *device, fixture, {stop});
            const auto t = turn(instance.get(), *s, prompt, 50, 3);
            CHECK(t->result->end == TurnEnd::Cancelled);
            CHECK(t->tokens == std::vector<std::uint32_t>(d.begin(), d.begin() + 3));
            // Draws 0 to 2 held; the position the queued step wrote, for the
            // draw never accepted, truncated.
            check_follows_as_fresh(instance.get(), *device, *s, fixture, {stop},
                                   joined(joined(prompt, {d[0], d[1], d[2]}), continuation), {}, 4, n + 3);
        }
        {
            INFO("the next prompt the whole accepted history, the stop token's position written again");
            auto s = open(instance.get(), *device, fixture, {d[j]});
            const auto t = turn(instance.get(), *s, prompt, 50);
            CHECK(t->result->end == TurnEnd::Stop);
            const std::vector<std::uint32_t> accepted(d.begin(), d.begin() + j + 1);
            check_follows_as_fresh(instance.get(), *device, *s, fixture, {d[j]}, joined(prompt, accepted), {}, 4,
                                   n + static_cast<std::uint32_t>(j));
        }
    }
}

TEST_CASE("a cancel in prefill keeps the blocks already run") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto prompt = pseudo_random(1300, 6);
    auto s = open(instance.get(), *device, "forward_gemma3", {0});
    Turned t;
    std::vector<TokenId> ids;
    for (const std::uint32_t id : prompt) ids.push_back(static_cast<TokenId>(id));
    REQUIRE(s->runtime->start(ids, kSampled, on_token, on_turn, &t).error == StartError::Ok);
    s->runtime->cancel();   // two blocks already run
    pump_turn(instance.get(), t);
    CHECK(t.result->end == TurnEnd::Cancelled);
    CHECK(t.tokens.empty());
    check_follows_as_fresh(instance.get(), *device, *s, "forward_gemma3", {0}, prompt, {}, 3, 1024);
}

TEST_CASE("destroying a runtime mid-turn finishes the turn Cancelled") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    auto s = open(instance.get(), *device, "forward_model", {0});
    Turned t;
    std::vector<TokenId> ids(30, TokenId{7});
    REQUIRE(s->runtime->start(ids, kSampled, on_token, on_turn, &t).error == StartError::Ok);
    s->runtime.reset();
    pump_turn(instance.get(), t);
    CHECK(t.result->failure == TurnFailure::None);
    CHECK(t.result->end == TurnEnd::Cancelled);
}

TEST_CASE("a turn is refused, by name, when it cannot run") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    auto s = open(instance.get(), *device, "forward_model", {0}, 64);
    Turned t;
    const auto start = [&](std::size_t length, policy::TurnPolicy policy) {
        const std::vector<TokenId> ids(length, TokenId{7});
        return s->runtime->start(ids, policy, on_token, on_turn, &t);
    };
    CHECK(start(0, kSampled).error == StartError::EmptyPrompt);
    const auto long_prompt = start(65, kSampled);
    CHECK(long_prompt.error == StartError::PromptTooLong);
    CHECK(long_prompt.subject.find("65 tokens") != std::string::npos);
    CHECK(long_prompt.subject.find("64") != std::string::npos);
    policy::TurnPolicy bad = kSampled;
    bad.sampling.top_k = 0;
    CHECK(start(4, bad).error == StartError::Settings);
    bad = kSampled;
    bad.max_tokens = 0;
    CHECK(start(4, bad).error == StartError::Settings);
    CHECK(!t.result);   // none of them called back
    REQUIRE(start(64, kSampled).error == StartError::Ok);
    CHECK(start(4, kSampled).error == StartError::Busy);
    pump_turn(instance.get(), t);
    // A prompt filling the context: one draw, emitted or a stop.
    CHECK(t.result->end == (t.result->stop ? TurnEnd::Stop : TurnEnd::Context));
    CHECK(t.result->emitted + (t.result->stop ? 1u : 0u) == 1);
}
