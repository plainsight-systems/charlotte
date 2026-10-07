#include "core/runtime/runtime.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "core/cache/prefix.h"
#include "core/sampler/sampler.h"

namespace bllm::runtime {

// The runtime's state, shared with the program's in-flight callbacks.
struct Runtime::State {
    State(std::unique_ptr<kernels::Program> p, cache::KvCache c, StopSet s)
        : program(std::move(p)), cache(std::move(c)), stops(s) {
        // The prompt and the reply's accepted draws: at most the context
        // offered, and the last draw, which no step fed.
        held.reserve(std::size_t{cache.capacity()} + 1);
        message.reserve(1024);
    }

    std::unique_ptr<kernels::Program> program;   // null once the runtime is destroyed
    cache::KvCache cache;
    StopSet stops;
    // The tokens whose keys and values the cache holds, between turns; during
    // one, also the prompt's uncached tokens and the accepted draws.
    std::vector<tokenizer::TokenId> held;
    kernels::Step step{};
    bool lost = false;   // a step reported the device lost

    // The running turn.
    std::optional<Turn> turn;
    TokenCallback on_token = nullptr;
    TurnCallback on_turn = nullptr;
    void* userdata = nullptr;
    std::uint32_t prompt = 0;
    std::uint32_t reused = 0;
    std::optional<tokenizer::TokenId> stop;
    std::string message;   // the first failed step's, reserved so a failure allocates nothing
#if BLLM_DIAGNOSTICS_ENABLED
    // The step observer, and the steps run and not yet reported, at most two,
    // in run order: what the observer is told of each.
    StepObserver observer = nullptr;
    void* observer_data = nullptr;
    std::array<StepTimes, 2> outstanding{};
    std::uint32_t outstanding_count = 0;
#endif
    // This state, held while a turn runs, so its callbacks outlive the runtime.
    std::shared_ptr<State> in_flight;
};

namespace {

void on_step(kernels::ProgramError error, std::string_view message, std::span<const std::byte> bytes,
             void* userdata);
#if BLLM_DIAGNOSTICS_ENABLED
void on_profiled_step(kernels::ProgramError error, std::string_view message, std::span<const std::byte> bytes,
                      const kernels::Program::Timestamps& times, void* userdata);
#endif

// Runs every step the turn plans now. The turn plans at most two
// outstanding, so the program never refuses one at once; and every position
// is below the context offered, within kMaxPositions.
void run_planned(Runtime::State& s) {
    while (const std::optional<Planned> p = s.turn->next()) {
        s.step.position = p->position;
        s.step.tokens = p->tokens;
        s.step.logits = p->logits ? 1 : 0;
        s.step.fed = p->fed ? 1 : 0;
        if (!p->fed) {
            const auto first = s.held.begin() + p->first;
            std::transform(first, first + p->tokens, s.step.ids.begin(),
                           [](tokenizer::TokenId t) { return static_cast<std::uint32_t>(t); });
        }
        s.cache.advance(p->tokens);
#if BLLM_DIAGNOSTICS_ENABLED
        if (s.observer != nullptr) {
            s.outstanding[s.outstanding_count++] = {p->tokens > 1 || !p->fed, p->position, p->tokens, 0, 0};
            s.program->run_profiled(s.step, s.program->launch_count(), on_profiled_step, &s);
            continue;
        }
#endif
        s.program->run(s.step, on_step, &s);
    }
}

// The turn has finished: the cache and the held tokens kept to what it keeps,
// then its callback.
void finish(Runtime::State& s) {
    const std::shared_ptr<Runtime::State> keep = std::move(s.in_flight);
    Turn& turn = *s.turn;
    const std::uint32_t kept = s.program ? s.cache.truncate(turn.kept()) : 0;
    s.held.resize(kept);
    const TurnResult result{turn.failure(), turn.end(), s.message, s.prompt, s.reused, turn.emitted(), s.stop};
    const TurnCallback on_turn = s.on_turn;
    void* const userdata = s.userdata;
    s.turn.reset();   // so the callback may start the next turn
    on_turn(result, userdata);
    s.message.clear();
}

StepFailure failure_of(kernels::ProgramError error) {
    switch (error) {
        case kernels::ProgramError::DeviceLost: return StepFailure::DeviceLost;
        case kernels::ProgramError::Cancelled: return StepFailure::Cancelled;
        default: return StepFailure::Step;
    }
}

// A step's report, first half: the turn decides its draw, and the steps it
// plans next go to the GPU before the CPU's work on the token (GPU.7).
// Returns the token to emit, if any.
std::optional<tokenizer::TokenId> decide(Runtime::State& s, kernels::ProgramError error, std::string_view message,
                                         std::span<const std::byte> bytes) {
    Turn& turn = *s.turn;
    std::optional<tokenizer::TokenId> emitted;
    if (error == kernels::ProgramError::Ok) {
        std::optional<sampler::SampledRecord> record;
        if (!bytes.empty()) {
            record.emplace();
            std::memcpy(&*record, bytes.data(), sizeof *record);
        }
        const auto token = static_cast<tokenizer::TokenId>(record ? record->token : 0);
        switch (turn.report(record, record && s.stops.contains(token))) {
            case Draw::Emit:
                s.held.push_back(token);
                emitted = token;
                break;
            case Draw::Stop:
                s.held.push_back(token);
                s.stop = token;
                break;
            case Draw::None:
            case Draw::Discard:
            case Draw::Failed: break;
        }
    } else {
        if (error == kernels::ProgramError::DeviceLost) s.lost = true;
        if (s.message.empty()) s.message.assign(message.substr(0, s.message.capacity()));
        turn.fail(failure_of(error));
    }
    if (s.program) run_planned(s);
    return emitted;
}

// Second half: the token passed on, and the turn finished if it has.
void deliver(Runtime::State& s, std::optional<tokenizer::TokenId> emitted) {
    if (emitted) s.on_token(*emitted, s.userdata);
    if (s.turn->finished()) finish(s);
}

void on_step(kernels::ProgramError error, std::string_view message, std::span<const std::byte> bytes,
             void* userdata) {
    Runtime::State& s = *static_cast<Runtime::State*>(userdata);
    const std::shared_ptr<Runtime::State> keep = s.in_flight;   // the token callback may destroy the runtime
    deliver(s, decide(s, error, message, bytes));
}

#if BLLM_DIAGNOSTICS_ENABLED
// A profiled step's report: as on_step, with the observer told of the step
// between the two halves — after the GPU has its next steps.
void on_profiled_step(kernels::ProgramError error, std::string_view message, std::span<const std::byte> bytes,
                      const kernels::Program::Timestamps& times, void* userdata) {
    Runtime::State& s = *static_cast<Runtime::State*>(userdata);
    const std::shared_ptr<Runtime::State> keep = s.in_flight;
    StepTimes step = s.outstanding[0];
    s.outstanding[0] = s.outstanding[1];
    --s.outstanding_count;
    step.begin_ns = times.begin_ns;
    step.end_ns = times.end_ns;
    const std::optional<tokenizer::TokenId> emitted = decide(s, error, message, bytes);
    if (error == kernels::ProgramError::Ok && s.observer != nullptr) s.observer(step, s.observer_data);
    deliver(s, emitted);
}
#endif

}  // namespace

Runtime::Runtime(std::unique_ptr<kernels::Program> program, cache::KvCache cache, StopSet stops)
    : state_(std::make_shared<State>(std::move(program), std::move(cache), stops)) {}

Runtime::~Runtime() {
    // A turn running finishes Cancelled: the program, destroyed, reports its
    // outstanding steps so, and this state lives until they have.
    if (state_->turn) state_->turn->cancel();
    state_->program.reset();
}

StartResult Runtime::start(std::span<const tokenizer::TokenId> prompt, const policy::TurnPolicy& policy,
                           TokenCallback on_token, TurnCallback on_turn, void* userdata) {
    State& s = *state_;
    if (s.turn) return {StartError::Busy, "a turn is running"};
    if (s.lost) return {StartError::DeviceLost, "the device was lost"};
    const auto length = static_cast<std::uint32_t>(std::min<std::size_t>(prompt.size(), UINT32_MAX));
    if (length == 0) return {StartError::EmptyPrompt, "the prompt is empty"};
    if (prompt.size() > s.cache.capacity()) {
        return {StartError::PromptTooLong,
                "the prompt is " + std::to_string(prompt.size()) + " tokens, and the context offered " +
                    std::to_string(s.cache.capacity()),
                prompt.size(), s.cache.capacity()};
    }
    if (const sampler::SettingsResult checked = sampler::check(policy.sampling); !checked.ok) {
        return {StartError::Settings, checked.subject};
    }
    if (policy.max_tokens == 0) return {StartError::Settings, "max_tokens is 0; a reply draws at least 1"};

    // The diff, less the last token when it is the whole prompt: the first
    // draw needs that token's logits, which the cache does not hold.
    const auto common = static_cast<std::uint32_t>(cache::longest_common_prefix(s.held, prompt));
    const std::uint32_t kept = s.cache.truncate(std::min(common, length - 1));
    s.held.resize(kept);
    s.held.insert(s.held.end(), prompt.begin() + kept, prompt.end());

    s.step = {};
    sampler::apply(policy.sampling, policy.seed, s.step);
    s.turn.emplace(kept, length, s.cache.capacity(), policy.max_tokens);
    s.on_token = on_token;
    s.on_turn = on_turn;
    s.userdata = userdata;
    s.prompt = length;
    s.reused = kept;
    s.stop.reset();
    s.in_flight = state_;
    run_planned(s);
    return {};
}

std::uint32_t Runtime::capacity() const noexcept { return state_->cache.capacity(); }

#if BLLM_DIAGNOSTICS_ENABLED
bool Runtime::observe_steps(StepObserver observer, void* userdata) noexcept {
    State& s = *state_;
    // Refused while a turn runs, and for a program that cannot profile: its
    // every step would be refused, at once, inside the turn's own loop.
    if (s.turn || (observer != nullptr && (s.program == nullptr || !s.program->can_profile()))) return false;
    s.observer = observer;
    s.observer_data = userdata;
    return true;
}
#endif

void Runtime::cancel() noexcept {
    if (state_->turn) state_->turn->cancel();
}

}  // namespace bllm::runtime
