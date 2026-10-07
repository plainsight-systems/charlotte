#include "core/runtime/generator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "core/tokenizer/detokenizer.h"

namespace bllm::runtime {

struct Generator::State {
    std::unique_ptr<tokenizer::Tokenizer> tokenizer;
    std::unique_ptr<Runtime> runtime;   // null once the generator is destroyed
    std::uint32_t capacity = 0;
    std::optional<tokenizer::Detokenizer> text;   // over `tokenizer`, made once it is set
    std::vector<tokenizer::TokenId> tokens;

    // The running turn.
    TextCallback on_text = nullptr;
    EndCallback on_end = nullptr;
    void* userdata = nullptr;
    // This state, held while a turn runs, so its callbacks outlive the generator.
    std::shared_ptr<State> in_flight;
};

namespace {

void on_token(tokenizer::TokenId token, void* userdata) {
    Generator::State& s = *static_cast<Generator::State*>(userdata);
    if (const std::string_view text = s.text->push(token); !text.empty()) s.on_text(text, s.userdata);
}

void on_turn(const TurnResult& result, void* userdata) {
    Generator::State& s = *static_cast<Generator::State*>(userdata);
    const std::shared_ptr<Generator::State> keep = std::move(s.in_flight);
    // Bytes still pending, the turn stopped inside a character: U+FFFD.
    if (const std::string_view text = s.text->finish(); !text.empty()) s.on_text(text, s.userdata);
    s.on_end(result, s.userdata);
}

}  // namespace

Generator::Generator(std::unique_ptr<tokenizer::Tokenizer> tokenizer, std::unique_ptr<Runtime> runtime)
    : state_(std::make_shared<State>()) {
    State& s = *state_;
    s.tokenizer = std::move(tokenizer);
    s.runtime = std::move(runtime);
    s.capacity = s.runtime->capacity();
    s.text.emplace(*s.tokenizer);
    s.tokens.reserve(std::size_t{s.capacity} + 1);
}

Generator::~Generator() {
    // A turn running finishes Cancelled: the runtime, destroyed, ends it, and
    // this state lives until its end callback has run.
    state_->runtime.reset();
}

GenerateResult Generator::start(std::string_view text, const policy::TurnPolicy& policy, TextCallback on_text,
                                EndCallback on_end, void* userdata) {
    State& s = *state_;
    if (s.runtime == nullptr || s.in_flight != nullptr) return {{}, {StartError::Busy, "a turn is running"}};
    // A token covers at most `longest` bytes, so text longer than the context
    // × that makes more tokens than the context holds: refused unencoded.
    const std::uint64_t longest = s.text->longest();
    const std::uint64_t bound = std::uint64_t{s.capacity} * longest;
    if (text.size() > bound) {
        const std::uint64_t least = (text.size() + longest - 1) / longest;
        return {{},
                {StartError::PromptTooLong, "the prompt is " + std::to_string(text.size()) + " bytes, at least " +
                                                std::to_string(least) + " tokens, and the context offered " +
                                                std::to_string(s.capacity)}};
    }
    s.tokens.clear();
    if (const tokenizer::EncodeError e = s.tokenizer->encode(text, s.tokens); e != tokenizer::EncodeError::Ok) {
        return {e, {}};
    }
    s.on_text = on_text;
    s.on_end = on_end;
    s.userdata = userdata;
    s.in_flight = state_;
    StartResult started = s.runtime->start(s.tokens, policy, runtime::on_token, runtime::on_turn, &s);
    if (started.error != StartError::Ok) s.in_flight.reset();
    return {{}, std::move(started)};
}

void Generator::cancel() noexcept {
    if (state_->runtime) state_->runtime->cancel();
}

}  // namespace bllm::runtime
