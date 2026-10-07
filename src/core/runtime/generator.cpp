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

// NFC's greatest shrinking of UTF-8: a character's canonical decomposition is
// at most 3 times its own bytes — U+0390's 6 to 2, a Hangul syllable's 9 to
// 3 — in Unicode 16.0, the tables' version (tokenizer/nfc.h).
constexpr std::uint64_t kNfcShrink = 3;

void on_token(tokenizer::TokenId token, void* userdata) {
    Generator::State& s = *static_cast<Generator::State*>(userdata);
    if (const std::string_view text = s.text->push(token); !text.empty()) s.on_text(text, s.userdata);
}

void on_turn(const TurnResult& result, void* userdata) {
    Generator::State& s = *static_cast<Generator::State*>(userdata);
    // This turn's callbacks, taken before the generator is idle: the last
    // text callback may start the next turn, which replaces them.
    const TextCallback on_text = s.on_text;
    const EndCallback on_end = s.on_end;
    void* const caller = s.userdata;
    const std::string_view text = s.text->finish();
    const std::shared_ptr<Generator::State> keep = std::move(s.in_flight);
    // Bytes still pending, the turn stopped inside a character: U+FFFD.
    if (!text.empty()) on_text(text, caller);
    on_end(result, caller);
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
    // A token covers at most `longest` bytes of normalized text, and NFC
    // shrinks text at most 3-fold, so text longer than 3 × the context × that
    // makes more tokens than the context holds: refused unencoded.
    const std::uint64_t longest = s.text->longest();
    const std::uint64_t covered = kNfcShrink * longest;
    const std::uint64_t bound = std::uint64_t{s.capacity} * covered;
    if (text.size() > bound) {
        const std::uint64_t least = (text.size() + covered - 1) / covered;
        return {{},
                {StartError::PromptTooLong,
                 "the prompt is " + std::to_string(text.size()) + " bytes, at least " + std::to_string(least) +
                     " tokens, and the context offered " + std::to_string(s.capacity),
                 least, s.capacity}};
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
