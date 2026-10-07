#include "core/runtime/stops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>


namespace bllm::runtime {

StopSet::StopSet(std::span<const tokenizer::TokenId> tokens) noexcept : size_(tokens.size()) {
    std::copy(tokens.begin(), tokens.end(), tokens_.begin());
}

bool StopSet::contains(tokenizer::TokenId token) const noexcept {
    return std::find(tokens_.begin(), tokens_.begin() + size_, token) != tokens_.begin() + size_;
}

std::span<const tokenizer::TokenId> StopSet::tokens() const noexcept { return {tokens_.data(), size_}; }

namespace {

// The keys llama.cpp reads for its end-of-generation set.
constexpr std::array<std::string_view, 3> kFileKeys{
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
};

}  // namespace

StopsResult resolve_stops(const gguf::TensorIndex& index, const tokenizer::Vocabulary& vocabulary,
                          std::span<const std::string> policy_stops, StopSet& out) {
    // Load time, not the per-token path: a vector, discarded once the set is
    // made.
    std::vector<tokenizer::TokenId> found;
    const auto add = [&](tokenizer::TokenId id) {
        if (std::find(found.begin(), found.end(), id) == found.end()) found.push_back(id);
    };
    for (const std::string_view key : kFileKeys) {
        std::uint32_t id = 0;
        switch (index.read_u32(key, id)) {
            case gguf::MetadataError::Ok: break;
            case gguf::MetadataError::MissingKey: continue;
            case gguf::MetadataError::WrongType:
                return {StopsError::WrongType, std::string(key) + " is not an unsigned 32-bit integer"};
        }
        if (id >= vocabulary.size()) {
            return {StopsError::OutOfVocabulary, std::string(key) + " is " + std::to_string(id) +
                                                     ", past the vocabulary of " + std::to_string(vocabulary.size())};
        }
        add(static_cast<tokenizer::TokenId>(id));
    }
    for (const std::string& text : policy_stops) {
        const std::optional<tokenizer::TokenId> id = vocabulary.find(text);
        if (!id) return {StopsError::NotAToken, "stop \"" + text + "\" is not a token of this vocabulary"};
        if (vocabulary.type(*id) != tokenizer::TokenType::Control) {
            return {StopsError::NotControl, "stop \"" + text + "\" is not a control token"};
        }
        add(*id);
    }
    if (found.empty()) {
        return {StopsError::None, "the file names no end-of-generation token, and the policy lists no stop"};
    }
    if (found.size() > kMaxStops) {
        return {StopsError::TooMany, std::to_string(found.size()) + " stop tokens, more than the " +
                                         std::to_string(kMaxStops) + " held"};
    }
    out = StopSet{found};
    return {};
}

}  // namespace bllm::runtime
