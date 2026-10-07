#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// The tokens that end a turn when drawn. Made once at load, read once a draw.
//
//   - From the file: tokenizer.ggml.eos_token_id, and eot_token_id and
//     eom_token_id where present — the keys llama.cpp reads for its
//     end-of-generation set. Each of the listed models' files names its chat
//     turn's end as eos: Qwen3 <|im_end|>, Llama 3.2 <|eot_id|>, Gemma 3
//     <end_of_turn>.
//   - From the model's load policy (policy/policy.h): further stop tokens, by
//     their text, where the model's own generation config lists more than the
//     file carries — Qwen3's <|endoftext|>, Llama 3.2's <|eom_id|> and
//     <|end_of_text|>, Gemma 3's <eos>. They are measured per model, as the
//     sampling settings are (principle 8); an unmeasured model stops on its
//     file's alone. A text, not an identifier, since the page never sees an
//     identifier (cache/prefix.h).
//   - Refused by name, never dropped: a file's key that is not an unsigned
//     32-bit integer; a file's identifier at or past the vocabulary; a
//     policy text that is not exactly one token of the vocabulary, or names
//     one that is not a control token, since a stop on an ordinary word would
//     end replies at that word; more than kMaxStops distinct tokens; and
//     none at all, since a turn would then end only at its limit or the
//     context. A token named twice is held once.
//   - Held in a fixed array, searched in order: at most 8 comparisons a draw,
//     the listed models' sets holding 2 or 3.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant — distinct tokens, at
//            most kMaxStops.
//     E.27   Use error codes systematically — StopsError, with the subject
//            named.
//   C++ performance guidelines
//     EMB.2  Prefer fixed-capacity containers — the set, inline, its
//            capacity stated.
//     MEM.9  Allocate at init — made at load; a draw's test allocates
//            nothing.

inline constexpr std::size_t kMaxStops = 8;

class StopSet {
public:
    StopSet() = default;   // empty: what resolve_stops fills
    // Precondition: the tokens are distinct and at most kMaxStops.
    explicit StopSet(std::span<const tokenizer::TokenId> tokens) noexcept;

    [[nodiscard]] bool contains(tokenizer::TokenId token) const noexcept;
    [[nodiscard]] std::span<const tokenizer::TokenId> tokens() const noexcept;

private:
    std::array<tokenizer::TokenId, kMaxStops> tokens_{};
    std::size_t size_ = 0;
};

enum class StopsError {
    Ok,
    WrongType,         // a file's key present, but not an unsigned 32-bit integer
    OutOfVocabulary,   // a file's identifier at or past the vocabulary
    NotAToken,         // a policy text that is not exactly one token
    NotControl,        // a policy text naming a token that is not a control token
    TooMany,           // more than kMaxStops distinct tokens
    None,              // neither the file nor the policy names one
};

struct StopsResult {
    StopsError error = StopsError::Ok;
    std::string subject;   // the key or text refused, and why
};

// The stop set of `index`'s file and `policy_stops`, into `out`; `out` is
// left untouched on failure. Precondition: `vocabulary` was loaded from
// `index`'s file.
[[nodiscard]] StopsResult resolve_stops(const gguf::TensorIndex& index, const tokenizer::Vocabulary& vocabulary,
                                        std::span<const std::string> policy_stops, StopSet& out);

}  // namespace bllm::runtime
