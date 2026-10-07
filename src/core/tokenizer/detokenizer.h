#pragma once

#include <string>
#include <string_view>

#include "core/tokenizer/tokenizer.h"

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// A reply's tokens as text: each token decoded (tokenizer.h) and pushed
// through a UTF-8 stream, giving the whole characters it completes — none
// for a token whose bytes only begin a character another token finishes —
// and, at the end, U+FFFD for bytes still pending, rather than dropping them
// unseen. Pure: the generator (runtime/generator.h) passes what it gives to
// the page, and it is tested on chosen tokens, without a model drawing them.
//
// What it costs: a token's bytes appended to a buffer reserved, at
// construction, for the vocabulary's longest token, and the characters they
// complete to another, reserved for the worst — each of the token's bytes
// written as U+FFFD's three, and the up to three held before them as their
// own three or as one U+FFFD in their place; nothing is allocated a token.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — no GPU, no callback; a token in, its
//            text out.
//   C++ performance guidelines
//     MEM.9  Allocate at init — both buffers, at construction.
class Detokenizer {
public:
    // Borrows `tokenizer`, which outlives it (I.11).
    explicit Detokenizer(const Tokenizer& tokenizer);

    // The characters `token` completes, possibly none; valid until the next
    // call. Precondition: the token is below the vocabulary.
    [[nodiscard]] std::string_view push(TokenId token);

    // Ends the reply: U+FFFD where bytes were pending, else nothing; valid
    // until the next call. The stream is then empty, ready for another.
    [[nodiscard]] std::string_view finish();

    // The vocabulary's longest token, in bytes.
    [[nodiscard]] std::size_t longest() const noexcept { return longest_; }

private:
    const Tokenizer* tokenizer_;
    std::size_t longest_ = 1;
    Utf8Stream stream_;
    std::string bytes_;
    std::string text_;
};

}  // namespace bllm::tokenizer
