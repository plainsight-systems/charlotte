#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Special tokens found in raw text. The chat template writes them as text, and
// each encodes as its one identifier, never as the characters that spell it.
// Every control and user-defined token is matched as the text stands, before
// any normalization or splitting: no reference marks one to be normalized,
// stripped or matched as a whole word. (SentencePiece matches after spaces
// become "▁", the one change it makes, with its tokens spelled to match:
// sentencepiece_bpe.h.) Text is searched from the left, and at
// each position the longest matching token wins — Hugging Face's
// leftmost-longest match over its added tokens. That matters: Gemma 3's
// special tokens include runs of spaces, each a prefix of the next.

// A run of ordinary text, or one special token, by its byte range.
struct Segment {
    std::uint32_t offset;
    std::uint32_t length;
    std::optional<TokenId> special;   // empty for ordinary text
};

class SpecialTokens {
public:
    // A token and the text that matches it.
    struct Entry {
        std::string text;
        TokenId id;
    };

    // None: every text is ordinary.
    SpecialTokens() = default;

    // Every control and user-defined token with text. It keeps its own copy
    // of their text, so it does not depend on the vocabulary living on.
    explicit SpecialTokens(const Vocabulary& vocabulary);

    // Exactly these, matched by the text each is given. Precondition: no text
    // is empty, and no two are the same.
    explicit SpecialTokens(std::vector<Entry> entries);

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    // The longest text a special token matches, 0 with none: what one
    // special token can take of a text's bytes (tokenizer.h).
    [[nodiscard]] std::size_t longest() const noexcept { return longest_; }

    // Splits `text` into ordinary text and special tokens, in order, covering
    // it exactly. Ordinary runs are never empty. Stops, false, once it has
    // found more than `max_specials` special tokens, `out` holding what it
    // found up to them, so a text of special tokens grows `out` by at most
    // two segments a token allowed (tokenizer.h, encoding is bounded).
    // Precondition: `text` is no longer than a Segment can address (4 GiB).
    bool segment(std::string_view text, std::vector<Segment>& out,
                 std::uint64_t max_specials = UINT64_MAX) const;

private:
    // Optimization (practice): a byte trie of the tokens' texts, laid out
    // flat. A position whose byte starts no token costs one read of first_;
    // from there each byte follows one edge, so a match costs its length, and
    // the deepest token the walk passes is the longest match. Matching the
    // tokens one by one cost Gemma 3, whose 6,414 special tokens mostly start
    // "<", a scan of thousands at every "<" in the text: 14.0 ms over the
    // bench corpus, now 0.8. Hugging Face matches with an Aho-Corasick
    // automaton, linear in the text; a walk from each position finds the same
    // matches, its depth bounded by the longest token.
    static constexpr std::uint32_t kNone = ~std::uint32_t{0};

    std::array<std::uint32_t, 256> first_{};   // the node each first byte leads to, or kNone
    // Node n's edges are [edge_start_[n], edge_start_[n + 1]), ordered by byte.
    std::vector<std::uint32_t> edge_start_;
    std::vector<unsigned char> edge_byte_;
    std::vector<std::uint32_t> edge_child_;
    std::vector<std::uint32_t> token_;   // the token node n spells in full, or kNone
    std::size_t size_ = 0;
    std::size_t longest_ = 0;
};

}  // namespace bllm::tokenizer
