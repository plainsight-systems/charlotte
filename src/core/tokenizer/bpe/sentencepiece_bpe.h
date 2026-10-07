#pragma once

#include <array>
#include <memory>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// SentencePiece BPE, the algorithm GGUF names "llama", as Gemma 3 uses it.
//
// It is BPE. The file lists no merges, only a score for each token, and
// llama.cpp merges by score: any two adjacent symbols whose text joined is a
// token, highest score first, leftmost on a tie. Hugging Face lists merges
// instead, ranked. They are one algorithm: the merges a vocabulary implies —
// every split of a normal token into two normal tokens, ranked by the score
// of the token they make (sentencepiece_merges.h) — are exactly Hugging
// Face's merges for Gemma 3, 513,511 of them, ranked in the same order of the
// tokens they make. Its other 1,395 make runs of tabs, newlines and spaces,
// which are user-defined tokens, matched in the raw text before anything
// merges. So the merges are derived from the vocabulary once, at load, and
// run through the same merge step as byte-level BPE (merge.h). Encoding a
// rendered prompt:
//
//   1. Each space becomes "▁" (U+2581), as the vocabulary spells it. Nothing
//      else is normalized, and no space is added before the text.
//   2. Special tokens are found in that text and become their own identifiers
//      (special.h), each spelled the same way. So Gemma 3's user-defined runs
//      of spaces match runs of "▁", whether the text held spaces or the
//      character ▁ itself. That is SentencePiece's own rule: it matches its
//      user-defined pieces after spaces become ▁. Both references differ from
//      it where ▁ is typed beside a space, as a sparkline's lowest bar is:
//      llama.cpp spells those tokens in spaces and never matches a typed ▁,
//      and Hugging Face matches typed runs only where they stand alone.
//   3. Each character between them becomes its token, or, where the
//      vocabulary has none, one byte token (<0x00> to <0xFF>) for each of
//      its bytes.
//   4. Each stretch is merged as SentencePiece merges it, whole. It is cut
//      before a "▁" only where no merge could cross: see cut_before.
//
// BOS is never added: the chat template writes it as text.
//
// Decoding a token gives a normal token's text with each "▁" a space again, a
// byte token's byte, and any other token's text as written. A "▁" typed in the
// text comes back as a space, as it does in SentencePiece and both references:
// the vocabulary cannot tell the two apart.
class SentencePieceBpe {
public:
    SentencePieceBpe() = default;

    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocabulary_; }

    // Appends the tokens `raw` encodes to. `out` is left untouched if it
    // cannot: text that is not UTF-8, or longer than 4 GiB once spaces are ▁.
    [[nodiscard]] EncodeError encode(std::string_view raw, std::vector<TokenId>& out) const;

    // Appends the bytes `token` stands for. They can end partway through a
    // character; Utf8Stream makes text of them. Precondition: loaded, and the
    // token is below vocabulary().size().
    void decode(TokenId token, std::string& out) const;

private:
    friend LoadResult load_sentencepiece_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                             SentencePieceBpe& out);

    // A normal token holding "▁" after its first character, and where.
    struct Straddle {
        std::string text;
        std::vector<std::uint32_t> marks;   // the byte offset of each such ▁
    };

    // Whether a stretch may be cut before the "▁" at text[at].
    //
    // Optimization (practice): merging a stretch whole costs a heap over every
    // symbol in it, and a stretch runs from one special token to the next —
    // in Gemma 3, a whole line. A merge that crossed a cut would make a
    // normal token holding that ▁ after its first character, whose text would
    // stand in the text across the cut, so where none does, the pieces on
    // either side merge exactly as they would together: each is cut off and
    // merged alone, a word at a time, short enough for the rescan in merge.h:
    // Gemma 3 encodes the bench corpus in 16.7 ms, not 33.3, and finding such
    // tokens adds 3 ms to its load. It has one, ">▁</". A vocabulary with more
    // than kMaxStraddles is not cut at all.
    [[nodiscard]] bool cut_before(std::string_view text, std::size_t at) const noexcept;

    static constexpr std::size_t kMaxStraddles = 64;

    Vocabulary vocabulary_;
    SpecialTokens special_;
    MergeTable merges_;
    std::array<TokenId, 256> byte_tokens_{};   // <0x00> to <0xFF>
    std::vector<Straddle> straddles_;
    bool cuts_ = false;   // false when there are more than kMaxStraddles
};

// The algorithm, as the capability table lists it. It splits no text first,
// so it needs no pre-tokenizer.
extern const Algorithm kSentencePiece;

// Reads the vocabulary and scores, derives the merges, and finds the byte
// tokens, which must be exactly <0x00> to <0xFF>. Refuses a vocabulary with
// two special tokens spelled alike once their spaces are ▁, and one with a
// one-character token that is neither
// normal nor special, which the references would use and this would spell in
// bytes, and a file that asks for a space before the text
// (tokenizer.ggml.add_space_prefix true, or not declared, which llama.cpp
// reads as true): the references place that space differently, llama.cpp
// after every special token and Hugging Face only at the start, and no model
// listed here asks for it. `out` is left untouched unless it succeeds.
[[nodiscard]] LoadResult load_sentencepiece_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                                SentencePieceBpe& out);

// The algorithm as a Tokenizer (tokenizer.h). Gemma 3 encodes the bench
// corpus, 428 KB, in 16.7 ms: about 39 ns a byte, so a conversation of 32,768
// tokens, near 130 KB, encodes in about 5 ms a turn.
class SentencePieceTokenizer final : public Tokenizer {
public:
    [[nodiscard]] const Vocabulary& vocabulary() const noexcept override;
    [[nodiscard]] EncodeError encode(std::string_view text, std::vector<TokenId>& out) override;
    void decode(TokenId token, std::string& out) const override;

private:
    friend LoadResult load_sentencepiece_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                                   const PreTokenizer* pretokenizer,
                                                   std::unique_ptr<Tokenizer>& out);
    SentencePieceBpe bpe_;
};

// kSentencePiece's load (tokenizer.h's LoadFn); `pretokenizer` is unused,
// the algorithm splitting no text first.
[[nodiscard]] LoadResult load_sentencepiece_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                                      const PreTokenizer* pretokenizer,
                                                      std::unique_ptr<Tokenizer>& out);

}  // namespace bllm::tokenizer::bpe
