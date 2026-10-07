#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/bpe/piece_cache.h"
#include "core/tokenizer/pretokenize.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Byte-level BPE, the algorithm GGUF names "gpt2", as Qwen3 and Llama 3 use
// it. Encoding a rendered prompt:
//
//   1. Special tokens are found in the raw text and become their own
//      identifiers (special.h).
//   2. The text between them is normalized as the pre-tokenizer says (NFC
//      for qwen2), then split into pieces (pretokenize.h).
//   3. Each piece's bytes become one token each, by GPT-2's byte map, and are
//      merged (merge.h) — unless the pre-tokenizer takes a piece that is
//      itself a token whole.
//
// BOS is never added: the chat template writes it as text.
//
// Decoding a token turns each character of a normal token back into the byte
// it stands for; any other token is its text as written. No clean-up follows:
// llama.cpp removes the space before punctuation when decoding llama-bpe, but
// that changes text the user wrote, and it needs tokens not yet generated.
class ByteLevelBpe {
public:
    ByteLevelBpe() = default;

    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocabulary_; }

    // Appends the tokens `text` encodes to. `out` is left untouched if it
    // cannot: text that is not UTF-8, or longer than 4 GiB. Precondition:
    // loaded by load_byte_level_bpe; an empty one has no pre-tokenizer.
    [[nodiscard]] EncodeError encode(std::string_view text, std::vector<TokenId>& out) const;

    // The same, consulting `cache` for short pieces and recording what it
    // finds. The result is identical; a cache last used by another tokenizer
    // is emptied first.
    [[nodiscard]] EncodeError encode(std::string_view text, std::vector<TokenId>& out, PieceCache& cache) const;

    // Appends the bytes `token` stands for. They can end partway through a
    // character; Utf8Stream makes text of them. Precondition: loaded, and the
    // token is below vocabulary().size().
    void decode(TokenId token, std::string& out) const;

private:
    friend LoadResult load_byte_level_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                          const PreTokenizer& pretokenizer, ByteLevelBpe& out);

    [[nodiscard]] EncodeError encode_into(std::string_view text, std::vector<TokenId>& out,
                                          PieceCache* cache) const;
    void encode_piece(std::string_view piece, std::vector<TokenId>& out, PieceCache* cache, std::string& spelled,
                      std::vector<TokenId>& symbols) const;

    Vocabulary vocabulary_;
    SpecialTokens special_;
    MergeTable merges_;
    std::array<TokenId, 256> byte_tokens_{};   // the token for each byte's character
    const PreTokenizer* pretokenizer_ = nullptr;
};

// The algorithm, as the capability table lists it: it splits text first, so a
// file must name its pre-tokenizer.
extern const Algorithm kByteLevel;

// Reads the vocabulary and merges, and checks every byte has a token and every
// normal token spells bytes. A token that does not is refused rather than
// decoded by a guess: Hugging Face would give its text as written, llama.cpp
// a marker naming the character. `out` is left untouched unless it succeeds.
[[nodiscard]] LoadResult load_byte_level_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                             const PreTokenizer& pretokenizer, ByteLevelBpe& out);

// The algorithm as a Tokenizer (tokenizer.h): the encoder and its piece
// cache, kept for the tokenizer's life, so a turn's encode of the whole
// conversation finds the pieces earlier turns merged. Warm, that cache took
// Qwen3's encode of the bench corpus — 385,417 bytes, the Makefile's
// BENCH_COMMIT — from 15.7 ms to 11.1 ms and Llama 3.2's from 20.9 to 11.7
// (piece_cache.h): 28.8 and 30.4 ns a byte, so a conversation of Qwen3's
// whole context, near 160 KB, encodes in about 4.6 ms a turn.
class ByteLevelTokenizer final : public Tokenizer {
public:
    [[nodiscard]] const Vocabulary& vocabulary() const noexcept override;
    [[nodiscard]] EncodeError encode(std::string_view text, std::vector<TokenId>& out) override;
    void decode(TokenId token, std::string& out) const override;

private:
    friend LoadResult load_byte_level_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                                const PreTokenizer* pretokenizer, std::unique_ptr<Tokenizer>& out);
    ByteLevelBpe bpe_;
    PieceCache cache_;
};

// kByteLevel's load (tokenizer.h's LoadFn). Precondition: `pretokenizer` is
// not null, the algorithm requiring one.
[[nodiscard]] LoadResult load_byte_level_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                                   const PreTokenizer* pretokenizer,
                                                   std::unique_ptr<Tokenizer>& out);

}  // namespace bllm::tokenizer::bpe
