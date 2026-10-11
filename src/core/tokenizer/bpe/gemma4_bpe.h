#pragma once

#include <array>
#include <cstdint>
#include <memory>
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
// The algorithm GGUF names "gemma4": BPE over characters, spaces spelled
// "▁" as SentencePiece spells them, with the merges the file lists and byte
// tokens for characters the vocabulary lacks — Gemma 4's, and EmbeddingGemma
// 2's. As llama.cpp's llama-vocab reads "gemma4" (its LLAMA_VOCAB_PRE_TYPE_GEMMA4),
// encoding a rendered text:
//
//   1. Special tokens are found in the raw text and become their own
//      identifiers (special.h), before any space is respelled: llama.cpp
//      partitions them out first and escapes each fragment after, the
//      order sentencepiece_bpe.h's step 2 reverses for Gemma 3.
//   2. In the text between them, each space becomes "▁" (U+2581). Nothing
//      else is normalized, and no space is added before the text
//      (tokenizer.ggml.add_space_prefix false, which the load requires).
//   3. The text is cut into lines: each run of characters other than "\n",
//      and each run of "\n" — llama.cpp's split for gemma4,
//      "[^\n]+|[\n]+". A run of "\n" that is itself a token is that token
//      whole, as llama.cpp takes it (its fix for Gemma 4's newline runs);
//      any other piece goes on to step 4.
//   4. Each character of a piece becomes its token, or, where the
//      vocabulary has none, one byte token (<0x00> to <0xFF>) for each of
//      its bytes; then the piece is merged by the file's merges
//      (tokenizer.ggml.merges, ranked by their place in the list) through
//      the merge step byte-level BPE uses (merge.h). A character with no
//      token can take part in no merge — every rule joins two tokens — so
//      spelling it in bytes before merging gives what llama.cpp's
//      byte fallback gives after.
//
// So it is sentencepiece_bpe.h's encoding with three differences, each
// following llama.cpp's reading of the identifier: the merges are listed,
// not implied by scores; special tokens are matched before spaces are
// respelled, not after; and text is cut at line breaks before merging. Its
// own algorithm, not a flag on SentencePieceBpe: an identifier selects one
// implementation (logical-overview.md, principle 2), and a change to either
// would otherwise re-verify both.
//
// BOS and EOS are never added: the caller's — a chat template writes them,
// and an embedder adds them where the file says (runtime/embedder.h).
//
// Decoding a token gives a normal token's text with each "▁" a space again,
// a byte token's byte, and any other token's text as written, as
// SentencePieceBpe decodes.
//
// What it costs, counted, for EmbeddingGemma 2's tokenizer — 262,144 tokens,
// 514,906 merges:
//   - At load: the vocabulary, its special tokens and its byte tokens read
//     as SentencePieceBpe reads them; the merges read and hashed once into
//     an open-addressing table (merge_table.h), 514,906 rules at three
//     quarters full in a power-of-two table: 1,048,576 slots of 16 bytes,
//     16 MiB, as Gemma 3's derived merges take. No merge is derived from
//     scores, so the load skips sentencepiece_merges.h's search over every
//     split of every token.
//   - An encode: about sentencepiece_bpe.h's 43 ns a byte of normalized
//     text, a piece being a line where Gemma 3's stretch runs from one
//     special token to the next, so a piece is no longer and the merge's
//     heap no larger. What is merged is bounded by admission, not by the
//     tokens it makes: text admitted under a limit of T tokens is at most T
//     × longest_cover() normalized bytes (tokenizer.h). This vocabulary's
//     longest token is a run of 31 "▁", 93 bytes, so an encode under the
//     embedder's 510 merges at most 47,430 bytes, about 2 ms.
//   - Memory during an encode: within tokenizer.h's bound for
//     kMaxEncodeBytes, whose count covers SentencePiece's escaped text, the
//     larger.
//
// Load refuses, naming what it found: a vocabulary whose byte tokens are not
// exactly <0x00> to <0xFF>; a merge naming a token the vocabulary lacks, or
// two merges of one pair (merge_table.h); two special tokens spelled alike;
// tokenizer.ggml.add_space_prefix true or undeclared, as SentencePieceBpe's
// load does; and a file with no tokenizer.ggml.merges, which this
// algorithm cannot run without.
//
// Verification the implementation is held to: tools/make_tokenizer_fixtures.py
// gains EmbeddingGemma 2, encoding its cases with Hugging Face's tokenizer
// on google/embeddinggemma-2's tokenizer.json and with llama-tokenize at the
// embeddings' pin (runtime/embedder.h), a case written only where the two
// agree, as for every model; its cases include runs of newlines that are
// tokens and ones that are not, spaces beside a typed "▁", a special token
// holding a space, characters outside the vocabulary in each UTF-8 length,
// and the task prompts the model's card gives. Each fixture's identifiers
// ours, and decoding them gives the text back with spaces for "▁". Its
// longest_cover() over the vocabulary, as tokenizer.h's admission needs.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     SL.io.2 When reading, always consider ill-formed input — each refusal
//            above, named.
//     E.27   Use error codes systematically — LoadResult and EncodeResult.
//     C.121  If a base class is used as an interface, make it a pure abstract
//            class — Gemma4Tokenizer implements tokenizer.h's Tokenizer,
//            final.
//   C++ performance guidelines
//     WASM.1 Size linear memory to the real high-water mark — the merge
//            table's 16 MiB, counted against the heap as tokenizer.h counts
//            Gemma 3's.
//     WASM.4 Reduce indirect dispatch in hot paths — one call an encode and
//            one a token, through tokenizer.h's interface.

class Gemma4Bpe {
public:
    Gemma4Bpe() = default;

    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocabulary_; }

    // Appends the tokens `raw` encodes to, bounded as tokenizer.h's encoding
    // is: refused, `out` untouched, for text making more than `max_tokens` at
    // least or past kMaxEncodeBytes once spaces are ▁, as for text that is
    // not UTF-8.
    [[nodiscard]] EncodeResult encode(std::string_view raw, std::uint32_t max_tokens,
                                      std::vector<TokenId>& out) const;

    // The most normalized bytes one token covers, ▁ for a space, found at
    // load (tokenizer.h).
    [[nodiscard]] std::size_t longest_cover() const noexcept { return cover_; }

    // Appends the bytes `token` stands for. Precondition: loaded, and the
    // token is below vocabulary().size().
    void decode(TokenId token, std::string& out) const;

private:
    friend LoadResult load_gemma4_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index, Gemma4Bpe& out);

    Vocabulary vocabulary_;
    SpecialTokens special_;
    MergeTable merges_;
    std::array<TokenId, 256> byte_tokens_{};   // <0x00> to <0xFF>
    std::size_t cover_ = 1;
};

// The algorithm, as the capability table lists it under "gemma4". It names
// no pre-tokenizer: its line split is part of what the identifier means.
extern const Algorithm kGemma4;

// Reads the vocabulary and the listed merges, and finds the byte tokens.
// Refuses as above; `out` is left untouched unless it succeeds.
[[nodiscard]] LoadResult load_gemma4_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index, Gemma4Bpe& out);

// The algorithm as a Tokenizer (tokenizer.h).
class Gemma4Tokenizer final : public Tokenizer {
public:
    [[nodiscard]] const Vocabulary& vocabulary() const noexcept override;
    [[nodiscard]] EncodeResult encode(std::string_view text, std::uint32_t max_tokens,
                                      std::vector<TokenId>& out) override;
    [[nodiscard]] std::size_t longest_cover() const noexcept override;
    void decode(TokenId token, std::string& out) const override;

private:
    friend LoadResult load_gemma4_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                            const PreTokenizer* pretokenizer, std::unique_ptr<Tokenizer>& out);
    Gemma4Bpe bpe_;
};

// kGemma4's load (tokenizer.h's LoadFn); `pretokenizer` is unused.
[[nodiscard]] LoadResult load_gemma4_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                               const PreTokenizer* pretokenizer, std::unique_ptr<Tokenizer>& out);

}  // namespace bllm::tokenizer::bpe
