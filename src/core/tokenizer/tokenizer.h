#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace bllm::gguf {
class ByteSource;
class TensorIndex;
}  // namespace bllm::gguf

namespace bllm::tokenizer {

class Vocabulary;
struct LoadResult;
struct PreTokenizer;

// Contract 9: the tokenizer.
//
// Each tokenizer/<algorithm>/ provides one Algorithm, and the capability table
// lists it under the tokenizer.ggml.model value it implements. It builds its
// tokenizer from the vocabulary arrays the tensor index locates.
//
//   - Encoding turns rendered text into identifiers. The chat template writes
//     special tokens as text; they encode as their single identifiers, never
//     as the characters that spell them.
//   - An algorithm that splits text first names its pre-tokenizer with
//     tokenizer.ggml.pre. One that does not, such as SentencePiece, needs none.
//   - Decoding gives each token's bytes back, special tokens included as the
//     text that encodes to them, and rewrites nothing: decoding what was
//     encoded gives the text back, normalized if the algorithm normalizes.
//     Whether a control token is shown is the caller's choice; it has the
//     token's type. The bytes become text through a stream (Utf8Stream below).
//   - A loaded tokenizer is used through Tokenizer, below, whatever its
//     algorithm: chosen once at load from the file's tokenizer.ggml.model by
//     the capability table, whose entry for the algorithm carries the
//     function that loads it. The table stays the one list of algorithms;
//     the runtime and the boundary name none. An abstract interface, since
//     the algorithms hold different state — a merge table and a piece cache,
//     or scores and byte tokens — and are chosen at run time; each is a
//     final class. It is called once a turn to encode and once a token to
//     decode, never in a loop over tokens' work, so its indirect call is paid
//     at those rates (WASM.4's caveat).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.25   Prefer empty abstract classes as interfaces — Tokenizer holds no
//            data; its virtual destructor public (C.127).
//     C.121  If a base class is used as an interface, make it a pure abstract
//            class.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — the dispatch is once a
//            turn and once a token; implementations are final.

// A token's identifier. Its own type, so it cannot be passed where a count or
// a position is meant.
enum class TokenId : std::uint32_t {};

enum class EncodeError {
    Ok,
    InvalidUtf8,
    TooLong,   // longer than one encode can address (4 GiB)
};

// A loaded tokenizer, whatever its algorithm.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    [[nodiscard]] virtual const Vocabulary& vocabulary() const noexcept = 0;

    // Appends the tokens `text` encodes to; `out` is left untouched if it
    // cannot. Not const: an algorithm may keep what it has found, as
    // byte-level BPE keeps its piece cache, which changes how long encoding
    // takes and never what it returns.
    [[nodiscard]] virtual EncodeError encode(std::string_view text, std::vector<TokenId>& out) = 0;

    // Appends the bytes `token` stands for. Precondition: the token is below
    // vocabulary().size().
    virtual void decode(TokenId token, std::string& out) const = 0;
};

// Loads an algorithm's tokenizer from the file the index describes: its
// pre-tokenizer, for an algorithm that requires one, else null. `out` is left
// untouched unless it succeeds.
using LoadFn = LoadResult (*)(gguf::ByteSource& source, const gguf::TensorIndex& index,
                              const PreTokenizer* pretokenizer, std::unique_ptr<Tokenizer>& out);

struct Algorithm {
    std::string_view name;
    bool requires_pretokenizer;
    LoadFn load;
};

// Turns the bytes of successive tokens into text. A token can end partway
// through a UTF-8 character; those bytes are held until the next token
// completes them, so every string emitted is whole characters. Bytes that
// cannot begin or continue a character are replaced with U+FFFD.
class Utf8Stream {
public:
    // Appends every character completed by `token_bytes` to `out`.
    void push(std::string_view token_bytes, std::string& out);

    // Ends the stream. Returns false if bytes were pending — generation
    // stopped inside a character — in which case U+FFFD is appended in their
    // place rather than dropping them unseen. The stream is then empty, ready
    // for another.
    [[nodiscard]] bool finish(std::string& out);

private:
    std::array<char, 3> pending_{};
    std::uint8_t pending_count_ = 0;
};

}  // namespace bllm::tokenizer
