#pragma once

#include <array>
#include <cstddef>
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
//   - Encoding is bounded, by the caller's token limit and by the bytes one
//     encode takes, before any piece is merged — the work, and the memory,
//     that grow with a piece's length (bpe/merge.h). encode takes the
//     limit, max_tokens, and returns an EncodeResult: the error, the least
//     tokens and normalized bytes admission counted, and the raw bytes. In
//     order:
//       1. admit_raw (admission.h): raw text past 7/2 × kMaxEncodeBytes plus
//          max_tokens × the longest special token is TooLong, unread. Text
//          that could be admitted is within it: at most the limit's special
//          tokens, and ordinary bytes at most 7/2 times its normalized
//          bytes — NFC shrinks UTF-8 at most 7/2-fold, and SentencePiece's
//          escaping never shrinks it.
//       2. The special tokens split out, stopping past the limit's worth of
//          them as TooManyTokens, so the segments are at most two a token
//          allowed; and the rest normalized, in the algorithm's own order —
//          byte-level BPE segments the raw text and applies NFC to each
//          ordinary segment; SentencePiece escapes spaces as ▁ first and
//          segments after, its special tokens, Gemma 3's runs of spaces
//          among them, spelled that way.
//       3. admit: each special token is a token, and a segment of n
//          normalized bytes makes at least ceil(n / longest_cover()) tokens
//          — no token spans a special or two segments — so their sum, in 64
//          bits, over the limit is TooManyTokens, naming that least count;
//          then normalized text past kMaxEncodeBytes is TooLong, naming its
//          bytes. The token check first, so text a context could never hold
//          is named by its tokens, by which the page shortens a
//          conversation. The count is a lower bound: admitted text may still
//          make more tokens than the limit, which the caller counts after
//          encoding; none of the three refuses text that would fit.
//       4. Only admitted text is pre-tokenized and merged.
//     A refusal leaves `out` untouched. The memory bound below holds for a
//     limit of a context; kNoTokenLimit, a test's and the bench's, leaves
//     step 1 nothing to refuse.
//   - longest_cover() is the most normalized bytes one token covers,
//     computed at load over the whole vocabulary: a token's decoded bytes
//     for byte-level BPE, its spelling for SentencePiece, whose normalized
//     text is spelled alike: 128 for Qwen3 and Llama 3.2, runs of spaces,
//     and 48 for Gemma 3, each a normal token, every other type shorter.
//   - kMaxEncodeBytes, 512 KiB of normalized text, is sized from the
//     module's worst case during an encode (WASM.1, EMB.4). Resident: the
//     tokenizer, measured natively, 64-bit, at 12.5 MiB for Qwen3, 18.4 for
//     Llama 3.2 and 30.2 for Gemma 3, 16 MiB of it Gemma 3's merge table,
//     its 4-byte-pointer build no larger; the load's 16 MiB chunk buffer;
//     and the module, 450 KB: under 47 MiB. The encode, before admission:
//     the raw text, at most 1.75 MiB and the specials' allowance, 3.75 MiB
//     for Llama 3.2's 131,072 × 30 bytes, the largest — 5.5 MiB; its
//     segments, 16 bytes each, at most two a token allowed, 4 MiB; and its
//     normalized text, at most 3 times the ordinary bytes NFC reads, 5.25
//     MiB, or SentencePiece's escaped text, at most 3 times the raw, 8.2
//     MiB for Gemma 3 — under 18 MiB. After it, per normalized byte:
//     pre-tokenization's characters, 12, and pieces, 8, doubled by their
//     vector's growth; the merge of a piece holding all of it, its symbols,
//     4 doubled, its chain, 12, and its candidate heap, at most 2 a symbol
//     live, 24 bytes each, whose vector's capacity, at most 4 a symbol, is
//     copied once more as it grows, 144; and the tokens, 4 doubled, and the
//     generator's copy, 4 — 204 bytes, at most 102 MiB. The encode's 120
//     MiB, 240 with EMB.4's margin of 2, beside the resident 47: 287 MiB,
//     under the build's 512 MiB maximum (CMakeLists.txt). 512 KiB is
//     131,072 tokens of English at 4 bytes a token, Llama 3.2's whole
//     context: what it refuses that a context could hold is text averaging
//     more bytes a token than 512 KiB over the context — 32 for a
//     16,384-token context — such as long runs of letters, a stated limit
//     (WASM.1's caveat), named.
//   - Verification: admit, on the CPU — the least count at a segment of a
//     multiple of the cover and one byte past it, specials alone, segments
//     that each round up; the limit met exactly and passed by one;
//     kMaxEncodeBytes met exactly and passed by one; the token refusal ahead
//     of the byte one when both hold. Each listed model's tokenizer: its
//     longest_cover() as above; a run of one letter, matching no special
//     token, whose least count meets the limit admitted, its tokens those an
//     encode without a limit gives, and one more letter refused naming its
//     count, out untouched; text one byte past kMaxEncodeBytes normalized
//     refused as TooLong, and raw text past step 1's bound refused unread;
//     61,167 of Llama 3.2's 30-byte special token, past 7/2 × kMaxEncodeBytes
//     but within its context, encoded; one special token past the limit
//     refused naming the limit and one; and every reference case encoded
//     with no limit, the same tokens as before.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.25   Prefer empty abstract classes as interfaces — Tokenizer holds no
//            data; its virtual destructor public (C.127).
//     C.121  If a base class is used as an interface, make it a pure abstract
//            class.
//     SL.io.2 When reading, always consider ill-formed input — a prompt of
//            any size is input; its size is checked before work grows with it.
//     E.27   Use error codes systematically — TooManyTokens and TooLong,
//            each with its counts.
//     F.8    Prefer pure functions — admit, counts in and a verdict out.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — the dispatch is once a
//            turn and once a token; implementations are final.
//     WASM.1 Size linear memory to the real high-water mark — the encode's
//            worst case counted against the heap's maximum; the cap stated.
//     EMB.4  Size the worst case of dynamic allocation, with margin.

// A token's identifier. Its own type, so it cannot be passed where a count or
// a position is meant.
enum class TokenId : std::uint32_t {};

// The most normalized bytes one encode takes (above).
inline constexpr std::uint64_t kMaxEncodeBytes = 512 * 1024;

// A token limit that refuses nothing by count, for an encode bounded by
// kMaxEncodeBytes alone.
inline constexpr std::uint32_t kNoTokenLimit = UINT32_MAX;

enum class EncodeError {
    Ok,
    InvalidUtf8,
    TooManyTokens,   // more tokens, at least, than the caller's limit
    TooLong,         // past kMaxEncodeBytes, normalized
};

// An encode's outcome: the least tokens the text makes and its normalized
// bytes, as admission counted them (admission.h), and its raw bytes. Text
// refused before it is segmented or normalized has only its raw bytes; text
// refused for its special tokens alone, its least tokens.
struct EncodeResult {
    EncodeError error = EncodeError::Ok;
    std::uint64_t least_tokens = 0;
    std::uint64_t normalized_bytes = 0;
    std::uint64_t raw_bytes = 0;
};

// A loaded tokenizer, whatever its algorithm.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    [[nodiscard]] virtual const Vocabulary& vocabulary() const noexcept = 0;

    // Appends the tokens `text` encodes to, refusing text that makes more
    // than `max_tokens` at least, or is too long to encode (above); `out` is
    // left untouched if it refuses or cannot. Not const: an algorithm may
    // keep what it has found, as byte-level BPE keeps its piece cache, which
    // changes how long encoding takes and never what it returns.
    [[nodiscard]] virtual EncodeResult encode(std::string_view text, std::uint32_t max_tokens,
                                              std::vector<TokenId>& out) = 0;

    // The most normalized bytes one token covers (above).
    [[nodiscard]] virtual std::size_t longest_cover() const noexcept = 0;

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
