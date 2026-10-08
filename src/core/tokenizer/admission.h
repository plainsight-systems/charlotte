#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/tokenizer/tokenizer.h"

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Whether an encode goes on to merge (tokenizer.h, encoding is bounded):
// pure functions of counts, the ones both algorithms call, so the bound is
// decided in one place and tested apart from either.
//
//   - admit_raw: raw text over 7/2 × kMaxEncodeBytes, plus the limit's worth
//     of the longest special token, is TooLong before it is segmented or
//     normalized, naming its raw bytes. Text an encode could admit has at
//     most the limit's special tokens, each at most that long, and ordinary
//     bytes that normalize to at most kMaxEncodeBytes — at most 7/2 times
//     that before NFC, which shrinks UTF-8 at most 7/2-fold, and no more
//     before SentencePiece's escaping, which never shrinks it — so this
//     never refuses text that would be admitted.
//   - admit: the least tokens the text makes are its specials, one each,
//     and each ordinary segment's ceil(n / cover) for its n normalized
//     bytes, since no token spans a special or two segments. Over the limit,
//     TooManyTokens with that count; then normalized bytes past
//     kMaxEncodeBytes, TooLong; otherwise Ok, with both counts. The count is
//     a lower bound, so it never refuses text that would fit.
//   - Sums are in 64 bits: a segment is at most the text, at most 4 GiB of
//     the module's memory, and there are at most as many as its bytes.
//
// What it costs: a division a segment, once a turn.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — counts in, a verdict out.
//     E.27   Use error codes systematically — the verdict is an
//            EncodeResult.
//   C++ performance guidelines
//     EMB.4  Size the worst case — the bound kMaxEncodeBytes enforces, and
//            the token bound ahead of it.

[[nodiscard]] EncodeResult admit_raw(std::uint64_t raw_bytes, std::uint32_t max_tokens,
                                     std::size_t longest_special) noexcept;

// Preconditions: cover >= 1.
[[nodiscard]] EncodeResult admit(std::uint64_t specials, std::span<const std::uint64_t> segment_bytes,
                                 std::size_t cover, std::uint32_t max_tokens) noexcept;

}  // namespace bllm::tokenizer
