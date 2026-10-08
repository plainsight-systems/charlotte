#include "core/tokenizer/admission.h"

namespace bllm::tokenizer {

EncodeResult admit_raw(std::uint64_t raw_bytes, std::uint32_t max_tokens, std::size_t longest_special) noexcept {
    // Doubled, in integers: 2 × bytes against 7 × kMaxEncodeBytes and twice
    // the specials' allowance, at most 2 × 2^32 × a special's length.
    const std::uint64_t allowed2 = 7 * kMaxEncodeBytes + 2 * std::uint64_t{max_tokens} * longest_special;
    if (2 * raw_bytes > allowed2) return {EncodeError::TooLong, 0, 0, raw_bytes};
    return {EncodeError::Ok, 0, 0, raw_bytes};
}

EncodeResult admit(std::uint64_t specials, std::span<const std::uint64_t> segment_bytes, std::size_t cover,
                   std::uint32_t max_tokens) noexcept {
    std::uint64_t least = specials;
    std::uint64_t bytes = 0;
    for (const std::uint64_t n : segment_bytes) {
        least += n / cover + (n % cover != 0 ? 1 : 0);
        bytes += n;
    }
    if (least > max_tokens) return {EncodeError::TooManyTokens, least, bytes, 0};
    if (bytes > kMaxEncodeBytes) return {EncodeError::TooLong, least, bytes, 0};
    return {EncodeError::Ok, least, bytes, 0};
}

}  // namespace bllm::tokenizer
