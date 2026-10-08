// Admission (tokenizer/admission.h): from the counts alone, whether an
// encode goes on to merge.

#include <doctest/doctest.h>

#include <array>
#include <cstdint>

#include "core/tokenizer/admission.h"

using namespace bllm::tokenizer;

TEST_CASE("the least count takes each special as one token and rounds each segment up") {
    // A multiple of the cover, and one byte past it.
    const std::array<std::uint64_t, 1> exact{256};
    CHECK(admit(0, exact, 128, kNoTokenLimit).least_tokens == 2);
    const std::array<std::uint64_t, 1> past{257};
    CHECK(admit(0, past, 128, kNoTokenLimit).least_tokens == 3);
    // Specials alone.
    CHECK(admit(5, {}, 128, kNoTokenLimit).least_tokens == 5);
    // Segments each round up: no token spans two.
    const std::array<std::uint64_t, 3> three{1, 1, 1};
    const EncodeResult r = admit(2, three, 128, kNoTokenLimit);
    CHECK(r.least_tokens == 5);
    CHECK(r.normalized_bytes == 3);
    CHECK(r.error == EncodeError::Ok);
}

TEST_CASE("the limit met is admitted and passed by one is refused, naming the count") {
    const std::array<std::uint64_t, 1> ten{1280};
    CHECK(admit(0, ten, 128, 10).error == EncodeError::Ok);
    const std::array<std::uint64_t, 1> eleven{1281};
    const EncodeResult r = admit(0, eleven, 128, 10);
    CHECK(r.error == EncodeError::TooManyTokens);
    CHECK(r.least_tokens == 11);
    // A special counts toward it as much as text.
    CHECK(admit(1, ten, 128, 10).error == EncodeError::TooManyTokens);
}

TEST_CASE("kMaxEncodeBytes met is admitted and passed by one is refused, the token refusal first") {
    const std::array<std::uint64_t, 1> at{kMaxEncodeBytes};
    CHECK(admit(0, at, 128, kNoTokenLimit).error == EncodeError::Ok);
    const std::array<std::uint64_t, 2> past{kMaxEncodeBytes, 1};
    const EncodeResult r = admit(0, past, 128, kNoTokenLimit);
    CHECK(r.error == EncodeError::TooLong);
    CHECK(r.normalized_bytes == kMaxEncodeBytes + 1);
    // Both hold: named by its tokens, which the page can shorten by.
    CHECK(admit(0, past, 128, 10).error == EncodeError::TooManyTokens);
}

TEST_CASE("raw text past 7/2 of kMaxEncodeBytes and the limit's specials is refused unsegmented") {
    // A limit of 10 and specials of at most 30 bytes allow 300 more.
    const std::uint64_t at = kMaxEncodeBytes * 7 / 2 + 10 * 30;
    CHECK(admit_raw(at, 10, 30).error == EncodeError::Ok);
    const EncodeResult r = admit_raw(at + 1, 10, 30);
    CHECK(r.error == EncodeError::TooLong);
    CHECK(r.raw_bytes == at + 1);
    CHECK(r.normalized_bytes == 0);
    // Special tokens' bytes are theirs: 61,167 of Llama 3.2's 30-byte
    // <|reserved_special_token_247|>, 1,835,010 bytes, two past 7/2 of
    // kMaxEncodeBytes, are 61,167 tokens, within its 131,072.
    CHECK(admit_raw(61'167 * 30, 131'072, 30).error == EncodeError::Ok);
}
