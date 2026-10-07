// The sampler on the CPU (sampler/sampler.h): the settings it refuses, and
// the reference Philox the GPU's draw is checked against, against
// Random123's published known-answer vectors.

#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "core/sampler/sampler.h"
#include "support/sampler_reference.h"

using namespace bllm;

TEST_CASE("Philox4x32-10 gives Random123's known answers") {
    struct Case {
        testing::Philox counter;
        std::uint32_t key0, key1;
        testing::Philox want;
    };
    for (const Case& c : {
             Case{{{0, 0, 0, 0}}, 0, 0, {{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}}},
             Case{{{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}}, 0xffffffff, 0xffffffff,
                  {{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}}},
             Case{{{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}}, 0xa4093822, 0x299f31d0,
                  {{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}}},
         }) {
        const testing::Philox got = testing::philox4x32_10(c.counter, c.key0, c.key1);
        for (int i = 0; i < 4; ++i) CHECK(got.words[i] == c.want.words[i]);
    }
}

TEST_CASE("the uniform lies in [2^-24, 1 - 2^-24], never 0 or 1") {
    CHECK(static_cast<float>(0u) * (1.0f / 8388608.0f) + (1.0f / 16777216.0f) == std::ldexp(1.0f, -24));
    CHECK(static_cast<float>(0x7FFFFFu) * (1.0f / 8388608.0f) + (1.0f / 16777216.0f) == 1.0f - std::ldexp(1.0f, -24));
}

TEST_CASE("settings outside their ranges are refused by name, never clamped") {
    const auto with = [](auto change) {
        policy::SamplingSettings s;
        change(s);
        return sampler::check(s);
    };
    CHECK(sampler::check(policy::SamplingSettings{}).ok);
    CHECK(with([](auto& s) { s.top_k = 64; }).ok);
    CHECK(with([](auto& s) { s.temperature = 0; }).ok);
    CHECK(with([](auto& s) { s.top_p = 1; }).ok);
    CHECK(with([](auto& s) { s.min_p = 0; }).ok);
    CHECK(with([](auto& s) { s.top_k = 0; }).subject == "top_k 0 is outside 1 to 64");
    CHECK(with([](auto& s) { s.top_k = 65; }).subject == "top_k 65 is outside 1 to 64");
    CHECK_FALSE(with([](auto& s) { s.temperature = -1; }).ok);
    CHECK_FALSE(with([](auto& s) { s.temperature = std::numeric_limits<float>::infinity(); }).ok);
    CHECK_FALSE(with([](auto& s) { s.temperature = std::numeric_limits<float>::quiet_NaN(); }).ok);
    CHECK_FALSE(with([](auto& s) { s.top_p = 0; }).ok);
    CHECK_FALSE(with([](auto& s) { s.top_p = 1.5f; }).ok);
    CHECK_FALSE(with([](auto& s) { s.min_p = 1; }).ok);
    CHECK_FALSE(with([](auto& s) { s.min_p = -0.1f; }).ok);
}
