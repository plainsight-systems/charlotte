// The launches a step can dispatch, planned at build (kernels/schedule.h).

#include <doctest/doctest.h>

#include <vector>

#include "core/kernels/schedule.h"

using namespace bllm::kernels;

TEST_CASE("a step's schedule lists the launches its token count and logits allow, in order") {
    const std::vector<Geometry> launches{
        {Rows::EveryToken, 64, 0, KeySplit::None, 0},                           // 0: every step
        {Rows::EveryToken, 64, 0, KeySplit::None, 0, tokens_of(Regime::Decode)}, // 1: decode
        {Rows::EveryToken, 64, 8, KeySplit::None, 0, {2, 8}},                    // 2: a narrow tile
        {Rows::EveryToken, 64, 32, KeySplit::None, 0, {9, UINT32_MAX}},          // 3: a wide tile
        {Rows::EveryToken, 64, 0, KeySplit::WhenSplit, 4096},                    // 4: a combine
        {Rows::LastToken, 64, 0, KeySplit::None, 0},                             // 5: the head
    };
    const Schedules s{launches};
    const auto is = [](std::span<const std::uint32_t> got, std::vector<std::uint32_t> want) {
        return std::vector<std::uint32_t>(got.begin(), got.end()) == want;
    };
    CHECK(is(s.of(1, true), {0, 1, 4, 5}));
    CHECK(is(s.of(1, false), {0, 1, 4}));
    CHECK(is(s.of(2, true), {0, 2, 4, 5}));
    CHECK(is(s.of(8, false), {0, 2, 4}));
    CHECK(is(s.of(9, true), {0, 3, 4, 5}));
    CHECK(is(s.of(512, false), {0, 3, 4}));
}

TEST_CASE("a launch over the step's tokens divided runs by its own rows, past a prefill block of tokens") {
    // A vision step of 2,520 patches: a launch over every patch, and the
    // pooled projection's tiles over the patches divided by 9 — its 8-row
    // tile at 1 to 8 pooled rows, its 32-row tile from 17.
    const std::vector<Geometry> launches{
        {Rows::EveryToken, 64, 0, KeySplit::None, 0, {}},
        {Rows::EveryToken, 64, 8, KeySplit::None, 0, {1, 8}, bllm::residency::kPrefillBlock, 9},
        {Rows::EveryToken, 64, 32, KeySplit::None, 0, {17, UINT32_MAX}, bllm::residency::kPrefillBlock, 9},
    };
    const Schedules s(launches);
    const auto at = [&](std::uint32_t tokens) {
        const auto list = s.of(tokens, true);
        return std::vector<std::uint32_t>(list.begin(), list.end());
    };
    CHECK(at(9) == std::vector<std::uint32_t>{0, 1});        // 1 pooled row
    CHECK(at(72) == std::vector<std::uint32_t>{0, 1});       // 8
    CHECK(at(81) == std::vector<std::uint32_t>{0});          // 9: neither tile
    CHECK(at(153) == std::vector<std::uint32_t>{0, 2});      // 17
    CHECK(at(2520) == std::vector<std::uint32_t>{0, 2});     // 280
    CHECK(workgroups_for(launches[2], 64, 0, 2520, true) == (280 + 31) / 32 * 32);
    CHECK(workgroups_for(launches[1], 64, 0, 2520, true) == 0);
    CHECK(workgroups_for(launches[0], 64, 0, 2520, true) == 2520);
}
