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
