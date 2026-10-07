// Top-k selection's launches, on the CPU (kernels/topk/topk.h): the passes a
// vocabulary takes, what each reads and writes, and when they run.

#include <doctest/doctest.h>

#include <vector>

#include "core/kernels/topk/topk.h"

using namespace bllm;

namespace {

const residency::BufferRange kLogits{residency::BufferIndex{1}, 0, 1u << 20}, kA{residency::BufferIndex{2}, 0, 1u << 17},
    kB{residency::BufferIndex{3}, 0, 1u << 17}, kCandidates{residency::BufferIndex{4}, 0, 512};

std::uint64_t workgroups(const kernels::Launch& l, bool logits) {
    return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                   l.workgroup_size, 0, 1, logits);
}

}  // namespace

TEST_CASE("each listed vocabulary is three passes, alternating buffers, the last writing the candidates") {
    struct Case {
        std::uint32_t vocabulary;
        std::vector<std::uint64_t> tiles;
    };
    for (const Case& c : {Case{151'936, {149, 10, 1}}, Case{128'256, {126, 8, 1}}, Case{262'144, {256, 16, 1}}}) {
        CAPTURE(c.vocabulary);
        const auto launches = kernels::topk_launches(kLogits, c.vocabulary, kA, kB, kCandidates);
        REQUIRE(launches.size() == 3);
        const std::vector<residency::BufferIndex> reads{kLogits.buffer, kA.buffer, kB.buffer},
            writes{kA.buffer, kB.buffer, kCandidates.buffer};
        for (std::size_t i = 0; i < 3; ++i) {
            CAPTURE(i);
            CHECK(launches[i].entry_point == (i == 0 ? "first" : "merge"));
            CHECK(launches[i].bindings[0].buffer == reads[i]);
            CHECK(launches[i].bindings[1].buffer == writes[i]);
            CHECK(workgroups(launches[i], true) == c.tiles[i]);
            CHECK(workgroups(launches[i], false) == 0);   // only a step that asks for logits
        }
    }
}

TEST_CASE("the largest vocabulary a u32 holds takes its passes, no tile count wrapping") {
    const auto launches = kernels::topk_launches(kLogits, UINT32_MAX, kA, kB, kCandidates);
    // Tiles of 4,194,304, then 262,144, 16,384, 1,024, 64, 4 and 1; a
    // dispatch that large the program refuses, but the launcher never loops.
    REQUIRE(launches.size() == 7);
    CHECK(workgroups(launches[0], true) == 4'194'304);
    CHECK(workgroups(launches[6], true) == 1);
}

TEST_CASE("a vocabulary of one tile is one pass, and one of 1,025 two") {
    CHECK(kernels::topk_launches(kLogits, 64, kA, kB, kCandidates).size() == 1);
    CHECK(kernels::topk_launches(kLogits, 1024, kA, kB, kCandidates).size() == 1);
    const auto two = kernels::topk_launches(kLogits, 1025, kA, kB, kCandidates);
    REQUIRE(two.size() == 2);
    CHECK(two[0].bindings[1].buffer == kA.buffer);
    CHECK(two[1].bindings[0].buffer == kA.buffer);
    CHECK(two[1].bindings[1].buffer == kCandidates.buffer);
}
