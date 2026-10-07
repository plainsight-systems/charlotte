// The matmul launcher, on the CPU (kernels/matmul/matmul.h): its launch
// pairs and regimes, entry points, geometry, and a fused group's spanning
// binding and members.

#include <doctest/doctest.h>

#include <cstring>
#include <string_view>

#include "core/kernels/matmul/matmul.h"

using namespace bllm;
using kernels::Epilogue;
using kernels::Regime;

namespace {

residency::WeightView q4_0(std::uint64_t columns, std::uint64_t rows, std::uint64_t offset,
                           std::uint64_t first_row = 0) {
    gguf::TensorShape shape{};
    shape.dimension_count = 2;
    shape.dimensions[0] = columns;
    shape.dimensions[1] = rows;
    shape.element_count = columns * rows;
    const std::uint64_t bytes = columns / 32 * 18 * rows;
    return residency::WeightView(gguf::TensorType::Q4_0, shape,
                                 {{residency::BufferIndex{0}, offset, bytes, first_row, rows}});
}

const residency::BufferRange kIn{residency::BufferIndex{1}, 0, 4096}, kQ{residency::BufferIndex{2}, 0, 4096},
    kK{residency::BufferIndex{3}, 0, 4096}, kV{residency::BufferIndex{4}, 0, 4096};

std::array<std::uint32_t, 12> members_of(const kernels::Launch& l) {
    std::array<std::uint32_t, 12> m{};
    std::memcpy(m.data(), l.constants.data(), sizeof m);
    return m;
}

}  // namespace

TEST_CASE("a product is a decode and a prefill launch, each in its regime; the head one launch in every regime") {
    const auto w = q4_0(1024, 3072, 0);
    const auto launches = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::EveryToken});
    REQUIRE(launches.size() == 2);
    CHECK(launches[0].entry_point == "decode_write");
    CHECK(launches[0].regime == Regime::Decode);
    CHECK(launches[1].entry_point == "prefill_write");
    CHECK(launches[1].regime == Regime::Prefill);
    CHECK(launches[1].rows_per_tile == 32);
    // Decode: 3,072 rows, 8 a workgroup. Prefill: 512 tokens in 16 tiles, 48
    // output tiles of 64.
    const auto workgroups = [](const kernels::Launch& l, std::uint32_t tokens) {
        return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.regime},
                                       l.workgroup_size, 0, tokens);
    };
    CHECK(workgroups(launches[0], 1) == 384);
    CHECK(workgroups(launches[0], 512) == 0);
    CHECK(workgroups(launches[1], 512) == 16 * 48);
    CHECK(workgroups(launches[1], 1) == 0);

    const auto head = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::LastToken});
    REQUIRE(head.size() == 1);
    CHECK(head[0].entry_point == "decode_write");
    CHECK_FALSE(head[0].regime.has_value());
    CHECK(workgroups(head[0], 1) == 384);
    CHECK(workgroups(head[0], 512) == 384);
}

TEST_CASE("a fused group binds the span of its members and reads each from its own first word") {
    // K at 0, something else between, Q, then V: file order, not output order.
    const auto k = q4_0(1024, 1024, 0), q = q4_0(1024, 2048, 1024 * 576 + 4096), v = q4_0(1024, 1024, 3 * 1024 * 576 + 4096);
    const auto launches = kernels::matmul_launches(
        {{&q, &k, &v}, 3, kIn, {kQ, kK, kV}, Epilogue::QKV, model::FeedForwardActivation::SiLU,
         kernels::Rows::EveryToken});
    REQUIRE(launches.size() == 2);
    CHECK(launches[0].entry_point == "decode_qkv");
    REQUIRE(launches[0].bindings.size() == 5);   // weights, input, query, key, value
    CHECK(launches[0].bindings[0].offset == 0);
    CHECK(launches[0].bindings[0].size == 4 * 1024 * 576 + 4096);
    const auto m = members_of(launches[0]);
    // Q: its first word, 2,048 rows of 32 blocks, output rows from 0.
    CHECK(m[0] == (1024 * 576 + 4096) / 4);
    CHECK(m[1] == 2048 * 32);
    CHECK(m[2] == 0);
    CHECK(m[3] == 2048);
    // K, from word 0, output rows from 2,048; V after both.
    CHECK(m[4] == 0);
    CHECK(m[6] == 2048);
    CHECK(m[7] == 1024);
    CHECK(m[8] == (3 * 1024 * 576 + 4096) / 4);
    CHECK(m[10] == 3072);
}

TEST_CASE("gate and up run a decode workgroup a 4 pairs and a prefill tile a 32") {
    const auto gate = q4_0(1024, 3072, 0), up = q4_0(1024, 3072, 3072 * 576);
    const auto launches = kernels::matmul_launches(
        {{&gate, &up, nullptr}, 2, kIn, {kQ, kK, kV}, Epilogue::GatedActivation,
         model::FeedForwardActivation::GeluTanh, kernels::Rows::EveryToken});
    REQUIRE(launches.size() == 2);
    CHECK(launches[0].entry_point == "decode_gated");
    REQUIRE(launches[0].bindings.size() == 3);   // weights, input, activation
    const auto workgroups = [](const kernels::Launch& l, std::uint32_t tokens) {
        return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.regime},
                                       l.workgroup_size, 0, tokens);
    };
    CHECK(workgroups(launches[0], 1) == 768);
    CHECK(workgroups(launches[1], 512) == 16 * 96);
    bool gelu = false;
    for (const auto& o : launches[0].overrides) {
        if (o.name == "activation") gelu = o.value == 1.0;
    }
    CHECK(gelu);
}
