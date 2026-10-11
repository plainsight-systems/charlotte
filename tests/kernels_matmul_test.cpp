// The matmul launcher, on the CPU (kernels/matmul/matmul.h): its launches
// and their token ranges, entry points, geometry, and a fused group's spanning
// binding and members.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>
#include <utility>

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

// A gate that never splits, so a test of the tiled form sees it alone; one
// that splits every product it may, up to its 1,024-workgroup cap.
constexpr kernels::SplitGate kNever{0, 1, 0};
constexpr kernels::SplitGate kAlways{1024, 1, 1024};
const residency::BufferRange kSums{residency::BufferIndex{7}, 0, 1 << 26};

std::array<std::uint32_t, 12> members_of(const kernels::Launch& l) {
    std::array<std::uint32_t, 12> m{};
    std::memcpy(m.data(), l.constants.data(), sizeof m);
    return m;
}

}  // namespace

TEST_CASE("a product is a decode launch and a prefill launch a tile width; the head one launch in every step") {
    const auto w = q4_0(1024, 3072, 0);
    const auto launches = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::EveryToken},
        kNever, kSums);
    REQUIRE(launches.size() == 4);
    CHECK(launches[0].entry_point == "decode_write");
    CHECK(launches[0].tokens == kernels::tokens_of(Regime::Decode));
    // The narrowest tile that holds the step whole, or the widest.
    const std::array<std::pair<std::uint32_t, kernels::TokenRange>, 3> tiles{
        {{8, {2, 8}}, {16, {9, 16}}, {32, {17, UINT32_MAX}}}};
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        CAPTURE(i);
        CHECK(launches[i + 1].entry_point == "prefill_write");
        CHECK(launches[i + 1].rows_per_tile == tiles[i].first);
        CHECK(launches[i + 1].tokens == tiles[i].second);
        // A 4-token × 4-output micro-tile an invocation: 4 a tile's token.
        CHECK(launches[i + 1].workgroup_size == 4 * tiles[i].first);
    }
    // Decode: 3,072 rows, 3 a set, 6 a workgroup. Prefill: 512 tokens in 16 tiles of
    // 32, 9 tokens in one of 16, 8 in one of 8; 48 output tiles of 64.
    const auto workgroups = [](const kernels::Launch& l, std::uint32_t tokens) {
        return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                       l.workgroup_size, 0, tokens, true);
    };
    CHECK(workgroups(launches[0], 1) == 512);
    CHECK(workgroups(launches[0], 512) == 0);
    CHECK(workgroups(launches[1], 8) == 48);
    CHECK(workgroups(launches[1], 9) == 0);
    CHECK(workgroups(launches[2], 9) == 48);
    CHECK(workgroups(launches[2], 17) == 0);
    CHECK(workgroups(launches[3], 512) == 16 * 48);
    CHECK(workgroups(launches[3], 16) == 0);
    CHECK(workgroups(launches[3], 1) == 0);

    const auto head = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::LastToken},
        kNever, kSums);
    REQUIRE(head.size() == 1);
    CHECK(head[0].entry_point == "decode_write");
    CHECK(head[0].tokens == kernels::TokenRange{});
    CHECK(workgroups(head[0], 1) == 512);
    CHECK(workgroups(head[0], 512) == 512);
}

TEST_CASE("a decode set takes the most rows, of 4 down to 1, that still makes 512 workgroups") {
    const auto set_rows = [](const kernels::Launch& l) {
        for (const auto& o : l.overrides) {
            if (o.name == "set_rows") return static_cast<std::uint32_t>(o.value);
        }
        return 0u;
    };
    // Rows, the set's rows, and the workgroups: Qwen3's output and down,
    // Llama 3.2's output and down and its QKV, Qwen3's QKV; the edges, 4,095
    // still 512 workgroups of 8 and 4,088 not, 2,047 still 512 of 4; and a
    // product too small for 512 at any.
    const std::array<std::array<std::uint32_t, 3>, 8> cases{{{1024, 1, 512},
                                                             {2048, 2, 512},
                                                             {3072, 3, 512},
                                                             {4096, 4, 512},
                                                             {4095, 4, 512},
                                                             {4088, 3, 682},
                                                             {2047, 2, 512},
                                                             {1000, 1, 500}}};
    for (const auto& [rows, want_rows, want_workgroups] : cases) {
        CAPTURE(rows);
        const auto w = q4_0(1024, rows, 0);
        const auto l = kernels::matmul_launches({{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write,
                                                 model::FeedForwardActivation::SiLU, kernels::Rows::EveryToken},
                                                kNever, kSums)[0];
        CHECK(set_rows(l) == want_rows);
        CHECK(kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                      l.workgroup_size, 0, 1, true) == want_workgroups);
    }
}

TEST_CASE("a fused group binds the span of its members and reads each from its own first word") {
    // K at 0, something else between, Q, then V: file order, not output order.
    const auto k = q4_0(1024, 1024, 0), q = q4_0(1024, 2048, 1024 * 576 + 4096), v = q4_0(1024, 1024, 3 * 1024 * 576 + 4096);
    const auto launches = kernels::matmul_launches(
        {{&q, &k, &v}, 3, kIn, {kQ, kK, kV}, Epilogue::QKV, model::FeedForwardActivation::SiLU,
         kernels::Rows::EveryToken},
        kNever, kSums);
    REQUIRE(launches.size() == 4);
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
         model::FeedForwardActivation::GeluTanh, kernels::Rows::EveryToken},
        kAlways, kSums);
    // Gate and up never split, whatever the gate (matmul.h).
    REQUIRE(launches.size() == 4);
    CHECK(launches[0].entry_point == "decode_gated");
    REQUIRE(launches[0].bindings.size() == 3);   // weights, input, activation
    const auto workgroups = [](const kernels::Launch& l, std::uint32_t tokens) {
        return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                       l.workgroup_size, 0, tokens, true);
    };
    CHECK(workgroups(launches[0], 1) == 768);
    CHECK(workgroups(launches[3], 512) == 16 * 96);
    bool gelu = false;
    for (const auto& o : launches[0].overrides) {
        if (o.name == "activation") gelu = o.value == 1.0;
    }
    CHECK(gelu);
}

TEST_CASE("a product gated by an input binds the input after its output and carries its stride and column") {
    const auto w = q4_0(512, 512, 0);
    const residency::BufferRange u{residency::BufferIndex{9}, 2048, 1 << 20};
    kernels::MatmulLaunch m{{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::GatedByInput,
                            model::FeedForwardActivation::GeluTanh, kernels::Rows::EveryToken};
    m.gate_input = u;
    m.gate_input_stride = 12288;
    m.gate_input_column = 5 * 512;
    const auto launches = kernels::matmul_launches(m, kNever, kSums);
    REQUIRE(launches.size() == 4);
    CHECK(launches[0].entry_point == "decode_gated_input");
    for (std::size_t i = 1; i < launches.size(); ++i) CHECK(launches[i].entry_point == "prefill_gated_input");
    for (const kernels::Launch& l : launches) {
        REQUIRE(l.bindings.size() == 4);
        CHECK(l.bindings[2].buffer == kQ.buffer);
        CHECK(l.bindings[3].buffer == u.buffer);
        CHECK(l.bindings[3].offset == u.offset);
        REQUIRE(l.constants.size() == 64);
        std::uint32_t gate[2];
        std::memcpy(gate, l.constants.data() + 48, sizeof gate);
        CHECK(gate[0] == 12288);
        CHECK(gate[1] == 5 * 512);
    }
}

namespace {

const auto workgroups_at = [](const kernels::Launch& l, std::uint32_t tokens) {
    return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                   l.workgroup_size, 0, tokens, true);
};

}  // namespace

TEST_CASE("split_tokens at each listed model's products, each need within the range sums' bound") {
    struct Case {
        const char* name;
        Epilogue epilogue;
        std::uint64_t rows, columns;
        std::uint32_t want;
    };
    // W = output tiles × token tiles splits while W ≤ 2G and W ≤ 128.
    const std::array<Case, 15> cases{{
        {"EmbeddingGemma 2 down", Epilogue::Write, 512, 2048, 512},
        {"EmbeddingGemma 2 sliding output", Epilogue::Write, 512, 1024, 256},
        {"EmbeddingGemma 2 per-layer gate", Epilogue::GatedByInput, 512, 512, 128},
        {"EmbeddingGemma 2 sliding QKV", Epilogue::QKV, 2048, 512, 32},
        {"EmbeddingGemma 2 global QKV", Epilogue::QKV, 3072, 512, 1},
        {"EmbeddingGemma 2 gate and up", Epilogue::GatedActivation, 4096, 512, 1},
        {"Qwen3 down", Epilogue::Write, 1024, 3072, 256},
        {"Qwen3 output", Epilogue::Write, 1024, 2048, 256},
        {"Qwen3 QKV", Epilogue::QKV, 4096, 1024, 32},
        {"Llama 3.2 down", Epilogue::Write, 2048, 8192, 128},
        {"Llama 3.2 QKV", Epilogue::QKV, 3072, 2048, 64},
        {"Llama 3.2 output", Epilogue::Write, 2048, 2048, 128},
        {"Gemma 3 QKV", Epilogue::QKV, 1536, 1152, 96},
        {"Gemma 3 output", Epilogue::Write, 1152, 1024, 96},
        {"Gemma 3 down", Epilogue::Write, 1152, 6912, 224},
    }};
    for (const Case& c : cases) {
        CAPTURE(c.name);
        const std::uint32_t split = kernels::split_tokens(c.epilogue, c.rows, c.columns, kernels::kSplitGate);
        CHECK(split == c.want);
        // Its range sums at its last split step, within the cap's 32 MiB.
        const std::uint64_t need = kernels::range_sum_floats(c.epilogue, c.rows, c.columns, kernels::kSplitGate);
        CHECK(need == (split < 2 ? 0 : std::uint64_t{split} * c.rows * 32));
        CHECK(need * 4 <= 32ull << 20);
    }
}

TEST_CASE("a step splits only while its range sums fit the buffer, and runs tiled past it") {
    const auto w = q4_0(1024, 512, 0);   // splits to 256 tokens under kSplitGate
    // Room for 100 tokens of 512 rows' 32 range sums.
    const residency::BufferRange small{residency::BufferIndex{7}, 0, 100 * 512 * 32 * 4};
    const auto launches = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::EveryToken},
        kernels::kSplitGate, small);
    const auto runs = [&](std::string_view entry, std::uint32_t tokens) {
        return std::any_of(launches.begin(), launches.end(), [&](const kernels::Launch& l) {
            return l.entry_point == entry && workgroups_at(l, tokens) > 0;
        });
    };
    CHECK(runs("split_ranges", 100));
    CHECK(runs("combine_write", 100));
    CHECK_FALSE(runs("split_ranges", 101));
    CHECK(runs("prefill_write", 101));
    // No buffer at all: never split.
    const auto none = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::EveryToken},
        kernels::kSplitGate, {});
    CHECK(none.size() == 4);
}

TEST_CASE("the gate splits on its boundary and not one workgroup past it") {
    // 512 rows of 1,024: 8 output tiles, G = 32. At one token tile W = 8,
    // so W × 4 ≤ G × 1 holds exactly, and W × 5 ≤ G × 1 does not.
    CHECK(kernels::split_tokens(Epilogue::Write, 512, 1024, {1, 4, 128}) == 32);
    CHECK(kernels::split_tokens(Epilogue::Write, 512, 1024, {1, 5, 128}) == 1);
    // The cap alike: W = 8 under a cap of 8, not 7; two token tiles at 16.
    CHECK(kernels::split_tokens(Epilogue::Write, 512, 1024, {2, 1, 8}) == 32);
    CHECK(kernels::split_tokens(Epilogue::Write, 512, 1024, {2, 1, 7}) == 1);
    CHECK(kernels::split_tokens(Epilogue::Write, 512, 1024, {2, 1, 16}) == 64);
    // Never a whole prefill block past kPrefillBlock, nor a gated activation.
    CHECK(kernels::split_tokens(Epilogue::Write, 64, 8192, kAlways) == residency::kPrefillBlock);
    CHECK(kernels::split_tokens(Epilogue::GatedActivation, 64, 8192, kAlways) == 1);
    CHECK(kernels::split_tokens(Epilogue::Write, 64, 8192, kNever) == 1);
}

TEST_CASE("each step runs one prefill form, the combine with the split, whatever the gate") {
    const auto w = q4_0(1024, 512, 0);
    for (const kernels::SplitGate gate : {kNever, kAlways, kernels::kSplitGate, kernels::SplitGate{1, 4, 128}}) {
        CAPTURE(gate.numerator);
        CAPTURE(gate.denominator);
        const auto launches = kernels::matmul_launches(
            {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
             kernels::Rows::EveryToken},
            gate, kSums);
        const std::uint32_t split = kernels::split_tokens(Epilogue::Write, 512, 1024, gate);
        for (std::uint32_t tokens = 1; tokens <= residency::kPrefillBlock; ++tokens) {
            int decode = 0, tiled = 0, ranges = 0, combine = 0;
            for (const kernels::Launch& l : launches) {
                if (workgroups_at(l, tokens) == 0) continue;
                decode += l.entry_point == "decode_write";
                tiled += l.entry_point == "prefill_write";
                ranges += l.entry_point == "split_ranges";
                combine += l.entry_point == "combine_write";
            }
            const bool splits = tokens >= 2 && tokens <= split;
            if ((decode != (tokens == 1)) || (tiled != (tokens >= 2 && !splits)) || (ranges != splits) ||
                (combine != splits)) {
                FAIL_CHECK("tokens " << tokens << ": " << decode << tiled << ranges << combine);
            }
        }
    }
}

TEST_CASE("a split launch is the tiled one's workgroups 32 times, writing range sums; its combine reads them") {
    const auto w = q4_0(512, 512, 0);
    const residency::BufferRange u{residency::BufferIndex{9}, 2048, 1 << 20};
    kernels::MatmulLaunch m{{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::GatedByInput,
                            model::FeedForwardActivation::GeluTanh, kernels::Rows::EveryToken};
    m.gate_input = u;
    const auto launches = kernels::matmul_launches(m, kernels::kSplitGate, kSums);
    const auto find = [&](std::string_view entry, std::uint32_t tokens) -> const kernels::Launch* {
        for (const kernels::Launch& l : launches) {
            if (l.entry_point == entry && workgroups_at(l, tokens) > 0) return &l;
        }
        return nullptr;
    };
    // 8 output tiles, one token tile at 21 tokens: 8 tiled, 256 split.
    const kernels::Launch* ranges = find("split_ranges", 21);
    REQUIRE(ranges != nullptr);
    CHECK(workgroups_at(*ranges, 21) == 8 * 32);
    REQUIRE(ranges->bindings.size() == 3);   // weights, input, range sums
    CHECK(ranges->bindings[1].buffer == kIn.buffer);
    CHECK(ranges->bindings[2].buffer == kSums.buffer);
    const kernels::Launch* combine = find("combine_gated_input", 21);
    REQUIRE(combine != nullptr);
    CHECK(workgroups_at(*combine, 21) == (21 * 512 + 63) / 64);
    REQUIRE(combine->bindings.size() == 3);   // range sums, output, u
    CHECK(combine->bindings[0].buffer == kSums.buffer);
    CHECK(combine->bindings[1].buffer == kQ.buffer);
    CHECK(combine->bindings[2].buffer == u.buffer);
    // Past the product's split steps, 128, the tiled form alone.
    CHECK(find("split_ranges", 129) == nullptr);
    CHECK(find("combine_gated_input", 129) == nullptr);
    CHECK(find("prefill_gated_input", 129) != nullptr);
    // The head never splits.
    const auto head = kernels::matmul_launches(
        {{&w, nullptr, nullptr}, 1, kIn, {kQ, kK, kV}, Epilogue::Write, model::FeedForwardActivation::SiLU,
         kernels::Rows::LastToken},
        kAlways, kSums);
    CHECK(head.size() == 1);
}
