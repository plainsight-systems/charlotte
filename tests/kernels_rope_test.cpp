// The rope launcher, on the CPU: its frequencies, bindings, geometry and
// variant (kernels/rope/rope.h).

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <numbers>
#include <string_view>

#include "bllm/shaders_generated.h"
#include "core/formats/f16/f16.h"
#include "core/kernels/rope/rope.h"

using namespace bllm;

namespace {

residency::WeightView vector(std::uint32_t buffer, std::uint64_t offset, std::uint64_t width) {
    gguf::TensorShape shape{};
    shape.dimension_count = 1;
    shape.dimensions[0] = width;
    shape.element_count = width;
    return residency::WeightView(gguf::TensorType::F32, shape,
                                 {{residency::BufferIndex{buffer}, offset, width * 4, 0, 1}});
}

double override_of(const kernels::Launch& l, std::string_view name) {
    for (const auto& o : l.overrides) {
        if (o.name == name) return o.value;
    }
    FAIL("no override " << name);
    return -1;
}

model::LayerDescription layer(std::uint32_t query_heads, std::uint32_t key_value_heads, std::uint32_t d,
                              float base) {
    return {query_heads, key_value_heads, d, 3072, 40960, base, {}};
}

const residency::BufferRange kQuery{residency::BufferIndex{3}, 0, 100}, kKey{residency::BufferIndex{4}, 0, 200},
    kValue{residency::BufferIndex{5}, 0, 300};
const residency::PlannedCacheLayer kCache{{residency::BufferIndex{6}, 512, 4096}, {residency::BufferIndex{6}, 4608, 4096},
                                          4608};

}  // namespace

TEST_CASE("each pair's turns a position are its f64 frequency over 2 pi, rounded once") {
    for (const float base : {1e6f, 5e5f, 1e4f}) {
        for (const std::uint32_t d : {64u, 128u, 256u}) {
            CAPTURE(base);
            CAPTURE(d);
            const auto l = kernels::rope_launch({layer(4, 1, d, base), model::RotaryPairing::Halves, nullptr, nullptr,
                                                 nullptr, 1e-6f, kQuery, kKey, kValue, kCache, &formats::kF16});
            REQUIRE(l.constants.size() == 528);
            float turns[128];
            std::memcpy(turns, l.constants.data() + 16, sizeof turns);
            for (std::uint32_t k = 0; k < 128; ++k) {
                if (k >= d / 2) {
                    CHECK(turns[k] == 0.0f);
                    continue;
                }
                const double exact = std::pow(static_cast<double>(base), -2.0 * k / d) / (2 * std::numbers::pi);
                // Within half a unit in f32's last place of the f64 value.
                CHECK(std::abs(turns[k] - exact) <= std::ldexp(std::abs(exact), -24));
            }
        }
    }
}

TEST_CASE("a rope launch binds its norms, factors, buffers and cache, and selects its variant") {
    const auto query_norm = vector(0, 256, 128), key_norm = vector(0, 1024, 128), factors = vector(0, 2048, 64);
    const auto qwen = kernels::rope_launch({layer(16, 8, 128, 1e6f), model::RotaryPairing::Halves, &query_norm,
                                            &key_norm, nullptr, 1e-6f, kQuery, kKey, kValue, kCache, &formats::kF16});
    CHECK(qwen.kernel == shaders::rope);
    CHECK(qwen.format == nullptr);
    CHECK(qwen.pack_format == &formats::kF16);
    REQUIRE(qwen.bindings.size() == 8);
    CHECK(qwen.bindings[0].offset == 256);
    CHECK(qwen.bindings[1].offset == 1024);
    CHECK(qwen.bindings[2].buffer == kKey.buffer);   // no factors: the key buffer, read-only
    CHECK(qwen.bindings[3].buffer == kQuery.buffer);
    CHECK(qwen.bindings[4].buffer == kKey.buffer);
    CHECK(qwen.bindings[5].buffer == kValue.buffer);
    CHECK(qwen.bindings[6].offset == 512);
    CHECK(qwen.bindings[7].offset == 4608);
    CHECK(qwen.invocations_per_row == 512);   // (16 + 2 × 8) heads × 16
    CHECK(qwen.workgroup_size == 64);
    CHECK(qwen.invocations_per_row % qwen.workgroup_size == 0);
    CHECK(override_of(qwen, "head_dimension") == 128);
    CHECK(override_of(qwen, "query_heads") == 16);
    CHECK(override_of(qwen, "key_value_heads") == 8);
    CHECK(override_of(qwen, "halves") == 1.0);
    CHECK(override_of(qwen, "qk_norm") == 1.0);
    CHECK(override_of(qwen, "factors") == 0.0);
    std::uint32_t slots = 0;
    float epsilon = 0;
    std::memcpy(&slots, qwen.constants.data(), 4);
    std::memcpy(&epsilon, qwen.constants.data() + 4, 4);
    CHECK(slots == 4608);
    CHECK(epsilon == 1e-6f);

    const auto llama = kernels::rope_launch({layer(32, 8, 64, 5e5f), model::RotaryPairing::Adjacent, nullptr, nullptr,
                                             &factors, 1e-5f, kQuery, kKey, kValue, kCache, &formats::kF16});
    CHECK(llama.bindings[0].buffer == kKey.buffer);   // no QK-norm
    CHECK(llama.bindings[1].buffer == kKey.buffer);
    CHECK(llama.bindings[2].offset == 2048);
    CHECK(llama.invocations_per_row == 384);   // (32 + 2 × 8) heads × 8
    CHECK(override_of(llama, "halves") == 0.0);
    CHECK(override_of(llama, "qk_norm") == 0.0);
    CHECK(override_of(llama, "factors") == 1.0);
}
