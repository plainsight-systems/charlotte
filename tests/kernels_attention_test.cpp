// The attention launcher, on the CPU: its two launches' geometry, bindings,
// constants and variant (kernels/attention/attention.h).

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <numbers>
#include <string_view>

#include "bllm/shaders_generated.h"
#include "core/formats/f16/f16.h"
#include "core/kernels/attention/attention.h"

using namespace bllm;

namespace {

double override_of(const kernels::Launch& l, std::string_view name) {
    for (const auto& o : l.overrides) {
        if (o.name == name) return o.value;
    }
    FAIL("no override " << name);
    return -1;
}

residency::BufferRange range(std::uint32_t buffer) { return {residency::BufferIndex{buffer}, 0, 4096}; }

}  // namespace

TEST_CASE("attention launches its tiles split by chunk, and a combine that runs only when split") {
    struct Case {
        std::uint32_t query_heads, key_value_heads, d, rows_per_tile;
    };
    for (const Case& c : {Case{16, 8, 128, 4}, Case{32, 8, 64, 4}, Case{4, 1, 256, 1}}) {
        CAPTURE(c.d);
        const model::LayerDescription layer{c.query_heads, c.key_value_heads, c.d, 0, 512, 1e6f, {}};
        const residency::PlannedCacheLayer cache{range(6), {residency::BufferIndex{6}, 8192, 4096}, 600};
        const float scale = 1.0f / std::sqrt(static_cast<float>(c.d));
        const auto [attend, combine] = kernels::attention_launches(
            {layer, scale, range(2), cache, &formats::kF16, range(3), range(4), range(5), 640});

        CHECK(attend.kernel == shaders::attention);
        CHECK(attend.entry_point == "main");
        CHECK(attend.pack_format == &formats::kF16);
        CHECK(attend.rows_per_tile == c.rows_per_tile);
        CHECK(attend.key_split == kernels::KeySplit::PerChunk);
        CHECK(attend.window == 512);
        // A tile's workgroups: one a key-value head.
        CHECK(attend.invocations_per_row * attend.rows_per_tile == c.key_value_heads * attend.workgroup_size);
        REQUIRE(attend.bindings.size() == 6);
        CHECK(attend.bindings[0].buffer == residency::BufferIndex{2});
        CHECK(attend.bindings[1].offset == 0);
        CHECK(attend.bindings[2].offset == 8192);
        CHECK(attend.bindings[3].buffer == residency::BufferIndex{3});
        CHECK(attend.bindings[4].buffer == residency::BufferIndex{4});
        CHECK(attend.bindings[5].buffer == residency::BufferIndex{5});
        CHECK(override_of(attend, "head_dimension") == c.d);
        CHECK(override_of(attend, "query_heads") == c.query_heads);
        CHECK(override_of(attend, "key_value_heads") == c.key_value_heads);
        // The partial buffers' rows, to the kernel and to the dispatch alike.
        CHECK(override_of(attend, "partial_rows") == 640);
        CHECK(attend.partial_rows == 640);
        CHECK(combine.partial_rows == 640);
        std::uint32_t slots = 0, window = 0;
        float scale_log2e = 0;
        std::memcpy(&slots, attend.constants.data(), 4);
        std::memcpy(&window, attend.constants.data() + 4, 4);
        std::memcpy(&scale_log2e, attend.constants.data() + 8, 4);
        CHECK(slots == 600);
        CHECK(window == 512);
        CHECK(scale_log2e == static_cast<float>(scale * std::numbers::log2e));

        CHECK(combine.entry_point == "combine");
        CHECK(combine.key_split == kernels::KeySplit::WhenSplit);
        CHECK(combine.rows_per_tile == 0);
        CHECK(combine.invocations_per_row == c.query_heads * c.d / 4);
        REQUIRE(combine.bindings.size() == 3);
        CHECK(combine.bindings[0].buffer == residency::BufferIndex{4});
        CHECK(combine.bindings[1].buffer == residency::BufferIndex{5});
        CHECK(combine.bindings[2].buffer == residency::BufferIndex{3});
    }
}
