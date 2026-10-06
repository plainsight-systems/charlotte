#include "core/kernels/attention/attention.h"

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;

// Binding 1, as attention.wgsl's Attention lays it out.
struct Constants {
    std::uint32_t slots;
    std::uint32_t window;
    float scale_log2e;
    std::uint32_t padding;
};

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

std::array<Launch, 2> attention_launches(const AttentionLaunch& a) {
    const model::LayerDescription& layer = a.layer;
    const std::uint32_t d = layer.head_dimension;
    const std::uint32_t group = layer.query_heads / layer.key_value_heads;
    const std::uint32_t rows_per_tile = 1024 / d / group;   // M / G

    // The scale in log2 units, so the kernel's exp2 is the softmax's exp.
    const Constants constants{a.cache.slots, layer.attention_window,
                              static_cast<float>(a.scale * std::numbers::log2e), 0};
    const auto bytes = std::as_bytes(std::span(&constants, 1));
    const std::vector<Override> shape{{"head_dimension", static_cast<double>(d)},
                                      {"query_heads", static_cast<double>(layer.query_heads)},
                                      {"key_value_heads", static_cast<double>(layer.key_value_heads)}};

    Launch attend{shaders::attention,
                  nullptr,
                  std::vector<std::byte>(bytes.begin(), bytes.end()),
                  {whole(a.query), whole(a.cache.keys), whole(a.cache.values), whole(a.attention), whole(a.partials),
                   whole(a.partial_stats)},
                  // A tile's workgroups are one a key-value head.
                  layer.key_value_heads * kWorkgroupSize / rows_per_tile,
                  kWorkgroupSize,
                  Rows::EveryToken,
                  shape,
                  a.cache_format};
    attend.rows_per_tile = rows_per_tile;
    attend.key_split = KeySplit::PerChunk;
    attend.window = layer.attention_window;

    Launch combine{shaders::attention,
                   nullptr,
                   std::vector<std::byte>(bytes.begin(), bytes.end()),
                   {whole(a.partials), whole(a.partial_stats), whole(a.attention)},
                   layer.query_heads * d / 4,   // a vec4 of every query's output
                   kWorkgroupSize,
                   Rows::EveryToken,
                   shape,
                   a.cache_format};
    combine.key_split = KeySplit::WhenSplit;
    combine.window = layer.attention_window;
    combine.entry_point = "combine";
    return {std::move(attend), std::move(combine)};
}

}  // namespace bllm::kernels
