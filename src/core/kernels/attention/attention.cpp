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
    // The tile's shape, derived here rather than in the kernel: WebKit
    // fails overrides initialized from overrides, and workgroup arrays
    // sized by expressions of them (attention.wgsl).
    const std::uint32_t queries_per_tile = 1024 / d;   // M
    const std::uint32_t keys_per_tile = 2048 / d;      // B
    const std::vector<Override> shape{{"head_dimension", static_cast<double>(d)},
                                      {"query_heads", static_cast<double>(layer.query_heads)},
                                      {"key_value_heads", static_cast<double>(layer.key_value_heads)},
                                      {"partial_rows", static_cast<double>(a.partial_rows)},
                                      {"queries_per_tile", static_cast<double>(queries_per_tile)},
                                      {"keys_per_tile", static_cast<double>(keys_per_tile)},
                                      {"group", static_cast<double>(group)},
                                      {"rows_per_tile", static_cast<double>(rows_per_tile)},
                                      {"lanes_per_query", static_cast<double>(kWorkgroupSize / queries_per_tile)},
                                      {"k_tile_words", static_cast<double>(keys_per_tile * (d / 2 + 1))},
                                      {"p_tile_floats", static_cast<double>(queries_per_tile * keys_per_tile)}};

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
    attend.partial_rows = a.partial_rows;

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
    combine.partial_rows = a.partial_rows;
    combine.entry_point = "combine";
    return {std::move(attend), std::move(combine)};
}

}  // namespace bllm::kernels
