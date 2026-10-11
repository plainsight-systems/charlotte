#include "core/kernels/block_attention/block_attention.h"

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 128;
constexpr std::uint32_t kVectors = 32;   // query vectors a tile (block_attention.wgsl)
constexpr std::uint32_t kDims = 64;      // output dimensions a weigh workgroup

// Binding 1, as block_attention.wgsl's BlockAttention lays it out.
struct Constants {
    std::uint32_t half_window;
    float scale_log2e;
};

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

std::array<Launch, 2> block_attention_launches(const BlockAttentionLaunch& a) {
    const model::LayerDescription& layer = a.layer;
    const std::uint32_t d = layer.head_dimension;
    const std::uint32_t group = layer.query_heads / layer.key_value_heads;
    const std::uint32_t rows_per_tile = kVectors / group;   // a tile's rows: 32 vectors of G heads each

    // The scale in log2 units, so the kernel's exp2 is the softmax's exp; the
    // window, half either side (block_attention.h).
    const Constants constants{layer.attention_window / 2, static_cast<float>(a.scale * std::numbers::log2e)};
    const auto bytes = std::as_bytes(std::span(&constants, 1));
    // Each a plain override, set on every launch, as WebKit requires.
    const std::vector<Override> shape{{"head_dimension", static_cast<double>(d)},
                                      {"query_heads", static_cast<double>(layer.query_heads)},
                                      {"key_value_heads", static_cast<double>(layer.key_value_heads)},
                                      {"group", static_cast<double>(group)}};

    // scores: a workgroup a key-value head, a vector tile and a 64-key chunk.
    // A tile is rows_per_tile rows; its workgroups one a key-value head; and
    // the chunks multiply them, as the step's key chunks do for a split: at
    // position 0, with no window and no limit on the partial rows, the
    // step's ceil(N / 64) (kernels/interface.h, key_chunks).
    Launch scores{shaders::block_attention,
                  nullptr,
                  std::vector<std::byte>(bytes.begin(), bytes.end()),
                  {whole(a.query), whole(a.key), whole(a.scores), whole(a.score_maxima)},
                  layer.key_value_heads * kWorkgroupSize / rows_per_tile,
                  kWorkgroupSize,
                  Rows::EveryToken,
                  shape};
    scores.entry_point = "scores_of";
    scores.rows_per_tile = rows_per_tile;
    scores.key_split = KeySplit::PerChunk;
    scores.window = kMaxPositions;
    scores.partial_rows = UINT32_MAX;

    // weigh: a workgroup a key-value head, a vector tile and 64 of d.
    Launch weigh{shaders::block_attention,
                 nullptr,
                 std::vector<std::byte>(bytes.begin(), bytes.end()),
                 {whole(a.scores), whole(a.score_maxima), whole(a.value), whole(a.attention)},
                 layer.key_value_heads * (d / kDims) * kWorkgroupSize / rows_per_tile,
                 kWorkgroupSize,
                 Rows::EveryToken,
                 shape};
    weigh.entry_point = "weigh";
    weigh.rows_per_tile = rows_per_tile;
    return {std::move(scores), std::move(weigh)};
}

}  // namespace bllm::kernels
