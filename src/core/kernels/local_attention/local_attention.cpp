#include "core/kernels/local_attention/local_attention.h"

#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

// Binding 1, as local_attention.wgsl's LocalAttention lays it out.
struct Constants {
    float q_scale;
    float k_scale;
    float softcap;
    std::uint32_t window;
};

// One workgroup a row: 8 heads of 16 lanes.
constexpr std::uint32_t kWorkgroupSize = 128;

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

Launch local_attention_launch(const LocalAttentionLaunch& a) {
    const Constants constants{a.q_scale, a.k_scale, a.softcap, a.window};
    std::vector<std::byte> bytes(sizeof constants);
    std::memcpy(bytes.data(), &constants, sizeof constants);
    return Launch{shaders::local_attention,
                  nullptr,
                  std::move(bytes),
                  {whole(*a.per_dim_scale), whole(a.relative), whole(a.query), whole(a.key), whole(a.value),
                   whole(a.attention)},
                  kWorkgroupSize,
                  kWorkgroupSize,
                  Rows::EveryToken,
                  {}};
}

}  // namespace bllm::kernels
