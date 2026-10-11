#include "core/kernels/norm/norm.h"

#include <cstdint>
#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

// Binding 1, as norm.wgsl's Norm lays it out.
struct Constants {
    float epsilon;
    float post_scale;
};

constexpr std::uint32_t kWorkgroupSize = 256;

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

Launch norm_launch(const NormLaunch& n) {
    const Constants constants{n.epsilon, n.post_scale};
    std::vector<std::byte> bytes(sizeof constants);
    std::memcpy(bytes.data(), &constants, sizeof constants);
    const bool post = n.post_gain != nullptr;
    const bool scaled = n.scale != nullptr;
    const bool renorm = n.renorm_gain != nullptr;
    const bool unit = n.gain == nullptr;
    // Every absent read-only binding takes a present one's place: the gain,
    // or with a unit gain the block's output, which the launch never writes.
    const Binding fill = unit ? whole(n.output) : whole(*n.gain);
    const std::uint64_t width = unit ? n.width : n.gain->shape().dimensions[0];
    return Launch{
        shaders::norm,
        nullptr,
        std::move(bytes),
        {fill, post ? whole(*n.post_gain) : fill, whole(n.output), whole(n.hidden), whole(n.normed),
         scaled ? whole(*n.scale) : fill, renorm ? whole(*n.renorm_gain) : fill},
        kWorkgroupSize * n.rows_per_token,
        kWorkgroupSize,
        n.rows,
        {{"add", n.add ? 1.0 : 0.0},
         {"post_norm", post ? 1.0 : 0.0},
         {"scaled", scaled ? 1.0 : 0.0},
         {"normalize", n.normalize ? 1.0 : 0.0},
         {"renorm", renorm ? 1.0 : 0.0},
         {"unit_gain", unit ? 1.0 : 0.0},
         {"rows_per_token", static_cast<double>(n.rows_per_token)},
         {"width", static_cast<double>(width)}},
    };
}

}  // namespace bllm::kernels
