#include "core/kernels/norm/norm.h"

#include <cstdint>
#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

// Binding 1, as norm.wgsl's Norm lays it out.
struct Constants {
    float epsilon;
};

constexpr std::uint32_t kWorkgroupSize = 256;

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

Launch norm_launch(const NormLaunch& n) {
    const Constants constants{n.epsilon};
    std::vector<std::byte> bytes(sizeof constants);
    std::memcpy(bytes.data(), &constants, sizeof constants);
    const bool post = n.post_gain != nullptr;
    return Launch{
        shaders::norm,
        nullptr,
        std::move(bytes),
        // Without a post-norm its gain is the norm's own: read-only, so the
        // two bindings may share it.
        {whole(n.gain), whole(post ? *n.post_gain : n.gain), whole(n.output), whole(n.hidden), whole(n.normed)},
        kWorkgroupSize,
        kWorkgroupSize,
        n.rows,
        {{"add", n.add ? 1.0 : 0.0},
         {"post_norm", post ? 1.0 : 0.0},
         {"width", static_cast<double>(n.gain.shape().dimensions[0])}},
    };
}

}  // namespace bllm::kernels
