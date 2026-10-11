#include "core/kernels/light_conv/light_conv.h"

#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

// Binding 1, as light_conv.wgsl's LightConv lays it out.
struct Constants {
    float epsilon;
    std::uint32_t kernel;
};

// One workgroup a row, as norm.cpp's.
constexpr std::uint32_t kWorkgroupSize = 256;

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

Launch light_conv_launch(const residency::WeightView& depthwise, const residency::WeightView& gain,
                         const residency::BufferRange& input, const residency::BufferRange& output,
                         std::uint32_t width, std::uint32_t kernel, float epsilon) {
    const Constants constants{epsilon, kernel};
    std::vector<std::byte> bytes(sizeof constants);
    std::memcpy(bytes.data(), &constants, sizeof constants);
    return Launch{shaders::light_conv,
                  nullptr,
                  std::move(bytes),
                  {whole(depthwise), whole(gain), whole(input), whole(output)},
                  kWorkgroupSize,
                  kWorkgroupSize,
                  Rows::EveryToken,
                  {{"width", static_cast<double>(width)}}};
}

}  // namespace bllm::kernels
