#include "core/kernels/pool/pool.h"

#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 256;

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

// One workgroup a step: a launch over the last token, one row's
// invocations (interface.h); its constants, the row's vec4s.
// Binding 1, as pool.wgsl's Pool lays it out; mean and unit read vec4s
// alone.
struct Constants {
    std::uint32_t vec4s;
    std::uint32_t pooling = 0;
    float epsilon = 0;
};

std::vector<std::byte> bytes_of(const Constants& c) {
    std::vector<std::byte> out(sizeof c);
    std::memcpy(out.data(), &c, sizeof c);
    return out;
}

Launch one_workgroup(std::string_view entry_point, const residency::BufferRange& in,
                     const residency::BufferRange& out, std::uint32_t width) {
    std::vector<std::byte> constants = bytes_of({width / 4});
    Launch launch{shaders::pool,
                  nullptr,
                  std::move(constants),
                  {whole(in), whole(out)},
                  kWorkgroupSize,
                  kWorkgroupSize,
                  Rows::LastToken,
                  {}};
    launch.entry_point = entry_point;
    return launch;
}

}  // namespace

Launch mean_launch(const residency::BufferRange& normed, const residency::BufferRange& pooled, std::uint32_t width) {
    return one_workgroup("mean", normed, pooled, width);
}

Launch unit_launch(const residency::BufferRange& projected, const residency::BufferRange& embedding,
                   std::uint32_t dimensions) {
    return one_workgroup("unit", projected, embedding, dimensions);
}

Launch grid_launch(const residency::BufferRange& patches, const residency::BufferRange& pooled, std::uint32_t width,
                   std::uint32_t pooling, float epsilon) {
    Launch launch{shaders::pool,
                  nullptr,
                  bytes_of({width / 4, pooling, epsilon}),
                  {whole(patches), whole(pooled)},
                  kWorkgroupSize,
                  kWorkgroupSize,
                  Rows::EveryToken,
                  {}};
    launch.entry_point = "grid";
    launch.row_divisor = pooling * pooling;
    return launch;
}

}  // namespace bllm::kernels
