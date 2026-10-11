#include "core/kernels/pool/pool.h"

#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 256;

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

// One workgroup a step: a launch over the last token, one row's
// invocations (interface.h); its constants, the row's vec4s.
Launch one_workgroup(std::string_view entry_point, const residency::BufferRange& in,
                     const residency::BufferRange& out, std::uint32_t width) {
    const std::uint32_t vec4s = width / 4;
    std::vector<std::byte> constants(sizeof vec4s);
    std::memcpy(constants.data(), &vec4s, sizeof vec4s);
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

}  // namespace bllm::kernels
