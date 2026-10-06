#pragma once

#include "bllm/shaders_generated.h"
#include "core/formats/device_layout.h"
#include "core/formats/format.h"

namespace bllm::formats {

// Axis B: changes with a new weight format.
//
// F16: one half-precision value a block, two bytes; the KV cache's storage
// at load policy's default precision, and the weights a file stores at half
// precision. Its unpack is f16.wgsl and its pack f16_pack.wgsl; its layout,
// one stream, is kF16Layout (device_layout.h).
inline constexpr Format kF16{kF16Layout, shaders::f16, shaders::f16_pack};

}  // namespace bllm::formats
