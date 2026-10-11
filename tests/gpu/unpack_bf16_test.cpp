// BF16's unpack on the GPU: every bfloat16 comes back as its exact f32, bit
// for bit, as ggml_bf16_to_fp32 gives it (formats/bf16/bf16.h).

#include <doctest/doctest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/bf16/bf16.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/unpack.h"

using namespace bllm;

namespace {

// ggml_bf16_to_fp32: the 16 bits are the top of an f32.
float bf16_to_fp32(std::uint16_t h) { return std::bit_cast<float>(std::uint32_t{h} << 16); }

}  // namespace

TEST_CASE("BF16 unpack returns every bfloat16 as its f32, -0 and subnormals included") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    // +-0, the smallest and largest subnormals, the smallest normal, the
    // largest finite of either sign, 1, -2, then random finite values.
    const std::uint16_t edges[] = {0x0000u, 0x8000u, 0x0001u, 0x807Fu, 0x0080u, 0x7F7Fu, 0xFF7Fu, 0x3F80u, 0xC000u};
    for (const std::uint32_t groups : {1u, 3u, 33u}) {
        CAPTURE(groups);
        std::vector<std::uint16_t> values(std::size_t{groups} * 32);
        std::uint32_t state = 0x2545F491u;
        for (std::size_t i = 0; i < values.size(); ++i) {
            state = state * 1664525u + 1013904223u;
            auto h = static_cast<std::uint16_t>(i < std::size(edges) ? edges[i] : state >> 16);
            if ((h & 0x7F80u) == 0x7F80u) h &= 0xBFFFu;   // never infinity or NaN
            values[i] = h;
        }
        std::vector<float> want(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) want[i] = bf16_to_fp32(values[i]);
        const auto got = testing::run_unpack(instance.get(), *device, formats::kBF16.unpack_wgsl(),
                                             formats::kBF16.layout(), std::as_bytes(std::span(values)), groups);
        testing::check_bitwise(got, want);
    }
}
