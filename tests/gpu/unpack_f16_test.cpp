// F16's unpack on the GPU: every half comes back as its exact f32, bit for
// bit within format.h's contract.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/f16/f16.h"
#include "core/gpu/wgpu_handles.h"
#include "core/quant/q4_0.h"
#include "support/acquire.h"
#include "support/unpack.h"

using namespace bllm;

TEST_CASE("F16 unpack returns every half as its f32, -0 and subnormals included") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    // +-0, the smallest and largest subnormals, the smallest normal, the
    // largest finite of either sign, 1, then random finite values.
    const std::uint16_t edges[] = {0x0000u, 0x8000u, 0x0001u, 0x83FFu, 0x0400u, 0x7BFFu, 0xFBFFu, 0x3C00u};
    for (const std::uint32_t groups : {1u, 3u, 33u}) {
        CAPTURE(groups);
        std::vector<std::uint16_t> halves(std::size_t{groups} * 32);
        std::uint32_t state = 0x6C078965u;
        for (std::size_t i = 0; i < halves.size(); ++i) {
            state = state * 1664525u + 1013904223u;
            auto h = static_cast<std::uint16_t>(i < std::size(edges) ? edges[i] : state >> 16);
            if ((h & 0x7C00u) == 0x7C00u) h &= 0xBFFFu;   // never infinity or NaN
            halves[i] = h;
        }
        std::vector<float> want(halves.size());
        for (std::size_t i = 0; i < halves.size(); ++i) want[i] = quant::fp16_to_fp32(halves[i]);
        const auto got = testing::run_unpack(instance.get(), *device, formats::kF16.unpack_wgsl(),
                                             formats::kF16.layout(), std::as_bytes(std::span(halves)), groups);
        testing::check_bitwise(got, want);
    }
}
