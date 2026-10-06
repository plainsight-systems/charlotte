// F16's pack on the GPU (formats/format.h): four values to two words, each
// value the nearest f16, ties to even, subnormals kept, and anything beyond
// f16's range saturated to ±65,504 — against the CPU reference.

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "core/formats/f16/f16.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/compute.h"
#include "support/f16_reference.h"

using namespace bllm;

namespace {

// The kernel side: four values an invocation, packed, written as stored.
constexpr std::string_view kPackHarness = R"(
@group(0) @binding(0) var<storage, read> values: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> packed: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.x) {
        packed[id.x] = pack(values[id.x]);
    }
}
)";

std::vector<std::uint16_t> pack_on_gpu(WGPUInstance instance, const gpu::Device& device, std::span<const float> values) {
    const auto quads = static_cast<std::uint32_t>(values.size() / 4);
    const std::string source = std::string(formats::kF16.pack_wgsl()) + std::string(kPackHarness);
    const auto out = testing::run_compute(instance, device, source, std::as_bytes(values), values.size() * 2,
                                          {quads, 0, 0, 0}, (quads + 63) / 64);
    std::vector<std::uint16_t> halves(values.size());
    std::memcpy(halves.data(), out.data(), out.size());
    return halves;
}

float from_bits(std::uint32_t bits) {
    float f = 0;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

TEST_CASE("F16 pack rounds to nearest even, keeps subnormals and saturates beyond the range") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    const float ulp_at_1 = std::ldexp(1.0f, -10);
    std::vector<float> values = {
        0.0f, 1.0f, -2.5f, 65504.0f,                                       // exact
        1.0f + ulp_at_1 / 2, 1.0f + 3 * ulp_at_1 / 2,                      // ties: down to even, up to even
        1.0f + ulp_at_1 / 4, 1.0f + 3 * ulp_at_1 / 4,                      // below and above a half
        std::ldexp(1.0f, -24), std::ldexp(1.0f, -25) * 3, std::ldexp(1.0f, -15),   // subnormal results
        std::ldexp(1.0f, -26), std::ldexp(1.0f, -14) * (1 - std::ldexp(1.0f, -12)),   // to zero; up to normal
        65519.0f, 65520.0f, 1e10f, -1e10f,                                 // to the largest; saturated
        std::numeric_limits<float>::max(), -65536.0f, -65504.0f, 3.0e-8f,
    };
    std::uint32_t state = 0x2545F491u;
    while (values.size() < 4096) {
        state = state * 1664525u + 1013904223u;
        // Finite f32s across f16's range and beyond it: exponents 2^-30 .. 2^20.
        const std::uint32_t exponent = 97u + (state >> 8) % 51u;
        values.push_back(from_bits((state & 0x80000000u) | (exponent << 23) | (state & 0x007FFFFFu)));
    }
    const auto got = pack_on_gpu(instance.get(), *device, values);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint16_t want = test::fp32_to_fp16_saturating(values[i]);
        // WGSL may drop a zero's sign (format.h).
        const bool zeros = (got[i] & 0x7FFFu) == 0 && (want & 0x7FFFu) == 0;
        if (got[i] != want && !zeros && differing++ < 8) {
            CAPTURE(i);
            CAPTURE(values[i]);
            CHECK(got[i] == want);
        }
    }
    CHECK(differing == 0);
}
