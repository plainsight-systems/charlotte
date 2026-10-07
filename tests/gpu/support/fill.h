#pragma once

// Test-only: a kernel that fills a buffer, run by the program like any
// launch, so a GPU test can make inputs on the device rather than commit
// them as fixtures.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "core/kernels/interface.h"

namespace bllm::testing {

// Test-only: fills a buffer with deterministic pseudo-random values in
// [−amplitude, amplitude): f32, or pairs of halves.
inline constexpr std::string_view kFill = R"(
struct Fill { seed: u32, count: u32, amplitude: f32, halves: u32 }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> fill: Fill;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
fn hash(x: u32) -> u32 {
    var v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}
fn unit(x: u32) -> f32 { return f32(hash(x) >> 8u) / 8388608.0 - 1.0; }
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= fill.count || step.tokens == 0u) { return; }
    if (fill.halves == 1u) {
        out[i] = pack2x16float(vec2<f32>(unit(fill.seed + 2u * i), unit(fill.seed + 2u * i + 1u)) * fill.amplitude);
    } else {
        out[i] = bitcast<u32>(unit(fill.seed + i) * fill.amplitude);
    }
}
)";

inline std::vector<std::byte> fill_constants(std::uint32_t seed, std::uint32_t count, float amplitude, bool halves) {
    std::vector<std::byte> bytes(16);
    const std::uint32_t h = halves ? 1 : 0;
    std::memcpy(bytes.data(), &seed, 4);
    std::memcpy(bytes.data() + 4, &count, 4);
    std::memcpy(bytes.data() + 8, &amplitude, 4);
    std::memcpy(bytes.data() + 12, &h, 4);
    return bytes;
}

// A launch filling `range` with `count` words of seed `seed`: f32 in
// [-amplitude, amplitude), or pairs of halves.
inline kernels::Launch fill_launch(const residency::BufferRange& range, std::uint32_t seed, float amplitude,
                                   bool halves) {
    const auto count = static_cast<std::uint32_t>(range.length / 4);
    return kernels::Launch{kFill, nullptr, fill_constants(seed, count, amplitude, halves),
                           {{range.buffer, range.offset, range.length}}, count, 64};
}

}  // namespace bllm::testing
