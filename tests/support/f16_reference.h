#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

// CPU reference for F16 — TEST ORACLE ONLY. Decoding is quant::fp16_to_fp32,
// exact. Encoding mirrors the F16 pack (formats/f16/f16_pack.wgsl): the value
// saturated to ±65,504, then rounded to the nearest f16, ties to even —
// ggml's ggml_compute_fp32_to_fp16 rounds the same way but overflows to
// infinity instead.
namespace bllm::test {

[[nodiscard]] inline std::uint16_t fp32_to_fp16_saturating(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::uint32_t magnitude = bits & 0x7FFFFFFFu;
    if (magnitude >= 0x477FE000u) return sign | 0x7BFFu;   // >= 65,504: saturated
    if (magnitude < 0x38800000u) {                          // below 2^-14: subnormal, or zero
        float a = 0;
        std::memcpy(&a, &magnitude, 4);
        // a × 2^24 is exact, and nearbyint rounds to nearest, ties to even.
        return sign | static_cast<std::uint16_t>(std::nearbyint(a * 16777216.0f));
    }
    // Rebias the exponent from 127 to 15, then round the 13 bits dropped to
    // nearest, ties to even.
    const std::uint32_t odd = (magnitude >> 13) & 1u;
    return sign | static_cast<std::uint16_t>((magnitude + 0xC8000FFFu + odd) >> 13);
}

}  // namespace bllm::test
