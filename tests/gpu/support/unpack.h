#pragma once

// Test-only: runs a format's unpack on the GPU over stored blocks, as a
// kernel would, and returns what it decoded. The blocks are laid out on the
// device by PieceWriter, as upload lays out a piece, so the test reads what
// upload writes. One invocation per 32-weight group writes its eight vec4s
// out; the output is read back whole. Shader errors fail the test with
// WebGPU's message.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/formats/device_layout.h"
#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/piece_writer.h"
#include "core/residency/routes.h"
#include "support/compute.h"
#include "support/pump.h"

namespace bllm::testing {

// The kernel side of format.h's contract, around the format's unpack.
inline constexpr std::string_view kUnpackHarnessHead = R"(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
)";

inline constexpr std::string_view kUnpackHarnessMain = R"(
struct Params { blocks_in_piece: u32, groups: u32 }
@group(0) @binding(1) var<storage, read_write> decoded: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let group = id.x;
    if (group >= params.groups) { return; }
    let v = unpack(params.blocks_in_piece, group);
    for (var i = 0u; i < 8u; i++) {
        decoded[group * 8u + i] = v[i];
    }
}
)";

// `stored` as one piece of `layout` lays out on the device.
inline std::vector<std::byte> lay_out(const formats::DeviceLayout& layout, std::span<const std::byte> stored) {
    const std::uint64_t blocks = stored.size() / layout.block_bytes;
    const std::uint64_t length = (stored.size() + 3) / 4 * 4;
    const residency::Route route{gguf::TensorId{0}, 0, blocks, &layout, residency::BufferIndex{0}, 0, length};
    residency::PieceWriter writer(std::span(&route, 1), stored.size());
    std::vector<residency::Write> writes;
    REQUIRE(writer.accept(0, stored, writes) == residency::WriteError::Ok);
    REQUIRE(writer.finish(stored.size()) == residency::WriteError::Ok);
    std::vector<std::byte> device(length);
    for (const auto& w : writes) std::memcpy(device.data() + w.offset, w.bytes.data(), w.bytes.size());
    return device;
}

inline std::vector<float> run_unpack(WGPUInstance instance, const gpu::Device& device, std::string_view unpack_wgsl,
                                     const formats::DeviceLayout& layout, std::span<const std::byte> stored,
                                     std::uint32_t groups) {
    const auto blocks = static_cast<std::uint32_t>(stored.size() / layout.block_bytes);
    const std::vector<std::byte> on_device = lay_out(layout, stored);
    const std::string source = std::string(kUnpackHarnessHead) + std::string(unpack_wgsl) +
                               std::string(kUnpackHarnessMain);
    const std::vector<std::byte> out = run_compute(instance, device, source, on_device,
                                                   std::uint64_t{groups} * 32 * sizeof(float), {blocks, groups, 0, 0},
                                                   (groups + 63) / 64);
    std::vector<float> decoded(std::size_t{groups} * 32);
    std::memcpy(decoded.data(), out.data(), out.size());
    return decoded;
}

// Bit for bit, but a zero of either sign equals a zero: WGSL may drop a
// zero's sign, and no kernel can tell (format.h).
inline bool same_weight(float got, float want) {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::memcpy(&a, &got, 4);
    std::memcpy(&b, &want, 4);
    return a == b || (got == 0.0f && want == 0.0f);
}

// Every weight the same as the reference's, as same_weight judges.
inline void check_bitwise(std::span<const float> got, std::span<const float> want) {
    REQUIRE(got.size() == want.size());
    std::size_t differing = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!same_weight(got[i], want[i]) && differing++ < 8) {
            CAPTURE(i);
            CHECK(got[i] == want[i]);
        }
    }
    CHECK(differing == 0);
}

}  // namespace bllm::testing
