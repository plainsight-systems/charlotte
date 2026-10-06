// The program on Dawn, for what its launches ask beyond one kernel's
// contract (kernels/program.h): constants that span several slots, each
// launch's read at its own offset, and a kernel composed with a format's
// pack.

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/kernels/program.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

// Test-only: writes its 528 bytes of constants, as 33 vec4s, to `out`.
constexpr std::string_view kWide = R"(
struct Step { position: u32, tokens: u32, ids: array<vec4<u32>, 128> }
struct Wide { words: array<vec4<f32>, 33> }
override workgroup_size: u32;
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> wide: Wide;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 33u && step.tokens > 0u) {
        out[id.x] = wide.words[id.x];
    }
}
)";

// Test-only: writes pack(1, 2, 3, 4), the two words a format's pack returns.
constexpr std::string_view kPacks = R"(
struct Step { position: u32, tokens: u32, ids: array<vec4<u32>, 128> }
struct Nothing { unused: u32 }
override workgroup_size: u32;
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> nothing: Nothing;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0u && step.tokens > 0u && nothing.unused == 0u) {
        let words = pack(vec4<f32>(1.0, 2.0, 3.0, 4.0));
        out[0] = words.x;
        out[1] = words.y;
    }
}
)";

// A format of the test's own whose pack is plainly checkable: the sum of
// the first two values and the product of the last two, as f32 bits.
inline constexpr formats::Format kSumProduct{
    formats::kF32Layout, "fn unpack_unused() {}",
    "fn pack(v: vec4<f32>) -> vec2<u32> { return vec2<u32>(bitcast<u32>(v.x + v.y), bitcast<u32>(v.z * v.w)); }"};

residency::BufferRange scratch(const Model& m, std::string_view purpose) {
    for (const auto& s : m.plan.scratch) {
        if (s.purpose == purpose) return s.range;
    }
    FAIL("no working buffer " << purpose);
    return {};
}

std::vector<std::byte> wide_constants(float first) {
    std::vector<std::byte> bytes(528);
    for (std::size_t i = 0; i < 132; ++i) {
        const float v = first + static_cast<float>(i);
        std::memcpy(bytes.data() + 4 * i, &v, 4);
    }
    return bytes;
}

}  // namespace

TEST_CASE("constants spanning several slots are each launch's own, at its own offset") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange first = scratch(m, "hidden"), second = scratch(m, "normed");

    // Each 528 bytes, three slots: the second's begin at 768, past the first's.
    std::vector<kernels::Launch> launches;
    launches.push_back({kWide, nullptr, wide_constants(0.0f), {{first.buffer, first.offset, first.length}}, 64, 64});
    launches.push_back(
        {kWide, nullptr, wide_constants(1000.0f), {{second.buffer, second.offset, second.length}}, 64, 64});
    const auto program = build_program(instance.get(), *upload, std::move(launches));
    run_step(instance.get(), *program, 1);

    const auto a = read_floats(instance.get(), *device, upload->buffer(first.buffer), first.offset, 132);
    const auto b = read_floats(instance.get(), *device, upload->buffer(second.buffer), second.offset, 132);
    for (std::size_t i = 0; i < 132; ++i) {
        CAPTURE(i);
        CHECK(a[i] == static_cast<float>(i));
        CHECK(b[i] == 1000.0f + static_cast<float>(i));
    }
}

TEST_CASE("a kernel composed with a format's pack calls it") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange out = scratch(m, "hidden");

    kernels::Launch launch{kPacks, nullptr, std::vector<std::byte>(4), {{out.buffer, out.offset, out.length}}, 64, 64};
    launch.pack_format = &kSumProduct;
    const auto program = build_program(instance.get(), *upload, {launch});
    run_step(instance.get(), *program, 1);
    const auto words = read_floats(instance.get(), *device, upload->buffer(out.buffer), out.offset, 2);
    CHECK(words[0] == 3.0f);
    CHECK(words[1] == 12.0f);
}

TEST_CASE("a launch whose pack format has no pack is refused, saying so") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange out = scratch(m, "hidden");

    kernels::Launch launch{kPacks, nullptr, std::vector<std::byte>(4), {{out.buffer, out.offset, out.length}}, 64, 64};
    launch.pack_format = &kF32;   // the test's F32 format: no pack
    const Built built = try_build(instance.get(), *upload, {launch});
    CHECK(built.program == nullptr);
    CHECK(built.error == kernels::ProgramError::Build);
    CHECK_MESSAGE(built.message.find("no pack") != std::string::npos, built.message);
}
