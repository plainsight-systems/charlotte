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

// Test-only: each workgroup adds 1 (main) or 1,000 (other) to its counter,
// so a step's counts are the workgroups each launch ran, and which entry.
constexpr std::string_view kCounts = R"(
struct Step { position: u32, tokens: u32, ids: array<vec4<u32>, 128> }
struct Nothing { unused: u32 }
override workgroup_size: u32;
override counter: u32 = 0;
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> nothing: Nothing;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) local: u32) {
    if (local == 0u && step.tokens > 0u && nothing.unused == 0u) {
        atomicAdd(&counts[counter], 1u);
    }
}
@compute @workgroup_size(workgroup_size)
fn other(@builtin(local_invocation_index) local: u32) {
    if (local == 0u && step.tokens > 0u && nothing.unused == 0u) {
        atomicAdd(&counts[counter], 1000u);
    }
}
)";

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

TEST_CASE("launches run their tiles and key splits, a combine only when split, each at its entry point") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange counts = scratch(m, "hidden");
    const auto launch = [&](std::uint32_t counter, std::uint32_t per_row, std::uint32_t rows_per_tile,
                            kernels::KeySplit split, std::string_view entry) {
        kernels::Launch l{kCounts, nullptr, std::vector<std::byte>(4),
                          {{counts.buffer, counts.offset, counts.length}}, per_row, 64};
        l.overrides = {{"counter", static_cast<double>(counter)}};
        l.rows_per_tile = rows_per_tile;
        l.key_split = split;
        l.window = 40960;
        l.entry_point = entry;
        return l;
    };
    const auto program = build_program(
        instance.get(), *upload,
        {launch(0, 128, 4, kernels::KeySplit::None, "main"), launch(1, 64, 0, kernels::KeySplit::PerChunk, "other"),
         launch(2, 64, 0, kernels::KeySplit::WhenSplit, "main")});
    const auto read = [&] {
        const auto words = read_floats(instance.get(), *device, upload->buffer(counts.buffer), counts.offset, 3);
        std::uint32_t c[3];
        std::memcpy(c, words.data(), sizeof c);
        return std::array<std::uint32_t, 3>{c[0], c[1], c[2]};
    };

    // Decode at 4,096: one tile of 4 rows, 8 workgroups; 16 chunks, each a
    // workgroup of the other entry; the step splits, so the combine runs.
    run_step(instance.get(), *program, 1, {}, 4095);
    CHECK(read() == std::array<std::uint32_t, 3>{8, 16000, 1});

    // Five rows at 100: two tiles; one chunk, so no split and no combine.
    run_step(instance.get(), *program, 5, {}, 100);
    CHECK(read() == std::array<std::uint32_t, 3>{8 + 16, 16000 + 5000, 1});
}
