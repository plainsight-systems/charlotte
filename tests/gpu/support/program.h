#pragma once

// Test-only: building a Program over an upload, running a step on it, and
// reading a buffer back, each wait pumped on the calling thread (pump.h), for
// the GPU tests of the kernels.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/kernels/program.h"
#include "core/residency/upload.h"
#include "support/pump.h"

namespace bllm::testing {

// How a build ended, as its callback reported it.
struct Built {
    std::unique_ptr<kernels::Program> program;
    kernels::ProgramError error = kernels::ProgramError::Build;
    std::string message;
    bool done = false;
};

// Builds a program for `launches` and returns how the build ended.
inline Built try_build(WGPUInstance instance, const residency::Upload& upload,
                       std::vector<kernels::Launch> launches) {
    Built built;
    kernels::Program::build(
        upload, std::move(launches),
        [](std::unique_ptr<kernels::Program> p, kernels::ProgramError e, std::string_view m, void* userdata) {
            auto& b = *static_cast<Built*>(userdata);
            b.program = std::move(p);
            b.error = e;
            b.message = m;
            b.done = true;
        },
        &built);
    pump_until(instance, built.done, "the program");
    return built;
}

// Builds a program, as try_build; it must succeed.
inline std::unique_ptr<kernels::Program> build_program(WGPUInstance instance, const residency::Upload& upload,
                                                       std::vector<kernels::Launch> launches) {
    Built built = try_build(instance, upload, std::move(launches));
    REQUIRE_MESSAGE(built.error == kernels::ProgramError::Ok, built.message);
    REQUIRE(built.program != nullptr);
    return std::move(built.program);
}

// How a step ended, as its callback reported it.
struct StepOutcome {
    kernels::ProgramError error;
    std::string message;
};

// Runs one step of `tokens` tokens from `position`, identifiers `ids` where
// given, and returns how it ended; it must be reported once.
inline StepOutcome try_step(WGPUInstance instance, kernels::Program& program, std::uint32_t tokens,
                            std::span<const std::uint32_t> ids = {}, std::uint32_t position = 0) {
    kernels::Step step{};
    step.position = position;
    step.tokens = tokens;
    std::copy(ids.begin(), ids.end(), step.ids.begin());
    struct Ran {
        kernels::ProgramError error = kernels::ProgramError::Step;
        std::string message;
        int calls = 0;
        bool done = false;
    } ran;
    program.run(step,
                [](kernels::ProgramError e, std::string_view message, void* userdata) {
                    auto& r = *static_cast<Ran*>(userdata);
                    r.error = e;
                    r.message = message;
                    ++r.calls;
                    r.done = true;
                },
                &ran);
    pump_until(instance, ran.done, "the step");
    REQUIRE(ran.calls == 1);
    return {ran.error, std::move(ran.message)};
}

// Runs one step, as try_step; it must succeed.
inline void run_step(WGPUInstance instance, kernels::Program& program, std::uint32_t tokens,
                     std::span<const std::uint32_t> ids = {}, std::uint32_t position = 0) {
    const StepOutcome ran = try_step(instance, program, tokens, ids, position);
    REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
}

// `count` floats of `buffer` from `offset`, read back through a mapping. The
// buffer must allow COPY_SRC, as the plan's working buffers do.
inline std::vector<float> read_floats(WGPUInstance instance, const gpu::Device& device, WGPUBuffer buffer,
                                      std::uint64_t offset, std::size_t count) {
    const std::uint64_t bytes = count * sizeof(float);
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    desc.size = bytes;
    const gpu::Buffer readback(wgpuDeviceCreateBuffer(device.handle(), &desc));
    const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(device.handle(), nullptr));
    wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), buffer, offset, readback.get(), 0, bytes);
    const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
    WGPUCommandBuffer raw = commands.get();
    wgpuQueueSubmit(device.queue(), 1, &raw);
    struct Mapped {
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        bool done = false;
    } mapped;
    WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    info.mode = gpu::kCallbackMode;
    info.userdata1 = &mapped;
    info.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata, void*) {
        auto& m = *static_cast<Mapped*>(userdata);
        m.status = status;
        m.done = true;
    };
    wgpuBufferMapAsync(readback.get(), WGPUMapMode_Read, 0, bytes, info);
    pump_until(instance, mapped.done, "a buffer read back");
    REQUIRE(mapped.status == WGPUMapAsyncStatus_Success);
    std::vector<float> out(count);
    std::memcpy(out.data(), wgpuBufferGetConstMappedRange(readback.get(), 0, bytes), bytes);
    wgpuBufferUnmap(readback.get());
    return out;
}

}  // namespace bllm::testing
