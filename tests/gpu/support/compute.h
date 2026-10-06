#pragma once

// Test-only: runs one compute shader once, outside the program, for the GPU
// tests of a format's WGSL. The shader's `main` binds 0, its input, read-only
// storage; 1, its output, storage; 2, a uniform of four words. Shader and
// validation errors fail the test with WebGPU's message.

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "support/pump.h"

namespace bllm::testing {

struct ScopeResult {
    WGPUErrorType type = WGPUErrorType_Unknown;
    std::string message;
    bool done = false;
};

// Dispatches `workgroups` workgroups of `source`'s main and returns the
// output buffer's `out_bytes`.
inline std::vector<std::byte> run_compute(WGPUInstance instance, const gpu::Device& device, std::string_view source,
                                          std::span<const std::byte> input, std::uint64_t out_bytes,
                                          std::array<std::uint32_t, 4> params, std::uint32_t workgroups) {
    WGPUDevice dev = device.handle();
    wgpuDevicePushErrorScope(dev, WGPUErrorFilter_Validation);
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = WGPUStringView{source.data(), source.size()};
    WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    module_desc.nextInChain = &wgsl.chain;
    const gpu::ShaderModule module(wgpuDeviceCreateShaderModule(dev, &module_desc));
    WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_desc.compute.module = module.get();
    pipeline_desc.compute.entryPoint = WGPUStringView{"main", 4};
    const gpu::ComputePipeline pipeline(wgpuDeviceCreateComputePipeline(dev, &pipeline_desc));

    const auto buffer = [&](WGPUBufferUsage usage, std::uint64_t size) {
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.usage = usage;
        d.size = size;
        return gpu::Buffer(wgpuDeviceCreateBuffer(dev, &d));
    };
    const std::uint64_t in_bytes = (input.size() + 3) / 4 * 4;
    const gpu::Buffer in = buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, in_bytes);
    const gpu::Buffer out = buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, out_bytes);
    const gpu::Buffer uniform = buffer(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 16);
    const gpu::Buffer readback = buffer(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, out_bytes);
    std::vector<std::byte> padded(in_bytes);
    std::memcpy(padded.data(), input.data(), input.size());
    wgpuQueueWriteBuffer(device.queue(), in.get(), 0, padded.data(), padded.size());
    wgpuQueueWriteBuffer(device.queue(), uniform.get(), 0, params.data(), sizeof(params));

    const gpu::BindGroupLayout bind_layout(wgpuComputePipelineGetBindGroupLayout(pipeline.get(), 0));
    WGPUBindGroupEntry entries[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                                     WGPU_BIND_GROUP_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].buffer = in.get();
    entries[0].size = in_bytes;
    entries[1].binding = 1;
    entries[1].buffer = out.get();
    entries[1].size = out_bytes;
    entries[2].binding = 2;
    entries[2].buffer = uniform.get();
    entries[2].size = 16;
    WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_desc.layout = bind_layout.get();
    bind_desc.entryCount = 3;
    bind_desc.entries = entries;
    const gpu::BindGroup bind_group(wgpuDeviceCreateBindGroup(dev, &bind_desc));

    const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(dev, nullptr));
    {
        const gpu::ComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr));
        wgpuComputePassEncoderSetPipeline(pass.get(), pipeline.get());
        wgpuComputePassEncoderSetBindGroup(pass.get(), 0, bind_group.get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass.get(), workgroups, 1, 1);
        wgpuComputePassEncoderEnd(pass.get());
    }
    wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), out.get(), 0, readback.get(), 0, out_bytes);
    const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
    WGPUCommandBuffer raw = commands.get();
    wgpuQueueSubmit(device.queue(), 1, &raw);

    ScopeResult scope;
    WGPUPopErrorScopeCallbackInfo pop = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    pop.mode = gpu::kCallbackMode;
    pop.userdata1 = &scope;
    pop.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* userdata, void*) {
        auto& s = *static_cast<ScopeResult*>(userdata);
        s.type = type;
        if (message.data != nullptr) {
            s.message = message.length == WGPU_STRLEN ? std::string(message.data)
                                                      : std::string(message.data, message.length);
        }
        s.done = true;
    };
    wgpuDevicePopErrorScope(dev, pop);
    pump_until(instance, scope.done, "the kernel's scope");
    REQUIRE_MESSAGE(scope.type == WGPUErrorType_NoError, scope.message);

    struct Mapped {
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        bool done = false;
    } mapped;
    WGPUBufferMapCallbackInfo map = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map.mode = gpu::kCallbackMode;
    map.userdata1 = &mapped;
    map.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata, void*) {
        auto& m = *static_cast<Mapped*>(userdata);
        m.status = status;
        m.done = true;
    };
    wgpuBufferMapAsync(readback.get(), WGPUMapMode_Read, 0, out_bytes, map);
    pump_until(instance, mapped.done, "the kernel's output");
    REQUIRE(mapped.status == WGPUMapAsyncStatus_Success);
    const void* range = wgpuBufferGetConstMappedRange(readback.get(), 0, out_bytes);
    REQUIRE(range != nullptr);
    std::vector<std::byte> result(out_bytes);
    std::memcpy(result.data(), range, out_bytes);
    wgpuBufferUnmap(readback.get());
    return result;
}

}  // namespace bllm::testing
