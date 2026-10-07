#pragma once

#include <webgpu/webgpu.h>

#include "core/gpu/unique_handle.h"

// RAII aliases for the WebGPU handles this project owns. Every wgpuXRelease
// call in the codebase is here; no other file releases a handle by hand.
namespace bllm::gpu {

using Instance = UniqueHandle<WGPUInstance, wgpuInstanceRelease>;
using Adapter = UniqueHandle<WGPUAdapter, wgpuAdapterRelease>;
using DeviceHandle = UniqueHandle<WGPUDevice, wgpuDeviceRelease>;
using Queue = UniqueHandle<WGPUQueue, wgpuQueueRelease>;
using Buffer = UniqueHandle<WGPUBuffer, wgpuBufferRelease>;
using ShaderModule = UniqueHandle<WGPUShaderModule, wgpuShaderModuleRelease>;
using ComputePipeline = UniqueHandle<WGPUComputePipeline, wgpuComputePipelineRelease>;
using BindGroup = UniqueHandle<WGPUBindGroup, wgpuBindGroupRelease>;
using BindGroupLayout = UniqueHandle<WGPUBindGroupLayout, wgpuBindGroupLayoutRelease>;
using CommandEncoder = UniqueHandle<WGPUCommandEncoder, wgpuCommandEncoderRelease>;
using CommandBuffer = UniqueHandle<WGPUCommandBuffer, wgpuCommandBufferRelease>;
using ComputePassEncoder = UniqueHandle<WGPUComputePassEncoder, wgpuComputePassEncoderRelease>;
using QuerySet = UniqueHandle<WGPUQuerySet, wgpuQuerySetRelease>;

// A counted reference of one's own to a handle someone else holds. The types
// above adopt the handle they are given — they release it, and never add a
// reference — so wrapping a borrowed handle directly would release one never
// taken. Every borrowed handle kept is taken through these. A null handle
// gives an empty one.
inline Instance retain(WGPUInstance handle) {
    if (handle != nullptr) wgpuInstanceAddRef(handle);
    return Instance{handle};
}

inline DeviceHandle retain(WGPUDevice handle) {
    if (handle != nullptr) wgpuDeviceAddRef(handle);
    return DeviceHandle{handle};
}

}  // namespace bllm::gpu
