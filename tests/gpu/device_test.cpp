#include <doctest/doctest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/self_check.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/pump.h"

using namespace bllm;
using bllm::testing::pump_until;

using bllm::testing::acquire;
using bllm::testing::Acquired;

TEST_CASE("a device is acquired natively, with the limits the harness requires") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    REQUIRE(instance);
    const auto device = acquire(instance.get());
    const auto& info = device->adapter_info();
    MESSAGE("adapter: " << info.description << " (" << info.backend << ")");
    CHECK(device->handle() != nullptr);
    CHECK(device->queue() != nullptr);
    CHECK(device->limits().max_buffer_size > 0);
}

TEST_CASE("no instance is a failure reported through the callback, at once") {
    Acquired acquired;
    gpu::Device::request(
        nullptr,
        [](std::unique_ptr<gpu::Device> device, const char* error, void* userdata) {
            auto& a = *static_cast<Acquired*>(userdata);
            a.device = std::move(device);
            if (error != nullptr) a.error = error;
            a.done = true;
        },
        &acquired);
    CHECK(acquired.done);
    CHECK(acquired.device == nullptr);
    CHECK(acquired.error == "no WebGPU instance was given");
}

TEST_CASE("an adapter of Dawn's Null backend, which computes nothing, is refused by name") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    REQUIRE(instance);
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.backendType = WGPUBackendType_Null;
    Acquired acquired;
    gpu::Device::request(
        instance.get(),
        [](std::unique_ptr<gpu::Device> device, const char* error, void* userdata) {
            auto& a = *static_cast<Acquired*>(userdata);
            a.device = std::move(device);
            if (error != nullptr) a.error = error;
            a.done = true;
        },
        &acquired, &options);
    pump_until(instance.get(), acquired.done, "a request for the Null backend");
    CHECK(acquired.device == nullptr);
    CHECK(acquired.error == "the WebGPU adapter selected is Dawn's Null backend, which accepts work "
                            "and computes nothing");
}

TEST_CASE("a retained instance stays usable after its first holder lets it go") {
    gpu::Instance first{wgpuCreateInstance(nullptr)};
    REQUIRE(first);
    const gpu::Instance kept = gpu::retain(first.get());
    first.reset();   // the first holder's reference goes; the kept one remains
    const auto device = acquire(kept.get());
    CHECK(device->handle() != nullptr);
    CHECK_FALSE(gpu::retain(static_cast<WGPUInstance>(nullptr)));
}

TEST_CASE("a lost device says so, and why, even to holders that outlive it") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    REQUIRE(instance);
    auto device = acquire(instance.get());
    const auto status = device->status();
    CHECK_FALSE(status->lost);

    wgpuDeviceDestroy(device->handle());
    pump_until(instance.get(), status->lost, "the device-lost callback");
    CHECK(status->reason == WGPUDeviceLostReason_Destroyed);

    // The status outlives the Device for whoever holds it.
    device.reset();
    CHECK(status->lost);
}

TEST_CASE("a device released without being destroyed is reported lost too") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    REQUIRE(instance);
    auto device = acquire(instance.get());
    const auto status = device->status();
    device.reset();   // the last reference: WebGPU destroys the device
    pump_until(instance.get(), status->lost, "the device-lost callback");
    CHECK(status->reason == WGPUDeviceLostReason_Destroyed);
}

TEST_CASE("a lost device cannot map a buffer, though its queue still reports work done") {
    // What Upload's success rests on (residency/upload.h): finished work is no
    // evidence the device is alive, a completed mapping is.
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    REQUIRE(instance);
    auto device = acquire(instance.get());
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.size = 4;
    desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    const gpu::Buffer buffer{wgpuDeviceCreateBuffer(device->handle(), &desc)};
    REQUIRE(buffer);
    wgpuDeviceDestroy(device->handle());

    struct Seen {
        bool done = false;
        WGPUQueueWorkDoneStatus status{};
    } work;
    WGPUQueueWorkDoneCallbackInfo work_cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    work_cb.mode = gpu::kCallbackMode;
    work_cb.userdata1 = &work;
    work_cb.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView, void* ud, void*) {
        auto& w = *static_cast<Seen*>(ud);
        w.status = status;
        w.done = true;
    };
    wgpuQueueOnSubmittedWorkDone(device->queue(), work_cb);
    pump_until(instance.get(), work.done, "queued work on a lost device");
    MESSAGE("work done on a lost device: status " << static_cast<int>(work.status));
    CHECK(work.status == WGPUQueueWorkDoneStatus_Success);   // no evidence of life

    struct Mapped {
        bool done = false;
        WGPUMapAsyncStatus status{};
    } mapped;
    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map_cb.mode = gpu::kCallbackMode;
    map_cb.userdata1 = &mapped;
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud, void*) {
        auto& m = *static_cast<Mapped*>(ud);
        m.status = status;
        m.done = true;
    };
    wgpuBufferMapAsync(buffer.get(), WGPUMapMode_Read, 0, 4, map_cb);
    pump_until(instance.get(), mapped.done, "a mapping on a lost device");
    MESSAGE("mapping on a lost device: status " << static_cast<int>(mapped.status));
    CHECK(mapped.status != WGPUMapAsyncStatus_Success);
}

TEST_CASE("the self-check runs vector_add on the GPU and reads back every value correctly") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    REQUIRE(instance);
    struct Checked {
        gpu::SelfCheckResult result;
        bool done = false;
    } checked;
    constexpr std::size_t kElements = 1 << 16;
    gpu::run_self_check(
        acquire(instance.get()), kElements,
        [](gpu::SelfCheckResult result, void* userdata) {
            auto& c = *static_cast<Checked*>(userdata);
            c.result = std::move(result);
            c.done = true;
        },
        &checked);
    pump_until(instance.get(), checked.done, "the self-check's readback");
    CHECK_MESSAGE(checked.result.ok, checked.result.error);
    CHECK(checked.result.elements == kElements);
    CHECK(checked.result.mismatches == 0);
    CHECK(checked.result.device != nullptr);   // handed back on every path
}

#if BLLM_DIAGNOSTICS_ENABLED
TEST_CASE("a diagnostic build's device grants timestamp queries; the harness's asks for none") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto harness = acquire(instance.get());
    CHECK(!harness->timestamps());
    CHECK(!harness->timestamps_inside_passes());
    const auto diagnostic = acquire(instance.get(), gpu::DiagnosticRequest{.timestamps = true});
    CHECK(diagnostic->timestamps());
    MESSAGE("timestamps inside passes: " << diagnostic->timestamps_inside_passes() << " on "
                                         << diagnostic->adapter_info().backend);
}
#endif
