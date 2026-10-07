#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/diagnostics.h"
#include "core/gpu/wgpu_handles.h"

namespace bllm::gpu {

// The limits that constrain how model weights can be laid out on the GPU, and
// how many of them a shader can see at once.
//
// These are four separate WebGPU constraints, not one. Conflating them yields
// a residency plan that uploads successfully and then cannot be bound:
//
//   max_buffer_size                       caps a physical buffer
//   max_storage_buffer_binding_size       caps a *bound range* within one
//   min_storage_buffer_offset_alignment   constrains suballocated offsets
//   max_storage_buffers_per_shader_stage  caps bindings visible to one shader
//
// The last matters more than it looks: every piece of a weight is a binding.
// A quantized piece lies on the device as streams — nibbles, then scales
// (formats/device_layout.h) — but all in one range, so a piece costs one
// binding, not one per stream.
struct DeviceLimits {
    std::uint64_t max_buffer_size = 0;
    std::uint64_t max_storage_buffer_binding_size = 0;
    std::uint32_t max_compute_workgroups_per_dimension = 0;
    std::uint32_t max_compute_invocations_per_workgroup = 0;
    std::uint32_t max_storage_buffers_per_shader_stage = 0;
    std::uint32_t min_storage_buffer_offset_alignment = 0;
};

// What the adapter says it is. Always queried: an adapter that cannot say is
// refused (Device::request), since its backend decides whether it computes
// at all. Fields it leaves empty are empty.
struct AdapterInfo {
    std::string vendor;
    std::string architecture;
    std::string device;
    std::string description;
    std::string backend;
};

// Whether a device has been lost, and why. Written once, by the device-lost
// callback. It names the cause of a failure; it is never proof that a device
// is alive. Once a device is lost WebGPU resolves error scopes clean and
// queued work as done, and it does not order the lost callback before them,
// so work can complete, and this still read not lost, after the loss. What
// shows a device alive is a completed mapping, which a lost one refuses
// (residency/upload.h).
struct DeviceStatus {
    bool lost = false;
    WGPUDeviceLostReason reason = WGPUDeviceLostReason_Unknown;
    std::string message;
};

// Owns a WebGPU instance, adapter and device. Every handle is an RAII alias,
// so release is not written by hand anywhere.
//
// It installs the device-lost callback, which WebGPU runs exactly once: when
// the device is lost, destroyed, or fails to be created. The callback holds
// its own reference to the DeviceStatus, since it may run after the Device is
// gone; status() hands others theirs (R.21: the one shared owner, because any
// of them may end first).
//
// Acquisition is asynchronous because WebGPU's adapter and device requests are
// promises in the browser and this build deliberately does not enable ASYNCIFY.
// The caller supplies a continuation rather than blocking.
//
// Ownership is transferred through a unique_ptr rather than a raw pointer, so
// the callee cannot forget to destroy it (I.11, R.20). On failure the pointer
// is null and `error` describes why. Exactly one of those states is delivered,
// exactly once.
//
// Contract:
//   - `callback` must not be null. It is invoked unconditionally.
//   - `error` is valid only for the duration of the callback. Copy it to keep
//     it; it points into a temporary that dies when the callback returns.
#if BLLM_DIAGNOSTICS_ENABLED
// What a diagnostic build may ask of a device beyond what the harness needs:
// the harness itself asks for no optional feature, so it runs wherever
// WebGPU's defaults do, and a clean build cannot ask (TLM.1).
//   - timestamps: the timestamp-query feature, which a profiled step needs
//     (kernels/program.h), and timestamps inside a pass where the adapter
//     offers them — natively behind Dawn's allow_unsafe_apis toggle, which
//     this enables; the target's Metal adapter does not offer them.
//     Natively it also disables Dawn's timestamp_quantization toggle, which
//     otherwise rounds every timestamp — a browser's protection against
//     timing attacks, and coarser than the dispatches it would time. An
//     adapter without the feature is a failure, named, not a device without
//     it.
struct DiagnosticRequest {
    bool timestamps = false;
};
#endif

class Device {
public:
    using RequestCallback = void (*)(std::unique_ptr<Device> device,
                                     const char* error,
                                     void* userdata);

    // Requests a device from `instance`, keeping a reference of its own to it.
    // Natively the caller runs the callbacks, through the same instance, with
    // wgpuInstanceProcessEvents (callback_mode.h). A null instance is a
    // failure reported through `callback`, at once, like any other (E.27).
    // `options`, if given, steers the adapter chosen: a test asks for one
    // backend by it.
    //
    // An adapter of Dawn's Null backend is refused, by name: it accepts work
    // and computes nothing, and natively, where no real backend is reachable,
    // Dawn can hand one out. A harness that ran on it would report results it
    // never computed — the self-check reads back zeros and fails, but a test
    // that checked less would pass. Tested natively by asking for it.
    static void request(WGPUInstance instance, RequestCallback callback, void* userdata,
                        const WGPURequestAdapterOptions* options = nullptr);

#if BLLM_DIAGNOSTICS_ENABLED
    // The same, asking for what `diagnostic` names as well.
    static void request(WGPUInstance instance, RequestCallback callback, void* userdata,
                        const WGPURequestAdapterOptions* options, const DiagnosticRequest& diagnostic);

    // Whether the device granted timestamp queries, and timestamps inside a
    // pass.
    bool timestamps() const { return timestamps_; }
    bool timestamps_inside_passes() const { return timestamps_inside_passes_; }
#endif

    // The same, from an instance of its own: the browser's path, where the
    // event loop runs every callback and nothing else needs the instance.
    static void request(RequestCallback callback, void* userdata);

    ~Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    WGPUInstance instance() const { return instance_.get(); }
    WGPUDevice handle() const { return device_.get(); }
    WGPUQueue queue() const { return queue_.get(); }
    const AdapterInfo& adapter_info() const { return adapter_info_; }

    // Whether the device has been lost. Shared, so it outlives the Device for
    // whoever still holds it.
    std::shared_ptr<const DeviceStatus> status() const { return status_; }

    // Limits of the ACQUIRED DEVICE. These are what validation enforces and
    // what dispatch must be sized against.
    //
    // Not the same thing as adapter_maxima(). WebGPU grants a device the
    // DEFAULT limits unless better ones are named in requiredLimits at
    // acquisition, so an adapter advertising 4 GiB buffers can yield a device
    // capped at 256/128 MiB. Sizing work against the adapter's numbers would
    // pass our own checks and then fail device validation.
    const DeviceLimits& limits() const { return limits_; }

    // What the adapter could grant if asked. Informational only: never size a
    // dispatch or a buffer against these.
    const DeviceLimits& adapter_maxima() const { return adapter_maxima_; }

private:
    Device() = default;
    friend struct PendingDeviceRequest;
#if BLLM_DIAGNOSTICS_ENABLED
    static void request_with(WGPUInstance instance, RequestCallback callback, void* userdata,
                             const WGPURequestAdapterOptions* options, const DiagnosticRequest* diagnostic);
#endif

    // Declaration order is release order reversed by the compiler: members are
    // destroyed bottom-up, so queue releases before device, device before
    // adapter, adapter before instance.
    Instance instance_;
    Adapter adapter_;
    DeviceHandle device_;
    Queue queue_;
    AdapterInfo adapter_info_;
    std::shared_ptr<DeviceStatus> status_;   // shared with the device-lost callback
#if BLLM_DIAGNOSTICS_ENABLED
    bool timestamps_ = false;
    bool timestamps_inside_passes_ = false;
#endif
    DeviceLimits limits_;          // of the acquired device
    DeviceLimits adapter_maxima_;  // of the adapter, informational
};

}  // namespace bllm::gpu
