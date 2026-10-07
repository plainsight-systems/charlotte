#include "core/gpu/device.h"

#include "core/gpu/callback_mode.h"
#include "core/gpu/device_requirements.h"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bllm::gpu {
namespace {

std::string to_string(WGPUStringView view) {
    if (view.data == nullptr) {
        return {};
    }
    // WGPU_STRLEN means "null-terminated"; anything else is an explicit length.
    return view.length == WGPU_STRLEN ? std::string(view.data)
                                      : std::string(view.data, view.length);
}

const char* backend_name(WGPUBackendType type) {
    switch (type) {
        case WGPUBackendType_Null: return "Null";
        case WGPUBackendType_WebGPU: return "WebGPU";
        case WGPUBackendType_D3D11: return "D3D11";
        case WGPUBackendType_D3D12: return "D3D12";
        case WGPUBackendType_Metal: return "Metal";
        case WGPUBackendType_Vulkan: return "Vulkan";
        case WGPUBackendType_OpenGL: return "OpenGL";
        case WGPUBackendType_OpenGLES: return "OpenGLES";
        default: return "unknown";
    }
}

// WebGPU reports validation and out-of-memory failures here rather than by
// returning null from most creation calls. Without this they are silent, and
// a failed pipeline would surface only as wrong output much later.
//
// It runs spontaneously, natively perhaps on another thread
// (callback_mode.h), so it touches no harness state: it only writes to
// stderr. What the harness must act on is caught by an error scope.
void on_uncaptured_error(WGPUDevice const*, WGPUErrorType type,
                         WGPUStringView message, void*, void*) {
    std::fprintf(stderr, "[webgpu] uncaptured error (type %d): %s\n",
                 static_cast<int>(type), to_string(message).c_str());
}

// Records the loss in the status the callback was given, then releases the
// callback's reference to it. WebGPU runs this exactly once per device.
void on_device_lost(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void* status,
                    void*) {
    const std::unique_ptr<std::shared_ptr<DeviceStatus>> held{static_cast<std::shared_ptr<DeviceStatus>*>(status)};
    DeviceStatus& s = **held;
    s.lost = true;
    s.reason = reason;
    s.message = to_string(message);
}

// True when every limit the harness requires was actually granted.
[[nodiscard]] bool meets_requirements(const DeviceLimits& granted) noexcept {
    return granted.max_buffer_size >= kRequirements.max_buffer_size
        && granted.max_storage_buffer_binding_size >=
               kRequirements.max_storage_buffer_binding_size
        && granted.max_storage_buffers_per_shader_stage >=
               kRequirements.max_storage_buffers_per_shader_stage
        && granted.max_compute_invocations_per_workgroup >=
               kRequirements.max_compute_invocations_per_workgroup
        && granted.max_compute_workgroups_per_dimension >=
               kRequirements.max_compute_workgroups_per_dimension;
}

}  // namespace

// Carries the caller's continuation across the two async hops. Heap allocated
// because the callbacks outlive the call that started them; freed exactly once
// on whichever path completes.
//
// The `delete this` below is deliberate rather than an escape from the
// UniqueHandle system. Ownership genuinely moves forward along the chain: the
// adapter callback must NOT destroy this object, because the device request it
// starts still needs it. Reclaiming into a unique_ptr at each callback entry
// would destroy it one hop too early. Only the terminal paths -- fail() and
// succeed() -- own the end of the chain, and each is reached exactly once.
struct PendingDeviceRequest {
    std::unique_ptr<Device> device;
    WGPULimits adapter_limits = {};
    WGPULimits required_limits = {};
    Device::RequestCallback callback;
#if BLLM_DIAGNOSTICS_ENABLED
    // What a diagnostic build asked for, and the descriptors that carry it,
    // held until the request completes (device.h's DiagnosticRequest).
    DiagnosticRequest diagnostic = {};
    std::vector<WGPUFeatureName> features;
    WGPURequestAdapterOptions adapter_options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
#if BLLM_GPU_DAWN_NATIVE
    WGPUDawnTogglesDescriptor adapter_toggles = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
    WGPUDawnTogglesDescriptor device_toggles = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
#endif
#endif
    void* userdata;

    void fail(const std::string& message) {
        auto* self = this;
        self->callback(nullptr, message.c_str(), self->userdata);
        delete self;
    }

    void succeed() {
        auto* self = this;
        self->callback(std::move(self->device), nullptr, self->userdata);
        delete self;
    }
};

void Device::request(RequestCallback callback, void* userdata) {
    const Instance instance{wgpuCreateInstance(nullptr)};
    if (!instance) {
        callback(nullptr, "could not create a WebGPU instance; this browser may not support WebGPU", userdata);
        return;
    }
    request(instance.get(), callback, userdata);
}

#if BLLM_DIAGNOSTICS_ENABLED
// Dawn's own toggles and experimental features exist natively alone, where
// CMakeLists.txt sets this 1; in the browser the port's webgpu.h has neither.
#if !defined(BLLM_GPU_DAWN_NATIVE)
#error "BLLM_GPU_DAWN_NATIVE is set by CMakeLists.txt: 1 natively, 0 in the browser"
#endif
namespace {
// Natively: unsafe APIs allowed, so an adapter lists timestamps inside a
// pass where it has them, and timestamps unrounded (device.h).
#if BLLM_GPU_DAWN_NATIVE
constexpr const char* kUnsafe[] = {"allow_unsafe_apis"};
constexpr const char* kUnrounded[] = {"timestamp_quantization"};
#endif
}  // namespace

void Device::request(WGPUInstance instance, RequestCallback callback, void* userdata,
                     const WGPURequestAdapterOptions* options, const DiagnosticRequest& diagnostic) {
    request_with(instance, callback, userdata, options, &diagnostic);
}
#endif

void Device::request(WGPUInstance instance, RequestCallback callback, void* userdata,
                     const WGPURequestAdapterOptions* options) {
#if BLLM_DIAGNOSTICS_ENABLED
    request_with(instance, callback, userdata, options, nullptr);
}

void Device::request_with(WGPUInstance instance, RequestCallback callback, void* userdata,
                          const WGPURequestAdapterOptions* options, const DiagnosticRequest* diagnostic) {
#endif
    if (instance == nullptr) {
        callback(nullptr, "no WebGPU instance was given", userdata);
        return;
    }
    // Designated initialisers: positional init here silently misaligned when a
    // field was added, and the compiler only caught it because the types
    // happened to disagree.
    auto* pending = new PendingDeviceRequest{
        .device = std::unique_ptr<Device>(new Device()),
        .adapter_limits = {},
        .required_limits = {},
        .callback = callback,
        .userdata = userdata};

    pending->device->instance_ = retain(instance);   // the Device's own reference
#if BLLM_DIAGNOSTICS_ENABLED
    if (diagnostic != nullptr && diagnostic->timestamps) {
        pending->diagnostic = *diagnostic;
        if (options != nullptr) pending->adapter_options = *options;
#if BLLM_GPU_DAWN_NATIVE
        pending->adapter_toggles.chain.next = pending->adapter_options.nextInChain;
        pending->adapter_toggles.enabledToggleCount = 1;
        pending->adapter_toggles.enabledToggles = kUnsafe;
        pending->adapter_options.nextInChain = &pending->adapter_toggles.chain;
#endif
        options = &pending->adapter_options;
    }
#endif

    WGPURequestAdapterCallbackInfo adapter_cb = {};
    adapter_cb.mode = kCallbackMode;
    adapter_cb.userdata1 = pending;
    adapter_cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                             WGPUStringView message, void* ud1, void*) {
        auto* p = static_cast<PendingDeviceRequest*>(ud1);
        if (status != WGPURequestAdapterStatus_Success || adapter == nullptr) {
            const std::string detail = to_string(message);
            p->fail("no WebGPU adapter available" + (detail.empty() ? "" : ": " + detail));
            return;
        }
        p->device->adapter_.reset(adapter);

        // Which backend the adapter is decides whether it computes at all:
        // Dawn's Null backend accepts work and does nothing (device.h). An
        // adapter that cannot say is refused too, rather than run on unseen.
        WGPUAdapterInfo info = {};
        if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
            p->fail("could not tell which backend the WebGPU adapter is; refusing an adapter "
                    "that may compute nothing");
            return;
        }
        const WGPUBackendType backend = info.backendType;
        p->device->adapter_info_ = AdapterInfo{
            to_string(info.vendor), to_string(info.architecture),
            to_string(info.device), to_string(info.description),
            backend_name(backend)};
        wgpuAdapterInfoFreeMembers(info);
        if (backend == WGPUBackendType_Null) {
            p->fail("the WebGPU adapter selected is Dawn's Null backend, which accepts work and "
                    "computes nothing");
            return;
        }

        // Adapter maxima are DIAGNOSTICS. They are recorded and reported, and
        // deliberately not used as the request: promoting whatever this
        // machine advertises into the requirement encodes its GPU into the
        // contract and fails elsewhere untraceably (WASM.10).
        // Stored on the pending request, not the stack: the device request is
        // asynchronous and the descriptor points at this until it completes.
        WGPULimits& adapter_limits = p->adapter_limits;
        if (wgpuAdapterGetLimits(adapter, &adapter_limits) != WGPUStatus_Success) {
            p->fail("could not query adapter limits; refusing to dispatch "
                    "against unknown device capabilities");
            return;
        }
        p->device->adapter_maxima_ = DeviceLimits{
            adapter_limits.maxBufferSize, adapter_limits.maxStorageBufferBindingSize,
            adapter_limits.maxComputeWorkgroupsPerDimension,
            adapter_limits.maxComputeInvocationsPerWorkgroup,
            adapter_limits.maxStorageBuffersPerShaderStage,
            adapter_limits.minStorageBufferOffsetAlignment};

        // Request exactly what the harness requires — no more, no less
        // (WASM.10). Every field left UNDEFINED takes the spec default, so
        // only the limits we have a stated need for appear here.
        WGPULimits& required = p->required_limits;
        required = WGPU_LIMITS_INIT;
        required.maxBufferSize = kRequirements.max_buffer_size;
        required.maxStorageBufferBindingSize = kRequirements.max_storage_buffer_binding_size;
        required.maxStorageBuffersPerShaderStage =
            kRequirements.max_storage_buffers_per_shader_stage;
        required.maxComputeInvocationsPerWorkgroup =
            kRequirements.max_compute_invocations_per_workgroup;
        required.maxComputeWorkgroupsPerDimension =
            kRequirements.max_compute_workgroups_per_dimension;

        WGPUDeviceDescriptor device_desc = {};
        device_desc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;
        device_desc.requiredLimits = &required;
#if BLLM_DIAGNOSTICS_ENABLED
        if (p->diagnostic.timestamps) {
            if (!wgpuAdapterHasFeature(adapter, WGPUFeatureName_TimestampQuery)) {
                p->fail("the adapter does not offer timestamp queries, which profiling needs");
                return;
            }
            p->features.push_back(WGPUFeatureName_TimestampQuery);
#if BLLM_GPU_DAWN_NATIVE
            if (wgpuAdapterHasFeature(adapter, WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses)) {
                p->features.push_back(WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses);
            }
#endif
            device_desc.requiredFeatureCount = p->features.size();
            device_desc.requiredFeatures = p->features.data();
#if BLLM_GPU_DAWN_NATIVE
            p->device_toggles.enabledToggleCount = 1;
            p->device_toggles.enabledToggles = kUnsafe;
            p->device_toggles.disabledToggleCount = 1;
            p->device_toggles.disabledToggles = kUnrounded;
            device_desc.nextInChain = &p->device_toggles.chain;
#endif
        }
#endif
        // The callback's own reference to the status, released by the callback.
        p->device->status_ = std::make_shared<DeviceStatus>();
        device_desc.deviceLostCallbackInfo.mode = kCallbackMode;
        device_desc.deviceLostCallbackInfo.callback = on_device_lost;
        device_desc.deviceLostCallbackInfo.userdata1 = new std::shared_ptr<DeviceStatus>(p->device->status_);

        WGPURequestDeviceCallbackInfo device_cb = {};
        device_cb.mode = kCallbackMode;
        device_cb.userdata1 = p;
        device_cb.callback = [](WGPURequestDeviceStatus s, WGPUDevice device,
                                WGPUStringView msg, void* inner_ud1, void*) {
            auto* q = static_cast<PendingDeviceRequest*>(inner_ud1);
            if (s != WGPURequestDeviceStatus_Success || device == nullptr) {
                const std::string detail = to_string(msg);
                q->fail("could not acquire a WebGPU device" +
                        (detail.empty() ? "" : ": " + detail));
                return;
            }
            q->device->device_.reset(device);
            q->device->queue_.reset(wgpuDeviceGetQueue(device));
#if BLLM_DIAGNOSTICS_ENABLED
            q->device->timestamps_ = wgpuDeviceHasFeature(device, WGPUFeatureName_TimestampQuery);
#if BLLM_GPU_DAWN_NATIVE
            q->device->timestamps_inside_passes_ =
                wgpuDeviceHasFeature(device, WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses);
#endif
#endif

            // Read back what was actually granted rather than assuming the
            // request was honoured wholesale. These are the limits validation
            // enforces, and the only ones work may be sized against.
            //
            // With requiredLimits set to the adapter's own maxima these should
            // match adapter_maxima(), and the page shows both so a divergence
            // is visible rather than silently assumed away.
            WGPULimits device_limits = {};
            if (wgpuDeviceGetLimits(device, &device_limits) != WGPUStatus_Success) {
                q->fail("could not query device limits; refusing to dispatch "
                        "against unknown device capabilities");
                return;
            }
            q->device->limits_ = DeviceLimits{
                device_limits.maxBufferSize, device_limits.maxStorageBufferBindingSize,
                device_limits.maxComputeWorkgroupsPerDimension,
                device_limits.maxComputeInvocationsPerWorkgroup,
                device_limits.maxStorageBuffersPerShaderStage,
                device_limits.minStorageBufferOffsetAlignment};

            // A conforming implementation cannot grant less than was required,
            // so this should be unreachable. Checked anyway: silently planning
            // against capacity we were not given is the failure mode this whole
            // arrangement exists to prevent, and it would be invisible.
            if (!meets_requirements(q->device->limits_)) {
                q->fail("device granted less than the harness requires; "
                        "the request succeeded but the limits do not satisfy it");
                return;
            }
            q->succeed();
        };
        wgpuAdapterRequestDevice(adapter, &device_desc, device_cb);
    };

    wgpuInstanceRequestAdapter(pending->device->instance_.get(), options, adapter_cb);
}

}  // namespace bllm::gpu
