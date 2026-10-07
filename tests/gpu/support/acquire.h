#pragma once

#include <memory>
#include <string>
#include <utility>

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include "core/gpu/device.h"
#include "support/pump.h"

namespace bllm::testing {

// Test-only: a device acquired as the harness acquires one, for the GPU
// tests that need a device. The request is pumped to completion on the
// calling thread (pump.h), so its callback writes into an Acquired on this
// stack frame, which outlives it. A device that cannot be had fails the test; it never
// skips.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     SF.2   A header file must not contain non-inline function definitions —
//            acquire is inline, so every test file may include it.
//     F.26   Use a unique_ptr<T> to transfer ownership where a pointer is
//            needed — the device is the test's, by unique_ptr.

struct Acquired {
    std::unique_ptr<gpu::Device> device;
    std::string error;
    bool done = false;
};

// A device from `instance`, as the harness acquires one: the limits it
// requires, granted and read back. Fails the test, with the reason, if there
// is none — no adapter is a failure here, never a skip.
inline std::unique_ptr<gpu::Device> acquire(WGPUInstance instance) {
    Acquired acquired;
    gpu::Device::request(
        instance,
        [](std::unique_ptr<gpu::Device> device, const char* error, void* userdata) {
            auto& a = *static_cast<Acquired*>(userdata);
            a.device = std::move(device);
            if (error != nullptr) a.error = error;
            a.done = true;
        },
        &acquired);
    pump_until(instance, acquired.done, "a device");
    REQUIRE_MESSAGE(acquired.device != nullptr, acquired.error);
    return std::move(acquired.device);
}

#if BLLM_DIAGNOSTICS_ENABLED
// The same, asking for what a diagnostic build may (gpu/device.h).
inline std::unique_ptr<gpu::Device> acquire(WGPUInstance instance, const gpu::DiagnosticRequest& diagnostic) {
    Acquired acquired;
    gpu::Device::request(
        instance,
        [](std::unique_ptr<gpu::Device> device, const char* error, void* userdata) {
            auto& a = *static_cast<Acquired*>(userdata);
            a.device = std::move(device);
            if (error != nullptr) a.error = error;
            a.done = true;
        },
        &acquired, nullptr, diagnostic);
    pump_until(instance, acquired.done, "a device");
    REQUIRE_MESSAGE(acquired.device != nullptr, acquired.error);
    return std::move(acquired.device);
}
#endif

}  // namespace bllm::testing
