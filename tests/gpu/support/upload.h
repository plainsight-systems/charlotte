#pragma once

// Test-only: driving an Upload as the page does — a fixture loaded and
// planned, the buffers made, the file streamed a chunk at a time, each sent
// once the last is accepted, then finish — for the GPU tests of upload and
// its check. Each wait is pumped on the calling thread (pump.h), so the
// state a callback writes lives on the waiting frame.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/formats/format.h"
#include "core/gguf/reader.h"
#include "core/gpu/device.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "support/gguf_fixture.h"
#include "support/pump.h"

namespace bllm::testing {

using residency::Upload;
using residency::UploadError;

// Formats of the test's own: the capability table lists none until a
// format's unpack exists.
inline constexpr formats::Format kF32{formats::kF32Layout, "fn unpack_f32() {}"};
inline constexpr formats::Format kQ4_0{formats::kQ4_0Layout, "fn unpack_q4_0() {}"};

inline const formats::Format* both(gguf::TensorType type) noexcept {
    if (type == gguf::TensorType::F32) return &kF32;
    if (type == gguf::TensorType::Q4_0) return &kQ4_0;
    return nullptr;
}

inline const formats::Format* f32_only(gguf::TensorType type) noexcept {
    return type == gguf::TensorType::F32 ? &kF32 : nullptr;
}

struct Model {
    std::vector<std::byte> bytes;
    gguf::TensorIndex index;
    residency::ResidencyPlan plan;
};

inline Model load(const std::string& fixture, const policy::LoadPolicy& policy = {}) {
    Model m;
    m.bytes = testing::load_gguf_fixture(fixture);
    gguf::MemoryByteSource source{m.bytes};
    REQUIRE(gguf::read_index(source, m.index).error == gguf::ReadError::Ok);
    std::string_view name;
    REQUIRE(m.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    model::ModelDescription description;
    REQUIRE(capability::find_architecture(name)->describe(m.index, description).ok());
    (void)residency::plan_residency(m.index, description, residency::DeviceLimits{256ull << 20, 128ull << 20, 256},
                                    policy, m.plan);
    return m;
}

inline constexpr std::size_t kChunk = 256;

struct Ready {
    std::unique_ptr<Upload> upload;
    UploadError error = UploadError::Internal;
    std::string subject;
    bool done = false;
};

inline void on_ready(std::unique_ptr<Upload> upload, UploadError error, const char* subject, void* userdata) {
    auto& r = *static_cast<Ready*>(userdata);
    r.upload = std::move(upload);
    r.error = error;
    r.subject = subject;
    r.done = true;
}

inline std::unique_ptr<Upload> begin(const gpu::Device& device, const Model& m) {
    Ready ready;
    Upload::begin(device, m.index, m.plan, m.bytes.size(), both, {}, kChunk, on_ready, &ready);
    pump_until(device.instance(), ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == UploadError::Ok, ready.subject);
    REQUIRE(ready.upload != nullptr);
    return std::move(ready.upload);
}

// Counts every call, so "exactly once" is checked, not assumed.
struct Reported {
    UploadError error = UploadError::Internal;
    int calls = 0;
    bool done = false;
};

inline void on_reported(UploadError error, void* userdata) {
    auto& r = *static_cast<Reported*>(userdata);
    r.error = error;
    ++r.calls;
    r.done = true;
}

// Streams the file, sending each chunk only once the last is accepted, as
// the page does, and stops at the first chunk refused; returns its error.
inline UploadError stream(WGPUInstance instance, Upload& upload, const Model& m, std::size_t from = 0,
                   std::size_t to = SIZE_MAX) {
    for (std::size_t at = from; at < std::min(to, m.bytes.size()); at += kChunk) {
        Reported accepted;
        const auto chunk = std::span(m.bytes).subspan(at, std::min(kChunk, m.bytes.size() - at));
        upload.write(at, chunk, on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.calls == 1);
        if (accepted.error != UploadError::Ok) return accepted.error;
    }
    return UploadError::Ok;
}

inline UploadError finish(WGPUInstance instance, Upload& upload) {
    Reported finished;
    upload.finish(on_reported, &finished);
    pump_until(instance, finished.done, "finish");
    CHECK(finished.calls == 1);
    return finished.error;
}

}  // namespace bllm::testing
