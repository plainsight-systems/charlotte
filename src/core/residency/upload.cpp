#include "core/residency/upload.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

#include "core/diagnostics.h"
#include "core/gpu/callback_mode.h"
#include "core/gpu/userdata.h"

namespace bllm::residency {

// What every callback of one Upload shares, and all it touches: its own
// references, so it outlives the Upload and whoever else holds the device or
// the instance (R.20, R.21).
struct UploadState {
    gpu::Instance instance;
    gpu::DeviceHandle device;
    gpu::Queue queue;
    std::shared_ptr<const gpu::DeviceStatus> status;
    gpu::Buffer witness;                       // MAP_READ | COPY_DST, 4 bytes
    std::array<std::byte, 4> witness_bytes{};
    bool cancelled = false;                    // the Upload is gone
    UploadError failed = UploadError::Ok;      // the first failure; later calls report it

    // Chunk pacing: chunk n may be sent on once n chunks' writes are done.
    struct Waiting {
        std::uint64_t needs;
        UploadCallback callback;
        void* userdata;
    };
    std::uint64_t chunks_done = 0;
    std::optional<Waiting> waiting;

    // Chunks whose scopes have not yet reported, and a finish waiting on them.
    std::uint64_t scopes_pending = 0;
    struct Finish {
        UploadCallback done;
        void* userdata;
    };
    std::optional<Finish> finish_waiting;

    // What a callback reports to its caller: Cancelled once the Upload is
    // gone, else the first failure.
    [[nodiscard]] UploadError outcome() const noexcept { return cancelled ? UploadError::Cancelled : failed; }
};

namespace {

using gpu::hand_off;
using gpu::take_back;

std::string to_string(WGPUStringView view) {
    if (view.data == nullptr) return {};
    return view.length == WGPU_STRLEN ? std::string(view.data) : std::string(view.data, view.length);
}

UploadError from_route(RouteError error) {
    switch (error) {
        case RouteError::Ok: return UploadError::Ok;
        case RouteError::UnsupportedFormat: return UploadError::UnsupportedFormat;
        case RouteError::OutOfRange: return UploadError::OutOfRange;
        case RouteError::NotACandidate: return UploadError::NotACandidate;
        case RouteError::RowNotSteppable: return UploadError::RowNotSteppable;
    }
    return UploadError::Internal;
}

UploadError from_write(WriteError error) {
    switch (error) {
        case WriteError::Ok: return UploadError::Ok;
        case WriteError::OutOfOrder: return UploadError::OutOfOrder;
        case WriteError::ChunkTooLarge: return UploadError::ChunkTooLarge;
        case WriteError::Unfinished: return UploadError::Unfinished;
    }
    return UploadError::Internal;
}

// A popped scope: the pop's status, then the error type it caught (upload.h).
UploadError from_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type) {
    if (status == WGPUPopErrorScopeStatus_CallbackCancelled) return UploadError::Cancelled;
    if (status != WGPUPopErrorScopeStatus_Success) return UploadError::Internal;
    switch (type) {
        case WGPUErrorType_NoError: return UploadError::Ok;
        case WGPUErrorType_OutOfMemory: return UploadError::OutOfMemory;
        case WGPUErrorType_Validation: return UploadError::Validation;
        default: return UploadError::Internal;   // Internal, Unknown, and any later type
    }
}

UploadError from_work_done(WGPUQueueWorkDoneStatus status) {
    switch (status) {
        case WGPUQueueWorkDoneStatus_Success: return UploadError::Ok;
        case WGPUQueueWorkDoneStatus_CallbackCancelled: return UploadError::Cancelled;
        default: return UploadError::Internal;
    }
}

WGPUBufferUsage usage(Pool pool) {
    switch (pool) {
        case Pool::Weights:
            // A diagnostic build reads the weights back (upload_check.h).
            return WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                   (diagnostics_enabled() ? WGPUBufferUsage_CopySrc : WGPUBufferUsage_None);
        case Pool::Cache: return WGPUBufferUsage_Storage;
        // Written as well as read back: a vision step's patches are written
        // into `pixels`, and its rows copied into a text step's `hidden`
        // (kernels/program.h, runtime/vision.h).
        case Pool::Scratch: return WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    }
    return WGPUBufferUsage_None;
}

// Four bytes from the file's size, never all zero: the witness buffer is new,
// and WebGPU zeroed it, so reading these back shows the write ran.
std::array<std::byte, 4> witness_bytes(std::uint64_t file_size) {
    const auto v = static_cast<std::uint32_t>(file_size ^ (file_size >> 32)) | 1u;
    return {std::byte(v & 0xFF), std::byte((v >> 8) & 0xFF), std::byte((v >> 16) & 0xFF),
            std::byte((v >> 24) & 0xFF)};
}

// The results of scopes popped together, kept by the order they were pushed,
// so which error is reported does not depend on the order the pops complete.
template <std::size_t N>
struct Scopes {
    std::array<UploadError, N> errors{};
    std::array<std::string, N> messages;
    std::size_t remaining = N;

    // The first scope pushed that caught something.
    [[nodiscard]] std::size_t first_failed() const {
        for (std::size_t i = 0; i < N; ++i) {
            if (errors[i] != UploadError::Ok) return i;
        }
        return N;
    }
};

// Pops `Job::kScopes` scopes, innermost first, and calls job->popped() once
// every result has arrived. Slot i holds the scope pushed i-th.
template <typename Job>
void pop_scopes(WGPUDevice device, std::shared_ptr<Job> job) {
    struct Pop {
        std::shared_ptr<Job> job;
        std::size_t slot;
    };
    for (std::size_t i = Job::kScopes; i-- > 0;) {
        WGPUPopErrorScopeCallbackInfo info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message,
                           void* userdata, void*) {
            const auto pop = take_back<Pop>(userdata);
            Job& j = *pop->job;
            j.scopes.errors[pop->slot] = from_scope(status, type);
            j.scopes.messages[pop->slot] = to_string(message);
            if (--j.scopes.remaining == 0) j.popped();
        };
        info.userdata1 = hand_off(std::make_unique<Pop>(Pop{job, i}));
        wgpuDevicePopErrorScope(device, info);
    }
}

// Creation's scopes, pushed in this order: out of memory, validation,
// internal.
struct Creation {
    static constexpr std::size_t kScopes = 3;
    Scopes<kScopes> scopes;
    std::unique_ptr<Upload> upload;
    std::string refused;   // a buffer creation returned null for, if any
    ReadyCallback ready;
    void* userdata;

    void popped() {
        if (const std::size_t i = scopes.first_failed(); i < kScopes) {
            ready(nullptr, scopes.errors[i], scopes.messages[i].c_str(), userdata);
        } else if (!refused.empty()) {
            ready(nullptr, UploadError::OutOfMemory, refused.c_str(), userdata);
        } else {
            ready(std::move(upload), UploadError::Ok, "", userdata);
        }
    }
};

// Reads the witness back once it is written: the one Ok finish gives.
struct WitnessMap {
    std::shared_ptr<UploadState> state;
    UploadCallback done;
    void* userdata;
};

void map_witness(std::shared_ptr<UploadState> state, UploadCallback done, void* userdata) {
    UploadState& s = *state;
    wgpuQueueWriteBuffer(s.queue.get(), s.witness.get(), 0, s.witness_bytes.data(), s.witness_bytes.size());
    WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    info.mode = gpu::kCallbackMode;
    info.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        const auto m = take_back<WitnessMap>(userdata1);
        UploadState& s = *m->state;
        if (s.cancelled) {
            m->done(UploadError::Cancelled, m->userdata);
            return;
        }
        const void* range = status == WGPUMapAsyncStatus_Success
                                ? wgpuBufferGetConstMappedRange(s.witness.get(), 0, s.witness_bytes.size())
                                : nullptr;
        const std::span<const std::byte> mapped =
            range != nullptr ? std::span(static_cast<const std::byte*>(range), s.witness_bytes.size())
                             : std::span<const std::byte>();
        const UploadError result =
            witness_result(mapping_result(status, range != nullptr, *s.status), mapped, s.witness_bytes);
        if (range != nullptr) wgpuBufferUnmap(s.witness.get());
        if (s.failed == UploadError::Ok) s.failed = result;
        // The first failure, whichever callback recorded it: a chunk's work
        // done may land after the witness was written.
        m->done(s.outcome(), m->userdata);
    };
    info.userdata1 = hand_off(std::make_unique<WitnessMap>(WitnessMap{std::move(state), done, userdata}));
    wgpuBufferMapAsync(s.witness.get(), WGPUMapMode_Read, 0, s.witness_bytes.size(), info);
}

// The rest of finish, once every chunk's scopes have reported: the first
// failure, or the witness. Cancellation is decided once, where the witness's
// mapping lands, so an Upload destroyed before then reports Cancelled.
void finish_now(std::shared_ptr<UploadState> state, UploadCallback done, void* userdata) {
    UploadState& s = *state;
    if (s.failed != UploadError::Ok) {
        done(s.outcome(), userdata);
        return;
    }
    map_witness(std::move(state), done, userdata);
}

// One chunk's scopes, pushed in this order: validation, internal.
struct ChunkScopes {
    static constexpr std::size_t kScopes = 2;
    Scopes<kScopes> scopes;
    std::shared_ptr<UploadState> state;

    void popped() {
        UploadState& s = *state;
        if (const std::size_t i = scopes.first_failed(); i < kScopes && s.failed == UploadError::Ok) {
            s.failed = scopes.errors[i];
        }
        if (--s.scopes_pending == 0 && s.finish_waiting) {
            const auto finish = *std::exchange(s.finish_waiting, std::nullopt);
            finish_now(std::move(state), finish.done, finish.userdata);
        }
    }
};

// Counts a chunk's writes done, and accepts the chunk waiting on it.
struct WorkDone {
    std::shared_ptr<UploadState> state;
};

void request_work_done(const std::shared_ptr<UploadState>& state) {
    WGPUQueueWorkDoneCallbackInfo info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    info.mode = gpu::kCallbackMode;
    info.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView, void* userdata1, void*) {
        const auto w = take_back<WorkDone>(userdata1);
        UploadState& s = *w->state;
        if (const UploadError e = from_work_done(status); e != UploadError::Ok && s.failed == UploadError::Ok) {
            s.failed = e;
        }
        ++s.chunks_done;
        if (s.waiting && (s.chunks_done >= s.waiting->needs || s.outcome() != UploadError::Ok)) {
            const auto waiting = *std::exchange(s.waiting, std::nullopt);
            waiting.callback(s.outcome(), waiting.userdata);
        }
    };
    info.userdata1 = hand_off(std::make_unique<WorkDone>(WorkDone{state}));
    wgpuQueueOnSubmittedWorkDone(state->queue.get(), info);
}

}  // namespace

Upload::Upload(const gpu::Device& device, ResidencyPlan plan, std::vector<Route> routes,
               std::uint64_t file_size, std::size_t max_chunk, std::span<const FileRange> ranges)
    : instance_(gpu::retain(device.instance())),
      device_(gpu::retain(device.handle())),
      device_status_(device.status()),
      plan_(std::move(plan)),
      routes_(std::move(routes)),
      ranges_(ranges.begin(), ranges.end()),
      writer_(routes_, max_chunk, ranges_),
      file_size_(file_size),
      pending_(std::make_shared<UploadState>()) {
    pending_->instance = gpu::retain(device.instance());
    pending_->device = gpu::retain(device.handle());
    pending_->queue = gpu::Queue(wgpuDeviceGetQueue(device.handle()));
    pending_->status = device_status_;
    pending_->witness_bytes = witness_bytes(file_size);
}

void Upload::begin(const gpu::Device& device, const gguf::TensorIndex& index, const ResidencyPlan& plan,
                   std::uint64_t file_size, FindFormat find_format,
                   std::span<const gguf::TensorId> confirmed_duplicates, std::size_t max_chunk,
                   ReadyCallback ready, void* userdata, std::span<const FileRange> ranges) {
    std::vector<Route> routes;
    ResidencyPlan carried;
    if (const auto r =
            plan_routes(index, plan, file_size, find_format, confirmed_duplicates, routes, carried, ranges);
        !r.ok()) {
        ready(nullptr, from_route(r.error), r.subject.c_str(), userdata);
        return;
    }

    // The constructor is private, so make_unique cannot reach it; the
    // pointer is owned from the expression that makes it (R.11).
    auto job = std::make_shared<Creation>();
    job->upload.reset(new Upload(device, std::move(carried), std::move(routes), file_size, max_chunk, ranges));
    job->ready = ready;
    job->userdata = userdata;
    Upload& u = *job->upload;

    u.tensor_names_.resize(index.tensors().size());
    for (const Route& route : u.routes_) {
        u.tensor_names_[static_cast<std::size_t>(route.tensor)] = index.tensor(route.tensor).name;
    }

    WGPUDevice dev = u.device_.get();
    wgpuDevicePushErrorScope(dev, WGPUErrorFilter_OutOfMemory);
    wgpuDevicePushErrorScope(dev, WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(dev, WGPUErrorFilter_Internal);

    // wgpuDeviceCreateBuffer is declared nullable, so a null is a refusal
    // too, named by the buffer, alongside whatever the scopes catch.
    u.buffers_.resize(u.plan_.buffers.size());
    for (std::size_t i = 0; i < u.plan_.buffers.size(); ++i) {
        const PlannedBuffer& planned = u.plan_.buffers[i];
        if (planned.size == 0) continue;   // a confirmed duplicate's, left uncreated
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.usage = usage(planned.pool);
        desc.size = planned.size;
        u.buffers_[i] = gpu::Buffer(wgpuDeviceCreateBuffer(dev, &desc));
        if (!u.buffers_[i] && job->refused.empty()) job->refused = "buffer " + std::to_string(i);
    }
    WGPUBufferDescriptor witness = WGPU_BUFFER_DESCRIPTOR_INIT;
    witness.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    witness.size = u.pending_->witness_bytes.size();
    u.pending_->witness = gpu::Buffer(wgpuDeviceCreateBuffer(dev, &witness));
    if (!u.pending_->witness && job->refused.empty()) job->refused = "the witness buffer";

    pop_scopes(dev, std::move(job));
}

Upload::~Upload() noexcept { pending_->cancelled = true; }

void Upload::write(std::uint64_t file_offset, std::span<const std::byte> chunk, UploadCallback accepted,
                   void* userdata) {
    UploadState& s = *pending_;
    if (finishing_) {
        accepted(UploadError::OutOfOrder, userdata);   // refused, and nothing changes
        return;
    }
    if (s.failed == UploadError::Ok && s.waiting) {
        // The page sends the next chunk only on acceptance; one sent before
        // is out of order, and the chunk still waiting hears it too.
        s.failed = UploadError::OutOfOrder;
    }
    if (s.failed != UploadError::Ok) {
        accepted(s.failed, userdata);
        return;
    }
    writes_.clear();
    if (const WriteError e = writer_.accept(file_offset, chunk, writes_); e != WriteError::Ok) {
        s.failed = from_write(e);
        accepted(s.failed, userdata);
        return;
    }

    // The scopes open and close within this call, so none is left on the
    // device's stack while the page reads the next chunk.
    wgpuDevicePushErrorScope(device_.get(), WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(device_.get(), WGPUErrorFilter_Internal);
    for (const Write& w : writes_) {
        wgpuQueueWriteBuffer(s.queue.get(), buffers_[static_cast<std::size_t>(w.buffer)].get(), w.offset,
                             w.bytes.data(), w.bytes.size());
    }
    ++s.scopes_pending;
    auto scopes = std::make_shared<ChunkScopes>();
    scopes->state = pending_;
    pop_scopes(device_.get(), std::move(scopes));
    const std::uint64_t chunk_number = chunks_written_++;
    request_work_done(pending_);

    // Optimization (practice): chunk n is accepted once chunk n - 1 is done,
    // not chunk n, so the page reads the next chunk while the GPU copies
    // this one (GPU.7).
    if (s.chunks_done >= chunk_number) {
        accepted(UploadError::Ok, userdata);
    } else {
        s.waiting = UploadState::Waiting{chunk_number, accepted, userdata};
    }
}

void Upload::finish(UploadCallback done, void* userdata) {
    if (finishing_) {
        done(UploadError::OutOfOrder, userdata);   // refused, and nothing changes
        return;
    }
    finishing_ = true;
    UploadState& s = *pending_;
    if (s.failed == UploadError::Ok) s.failed = from_write(writer_.finish(file_size_));
    if (s.scopes_pending == 0) {
        finish_now(pending_, done, userdata);
    } else {
        s.finish_waiting = UploadState::Finish{done, userdata};
    }
}

WGPUBuffer Upload::buffer(BufferIndex index) const noexcept {
    const auto i = static_cast<std::size_t>(index);
    return i < buffers_.size() ? buffers_[i].get() : nullptr;
}

std::string_view Upload::tensor_name(gguf::TensorId tensor) const noexcept {
    const auto i = static_cast<std::size_t>(tensor);
    return i < tensor_names_.size() ? std::string_view(tensor_names_[i]) : std::string_view();
}

}  // namespace bllm::residency
