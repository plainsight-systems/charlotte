#include "core/residency/upload_check.h"

#include <algorithm>
#include <utility>

#include "core/gpu/callback_mode.h"
#include "core/gpu/userdata.h"
#include "core/residency/mapping.h"

namespace bllm::residency {

// What every callback of one check shares, and all it touches: its own
// references, so it outlives the check and whoever else holds the device or
// the instance.
struct CheckState {
    gpu::Instance instance;
    gpu::DeviceHandle device;
    gpu::Queue queue;
    std::shared_ptr<const gpu::DeviceStatus> status;
    gpu::Buffer staging;   // MAP_READ | COPY_DST
    bool cancelled = false;
    bool comparing = false;   // a chunk's readback is mapped or pending
    CheckError failed = CheckError::Ok;
    std::vector<Mismatch> mismatches;
};

namespace {

using gpu::hand_off;
using gpu::take_back;

CheckError from_write(WriteError error) {
    switch (error) {
        case WriteError::Ok: return CheckError::Ok;
        case WriteError::OutOfOrder: return CheckError::OutOfOrder;
        case WriteError::ChunkTooLarge: return CheckError::ChunkTooLarge;
        case WriteError::Unfinished: return CheckError::Unfinished;
    }
    return CheckError::Internal;
}

// A mapping's outcome as the check reports it (upload_check.h).
CheckError from_mapped(Mapped mapped) {
    switch (mapped) {
        case Mapped::Ok: return CheckError::Ok;
        case Mapped::Cancelled: return CheckError::Cancelled;
        case Mapped::Internal: return CheckError::Internal;
        case Mapped::DeviceLost: return CheckError::DeviceLost;
        case Mapped::Unexplained: return CheckError::MapFailed;
    }
    return CheckError::Internal;
}

// The route whose piece covers `offset` in `buffer`, or null for padding.
const Route* route_at(const Upload& upload, BufferIndex buffer, std::uint64_t offset) {
    for (const Route& r : upload.routes()) {
        if (r.buffer == buffer && offset >= r.buffer_offset && offset - r.buffer_offset < r.length) return &r;
    }
    return nullptr;
}

// One chunk's readback: the writes it compares against, whose spans point
// into the writer's staging, intact until it accepts.
struct Comparison {
    std::shared_ptr<CheckState> state;
    const Upload* upload;
    std::vector<Write> writes;
    std::uint64_t bytes;
    CheckCallback accepted;
    void* userdata;
};

void compare(WGPUMapAsyncStatus status, Comparison& c) {
    CheckState& s = *c.state;
    const void* range = status == WGPUMapAsyncStatus_Success
                            ? wgpuBufferGetConstMappedRange(s.staging.get(), 0, c.bytes)
                            : nullptr;
    const Mapped mapped = mapping_result(status, range != nullptr, *s.status);
    if (mapped != Mapped::Ok) {
        if (range != nullptr) wgpuBufferUnmap(s.staging.get());
        if (s.failed == CheckError::Ok) s.failed = from_mapped(mapped);
        c.accepted(s.failed, c.userdata);
        return;
    }
    const auto* device = static_cast<const std::byte*>(range);
    std::uint64_t at = 0;
    for (const Write& w : c.writes) {
        // A write may span pieces (piece_writer.h): the first differing byte
        // of each is reported, so every tensor a write covers is named.
        const auto* held = device + at;
        std::uint64_t i = 0;
        while (i < w.bytes.size()) {
            const auto from = w.bytes.begin() + static_cast<std::ptrdiff_t>(i);
            const auto mine = std::mismatch(from, w.bytes.end(), held + i).first;
            if (mine == w.bytes.end()) break;
            const std::uint64_t offset = w.offset + static_cast<std::uint64_t>(mine - w.bytes.begin());
            const Route* route = route_at(*c.upload, w.buffer, offset);
            s.mismatches.push_back(
                {w.buffer, offset, route ? std::string(c.upload->tensor_name(route->tensor)) : std::string()});
            i = (route ? route->buffer_offset + route->length : offset + 1) - w.offset;
        }
        at += w.bytes.size();
    }
    wgpuBufferUnmap(s.staging.get());
    c.accepted(CheckError::Ok, c.userdata);
}

}  // namespace

UploadCheck::UploadCheck(const Upload& upload, std::size_t max_chunk)
    : instance_(gpu::retain(upload.instance())),
      device_(gpu::retain(upload.device())),
      upload_(upload),
      writer_(upload.routes(), max_chunk, upload.ranges()),
      pending_(std::make_shared<CheckState>()) {
    CheckState& s = *pending_;
    s.instance = gpu::retain(upload.instance());
    s.device = gpu::retain(upload.device());
    s.queue = gpu::Queue(wgpuDeviceGetQueue(upload.device()));
    s.status = upload.device_status();
    // Every write is a whole number of words, so their sum is too.
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    desc.size = (PieceWriter::staging_bound(upload.routes().size(), max_chunk) + 3) / 4 * 4;
    s.staging = gpu::Buffer(wgpuDeviceCreateBuffer(upload.device(), &desc));
    if (!s.staging) s.failed = CheckError::Internal;   // no readback, so no claim
}

UploadCheck::~UploadCheck() noexcept { pending_->cancelled = true; }

void UploadCheck::check(std::uint64_t file_offset, std::span<const std::byte> chunk, CheckCallback accepted,
                        void* userdata) {
    CheckState& s = *pending_;
    if (s.failed == CheckError::Ok && s.comparing) {
        // The page sends the next chunk only on acceptance.
        s.failed = CheckError::OutOfOrder;
    }
    if (s.failed != CheckError::Ok) {
        accepted(s.failed, userdata);
        return;
    }
    writes_.clear();
    if (const WriteError e = writer_.accept(file_offset, chunk, writes_); e != WriteError::Ok) {
        s.failed = from_write(e);
        accepted(s.failed, userdata);
        return;
    }
    if (writes_.empty()) {
        accepted(CheckError::Ok, userdata);   // nothing of the chunk was routed
        return;
    }

    // Every write's range, back to back, in one command buffer and one mapping.
    const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(device_.get(), nullptr));
    std::uint64_t at = 0;
    for (const Write& w : writes_) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), upload_.buffer(w.buffer), w.offset, s.staging.get(), at,
                                             w.bytes.size());
        at += w.bytes.size();
    }
    const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
    WGPUCommandBuffer raw = commands.get();
    wgpuQueueSubmit(s.queue.get(), 1, &raw);

    s.comparing = true;
    auto record = std::make_unique<Comparison>(
        Comparison{pending_, &upload_, std::exchange(writes_, {}), at, accepted, userdata});
    WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    info.mode = gpu::kCallbackMode;
    info.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        const auto c = take_back<Comparison>(userdata1);
        CheckState& s = *c->state;
        s.comparing = false;
        // A check that is gone took the writes' staging with it: nothing is
        // compared, and nothing it pointed at is touched.
        if (s.cancelled) {
            c->accepted(CheckError::Cancelled, c->userdata);
            return;
        }
        compare(status, *c);
    };
    info.userdata1 = hand_off(std::move(record));
    wgpuBufferMapAsync(s.staging.get(), WGPUMapMode_Read, 0, at, info);
}

CheckError UploadCheck::finish(std::uint64_t file_size) {
    CheckState& s = *pending_;
    // A chunk still being compared is not yet compared.
    if (s.failed == CheckError::Ok && s.comparing) s.failed = CheckError::Unfinished;
    if (s.failed == CheckError::Ok) s.failed = from_write(writer_.finish(file_size));
    return s.failed;
}

std::span<const Mismatch> UploadCheck::mismatches() const noexcept { return pending_->mismatches; }

}  // namespace bllm::residency
