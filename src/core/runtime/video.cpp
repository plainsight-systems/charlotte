#include "core/runtime/video.h"

#include <algorithm>
#include <utility>

namespace bllm::runtime {

Video::Video(std::shared_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
             kernels::Binding pixels, std::uint32_t patch, std::uint32_t pooling, std::uint32_t room)
    : program_(std::move(program)),
      link_(std::move(link)),
      pixels_(pixels),
      patch_(patch),
      pooling_(pooling),
      room_(room) {
    // Every slot holds a frame's most patches, at load (MEM.9).
    const std::size_t floats = std::size_t{media::kFrameTokens} * pooling * pooling * 3 * patch * patch;
    for (ClipSet& set : sets_) {
        for (FrameSlot& slot : set.slots) slot.patches.resize(floats);
    }
}

Video::~Video() = default;

namespace {

// A frame size's refusal, before any grid is computed for it.
media::ImageError size_error(std::uint32_t width, std::uint32_t height) noexcept {
    if (width == 0 || height == 0) return media::ImageError::Empty;
    if (width > media::kMaxSide || height > media::kMaxSide) return media::ImageError::TooLarge;
    return media::ImageError::Ok;
}

}  // namespace

VideoResult Video::plan(std::uint32_t width, std::uint32_t height, double duration, VideoPlan& out) const noexcept {
    if (const auto e = size_error(width, height); e != media::ImageError::Ok) return {VideoError::Image, e};
    const std::uint32_t frames = media::frame_count(duration);
    if (frames == 0) return {VideoError::NoFrames, media::ImageError::Ok};
    out.frames = frames;
    out.grid = media::frame_grid(width, height, patch_, pooling_, media::kFrameTokens);
    out.clip_frames = media::frames_per_clip(out.grid.tokens, room_);
    return {};
}

VideoResult Video::begin(std::uint32_t width, std::uint32_t height, std::uint32_t frames) {
    // The set no encode is using: at most one is (video.h).
    filling_ = sets_[filling_].in_use ? 1 - filling_ : filling_;
    ClipSet& set = sets_[filling_];
    set.frames = 0;
    for (FrameSlot& slot : set.slots) slot.ready = false;
    if (const auto e = size_error(width, height); e != media::ImageError::Ok) return {VideoError::Image, e};
    const media::ImageGrid grid = media::frame_grid(width, height, patch_, pooling_, media::kFrameTokens);
    if (frames == 0 || frames > media::frames_per_clip(grid.tokens, room_)) {
        return {VideoError::TooManyFrames, media::ImageError::Ok};
    }
    const media::ImageFit fit = media::frame_fit(grid);
    for (std::uint32_t i = 0; i < frames; ++i) {
        if (const auto e = set.slots[i].resampler.begin(width, height, grid, fit, patch_, media::kFrameTokens);
            e != media::ImageError::Ok) {
            return {VideoError::Image, e};
        }
        set.segments[i] = grid.tokens;
    }
    set.grid = grid;
    set.frames = frames;
    return {};
}

VideoResult Video::rows(std::uint32_t frame, std::uint32_t first_row, std::span<const std::uint8_t> rgba) {
    ClipSet& set = sets_[filling_];
    if (set.in_use || frame >= set.frames) return {VideoError::BadFrame, media::ImageError::Ok};
    FrameSlot& slot = set.slots[frame];
    if (const auto e = slot.resampler.accept(rgba, first_row); e != media::ImageError::Ok) {
        return {VideoError::Image, e};
    }
    const std::size_t patches = std::size_t{set.grid.columns} * set.grid.rows * 3 * patch_ * patch_;
    if (slot.resampler.patches(std::span(slot.patches).first(patches)) == media::ImageError::Ok) slot.ready = true;
    return {};
}

std::span<const std::uint32_t> Video::ready_segments() const noexcept {
    const ClipSet& set = sets_[filling_];
    if (set.in_use || set.frames == 0) return {};
    const bool ready = std::all_of(set.slots.begin(), set.slots.begin() + set.frames,
                                   [](const FrameSlot& slot) { return slot.ready; });
    if (!ready) return {};
    return std::span(set.segments).first(set.frames);
}

void Video::run(std::uint32_t segment, const SegmentCopy& out, kernels::StepCallback done, void* userdata) {
    // The encode takes the clip with its first frame.
    if (segment == 0) {
        running_ = filling_;
        sets_[running_].in_use = true;
    }
    ClipSet& set = sets_[running_];
    FrameSlot& slot = set.slots[segment];
    kernels::Step step{};
    step.position = 0;
    step.tokens = set.grid.columns * set.grid.rows;
    step.columns = set.grid.columns;
    const std::size_t floats = std::size_t{step.tokens} * 3 * patch_ * patch_;
    const kernels::RunInput input{pixels_, std::as_bytes(std::span(slot.patches).first(floats))};
    const kernels::RunCopy copy{link_.get(), 0, out.to_offset, out.bytes, kernels::CopyAt::AfterPass};
    // The write copies the frame's patches before run() returns: its slot
    // is free, and with the last frame, the set.
    program_->run(step, std::span(&input, 1), std::span(&copy, 1), done, userdata);
    slot.ready = false;
    if (segment + 1 == set.frames) {
        set.frames = 0;
        set.in_use = false;
    }
}

}  // namespace bllm::runtime
