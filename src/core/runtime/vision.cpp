#include "core/runtime/vision.h"

#include <utility>

namespace bllm::runtime {

Vision::Vision(std::unique_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
               kernels::Binding pixels, media::ImageBudget budget, std::uint32_t patch, std::uint32_t pooling)
    : program_(std::move(program)),
      link_(std::move(link)),
      pixels_(pixels),
      budget_(budget),
      patch_(patch),
      pooling_(pooling) {
    // The slot holds the most patches the budget allows, at load (MEM.9).
    slot_.patches.resize(std::size_t{budget.most} * pooling * pooling * 3 * patch * patch);
}

Vision::~Vision() = default;

media::ImageError Vision::begin(std::uint32_t width, std::uint32_t height) {
    slot_.state = SlotState::Empty;
    if (width == 0 || height == 0) return media::ImageError::Empty;
    if (width > media::kMaxSide || height > media::kMaxSide) return media::ImageError::TooLarge;
    slot_.grid = media::image_grid(width, height, patch_, pooling_, budget_);
    if (const auto e = slot_.resampler.begin(width, height, slot_.grid, patch_, budget_.most);
        e != media::ImageError::Ok) {
        return e;
    }
    slot_.state = SlotState::Filling;
    return media::ImageError::Ok;
}

media::ImageError Vision::rows(std::uint32_t first_row, std::span<const std::uint8_t> rgba) {
    if (slot_.state != SlotState::Filling) return media::ImageError::BadBand;
    if (const auto e = slot_.resampler.accept(rgba, first_row); e != media::ImageError::Ok) {
        slot_.state = SlotState::Empty;
        return e;
    }
    const auto patches = std::size_t{slot_.grid.columns} * slot_.grid.rows * 3 * patch_ * patch_;
    if (slot_.resampler.patches(std::span(slot_.patches).first(patches)) == media::ImageError::Ok) {
        slot_.state = SlotState::Ready;
    }
    return media::ImageError::Ok;
}

std::uint32_t Vision::ready_rows() const noexcept {
    return slot_.state == SlotState::Ready ? slot_.grid.tokens : 0;
}

void Vision::run(kernels::StepCallback done, void* userdata) {
    kernels::Step step{};
    step.position = 0;
    step.tokens = slot_.grid.columns * slot_.grid.rows;
    step.columns = slot_.grid.columns;
    const std::size_t floats = std::size_t{step.tokens} * 3 * patch_ * patch_;
    const kernels::RunInput input{pixels_, std::as_bytes(std::span(slot_.patches).first(floats))};
    // The write copies the patches before run() returns: the slot is free.
    program_->run(step, std::span(&input, 1), {}, done, userdata);
    slot_.state = SlotState::Empty;
}

}  // namespace bllm::runtime
