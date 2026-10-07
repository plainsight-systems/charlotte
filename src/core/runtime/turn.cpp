#include "core/runtime/turn.h"

#include <algorithm>

#include "core/kernels/order.h"
#include "core/residency/plan.h"

namespace bllm::runtime {

namespace {

constexpr std::uint64_t kOutstanding = kernels::Order::kOutstanding;

}  // namespace

Turn::Turn(std::uint32_t cached, std::uint32_t prompt, std::uint32_t capacity, std::uint32_t max_tokens) noexcept
    : cached_(cached),
      prompt_(prompt),
      capacity_(capacity),
      max_tokens_(max_tokens),
      blocks_((prompt - cached + residency::kPrefillBlock - 1) / residency::kPrefillBlock) {}

std::optional<Planned> Turn::next() noexcept {
    if (ended_ || run_ - reported_ >= kOutstanding) return std::nullopt;
    if (prefilled_ < prompt_ - cached_) {
        const std::uint32_t first = cached_ + prefilled_;
        const std::uint32_t tokens = std::min(residency::kPrefillBlock, prompt_ - first);
        prefilled_ += tokens;
        ++run_;
        return Planned{first, tokens, first + tokens == prompt_, false, first};
    }
    // Decode step j feeds draw j at position prompt + j: never at or past
    // the context, and only when its own draw, j + 1, is within the limit.
    const std::uint32_t j = decodes_;
    if (prompt_ + j >= capacity_ || j + 1 >= max_tokens_) return std::nullopt;
    ++decodes_;
    ++run_;
    return Planned{prompt_ + j, 1, true, true, 0};
}

Draw Turn::report(const std::optional<sampler::SampledRecord>& record, bool stop) noexcept {
    ++reported_;
    if (!record) return Draw::None;
    const std::uint32_t i = drawn_++;
    if (ended_) return Draw::Discard;
    if (record->failed != 0) {
        ended_ = true;
        failure_ = TurnFailure::NonFinite;
        return Draw::Failed;
    }
    ++accepted_;
    if (stop) {
        ended_ = true;
        end_ = TurnEnd::Stop;
        return Draw::Stop;
    }
    ++emitted_;
    if (emitted_ == max_tokens_) {
        ended_ = true;
        end_ = TurnEnd::Limit;
    } else if (prompt_ + i >= capacity_) {   // no step can feed draw i
        ended_ = true;
        end_ = TurnEnd::Context;
    }
    return Draw::Emit;
}

void Turn::fail(StepFailure failure) noexcept {
    ++reported_;
    if (failure == StepFailure::Cancelled) {
        cancel();
        return;
    }
    if (failure_ != TurnFailure::None) return;
    ended_ = true;
    failure_ = failure == StepFailure::DeviceLost ? TurnFailure::DeviceLost : TurnFailure::Step;
}

void Turn::cancel() noexcept {
    if (ended_) return;
    ended_ = true;
    end_ = TurnEnd::Cancelled;
}

bool Turn::ended() const noexcept { return ended_; }

bool Turn::finished() const noexcept { return ended_ && reported_ == run_; }

TurnFailure Turn::failure() const noexcept { return failure_; }

TurnEnd Turn::end() const noexcept { return end_; }

std::uint32_t Turn::kept() const noexcept {
    if (failure_ == TurnFailure::Step || failure_ == TurnFailure::DeviceLost) return 0;
    return std::min(written(), prompt_ + accepted_);
}

std::uint32_t Turn::emitted() const noexcept { return emitted_; }

std::uint32_t Turn::written() const noexcept { return cached_ + prefilled_ + decodes_; }

}  // namespace bllm::runtime
