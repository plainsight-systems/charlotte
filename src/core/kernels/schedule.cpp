#include "core/kernels/schedule.h"

#include <algorithm>

namespace bllm::kernels {

Schedules::Schedules(std::span<const Geometry> launches) {
    // Every token count where some launch starts or stops running.
    // A launch's range is in its rows, the step's tokens over its divisor:
    // in tokens, from least × d to (most + 1) × d − 1.
    bounds_.push_back(1);
    for (const Geometry& g : launches) {
        const std::uint64_t d = g.row_divisor;
        const std::uint64_t least = std::uint64_t{g.tokens.least} * d;
        const std::uint64_t past = (std::uint64_t{g.tokens.most} + 1) * d;
        if (least > 1 && least <= UINT32_MAX) bounds_.push_back(static_cast<std::uint32_t>(least));
        if (past <= UINT32_MAX) bounds_.push_back(static_cast<std::uint32_t>(past));
    }
    std::sort(bounds_.begin(), bounds_.end());
    bounds_.erase(std::unique(bounds_.begin(), bounds_.end()), bounds_.end());

    lists_.resize(2 * bounds_.size());
    for (std::size_t i = 0; i < bounds_.size(); ++i) {
        // Within an interval every launch runs at all of it or at none, so
        // its least token count stands for it.
        const std::uint32_t tokens = bounds_[i];
        for (std::uint32_t k = 0; k < launches.size(); ++k) {
            const Geometry& g = launches[k];
            const std::uint32_t rows = tokens / g.row_divisor;
            if (rows < g.tokens.least || rows > g.tokens.most) continue;
            if (g.rows != Rows::LastToken) lists_[2 * i].push_back(k);
            lists_[2 * i + 1].push_back(k);
        }
    }
}

std::span<const std::uint32_t> Schedules::of(std::uint32_t tokens, bool logits) const noexcept {
    const auto above = std::upper_bound(bounds_.begin(), bounds_.end(), tokens);
    const auto interval = static_cast<std::size_t>(above - bounds_.begin()) - 1;
    return lists_[2 * interval + (logits ? 1 : 0)];
}

}  // namespace bllm::kernels
