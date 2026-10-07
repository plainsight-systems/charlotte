#include "core/kernels/schedule.h"

#include <algorithm>

namespace bllm::kernels {

Schedules::Schedules(std::span<const Geometry> launches) {
    // Every token count where some launch starts or stops running.
    bounds_.push_back(1);
    for (const Geometry& g : launches) {
        if (g.tokens.least > 1 && g.tokens.least <= residency::kPrefillBlock) bounds_.push_back(g.tokens.least);
        if (g.tokens.most < residency::kPrefillBlock) bounds_.push_back(g.tokens.most + 1);
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
            if (tokens < g.tokens.least || tokens > g.tokens.most) continue;
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
