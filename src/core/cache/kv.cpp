#include "core/cache/kv.h"

#include <algorithm>
#include <cstddef>

namespace bllm::cache {

KvCache::KvCache(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                 policy::CachePrecision precision, std::uint32_t capacity)
    : capacity_(capacity), precision_(precision) {
    layers_.reserve(plan.cache.size());
    for (std::size_t i = 0; i < plan.cache.size(); ++i) {
        layers_.push_back({plan.cache[i], model.layers[i].attention_window});
    }
}

std::uint32_t KvCache::length() const noexcept { return length_; }

std::uint32_t KvCache::capacity() const noexcept { return capacity_; }

policy::CachePrecision KvCache::precision() const noexcept { return precision_; }

std::uint32_t KvCache::truncate(std::uint32_t tokens) noexcept {
    for (const Layer& layer : layers_) {
        // The earlier positions the next query, at `tokens`, reads run from
        // tokens - window + 1 to tokens - 1; with a window of 1, or nothing
        // kept, there are none to have lost.
        if (tokens == 0 || layer.window <= 1) continue;
        const std::uint32_t slots = layer.planned.slots;
        const std::uint32_t oldest_held = written_ > slots ? written_ - slots : 0;
        const std::uint32_t oldest_read = tokens >= layer.window ? tokens - layer.window + 1 : 0;
        if (oldest_held > oldest_read) {
            length_ = 0;
            written_ = 0;
            return 0;
        }
    }
    length_ = tokens;
    if (tokens == 0) written_ = 0;
    return tokens;
}

void KvCache::advance(std::uint32_t tokens) noexcept {
    length_ += tokens;
    written_ = std::max(written_, length_);
}

const residency::PlannedCacheLayer& KvCache::layer(model::LayerIndex layer) const noexcept {
    return layers_[static_cast<std::size_t>(layer)].planned;
}

}  // namespace bllm::cache
