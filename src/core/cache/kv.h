#pragma once

#include <cstdint>
#include <vector>

#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace bllm::cache {

// Axis G: changes with a new cache or context mechanism.
//
// Contract 8: the KV cache.
//
//   - A full-attention layer holds every token of the context offered. A
//     sliding-window layer holds a ring of the slots the residency plan gives
//     it: its window, a prefill block and the policy's rollback reserve.
//     Position p lives in slot p mod slots, in every layer.
//   - Truncation resets a counter and moves no data. A ring has overwritten
//     what lies further back than its slots, and recomputing those entries
//     needs the ones before them, back to the first token. So a rollback
//     within the reserve keeps the cache, and a deeper one empties it.
//     Exactly: a query at position p reads keys at p - window + 1 .. p, its
//     own position included, as llama.cpp and Hugging Face mask a sliding
//     layer. What a ring holds is set by how far it has been written, not by
//     the length: a rollback leaves the entries past it in their slots until
//     new tokens overwrite them, and they had already overwritten older ones.
//     So the cache keeps a high-water mark, the most tokens written since it
//     was last emptied, and a layer holds positions mark - slots .. mark - 1,
//     or all of them while mark <= slots. Truncating to t keeps
//     the cache when, in every layer, every earlier position the next query
//     at t reads — t - window + 1 .. t - 1 — is still held; the entries from
//     t on are overwritten in place as the new tokens are written. A
//     full-attention layer's slots are the capacity, so it never overwrites
//     and never decides; one rule serves both kinds of layer.
//   - Capacity is the context offered. Each layer's window comes from the
//     model description, storage precision from load policy, and packing from
//     the format for that precision.
//   - Storage is planned by the residency plan and created by upload. The
//     cache names its buffers by plan index and holds no GPU object, so its
//     bookkeeping is tested without a device.
//   - Attention reads a layer's window and writes the step's new entries; the
//     runtime then advances the cache by the step's token count.
//
// What it costs: truncate visits each layer once, once per turn — 28 for
// Qwen3, reading two counts from each, a few hundred bytes — and advance and
// layer are constant time, once per step. Nothing is allocated after
// construction, which copies the plan's layers once at load, so the
// per-token path allocates nothing (MEM.9); no key or value moves on the CPU.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant — length() <= capacity(),
//            kept by the stated preconditions.
//     I.5    State preconditions — on truncate, advance and layer; stated, not
//            checked, as elsewhere in the core.
//   C++ performance guidelines
//     MEM.9  Allocate at init, not in steady state — the layers, at load.
//     GDSA.6 Account the bytes a stage moves — a few hundred bytes of
//            counts a turn; the cache's own bytes move on the GPU, in
//            attention.

class KvCache {
public:
    KvCache(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
            policy::CachePrecision precision, std::uint32_t capacity);

    // Tokens whose keys and values are held.
    [[nodiscard]] std::uint32_t length() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] policy::CachePrecision precision() const noexcept;

    // Keeps the first `tokens` tokens and returns how many were kept: `tokens`
    // when every sliding-window layer still holds the window before that
    // position, otherwise 0, and the caller prefills from the first token.
    // Precondition: tokens <= length().
    [[nodiscard]] std::uint32_t truncate(std::uint32_t tokens) noexcept;

    // Precondition: length() + tokens <= capacity().
    void advance(std::uint32_t tokens) noexcept;

    // Precondition: `layer` is a layer of the model.
    [[nodiscard]] const residency::PlannedCacheLayer& layer(model::LayerIndex layer) const noexcept;

private:
    struct Layer {
        residency::PlannedCacheLayer planned;
        std::uint32_t window;   // the model's attention window for the layer
    };

    std::vector<Layer> layers_;
    // The most tokens written since the cache was last emptied: what bounds
    // what each ring still holds, which length alone does not after a
    // rollback.
    std::uint32_t written_ = 0;
    std::uint32_t length_ = 0;
    std::uint32_t capacity_;
    policy::CachePrecision precision_;
};

}  // namespace bllm::cache
