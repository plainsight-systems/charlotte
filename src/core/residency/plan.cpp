#include "core/residency/plan.h"

#include <algorithm>
#include <array>
#include <span>

#include "core/formats/format.h"
#include "core/gguf/checked.h"

// How this file works.
//
// The plan is the blueprint for putting a model on the GPU. It is worked out
// from the file's header alone, so preflight can say whether a model fits, and
// with how much context, before a single weight byte is downloaded. Upload
// later carries out the same plan: it creates the buffers listed here and
// copies each weight's bytes, unchanged, to the place assigned to it. Kernels
// then find every weight, cache layer and working buffer through the plan.
//
// plan_residency fills three pools in a fixed order, because each step needs
// what the previous one left:
//
//   1. place_weights. Every tensor, in file order, packed into as few buffers
//      as the limits allow. A tensor larger than one binding is cut between
//      rows, never inside one, so no quantization block is split.
//   2. place_scratch. The working buffers the graph declared, shared by
//      every layer, each in a buffer of its own.
//   3. place_cache. For a graph that keeps a cache, the KV cache takes
//      whatever the budget has left. The
//      context offered is the largest whose cache fits, found by bisection
//      because the cost only grows with the context. A sliding-window layer
//      stops growing at its ring: its window, a prefill block, and the
//      rollback reserve.
//
// All three place ranges through Packer, a bump allocator over one pool's
// buffers: align the offset, open a new buffer when the range will not fit,
// pad the length to a multiple of 4.
//
// Where the numbers come from. The widths — embedding, heads, head size,
// feed-forward, vocabulary — are the file's, read by Describe into the model
// description. The limits are the ones the device granted, and the budget and
// cache precision are load policy. Three numbers are this design's own:
//
//   - kPrefillBlock, 512 rows: how many prompt tokens one prefill step
//     processes. A bigger block reads each weight once for more tokens and
//     spreads a step's dispatches further; a smaller one keeps the working
//     buffers small (they grow linearly with it), returns control to the
//     page sooner, and leaves more of the budget to the cache. 512 follows
//     llama.cpp's default micro-batch; prefill throughput across block sizes
//     on the target devices is what moves it.
//   - f32 working values: arithmetic is f32 everywhere, because f16
//     arithmetic needs WebGPU's optional shader-f16 feature, which the
//     harness does not require, for cross-browser compatibility
//     (gpu/device_requirements.h); and f16 overflows at 65,504, which
//     activations can reach. Only storage is narrower — the
//     cache, per policy — so cache rounding is the only error introduced.
//   - one row of logits: only a step's last token needs a prediction.

namespace bllm::residency {
namespace {

using gguf::checked_add;
using gguf::checked_mul;

constexpr std::uint64_t kBindingGranule = 4;   // a storage binding's size is a multiple of 4
constexpr std::uint64_t kF32Bytes = 4;

// v rounded up to a multiple of `to`, a power of two. False on overflow.
bool round_up(std::uint64_t v, std::uint64_t to, std::uint64_t& out) {
    std::uint64_t sum = 0;
    if (!checked_add(v, to - 1, sum)) return false;
    out = sum & ~(to - 1);
    return true;
}

PlanResult failure(PlanError error, std::string subject = {}) {
    return PlanResult{error, std::move(subject)};
}

// Places ranges into the buffers of one pool, filling a buffer before opening
// the next. Every range starts at an aligned offset and is padded to the
// binding granule.
class Packer {
public:
    Packer(std::vector<PlannedBuffer>& buffers, Pool pool, const DeviceLimits& limits)
        : buffers_(buffers), pool_(pool), limits_(limits) {}

    // Closes the open buffer if `length` more bytes, at the next aligned
    // offset, would not fit it, so a group placed next starts a new one.
    bool reserve(std::uint64_t length) {
        if (!open_) return true;
        std::uint64_t offset = 0;
        if (!round_up(buffers_.back().size, limits_.storage_offset_alignment, offset)) return false;
        if (offset > limits_.max_buffer_size || length > limits_.max_buffer_size - offset) open_ = false;
        return true;
    }

    // Places `length` bytes. `alone` gives the range a buffer nobody else
    // shares. Precondition: the padded length fits in one buffer.
    bool place(std::uint64_t length, bool alone, BufferRange& out) {
        std::uint64_t padded = 0;
        if (!round_up(length, kBindingGranule, padded)) return false;
        std::uint64_t offset = 0;
        if (open_ && !alone) {
            if (!round_up(buffers_.back().size, limits_.storage_offset_alignment, offset)) return false;
            if (offset > limits_.max_buffer_size || padded > limits_.max_buffer_size - offset) open_ = false;
        }
        if (!open_ || alone) {
            buffers_.push_back({pool_, 0});
            offset = 0;
        }
        buffers_.back().size = offset + padded;
        out = {static_cast<BufferIndex>(buffers_.size() - 1), offset, padded};
        open_ = !alone;
        return true;
    }

private:
    std::vector<PlannedBuffer>& buffers_;
    Pool pool_;
    DeviceLimits limits_;
    bool open_ = false;
};

// The most one range may hold: it must fit both a binding and a buffer.
std::uint64_t range_limit(const DeviceLimits& limits) {
    return std::min(limits.max_storage_binding_size, limits.max_buffer_size) & ~(kBindingGranule - 1);
}

bool is_tied_copy(const gguf::TensorIndex& index, const model::ModelDescription& model,
                  std::size_t tensor) {
    if (!model.output_head || static_cast<std::size_t>(*model.output_head) != tensor) return false;
    const gguf::TensorEntry& head = index.tensor(*model.output_head);
    if (!model.token_embedding) return false;
    const gguf::TensorEntry& embedding = index.tensor(*model.token_embedding);
    return head.type == embedding.type && head.data_length == embedding.data_length &&
           std::equal(std::begin(head.dimensions), std::end(head.dimensions),
                      std::begin(embedding.dimensions));
}

// A layer's candidate groups, members in output order: Q, K, V and gate,
// up, where the layer has them.
std::vector<PlannedGroup> candidate_groups(const model::ModelDescription& model) {
    std::vector<PlannedGroup> groups;
    for (const model::LayerDescription& l : model.layers) {
        const auto at = [&](model::Role r) { return l.tensors[static_cast<std::size_t>(r)]; };
        using model::Role;
        if (at(Role::Query) && at(Role::Key) && at(Role::Value)) {
            groups.push_back({{*at(Role::Query), *at(Role::Key), *at(Role::Value)}, 3, {}});
        }
        if (at(Role::Gate) && at(Role::Up)) {
            groups.push_back({{*at(Role::Gate), *at(Role::Up), gguf::TensorId{}}, 2, {}});
        }
    }
    return groups;
}

// The bytes from tensor `first` to tensor `last`, in file order, as the
// packer would place them in one buffer: each padded and aligned. False
// when one of them is more than one piece, so the group cannot be one span.
bool span_bytes(std::span<const gguf::TensorEntry> tensors, std::size_t first, std::size_t last,
                const DeviceLimits& limits, std::uint64_t& out) {
    out = 0;
    for (std::size_t k = first; k <= last; ++k) {
        std::uint64_t padded = 0, aligned = 0;
        if (tensors[k].data_length > range_limit(limits) || !round_up(tensors[k].data_length, kBindingGranule, padded) ||
            !round_up(padded, limits.storage_offset_alignment, aligned) || !checked_add(out, aligned, out)) {
            return false;
        }
    }
    return true;
}

// Every tensor the description names: what a load of part of a file must
// hold.
std::vector<gguf::TensorId> described_tensors(const model::ModelDescription& model) {
    std::vector<gguf::TensorId> out;
    const auto add = [&](const std::optional<gguf::TensorId>& t) {
        if (t) out.push_back(*t);
    };
    add(model.token_embedding);
    add(model.output_norm);
    add(model.output_head);
    add(model.rotary_factors);
    if (model.vision) {
        add(model.vision->patch_embedding);
        add(model.vision->position_table);
        add(model.vision->projection);
    }
    if (model.audio) {
        const model::AudioDescription& a = *model.audio;
        for (const gguf::TensorId t : {a.convolution[0], a.convolution[1], a.convolution_norm[0],
                                       a.convolution_norm[1], a.input_projection, a.projection}) {
            add(t);
        }
        for (const model::ConformerLayer& l : a.layers) {
            for (const gguf::TensorId t : {l.ffn_norm, l.ffn_up.weight, l.ffn_down.weight, l.ffn_post_norm,
                                           l.attention_norm, l.query.weight, l.key.weight, l.value.weight,
                                           l.per_dim_scale, l.relative, l.attention_output.weight,
                                           l.attention_post_norm, l.convolution_norm, l.pointwise_in.weight,
                                           l.depthwise, l.depthwise_norm, l.pointwise_out.weight, l.ffn_norm_1,
                                           l.ffn_up_1.weight, l.ffn_down_1.weight, l.ffn_post_norm_1,
                                           l.output_norm}) {
                add(t);
            }
        }
    }
    // Every member of a product span: the bounds and biases above leave out.
    for (const model::ProductSpan& s : model.product_spans) {
        for (const gguf::TensorId t : s.members) add(t);
    }
    if (model.encoder) {
        add(model.encoder->per_layer_projection);
        add(model.encoder->per_layer_norm);
        add(model.encoder->output_projection);
    }
    for (const model::LayerDescription& l : model.layers) {
        for (const auto& t : l.tensors) add(t);
    }
    return out;
}

// Whether tensor `t` lies wholly inside the ranges the load reads; every
// tensor does when the load reads the whole file.
bool present(const gguf::TensorEntry& t, std::span<const FileRange> ranges) {
    return ranges.empty() || within(ranges, t.data_offset, t.data_offset + t.data_length);
}

PlanResult place_weights(const gguf::TensorIndex& index, const model::ModelDescription& model,
                         const DeviceLimits& limits, std::span<const FileRange> ranges, ResidencyPlan& out) {
    Packer packer{out.buffers, Pool::Weights, limits};
    const std::uint64_t limit = range_limit(limits);
    const auto tensors = index.tensors();
    for (const gguf::TensorId id : described_tensors(model)) {
        const gguf::TensorEntry& t = index.tensor(id);
        if (!present(t, ranges)) return failure(PlanError::MissingTensorBytes, t.name);
    }

    // Groups whose members agree in format and input width, by the file
    // index of their first member, and the bytes of their span.
    std::vector<PlannedGroup> groups = candidate_groups(model);
    std::vector<std::uint64_t> reserve_before(tensors.size(), 0);
    std::erase_if(groups, [&](const PlannedGroup& g) {
        const gguf::TensorEntry& head = index.tensor(g.members[0]);
        std::size_t first = tensors.size(), last = 0;
        for (std::uint32_t m = 0; m < g.count; ++m) {
            const gguf::TensorEntry& t = index.tensor(g.members[m]);
            if (t.type != head.type || t.dimensions[0] != head.dimensions[0]) return true;
            first = std::min(first, static_cast<std::size_t>(g.members[m]));
            last = std::max(last, static_cast<std::size_t>(g.members[m]));
        }
        // A span holding a tensor the load does not read is not a group.
        for (std::size_t k = first; k <= last; ++k) {
            if (!present(tensors[k], ranges)) return true;
        }
        std::uint64_t bytes = 0;
        if (!span_bytes(tensors, first, last, limits, bytes) || bytes > limit) return true;
        reserve_before[first] = std::max(reserve_before[first], bytes);
        return false;
    });
    // Product spans, which unlike groups must hold: each kept in one buffer
    // by reserving its bytes before its first member, as a group's are.
    for (const model::ProductSpan& s : model.product_spans) {
        std::size_t first = tensors.size(), last = 0;
        for (const gguf::TensorId t : s.members) {
            first = std::min(first, static_cast<std::size_t>(t));
            last = std::max(last, static_cast<std::size_t>(t));
        }
        const std::string& weight = index.tensor(s.weight).name;
        for (std::size_t k = first; k <= last; ++k) {
            if (!present(tensors[k], ranges)) return failure(PlanError::UnboundSpan, weight);
        }
        std::uint64_t bytes = 0;
        if (!span_bytes(tensors, first, last, limits, bytes) || bytes > limit) {
            return failure(PlanError::UnboundSpan, weight);
        }
        reserve_before[first] = std::max(reserve_before[first], bytes);
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (reserve_before[i] != 0 && !packer.reserve(reserve_before[i])) return failure(PlanError::Overflow);
        const gguf::TensorEntry& t = tensors[i];
        const gguf::TensorShape shape{t.dimension_count,
                                      {t.dimensions[0], t.dimensions[1], t.dimensions[2], t.dimensions[3]},
                                      t.element_count};
        if (!present(t, ranges)) {
            out.tensors.push_back({static_cast<gguf::TensorId>(i), WeightView{t.type, shape, {}}, std::nullopt,
                                   false});
            continue;
        }
        const bool tied = is_tied_copy(index, model, i);
        std::vector<WeightPiece> pieces;

        // A row is every dimension but the last (formats/format.h): whole
        // blocks, never split.
        const std::uint64_t row = formats::row_elements(t);
        const std::uint64_t rows = row == 0 ? 0 : t.element_count / row;
        if (rows != 0) {
            const std::uint64_t row_bytes = t.data_length / rows;
            if (row_bytes > limit) return failure(PlanError::RowExceedsBinding, t.name);
            const std::uint64_t rows_per_piece = limit / row_bytes;
            for (std::uint64_t first = 0; first < rows; first += rows_per_piece) {
                const std::uint64_t count = std::min(rows_per_piece, rows - first);
                BufferRange range{};
                if (!packer.place(count * row_bytes, tied, range)) return failure(PlanError::Overflow, t.name);
                pieces.push_back({range.buffer, range.offset, range.length, first, count});
            }
        }
        const bool whole = std::find(model.read_whole.begin(), model.read_whole.end(),
                                     static_cast<gguf::TensorId>(i)) != model.read_whole.end();
        out.tensors.push_back({static_cast<gguf::TensorId>(i), WeightView{t.type, shape, std::move(pieces)},
                               tied ? model.token_embedding : std::nullopt, true, whole});
    }

    // Each group's span, now its members are placed: one buffer, from the
    // first member's first byte to the last member's last.
    for (PlannedGroup& g : groups) {
        const WeightPiece& head = out.tensors[static_cast<std::size_t>(g.members[0])].view.pieces().front();
        std::uint64_t begin = head.offset, end = head.offset + head.length;
        bool one_buffer = true;
        for (std::uint32_t m = 0; m < g.count; ++m) {
            const WeightPiece& p = out.tensors[static_cast<std::size_t>(g.members[m])].view.pieces().front();
            one_buffer = one_buffer && p.buffer == head.buffer;
            begin = std::min(begin, p.offset);
            end = std::max(end, p.offset + p.length);
        }
        if (one_buffer && end - begin <= limit) {
            g.span = {head.buffer, begin, end - begin};
            out.groups.push_back(g);
        }
    }
    for (const model::ProductSpan& s : model.product_spans) {
        const WeightPiece& head = out.tensors[static_cast<std::size_t>(s.weight)].view.pieces().front();
        std::uint64_t begin = head.offset, end = head.offset + head.length;
        for (const gguf::TensorId t : s.members) {
            const auto& pieces = out.tensors[static_cast<std::size_t>(t)].view.pieces();
            if (pieces.size() != 1 || pieces.front().buffer != head.buffer) {
                return failure(PlanError::UnboundSpan, index.tensor(s.weight).name);
            }
            begin = std::min(begin, pieces.front().offset);
            end = std::max(end, pieces.front().offset + pieces.front().length);
        }
        if (end - begin > limit) return failure(PlanError::UnboundSpan, index.tensor(s.weight).name);
        out.spans.push_back({s.weight, {head.buffer, begin, end - begin}});
    }
    return {};
}

// The working buffers the graph declared, shared by every layer, in order.
PlanResult place_scratch(const WorkingSet& working, const DeviceLimits& limits, ResidencyPlan& out) {
    out.partial_rows = working.partial_rows;
    Packer packer{out.buffers, Pool::Scratch, limits};
    for (const ScratchNeed& need : working.needs) {
        std::uint64_t bytes = 0;
        std::uint64_t values = 0;
        if (!checked_mul(need.rows, need.width, values) || !checked_mul(values, kF32Bytes, bytes)) {
            return failure(PlanError::Overflow, std::string(need.purpose));
        }
        if (bytes > range_limit(limits)) return failure(PlanError::ScratchExceedsBinding, std::string(need.purpose));
        BufferRange range{};
        // Alone: a kernel binds one working buffer read-only and another
        // writable, which WebGPU refuses within one buffer (plan.h).
        if (!packer.place(bytes, true, range)) return failure(PlanError::Overflow, std::string(need.purpose));
        out.scratch.push_back({need.purpose, range});
    }
    return {};
}

std::uint64_t pool_bytes(const std::vector<PlannedBuffer>& buffers, Pool pool) {
    std::uint64_t total = 0;
    for (const PlannedBuffer& b : buffers) {
        if (b.pool == pool) total += b.size;
    }
    return total;
}

gguf::TensorType storage_type(policy::CachePrecision precision) {
    switch (precision) {
        case policy::CachePrecision::F16: return gguf::TensorType::F16;
        case policy::CachePrecision::BF16: return gguf::TensorType::BF16;
        case policy::CachePrecision::Q8_0: return gguf::TensorType::Q8_0;
    }
    return gguf::TensorType::F16;
}

// What one layer's cache costs: the bytes of one slot of keys (and as many
// again of values), and the most slots the layer ever holds.
struct CacheCost {
    std::uint64_t slot_bytes;
    std::uint64_t ring;
};

// Slots a layer holds at a context of `context` tokens.
std::uint64_t slots_at(const CacheCost& layer, std::uint64_t context) {
    return std::min(context, layer.ring);
}

// The cache's bytes at a context of `context` tokens, before alignment
// padding. False when a size overflows or a layer's keys outgrow a binding.
bool cache_bytes_at(std::span<const CacheCost> layers, std::uint64_t context, std::uint64_t binding,
                    std::uint64_t& out) {
    out = 0;
    for (const CacheCost& layer : layers) {
        std::uint64_t length = 0;
        if (!checked_mul(slots_at(layer, context), layer.slot_bytes, length) || length > binding) return false;
        if (!checked_add(out, 2 * length, out)) return false;   // length <= binding, so 2 * length cannot wrap
    }
    return true;
}

// Sizes the cache to the largest context the budget and the binding limit
// allow, and places it.
PlanResult place_cache(const model::ModelDescription& model, const DeviceLimits& limits,
                       const policy::LoadPolicy& policy, ResidencyPlan& out) {
    const gguf::FormatLayout& layout = *gguf::format_layout(storage_type(policy.cache_precision));
    const std::uint64_t limit = range_limit(limits);

    // A full-attention layer's window is the trained context, so its ring
    // never binds; a sliding-window layer's ring is its window, a prefill
    // block and the rollback reserve.
    std::vector<CacheCost> layers;
    layers.reserve(model.layers.size());
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const model::LayerDescription& l = model.layers[i];
        if (l.head_dimension % layout.block_elements != 0) {
            return failure(PlanError::UnsupportedCachePrecision, "layer " + std::to_string(i));
        }
        CacheCost cost{0, std::uint64_t{l.attention_window} + kPrefillBlock + policy.rollback_reserve};
        const std::uint64_t blocks = std::uint64_t{l.key_value_heads} * (l.head_dimension / layout.block_elements);
        if (!checked_mul(blocks, layout.block_bytes, cost.slot_bytes)) {
            return failure(PlanError::Overflow, "layer " + std::to_string(i));
        }
        layers.push_back(cost);
    }
    // The binding never limits the cache below a prefill block: the "key"
    // working buffer holds kPrefillBlock tokens of the same width at 4 bytes a
    // value and already fits one binding, and every cache precision stores
    // fewer bytes a value than that.
    const std::uint64_t shortest = std::min<std::uint64_t>(kPrefillBlock, model.trained_context);

    // Alignment padding the cache may add: at most one granule per range.
    const std::uint64_t slack = 2 * model.layers.size() * (limits.storage_offset_alignment + kBindingGranule);
    const std::uint64_t fixed = out.weight_bytes + out.scratch_bytes + slack;
    const auto fits = [&](std::uint64_t context) {
        std::uint64_t bytes = 0;
        std::uint64_t total = 0;
        return cache_bytes_at(layers, context, limit, bytes) && checked_add(fixed, bytes, total) &&
               total <= policy.memory_budget;
    };
    if (!fits(shortest)) {
        std::uint64_t bytes = 0;
        if (!cache_bytes_at(layers, shortest, limit, bytes) || !checked_add(fixed, bytes, out.total_bytes)) {
            return failure(PlanError::Overflow);
        }
        return failure(PlanError::ExceedsBudget);
    }

    // The cost never falls as the context grows, so the largest context that
    // fits is found by bisection between the shortest and the trained context.
    std::uint64_t low = shortest;
    std::uint64_t high = model.trained_context;
    while (low < high) {
        const std::uint64_t mid = low + (high - low + 1) / 2;
        if (fits(mid)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    out.context_offered = static_cast<std::uint32_t>(low);

    Packer packer{out.buffers, Pool::Cache, limits};
    out.cache.reserve(layers.size());
    for (const CacheCost& layer : layers) {
        PlannedCacheLayer planned{};
        planned.slots = static_cast<std::uint32_t>(slots_at(layer, out.context_offered));
        const std::uint64_t length = std::uint64_t{planned.slots} * layer.slot_bytes;   // checked by fits()
        if (!packer.place(length, false, planned.keys) || !packer.place(length, false, planned.values)) {
            return failure(PlanError::Overflow);
        }
        out.cache.push_back(planned);
    }
    return {};
}

// Without a cache: the context one step holds, and the fit of what is placed.
PlanResult place_no_cache(const model::ModelDescription& model, const policy::LoadPolicy& policy,
                          const WorkingSet& working, ResidencyPlan& out) {
    out.context_offered = working.step_rows != 0 ? working.step_rows
                                                 : std::min<std::uint32_t>(kPrefillBlock, model.trained_context);
    std::uint64_t total = 0;
    if (!checked_add(out.weight_bytes, out.scratch_bytes, total)) return failure(PlanError::Overflow);
    if (total > policy.memory_budget) {
        out.total_bytes = total;
        return failure(PlanError::ExceedsBudget);
    }
    return {};
}

}  // namespace

PlanResult plan_residency(const gguf::TensorIndex& index, const model::ModelDescription& model,
                          const DeviceLimits& limits, const policy::LoadPolicy& policy, const WorkingSet& working,
                          ResidencyPlan& out, std::span<const FileRange> ranges) {
    out = ResidencyPlan{};
    out.cache_type = storage_type(policy.cache_precision);
    if (auto r = place_weights(index, model, limits, ranges, out); !r.ok()) return r;
    out.weight_bytes = pool_bytes(out.buffers, Pool::Weights);
    if (auto r = place_scratch(working, limits, out); !r.ok()) return r;
    out.scratch_bytes = pool_bytes(out.buffers, Pool::Scratch);
    if (auto r = working.cache ? place_cache(model, limits, policy, out) : place_no_cache(model, policy, working, out);
        !r.ok()) {
        return r;
    }
    out.cache_bytes = pool_bytes(out.buffers, Pool::Cache);
    out.total_bytes = out.weight_bytes + out.scratch_bytes + out.cache_bytes;
    return {};
}

}  // namespace bllm::residency
