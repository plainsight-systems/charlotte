#include "core/graph/vision.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "core/kernels/block_attention/block_attention.h"
#include "core/kernels/gather/gather.h"
#include "core/kernels/matmul/matmul.h"
#include "core/kernels/norm/norm.h"
#include "core/kernels/pool/pool.h"
#include "core/kernels/rope/rope.h"

namespace bllm::graph {
namespace {

using model::Role;

constexpr std::uint32_t kWidest = 4096;   // norm.h and pool.h: rows up to 4,096 wide
constexpr std::uint64_t kGroup = 32;      // format.h: a row a whole number of 32-weight groups
constexpr std::uint32_t kVectors = 32;    // block_attention.h: the group divides 32

std::size_t index_of(gguf::TensorId tensor) { return static_cast<std::size_t>(tensor); }
std::size_t index_of(Role role) { return static_cast<std::size_t>(role); }

const residency::BufferRange& scratch(const residency::ResidencyPlan& plan, std::string_view purpose) {
    for (const residency::PlannedScratch& s : plan.scratch) {
        if (s.purpose == purpose) return s.range;
    }
    std::abort();   // a plan not made with vision_working_set: a broken precondition
}

GraphResult unsupported(std::string_view where, const std::string& what) {
    return {GraphError::UnsupportedShape, std::string(where) + ": " + what};
}

void append(std::vector<kernels::Launch>& out, std::vector<kernels::Launch> launches) {
    for (kernels::Launch& l : launches) out.push_back(std::move(l));
}

void named(std::vector<kernels::Launch>& out, std::size_t from, std::string_view role,
           std::uint32_t layer = kernels::kNoLayer) {
    for (std::size_t i = from; i < out.size(); ++i) {
        out[i].role = role;
        out[i].layer = layer;
    }
}

bool whole_groups(std::uint64_t width) { return width <= kWidest && width % kGroup == 0; }

}  // namespace

residency::WorkingSet vision_working_set(const model::ModelDescription& model, std::uint32_t rows) {
    std::uint64_t query = 0, key_value = 0, feed_forward = 0;
    for (const model::LayerDescription& l : model.layers) {
        query = std::max<std::uint64_t>(query, std::uint64_t{l.query_heads} * l.head_dimension);
        key_value = std::max<std::uint64_t>(key_value, std::uint64_t{l.key_value_heads} * l.head_dimension);
        feed_forward = std::max<std::uint64_t>(feed_forward, l.feed_forward_width);
    }
    const model::VisionDescription& v = *model.vision;
    const std::uint64_t patch = std::uint64_t{3} * v.patch_size * v.patch_size;
    const std::uint64_t pooled_rows = rows / (std::uint64_t{v.pooling} * v.pooling);
    const std::uint64_t w = model.embedding_width;
    residency::WorkingSet set{{
                                  {"pixels", rows, patch},
                                  {"hidden", rows, w},
                                  // Also the pooled rows, after the last layer's
                                  // feed-forward has read it (graph/vision.h).
                                  {"normed", rows, w},
                                  {"query", rows, query},
                                  {"key", rows, key_value},
                                  {"value", rows, key_value},
                                  {"attention", rows, query},
                                  {"output", rows, w},
                                  {"activation", rows, feed_forward},
                                  {"soft_tokens", pooled_rows, v.projection_width},
                                  // One float: no product of a vision step splits
                                  // along K (graph/vision.h).
                                  {"range_sums", 1, 1},
                              },
                              false,
                              0};
    set.step_rows = rows;
    return set;
}

}  // namespace bllm::graph

namespace bllm::graph {

VisionBuilder::VisionBuilder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                             std::vector<kernels::Launch>& out)
    : model_(&model), plan_(&plan), out_(&out) {
    for (const residency::PlannedTensor& t : plan.tensors) {
        if (!t.present) continue;
        const std::size_t i = index_of(t.tensor);
        if (i >= views_.size()) views_.resize(i + 1, nullptr);
        views_[i] = &t.view;
    }
    buffers_ = {scratch(plan, "pixels"), scratch(plan, "hidden"),     scratch(plan, "normed"),
                scratch(plan, "query"),  scratch(plan, "key"),        scratch(plan, "value"),
                scratch(plan, "attention"), scratch(plan, "output"), scratch(plan, "activation"),
                scratch(plan, "soft_tokens"), scratch(plan, "range_sums")};
}

const residency::WeightView& VisionBuilder::view(gguf::TensorId tensor) const {
    const std::size_t i = index_of(tensor);
    if (i >= views_.size() || views_[i] == nullptr) std::abort();   // the plan was not made from this model
    return *views_[i];
}

const residency::WeightView* VisionBuilder::role(std::uint32_t layer, Role r) const {
    const std::optional<gguf::TensorId>& tensor = model_->layers[layer].tensors[index_of(r)];
    return tensor ? &view(*tensor) : nullptr;
}

const residency::PlannedGroup* VisionBuilder::group(std::initializer_list<std::optional<gguf::TensorId>> members) const {
    for (const residency::PlannedGroup& g : plan_->groups) {
        if (g.count != members.size()) continue;
        std::size_t i = 0;
        bool same = true;
        for (const std::optional<gguf::TensorId>& m : members) same = same && m == g.members[i++];
        if (same) return &g;
    }
    return nullptr;
}

void VisionBuilder::product(const residency::WeightView& weight, const residency::BufferRange& input,
                            const residency::BufferRange& output) {
    append(*out_, kernels::matmul_launches({{&weight, nullptr, nullptr},
                                            1,
                                            input,
                                            {output, {}, {}},
                                            kernels::Epilogue::Write,
                                            model_->activation,
                                            kernels::Rows::EveryToken},
                                           kernels::kSplitGate, buffers_.range_sums));
}

kernels::Launch VisionBuilder::norm(const residency::WeightView& gain, bool normalize) {
    kernels::NormLaunch n{&gain,           pending_.post_gain, pending_.written, buffers_.output,
                          buffers_.hidden, buffers_.normed,    model_->norm_epsilon, kernels::Rows::EveryToken};
    n.normalize = normalize;
    return kernels::norm_launch(n);
}

GraphResult VisionBuilder::patch_embedding() {
    const model::VisionDescription& v = *model_->vision;
    const std::uint64_t patch = std::uint64_t{3} * v.patch_size * v.patch_size;
    if (!whole_groups(patch) || !whole_groups(model_->embedding_width)) {
        return unsupported("patch embedding", "a patch of " + std::to_string(patch) + " or a width of " +
                                                  std::to_string(model_->embedding_width) +
                                                  " is over 4,096 or no whole number of groups");
    }
    const std::size_t at = out_->size();
    product(view(v.patch_embedding), buffers_.pixels, buffers_.hidden);
    named(*out_, at, "patch.embed");
    return {};
}

GraphResult VisionBuilder::positions() {
    const model::VisionDescription& v = *model_->vision;
    const residency::WeightView& table = view(v.position_table);
    if (table.pieces().size() != 1) return unsupported("positions", "the table is more than one binding");
    const std::size_t at = out_->size();
    out_->push_back(kernels::position_launch(table, v.positions, buffers_.output));
    named(*out_, at, "patch.positions");
    // The positions pending, with no post-norm: the first norm adds them.
    pending_ = {true, nullptr};
    return {};
}

GraphResult VisionBuilder::attention(std::uint32_t layer) {
    const model::LayerDescription& l = model_->layers[layer];
    const std::string where = "layer " + std::to_string(layer);
    if (l.head_dimension != 64) return unsupported(where, "a vision layer's heads are 64 wide");
    if (l.query_heads != l.key_value_heads || kVectors % (l.query_heads / l.key_value_heads) != 0) {
        return unsupported(where, "query heads differ from key-value heads");
    }
    if (!l.tensors[index_of(Role::QueryNorm)] || !l.tensors[index_of(Role::KeyNorm)]) {
        return unsupported(where, "a vision layer normalizes queries and keys, and this one has no QK-norm");
    }
    if (!whole_groups(model_->embedding_width) || !whole_groups(l.feed_forward_width)) {
        return unsupported(where, "a width over 4,096 or no whole number of groups");
    }
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(norm(*role(layer, Role::AttentionNorm), true));
    named(*out_, at, "attention.norm", layer);

    const auto& t = l.tensors;
    if (group({t[index_of(Role::Query)], t[index_of(Role::Key)], t[index_of(Role::Value)]}) == nullptr) {
        return {GraphError::UngroupedFeedForward, where + ": Q, K and V are not one group"};
    }
    at = out_->size();
    append(*out_, kernels::matmul_launches({{role(layer, Role::Query), role(layer, Role::Key), role(layer, Role::Value)},
                                            3,
                                            b.normed,
                                            {b.query, b.key, b.value},
                                            kernels::Epilogue::QKV,
                                            model_->activation,
                                            kernels::Rows::EveryToken},
                                           kernels::kSplitGate, b.range_sums));
    named(*out_, at, "attention.qkv", layer);

    at = out_->size();
    out_->push_back(kernels::rope_encode_launch({l, model_->rotary_pairing, role(layer, Role::QueryNorm),
                                                 role(layer, Role::KeyNorm), nullptr, model_->value_norm,
                                                 model_->norm_epsilon, b.query, b.key, b.value}));
    named(*out_, at, "attention.rope", layer);

    at = out_->size();
    out_->push_back(kernels::streamed_attention_launch(
        {l, model_->attention_scale, b.query, b.key, b.value, b.attention}));
    named(*out_, at, "attention.stream", layer);

    at = out_->size();
    product(*role(layer, Role::AttentionOutput), b.attention, b.output);
    named(*out_, at, "attention.output", layer);
    pending_ = {true, role(layer, Role::PostAttentionNorm)};
    return {};
}

GraphResult VisionBuilder::gated_feed_forward(std::uint32_t layer) {
    const auto& t = model_->layers[layer].tensors;
    if (group({t[index_of(Role::Gate)], t[index_of(Role::Up)]}) == nullptr) {
        return {GraphError::UngroupedFeedForward, "layer " + std::to_string(layer)};
    }
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(norm(*role(layer, Role::FeedForwardNorm), true));
    named(*out_, at, "ffn.norm", layer);
    at = out_->size();
    append(*out_, kernels::matmul_launches({{role(layer, Role::Gate), role(layer, Role::Up), nullptr},
                                            2,
                                            b.normed,
                                            {b.activation, {}, {}},
                                            kernels::Epilogue::GatedActivation,
                                            model_->activation,
                                            kernels::Rows::EveryToken},
                                           kernels::kSplitGate, b.range_sums));
    named(*out_, at, "ffn.gate_up", layer);
    at = out_->size();
    product(*role(layer, Role::Down), b.activation, b.output);
    named(*out_, at, "ffn.down", layer);
    pending_ = {true, role(layer, Role::PostFeedForwardNorm)};
    return {};
}

GraphResult VisionBuilder::pooled_projection() {
    const model::VisionDescription& v = *model_->vision;
    if (!whole_groups(v.projection_width) || model_->embedding_width % 4 != 0) {
        return unsupported("projection", "width " + std::to_string(v.projection_width) +
                                             " is over 4,096 or no whole number of groups");
    }
    const Buffers& b = buffers_;
    const std::uint32_t area = v.pooling * v.pooling;
    // 1. The last block's result added into X, not normalized; its post-norm's
    // gain takes the norm's own gain's place, read-only.
    std::size_t at = out_->size();
    out_->push_back(norm(*pending_.post_gain, false));
    named(*out_, at, "pool.add");
    // 2. k × k patches a soft token, RMS-normalized at ε / W, into `normed`,
    // free once the last feed-forward has read it (vision.h).
    at = out_->size();
    out_->push_back(kernels::grid_launch(b.hidden, b.normed, model_->embedding_width, v.pooling,
                                         model_->norm_epsilon / static_cast<float>(model_->embedding_width)));
    named(*out_, at, "pool.grid");
    // 3. To the text width, over the pooled rows.
    at = out_->size();
    product(view(v.projection), b.normed, b.soft_tokens);
    for (std::size_t i = at; i < out_->size(); ++i) (*out_)[i].row_divisor = area;
    named(*out_, at, "pool.project");
    return {};
}

}  // namespace bllm::graph
