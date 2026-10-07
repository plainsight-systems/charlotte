#include "core/graph/graph.h"

#include <cstdlib>
#include <string_view>

#include "core/kernels/attention/attention.h"
#include "core/kernels/gather/gather.h"
#include "core/kernels/matmul/matmul.h"
#include "core/kernels/norm/norm.h"
#include "core/kernels/rope/rope.h"
#include "core/kernels/topk/topk.h"
#include "core/sampler/sampler.h"

namespace bllm::graph {
namespace {

using model::Role;

constexpr std::uint32_t kNormWidest = 4096;     // norm.h: rows up to 4,096 wide
constexpr std::uint32_t kRopeHeadsTile = 64;    // rope.h: (heads) × d / 8 a multiple of 64
constexpr std::uint32_t kAttentionLanes = 1024; // attention.h: the group divides 1,024 / d

std::size_t index_of(gguf::TensorId tensor) { return static_cast<std::size_t>(tensor); }
std::size_t index_of(Role role) { return static_cast<std::size_t>(role); }

// The plan names every working buffer the blocks bind (residency/plan.h).
const residency::BufferRange& scratch(const residency::ResidencyPlan& plan, std::string_view purpose) {
    for (const residency::PlannedScratch& s : plan.scratch) {
        if (s.purpose == purpose) return s.range;
    }
    std::abort();   // a plan not made by plan_residency: a broken precondition
}

GraphResult unsupported(std::uint32_t layer, const std::string& what) {
    return {GraphError::UnsupportedShape, "layer " + std::to_string(layer) + ": " + what};
}

// The preconditions rope, attention and norm state for a layer.
GraphResult check_shape(std::uint32_t layer, const model::LayerDescription& l, std::uint32_t width) {
    const std::uint32_t d = l.head_dimension;
    if (d != 64 && d != 128 && d != 256) {
        return unsupported(layer, "head dimension " + std::to_string(d) + " is not 64, 128 or 256");
    }
    if (l.key_value_heads == 0 || l.query_heads % l.key_value_heads != 0) {
        return unsupported(layer, std::to_string(l.query_heads) + " query heads do not share " +
                                      std::to_string(l.key_value_heads) + " key-value heads evenly");
    }
    const std::uint32_t group = l.query_heads / l.key_value_heads;
    if ((kAttentionLanes / d) % group != 0) {
        return unsupported(layer, std::to_string(group) + " query heads a key-value head do not divide " +
                                      std::to_string(kAttentionLanes / d));
    }
    if ((l.query_heads + 2 * l.key_value_heads) * (d / 8) % kRopeHeadsTile != 0) {
        return unsupported(layer, "(query heads + 2 × key-value heads) × head dimension / 8 is not a multiple of " +
                                      std::to_string(kRopeHeadsTile));
    }
    if (l.tensors[index_of(Role::QueryNorm)].has_value() != l.tensors[index_of(Role::KeyNorm)].has_value()) {
        return unsupported(layer, "QK-norm on queries or keys alone");
    }
    if (width > kNormWidest) {
        return unsupported(layer, "hidden width " + std::to_string(width) + " is over " + std::to_string(kNormWidest));
    }
    return {};
}

void append(std::vector<kernels::Launch>& out, std::vector<kernels::Launch> launches) {
    for (kernels::Launch& l : launches) out.push_back(std::move(l));
}

// Names the launches appended since `from` (graph.h's names).
void named(std::vector<kernels::Launch>& out, std::size_t from, std::string_view role,
           std::uint32_t layer = kernels::kNoLayer) {
    for (std::size_t i = from; i < out.size(); ++i) {
        out[i].role = role;
        out[i].layer = layer;
    }
}

}  // namespace

Builder::Builder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                 const formats::Format& cache_format, std::vector<kernels::Launch>& out)
    : model_(&model), plan_(&plan), cache_format_(&cache_format), out_(&out) {
    for (const residency::PlannedTensor& t : plan.tensors) {
        const std::size_t i = index_of(t.tensor);
        if (i >= views_.size()) views_.resize(i + 1, nullptr);
        views_[i] = &t.view;
    }
    buffers_ = {scratch(plan, "hidden"),   scratch(plan, "normed"),        scratch(plan, "query"),
                scratch(plan, "key"),      scratch(plan, "value"),         scratch(plan, "attention"),
                scratch(plan, "partials"), scratch(plan, "partial_stats"), scratch(plan, "output"),
                scratch(plan, "activation"), scratch(plan, "logits"), scratch(plan, "partials_a"),
                scratch(plan, "partials_b"), scratch(plan, "candidates"), scratch(plan, "sampled")};
}

const residency::WeightView& Builder::view(gguf::TensorId tensor) const {
    const std::size_t i = index_of(tensor);
    if (i >= views_.size() || views_[i] == nullptr) std::abort();   // the plan was not made from this model
    return *views_[i];
}

const residency::WeightView* Builder::role(std::uint32_t layer, Role r) const {
    const std::optional<gguf::TensorId>& tensor = model_->layers[layer].tensors[index_of(r)];
    return tensor ? &view(*tensor) : nullptr;
}

const residency::PlannedGroup* Builder::group(std::initializer_list<std::optional<gguf::TensorId>> members) const {
    for (const residency::PlannedGroup& g : plan_->groups) {
        if (g.count != members.size()) continue;
        std::size_t i = 0;
        bool same = true;
        for (const std::optional<gguf::TensorId>& m : members) same = same && m == g.members[i++];
        if (same) return &g;
    }
    return nullptr;
}

void Builder::product(const residency::WeightView& weight, const residency::BufferRange& input,
                      const residency::BufferRange& output, kernels::Rows rows) {
    append(*out_, kernels::matmul_launches({{&weight, nullptr, nullptr},
                                            1,
                                            input,
                                            {output, {}, {}},
                                            kernels::Epilogue::Write,
                                            model_->activation,
                                            rows}));
}

void Builder::embed(float scale) {
    const std::size_t at = out_->size();
    append(*out_, kernels::gather_launches(view(model_->token_embedding), buffers_.hidden, buffers_.sampled, scale));
    named(*out_, at, "embed");
}

GraphResult Builder::attention(std::uint32_t layer) {
    const model::LayerDescription& l = model_->layers[layer];
    if (GraphResult r = check_shape(layer, l, model_->embedding_width); !r.ok()) return r;
    const Buffers& b = buffers_;

    std::size_t at = out_->size();
    out_->push_back(kernels::norm_launch({role(layer, Role::AttentionNorm), pending_.post_gain, pending_.written,
                                          b.output, b.hidden, b.normed, model_->norm_epsilon,
                                          kernels::Rows::EveryToken}));
    named(*out_, at, "attention.norm", layer);

    const auto& t = l.tensors;
    at = out_->size();
    if (group({t[index_of(Role::Query)], t[index_of(Role::Key)], t[index_of(Role::Value)]}) != nullptr) {
        append(*out_, kernels::matmul_launches({{role(layer, Role::Query), role(layer, Role::Key),
                                                 role(layer, Role::Value)},
                                                3,
                                                b.normed,
                                                {b.query, b.key, b.value},
                                                kernels::Epilogue::QKV,
                                                model_->activation,
                                                kernels::Rows::EveryToken}));
        named(*out_, at, "attention.qkv", layer);
    } else {
        product(*role(layer, Role::Query), b.normed, b.query, kernels::Rows::EveryToken);
        named(*out_, at, "attention.q", layer);
        at = out_->size();
        product(*role(layer, Role::Key), b.normed, b.key, kernels::Rows::EveryToken);
        named(*out_, at, "attention.k", layer);
        at = out_->size();
        product(*role(layer, Role::Value), b.normed, b.value, kernels::Rows::EveryToken);
        named(*out_, at, "attention.v", layer);
    }

    at = out_->size();
    out_->push_back(kernels::rope_launch({l, model_->rotary_pairing, role(layer, Role::QueryNorm),
                                          role(layer, Role::KeyNorm),
                                          model_->rotary_factors ? &view(*model_->rotary_factors) : nullptr,
                                          model_->norm_epsilon, b.query, b.key, b.value, plan_->cache[layer],
                                          cache_format_}));
    named(*out_, at, "attention.rope", layer);

    for (kernels::Launch& a : kernels::attention_launches({l, model_->attention_scale, b.query, plan_->cache[layer],
                                                           cache_format_, b.attention, b.partials,
                                                           b.partial_stats})) {
        at = out_->size();
        const bool combine = a.entry_point == "combine";
        out_->push_back(std::move(a));
        named(*out_, at, combine ? "attention.combine" : "attention.scores", layer);
    }

    at = out_->size();
    product(*role(layer, Role::AttentionOutput), b.attention, b.output, kernels::Rows::EveryToken);
    named(*out_, at, "attention.output", layer);
    pending_ = {true, role(layer, Role::PostAttentionNorm)};
    return {};
}

GraphResult Builder::gated_feed_forward(std::uint32_t layer) {
    const auto& t = model_->layers[layer].tensors;
    if (group({t[index_of(Role::Gate)], t[index_of(Role::Up)]}) == nullptr) {
        return {GraphError::UngroupedFeedForward, "layer " + std::to_string(layer)};
    }
    const Buffers& b = buffers_;

    std::size_t at = out_->size();
    out_->push_back(kernels::norm_launch({role(layer, Role::FeedForwardNorm), pending_.post_gain, pending_.written,
                                          b.output, b.hidden, b.normed, model_->norm_epsilon,
                                          kernels::Rows::EveryToken}));
    named(*out_, at, "ffn.norm", layer);
    at = out_->size();
    append(*out_, kernels::matmul_launches({{role(layer, Role::Gate), role(layer, Role::Up), nullptr},
                                            2,
                                            b.normed,
                                            {b.activation, {}, {}},
                                            kernels::Epilogue::GatedActivation,
                                            model_->activation,
                                            kernels::Rows::EveryToken}));
    named(*out_, at, "ffn.gate_up", layer);
    at = out_->size();
    product(*role(layer, Role::Down), b.activation, b.output, kernels::Rows::EveryToken);
    named(*out_, at, "ffn.down", layer);
    pending_ = {true, role(layer, Role::PostFeedForwardNorm)};
    return {};
}

GraphResult Builder::output() {
    if (model_->vocabulary_size < kernels::kCandidates) {
        return {GraphError::UnsupportedShape, "a vocabulary of " + std::to_string(model_->vocabulary_size) +
                                                  " is under the " + std::to_string(kernels::kCandidates) +
                                                  " candidates selection keeps"};
    }
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(kernels::norm_launch({&view(model_->output_norm), pending_.post_gain, pending_.written, b.output,
                                          b.hidden, b.normed, model_->norm_epsilon, kernels::Rows::LastToken}));
    named(*out_, at, "output.norm");
    const residency::WeightView& head = view(model_->output_head.value_or(model_->token_embedding));
    at = out_->size();
    product(head, b.normed, b.logits, kernels::Rows::LastToken);
    named(*out_, at, "output.head");
    at = out_->size();
    append(*out_, kernels::topk_launches(b.logits, model_->vocabulary_size, b.partials_a, b.partials_b, b.candidates));
    named(*out_, at, "output.select");
    at = out_->size();
    out_->push_back(sampler::draw_launch(b.candidates, b.sampled));
    named(*out_, at, "output.draw");
    return {};
}

}  // namespace bllm::graph
