#include "core/graph/encoder.h"

#include <algorithm>
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

constexpr std::uint32_t kWidest = 4096;          // norm.h and pool.h: rows up to 4,096 wide
constexpr std::uint32_t kRopeHeadsTile = 64;     // rope.h: (heads) × d / 8 a multiple of 64
constexpr std::uint32_t kVectors = 32;           // block_attention.h: the group divides 32
constexpr std::uint64_t kGroup = 32;             // format.h: a row a whole number of 32-weight groups

std::size_t index_of(gguf::TensorId tensor) { return static_cast<std::size_t>(tensor); }
std::size_t index_of(Role role) { return static_cast<std::size_t>(role); }

const residency::BufferRange& scratch(const residency::ResidencyPlan& plan, std::string_view purpose) {
    for (const residency::PlannedScratch& s : plan.scratch) {
        if (s.purpose == purpose) return s.range;
    }
    std::abort();   // a plan not made with encoder_working_set: a broken precondition
}

GraphResult unsupported(std::string_view where, const std::string& what) {
    return {GraphError::UnsupportedShape, std::string(where) + ": " + what};
}

// The preconditions rope's encode form and block attention state for a
// layer (encoder.h, Shapes).
GraphResult check_layer(std::uint32_t layer, const model::LayerDescription& l, std::uint32_t width) {
    const std::string where = "layer " + std::to_string(layer);
    const std::uint32_t d = l.head_dimension;
    if (d != 64 && d != 128 && d != 256 && d != 512) {
        return unsupported(where, "head dimension " + std::to_string(d) + " is not 64, 128, 256 or 512");
    }
    if (l.key_value_heads == 0 || l.query_heads % l.key_value_heads != 0) {
        return unsupported(where, std::to_string(l.query_heads) + " query heads do not share " +
                                      std::to_string(l.key_value_heads) + " key-value heads evenly");
    }
    const std::uint32_t group = l.query_heads / l.key_value_heads;
    if (kVectors % group != 0) {
        return unsupported(where, std::to_string(group) + " query heads a key-value head do not divide 32");
    }
    if ((l.query_heads + 2 * l.key_value_heads) * (d / 8) % kRopeHeadsTile != 0) {
        return unsupported(where, "(query heads + 2 × key-value heads) × head dimension / 8 is not a multiple of 64");
    }
    if (!l.tensors[index_of(Role::QueryNorm)] || !l.tensors[index_of(Role::KeyNorm)]) {
        return unsupported(where, "an encoder's rope normalizes queries and keys, and this layer has no QK-norm");
    }
    if (width > kWidest || width % kGroup != 0) {
        return unsupported(where, "hidden width " + std::to_string(width) + " is over 4,096 or no whole number of groups");
    }
    return {};
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

}  // namespace

residency::WorkingSet encoder_working_set(const model::ModelDescription& model) {
    using residency::kPrefillBlock;
    std::uint64_t query = 0, key_value = 0, feed_forward = 0, query_heads = 0;
    for (const model::LayerDescription& l : model.layers) {
        query = std::max<std::uint64_t>(query, std::uint64_t{l.query_heads} * l.head_dimension);
        query_heads = std::max<std::uint64_t>(query_heads, l.query_heads);
        key_value = std::max<std::uint64_t>(key_value, std::uint64_t{l.key_value_heads} * l.head_dimension);
        feed_forward = std::max<std::uint64_t>(feed_forward, l.feed_forward_width);
    }
    const model::EncoderDescription& e = *model.encoder;
    const std::uint64_t per_layer = std::uint64_t{e.per_layer_width} * model.layers.size();
    // The range sums of a product split along K (kernels/matmul/matmul.h):
    // the most any needs — a decoder layer's, the per-layer inputs'
    // projection, and each layer's per-layer gate and projection.
    using kernels::Epilogue;
    const auto need = [](Epilogue epilogue, std::uint64_t rows, std::uint64_t columns) {
        return kernels::range_sum_floats(epilogue, rows, columns, kernels::kSplitGate);
    };
    std::uint64_t range_sums = std::max({std::uint64_t{1}, need(Epilogue::Write, per_layer, model.embedding_width),
                                         need(Epilogue::GatedByInput, e.per_layer_width, model.embedding_width),
                                         need(Epilogue::Write, model.embedding_width, e.per_layer_width)});
    for (const model::LayerDescription& l : model.layers) {
        range_sums = std::max(range_sums, layer_range_sums(l, model.embedding_width));
    }
    // The gate's output, the per-layer width, shares `activation` with the
    // feed-forward block's: neither is live when the other is written.
    feed_forward = std::max<std::uint64_t>(feed_forward, e.per_layer_width);
    const std::uint64_t chunks = (kPrefillBlock + 63) / 64;
    return {{
                {"hidden", kPrefillBlock, model.embedding_width},
                {"normed", kPrefillBlock, model.embedding_width},
                {"query", kPrefillBlock, query},
                {"key", kPrefillBlock, key_value},
                {"value", kPrefillBlock, key_value},
                {"attention", kPrefillBlock, query},
                {"output", kPrefillBlock, model.embedding_width},
                {"activation", kPrefillBlock, feed_forward},
                {"per_layer_raw", kPrefillBlock, per_layer},
                {"per_layer", kPrefillBlock, per_layer},
                // Block attention's scores, every query head's N × N, and
                // each row's chunk maxima (block_attention.h).
                {"scores", query_heads * kPrefillBlock, kPrefillBlock},
                {"score_maxima", query_heads * kPrefillBlock, chunks},
                // The mean, at row N − 1, where a launch over the last token
                // reads it; the projection's one row; the vector read back.
                {"pooled", kPrefillBlock, model.embedding_width},
                {"projected", 1, e.embedding_length},
                {"embedding", 1, e.embedding_length},
                // The range sums of a product split along K, the most any of
                // the products needs (kernels/matmul/matmul.h).
                {"range_sums", 1, range_sums},
            },
            false,
            0};
}

EncoderBuilder::EncoderBuilder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                               std::vector<kernels::Launch>& out)
    : model_(&model), plan_(&plan), out_(&out) {
    for (const residency::PlannedTensor& t : plan.tensors) {
        const std::size_t i = index_of(t.tensor);
        if (i >= views_.size()) views_.resize(i + 1, nullptr);
        views_[i] = &t.view;
    }
    buffers_ = {scratch(plan, "hidden"),        scratch(plan, "normed"),     scratch(plan, "query"),
                scratch(plan, "key"),           scratch(plan, "value"),      scratch(plan, "attention"),
                scratch(plan, "output"),        scratch(plan, "activation"), scratch(plan, "per_layer_raw"),
                scratch(plan, "per_layer"),     scratch(plan, "scores"),     scratch(plan, "score_maxima"),
                scratch(plan, "pooled"),        scratch(plan, "projected"),  scratch(plan, "embedding"),
                scratch(plan, "range_sums")};
}

const residency::WeightView& EncoderBuilder::view(gguf::TensorId tensor) const {
    const std::size_t i = index_of(tensor);
    if (i >= views_.size() || views_[i] == nullptr) std::abort();   // the plan was not made from this model
    return *views_[i];
}

const residency::WeightView* EncoderBuilder::role(std::uint32_t layer, Role r) const {
    const std::optional<gguf::TensorId>& tensor = model_->layers[layer].tensors[index_of(r)];
    return tensor ? &view(*tensor) : nullptr;
}

const residency::PlannedGroup* EncoderBuilder::group(
    std::initializer_list<std::optional<gguf::TensorId>> members) const {
    for (const residency::PlannedGroup& g : plan_->groups) {
        if (g.count != members.size()) continue;
        std::size_t i = 0;
        bool same = true;
        for (const std::optional<gguf::TensorId>& m : members) same = same && m == g.members[i++];
        if (same) return &g;
    }
    return nullptr;
}

void EncoderBuilder::product(const residency::WeightView& weight, const residency::BufferRange& input,
                             const residency::BufferRange& output, kernels::Rows rows) {
    append(*out_, kernels::matmul_launches({{&weight, nullptr, nullptr},
                                            1,
                                            input,
                                            {output, {}, {}},
                                            kernels::Epilogue::Write,
                                            model_->activation,
                                            rows},
                                           kernels::kSplitGate, buffers_.range_sums));
}

kernels::Launch EncoderBuilder::norm(const residency::WeightView& gain, bool normalize) {
    kernels::NormLaunch n{&gain,           pending_.post_gain, pending_.written, buffers_.output,
                          buffers_.hidden, buffers_.normed,    model_->norm_epsilon, kernels::Rows::EveryToken};
    n.scale = pending_.scale;
    n.normalize = normalize;
    return kernels::norm_launch(n);
}

void EncoderBuilder::embed(float scale) {
    const std::size_t at = out_->size();
    // An encoder never feeds a step, so the gather's record binding, which
    // only a fed step reads, takes the embedding buffer, read-only and unread
    // (kernels/gather/gather.h).
    append(*out_, kernels::gather_launches(view(*model_->token_embedding), buffers_.hidden, buffers_.embedding, scale));
    named(*out_, at, "embed");
}

GraphResult EncoderBuilder::per_layer_inputs() {
    const model::EncoderDescription& e = *model_->encoder;
    if (e.per_layer_width > kWidest || e.per_layer_width % kGroup != 0) {
        return unsupported("per-layer inputs", "width " + std::to_string(e.per_layer_width) +
                                                   " is over 4,096 or no whole number of groups");
    }
    std::size_t at = out_->size();
    product(view(e.per_layer_projection), buffers_.hidden, buffers_.per_layer_raw, kernels::Rows::EveryToken);
    named(*out_, at, "per_layer.inputs");
    // Each token's slices, a row each, normalized on their own; the
    // projection's 1 / √W folded into ε, ε × W (encoder.h).
    kernels::NormLaunch n{&view(e.per_layer_norm),
                          nullptr,
                          false,
                          buffers_.output,
                          buffers_.per_layer_raw,
                          buffers_.per_layer,
                          model_->norm_epsilon * static_cast<float>(model_->embedding_width),
                          kernels::Rows::EveryToken};
    n.rows_per_token = static_cast<std::uint32_t>(model_->layers.size());
    at = out_->size();
    out_->push_back(kernels::norm_launch(n));
    named(*out_, at, "per_layer.inputs_norm");
    return {};
}

GraphResult EncoderBuilder::attention(std::uint32_t layer) {
    const model::LayerDescription& l = model_->layers[layer];
    if (GraphResult r = check_layer(layer, l, model_->embedding_width); !r.ok()) return r;
    const Buffers& b = buffers_;

    std::size_t at = out_->size();
    out_->push_back(norm(*role(layer, Role::AttentionNorm), true));
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
                                                kernels::Rows::EveryToken},
                                               kernels::kSplitGate, b.range_sums));
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
    out_->push_back(kernels::rope_encode_launch({l, model_->rotary_pairing, role(layer, Role::QueryNorm),
                                                 role(layer, Role::KeyNorm),
                                                 model_->rotary_factors ? &view(*model_->rotary_factors) : nullptr,
                                                 model_->value_norm, model_->norm_epsilon, b.query, b.key, b.value}));
    named(*out_, at, "attention.rope", layer);

    auto pair = kernels::block_attention_launches(
        {l, model_->attention_scale, b.query, b.key, b.value, b.scores, b.score_maxima, b.attention});
    at = out_->size();
    out_->push_back(std::move(pair[0]));
    named(*out_, at, "attention.scores", layer);
    at = out_->size();
    out_->push_back(std::move(pair[1]));
    named(*out_, at, "attention.weigh", layer);

    at = out_->size();
    product(*role(layer, Role::AttentionOutput), b.attention, b.output, kernels::Rows::EveryToken);
    named(*out_, at, "attention.output", layer);
    pending_ = {true, role(layer, Role::PostAttentionNorm), nullptr};
    return {};
}

GraphResult EncoderBuilder::gated_feed_forward(std::uint32_t layer) {
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
    product(*role(layer, Role::Down), b.activation, b.output, kernels::Rows::EveryToken);
    named(*out_, at, "ffn.down", layer);
    pending_ = {true, role(layer, Role::PostFeedForwardNorm), nullptr};
    return {};
}

GraphResult EncoderBuilder::per_layer_block(std::uint32_t layer) {
    const model::EncoderDescription& e = *model_->encoder;
    const Buffers& b = buffers_;
    const residency::WeightView* gate = role(layer, Role::PerLayerGate);
    const residency::WeightView* projection = role(layer, Role::PerLayerProjection);
    if (gate == nullptr || projection == nullptr || role(layer, Role::PerLayerPostNorm) == nullptr ||
        role(layer, Role::OutputScale) == nullptr) {
        return unsupported("layer " + std::to_string(layer), "the per-layer block lacks a weight");
    }

    // 1. The pending block added into X, not normalized: the gate reads X.
    // Its post-norm's gain, read-only, takes the norm's own gain's place.
    std::size_t at = out_->size();
    out_->push_back(norm(pending_.post_gain != nullptr ? *pending_.post_gain : *role(layer, Role::PerLayerPostNorm),
                         false));
    named(*out_, at, "per_layer.add", layer);

    // 2. gelu(W X) × the layer's per-layer input, its slice of `per_layer`.
    at = out_->size();
    kernels::MatmulLaunch m{{gate, nullptr, nullptr},
                            1,
                            b.hidden,
                            {b.activation, {}, {}},
                            kernels::Epilogue::GatedByInput,
                            model_->activation,
                            kernels::Rows::EveryToken};
    m.gate_input = b.per_layer;
    m.gate_input_stride = e.per_layer_width * static_cast<std::uint32_t>(model_->layers.size());
    m.gate_input_column = e.per_layer_width * layer;
    append(*out_, kernels::matmul_launches(m, kernels::kSplitGate, b.range_sums));
    named(*out_, at, "per_layer.gate", layer);

    // 3. Back to the hidden width; its post-norm and the layer's scale
    // pending, for the next norm's add.
    at = out_->size();
    product(*projection, b.activation, b.output, kernels::Rows::EveryToken);
    named(*out_, at, "per_layer.project", layer);
    pending_ = {true, role(layer, Role::PerLayerPostNorm), role(layer, Role::OutputScale)};
    return {};
}

GraphResult EncoderBuilder::pooled_output() {
    const model::EncoderDescription& e = *model_->encoder;
    if (e.embedding_length > kWidest || e.embedding_length % kGroup != 0) {
        return unsupported("output", "embedding length " + std::to_string(e.embedding_length) +
                                         " is over 4,096 or no whole number of groups");
    }
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(norm(view(*model_->output_norm), true));
    named(*out_, at, "output.norm");
    at = out_->size();
    out_->push_back(kernels::mean_launch(b.normed, b.pooled, model_->embedding_width));
    named(*out_, at, "output.pool");
    at = out_->size();
    product(view(e.output_projection), b.pooled, b.projected, kernels::Rows::LastToken);
    named(*out_, at, "output.project");
    at = out_->size();
    out_->push_back(kernels::unit_launch(b.projected, b.embedding, e.embedding_length));
    named(*out_, at, "output.unit");
    return {};
}

}  // namespace bllm::graph
