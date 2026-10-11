#include "core/arch/embedding_gemma2/embedding_gemma2.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "core/arch/describe.h"
#include "core/graph/encoder.h"

namespace bllm::arch {
namespace {

using model::Role;

constexpr std::string_view kArch = "gemma-embedding2";

// Gemma 3's thirteen roles, named as the file names them; the per-layer
// block's four are read below with their own widths.
constexpr std::array kRoles{
    RoleName{Role::AttentionNorm, "attn_norm.weight"},
    RoleName{Role::Query, "attn_q.weight"},
    RoleName{Role::Key, "attn_k.weight"},
    RoleName{Role::Value, "attn_v.weight"},
    RoleName{Role::QueryNorm, "attn_q_norm.weight"},
    RoleName{Role::KeyNorm, "attn_k_norm.weight"},
    RoleName{Role::AttentionOutput, "attn_output.weight"},
    RoleName{Role::PostAttentionNorm, "post_attention_norm.weight"},
    RoleName{Role::FeedForwardNorm, "ffn_norm.weight"},
    RoleName{Role::Gate, "ffn_gate.weight"},
    RoleName{Role::Up, "ffn_up.weight"},
    RoleName{Role::Down, "ffn_down.weight"},
    RoleName{Role::PostFeedForwardNorm, "post_ffw_norm.weight"},
};

// llama.cpp's pooling_type for the mean (LLAMA_POOLING_TYPE_MEAN).
constexpr std::uint32_t kMeanPooling = 1;

std::string key(std::string_view k) { return std::string(kArch) + "." + std::string(k); }

DescribeResult refused(DescribeError error, std::string subject) { return {error, std::move(subject)}; }

DescribeResult metadata(gguf::MetadataError e, std::string subject) {
    return refused(e == gguf::MetadataError::MissingKey ? DescribeError::MissingKey : DescribeError::WrongKeyType,
                   std::move(subject));
}

DescribeResult read_u32(const gguf::TensorIndex& index, std::string_view k, std::uint32_t& out) {
    if (const auto e = index.read_u32(key(k), out); e != gguf::MetadataError::Ok) return metadata(e, key(k));
    return {};
}

// A tensor the architecture reads itself: present, of this shape, and F32
// where a kernel binds it as f32.
DescribeResult find(const gguf::TensorIndex& index, const std::string& name, std::uint32_t dimension_count,
                    std::uint64_t d0, std::uint64_t d1, bool f32, gguf::TensorId& out) {
    const std::optional<gguf::TensorId> id = index.find(name);
    if (!id) return refused(DescribeError::MissingTensor, name);
    const gguf::TensorEntry& t = index.tensor(*id);
    if (t.dimension_count != dimension_count || t.dimensions[0] != d0 ||
        (dimension_count == 2 && t.dimensions[1] != d1)) {
        return refused(DescribeError::ShapeMismatch, name);
    }
    if (f32 && t.type != gguf::TensorType::F32) return refused(DescribeError::UnsupportedValue, name);
    out = *id;
    return {};
}

DescribeResult describe(const gguf::TensorIndex& index, model::ModelDescription& out) {
    Hyperparameters hp{};
    if (auto r = read_hyperparameters(index, kArch, hp); !r.ok()) return r;

    // What makes it an encoder: attention both ways, mean pooling, no keys
    // and values shared across layers.
    bool causal = true;
    if (const auto e = index.read_bool(key("attention.causal"), causal); e != gguf::MetadataError::Ok) {
        return metadata(e, key("attention.causal"));
    }
    if (causal) return refused(DescribeError::UnsupportedValue, key("attention.causal"));
    std::uint32_t pooling = 0;
    if (auto r = read_u32(index, "pooling_type", pooling); !r.ok()) return r;
    if (pooling != kMeanPooling) return refused(DescribeError::UnsupportedValue, key("pooling_type"));
    std::uint32_t shared = 0;
    if (auto r = read_u32_or(index, kArch, "attention.shared_kv_layers", 0, shared); !r.ok()) return r;
    if (shared != 0) return refused(DescribeError::UnsupportedValue, key("attention.shared_kv_layers"));

    // Which layers slide, and each kind's head.
    std::uint32_t window = 0, swa_head = 0, swa_value = 0, swa_rotated = 0;
    if (auto r = read_u32(index, "attention.sliding_window", window); !r.ok()) return r;
    if (window == 0) return refused(DescribeError::InvalidValue, key("attention.sliding_window"));
    std::vector<bool> sliding;
    if (const auto e = index.read_bool_array(key("attention.sliding_window_pattern"), sliding);
        e != gguf::MetadataError::Ok) {
        return metadata(e, key("attention.sliding_window_pattern"));
    }
    if (sliding.size() != hp.block_count) {
        return refused(DescribeError::InvalidValue, key("attention.sliding_window_pattern"));
    }
    if (auto r = read_u32(index, "attention.key_length_swa", swa_head); !r.ok()) return r;
    if (auto r = read_u32_or(index, kArch, "attention.value_length_swa", swa_head, swa_value); !r.ok()) return r;
    if (swa_value != swa_head) return refused(DescribeError::UnsupportedValue, key("attention.value_length_swa"));
    if (auto r = read_u32_or(index, kArch, "rope.dimension_count_swa", swa_head, swa_rotated); !r.ok()) return r;
    if (swa_rotated != swa_head) return refused(DescribeError::UnsupportedValue, key("rope.dimension_count_swa"));
    if (swa_head == 0 || swa_head % 2 != 0) return refused(DescribeError::UnsupportedValue, key("attention.key_length_swa"));
    float swa_base = 0;
    if (auto r = read_f32_or(index, kArch, "rope.freq_base_swa", hp.rope_base, swa_base); !r.ok()) return r;

    std::vector<LayerHeads> heads;
    for (std::uint32_t l = 0; l < hp.block_count; ++l) {
        const std::uint32_t kv = hp.head_count_kv_per_layer.empty() ? hp.head_count_kv : hp.head_count_kv_per_layer[l];
        heads.push_back({kv, sliding[l] ? swa_head : hp.head_dimension});
    }
    // NEOX pairing and GELU's tanh form, as llama.cpp gives Gemma 4; its
    // output.weight is the projection to the embedding, read below.
    Conventions conventions{model::RotaryPairing::Halves, model::FeedForwardActivation::GeluTanh};
    conventions.vocabulary_head = false;
    if (auto r = describe_layers(index, hp, kRoles, conventions, out, heads); !r.ok()) return r;

    for (std::uint32_t l = 0; l < hp.block_count; ++l) {
        if (sliding[l]) {
            out.layers[l].attention_window = window;
            out.layers[l].rope_base = swa_base;
        }
    }

    // The encoder's own: the per-layer inputs, the per-layer block of each
    // layer, and the output projection.
    std::uint32_t per_layer = 0, embedding = 0;
    if (auto r = read_u32(index, "embedding_length_per_layer_input", per_layer); !r.ok()) return r;
    if (auto r = read_u32(index, "embedding_length_out", embedding); !r.ok()) return r;
    if (per_layer == 0) return refused(DescribeError::InvalidValue, key("embedding_length_per_layer_input"));
    if (embedding == 0) return refused(DescribeError::InvalidValue, key("embedding_length_out"));
    const std::uint64_t w = hp.embedding_length;
    model::EncoderDescription e{};
    e.per_layer_width = per_layer;
    e.embedding_length = embedding;
    e.pooling = model::Pooling::Mean;
    if (auto r = find(index, "per_layer_model_proj.weight", 2, w, std::uint64_t{per_layer} * hp.block_count, false,
                      e.per_layer_projection);
        !r.ok()) {
        return r;
    }
    if (auto r = find(index, "per_layer_proj_norm.weight", 1, per_layer, 0, true, e.per_layer_norm); !r.ok()) return r;
    if (auto r = find(index, "output.weight", 2, w, embedding, false, e.output_projection); !r.ok()) return r;
    for (std::uint32_t l = 0; l < hp.block_count; ++l) {
        const std::string prefix = "blk." + std::to_string(l) + ".";
        struct PerLayer {
            Role role;
            const char* suffix;
            std::uint32_t dimension_count;
            std::uint64_t d0, d1;
            bool f32;
        };
        for (const PerLayer& p : {PerLayer{Role::PerLayerGate, "inp_gate.weight", 2, w, per_layer, false},
                                  PerLayer{Role::PerLayerProjection, "proj.weight", 2, per_layer, w, false},
                                  PerLayer{Role::PerLayerPostNorm, "post_norm.weight", 1, w, 0, true},
                                  PerLayer{Role::OutputScale, "layer_output_scale.weight", 1, 1, 0, true}}) {
            gguf::TensorId id{};
            if (auto r = find(index, prefix + p.suffix, p.dimension_count, p.d0, p.d1, p.f32, id); !r.ok()) return r;
            out.layers[l].tensors[static_cast<std::size_t>(p.role)] = id;
        }
    }

    // QK-norm takes the attention scale's place (llama.cpp's
    // f_attention_scale = 1); values normalized per head, without a gain.
    out.attention_scale = 1.0f;
    out.mask = model::AttentionMask::Bidirectional;
    out.value_norm = true;
    out.encoder = e;
    return {};
}

// The embedding scaled by √W in f32, as llama.cpp's build_inp_embd; the
// per-layer inputs; each layer's attention, feed-forward block and
// per-layer block; the pooled output (graph/encoder.h).
graph::GraphResult encode(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                          std::vector<kernels::Launch>& out) {
    graph::EncoderBuilder b(model, plan, out);
    b.embed(std::sqrt(static_cast<float>(model.embedding_width)));
    if (auto r = b.per_layer_inputs(); !r.ok()) return r;
    for (std::uint32_t layer = 0; layer < model.layers.size(); ++layer) {
        if (auto r = b.attention(layer); !r.ok()) return r;
        if (auto r = b.gated_feed_forward(layer); !r.ok()) return r;
        if (auto r = b.per_layer_block(layer); !r.ok()) return r;
    }
    return b.pooled_output();
}

}  // namespace

const Architecture kEmbeddingGemma2{kArch, describe, EncodeFn{encode}};

}  // namespace bllm::arch
