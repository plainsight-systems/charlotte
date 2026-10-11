#include "core/arch/describe.h"

#include <array>
#include <cmath>
#include <string>

namespace bllm::arch {
namespace {

std::string key_of(std::string_view arch, std::string_view key) {
    return std::string(arch) + "." + std::string(key);
}

DescribeResult failure(DescribeError error, std::string subject) {
    return DescribeResult{error, std::move(subject)};
}

DescribeResult from_metadata(gguf::MetadataError error, std::string subject) {
    return error == gguf::MetadataError::MissingKey
               ? failure(DescribeError::MissingKey, std::move(subject))
               : failure(DescribeError::WrongKeyType, std::move(subject));
}

DescribeResult read_u32(const gguf::TensorIndex& index, std::string_view arch,
                        std::string_view key, std::uint32_t& out) {
    const std::string name = key_of(arch, key);
    if (const auto e = index.read_u32(name, out); e != gguf::MetadataError::Ok) {
        return from_metadata(e, name);
    }
    return {};
}

DescribeResult read_f32(const gguf::TensorIndex& index, std::string_view arch,
                        std::string_view key, float& out) {
    const std::string name = key_of(arch, key);
    if (const auto e = index.read_f32(name, out); e != gguf::MetadataError::Ok) {
        return from_metadata(e, name);
    }
    return {};
}

// The shape a weight in `role` has: GGUF lists a matrix's input width first.
struct Shape {
    std::uint32_t dimension_count;
    std::array<std::uint64_t, 2> dimensions;
};

Shape shape_of(model::Role role, const Hyperparameters& hp, const LayerHeads& heads) {
    const std::uint64_t e = hp.embedding_length;
    const std::uint64_t q = std::uint64_t{hp.head_count} * heads.head_dimension;
    const std::uint64_t kv = std::uint64_t{heads.key_value_heads} * heads.head_dimension;
    const std::uint64_t f = hp.feed_forward_length;
    using model::Role;
    switch (role) {
        case Role::Query: return {2, {e, q}};
        case Role::Key: case Role::Value: return {2, {e, kv}};
        case Role::AttentionOutput: return {2, {q, e}};
        case Role::Gate: case Role::Up: return {2, {e, f}};
        case Role::Down: return {2, {f, e}};
        case Role::QueryNorm: case Role::KeyNorm: return {1, {heads.head_dimension, 0}};
        case Role::AttentionNorm: case Role::PostAttentionNorm:
        case Role::FeedForwardNorm: case Role::PostFeedForwardNorm: case Role::PerLayerPostNorm: return {1, {e, 0}};
        // The per-layer block's shapes are its architecture's, which checks
        // them with its own widths (arch/embedding_gemma2).
        case Role::PerLayerGate: case Role::PerLayerProjection: case Role::OutputScale:
        case Role::Count: break;
    }
    return {0, {0, 0}};
}

// Norm gains: the norm and rope kernels bind them as f32 and read no format
// (kernels/norm/norm.h, kernels/rope/rope.h), so they must be stored as F32.
bool is_gain(model::Role role) {
    using model::Role;
    switch (role) {
        case Role::AttentionNorm: case Role::QueryNorm: case Role::KeyNorm: case Role::PostAttentionNorm:
        case Role::FeedForwardNorm: case Role::PostFeedForwardNorm: case Role::PerLayerPostNorm:
        case Role::OutputScale: return true;
        default: return false;
    }
}

// A tensor a kernel binds as f32 must be F32; any other format, however well
// supported, would be read as the wrong bits.
DescribeResult require_f32(const gguf::TensorIndex& index, const std::optional<gguf::TensorId>& tensor,
                           const std::string& name) {
    if (tensor && index.tensor(*tensor).type != gguf::TensorType::F32) {
        return failure(DescribeError::UnsupportedValue, name);
    }
    return {};
}

bool has_shape(const gguf::TensorEntry& t, std::uint32_t dimension_count,
               std::uint64_t d0, std::uint64_t d1) {
    if (t.dimension_count != dimension_count || t.dimensions[0] != d0) return false;
    return dimension_count == 1 || t.dimensions[1] == d1;
}

// Finds `name` and checks its shape; `required` decides whether absence fails.
DescribeResult find_tensor(const gguf::TensorIndex& index, const std::string& name,
                           std::uint32_t dimension_count, std::uint64_t d0, std::uint64_t d1,
                           bool required, std::optional<gguf::TensorId>& out) {
    out = index.find(name);
    if (!out.has_value()) {
        return required ? failure(DescribeError::MissingTensor, name) : DescribeResult{};
    }
    if (!has_shape(index.tensor(*out), dimension_count, d0, d1)) {
        return failure(DescribeError::ShapeMismatch, name);
    }
    return {};
}

}  // namespace

DescribeResult read_u32_or(const gguf::TensorIndex& index, std::string_view arch,
                           std::string_view key, std::uint32_t fallback, std::uint32_t& out) {
    const std::string name = key_of(arch, key);
    const auto e = index.read_u32(name, out);
    if (e == gguf::MetadataError::MissingKey) out = fallback;
    else if (e != gguf::MetadataError::Ok) return from_metadata(e, name);
    return {};
}

DescribeResult read_f32_or(const gguf::TensorIndex& index, std::string_view arch,
                           std::string_view key, float fallback, float& out) {
    const std::string name = key_of(arch, key);
    const auto e = index.read_f32(name, out);
    if (e == gguf::MetadataError::MissingKey) out = fallback;
    else if (e != gguf::MetadataError::Ok) return from_metadata(e, name);
    return {};
}

DescribeResult read_hyperparameters(const gguf::TensorIndex& index, std::string_view arch,
                                    Hyperparameters& out) {
    struct Required {
        std::string_view key;
        std::uint32_t* value;
    };
    for (const Required& r : {
             Required{"block_count", &out.block_count},
             Required{"context_length", &out.context_length},
             Required{"embedding_length", &out.embedding_length},
             Required{"feed_forward_length", &out.feed_forward_length},
             Required{"attention.head_count", &out.head_count},
         }) {
        if (auto result = read_u32(index, arch, r.key, *r.value); !result.ok()) return result;
        if (*r.value == 0) return failure(DescribeError::InvalidValue, key_of(arch, r.key));
    }
    // Key-value heads: one number, or one a layer (llama.cpp's
    // get_key_or_arr).
    const std::string kv_key = key_of(arch, "attention.head_count_kv");
    out.head_count_kv_per_layer.clear();
    if (const auto e = index.read_u32(kv_key, out.head_count_kv); e == gguf::MetadataError::WrongType) {
        if (auto r = index.read_u32_array(kv_key, out.head_count_kv_per_layer); r != gguf::MetadataError::Ok) {
            return from_metadata(r, kv_key);
        }
        if (out.head_count_kv_per_layer.size() != out.block_count) return failure(DescribeError::InvalidValue, kv_key);
        for (const std::uint32_t kv : out.head_count_kv_per_layer) {
            if (kv == 0 || out.head_count % kv != 0) return failure(DescribeError::InvalidValue, kv_key);
        }
        out.head_count_kv = out.head_count_kv_per_layer.front();
    } else if (e != gguf::MetadataError::Ok) {
        return from_metadata(e, kv_key);
    } else if (out.head_count_kv == 0) {
        return failure(DescribeError::InvalidValue, kv_key);
    }
    if (auto r = read_f32(index, arch, "attention.layer_norm_rms_epsilon", out.norm_epsilon); !r.ok()) return r;
    if (auto r = read_f32(index, arch, "rope.freq_base", out.rope_base); !r.ok()) return r;

    if (out.head_count % out.head_count_kv != 0) {
        return failure(DescribeError::InvalidValue, key_of(arch, "attention.head_count_kv"));
    }
    if (auto r = read_u32_or(index, arch, "attention.key_length",
                             out.embedding_length / out.head_count, out.head_dimension);
        !r.ok()) {
        return r;
    }
    if (out.head_dimension == 0) return failure(DescribeError::InvalidValue, key_of(arch, "attention.key_length"));

    // The rope kernel rotates every dimension of a head, at unscaled
    // positions (kernels/rope/rope.h). A file that rotates part of each head
    // or scales its positions would run with every position wrong, so it is
    // refused by name; so is an odd head, whose last dimension has no pair.
    if (out.head_dimension % 2 != 0) {
        return failure(DescribeError::UnsupportedValue, key_of(arch, "attention.key_length"));
    }
    std::uint32_t rotated = 0;
    if (auto r = read_u32_or(index, arch, "rope.dimension_count", out.head_dimension, rotated); !r.ok()) return r;
    if (rotated != out.head_dimension) {
        return failure(DescribeError::UnsupportedValue, key_of(arch, "rope.dimension_count"));
    }
    // No kernel caps logits with tanh, as Gemma 2 does: a file declaring a
    // nonzero cap, on attention scores or on the output, would run uncapped.
    for (const std::string_view key : {std::string_view{"attn_logit_softcapping"},
                                       std::string_view{"final_logit_softcapping"}}) {
        float cap = 0;
        if (auto r = read_f32_or(index, arch, key, 0.0f, cap); !r.ok()) return r;
        if (cap != 0.0f) return failure(DescribeError::UnsupportedValue, key_of(arch, key));
    }

    const std::string scaling_key = key_of(arch, "rope.scaling.type");
    std::string_view scaling;
    if (const auto e = index.read_string(scaling_key, scaling); e == gguf::MetadataError::Ok) {
        if (scaling != "none") return failure(DescribeError::UnsupportedValue, scaling_key);
    } else if (e != gguf::MetadataError::MissingKey) {
        return from_metadata(e, scaling_key);
    }

    // The description has one head dimension; a model whose values are a
    // different width from its keys is valid, and not one this reads.
    std::uint32_t value_length = 0;
    if (auto r = read_u32_or(index, arch, "attention.value_length", out.head_dimension, value_length); !r.ok()) return r;
    if (value_length != out.head_dimension) {
        return failure(DescribeError::UnsupportedValue, key_of(arch, "attention.value_length"));
    }

    // Every layer needs tensors of its own, so a file cannot have more layers
    // than tensors. Checked before the count sizes anything.
    if (out.block_count > index.tensors().size()) {
        return failure(DescribeError::InvalidValue, key_of(arch, "block_count"));
    }
    return {};
}

DescribeResult describe_layers(const gguf::TensorIndex& index, const Hyperparameters& hp,
                               std::span<const RoleName> roles, Conventions conventions,
                               model::ModelDescription& out, std::span<const LayerHeads> heads) {
    gguf::ArrayLocation tokens{};
    if (const auto e = index.read_array("tokenizer.ggml.tokens", tokens); e != gguf::MetadataError::Ok) {
        return from_metadata(e, "tokenizer.ggml.tokens");
    }
    out.vocabulary_size = static_cast<std::uint32_t>(tokens.element_count);
    out.embedding_width = hp.embedding_length;
    out.trained_context = hp.context_length;
    out.norm_epsilon = hp.norm_epsilon;

    const std::uint64_t e = hp.embedding_length;
    const std::uint64_t vocab = out.vocabulary_size;
    std::optional<gguf::TensorId> found;
    if (auto r = find_tensor(index, "token_embd.weight", 2, e, vocab, true, found); !r.ok()) return r;
    out.token_embedding = *found;
    if (auto r = find_tensor(index, "output_norm.weight", 1, e, 0, true, found); !r.ok()) return r;
    if (auto r = require_f32(index, found, "output_norm.weight"); !r.ok()) return r;
    out.output_norm = *found;
    // Absent when the output head reads the token embedding. An
    // architecture whose output.weight is no vocabulary head reads it itself.
    out.output_head.reset();
    if (conventions.vocabulary_head) {
        if (auto r = find_tensor(index, "output.weight", 2, e, vocab, false, out.output_head); !r.ok()) return r;
    }

    out.rotary_pairing = conventions.pairing;
    out.activation = conventions.activation;
    out.attention_scale = 1.0f / std::sqrt(static_cast<float>(hp.head_dimension));
    // Llama 3's frequency factors, one a pair; the rope kernel binds them as
    // f32.
    if (auto r = find_tensor(index, "rope_freqs.weight", 1, hp.head_dimension / 2, 0, false, out.rotary_factors);
        !r.ok()) {
        return r;
    }
    if (auto r = require_f32(index, out.rotary_factors, "rope_freqs.weight"); !r.ok()) return r;

    if (!heads.empty() && heads.size() != hp.block_count) return failure(DescribeError::InvalidValue, "layer heads");
    out.layers.clear();
    for (std::uint32_t layer = 0; layer < hp.block_count; ++layer) {
        LayerHeads h{hp.head_count_kv, hp.head_dimension};
        if (!heads.empty()) {
            h = heads[layer];
        } else if (!hp.head_count_kv_per_layer.empty()) {
            h.key_value_heads = hp.head_count_kv_per_layer[layer];
        }
        out.layers.push_back(model::LayerDescription{hp.head_count, h.key_value_heads, h.head_dimension,
                                                     hp.feed_forward_length, hp.context_length, hp.rope_base, {}});
        if (auto r = describe_layer_tensors(index, "blk.", layer, hp, h, roles, out.layers[layer]); !r.ok()) {
            return r;
        }
    }
    return {};
}

DescribeResult describe_layer_tensors(const gguf::TensorIndex& index, std::string_view prefix, std::uint32_t layer,
                                      const Hyperparameters& hp, LayerHeads heads, std::span<const RoleName> roles,
                                      model::LayerDescription& out) {
    const std::string layer_prefix = std::string(prefix) + std::to_string(layer) + ".";
    for (const RoleName& rn : roles) {
        const Shape shape = shape_of(rn.role, hp, heads);
        auto& slot = out.tensors[static_cast<std::size_t>(rn.role)];
        const std::string name = layer_prefix + std::string(rn.suffix);
        if (auto r = find_tensor(index, name, shape.dimension_count, shape.dimensions[0], shape.dimensions[1], true,
                                 slot);
            !r.ok()) {
            return r;
        }
        if (is_gain(rn.role)) {
            if (auto r = require_f32(index, slot, name); !r.ok()) return r;
        }
    }
    return {};
}

}  // namespace bllm::arch
