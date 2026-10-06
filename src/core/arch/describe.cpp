#include "core/arch/describe.h"

#include <array>
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

Shape shape_of(model::Role role, const Hyperparameters& hp) {
    const std::uint64_t e = hp.embedding_length;
    const std::uint64_t q = std::uint64_t{hp.head_count} * hp.head_dimension;
    const std::uint64_t kv = std::uint64_t{hp.head_count_kv} * hp.head_dimension;
    const std::uint64_t f = hp.feed_forward_length;
    using model::Role;
    switch (role) {
        case Role::Query: return {2, {e, q}};
        case Role::Key: case Role::Value: return {2, {e, kv}};
        case Role::AttentionOutput: return {2, {q, e}};
        case Role::Gate: case Role::Up: return {2, {e, f}};
        case Role::Down: return {2, {f, e}};
        case Role::QueryNorm: case Role::KeyNorm: return {1, {hp.head_dimension, 0}};
        case Role::AttentionNorm: case Role::PostAttentionNorm:
        case Role::FeedForwardNorm: case Role::PostFeedForwardNorm: return {1, {e, 0}};
        case Role::Count: break;
    }
    return {0, {0, 0}};
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
             Required{"attention.head_count_kv", &out.head_count_kv},
         }) {
        if (auto result = read_u32(index, arch, r.key, *r.value); !result.ok()) return result;
        if (*r.value == 0) return failure(DescribeError::InvalidValue, key_of(arch, r.key));
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
    // refused by name.
    std::uint32_t rotated = 0;
    if (auto r = read_u32_or(index, arch, "rope.dimension_count", out.head_dimension, rotated); !r.ok()) return r;
    if (rotated != out.head_dimension) {
        return failure(DescribeError::UnsupportedValue, key_of(arch, "rope.dimension_count"));
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
                               std::span<const RoleName> roles, model::RotaryPairing pairing,
                               model::ModelDescription& out) {
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
    out.output_norm = *found;
    // Absent when the output head reads the token embedding.
    if (auto r = find_tensor(index, "output.weight", 2, e, vocab, false, out.output_head); !r.ok()) return r;

    out.rotary_pairing = pairing;
    // Llama 3's frequency factors, one a pair; the rope kernel binds them as
    // f32.
    if (auto r = find_tensor(index, "rope_freqs.weight", 1, hp.head_dimension / 2, 0, false, out.rotary_factors);
        !r.ok()) {
        return r;
    }
    if (out.rotary_factors && index.tensor(*out.rotary_factors).type != gguf::TensorType::F32) {
        return failure(DescribeError::UnsupportedValue, "rope_freqs.weight");
    }

    out.layers.assign(hp.block_count, model::LayerDescription{
        hp.head_count, hp.head_count_kv, hp.head_dimension, hp.feed_forward_length,
        hp.context_length, hp.rope_base, {}});
    for (std::uint32_t layer = 0; layer < hp.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const RoleName& rn : roles) {
            const Shape shape = shape_of(rn.role, hp);
            auto& slot = out.layers[layer].tensors[static_cast<std::size_t>(rn.role)];
            if (auto r = find_tensor(index, prefix + std::string(rn.suffix), shape.dimension_count,
                                     shape.dimensions[0], shape.dimensions[1], true, slot);
                !r.ok()) {
                return r;
            }
        }
    }
    return {};
}

}  // namespace bllm::arch
