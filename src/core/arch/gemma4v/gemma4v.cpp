#include "core/arch/gemma4v/gemma4v.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/describe.h"
#include "core/graph/vision.h"

namespace bllm::arch {
namespace {

using model::Role;

constexpr std::string_view kProjector = "gemma4v";

// Each layer's thirteen roles, named as the mmproj names them.
constexpr std::array kRoles{
    RoleName{Role::AttentionNorm, "ln1.weight"},
    RoleName{Role::Query, "attn_q.weight"},
    RoleName{Role::Key, "attn_k.weight"},
    RoleName{Role::Value, "attn_v.weight"},
    RoleName{Role::QueryNorm, "attn_q_norm.weight"},
    RoleName{Role::KeyNorm, "attn_k_norm.weight"},
    RoleName{Role::AttentionOutput, "attn_out.weight"},
    RoleName{Role::PostAttentionNorm, "attn_post_norm.weight"},
    RoleName{Role::FeedForwardNorm, "ln2.weight"},
    RoleName{Role::Gate, "ffn_gate.weight"},
    RoleName{Role::Up, "ffn_up.weight"},
    RoleName{Role::Down, "ffn_down.weight"},
    RoleName{Role::PostFeedForwardNorm, "ffn_post_norm.weight"},
};

// The numbers the file does not carry, the encoder's convention: θ 100 for
// the axial rotation, as llama.cpp sets it for gemma4v and Google's config
// states; the pooling where the file omits it.
constexpr float kRopeBase = 100.0f;
constexpr std::uint32_t kDefaultPooling = 3;

DescribeResult refused(DescribeError error, std::string subject) { return {error, std::move(subject)}; }

DescribeResult metadata(gguf::MetadataError e, std::string subject) {
    return refused(e == gguf::MetadataError::MissingKey ? DescribeError::MissingKey : DescribeError::WrongKeyType,
                   std::move(subject));
}

DescribeResult read_u32(const gguf::TensorIndex& index, const std::string& key, std::uint32_t& out) {
    if (const auto e = index.read_u32(key, out); e != gguf::MetadataError::Ok) return metadata(e, key);
    return {};
}

// An optional key: `fallback` where the file omits it.
DescribeResult read_u32_or(const gguf::TensorIndex& index, const std::string& key, std::uint32_t fallback,
                           std::uint32_t& out) {
    if (const auto e = index.read_u32(key, out); e == gguf::MetadataError::MissingKey) {
        out = fallback;
    } else if (e != gguf::MetadataError::Ok) {
        return metadata(e, key);
    }
    return {};
}

DescribeResult read_bool_or(const gguf::TensorIndex& index, const std::string& key, bool fallback, bool& out) {
    if (const auto e = index.read_bool(key, out); e == gguf::MetadataError::MissingKey) {
        out = fallback;
    } else if (e != gguf::MetadataError::Ok) {
        return metadata(e, key);
    }
    return {};
}

DescribeResult expect_string(const gguf::TensorIndex& index, const std::string& key, std::string_view want) {
    std::string_view got;
    if (const auto e = index.read_string(key, got); e != gguf::MetadataError::Ok) return metadata(e, key);
    if (got != want) return refused(DescribeError::UnsupportedValue, key);
    return {};
}

// A float array that, where the file has it, holds `want` in every element.
DescribeResult expect_all(const gguf::TensorIndex& index, const std::string& key, float want) {
    std::vector<float> values;
    if (const auto e = index.read_f32_array(key, values); e == gguf::MetadataError::MissingKey) {
        return {};
    } else if (e != gguf::MetadataError::Ok) {
        return metadata(e, key);
    }
    for (const float v : values) {
        if (v != want) return refused(DescribeError::UnsupportedValue, key);
    }
    return {};
}

// A tensor of exactly `dims`, F32 where `f32`.
DescribeResult find(const gguf::TensorIndex& index, const std::string& name, std::initializer_list<std::uint64_t> dims,
                    bool f32, gguf::TensorId& out) {
    const std::optional<gguf::TensorId> id = index.find(name);
    if (!id) return refused(DescribeError::MissingTensor, name);
    const gguf::TensorEntry& t = index.tensor(*id);
    if (t.dimension_count != dims.size()) return refused(DescribeError::ShapeMismatch, name);
    std::size_t i = 0;
    for (const std::uint64_t d : dims) {
        if (t.dimensions[i++] != d) return refused(DescribeError::ShapeMismatch, name);
    }
    if (f32 && t.type != gguf::TensorType::F32) return refused(DescribeError::UnsupportedValue, name);
    out = *id;
    return {};
}

DescribeResult describe(const gguf::TensorIndex& index, model::ModelDescription& out) {
    if (auto r = expect_string(index, "general.architecture", "clip"); !r.ok()) return r;
    if (auto r = expect_string(index, "general.type", "mmproj"); !r.ok()) return r;
    if (auto r = expect_string(index, "clip.vision.projector_type", kProjector); !r.ok()) return r;
    bool has_vision = false;
    if (const auto e = index.read_bool("clip.has_vision_encoder", has_vision); e != gguf::MetadataError::Ok) {
        return metadata(e, "clip.has_vision_encoder");
    }
    if (!has_vision) return refused(DescribeError::UnsupportedValue, "clip.has_vision_encoder");

    Hyperparameters hp{};
    if (auto r = read_u32(index, "clip.vision.block_count", hp.block_count); !r.ok()) return r;
    if (auto r = read_u32(index, "clip.vision.embedding_length", hp.embedding_length); !r.ok()) return r;
    if (auto r = read_u32(index, "clip.vision.feed_forward_length", hp.feed_forward_length); !r.ok()) return r;
    if (auto r = read_u32(index, "clip.vision.attention.head_count", hp.head_count); !r.ok()) return r;
    if (const auto e = index.read_f32("clip.vision.attention.layer_norm_epsilon", hp.norm_epsilon);
        e != gguf::MetadataError::Ok) {
        return metadata(e, "clip.vision.attention.layer_norm_epsilon");
    }
    if (hp.block_count == 0 || hp.block_count > index.tensors().size()) {
        return refused(DescribeError::InvalidValue, "clip.vision.block_count");
    }
    if (hp.head_count == 0 || hp.embedding_length % hp.head_count != 0) {
        return refused(DescribeError::InvalidValue, "clip.vision.attention.head_count");
    }
    hp.head_dimension = hp.embedding_length / hp.head_count;
    if (auto r = read_u32_or(index, "clip.vision.attention.head_count_kv", hp.head_count, hp.head_count_kv); !r.ok()) {
        return r;
    }
    if (hp.head_count_kv != hp.head_count) {
        return refused(DescribeError::UnsupportedValue, "clip.vision.attention.head_count_kv");
    }
    std::uint32_t head_dim = 0;
    if (auto r = read_u32_or(index, "clip.vision.attention.head_dim", hp.head_dimension, head_dim); !r.ok()) return r;
    if (head_dim != hp.head_dimension) return refused(DescribeError::UnsupportedValue, "clip.vision.attention.head_dim");
    hp.rope_base = kRopeBase;
    hp.context_length = std::numeric_limits<std::uint32_t>::max();

    model::VisionDescription v{};
    if (auto r = read_u32(index, "clip.vision.patch_size", v.patch_size); !r.ok()) return r;
    if (v.patch_size == 0) return refused(DescribeError::InvalidValue, "clip.vision.patch_size");
    if (auto r = read_u32(index, "clip.vision.projection_dim", v.projection_width); !r.ok()) return r;
    if (auto r = read_u32_or(index, "clip.vision.projector.scale_factor", kDefaultPooling, v.pooling); !r.ok()) return r;
    if (v.pooling == 0) return refused(DescribeError::InvalidValue, "clip.vision.projector.scale_factor");
    if (auto r = expect_all(index, "clip.vision.image_mean", 0.0f); !r.ok()) return r;
    if (auto r = expect_all(index, "clip.vision.image_std", 1.0f); !r.ok()) return r;

    // The activation: Google's GELU, tanh form, whatever llama.cpp's default
    // (gemma4v.h); a file asking for SiLU is not this encoder.
    bool silu = false;
    if (auto r = read_bool_or(index, "clip.use_silu", false, silu); !r.ok()) return r;
    if (silu) return refused(DescribeError::UnsupportedValue, "clip.use_silu");

    const std::uint64_t w = hp.embedding_length;
    if (auto r = find(index, "v.patch_embd.weight", {v.patch_size, v.patch_size, 3, w}, true, v.patch_embedding);
        !r.ok()) {
        return r;
    }
    const std::optional<gguf::TensorId> table = index.find("v.position_embd.weight");
    if (!table) return refused(DescribeError::MissingTensor, "v.position_embd.weight");
    v.positions = static_cast<std::uint32_t>(index.tensor(*table).dimensions[1]);
    if (auto r = find(index, "v.position_embd.weight", {w, v.positions, 2}, true, v.position_table); !r.ok()) return r;
    if (auto r = find(index, "mm.input_projection.weight", {w, v.projection_width}, false, v.projection); !r.ok()) {
        return r;
    }

    out = model::ModelDescription{};
    out.vocabulary_size = 0;
    out.embedding_width = hp.embedding_length;
    // No context: a vision step holds patches, as many as the image budget
    // allows (graph/vision.h).
    out.trained_context = 0;
    out.norm_epsilon = hp.norm_epsilon;
    out.rotary_pairing = model::RotaryPairing::Axial;
    out.activation = model::FeedForwardActivation::GeluTanh;
    out.attention_scale = 1.0f;   // QK-norm takes its place
    out.mask = model::AttentionMask::Bidirectional;
    out.value_norm = true;
    out.layers.reserve(hp.block_count);
    for (std::uint32_t l = 0; l < hp.block_count; ++l) {
        // Every patch reads every other: the widest window there is.
        out.layers.push_back(model::LayerDescription{hp.head_count, hp.head_count, hp.head_dimension,
                                                     hp.feed_forward_length, std::numeric_limits<std::uint32_t>::max(),
                                                     kRopeBase, {}});
        if (auto r = describe_layer_tensors(index, "v.blk.", l, hp, {hp.head_count, hp.head_dimension}, kRoles,
                                            out.layers[l]);
            !r.ok()) {
            return r;
        }
    }
    out.vision = v;

    // Principle 5: every tensor of the vision encoder is accounted for. One
    // the description does not name — a bias, a layer scale, a
    // standardization, a clamp on the projection — would need a block this
    // graph does not have, so it is refused by name.
    std::vector<bool> described(index.tensors().size(), false);
    for (const gguf::TensorId id : {v.patch_embedding, v.position_table, v.projection}) {
        described[static_cast<std::size_t>(id)] = true;
    }
    for (const model::LayerDescription& l : out.layers) {
        for (const auto& t : l.tensors) {
            if (t) described[static_cast<std::size_t>(*t)] = true;
        }
    }
    for (std::size_t i = 0; i < index.tensors().size(); ++i) {
        const std::string_view name = index.tensors()[i].name;
        const bool vision = name.starts_with("v.") || (name.starts_with("mm.") && !name.starts_with("mm.a."));
        if (vision && !described[i]) return refused(DescribeError::UnsupportedValue, std::string(name));
    }
    return {};
}

DescribeResult pairs(const model::ModelDescription& vision, const model::ModelDescription& text,
                     std::string_view text_architecture) {
    if (text_architecture != "gemma-embedding2") {
        return refused(DescribeError::UnsupportedValue, std::string(text_architecture));
    }
    if (!vision.vision || !text.encoder || text.embedding_width != vision.vision->projection_width) {
        return refused(DescribeError::ShapeMismatch, "clip.vision.projection_dim");
    }
    return {};
}

// The patch embedding and the positions; each layer's attention and gated
// feed-forward block; the pooled projection (graph/vision.h).
graph::GraphResult vision(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                          std::vector<kernels::Launch>& out) {
    graph::VisionBuilder b(model, plan, out);
    if (auto r = b.patch_embedding(); !r.ok()) return r;
    if (auto r = b.positions(); !r.ok()) return r;
    for (std::uint32_t layer = 0; layer < model.layers.size(); ++layer) {
        if (auto r = b.attention(layer); !r.ok()) return r;
        if (auto r = b.gated_feed_forward(layer); !r.ok()) return r;
    }
    return b.pooled_projection();
}

}  // namespace

const MediaEncoder kGemma4Vision{MediaKind::Image, kProjector, describe, vision, nullptr, pairs,
                                  {"<|image|>", "<|image>", "<image|>"}};

}  // namespace bllm::arch
