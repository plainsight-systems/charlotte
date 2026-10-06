#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/gguf/index.h"

namespace bllm::model {

// Contract 4: the model description.
//
// Every number the planner, the cache and the graph need, with no reference to
// which architecture produced it. An architecture fills it from the tensor
// index; nothing downstream reads the file's metadata again.
//
// Values that can differ between layers are per layer. Models vary head
// counts, attention window and rotary base by layer, and a field that is
// global here would force every such model into a special case downstream.
//
// Weights are named by role and identifier. The graph and the kernels address
// a weight as "layer 3, key projection", never by the tensor's name in the
// file, which is the architecture's business alone.

enum class LayerIndex : std::uint32_t {};

// The roles a weight plays in a layer: the vocabulary every architecture's
// graph shares. Adding a role is a change to this contract.
enum class Role {
    AttentionNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    QueryNorm,
    KeyNorm,
    PostAttentionNorm,
    FeedForwardNorm,
    Gate,
    Up,
    Down,
    PostFeedForwardNorm,
    Count,
};

inline constexpr std::size_t kRoleCount = static_cast<std::size_t>(Role::Count);

// Which of a head's dimensions RoPE rotates together. Not a reparametrization
// a converter could fold into the weights, so a kernel variant: llama.cpp's
// llama_model_rope_type gives each architecture's.
enum class RotaryPairing {
    // Dimensions 2k and 2k + 1: llama.cpp's NORM. Llama's files, whose
    // converter permuted Q and K into this order.
    Adjacent,
    // Dimensions k and k + d / 2: NEOX, Hugging Face's rotate_half. Qwen3,
    // Gemma 3.
    Halves,
};

struct LayerDescription {
    std::uint32_t query_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dimension;
    std::uint32_t feed_forward_width;
    // Tokens this layer attends back over: the context length for a
    // full-attention layer, the window for a sliding-window layer. A query at
    // position p reads keys at p - attention_window + 1 .. p.
    std::uint32_t attention_window;
    float rope_base;
    // The tensor filling each role. Empty where this layer has no such weight.
    std::array<std::optional<gguf::TensorId>, kRoleCount> tensors;
};

struct ModelDescription {
    std::uint32_t vocabulary_size;
    std::uint32_t embedding_width;
    std::uint32_t trained_context;
    float norm_epsilon;
    gguf::TensorId token_embedding;
    gguf::TensorId output_norm;
    // Empty when the output head reads the token embedding.
    std::optional<gguf::TensorId> output_head;
    // The architecture's, set by its describe.
    RotaryPairing rotary_pairing;
    // What attention multiplies q · k by: 1 / sqrt(head dimension), but for
    // Gemma 3 27B's 1 / sqrt(embedding width / query heads), which no GGUF
    // key states and llama.cpp infers from its 62 layers, after Google's
    // gemma_pytorch configuration. Set by describe.
    float attention_scale;
    // Llama 3's long-context scaling, rope_freqs.weight: F32, one factor a
    // pair, each pair's frequency divided by its own, in every layer. Empty
    // where the file has none. Every head is rotated whole: describe refuses
    // a file that rotates part of a head or declares another scaling.
    std::optional<gguf::TensorId> rotary_factors;
    std::vector<LayerDescription> layers;
};

}  // namespace bllm::model
