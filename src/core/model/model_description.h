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
    std::vector<LayerDescription> layers;
};

}  // namespace bllm::model
