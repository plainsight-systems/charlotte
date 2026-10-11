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
    // Gemma 4's per-layer block (graph/encoder.h): the gate from X to the
    // per-layer width, the projection back, its post-norm, and the scale X
    // takes once the block is added — one F32 value.
    PerLayerGate,
    PerLayerProjection,
    PerLayerPostNorm,
    OutputScale,
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
    // Each half of the head NEOX-paired within itself — dimensions k and k +
    // d / 4 of the first half, turned by the patch's column, and the same
    // within the second half, turned by its row: Gemma 4's vision encoder,
    // rope_type axial (kernels/rope/rope.h).
    Axial,
};

// The activation a gated feed-forward block applies to its gate before
// multiplying by up: SiLU (SwiGLU) for Qwen3 and Llama, GELU in its tanh
// form (GeGLU) for Gemma 3, as llama.cpp's graphs apply them.
enum class FeedForwardActivation {
    SiLU,
    GeluTanh,
    // A GLU's gate, σ(x): a Conformer's convolution module (graph/audio.h).
    Sigmoid,
};

// Which keys a query reads: those behind it, a decoder's, or those either
// side, an encoder's (kernels/block_attention/block_attention.h).
enum class AttentionMask {
    Causal,
    Bidirectional,
};

// How an encoder's rows become one vector: the mean of every row.
enum class Pooling {
    Mean,
};

struct LayerDescription {
    std::uint32_t query_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dimension;
    std::uint32_t feed_forward_width;
    // Tokens this layer attends over: the context length for a
    // full-attention layer, the window for a sliding-window layer. Under a
    // causal mask a query at position p reads keys at p - attention_window +
    // 1 .. p; under a bidirectional one, keys within attention_window / 2 of
    // p either side, llama.cpp's symmetric window.
    std::uint32_t attention_window;
    float rope_base;
    // The tensor filling each role. Empty where this layer has no such weight.
    std::array<std::optional<gguf::TensorId>, kRoleCount> tensors;
};

// What an encoder has beside its layers (graph/encoder.h): the projection of
// the embedding to every layer's per-layer input, W to layers × width, and
// that input's norm; the output projection, W to the embedding's length,
// after pooling; and the pooling.
struct EncoderDescription {
    gguf::TensorId per_layer_projection;
    gguf::TensorId per_layer_norm;
    std::uint32_t per_layer_width;
    gguf::TensorId output_projection;
    std::uint32_t embedding_length;
    Pooling pooling;
};

// What a vision encoder has beside its layers (graph/vision.h): the patch
// embedding, F32, 3 P² to W; the learned position table, F32, W ×
// `positions` × 2, the column table then the row table; the pooling of the
// patch grid, k × k; and the projection, W to the text model's width.
struct VisionDescription {
    gguf::TensorId patch_embedding;
    std::uint32_t patch_size;
    gguf::TensorId position_table;
    std::uint32_t positions;
    std::uint32_t pooling;
    gguf::TensorId projection;
    std::uint32_t projection_width;
};

// A product's clamp bounds: four one-float F32 tensors beside its weight
// (arch/gemma4a/gemma4a.h), what its input and its output are clamped to.
struct ClampBounds {
    gguf::TensorId input_min;
    gguf::TensorId input_max;
    gguf::TensorId output_min;
    gguf::TensorId output_max;
};

// A product of an audio encoder: its weight, and its bounds where it clamps.
struct AudioProduct {
    gguf::TensorId weight;
    std::optional<ClampBounds> bounds;
};

// One layer of a Conformer (graph/audio.h): two half-step feed-forwards,
// attention over a window, a convolution module, a closing norm.
struct ConformerLayer {
    gguf::TensorId ffn_norm;
    AudioProduct ffn_up;
    AudioProduct ffn_down;
    gguf::TensorId ffn_post_norm;
    gguf::TensorId attention_norm;
    AudioProduct query;
    AudioProduct key;
    AudioProduct value;
    gguf::TensorId per_dim_scale;
    gguf::TensorId relative;
    AudioProduct attention_output;
    gguf::TensorId attention_post_norm;
    gguf::TensorId convolution_norm;
    AudioProduct pointwise_in;
    gguf::TensorId depthwise;
    gguf::TensorId depthwise_norm;
    AudioProduct pointwise_out;
    gguf::TensorId ffn_norm_1;
    AudioProduct ffn_up_1;
    AudioProduct ffn_down_1;
    gguf::TensorId ffn_post_norm_1;
    gguf::TensorId output_norm;
};

// What an audio encoder is (arch/gemma4a/gemma4a.h): its subsampling, its
// Conformer layers, its output and multimodal projections, and the numbers
// its file does not carry.
struct AudioDescription {
    std::uint32_t mel_bins;
    std::array<gguf::TensorId, 2> convolution;        // F32, 3 × 3, stride 2
    std::array<gguf::TensorId, 2> convolution_norm;   // F32, a gain a channel
    std::array<std::uint32_t, 2> channels;
    gguf::TensorId input_projection;                  // F32
    std::vector<ConformerLayer> layers;
    std::uint32_t width;
    std::uint32_t heads;
    std::uint32_t feed_forward;
    std::uint32_t kernel;                             // the depthwise convolution's
    std::uint32_t chunk;
    std::uint32_t window;                             // a row and its window − 1 before
    float residual_weight;
    float softcap;
    float q_scale;
    float k_scale;
    gguf::TensorId output_projection;
    gguf::TensorId output_bias;
    std::uint32_t output_width;
    gguf::TensorId projection;
    std::uint32_t projection_width;
};

// A weight and the F32 tensors its product reads from the weight's own
// binding — clamp bounds, a bias — which the plan keeps in one buffer within
// one binding (residency/plan.h).
struct ProductSpan {
    gguf::TensorId weight;
    std::vector<gguf::TensorId> members;   // the weight among them
};

struct ModelDescription {
    std::uint32_t vocabulary_size;
    std::uint32_t embedding_width;
    std::uint32_t trained_context;
    float norm_epsilon;
    // Every text architecture's; absent only for a vision encoder, which has
    // no vocabulary (`vision` below). The decoder's and encoder's builders
    // require both.
    std::optional<gguf::TensorId> token_embedding;
    std::optional<gguf::TensorId> output_norm;
    // Empty when the output head reads the token embedding.
    std::optional<gguf::TensorId> output_head;
    // The architecture's, set by its describe.
    RotaryPairing rotary_pairing;
    FeedForwardActivation activation;
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
    // Causal for a decoder; Bidirectional for an encoder.
    AttentionMask mask = AttentionMask::Causal;
    // Values normalized per head without a gain, as Gemma 4's are, in rope's
    // encode form (kernels/rope/rope.h).
    bool value_norm = false;
    // Set by an architecture whose graph encodes, and by no other
    // (arch/architecture.h): what the planner sizes its working buffers by,
    // and what makes it plan no cache.
    std::optional<EncoderDescription> encoder;
    // Set by a vision encoder's describe and by no other
    // (arch/media_encoder.h).
    std::optional<VisionDescription> vision;
    // Set by an audio encoder's describe and by no other, as `vision` is.
    // `layers` stays empty: a Conformer's layer is not LayerDescription.
    std::optional<AudioDescription> audio;
    // The products whose weights carry their bounds or bias in their binding.
    std::vector<ProductSpan> product_spans;
    // F32 tensors no kernel steps through 32 weights at a time, each read
    // whole by the one kernel that reads it — an audio encoder's
    // convolution weights (kernels/subsample, kernels/light_conv) — so their
    // rows need not be a whole number of groups (formats/format.h).
    std::vector<gguf::TensorId> read_whole;
    std::vector<LayerDescription> layers;
};

}  // namespace bllm::model
