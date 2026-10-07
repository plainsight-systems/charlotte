#include "core/arch/gemma3/gemma3.h"

#include <array>
#include <cmath>

#include "core/arch/describe.h"

namespace bllm::arch {
namespace {

using model::Role;

// Gemma 3 normalises query and key heads, and normalises again after
// attention and after the feed-forward block.
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

// Defaults llama.cpp applies when the file does not say: in every run of six
// layers, the first five attend over a sliding window and the sixth over the
// whole context; window layers rotate with a base of 10,000.
constexpr std::uint32_t kDefaultPattern = 6;
constexpr float kDefaultWindowRopeBase = 10000.0f;

// Sliding-window layers attend over the window, with their own rotary base.
DescribeResult apply_sliding_window(const gguf::TensorIndex& index, model::ModelDescription& out) {
    std::uint32_t window = 0;
    if (auto r = read_u32_or(index, "gemma3", "attention.sliding_window", 0, window); !r.ok()) return r;
    if (window == 0) return {};

    // The pattern may be stored as one number or as a flag per layer; the
    // per-layer form is valid and not one this reads.
    gguf::ArrayLocation per_layer{};
    if (index.read_array("gemma3.attention.sliding_window_pattern", per_layer) == gguf::MetadataError::Ok) {
        return DescribeResult{DescribeError::UnsupportedValue, "gemma3.attention.sliding_window_pattern"};
    }
    std::uint32_t pattern = 0;
    if (auto r = read_u32_or(index, "gemma3", "attention.sliding_window_pattern", kDefaultPattern, pattern);
        !r.ok()) {
        return r;
    }
    if (pattern == 0) return DescribeResult{DescribeError::InvalidValue, "gemma3.attention.sliding_window_pattern"};
    float window_rope_base = 0;
    if (auto r = read_f32_or(index, "gemma3", "rope.freq_base_swa", kDefaultWindowRopeBase, window_rope_base);
        !r.ok()) {
        return r;
    }

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        if (layer % pattern == pattern - 1) continue;   // the global layer of each run
        out.layers[layer].attention_window = window;
        out.layers[layer].rope_base = window_rope_base;
    }
    return {};
}

// Gemma 3 27B scales attention by 1 / sqrt(embedding width / query heads),
// not by the head dimension. No GGUF key states it; llama.cpp infers the
// 27B from its 62 layers, after Google's gemma_pytorch configuration, and so
// does this.
constexpr std::uint32_t k27BLayers = 62;

DescribeResult describe(const gguf::TensorIndex& index, model::ModelDescription& out) {
    Hyperparameters hp{};
    if (auto r = read_hyperparameters(index, "gemma3", hp); !r.ok()) return r;
    // NEOX pairing, as llama.cpp's llama_model_rope_type gives Gemma 3, and
    // GeGLU with GELU's tanh form, its graph's feed-forward and Hugging
    // Face's gelu_pytorch_tanh.
    if (auto r = describe_layers(index, hp, kRoles,
                                 {model::RotaryPairing::Halves, model::FeedForwardActivation::GeluTanh}, out);
        !r.ok()) {
        return r;
    }
    if (hp.block_count == k27BLayers) {
        out.attention_scale = 1.0f / std::sqrt(static_cast<float>(hp.embedding_length / hp.head_count));
    }
    return apply_sliding_window(index, out);
}

// Each layer attention then the gated feed-forward block, each block's
// output through its post-norm before it is added (graph/graph.h), and the
// embedding scaled by the square root of its width in f32, as llama.cpp's
// build_inp_embd(tok_embd, sqrtf(n_embd)) does for Gemma 3.
graph::GraphResult graph(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                         const formats::Format& cache_format, std::vector<kernels::Launch>& out) {
    graph::Builder b(model, plan, cache_format, out);
    b.embed(std::sqrt(static_cast<float>(model.embedding_width)));
    for (std::uint32_t layer = 0; layer < model.layers.size(); ++layer) {
        if (auto r = b.attention(layer); !r.ok()) return r;
        if (auto r = b.gated_feed_forward(layer); !r.ok()) return r;
    }
    return b.output();
}

}  // namespace

const Architecture kGemma3{"gemma3", describe, graph};

}  // namespace bllm::arch
