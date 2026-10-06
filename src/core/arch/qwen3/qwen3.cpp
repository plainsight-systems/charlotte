#include "core/arch/qwen3/qwen3.h"

#include <array>

#include "core/arch/describe.h"

namespace bllm::arch {
namespace {

using model::Role;

// Qwen3 normalises each query and key head before attention, so its layers
// carry two weights a plain transformer layer does not.
constexpr std::array kRoles{
    RoleName{Role::AttentionNorm, "attn_norm.weight"},
    RoleName{Role::Query, "attn_q.weight"},
    RoleName{Role::Key, "attn_k.weight"},
    RoleName{Role::Value, "attn_v.weight"},
    RoleName{Role::QueryNorm, "attn_q_norm.weight"},
    RoleName{Role::KeyNorm, "attn_k_norm.weight"},
    RoleName{Role::AttentionOutput, "attn_output.weight"},
    RoleName{Role::FeedForwardNorm, "ffn_norm.weight"},
    RoleName{Role::Gate, "ffn_gate.weight"},
    RoleName{Role::Up, "ffn_up.weight"},
    RoleName{Role::Down, "ffn_down.weight"},
};

DescribeResult describe(const gguf::TensorIndex& index, model::ModelDescription& out) {
    Hyperparameters hp{};
    if (auto r = read_hyperparameters(index, "qwen3", hp); !r.ok()) return r;
    // NEOX pairing, as llama.cpp's llama_model_rope_type gives Qwen3.
    return describe_layers(index, hp, kRoles, model::RotaryPairing::Halves, out);
}

}  // namespace

const Architecture kQwen3{"qwen3", describe};

}  // namespace bllm::arch
