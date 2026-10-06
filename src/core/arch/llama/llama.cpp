#include "core/arch/llama/llama.h"

#include <array>

#include "core/arch/describe.h"

namespace bllm::arch {
namespace {

using model::Role;

// The plain transformer layer. Llama 3's long-context rotary scaling is the
// file's rope_freqs.weight, which describe_layers finds; it is not a layer
// weight.
constexpr std::array kRoles{
    RoleName{Role::AttentionNorm, "attn_norm.weight"},
    RoleName{Role::Query, "attn_q.weight"},
    RoleName{Role::Key, "attn_k.weight"},
    RoleName{Role::Value, "attn_v.weight"},
    RoleName{Role::AttentionOutput, "attn_output.weight"},
    RoleName{Role::FeedForwardNorm, "ffn_norm.weight"},
    RoleName{Role::Gate, "ffn_gate.weight"},
    RoleName{Role::Up, "ffn_up.weight"},
    RoleName{Role::Down, "ffn_down.weight"},
};

DescribeResult describe(const gguf::TensorIndex& index, model::ModelDescription& out) {
    Hyperparameters hp{};
    if (auto r = read_hyperparameters(index, "llama", hp); !r.ok()) return r;
    // NORM pairing, as llama.cpp's llama_model_rope_type gives Llama: its
    // converter permuted Q and K into that order.
    return describe_layers(index, hp, kRoles, model::RotaryPairing::Adjacent, out);
}

}  // namespace

const Architecture kLlama{"llama", describe};

}  // namespace bllm::arch
