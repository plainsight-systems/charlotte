#pragma once

#include "core/arch/architecture.h"

namespace bllm::arch {

// Axis A: changes with this architecture.
//
// EmbeddingGemma 2's text model, the general.architecture GGUF names
// "gemma-embedding2": a Gemma 4 text stack run as an encoder, every token
// attending to every other, mean-pooled and projected to an embedding. As
// llama.cpp's src/models/gemma-embedding2.cpp builds it, at the embeddings'
// pin (runtime/embedder.h). Its vision and audio encoders are a separate
// file, the mmproj, which this architecture does not read.
//
// describe reads, under "gemma-embedding2.":
//   - What every architecture reads (describe.h): block_count 24,
//     context_length 262,144, embedding_length 512, feed_forward_length
//     2,048, attention.head_count 4, attention.layer_norm_rms_epsilon,
//     rope.freq_base 1,000,000 — and attention.head_count_kv, which this
//     file stores one a layer, 2 for a sliding layer and 1 for a global one.
//   - Which layers slide: attention.sliding_window_pattern, one flag a layer
//     — five sliding then one global, six times — and
//     attention.sliding_window, 1,024: a sliding layer attends 512 tokens
//     either side (kernels/block_attention/block_attention.h); a global
//     layer over its trained context.
//   - Each kind's head: attention.key_length 512 and key_length_swa 256 its
//     head dimension, value_length and value_length_swa equal to them —
//     refused otherwise, as llama.cpp refuses them; rope.dimension_count 512
//     and dimension_count_swa 256 equal to them, every head rotated whole;
//     rope.freq_base_swa 10,000 a sliding layer's base.
//   - What makes it an encoder: attention.causal, false — a file declaring
//     true, or nothing, is refused, since this graph attends both ways;
//     pooling_type 1, mean — any other refused, naming it;
//     embedding_length_per_layer_input 512, the per-layer width; and
//     embedding_length_out 768, the embedding's length.
//   - attention.shared_kv_layers, 0 — a file sharing keys and values across
//     layers is refused, as no block here shares them.
// Its tensors, each checked against the shape its role requires: the token
// embedding; per_layer_model_proj, W × L·P, and per_layer_proj_norm, P;
// output_norm, W; output, W × 768 — the projection to the embedding, not a
// vocabulary head; and in each layer, beside Gemma 3's thirteen roles
// (gemma3.cpp), inp_gate W × P, proj P × W, post_norm W and
// layer_output_scale, one F32.
//
// Conventions: NEOX pairing and GELU's tanh form, as llama.cpp's
// llama_model_rope_type and graph give it; an attention scale of 1, which
// llama.cpp sets because QK-norm takes the scale's place; values normalized
// per head without a gain (model_description.h's value_norm), its graph's
// ggml_rms_norm of Vcur.
//
// Its graph (graph/encoder.h): the embedding scaled by √512 in f32; the
// per-layer inputs; then each layer's attention, gated feed-forward block
// and per-layer block; then the pooled output.
//
// Verification the implementation is held to, on the CPU from ggml-org's
// Q8_0 file's header, pinned and fetched as test data (tests/encoder_test.cpp):
// every layer's heads, window and base, the mask, the value norm and the
// attention scale, the encoder description, every per-layer tensor by role;
// the plan and graph built over it; and preflight taking it to Run. The
// whole, against llama.cpp, is runtime/embedder.h's.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.181  Avoid "naked" unions — a graph's kind is a variant.
//     SL.io.2 When reading, always consider ill-formed input — every key
//            checked, each refusal naming it.
//     E.27   Use error codes systematically — DescribeResult, its subject the
//            key or tensor at fault.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — the architecture and its
//            graph's kind chosen once, at load; a step calls neither.

extern const Architecture kEmbeddingGemma2;

}  // namespace bllm::arch
