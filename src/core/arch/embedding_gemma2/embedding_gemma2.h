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
// What it asks of the other contracts:
//   - arch/architecture.h: an architecture's graph is one of two kinds,
//     `std::variant<GraphFn, EncodeFn>` — a decoder's, given the cache
//     format, or an encoder's, EncodeFn(const ModelDescription&, const
//     ResidencyPlan&, std::vector<Launch>&) -> GraphResult, which has no
//     cache. The kind is the architecture's, fixed in the capability table,
//     so it cannot disagree with the graph that runs; preflight and load
//     read it once, at load, and choose a generator or an embedder
//     (runtime/embedder.h). A variant, not two pointers one of which is
//     null, so no entry can hold both or neither (C.181).
//   - model/model_description.h:
//       - `AttentionMask mask`: Causal, every decoder's, or Bidirectional;
//         for Bidirectional a layer's attention_window W means keys within
//         W / 2 either side, as llama.cpp's symmetric window.
//       - `bool value_norm`: values normalized per head without a gain, in
//         rope's encode form.
//       - Four roles: PerLayerGate, PerLayerProjection, PerLayerPostNorm and
//         OutputScale — a change to the role vocabulary, which every
//         architecture's describe leaves empty but this one.
//       - `std::optional<EncoderDescription> encoder`, set by an encoding
//         architecture and by no other: the per-layer projection and its
//         norm, the per-layer width, the output projection, the embedding's
//         length, and the pooling, Mean. What the planner sizes an
//         encoder's working buffers by, and what makes it plan no cache
//         (graph/encoder.h's asks of residency/plan.h). An architecture
//         whose graph is an EncodeFn always sets it, and a GraphFn one never
//         does — each architecture's describe test holds it to that.
//   - arch/describe.h: read_hyperparameters reads attention.head_count_kv
//     as one number or one a layer, as llama.cpp's get_key_or_arr does, and
//     describe_layers takes the per-layer form; the other listed files
//     store one number and describe as before.
//   - capability/capability.h: one row, "gemma-embedding2".
//
// Verification the implementation is held to:
//   - On the CPU, from ggml-org's Q8_0 file's header, fetched as test data:
//     every number above, each layer's head shape, window and base, the
//     encoder description, every tensor by role; and each refusal named —
//     attention.causal true, pooling_type 2, a value length differing from
//     its key length, shared_kv_layers 1, a per-layer tensor missing —
//     over headers tools/make_fixture_gguf.py writes with that one change.
//   - The graph: graph/encoder.h's verification.
//   - The whole: runtime/embedder.h's reference against llama.cpp.
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
