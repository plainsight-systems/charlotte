#pragma once

#include "core/arch/media_encoder.h"

namespace bllm::arch {

// Axis A: changes with this vision encoder.
//
// Gemma 4's vision encoder, the clip.vision.projector_type GGUF names
// "gemma4v": EmbeddingGemma 2's, in its mmproj (docs/research/
// 2026-10-11-embeddinggemma2-vision.md). A 16-layer transformer over an
// image's 16 × 16 patches, every patch attending to every other, pooled 3 × 3
// and projected to the text model's width. As llama.cpp's
// tools/mtmd/models/gemma4v.cpp builds it at the embeddings' pin
// (runtime/embedder.h), with one exception, the feed-forward's activation,
// below.
//
// describe reads, under "clip.":
//   - general.architecture "clip" and general.type "mmproj", each refused
//     otherwise; has_vision_encoder true. The audio encoder the same file
//     holds is not this encoder's and is not read: its keys are ignored and
//     its tensors are not in the vision span the load reads
//     (residency/file_ranges.h).
//   - vision.block_count 16, vision.embedding_length 768,
//     vision.feed_forward_length 3,072, vision.attention.head_count 12,
//     vision.attention.layer_norm_epsilon, vision.patch_size 16,
//     vision.projection_dim 512. vision.attention.head_count_kv and
//     head_dim, where present, must give 12 and 64 — the width over the
//     heads — and are refused by name otherwise, as no block here groups
//     the vision encoder's heads.
//   - vision.projector.scale_factor, the pooling, 3 where absent, as
//     llama.cpp defaults it for gemma4v. vision.image_size is not read: the
//     size is the budget's (media/image.h).
//   - vision.image_mean all 0 and vision.image_std all 1, as this file's are:
//     the patches are x / 255 scaled to [−1, 1] and nothing else
//     (media/image.h); any other value is refused, naming the key, since no
//     step here would apply it.
//   - The activation. Google's config gives gelu_pytorch_tanh, GELU's tanh
//     form. llama.cpp chooses from clip.use_gelu and clip.use_silu and,
//     with neither, as in this file, runs GELU-quick — a difference of up
//     to 0.021 an activation, which the research note traces. So: GeluTanh
//     where neither key is set or use_gelu is true; use_silu true is
//     refused, naming it, since it would say the file is not this encoder.
//     The reference tool sets clip.use_gelu on its copy of the header so
//     llama.cpp computes the activation the model was trained with
//     (runtime/vision.h).
// Numbers the file does not carry, each the encoder's convention, as NEOX
// pairing is a text architecture's (logical-overview.md, principle 3, as
// Gemma 3 27B's attention scale is): the rotary base θ, 100, which
// llama.cpp sets for gemma4v and Google's config states, rope_type axial;
// an attention scale of 1, QK-norm taking its place.
//
// Its tensors, each checked against the shape its role requires, with W =
// 768, P = 16, F the feed-forward width, L the layers:
//   - v.patch_embd.weight, F32, (P, P, 3, W): the patch embedding, a matrix
//     of W rows of 3 P² — the convolution as a product (graph/vision.h);
//   - v.position_embd.weight, F32, (W, positions, 2): the column table, then
//     the row table, each `positions` rows of W, 10,240 in this file;
//   - mm.input_projection.weight, (W, 512): the projection to the text
//     model's width;
//   - in each layer v.blk.l.: ln1 (AttentionNorm), attn_q, attn_k, attn_v
//     (Query, Key, Value, W × W), attn_q_norm and attn_k_norm (QueryNorm,
//     KeyNorm, 64), attn_out (AttentionOutput), attn_post_norm
//     (PostAttentionNorm), ln2 (FeedForwardNorm), ffn_gate, ffn_up (W × F),
//     ffn_down (F × W), ffn_post_norm (PostFeedForwardNorm).
// A tensor the graph has no block for is refused by name where gemma4v.cpp
// would apply it: v.patch_embd.bias, v.std_bias and v.std_scale, a layer's
// biases or layer scales, and clamp values on mm.input_projection (Google's
// use_clipped_linears is false; a file with them would need a clamped
// product). Principle 5: structure detected, never assumed.
//
// The description it fills (model_description.h): each layer's heads, head
// dimension 64, feed-forward width, rotary base 100, its tensors by role,
// and an attention window wide enough that every patch reads every other;
// RotaryPairing::Axial; GeluTanh; value_norm; mask Bidirectional; attention
// scale 1; the file's ε; no token embedding, no output norm and no
// vocabulary; and its vision description: the patch embedding and size, the
// position table and its length, the pooling, the projection and its width.
//
// pairs: the text model reads these rows when its architecture is
// "gemma-embedding2" and its hidden width is the projection's, 512.
// EmbeddingGemma 2's own mmproj is the only pairing listed; another text
// model, or another width, is refused by name.
//
// Its graph (graph/vision.h): the patch embedding; the positions; then each
// layer's attention and gated feed-forward block; then the pooled
// projection.
//
// What it asks of the other contracts:
//   - capability/capability.h: find_media_encoder(MediaKind::Image,
//     "gemma4v") returning it; preflight turns null into a blocker naming
//     the projector type, as it names an architecture.
// Its tokens: `<|image|>`, `<|image>` and `<image|>`, llama.cpp's img_beg
// and img_end for gemma4v. No setup.
//   - model/model_description.h: a vision description,
//       struct VisionDescription { TensorId patch_embedding;
//         uint32_t patch_size; TensorId position_table; uint32_t positions;
//         uint32_t pooling; TensorId projection; uint32_t projection_width; };
//     as std::optional<VisionDescription> vision, set by a vision encoder's
//     describe and no other, as `encoder` is by an embedder's; token
//     embedding and output norm become optional, absent only where `vision`
//     is set — the decoder's and encoder's builders keep requiring them; a
//     rotary pairing, Axial: each half of a head NEOX-paired within itself,
//     the first half turned by the patch's column and the second by its
//     row (kernels/rope/rope.h).
//   - arch/describe.h: its key and tensor naming taken with a prefix —
//     "clip.vision." and "v.blk." — beside the text architectures'
//     "<arch>." and "blk.", so the shape checks by role are shared.
//
// Verification the implementation is held to, on the CPU from ggml-org's
// Q8_0 mmproj's header, pinned and fetched as test data
// (tests/vision_test.cpp): every layer's heads, base, window and tensors by
// role; the vision description; each refusal above over the header with the
// key changed or the tensor renamed; pairs with EmbeddingGemma 2's text
// header and refusing Qwen3's; the plan and graph built over it; preflight
// taking it to Run. The whole, against llama.cpp, is runtime/vision.h's.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     SL.io.2 When reading, always consider ill-formed input — every key and
//            tensor checked, each refusal naming it.
//     E.27   Use error codes systematically — DescribeResult, its subject the
//            key or tensor at fault.
//     I.4    Make interfaces precisely and strongly typed — roles, not names,
//            past describe.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — chosen once, at load.

extern const MediaEncoder kGemma4Vision;

}  // namespace bllm::arch
