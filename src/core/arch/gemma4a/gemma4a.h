#pragma once

#include "core/arch/media_encoder.h"

namespace bllm::arch {

// Axis A: changes with this audio encoder.
//
// Gemma 4's audio encoder, the clip.audio.projector_type GGUF names
// "gemma4a": EmbeddingGemma 2's, in its mmproj (docs/research/
// 2026-10-11-embeddinggemma2-audio.md). A Conformer: two strided
// convolutions over a clip's log-mel spectrogram, 12 layers of a half-step
// feed-forward, local attention over each row's 12 latest rows with a
// relative position term, a convolution module and a second half-step
// feed-forward, then a projection to the text model's width. As llama.cpp's
// tools/mtmd/models/gemma4a.cpp builds it at the embeddings' pin
// (runtime/embedder.h); the one difference from llama.cpp is in preparing
// the spectrogram, not in the encoder (media/audio.h).
//
// describe reads, under "clip.":
//   - general.architecture "clip" and general.type "mmproj", each refused
//     otherwise; has_audio_encoder true. The vision encoder the same file
//     holds is not this encoder's and is not read: its keys are ignored and
//     its tensors are not in the audio span the load reads
//     (residency/file_ranges.h).
//   - audio.block_count 12, audio.embedding_length 1,024,
//     audio.feed_forward_length 4,096, audio.attention.head_count 8 — a head
//     of 128 — audio.num_mel_bins 128, which must be media/audio.h's
//     kMelBins and is refused by name otherwise, audio.projection_dim 512,
//     audio.attention.layer_norm_epsilon, 10⁻⁶ in this file.
// Numbers the file does not carry, each the encoder's convention, written in
// Google's audio_config and hard-coded in llama.cpp's graph, which agree on
// every one (the research note): attention chunks of 12 rows, each row
// attending to itself and the 11 before it — attention_context_left 13, its
// window 12 — and none after; relative positions 0 to 12, sinusoidal with a
// timescale of 10,000; Q scaled by (1 / √128) / ln 2 and K by
// ln(1 + e) / ln 2; scores soft-capped at 50; a residual weight of 0.5 on
// each half-step feed-forward; SiLU; the subsampling's convolutions 3 × 3 of
// stride 2 and padding 1, each followed by a layer norm over its channels and
// ReLU. The depthwise convolution's kernel, 5, is its tensor's width.
//
// Its tensors, each checked against the shape its role requires, with W =
// 1,024, F = 4,096, M = 128 mel bins, L = 12:
//   - a.conv1d.0.weight, F32, (3, 3, 1, 128), and a.conv1d.0.norm.weight,
//     F32, 128: the first convolution and its layer norm's gain;
//   - a.conv1d.1.weight, F32, (3, 3, 128, 32), and a.conv1d.1.norm.weight,
//     F32, 32: the second;
//   - a.input_projection.weight, F32, (32 × M / 4, W): the flattened
//     subsampling to the encoder's width;
//   - in each layer a.blk.l.:
//       ffn_norm, ffn_up (W, F), ffn_down (F, W), ffn_post_norm — the first
//         half-step feed-forward;
//       attn_pre_norm, attn_q, attn_k, attn_v, attn_out (W, W),
//         per_dim_scale (128, F32, softplus already applied by the
//         converter), attn_k_rel (W, W), attn_post_norm — the attention;
//       conv_norm — the convolution module's pre-norm, despite its name —
//         conv_pw1 (W, 2W), conv_dw (5, W, F32), norm_conv — the norm after
//         the depthwise convolution — conv_pw2 (W, W);
//       ffn_norm_1, ffn_up_1, ffn_down_1, ffn_post_norm_1 — the second;
//       ln2 — the layer's output norm;
//     and for each of its ten products — ffn_up, ffn_down, attn_q, attn_k,
//     attn_v, attn_out, conv_pw1, conv_pw2, ffn_up_1, ffn_down_1 — its
//     clamp bounds: <product>.input_min, input_max, output_min, output_max,
//     F32, one element each. attn_k_rel has none, nor do the subsampling's
//     or the output's products.
//   - a.pre_encode.out.weight (W, 1,536) and a.pre_encode.out.bias (1,536,
//     F32): the encoder's output projection;
//   - mm.a.input_projection.weight (1,536, 512): the multimodal projection,
//     after an RMS norm without a gain.
// A tensor the graph has no block for is refused by name where gemma4a.cpp
// would apply it: a.conv1d.N.bias, mm.a.soft_emb_norm.weight, a layer's
// per_dim_k_scale or conv_dw.bias, any product's bias but the output
// projection's, a product with some but not all four bounds. Principle 5:
// structure detected, never assumed.
//
// The description it fills (model_description.h): no layer in `layers` —
// the Conformer's blocks are not the transformer's — no token embedding, no
// output norm, no vocabulary, the file's ε; its AudioDescription, each
// layer's tensors by role and each product with its bounds; and a product
// span for every product with bounds, and for the output projection's
// weight and bias (residency/plan.h).
//
// pairs: the text model reads these rows when its architecture is
// "gemma-embedding2" and its hidden width is the projection's, 512, as
// gemma4v's pairs.
//
// Its graph (graph/audio.h): the subsampling; each layer's five blocks; the
// output. Its setup: each layer's relative position table, attn_k_rel times
// the 13 sinusoidal positions, once at load.
//
// What it asks of the other contracts:
//   - capability/capability.h: find_media_encoder(MediaKind::Audio,
//     "gemma4a") returning it; the table keyed by kind and type.
//   - model/model_description.h: the audio description above.
//   - residency/plan.h: each product with bounds a product span — its
//     weight and four bounds kept in one buffer within one binding — and the
//     output projection's weight and bias the same (kernels/matmul/matmul.h).
//   - arch/describe.h: key and tensor naming with the prefixes
//     "clip.audio." and "a.blk.", as gemma4v takes "clip.vision." and
//     "v.blk.".
//
// Verification the implementation is held to, on the CPU from ggml-org's
// Q8_0 mmproj's header, pinned and fetched as test data
// (tests/audio_test.cpp): every layer's tensors by role and its products'
// bounds; the audio description; each refusal above over the header with the
// key changed or the tensor renamed or dropped; pairs with EmbeddingGemma
// 2's text header and refusing Qwen3's; the plan over the audio range, every
// product span in one binding; the setup and step graphs built over it;
// preflight taking it to Run beside the text model, and beside the text
// model and the vision encoder. The whole, against llama.cpp, is
// runtime/audio.h's.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     SL.io.2 When reading, always consider ill-formed input — every key and
//            tensor checked, each refusal naming it.
//     E.27   Use error codes systematically — DescribeResult, its subject the
//            key or tensor at fault.
//     I.4    Make interfaces precisely and strongly typed — roles, not names,
//            past describe; the conformer's layer its own struct.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — chosen once, at load.

// Its describe and pairs, which kGemma4Audio composes with its graphs.
[[nodiscard]] DescribeResult describe_gemma4a(const gguf::TensorIndex& index, model::ModelDescription& out);
[[nodiscard]] DescribeResult pairs_gemma4a(const model::ModelDescription& audio, const model::ModelDescription& text,
                                           std::string_view text_architecture);

extern const MediaEncoder kGemma4Audio;

}  // namespace bllm::arch
