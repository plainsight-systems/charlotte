# EmbeddingGemma 2's vision encoder: what it is, what llama.cpp runs, what it costs

**Date:** 2026-10-11
**Method:** the files' headers and tensor indexes read by HTTP range request;
Google's `config.json`, `processor_config.json` and model card; llama.cpp read
from source at the embeddings' pin, `a9d27ac693ff0a1d3cba449c912c2b891904a4c7`
(`runtime/embedder.h`). Nothing here was run. The costs are derived from
shapes, and where a rate is used it is a measured one, named.

## Why

EmbeddingGemma 2 maps text, images, video and audio into one 768-dimensional
space, and that is what its card leads with. Charlotte runs its text model.
This note is what the image path needs: where the vision encoder lives, what
it computes, how llama.cpp prepares and runs an image, and what it would cost
here.

## The files

Google ships one checkpoint, `google/embeddinggemma-2`: a single
`model.safetensors` holding a 270M-parameter text model and 170M vision and
300M audio encoders. ggml-org's conversion (`ggml-org/embeddinggemma-2-GGUF`)
splits it as llama.cpp loads it: the text model in one file, which Charlotte
reads, and both encoders in a second, the mmproj, which llama.cpp's
multimodal library (`tools/mtmd`) loads beside it.

| File | Size |
|---|---|
| `embeddinggemma-2-Q8_0.gguf`, the text model | 309,855,456 bytes |
| `mmproj-embeddinggemma-2-Q8_0.gguf`, both encoders | 554,821,024 bytes |
| `mmproj-embeddinggemma-2-BF16.gguf` | 982,074,784 bytes |

The Q8_0 mmproj, by its index (`general.architecture` `clip`, `general.type`
`mmproj`, `clip.vision.projector_type` `gemma4v`, `clip.audio.projector_type`
`gemma4a`):

| Bytes | What |
|---|---|
| 0 .. 58,656 | header and tensor index; data aligned to 32 |
| 58,656 .. 328,492,448 | the audio encoder: 752 tensors, 328.4 MB |
| 328,492,448 .. 554,821,024 | the vision encoder and its projection: 211 tensors, 226.3 MB, to the end of the file |

So the vision encoder is one contiguous span. A page that wants images alone
needs the header and that span, 226.4 MB of the 554.8.

## The vision encoder

From the file's keys and tensors, Google's `vision_config`, and llama.cpp's
graph (`tools/mtmd/models/gemma4v.cpp`, `clip.cpp`'s `build_vit`):

- **Patches.** The image, resized (below), is cut into 16 × 16 patches, each
  a row of 768 values, channel then row then column, scaled from [0, 1] to
  [−1, 1]. `v.patch_embd.weight`, F32, 768 × 768: a 16 × 16 convolution of
  stride 16 with no bias, which is a matrix product over those rows.
- **Positions, twice.** A learned table, `v.position_embd.weight`, F32,
  768 × 10,240 × 2, 62.9 MB: patch (x, y) adds row x of the first half and
  row y of the second. Then, in every layer, 2-D rotary positions ("axial"
  in Google's config, θ 100): the first half of each 64-wide head rotated by
  x, the second half by y, each half NEOX-paired within itself. θ is not in
  the file; llama.cpp sets 100 for `gemma4v`, as Google's config states.
- **16 layers, width 768**, 12 heads of 64, no grouping; RMS norms, ε 10⁻⁶;
  per layer: a pre-norm, Q, K and V (Q8_0, 768 × 768 each), QK-norm with
  64-wide gains, values normalized per head without a gain, attention in
  both directions over every patch with scale 1, the output projection, a
  post-attention norm and the residual add; then a pre-norm, a gated
  feed-forward of 3,072 (gate and up, Q8_0), down, a post-feed-forward norm
  and the add. No biases, no layer scales. The same block as the text
  model's, at other sizes, without the text model's per-layer inputs.
- **Pooling.** The patch grid averaged 3 × 3 (`pooling_kernel_size`), so a
  grid of c × r patches becomes c/3 × r/3 rows; each scaled by √768, RMS
  normalized without a gain, then projected by `mm.input_projection.weight`,
  Q8_0, 768 to 512: the text model's width. Google's config has
  `use_clipped_linears` false and the file has no clamp values for the
  vision side, so the projection is a plain product.
- **Into the text model.** Those rows are soft tokens. The text, with
  `<|image|>` where the image goes, is tokenized; llama.cpp replaces the
  placeholder with `<|image>`, the image's rows, `<image|>`. The rows enter
  the text model's input as they are: llama.cpp scales token rows by √512
  and media rows by 1 (`gemma-embedding2.cpp`, "do not normalize weights for
  raw embeddings input"). The per-layer inputs are projected from the input
  rows, token or media alike, since this architecture has no per-layer token
  table. Then the text encoder runs as it does for text: every row attends
  to every other, mean pooled over all rows, projected to 768, unit length.

## How many tokens an image is

The card's default is 280 soft tokens an image, configurable from 70 to
1,120. Google's processor configuration gives `max_soft_tokens` 280 and
`pooling_kernel_size` 3 (`processor_config.json`): at most 280 rows, from
at most 2,520 patches.

llama.cpp's dynamic-size preprocessor at the pin
(`mtmd_image_preprocessor_dyn_size`):

1. The target size by `calc_size_preserved_ratio`: each side rounded to a
   multiple of 48 (16 × 3); if the area exceeds the maximum pixels, both
   sides scaled by √(area / max) and floored to 48; if under the minimum,
   scaled up and ceiled. The limits are tokens × 2,304 pixels: 70 and 1,120
   tokens by default for `gemma4v`, set with `--image-min-tokens` and
   `--image-max-tokens`.
2. A Pillow-compatible separable resampler (`resize_pillow`, adapted from
   Pillow's `Resample.c`), bicubic with a = −0.5, its support widened by the
   scale when shrinking, so it filters rather than aliases: weights
   precomputed per output column and row, in 22-bit fixed point; a
   horizontal pass, then a vertical one, each rounding to 8 bits and
   clipping. Integer arithmetic throughout, so the same input gives the same
   bytes anywhere.
3. Bytes to floats, x / 255; the graph then takes 2x − 1.

llama.cpp's default maximum, 1,120 tokens, is four times the card's. A
reference made to match Google's processor sets `--image-max-tokens 280`.
The image is decoded by stb_image, which ignores EXIF orientation and ICC
profiles and drops alpha.

## One discrepancy: the feed-forward's activation

Google's `vision_config.hidden_activation` is `gelu_pytorch_tanh`. llama.cpp
chooses a vision encoder's feed-forward activation from the file's
`clip.use_gelu` and `clip.use_silu`, and with neither set falls back to
`FFN_GELU_QUICK`, x · σ(1.702 x) (`clip.cpp`, `load_hparams`). Its converter
writes `clip.use_gelu` for Gemma 3's vision encoder (`conversion/gemma.py`,
`Gemma3VisionModel`) but not for Gemma 4's (`Gemma4VisionAudioModel`), and
this file has neither key; `gemma4v`'s own case in `load_hparams` does not set
the activation, and `gemma4v.cpp` passes `hparams.ffn_op` to `build_vit`.

So by reading the code, llama.cpp at the pin runs this vision encoder's
feed-forward with GELU-quick where the model was trained with GELU's tanh
form. The two differ by up to 0.021 (at x ≈ −2.29), about 1 to 2% of the
activation over the range that matters, in every one of 16 layers: far
outside the spread between llama.cpp's own backends that Charlotte's
reference tests take as their tolerance. This is from reading, not running;
the reference tool confirms it when it is built, by embedding one image both
ways.

What follows for Charlotte:

- Its vision encoder computes Google's activation, GELU's tanh form, which
  its matrix product's gated epilogue already has.
- Its oracle is llama.cpp run on a copy of the mmproj whose header adds
  `clip.use_gelu = true`: the same weights, Google's activation, so the
  comparison is still against llama.cpp's own two backends' spread.
- Once, while the reference tool is built, the corrected llama.cpp's
  embedding of an image is checked against Google's transformers
  implementation of the same image, to confirm that the correction is the
  only one; that figure belongs in this note when it exists.

## llama.cpp as the oracle for an image and text together

llama.cpp's server embeds a prompt holding media with every row in one
batch: tokens and the image's rows are added to the same batch
(`server-context.cpp`, `add_prompt_mixed`, `add_mtmd_chunk`), and
`gemma-embedding2` supports mixed batches (`llm_arch_supports_mixed_batch`).
Its graph has no cache (`build_attn_inp_no_cache`), so one batch is what lets
the image's rows and the text's tokens attend to each other. A reference
made through the server is therefore the computation Charlotte's one step
makes.

## What an image costs here, derived

For 280 tokens, 2,520 patches, and the text step that follows of about 284
rows (`<|image>`, the rows, `<image|>` and the ends):

| | Multiply-adds |
|---|---|
| Patch embedding, 2,520 × 768 × 768 | 1.5 × 10⁹ |
| 16 layers' products, 9.44 × 10⁶ a patch a layer | 3.81 × 10¹¹ |
| 16 layers' attention, 2 × 12 × 2,520² × 64 a layer | 1.56 × 10¹¹ |
| Pooling and projection, 280 × 768 × 512 | 1.1 × 10⁸ |
| **Vision encoder** | **5.39 × 10¹¹**, 1.08 × 10¹² operations |

- At the 2.9 TFLOPS Qwen3's 951-token prefill reached on the target
  (`2026-10-09-webkit-shader-compiler.md`), the vision encoder would take
  about 0.37 s; its attention, 29% of its work, runs on a kernel of its own
  whose rate is not yet known, so 0.4 to 0.5 s an image is the estimate. The
  text step after it is measured: 26.0 ms at 254 tokens and 33.5 ms at 336
  (`2026-10-11-chrome-encode-benchmark.md`), so about 29 ms.
- Attention's scores at 2,520 patches and 12 heads are 305 MB a layer as
  f32: more than WebGPU's default 128 MiB binding and 256 MiB buffer, so the
  text encoder's form, which writes its scores, cannot run here. At a head
  dimension of 64 the streaming form that form's header weighed and set aside
  at 512 fits the default 16 KiB of workgroup memory.
- On the CPU, the resampler: a 4,032 × 3,024 photo to 912 × 672, about 280
  tokens, filters with 19 taps a side: about 1.6 × 10⁸ integer multiply-adds
  for the horizontal pass over every source row and 3.5 × 10⁷ for the
  vertical, about 0.1 to 0.2 s in the module, by estimate.
- Memory: the vision span, 226 MB of weights, and its working buffers at
  2,520 rows, about 93 MB, of which the feed-forward's 31 MB is the largest:
  about 320 MB beside the text model's 384.
- Download: 226.4 MB for the header and the vision span, against 554.8 MB
  for the whole mmproj.

## What it says

- **The image path is the text path with an encoder in front of it.** The
  vision encoder's output is 280 rows of the text model's width, placed in
  the text step between two tokens; everything after is the encode path that
  exists.
- **The vision encoder's layer is the text encoder's block at other sizes**,
  with three things the text encoder has no form of: rotary positions in two
  dimensions, a learned position table indexed by column and row, and 3 × 3
  pooling over the patch grid.
- **One step of 2,520 rows needs attention that streams.** Its scores do
  not fit a binding; at d = 64 the online softmax fits the workgroup memory
  WebGPU guarantees.
- **The preprocessing can be matched exactly.** llama.cpp's resampler is
  integer arithmetic, so a port can be held to its bytes, not a tolerance.
- **The oracle needs one correction**, the feed-forward's activation, which
  llama.cpp at the pin takes from a key this file lacks.
- **Download what the modality needs.** The vision encoder is the last
  226 MB of the file; the audio encoder before it is 328 MB an image search
  never reads.
