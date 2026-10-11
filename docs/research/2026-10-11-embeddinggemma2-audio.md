# EmbeddingGemma 2's audio encoder: what it is, what llama.cpp runs, what it costs

**Date:** 2026-10-11
**Method:** the mmproj's header and tensor index read from the pinned file;
F32 tensors read from its data; Google's `config.json`,
`processor_config.json` and model card; Hugging Face's Gemma 4 feature
extractor, processor and audio model read from source; llama.cpp read from
source at the embeddings' pin, `a9d27ac693ff0a1d3cba449c912c2b891904a4c7`,
and run on the target (Apple M3 Max, Metal) through a probe built against it
and against a copy changed in one line (below). Costs are derived from
shapes; the rates used are measured ones, named.

## Why

Images are in (`2026-10-11-embeddinggemma2-vision.md`). The card's other
modalities are audio and video, and EmbeddingGemma 2 isn't multimodal here
until it takes them. This note is what the audio path needs: where the audio
encoder lives, what it computes, how llama.cpp and Google prepare a waveform,
where they disagree, and what a clip costs.

## The file

The Q8_0 mmproj's audio half, by its index (`clip.has_audio_encoder` true,
`clip.audio.projector_type` `gemma4a`):

| Bytes | What |
|---|---|
| 0 .. 58,656 | header and tensor index |
| 58,656 .. 328,492,448 | the audio encoder and its projection: 752 tensors, 328.4 MB |

The audio span starts where the header ends, so the header and the audio
encoder are one range, `[0, 328,492,448)`: 328.5 MB of the file's 554.8.

Keys: `clip.audio.embedding_length` 1,024, `feed_forward_length` 4,096,
`block_count` 12, `attention.head_count` 8, `num_mel_bins` 128,
`projection_dim` 512, `attention.layer_norm_epsilon` 10⁻⁶. The file has no
key for the chunked attention's geometry, the softcap, the residual weight or
the convolution's kernel; llama.cpp hard-codes them, and each matches
Google's `audio_config` (below).

## The audio encoder

A Conformer, from the tensors, Google's `audio_config` and llama.cpp's graph
(`tools/mtmd/models/gemma4a.cpp`):

- **Subsampling.** The log-mel spectrogram, 128 bins a 10 ms frame, is an
  image of frames × bins. Two 3 × 3 convolutions of stride 2 and padding 1,
  F32 and without bias (`a.conv1d.0`, 1 to 128 channels; `a.conv1d.1`, 128 to
  32), each followed by a layer norm over the channels (mean-centred, a gain,
  no bias, ε 10⁻⁶) and ReLU. Frames and bins each shrink by 4: T frames
  become R = ⌈⌈T / 2⌉ / 2⌉ rows of 32 bins × 32 channels, flattened bin then
  channel to 1,024, then `a.input_projection`, F32, 1,024 to 1,024.
- **12 layers, width 1,024**, 8 heads of 128. Each is, in order:
  1. A half-step feed-forward: RMS norm, up (1,024 to 4,096), SiLU, down,
     a post-norm, added to the residual times 0.5. Not gated.
  2. Chunked local self-attention: RMS norm; Q, K and V (1,024 each, no
     bias); Q scaled by (1 / √128) / ln 2 and per dimension by
     `per_dim_scale`, K by ln(1 + e) / ln 2; each query attends to itself
     and the 11 rows before it, nothing after (`attention_context_left` 13,
     so 12 positions, `context_right` 0); a relative position term, Q
     against a projection (`attn_k_rel`) of 13 sinusoidal positions,
     distances 12 down to 0; scores soft-capped at 50 (50 tanh(s / 50));
     softmax; the output projection; a post-norm; added to the residual.
  3. A convolution module: RMS norm; a pointwise projection to 2,048; a GLU,
     the first half times the sigmoid of the second; a depthwise convolution
     over time, kernel 5, causal (4 rows of left padding), no bias; an RMS
     norm with a gain; SiLU; a pointwise projection to 1,024; added to the
     residual, no norm.
  4. The second half-step feed-forward, as the first.
  5. An RMS norm whose output replaces the residual.
- **Clipped linears.** Every product in a layer clamps its input to
  `input_min`..`input_max` and its output to `output_min`..`output_max`,
  four F32 scalars stored beside each weight (`use_clipped_linears` true in
  Google's config). The bounds vary by layer and product: layer 0's
  attention projections clamp inputs to ±20 and outputs to ±34, its
  output projection's outputs to ±100, its second pointwise projection's
  inputs to ±5.8; layer 5's Q to ±9.5 and ±17. In the file each weight's
  four scalars lie around it in index order: `input_max`, `input_min`, the
  weight, `output_max`, `output_min`.
- **Out.** `a.pre_encode.out` (1,024 to 1,536, with a bias: Google's
  `output_proj_dims`), then the multimodal embedder: an RMS norm without a
  gain, then `mm.a.input_projection`, 1,536 to 512, the text model's width.
- **Into the text model.** As an image's: the text writes `<|audio|>`; it is
  replaced by `<|audio>`, the R rows, `<audio|>` (token ids 256,000 and
  258,883; llama.cpp's `aud_beg` and `aud_end` for `gemma4a`), the rows
  unscaled, and the text encoder runs as for text.

`per_dim_scale` is stored with softplus already applied: Hugging Face's
model takes softplus of the raw parameter at run time
(`modeling_gemma4.py`), and llama.cpp's converter applies it when it writes
the file (`conversion/gemma.py`), so llama.cpp multiplies by the stored value
as it is. The stored values run from 0.0048 to 0.128.

Everything above matches between Google's config, Hugging Face's model and
llama.cpp's graph: SiLU (`hidden_act`), chunk 12 with 12 past positions,
softcap 50, residual weight 0.5, kernel 5, channels 128 and 32, output 1,536.
Hugging Face's mask admits distances 0 to 11 (`dist < left_window_size`, the
window passed as `context_left − 1`), as llama.cpp's does (`gq − gk <
max_past`).

## Preparing a waveform

Google's feature extractor (`Gemma4AudioFeatureExtractor`,
`processor_config.json`; `feature_extraction_gemma4.py`): 16 kHz mono, as the
card says audio should be supplied; frames of 320 samples (20 ms) every 160
(10 ms), after 160 samples of zeros on the left; a periodic Hann window of
320; an FFT of 512, the frame zero-padded; the magnitude, not the power; 128
HTK mel filters from 0 to 8 kHz without area normalization; then
log(mel + 0.001). No pre-emphasis, no dither, no per-bin normalization.
Frame i is kept when its last sample is real audio, so n samples make T =
⌊(n − 161) / 160⌋ + 1 frames.

llama.cpp (`mtmd_audio_preprocessor_gemma4a`, `mtmd-audio.cpp`) makes the
same frames, window, FFT, magnitude and filters, and trims to the same T.
It differs in one place: its log is of **max(mel, 0.001)**, not mel + 0.001.
Its `filter_params` has the other form (`mel_floor_add`), and the gemma4a
preprocessor doesn't set it.

The two agree wherever a bin's energy is well above 0.001 and differ by up
to ln 2 where it is near it. Measured through llama.cpp on Metal, each
clip's embedding with llama.cpp's form against its embedding with
`mel_floor_add` set:

| Clip | Cosine |
|---|---|
| 1 kHz tone, 3 s | 0.99684 |
| a 600 to 1,200 Hz siren, 4 s | 0.99649 |
| white noise, 3 s | 0.999999 |
| clicks twice a second, 3 s | 0.999999 |
| silence, 3 s | 1.000000 |
| three spoken sentences, 3 to 4 s | 0.99979 to 0.99997 |

Small for speech, larger for a pure tone, whose spectrum is near the floor
almost everywhere but one bin. llama.cpp's CPU and Metal disagree by
cosines of 0.9997 to 0.99996 on images (the vision note), so the tones'
difference is ten times the backends' spread and would show; speech's is
within it.

What follows for Charlotte: its preprocessing computes Google's form, and
its oracle is llama.cpp with that one line set, `params.mel_floor_add =
true` in the gemma4a preprocessor, applied by the reference script to its
copy of the pin; the fixture records each clip's cosine with unchanged
llama.cpp beside it, as the vision fixture records GELU-quick's.

The FFT and the mel sums are floating point, so unlike the image resampler
the spectrogram can't be held to llama.cpp's bytes; it's held to a
tolerance, and the embedding to llama.cpp's backends' spread.

## How long a clip can be

The card gives 25 tokens a second of audio and a context of 8,192 tokens,
about 327 s of audio alone. Google's processor configuration says
`audio_seq_length` 280; its code caps one count of placeholders at that and
not the other (`processing_gemma4.py`), so it isn't a limit the processor
enforces consistently, and the card's figure is what the model is offered
for. llama.cpp cuts audio into 30 s chunks, 750 rows each, and encodes each
separately.

Charlotte's text encoder runs one step of at most 512 rows
(`runtime/embedder.h`): the ends, `<|audio>`, `<audio|>` and R rows, so R ≤
508, T ≤ 2,032 frames, n ≤ 325,120 samples: 20.32 s. A longer recording is
more than one embedding however it's cut: Charlotte cuts it into pieces of
equal length, each at most 20 s, and embeds each alone. llama.cpp would
embed 30 s chunks, which don't fit; a 20 s piece is a valid input of the
model's, and a search over pieces finds where in a recording a query is
answered, as a search over passages does.

## Rows, measured

Through llama.cpp, the rows a clip makes: 3 s, 48,000 samples, 75 rows; 4 s,
100; 20 s, 500. 25 a second, as the card says.

## Does it find what was said

Through llama.cpp on Metal, queries written as the search panel writes
them (`task: search result | query: …`) against clips embedded alone:

- **Synthetic sounds are weak tests.** A 1 kHz tone, a siren, white noise,
  clicks and silence, each 3 to 4 s: "static white noise" ranks the noise
  first by 0.029, "a wailing siren" the siren by 0.011, "a ticking clock"
  the clicks by 0.009; "a steady high-pitched beep" ranks the tone first by
  0.003 with llama.cpp's mel form and the siren first with Google's;
  "silence" ranks the silence last.
- **Speech is a strong test.** Three sentences synthesized by the
  development machine's speech synthesizer, used for this probe alone: each
  sentence's paraphrased query ranks its own clip first, 0.75 to 0.78
  against 0.59 at most for anything else.
- **Public-domain recordings.** Seven clips from NASA's historical sounds
  page, US government works, decoded to 16 kHz mono and cut to 20 s. Four
  rank their own query first by clear margins:

| Clip | Query | Own | Best other |
|---|---|---|---|
| President Kennedy, "we choose to go to the moon", first 20 s | a president explaining why we go to the moon | 0.716 | 0.677 |
| Apollo 8's Christmas message, 13.5 s | a christmas greeting from astronauts orbiting the moon | 0.681 | 0.588 |
| STS-135's countdown, 20 s | a countdown to a rocket launch | 0.700 | 0.603 |
| Sputnik's beep, 4.4 s | the beeping signal of the first satellite | 0.702 | 0.655 |

  The other three, Apollo 13's "Houston, we've had a problem", "the Eagle
  has landed" and "one small step", are older and noisier, and the
  countdown, which is mostly radio voices, outranks them for their own
  queries. Those four are the tests' speech; the synthetic sounds stay for
  the spectrogram's accuracy, where a tone shows the mel floor's form.

## What a clip costs here, derived

For a 20 s piece: T = 2,000 frames, R = 500 rows, and a text step of about
504 rows.

| | Multiply-adds |
|---|---|
| Subsampling, first convolution, 1,000 × 64 × 128 × 9 | 7.4 × 10⁷ |
| Subsampling, second, 500 × 32 × 32 × 1,152 | 5.9 × 10⁸ |
| Input projection, 500 × 1,024² | 5.2 × 10⁸ |
| 12 layers' products, 24.1 × 10⁶ a row a layer | 1.45 × 10¹¹ |
| 12 layers' attention and relative terms, 3.8 × 10⁴ a row a layer | 2.3 × 10⁸ |
| Out projection and multimodal projection, 500 × 2.36 × 10⁶ | 1.2 × 10⁹ |
| **Audio encoder** | **1.48 × 10¹¹** |

- At 2.9 TFLOPS that's about 0.10 s. llama.cpp on Metal, warm, measured
  62 to 65 ms for 500 rows; its first encodes after loading took 300 to
  470 ms. Unlike the vision encoder, attention is a sliver of the work: each
  query sees 12 keys, so the encoder is its matrix products.
- The relative position term's projection, 13 positions × 1,024² a layer,
  depends on the weights alone: 1.6 × 10⁸ multiply-adds once, at load.
- On the CPU, the spectrogram: 2,000 FFTs of 512, about 2,300 butterflies
  each, and 128 sparse filters, about 0.5 × 10⁷ operations together: a few
  milliseconds in the module.
- The text step after it, about 504 rows: about 45 ms, between the
  measured 39.2 ms at 418 tokens and the estimate at 512
  (`2026-10-11-chrome-encode-benchmark.md`).
- Memory: the audio span, 328 MB of weights; working buffers at 500 rows,
  about 70 MB, of which the first convolution's output, ⌈T / 2⌉ × 64 × 128
  floats, 33 MB, is the largest: about 400 MB beside the text model's 384
  and the vision encoder's 320, inside the 2 GiB default.
- Download: 328.5 MB, the header and the audio span.

## What it says

- **The audio path is the image path with another encoder in front of it.**
  Rows of the text model's width, placed between two tokens; the encode
  after them exists.
- **The encoder is matrix products with five things the harness has no form
  of**: a convolutional front end with a mean-centred layer norm; clamped
  products, input and output, with bounds stored beside each weight; a
  half-step residual; a GLU and a causal depthwise convolution; and local
  attention with a relative position term and a softcap.
- **The oracle needs one correction**, the log's floor, which llama.cpp at
  the pin takes in the other form.
- **A clip is at most 20 s here**, by the text encoder's one step; a
  recording is pieces.
- **Speech, not tones, tells whether it works.**
