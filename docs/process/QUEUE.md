# Work Queue

This file tracks active and accepted work.

## Active

- **Embedding in the browser** — EmbeddingGemma 2's text encode path,
  designed in its file headers (`runtime/embedder.h`, `graph/encoder.h`,
  `arch/embedding_gemma2`).

  In: the module loads the model, encodes a text of up to 512 tokens with
  its ends as one step, and answers the worker's EMBED with the unit
  vector. Each pinned text's embedding falls within llama.cpp's own
  CPU-to-Metal spread of its Metal output, natively
  (`tests/gpu/embedder_test.cpp`) and through the worker in the app's
  Chromium; a swapped RoPE pairing falls 30 to 60 times outside it.

  The page searches a pasted document (`web/search.js`): its paragraphs
  embedded with the card's task prompts, one too long cut and its pieces
  embedded, a query ranked against them. In the app's Chromium each of the
  fixture's seven queries (`tests/fixtures/search/`) ranks its own
  paragraph first.

  In Chrome a query encodes in 7.0 ms and 418 tokens in 39.2 ms, a short
  step's matrix products split along K; the worker's hops take under 0.1
  ms (`research/2026-10-11-chrome-encode-benchmark.md`).

  Open: the panel with an iPhone's on-screen keyboard up.

- **Images, through EmbeddingGemma 2's vision encoder** — photos embedded
  into the same space as text, so one query ranks paragraphs and photos
  together; the companion file read in part, its vision span alone
  (`research/2026-10-11-embeddinggemma2-vision.md`). Designed in its file
  headers (`media/image.h`, `arch/gemma4v`, `graph/vision.h`,
  `runtime/vision.h`, `residency/file_ranges.h`, block attention's
  streaming form, `web/companion.js`, `web/image_input.js`). Audio and
  video follow it, each its own design, before EmbeddingGemma 2 is
  announced as multimodal.

  In: the module prepares an image byte for byte as llama.cpp does, and
  each image's embedding falls within llama.cpp's own CPU-to-Metal spread
  of its Metal output (`tests/gpu/vision_embed_test.cpp`). The page
  downloads the companion's header and vision span, 226 MB of its 555,
  checks them against a pinned SHA-256 and loads them the first time a
  photo is indexed. In the app's Chromium, photos and the fixture's
  paragraphs share one index; each query for the four test scenes ranks
  its own scene first, every score llama.cpp's to three places; a
  12-megapixel photo indexes in 0.49 s. In iOS Safari, under the fake
  runtime, photos come from the library and the ranged download is
  verified and cached.

- **Audio, through EmbeddingGemma 2's audio encoder** — recordings embedded
  into the same space as text and photos, cut into clips of up to 20 s, so
  a query finds where in a recording it is answered; the companion file's
  header and audio span, one 328.5 MB range
  (`research/2026-10-11-embeddinggemma2-audio.md`). Designed in its file
  headers (`media/audio.h`, `arch/media_encoder.h`, `arch/gemma4a`,
  `graph/audio.h`, `runtime/audio.h`, `runtime/media_stage.h`, the
  subsample, local attention and light convolution kernels, the matrix
  product's clamped, biased and activated options, the norm's audio
  variants, the plan's product spans, `web/audio_input.js`,
  `web/recordings.js`). The spectrogram follows Google's log floor where
  llama.cpp's differs; the oracle is llama.cpp with that one line set.

  In: the module computes the spectrogram as that llama.cpp does, within
  its tiered tolerance (`tests/media_audio_test.cpp`), and each computed
  sound, NASA recording, captioned clip and query embeds within llama.cpp's
  own CPU-to-Metal spread of its Metal output
  (`tests/gpu/audio_embed_test.cpp`); llama.cpp's own floor falls outside
  on the tone and the siren, and the relative rows read unshifted fall
  outside on every recording. The page reads a 16 kHz 16-bit WAV itself,
  sample for sample as llama.cpp does, and loads the companion's audio
  range the first time a recording is indexed. In the app's Chromium,
  recordings, photos and paragraphs share one index; each of the four
  recordings' queries ranks its own recording first, 14 of 16 scores
  llama.cpp's to three places and the other two at a rounding boundary,
  within llama.cpp's own spread; a 5 minute recording becomes 15 clips in
  about 2 s, its matching clip ranked with its span. In iOS Safari, under
  the fake runtime, the recordings list and a clip's player fit the phone
  with the on-screen keyboard up.

- **Video, through EmbeddingGemma 2's vision encoder** — videos embedded
  into the same space as text, photos and recordings, cut into clips of a
  frame a second, four frames to a 16:9 clip, so a query finds where in a
  video it is answered; frames read by the image companion's encoder, no
  download of their own (`research/2026-10-11-embeddinggemma2-video.md`).
  Designed in its file headers (`media/video.h`, `runtime/video.h`,
  `runtime/media_stage.h`'s segments, the step's media rows by identifier
  in `kernels/interface.h` and `kernels/gather`, `kernels/program.h`'s copy
  after a pass, `arch/media_encoder.h`'s video kind, `web/video_input.js`,
  `web/videos.js`). Frames follow EmbeddingGemma 2's own processor, not
  llama.cpp's video helper; the oracle is llama.cpp's encoder per frame
  over Google's token sequence.

  In: each computed clip and query embeds within llama.cpp's own
  CPU-to-Metal spread of its Metal output, at about a tenth of it
  (`tests/gpu/video_embed_test.cpp`); frames given in reverse fall three
  to four times outside it, frames letterboxed as an image is ten to thirty
  times; each query ranks its own clip first. On the way, an encode now
  delivers only once every step it submitted has reported, which image and
  audio needed too. In the app's Chromium, four NASA clips
  (`tests/fixtures/video/`) index beside paragraphs, photos and recordings;
  each clip's query ranks its own video's clips first, each score within
  0.005 of llama.cpp's on FFmpeg's frames, the browser's decoded frames
  within half a unit a channel of FFmpeg's on average. In iOS Safari, under
  the fake runtime, videos chosen from the photo library index and play
  from their span, and the query stays above the on-screen keyboard.

  Open: the cost of a clip, measured in a visible pane. Timed with the
  pane hidden, which throttles it, eight clips took 10 s and eight
  frame-sized photos 3.3 s on this build and 3.5 s on the build before
  video; not a measurement of either.

## Accepted

- **Generation in the browser** — accepted 2026-10-07, designed in its file
  headers (`workflow.md`), so it has no packet.

  A model chosen in the page is downloaded, cached and checked against this
  device, uploaded as stored, and run by the module: both tokenizers
  (byte-level and SentencePiece BPE), the forward pass as WebGPU kernels,
  sampling on the GPU, and a chat with streamed replies and thinking. Three
  models: Qwen3 0.6B, Llama 3.2 1B Instruct and Gemma 3 1B, all Q4_0.

  Verification: the tokenizers match their references on every fixture
  case; each model's logits fall within llama.cpp's own CPU-to-Metal spread
  (`tests/gpu/reference_test.cpp`), and a deliberate error in position
  encoding or attention's window falls 33 to 66 times outside it. In Chrome
  on the development machine, Qwen3 reaches its first token in 21 ms on a
  short prompt and decodes at 270 tok/s
  (`research/2026-10-07-chrome-page-benchmark.md`); Llama 3.2 and Gemma 3
  each downloaded, loaded and held a two-turn chat there, and Gemma 3 a
  905-token reply, well past its local layers' 512-key window.

  Accepted without the per-commit review `workflow.md` calls for: the
  commits from `538e95d` onward, except `ab669eb`, were not reviewed.

- **BLLM-002: GGUF reading and Q4_0 layout** — accepted 2026-10-07.
  `packets/2026-08-31-model-selection-and-weight-loading.md`. The container
  reader over bounded windows, its named failures, and the Q4_0 layout with
  its bit-exact oracle. Criterion 4b, left open in the packet, is met where
  the packet said it would be: `residency::Upload::begin` takes tensors by
  their index entries, so no weight reaches the GPU apart from its type.

- **BLLM-003: Rename the project to Charlotte** — accepted 2026-09-30.
  `packets/2026-09-30-rename-to-charlotte.md`. Identity only; deployed at
  `plainsight-systems.github.io/charlotte/` and verified there.

- **BLLM-001: Repo skeleton and build system** — accepted 2026-08-31.
  `packets/2026-08-29-repo-skeleton-and-build-system.md`.

  Dual-target CMake build, platform-neutral core, WebGPU device path, wasm
  bindings, and a static page deployed by CI. Two independent reviews by a
  separate identity, both `changes_requested`, both sets of findings worked.

## Parking Lot

Speed, none of it a correctness gap:

- Attention's combine at depth: 12% of a native decode step at 8,192
  positions (`research/2026-10-07-app-chromium-decode-profile.md`).
- Prefill attention: 20.7% of a 512-token step natively (the same note).
- Decode's products of few rows, at 150 to 210 GB/s against the larger
  products' 300 and more (the same note).
- Prefill's products of many tiles over a short K at short steps, which
  the split along K leaves: gate and up, 35% of a 16-token Qwen3 step
  natively; a query still 7.0 ms against an estimate of 2
  (`research/2026-10-11-chrome-encode-benchmark.md`).
- The re-prefill when the chat template rewrites history, as after a
  reasoning turn.
