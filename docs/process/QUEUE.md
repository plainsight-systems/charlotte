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

  In Chrome a query encodes in 13.9 ms and 418 tokens in 38.5 ms; the
  worker's hops take under 0.1 ms
  (`research/2026-10-11-chrome-encode-benchmark.md`).

  Open: the panel with an iPhone's on-screen keyboard up.

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
- Prefill's workgroups for short prompts.
- The re-prefill when the chat template rewrites history, as after a
  reasoning turn.
- The encode's 11 to 13 ms that does not scale with its tokens, against
  about 1 ms counted: 7 times the estimate for a short query
  (`research/2026-10-11-chrome-encode-benchmark.md`). A native step
  profile would place it, and whether batching passages would share it.
