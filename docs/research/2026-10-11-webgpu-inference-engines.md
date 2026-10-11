# Five WebGPU inference engines against Charlotte's kernels

**Date:** 2026-10-11
**Method:** each engine's WebGPU kernels read from source at a pinned commit,
and its shipped WGSL where it ships some; its published numbers taken with
their conditions. No engine was run here. Every figure below is the engine's
own or a third party's, never measured on this machine, unless it says
Charlotte.

| Engine | Source read | What runs in a browser |
|---|---|---|
| llama.cpp, ggml-webgpu | `ggml-org/llama.cpp` `36a73916ee`, checked against `f2cc728` | the WebGPU backend compiled with Emscripten (wllama) |
| MLC-LLM / WebLLM | `apache/tvm` `4b86188`, `mlc-ai/mlc-llm` `8978ea9`; the WGSL in WebLLM's Qwen3-0.6B q4f32_1 and q4f16_1 libraries | TVM-generated WGSL |
| ONNX Runtime Web | `microsoft/onnxruntime` `6ce927e`; `huggingface/transformers.js` `511bb61` | the C++ WebGPU execution provider compiled to WASM |
| Google ML Drift, MediaPipe, LiteRT-LM | `google-ai-edge/ml-drift` `7c58cf0`; arXiv 2505.00232 | ML Drift's WebGPU backend through Emscripten |
| CubeCL / CubeK (Burn), Ratchet | `tracel-ai/cubek`, `huggingface/ratchet` at their pinned heads | wgpu-generated WGSL |

## Why

The encode benchmark (`2026-10-11-chrome-encode-benchmark.md`) found short
steps' matrix products running a seventh of the rate long ones reach, and a
512-token prefill step at about a quarter of the GPU's f32 arithmetic peak.
Before designing the next kernels, the question is what the engines that
already run models on WebGPU do, and how much of their speed is available
under Charlotte's constraints: WebGPU's default limits, 16 KiB of workgroup
memory and 256 invocations, and no optional feature — no `shader-f16`, no
`subgroups`.

## What they need that Charlotte does not take

| Engine | Optional features | Limits |
|---|---|---|
| ggml-webgpu | `shader-f16` required: no device without it; `subgroups` when offered | every adapter maximum |
| WebLLM | `shader-f16` and `subgroups` when offered | 32 KiB workgroup memory required; prefill attention uses 24.8 KiB at f32 |
| ORT Web | `shader-f16`, `subgroups`, subgroup matrices, subgroup size control when offered | every adapter maximum |
| ML Drift in the browser | MediaPipe requires `shader-f16`; LiteRT-LM requests it; `subgroups` requested, unused on Apple | **the defaults, 16 KiB and 256** |
| CubeCL, Ratchet | Ratchet requires `shader-f16` and `subgroups` | adapter maxima; CubeCL's f32 tile needs 32 KiB on Apple |

Only Google's browser path runs at the default limits, and it too takes
`shader-f16`. None uses subgroup matrices — Apple's 8×8 matrix units through
WebGPU — in a browser: the feature is behind an unsafe flag in Chrome, and
ggml and ML Drift compile it out under Emscripten.

## What each does

**Prefill matrix products.**

| Engine | Workgroup tile | Per invocation | Workgroup memory | Weights | Split along K |
|---|---|---|---|---|---|
| Charlotte | τ tokens × 64 outputs, τ 8, 16 or 32 | 4 × 4, f32 | 9 KiB of weights, up to 4 of inputs | decoded into workgroup memory each K-step | yes, for few tiles (`kernels/matmul/matmul.h`) |
| ggml-webgpu | 32 × 32, 64 invocations | 4 × 4 | 4 KiB, f16 | decoded to f16 in workgroup memory; activations rounded to f16 | no |
| WebLLM | 32 × 32, 64 invocations | 4 × 4, scalar loads | 2 KiB | decoded in the copy to workgroup memory | no |
| ORT Web, no features | 1 output × 16 rows a thread, 128 threads | 16 accumulators | 2 KiB | decoded a 32-block a K-step | no, for quantized weights |
| ML Drift, browser | a plain float convolution, no workgroup tiling | up to 2 tokens × 16 outputs | none | decoded in registers; from about 200 tokens (f16) or 360 (f32), converted once to a float scratch buffer | no |
| CubeK, no features | 128 × 128 × 16 | 8 × 8 | 32 KiB on Apple | packed words staged, decoded at the register load | benchmarked, not shipped |

**Short steps.** Every engine but CubeK routes the shortest steps away from
its tiled kernel to a matrix–vector form that reads each weight once for a
few tokens: ggml up to 4 tokens, WebLLM up to 8 (padded to 2 or 4), ORT up
to 3, ML Drift up to 8 at f32 and 16 at f16. None splits a quantized product
along K; Charlotte's split is the only one, and the only one that keeps a
token's bits the same at every step size.

**Decode.** ORT's and Ratchet's no-subgroup forms are Charlotte's: 32 lanes
along K a row, partial sums meeting in workgroup memory. WebLLM uses 64
lanes and one row a workgroup; ggml 256 threads for 4 rows.

**Attention.** ggml (with subgroups), ORT and WebLLM fuse prefill attention
as flash attention; ML Drift in the browser materializes the scores in three
dispatches, as Charlotte's encoder does. ggml does not share K and V tiles
across the query heads of a group.

**Determinism.** Every other engine changes kernel, and so a token's
reduction order, with the step's length: none is batch-invariant. ORT turns
off its only split-K, which uses atomics, when determinism is asked for.

## Published numbers nearest Charlotte's

| Engine | Model | Device | Prefill | Decode | Source |
|---|---|---|---|---|---|
| ML Drift, browser, f32 | Gemma 3 1B, int4 / Q4_0 | M4 Max | 2,837–2,943 tok/s | 134 tok/s | LiteRT community model card, 2026-09-01 |
| ML Drift, browser, f16 | Gemma 3 1B, int4 / Q4_0 | M4 Max | 4,339–4,531 | 133 | the same |
| ggml-webgpu, native Dawn | Llama 3.2 1B Q4_0 | M3 | 960 (pp512) | 41.8 | llama.cpp PR #17031 |
| ggml-webgpu, Chrome | Llama 3.2 1B q4_k_m | M3 | — | about 52 | arXiv 2605.20706 §4 |
| WebLLM | Phi-3.5-mini, Llama 3.1 8B | M3 Max | — | 71.1, 41.1 | arXiv 2412.15803 |
| ORT Web, subgroup matrices | Phi-3.5-mini | unnamed | 68.8 → 184.5 tok/s | — | onnxruntime PR #23729 |
| **Charlotte, native** | Qwen3 0.6B Q4_0 | M3 Max | 512 tokens in 129 ms, about 3,970 tok/s | — | `make profile`, 2026-10-11 |
| **Charlotte, Chrome** | Qwen3 0.6B Q4_0 | M3 Max | 951 tokens to first token in 296 ms | 270 tok/s | `2026-10-07-chrome-page-benchmark.md` |

No engine publishes a figure on Charlotte's models and machine. The nearest
is Google's browser path on Gemma 3 1B: about 4.0 to 4.1 TFLOP/s of its
products' arithmetic at f32 on an M4 Max, against Charlotte's 3.5 on Qwen3
0.6B on an M3 Max, natively. Different models, chips and clocks: no closer
than within about a quarter, either way.

## What it says

- **Charlotte is not behind under its own constraints.** Each engine's speed
  over Charlotte's kind of kernel comes from what Charlotte does not take:
  f16, subgroups, subgroup matrices or larger workgroup memory. Without them
  ORT's, WebLLM's and ggml's prefill kernels are smaller tiles than
  Charlotte's, and Google's f32 path, the one engine at the default limits,
  comes out within about a quarter of it, on another chip.
- **f16 is the lever with a measured size.** Google's browser path gains
  1.53 to 1.55 times in prefill from f16 on Gemma 3 1B, and nothing in
  decode. It changes every activation's rounding, so Charlotte's reference
  tests would hold it to a tolerance f16 would need, and it is an optional
  feature: a decision about the target, not a kernel change.
- **Within the constraints, four levers, each to measure before it is built:**
  1. **A matrix–vector form for the shortest steps.** Four engines route
     steps of up to 3 to 16 tokens to one. Timed in Chrome on this machine
     while the split was designed (`kernels/matmul/matmul.h`, levers not
     taken), such a form, written to decode's order and so to the same bits,
     beat both the tiled form and the split at 8 tokens on EmbeddingGemma 2's
     gate-and-up shape on Q4_0 weights, 41 µs against 58 and 54, and lost
     past 16 — and gate and up is the product the split leaves.
  2. **A larger per-invocation tile.** CubeK keeps 8 × 8 accumulators an
     invocation, WebGPT 4 × 8; Charlotte 4 × 4, for its batch-invariant
     order's range sums beside the totals (`matmul.h`'s costs). Its effect on
     registers and occupancy at 256 invocations is unmeasured.
  3. **Packed weights in workgroup memory.** CubeK stages Q4 words packed and
     decodes them at the register load, reporting it as fast as staging
     decoded values in a quarter of the memory; at 16 KiB that room is what
     a deeper K-stage or a wider tile needs.
  4. **Weights converted once for long steps.** ML Drift converts int4 to a
     float buffer once a product from 200 to 360 tokens up, where Charlotte
     decodes each weight once a 32-token tile, 16 times at 512 tokens.
- **The comparison to make next is on this machine.** ML Drift ships Qwen3
  0.6B and Gemma 3 1B at Q4_0 as samples for its WebGPU backend; run here,
  natively and in Chrome, they would put Google's numbers and Charlotte's on
  one chip and one model.
