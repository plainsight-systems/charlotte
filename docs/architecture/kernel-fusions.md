# Kernel fusions

Which steps of a pass share a kernel, and why. A pass is a fixed sequence of
GPU dispatches, and in the browser each dispatch carries a cost of its own,
so how the steps of [`logical-overview.md`](logical-overview.md) group into
launches is part of the design, not a detail of each kernel.

## Why fusion is the lever

On the target (Apple M3 Max, Metal, measured in the desktop app's
Chromium 152), holding a pass's work
constant and splitting it into more dispatches adds about 1.5 µs of GPU time
for each extra dispatch (`core/kernels/interface.h`). Reading Qwen3 0.6B's
380 MB of weights once, which every decode step does, takes at least 0.95 ms
at the M3 Max's 400 GB/s. So 300 launches would add about half again on top
of the step's floor.

Native engines remove launch cost by capturing a sequence once and replaying
it — CUDA graphs. WebGPU has no captured compute sequence: every step records
its dispatches anew, and the GPU pays for each. What is left is to have fewer
of them.

## What the established engines fuse

- **vLLM** (`docs/design/fusions.md`, `csrc/`): RMSNorm is a kernel of its
  own, with the residual add before it fused in (`fused_add_rms_norm`), and
  optionally the quantization after it. QK-norm and RoPE are one kernel
  (`fused_qk_norm_rope`, for head sizes 64, 128 and 256), quoted at 2–3% end
  to end; a newer variant writes the KV cache in the same kernel. RoPE and the
  cache write are one kernel on AMD. The activation joins the matmul around
  it. No norm is fused into a GEMM.
- **llama.cpp, Metal** (`ggml/src/ggml-metal/ggml-metal-ops.cpp`): the norm
  kernel absorbs the element-wise operations after it — RMS_NORM + MUL (the
  gain), + MUL + ADD, or + SCALE — and stays its own dispatch.

The pattern they share: a reduction across a row is its own kernel, and the
cheap element-wise work beside it joins it.

## The rules here

- **A norm is its own kernel,** since it reduces across a whole row before
  any of the row can be written. It does the residual add before it and the
  gain after it. Where an architecture normalizes a block's output before
  adding it — Gemma 3's post-norms — that norm is a stage of the same kernel.
- **The residual add belongs to the norm that begins the next block,** not to
  the matmul that ends the last one. A block's last matmul writes its result
  to a working buffer; the norm after it adds that into X. That is the only
  arrangement in which Gemma 3's post-norms, which come between the block's
  output and the add, need no kernel of their own. The final norm adds the
  last block's output the same way.
- **Projections that read the same input are one matmul:** Q, K and V; gate
  and up.
- **The activation is the epilogue of the gate and up matmul,** which writes
  activation(gate) × up and never the two separately.
- **QK-norm, RoPE and the cache write are one kernel.** Each head's query and
  key is normalized where the file has QK-norm, rotated by position, and the
  key and value appended to the layer's cache, in one dispatch.

## Launches in a pass

For one layer, in order:

| # | Launch | Fused into it |
|---|---|---|
| 1 | norm | the last block's output added into X; Gemma 3's post-norm of it first; the attention norm and its gain |
| 2 | matmul | the Q, K and V projections |
| 3 | rope | QK-norm where the file has it; RoPE on queries and keys; the keys and values appended to the cache |
| 4 | attention | scores, causal and window mask, softmax and the weighted sum, never writing the scores out; where a step is split across the context, a combine launch after it folds the splits |
| 5 | matmul | the output projection, writing the block's output |
| 6 | norm | that output added into X; Gemma 3's post-norm of it first; the feed-forward norm and its gain |
| 7 | matmul | the gate and up projections, and the activation of their product |
| 8 | matmul | the down projection, writing the block's output |

Around the layers: the gather, one launch for each piece of the embedding
table; after them, the final norm, adding the last block's output, the
output head, top-k selection's three passes and the draw, all on the step's
last token only (`core/kernels/topk/topk.h`, `core/sampler/sampler.h`).

| Model | Layers | Gather | Layers × 8 | Final norm, head, selection, draw | Launches a pass | At 1.5 µs each |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | 28 | 1 | 224 | 6 | 231 | 0.35 ms |
| Llama 3.2 1B | 16 | 2 | 128 | 6 | 136 | 0.20 ms |
| Gemma 3 1B | 26 | 3 | 208 | 6 | 217 | 0.33 ms |

A layer split across the context — a step whose rows times the layer's
64-key chunks fit the partial buffers, as a decode step's always do, once its
keys span two chunks, from position 64 — adds attention's combine in that layer: up to 28
more for Qwen3, 16 for Llama 3.2, 26 for Gemma 3
(`core/kernels/attention/attention.h`).

Without these fusions Qwen3's pass would be over 300 launches: a norm, three
projections, two QK-norms, RoPE, a cache write, attention, a projection and
an add for attention, and a norm, two projections, an activation, a
projection and an add for the feed-forward, in each of 28 layers.

## Levers not taken

- **The norm folded into the matmul after it.** Each matmul workgroup would
  compute its input row's RMS as it loads the row, saving the two norm
  launches a layer — 56 a pass for Qwen3, about 0.08 ms. Neither engine above
  does it, and in prefill every workgroup of a tile would repeat the row's
  reduction. The matmul's design weighs it against that repetition.
- **One gather launch for every piece of the table.** It would save 1.5 µs a
  step for Llama 3.2 and 3 µs for Gemma 3, and nothing for Qwen3, at the cost
  of giving the format's unpack more than its one weights binding
  (`core/kernels/gather/gather.h`).
