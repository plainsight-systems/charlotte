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

A prefill step that splits a matmul along K — a product whose tiles are few
beside its length along K, as an output or down projection's are in a short
step — adds that product's combine, which adds each output's 32 range sums in
order and writes them as the product would have
(`core/kernels/matmul/matmul.h`): in a step of up to 32 tokens, 3 a layer —
Q, K and V, the output and down projections — 84 more for Qwen3, 48 for
Llama 3.2, 78 for Gemma 3.

Without these fusions Qwen3's pass would be over 300 launches: a norm, three
projections, two QK-norms, RoPE, a cache write, attention, a projection and
an add for attention, and a norm, two projections, an activation, a
projection and an add for the feed-forward, in each of 28 layers.

## An encoder's step

An embedder runs one step over a whole text (`core/graph/encoder.h`). The
rules above hold, and four more follow from them:

- **The layer output scale rides in the next norm's add.** Gemma 4 multiplies
  all of X by a per-layer scalar once a layer's last block is added; the norm
  that adds that block does it, after the add and before its reduction.
- **The per-layer gate's product with the layer's input is its matmul's
  epilogue,** as the activation is gate and up's: GELU of the product, times
  the layer's slice of the per-layer inputs, written once.
- **Every layer's per-layer input is one matmul and one norm,** before the
  first layer, from the embedding: the projection to every layer's width at
  once, then each layer's slice normalized on its own, with the projection's
  scale folded into the norm's ε. Each layer's gate reads its slice in place.
- **Pooling comes before the output projection,** so the projection runs over
  one row, as a decoder's head does.

And one launch is two: **attention's scores and its weighted sum are separate
launches,** the scores written between them. At a global layer's head size of
512, WebGPU's default 16 KiB of workgroup memory leaves an online-softmax tile
either too many accumulators an invocation or a recomputation of the scores
for every slice of the output; one write and read of the scores, 4 MiB a
layer at 512 tokens, costs less (`core/kernels/block_attention/block_attention.h`).

For one layer, in order:

| # | Launch | Fused into it |
|---|---|---|
| 1 | norm | the last layer's per-layer output added into X through its post-norm, then X times the last layer's output scale; the attention norm and its gain |
| 2 | matmul | the Q, K and V projections |
| 3 | rope, encode form | QK-norm, the value norm, RoPE on queries and keys, all in place; no cache |
| 4 | block attention, scores | scores in both directions within the window, scaled, and each chunk's maximum |
| 5 | block attention, weighted sum | softmax and the weighted sum |
| 6 | matmul | the output projection |
| 7 | norm | that output added into X through its post-norm; the feed-forward norm and its gain |
| 8 | matmul | the gate and up projections, and the activation of their product |
| 9 | matmul | the down projection |
| 10 | norm, add only | that output added into X through its post-norm; X not normalized, since the gate reads it as it is |
| 11 | matmul | the per-layer gate, its GELU, and its product with the layer's input |
| 12 | matmul | the per-layer projection back to the hidden width |

Around the layers: the gather, one launch for each piece of the embedding
table; the per-layer inputs' projection and norm; and after the layers the
final norm over every row, the mean, the output projection over one row, and
the scaling to unit length. A step that splits a matmul along K adds its
combine, as a decoder's prefill does: up to 5 a layer, Q, K and V in the
sliding layers, the output, down and per-layer gate and projection, 116 more
in a step of up to 32 tokens.

| Model | Layers | Gather | Per-layer inputs | Layers × 12 | Final norm, mean, projection, unit | Launches a step | At 1.5 µs each |
|---|---|---|---|---|---|---|---|
| EmbeddingGemma 2 | 24 | 2 | 2 | 288 | 4 | 296 | 0.44 ms |

## A vision encoder's step

A vision encoder runs one step over an image's patches
(`core/graph/vision.h`). The rules above hold, with three more:

- **The learned positions ride in the first norm's add.** One gather sums
  each patch's column row and row row into the block output the first
  layer's attention norm adds into X anyway, so the positions cost no pass
  over X and no launch of their own to add.
- **Attention is one launch, streamed.** At a head of 64 the online
  softmax's tile fits the default 16 KiB, and at thousands of patches the
  scores could not be written in any case: 305 MB a layer at 2,520
  patches, over a binding (`core/kernels/block_attention/block_attention.h`).
- **Pooling and the norm after it are one launch,** a workgroup a pooled
  row, the √W scale folded into the norm's ε.

For one layer, in order:

| # | Launch | Fused into it |
|---|---|---|
| 1 | norm | the positions, or the last layer's output through its post-norm, added into X; the attention norm and its gain |
| 2 | matmul | the Q, K and V projections |
| 3 | rope, encode form, axial | QK-norm, the value norm, each head's halves rotated by the patch's column and row, all in place |
| 4 | block attention, streaming | scores, softmax and the weighted sum over every patch, the scores never written |
| 5 | matmul | the output projection |
| 6 | norm | that output added into X through its post-norm; the feed-forward norm and its gain |
| 7 | matmul | the gate and up projections, and the activation of their product |
| 8 | matmul | the down projection |

Around the layers: before them the patch embedding and the positions'
gather; after them the last layer's output added into X, the pooling with
its norm, and the projection to the text width. No product of a step this
long splits along K.

| Model | Layers | Patches, positions | Layers × 8 | Add, pool, projection | Launches a step | At 1.5 µs each |
|---|---|---|---|---|---|---|
| EmbeddingGemma 2's vision encoder | 16 | 2 | 128 | 3 | 133 | 0.20 ms |

The image's rows then reach the text step by one copy between the two
programs before the text step's pass, not a launch.

## An audio encoder's step

An audio encoder runs one step over a clip's spectrogram
(`core/graph/audio.h`). The rules above hold, with four more:

- **Every block boundary is one norm.** A Conformer layer has four
  residual adds, two at half weight, three post-norms and a closing norm
  that replaces the residual stream. Each add, its post-norm, its weight,
  the closing norm and the next block's norm are one launch, so a layer's
  norms are four launches, one a block.
- **The clamps ride in the products.** Each product reads its four bounds
  from its own weight's binding and clamps its input as it stages it and
  its output before it writes; no clamp is a pass.
- **The light convolution is two launches.** The GLU rides in the first
  pointwise product's gated epilogue, the weight's two halves its members;
  the depthwise convolution over 5 rows, its norm and SiLU are one launch, a
  workgroup a row, where llama.cpp's graph makes five passes.
- **Attention is the window alone.** Each row against its 11 predecessors
  and itself, the relative term read from a table computed once at load,
  where llama.cpp's blocked form scores 24 keys a row and masks half.

For one layer, in order:

| # | Launch | Fused into it |
|---|---|---|
| 1 | norm | the last layer's output through its post-norm at half weight, the layer's closing norm, then this layer's feed-forward norm; or, in layer 0, the input projection's rows normalized |
| 2 | matmul | the up projection, clamped, through SiLU |
| 3 | matmul | the down projection, clamped |
| 4 | norm | that output through its post-norm at half weight; the attention norm |
| 5–7 | matmul | Q, K and V, each clamped |
| 8 | local attention | scores, the relative term, the cap, softmax and the weighted sum over the window |
| 9 | matmul | the output projection, clamped |
| 10 | norm | that output through its post-norm; the convolution module's norm |
| 11 | matmul | the first pointwise projection, clamped, and the GLU of its halves |
| 12 | light convolution | depthwise convolution, norm, SiLU |
| 13 | matmul | the second pointwise projection, clamped |
| 14 | norm | that output added; the second feed-forward's norm |
| 15 | matmul | its up projection, clamped, through SiLU |
| 16 | matmul | its down projection, clamped |

Around the layers: before them the two convolutions and the input
projection; after them the last layer's output added and closed, the
output projection with its bias, a norm without a gain, and the projection
to the text width. A short clip's products may split along K, adding their
combines. Once, at load, a setup step of 12 products computes the relative
tables.

| Model | Layers | Subsampling | Layers × 16 | Close, project, norm, project | Launches a step | At 1.5 µs each |
|---|---|---|---|---|---|---|
| EmbeddingGemma 2's audio encoder | 12 | 3 | 192 | 4 | 199 | 0.30 ms |

The clip's rows reach the text step as an image's do, by one copy.

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
