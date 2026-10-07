#pragma once

#include <string>
#include <vector>

#include "core/formats/format.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace bllm::graph {

// Axis A: changes when an architecture brings a structure the layers here do
// not have — a plain feed-forward block, experts, attention and feed-forward
// side by side.
//
// The forward pass: every kernel launch of a step, in order, described from
// the model description and the residency plan as upload carried it out
// (Upload::plan(), where a confirmed duplicate head reads the embedding's
// bytes). Pure, as the kernel launchers are: no GPU object, no fetch, so the
// program builds from it at load (kernels/program.h) and preflight runs it
// before any weight is fetched, a failure naming what stops the Run stage
// (preflight/preflight.h).
//
// One graph serves every architecture. Each family's differences are stated
// in the model description by its describe — which weights a layer has, its
// window and rotary base, the pairing, the activation, the attention and
// embedding scales — so the graph reads structure, never a family's name
// (docs/architecture/logical-overview.md, principle 1). Three graphs would
// be three copies of this order.
//
// The order, a step:
//   - Embed: the token embedding's gather, one launch a piece, into `hidden`,
//     each weight times the description's embedding scale
//     (kernels/gather/gather.h).
//   - Each layer, in order:
//       1. Attention norm: `output` added into `hidden` and `hidden`
//          normalized into `normed` with the layer's gain. Layer 0 does not
//          add: the gather wrote `hidden`. Where the layer before has a
//          post-feed-forward norm, `output` is normalized with it before it
//          is added (kernels/norm/norm.h).
//       2. Q, K and V from `normed`: the plan's group as one product into
//          `query`, `key` and `value`; where the plan made none, three
//          products, each writing its own (kernels/matmul/matmul.h).
//       3. QK-norm where the layer has it, RoPE, and the key and value
//          appended to the layer's cache (kernels/rope/rope.h).
//       4. Attention over the layer's cache into `attention`, and its
//          combine (kernels/attention/attention.h).
//       5. Output projection: `attention` into `output`.
//       6. Feed-forward norm: `output` added into `hidden`, normalized into
//          `normed`; where the layer has a post-attention norm, `output` is
//          normalized with it first.
//       7. Gate and up from `normed`, the plan's group as one product writing
//          activation(gate) × up into `activation`.
//       8. Down: `activation` into `output`.
//   - Final norm: the last layer's `output` added into `hidden` — through
//     its post-feed-forward norm where it has one — and normalized with the
//     output norm's gain, the step's last token only.
//   - Head: `normed` into `logits`, the last token only; the output head's
//     weight, or the token embedding's where the file ties them.
// A matrix product is all its launcher's launches — a decode launch and one
// a prefill tile width, for each piece — and attention its main launch and
// its combine; the program dispatches the ones a step's token count and
// position select (kernels/interface.h). Which steps share a launch, and
// why, is docs/architecture/kernel-fusions.md.
//
//   - Residual: the norm that begins each block adds the last block's
//     output into X, so X is never written by a matrix product
//     (kernel-fusions.md). Post-norms are detected — present where the file
//     has the weights — never assumed.
//   - Buffers: every launch reads working buffers other than the ones it
//     writes, each its own GPU buffer (residency/plan.h), so no dispatch
//     binds one buffer both read-only and writable. WebGPU orders a pass's
//     dispatches by their buffer use, so each launch sees what the one before
//     it wrote.
//   - Gate and up must be one group. Their epilogue reads both in one
//     product, and the plan holds no buffer for either alone; a layer whose
//     gate and up the plan could not group — formats or widths that differ,
//     a member in pieces, a span wider than a binding — is refused, naming
//     the layer. Every listed file groups them. Q, K and V, each written to
//     its own buffer, need no group.
//   - Kernel shapes: the preconditions rope, attention and norm state for a
//     layer — a head dimension of 64, 128 or 256; (query heads + 2 ×
//     key-value heads) × head dimension / 8 a multiple of 64; the query heads
//     a key-value head serves dividing 1,024 / head dimension; QK-norm on
//     both queries and keys or neither; a hidden width of at most 4,096 — are
//     checked here, their one caller, and a layer outside them refused naming
//     the layer and the value. A weight's rows a whole number of 32-weight
//     groups, which matrix products and the gather need, is routes' check
//     (residency/routes.h), at preflight's Upload stage.
//   - The cache format is the one the plan sized the cache in, from load
//     policy's precision (policy/policy.h); rope packs with it and attention
//     unpacks with it.
//   - Determinism (GDSA.2): every kernel on the path is batch-invariant —
//     the gather, norms and rope work a row alone, attention folds chunks
//     fixed by position, each matrix product adds in an order fixed by K —
//     and the graph adds no reduction of its own. So a token's logits are
//     the same bits whether its prompt was prefilled in one step, in
//     several, or decoded a token at a time, on the target's compiler, as
//     each kernel states.
//
// What a step costs, counted, for Qwen3 0.6B at Q4_0 with an F16 cache:
//   - Launches in the graph: 21 a layer — the two norms, rope, attention and
//     its combine, and 4 for each of the 4 products — 588, with the gather,
//     the final norm and the head 591. Each is a bind group and a 256-byte
//     constants slot, rope's 3: 647 slots, 162 KiB, written once at load
//     (kernels/program.h). 28 pipelines: the gather, 3 norm variants, rope,
//     attention and its combine, 4 forms of each of the 4 products, 4 more
//     for the Q4_1 down projections, and the head.
//   - Dispatched a step: 8 a layer, and the gather, final norm and head: 227
//     (kernel-fusions.md); 255 once a decode step's keys span two 256-key
//     chunks and every layer's combine runs. The program walks all 591
//     launches, a constant-time workgroups_for each, and calls out of the
//     module 14 times and 3 a dispatch — a bind group, the dispatch, and a
//     pipeline, which changes at every dispatch since no two consecutive
//     launches share one: 695 calls a step, 779 when split. At about 1.5 µs
//     a dispatch (kernels/interface.h), 0.34 to 0.38 ms of GPU time.
//   - Bytes, a decode step at position p: the weights once, 376 MB
//     (kernels/matmul/matmul.h); the cache, 4 KiB a position a layer — 8
//     key-value heads of 128 F16 keys and values — 112 KiB a position, read
//     by attention; activations under 1 MiB. At 400 GB/s, 0.94 ms + 0.29 µs
//     × p: 2.1 ms at p = 4,096, before the dispatches.
//   - A 512-token prefill step: about 32 ms of the products' arithmetic at
//     the M3 Max's peak (matmul.h), 1.46 ms of norms (norm.h), attention as
//     attention.h counts it; the same 227 dispatches.
// Optimization (practice): 8 dispatches a layer, where a launch an
// operation would be 17 as kernel-fusions.md counts them — 9 fewer a layer,
// 252 a Qwen3 step, about 0.38 ms (GPU.6).
//
// Levers not taken:
//   - The head skipped on a prefill step that does not end the prompt: its
//     last token's logits are never sampled. It reads the 127.6 MB head,
//     0.32 ms, about 1% of the step's 32 ms or more; skipping it would add a
//     flag to every step's parameters and a token range the head alone
//     reads.
//
// Verification the implementation is held to:
//   - On the CPU, for a description of each listed family's structure —
//     Qwen3's QK-norm and halves pairing; Llama 3.2's rotary factors and
//     adjacent pairing; Gemma 3's post-norms, √(width) embedding scale,
//     sliding-window layers with their own base, and GELU — the launches
//     equal, one for one, what the kernel launchers return for the arguments
//     the order above states; the head on the token embedding where the file
//     has none; Q, K and V without a group as three products; and each
//     refusal naming its layer.
//   - On the GPU, a generated two-layer Qwen3-shaped model with Q4_0
//     weights: the last token's logits the same bits prefilled in one step,
//     in steps that change prefill tile width and attention's split, and
//     decoded a token at a time, at positions either side of a 256-key
//     chunk boundary. The test reads `logits` back through a mapping of its
//     own; the harness reads only the sampler's candidates
//     (sampler/sampler.h).
//   - Against the reference, Qwen3 0.6B at Q4_0, the file pinned by SHA-256
//     and fetched as test data: a pinned prompt of 64 token identifiers
//     decoded a token at a time, and at every position the log-probabilities
//     of llama.cpp's 20 most likely tokens within the largest difference
//     llama.cpp shows between its own CPU and Metal backends on the same file
//     and tokens, and its top token ours wherever its top two are further
//     apart than that. llama.cpp's logits are produced at a pinned commit by
//     tools/reference_logits and kept as a fixture — each position's top 20
//     and log-sum-exp. A pairing swapped from halves to adjacent fails it.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — no GPU object, no I/O; preflight and the
//            program run the same function.
//     E.27   Use error codes systematically — GraphResult, as describe and
//            the plan report theirs.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — 8 dispatches a layer, counted above.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            batch-invariant end to end, as above.
//     MEM.9  Allocate at init, not in steady state — every launch described
//            once, at load; a step allocates nothing for the graph.

enum class GraphError {
    Ok,
    // A layer's gate and up are not one group in the plan. The subject names
    // the layer.
    UngroupedFeedForward,
    // A layer's shape is outside a kernel's preconditions. The subject names
    // the layer and the value.
    UnsupportedShape,
};

struct GraphResult {
    GraphError error = GraphError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == GraphError::Ok; }
};

// Fills `out` with every launch of a step, in the order above.
// Preconditions: `model` is a describe's; `plan` was made from it, as upload
// carried it out; `cache_format` has a pack and is the format the plan sized
// the cache in.
[[nodiscard]] GraphResult build_graph(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                                      const formats::Format& cache_format, std::vector<kernels::Launch>& out);

}  // namespace bllm::graph
