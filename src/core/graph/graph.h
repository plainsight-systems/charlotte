#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "core/formats/format.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::graph {

// Axis A: changes when an architecture needs a block the ones here do not
// cover — a plain feed-forward block, LayerNorm, biases, learned positions.
//
// The blocks every architecture's graph composes (arch/architecture.h). An
// architecture's graph is the order its family runs them in, and what its
// family does that the model description does not carry, such as Gemma 3's
// embedding scale; the blocks are what families share, so a decoder layer
// is written once and each family's graph is a few lines. A family that
// needs a block not here adds one, and no other family's graph changes
// (docs/architecture/change-axes.md, axis A). Composition, not a base class
// holding the order: a base whose order every family overrides would change
// for each family added (C.129).
//
// Pure, as the kernel launchers are: no GPU object, no fetch. The program
// builds from a graph's launches at load (kernels/program.h), and preflight
// runs the graph before any weight is fetched, a failure naming what stops
// the Run stage (preflight/preflight.h). Views come from the plan as upload
// carried it out (Upload::plan()), where a confirmed duplicate head reads the
// embedding's bytes.
//
// The blocks, each appending its launches in order:
//   - embed(scale): the token embedding's gather, one launch a piece, into
//     `hidden`, each weight times `scale` (kernels/gather/gather.h).
//   - attention(layer):
//       1. Attention norm: `output` added into `hidden` and `hidden`
//          normalized into `normed` with the layer's gain — without the add
//          before any block has written `output`, since the gather wrote
//          `hidden` (kernels/norm/norm.h).
//       2. Q, K and V from `normed`: the plan's group as one product into
//          `query`, `key` and `value`; where the plan made none, three
//          products, each writing its own (kernels/matmul/matmul.h).
//       3. QK-norm where the layer has it, RoPE, and the key and value
//          appended to the layer's cache (kernels/rope/rope.h).
//       4. Attention over the layer's cache into `attention`, and its
//          combine (kernels/attention/attention.h).
//       5. Output projection: `attention` into `output`.
//   - gated_feed_forward(layer):
//       6. Feed-forward norm: `output` added into `hidden`, normalized into
//          `normed`.
//       7. Gate and up from `normed`, the plan's group as one product writing
//          activation(gate) × up into `activation`.
//       8. Down: `activation` into `output`.
//   - output(): the final norm, `output` added into `hidden` and normalized
//     with the output norm's gain; then the head, `normed` into `logits` —
//     the output head's weight, or the token embedding's where the file ties
//     them. Both cover the step's last token only, and run only in a step
//     that asks for logits.
// A matrix product is all its launcher's launches — a decode launch and one
// a prefill tile width, for each piece — and attention its main launch and
// its combine; the program dispatches those a step's token count, position
// and logits select (kernels/interface.h). Which steps share a launch, and
// why, is docs/architecture/kernel-fusions.md.
//
//   - Residual: the norm that begins a block adds the last block's result
//     into X, so X is never written by a matrix product (kernel-fusions.md).
//     The builder carries what `output` holds — whether a block has written
//     it, and the post-norm that block's layer applies to it before the add:
//     its post-attention norm after attention, its post-feed-forward norm
//     after the feed-forward block, where the layer has the weight. Post-norms
//     are detected, never assumed (logical-overview.md, principle 5).
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
//     checked by the blocks, their one caller, and a layer outside them
//     refused naming the layer and the value. A weight's rows a whole number
//     of 32-weight groups, which matrix products and the gather need, is
//     routes' check (residency/routes.h), at preflight's Upload stage.
//   - The cache format is the one the plan sized the cache in, from load
//     policy's precision (policy/policy.h); rope packs with it and attention
//     unpacks with it.
//   - Determinism (GDSA.2): every kernel the blocks launch is
//     batch-invariant — the gather, norms and rope work a row alone,
//     attention folds chunks fixed by position, each matrix product adds in
//     an order fixed by K — and no block adds a reduction of its own. So a
//     token's logits are the same bits whether its prompt was prefilled in
//     one step, in several, or decoded a token at a time, on the target's
//     compiler, as each kernel states.
//
// What it asks of the other contracts:
//   - kernels/interface.h: Step gains `logits`, 1 in a step whose last
//     token's logits are sampled and 0 in a prefill step that does not end
//     the prompt, in a word that was padding, so the step's write is no
//     larger. A launch over the last token alone — the final norm and the
//     head — runs only when it is 1; workgroups_for takes it.
//   - arch/architecture.h: each architecture supplies its graph, composed
//     from these blocks.
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
//     chunks and every layer's combine runs; 225 in a prefill step that does
//     not end the prompt. The program walks the 253 to 255 launches its
//     schedule lists for the step (kernels/schedule.h), a constant-time
//     workgroups_for each, and calls out of the module 14 times and 3 a
//     dispatch — a bind group, the dispatch, and a pipeline,
//     which changes at every dispatch since no two consecutive launches
//     share one: 695 calls a step, 779 when split, 689 without logits. At
//     about 1.5 µs a dispatch (kernels/interface.h), 0.34 to 0.38 ms of GPU
//     time.
//   - Bytes, a decode step at position p: the weights once, 376 MB
//     (kernels/matmul/matmul.h); the cache, 4 KiB a position a layer — 8
//     key-value heads of 128 F16 keys and values — 112 KiB a position, read
//     by attention. From device memory at 400 GB/s, 0.94 ms + 0.29 µs × p:
//     2.1 ms at p = 4,096, before the dispatches. The activations are a
//     working set under 1 MiB, but read many times over: each matrix
//     product's invocations read its whole input again, 15 MiB a layer and
//     148 MiB for the head — 568 MiB of storage reads a step, from the GPU's
//     caches (matmul.h) — and the norms move 1.1 MiB (norm.h). The floor
//     above prices device memory alone; those cached reads are counted, not
//     priced in it.
//   - A 512-token prefill step: about 32 ms of the products' arithmetic at
//     the M3 Max's peak (matmul.h), 1.46 ms of norms (norm.h), attention as
//     attention.h counts it; the same dispatches.
// Optimization (practice): 8 dispatches a layer, where a launch an
// operation would be 17 as kernel-fusions.md counts them — 9 fewer a layer,
// 252 a Qwen3 step, about 0.38 ms (GPU.6).
// Optimization (practice): logits only for a step whose last token is
// sampled, as llama.cpp computes them only for the tokens a batch marks and
// vLLM only for each sequence's last. A prefill step that does not end the
// prompt skips the final norm and the head: the head's 127.6 MB, 0.32 ms
// at 400 GB/s, and two dispatches — 2.2 ms over the 7 such steps of a
// 4,096-token prompt (GDSA.6).
//
// Verification the implementation is held to:
//   - On the CPU, each listed architecture's graph — Qwen3's QK-norm and
//     halves pairing; Llama 3.2's rotary factors and adjacent pairing;
//     Gemma 3's post-norms, √(width) embedding scale, sliding-window layers
//     with their own base, and GELU — over a description of its structure:
//     the launches equal, one for one, what the kernel launchers return for
//     the arguments the blocks above state; the head on the token embedding
//     where the file has none; Q, K and V without a group as three products;
//     and each refusal naming its layer. The final norm and head dispatch no
//     workgroups in a step that asks for no logits.
//   - On the GPU, a generated two-layer Qwen3-shaped model with Q4_0
//     weights: the last token's logits the same bits prefilled in one step,
//     in steps that change prefill tile width and attention's split, and
//     decoded a token at a time, at positions either side of a 256-key
//     chunk boundary. The test reads `logits` back through a mapping of its
//     own; the harness reads only the sampler's drawn token
//     (sampler/sampler.h).
//   - Against the reference, Qwen3 0.6B at Q4_0, the file pinned by SHA-256
//     and fetched as test data: a pinned prompt of 64 token identifiers
//     decoded a token at a time, and at every position the log-probabilities
//     of llama.cpp's 20 most likely tokens within the largest difference
//     llama.cpp shows at that position between its own CPU and Metal
//     backends on the same file and tokens — 0.1 to 1.0 nats, by position —
//     and its top token ours wherever its top two are further apart than
//     that. llama.cpp's logits are produced at a pinned commit by
//     tools/reference_logits and kept as a fixture — each position's top 20
//     and log-sum-exp. A pairing swapped from halves to adjacent fails it.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.129  Distinguish implementation inheritance from interface
//            inheritance — families share blocks by composition; the
//            interface is architecture.h's function pointer.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — 8 dispatches a layer, counted above.
//     GDSA.6 Count passes over global memory — the head's read skipped where
//            its logits are not sampled.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            batch-invariant end to end, as above.
//     WASM.4 Reduce indirect dispatch in hot paths — the graph is chosen
//            and built once, at load; a step makes no call through it.

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

// Appends a step's launches to `out`, block by block, in the order an
// architecture's graph calls them. It holds what `output` holds between
// blocks (above). Preconditions, as architecture.h's GraphFn states: `model`
// is a describe's; `plan` was made from it, as upload carried it out;
// `cache_format` has a pack and is the format the plan sized the cache in.
// embed is called once, first, and output once, last; `out` outlives the
// builder.
class Builder {
public:
    Builder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
            const formats::Format& cache_format, std::vector<kernels::Launch>& out);

    void embed(float scale);
    [[nodiscard]] GraphResult attention(std::uint32_t layer);
    [[nodiscard]] GraphResult gated_feed_forward(std::uint32_t layer);
    void output();

private:
    // What `output` holds when the next norm adds it.
    struct Pending {
        bool written = false;                                 // no block yet: the first norm does not add
        const residency::WeightView* post_gain = nullptr;     // the writer's post-norm, or null
    };

    // The plan's working buffers, by what they hold.
    struct Buffers {
        residency::BufferRange hidden, normed, query, key, value, attention, partials, partial_stats, output,
            activation, logits, sampled;
    };

    [[nodiscard]] const residency::WeightView& view(gguf::TensorId tensor) const;
    // The layer's weight in `role`, or null where it has none.
    [[nodiscard]] const residency::WeightView* role(std::uint32_t layer, model::Role role) const;
    // The plan's group whose members, in order, are `members`, or null.
    [[nodiscard]] const residency::PlannedGroup* group(std::initializer_list<std::optional<gguf::TensorId>> members) const;
    void product(const residency::WeightView& weight, const residency::BufferRange& input,
                 const residency::BufferRange& output, kernels::Rows rows);

    const model::ModelDescription* model_;
    const residency::ResidencyPlan* plan_;
    const formats::Format* cache_format_;
    std::vector<kernels::Launch>* out_;
    std::vector<const residency::WeightView*> views_;   // by tensor id; null for a tensor the plan did not place
    Buffers buffers_;
    Pending pending_;
};

}  // namespace bllm::graph
