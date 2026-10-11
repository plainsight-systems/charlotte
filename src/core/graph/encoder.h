#pragma once

#include <cstdint>
#include <vector>

#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::graph {

// Axis A: changes when an encoding architecture needs a block the ones here
// do not cover.
//
// The blocks an encoder's graph composes (arch/architecture.h's EncodeFn),
// as graph.h's are a decoder's. An encoder runs one step over a whole text,
// every token attending to every other, and ends in one pooled, unit-length
// vector, not in logits: so no block here touches a KV cache, the sampler or
// a decode regime, and its attention is its own kernel
// (kernels/block_attention/block_attention.h). What the two share is the
// kernels — gather, norm, rope, the matrix products — and the residual
// convention: the norm that begins a block adds the last block's result
// into X, so X is never written by a matrix product (graph.h). A separate
// builder, not graph.h's with flags: its blocks end differently, bind
// different buffers and check different shapes, and a builder serving both
// would change whenever either did (C.129).
//
// Pure, as graph.h's builder is: no GPU object, no fetch. Preflight runs it
// before any weight is fetched; the program builds from its launches.
//
// The blocks, each appending its launches in order, for a step of N tokens
// from position 0:
//   - embed(scale): the token embedding's gather into `hidden`, each weight
//     times `scale` (kernels/gather/gather.h) — √512 for EmbeddingGemma 2,
//     as llama.cpp's build_inp_embd(tok_embd, sqrtf(n_embd)).
//   - per_layer_inputs(): the input every layer's per-layer block gates by
//     (below), all layers' at once, from the scaled embedding X₀ while
//     `hidden` still holds it — before the first layer, whose first norm
//     does not add. Two launches:
//       1. The projection: `hidden` (N × W) times per_layer_model_proj (W
//          to L × P, L layers, P the per-layer width) into
//          `per_layer_raw`, N × L × P: one product (kernels/matmul).
//       2. Its norm: each P-wide slice of each token normalized on its own,
//          with per_layer_proj_norm's gain, into `per_layer` — one norm
//          launch over N × L rows (kernels/norm, rows_per_token L, below).
//          llama.cpp scales the projection by 1 / √W before this norm;
//          here the scale is folded into the norm's ε: RMSNorm(c y) =
//          RMSNorm(y) with ε / c² in place of ε, for c = 1 / √W — ε × W,
//          exact in f32 for W a power of two — so no launch or pass
//          multiplies by it (principle 4, logical-overview.md).
//     Layer l's input is then columns l × P .. l × P + P − 1 of each row of
//     `per_layer`: its slice, read in place.
//   - attention(layer): bidirectional, against the step's own keys and
//     values:
//       1. Attention norm: `output` added into `hidden` — through its
//          post-norm, and then X multiplied by the last layer's output scale,
//          where the last block set one (below) — and `hidden` normalized
//          into `normed`. The first layer's adds nothing.
//       2. Q, K and V from `normed`: one product, the plan's group, into
//          `query`, `key` and `value`.
//       3. Rope's encode form (below): queries and keys normalized per head
//          with their gains and rotated by position, in place; values
//          normalized per head without a gain, in place, where the model
//          says (model_description.h's value_norm). Nothing is written to a
//          cache: an encoder has none.
//       4. Block attention: `query` against `key` and `value`, every
//          token's keys within the layer's window either side, into
//          `attention` — its scores launch and its weighted-sum launch
//          (block_attention.h).
//       5. Output projection: `attention` into `output`; the post-attention
//          norm pending.
//   - gated_feed_forward(layer): graph.h's block, unchanged: the norm adding
//     `output` into `hidden` and normalizing, gate and up as one product
//     writing activation(gate) × up into `activation`, down into `output`;
//     the post-feed-forward norm pending.
//   - per_layer_block(layer), Gemma 4's per-layer embedding, as llama.cpp's
//     gemma-embedding2 graph builds it:
//       1. The pending block added: `output` through the post-feed-forward
//          norm into `hidden`, which is not normalized — the gate reads X
//          itself (norm's add-only form, below).
//       2. The gate: `hidden` times the layer's inp_gate (W to P) into
//          `activation`, each output GELU's tanh form of the product times
//          the layer's per-layer input at its column — gelu(W x) ⊙ u — one
//          product whose epilogue reads `per_layer` (matmul's GatedByInput,
//          below).
//       3. The projection: `activation` times the layer's proj (P to W)
//          into `output`. Pending: its post_norm, then the layer's output
//          scale, a scalar, multiplying all of X once the post-normed
//          projection is added — which the next block's norm does in its
//          add (step 1 of attention, and output's).
//   - pooled_output(): the final norm over every token — `output` added
//     with its post-norm and output scale, `hidden` normalized with the
//     output norm into `normed`, every row, not the last alone; then the
//     mean of those N rows (kernels/pool), the output projection (W to D,
//     D the embedding's length) as the head runs, over one row; and the
//     vector made unit length into `embedding`, the range the program reads
//     back (runtime/embedder.h). Pooling before projecting is the
//     projection's linearity: mean(P x_i) = P mean(x_i), so the product
//     runs over one row, not N — what llama.cpp's graph notes and
//     sentence-transformers computes in the other order; the two differ only
//     in rounding.
//
// The mean is written at row N − 1 of `pooled`, where a launch over the last
// token reads its input (kernels/interface.h, Rows::LastToken): so the
// projection is a matrix product over the last token, exactly as a decoder's
// head is, reading row N − 1 and writing row 0 of `projected`; and the unit
// launch reads that row.
//
// Names, as graph.h's, for a profile: "embed"; "per_layer.inputs",
// "per_layer.inputs_norm"; per layer "attention.norm", "attention.qkv",
// "attention.rope", "attention.scores", "attention.weigh",
// "attention.output", "ffn.norm", "ffn.gate_up", "ffn.down",
// "per_layer.add", "per_layer.gate", "per_layer.project"; and
// "output.norm", "output.pool", "output.project", "output.unit".
//
// Shapes: the encoder checks its own, as graph.h's blocks check theirs — a
// head dimension of 64, 128, 256 or 512 (rope, below); (query heads + 2 ×
// key-value heads) × head dimension / 8 a multiple of 64; query heads a
// whole number of key-value heads, and that group dividing 32
// (block_attention.h); QK-norm on both or neither; a hidden width, a
// per-layer width and the embedding's length each at most 4,096 and a whole
// number of 32-weight groups; N at most kPrefillBlock. A layer outside them
// is refused, naming the layer and the value. EmbeddingGemma 2's: 20
// sliding layers of 4 query heads, 2 key-value heads, d = 256, window 1,024
// (512 either side); 4 global layers of 4 and 1, d = 512; W = P = 512, L =
// 24, D = 768 — inside each.
//
// Determinism (GDSA.2): run to run on the target — every kernel the blocks
// launch adds in an order fixed by its shape and the token's position,
// never by tiling or workgroup scheduling, and block attention's maxima are
// exact. An encode is one step, so batch invariance across steps, which a
// decoder needs, does not arise; a text's vector does not depend on any
// other text, since nothing is batched.
//
// What it asks of the other contracts:
//   - kernels/rope/rope.h: an `encode` entry point — queries and keys
//     normalized with their gains and rotated in place, values normalized
//     without one where `value_norm` is set, nothing packed into a cache;
//     bindings 2 the query gain, 3 the key gain, 4 the factors or the key
//     gain in their place, 5 queries, 6 keys and 7 values, read and written.
//     And a head dimension of 512: d / 8 = 64 invocations a head, one
//     workgroup; its norm's tree 6 levels; 256 angle pairs, 4 an
//     invocation, 2 KiB of workgroup memory beside the 256 bytes of partial
//     sums. Its constants' `turns` grow from 32 vec4s to 64 — 1,040 bytes,
//     5 slots of 256 where they took 3 — for every rope launch: for Qwen3,
//     28 launches × 512 bytes, 14 KiB more of constants written once at
//     load, and nothing a step.
//   - kernels/interface.h: kMaxLaunchConstants from 1,024 to 1,280, for
//     rope's 1,040.
//   - kernels/norm/norm.h: three variants, override constants, as its
//     `add` and `post_norm` are:
//       `normalize` (default true): false writes X after the add and nothing
//         else — per_layer_block's step 1;
//       `scaled`: X multiplied by a scalar after the add and before the
//         reduction, the scalar an F32 view of one float at binding 7 — the
//         layer output scale;
//       `rows_per_token` (default 1): a launch covers N × rows_per_token rows
//         of `width`, row r of token r / rows_per_token — the per-layer
//         inputs' norm, 24 rows of 512 a token.
//     Each changes one launch's form; the decoder's launches set none and
//     are unchanged.
//   - kernels/matmul/matmul.h: an epilogue, GatedByInput — activation(y) ×
//     u[t × u_stride + u_offset + o], `u` a working buffer bound read-only
//     at the binding after the output, `u_stride` and `u_offset` in the
//     launch's constants — in both forms, as every epilogue has. Five
//     bindings, as QKV's. And BF16 weights, through its format's unpack
//     (formats/bf16/bf16.h), as any format: no change to the kernel.
//   - residency/plan.h: it places a working set it is given, and no longer
//     lists one. Which working buffers a graph binds, and how large, is the
//     graph's knowledge (axis A), and where they lie the planner's (axis D);
//     the list in plan.cpp — every kernel's buffers, the attention
//     partials and selection's among them — is A and E knowledge in a D
//     file, and an encoder would add a second list keyed on the model's
//     kind. So plan.h states the shape of a need, ScratchNeed {purpose,
//     rows, width}, and a WorkingSet — the needs, in order, and whether the
//     graph keeps a KV cache — and plan_residency takes one. Each builder
//     declares its own: graph.h's decoder_working_set(model), its list
//     moved unchanged, and this file's encoder_working_set(model). The
//     caller, preflight's plan_load, picks by the architecture's graph kind
//     (arch/architecture.h). plan.h never includes a graph header, so the
//     dependency still points one way.
//     The move is its own change, first, intending no behavior change: the
//     same buffers in the same order and the same bytes for each listed
//     model, which the plan's tests hold. The encoder's working set:
//     no KV cache, so the context offered is the smaller of kPrefillBlock
//     and the trained context; and these working buffers, each
//     kPrefillBlock rows at f32 — `hidden`, `normed`, `output` W wide;
//     `query` and `attention` the widest layer's query heads × d, 2,048;
//     `key` and `value` its key-value heads × d, 512; `activation` the
//     feed-forward width, 2,048, which the gate reuses; `per_layer_raw` and
//     `per_layer`, L × P, 12,288 wide, 24 MiB each; `scores` and
//     `score_maxima` (block_attention.h), 4 MiB and 64 KiB at the widest
//     layer's query heads; `pooled`, W wide; and `projected` and
//     `embedding`, one row of D. About 70 MiB for EmbeddingGemma 2, beside
//     its 294 MB of weights: 364 MB, under the 2 GiB default budget and a
//     phone's 512 MiB (web/platform_policy.js).
//
// What a step costs, counted, for EmbeddingGemma 2 at Q8_0 and N tokens:
//   - Launches dispatched: the gather's 2 — the table's 142.6 MB is two
//     pieces at a 128 MiB binding — the per-layer inputs' 2; 12 a layer,
//     288; and the output block's 4: 296, about 0.44 ms at 1.5 µs each.
//     Compiled: each product's decode form and three prefill tiles beside
//     the forms that run, as graph.h's.
//   - Bytes: each 32-token tile of a product reads its weight once — the
//     layers' 138.1 MB and the per-layer projection's 12.6 MB of BF16 —
//     from device memory once for N up to 32, 0.38 ms at 400 GB/s; at N =
//     512, 16 tiles, 2.4 GB from the GPU's caches at best and device
//     memory at worst, as matmul.h counts a prefill step. The per-layer
//     inputs: 24 MiB written and read twice, 72 MiB, about 0.2 ms at 400
//     GB/s; their slices read again by the 24 gates, 24 MiB. Attention's
//     scores: 4 MiB a layer written and read at N = 512 (block_attention.h).
//     The norms: W × N × 4 bytes a row's read and write, about 4 MiB a
//     launch at N = 512, 73 launches.
//   - Multiply-adds a token: a sliding layer's products 5,242,880 — QKV
//     1,048,576, output 524,288, gate and up 2,097,152, down 1,048,576, the
//     per-layer gate and projection 524,288 — a global layer's 6,291,456,
//     the per-layer inputs' projection 6,291,456: 1.36 × 10⁸; attention
//     57,344 N² for the step (block_attention.h).
// Optimization (practice): every layer's per-layer input from one product
// and one norm, before the first layer — 2 launches where a product and a
// norm a layer would be 48, and the projection read once, not as 24 slices
// of a weight split by rows (GPU.6, CDSA.32's once-at-load reasoning applied
// once a step).
// Optimization (practice): the 1 / √W scale folded into the norm's ε, so
// no pass multiplies the 24 MiB by it (GDSA.6).
// Optimization (practice): the gate's multiplication by the per-layer input
// rides in its product's epilogue, so the gelu output is never written and
// read back to be multiplied, 1 MiB a layer at N = 512 (GDSA.6).
// Optimization (practice): pooling before the output projection, one row's
// product where N rows' would be 393,216 × N multiply-adds and one more
// launch's worth of rows written (GDSA.6).
// Optimization (practice): the layer output scale rides in the next norm's
// add, not a launch of its own: 24 launches and 24 passes over X saved
// (GPU.6).
//
// Verification the implementation is held to:
//   - On the CPU, EmbeddingGemma 2's encoder graph over a description of its
//     structure: the launches equal, one for one, what the kernel launchers
//     return for the arguments the blocks above state; the per-layer slice
//     each gate reads; the ε the per-layer norm is given, W times the file's;
//     every pending post-norm and scale handed to the norm that adds it; and
//     each shape refusal naming its layer.
//   - On the GPU, a generated two-layer model of its shape, one sliding
//     layer and one global, with Q8_0 weights and a BF16 projection: each
//     working buffer after each block against an f64 reference of the same
//     arithmetic over the same decoded weights, within the margins each
//     kernel states; and the output against runtime/embedder.h's
//     llama.cpp reference for the real file.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.129  Distinguish implementation inheritance from interface
//            inheritance — the encoder's builder composes the shared
//            kernels; it does not extend graph.h's builder.
//     I.4    Make interfaces precisely and strongly typed — every block takes
//            a layer index and reads the description; buffers by purpose.
//     C.12   Don't make data members const or references in a copyable type —
//            the builder holds pointers, as graph.h's does.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — 12 dispatches a layer, and the fusions
//            above.
//     GDSA.6 Count passes over global memory — the bytes above.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            run to run.
//     CDSA.32 Transform static data once into the layout its consumer reads —
//            the per-layer inputs computed once and read in place by slice.

// Appends an encode step's launches to `out`, block by block, in the order an
// encoding architecture's graph calls them. Preconditions: `model` is a
// describe's and has an encoder (model_description.h); `plan` was made from
// it, as upload carried it out. embed is called once, first; per_layer_inputs
// once, after it, where the model has per-layer inputs; pooled_output once,
// last; `out` outlives the builder.
class EncoderBuilder {
public:
    EncoderBuilder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                   std::vector<kernels::Launch>& out);

    void embed(float scale);
    [[nodiscard]] GraphResult per_layer_inputs();
    [[nodiscard]] GraphResult attention(std::uint32_t layer);
    [[nodiscard]] GraphResult gated_feed_forward(std::uint32_t layer);
    [[nodiscard]] GraphResult per_layer_block(std::uint32_t layer);
    [[nodiscard]] GraphResult pooled_output();

private:
    // What `output` holds when the next norm adds it: whether a block has
    // written it, its post-norm, and the scale X takes after the add.
    struct Pending {
        bool written = false;
        const residency::WeightView* post_gain = nullptr;
        const residency::WeightView* scale = nullptr;
    };

    struct Buffers {
        residency::BufferRange hidden, normed, query, key, value, attention, output, activation, per_layer_raw,
            per_layer, scores, score_maxima, pooled, projected, embedding;
    };

    const model::ModelDescription* model_;
    const residency::ResidencyPlan* plan_;
    std::vector<kernels::Launch>* out_;
    std::vector<const residency::WeightView*> views_;   // by tensor id; null for a tensor the plan did not place
    Buffers buffers_;
    Pending pending_;
};

}  // namespace bllm::graph
