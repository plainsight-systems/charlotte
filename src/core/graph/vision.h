#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::graph {

// Axis A: changes when a vision encoder needs a block the ones here do not
// cover.
//
// The blocks a vision encoder's graph composes (arch/media_encoder.h), as
// encoder.h's are an embedder's. A vision encoder runs one step over an
// image's patches, every patch attending to every other, and ends in rows
// of the text model's width — soft tokens — which the text model's own step
// reads in place of tokens (runtime/vision.h). So no block here touches a
// cache, a vocabulary, the sampler or pooling to one vector. A separate
// builder, not encoder.h's with flags, for encoder.h's reason: its blocks
// begin and end differently and bind different buffers (C.129). What they
// share is the kernels and the residual convention: the norm that begins a
// block adds the last block's result into X (graph.h).
//
// Pure, as the other builders are: no GPU object, no fetch. Preflight runs
// it before any weight is fetched; the program builds from its launches.
//
// The step: N rows, one a patch, row-major over the image's grid of C
// columns, N = C × R; its position is 0 and its token count N, its grid's
// columns in the step's `columns` (kernels/interface.h, below). N is at
// most the plan's rows, the image budget's most tokens × pooling², 2,520 for
// EmbeddingGemma 2 at the card's 280.
//
// The blocks, each appending its launches in order:
//   - patch_embedding(): `pixels`, N rows of 3 P² floats as media/image.h
//     writes them, times the patch embedding (F32, 3 P² to W) into
//     `hidden`: one product, the convolution of stride P as a matrix product
//     over patches (kernels/matmul). F32 weights through the product for the
//     first time: F32's unpack, as every format's (formats/f32/f32.h).
//   - positions(): each patch's two learned positions summed into `output`:
//     row i, at column x = i mod C and row y = i / C, is table[x] +
//     table[positions + y], the column table's row then the row table's —
//     gather's position form (below). Left pending, as a block's output is,
//     with no post-norm: the first layer's attention norm adds it into X.
//     llama.cpp adds the column's row into the patch and then the row's;
//     here the two are summed first and then added, which differs only in
//     rounding.
//   - attention(layer), as encoder.h's, every patch against every other:
//       1. Attention norm: what `output` holds added into `hidden` — the
//          positions before the first layer, the last block's result
//          through its post-norm after — and `hidden` normalized into
//          `normed`.
//       2. Q, K and V from `normed`: one product, the plan's group, into
//          `query`, `key` and `value`.
//       3. Rope's encode form, axial (below): queries and keys normalized
//          per head with their 64-wide gains and rotated, the first half of
//          each head by the patch's column and the second by its row;
//          values normalized per head without a gain. In place.
//       4. Block attention's streaming form (block_attention.h): every
//          patch's query against every patch's key and value, scores never
//          written, into `attention`.
//       5. Output projection: `attention` into `output`; the
//          post-attention norm pending.
//   - gated_feed_forward(layer): graph.h's block, unchanged — the norm
//     adding `output` through the post-attention norm into `hidden` and
//     normalizing, gate and up as one product writing gelu(gate) × up into
//     `activation`, down into `output`; the post-feed-forward norm pending.
//   - pooled_projection():
//       1. The last block's result added: `output` through the
//          post-feed-forward norm into `hidden`, not normalized (the norm's
//          add-only form, as encoder.h's per-layer block uses).
//       2. Pooling over the grid and its norm, one launch (pool's grid
//          form, below): soft token r, at column u and row v of the pooled
//          grid of C / k columns, the mean of the k × k patches
//          (k u .. k u + k − 1, k v .. k v + k − 1), then RMS-normalized
//          without a gain, into row r of `pooled`. llama.cpp scales the mean
//          by √W before that norm; RMSNorm(c y) is RMSNorm(y) with ε / c² in
//          place of ε, so the scale is folded into the norm's ε, ε / W,
//          exact in f32 only for W a power of two, so ε / 768 rounds once:
//          one rounding of ε, not one of every component (principle 4,
//          logical-overview.md; encoder.h folds its 1 / √W the same way).
//       3. The projection: `pooled` times the projection (W to the text
//          width, 512) into `soft_tokens`: one product. Its rows are the
//          rows the text step reads (runtime/vision.h).
//     The pooled grid has M = N / k² rows, and steps 2 and 3 run over those
//     rows alone: each launch's geometry carries a row divisor, k², so it
//     covers N / k² rows, and its kernel bounds its rows by the same
//     division (kernels/interface.h, below). The add in step 1 covers all
//     N.
//
// Names, for a profile: "patch.embed", "patch.positions"; per layer
// "attention.norm", "attention.qkv", "attention.rope", "attention.stream",
// "attention.output", "ffn.norm", "ffn.gate_up", "ffn.down"; and
// "pool.add", "pool.grid", "pool.project".
//
// Shapes the builder checks, refusing the layer and value otherwise: a head
// dimension of 64 (block attention's streaming form and rope's axial form);
// query heads equal to key-value heads; QK-norm on both; a width and a
// feed-forward width each at most 4,096 and a whole number of 32-weight
// groups; 3 P² a whole number of 32-weight groups (768); a projection width
// of at most 4,096; the plan's rows a multiple of k²; positions at least
// the most columns and rows a grid of the budget can have — a 1 × 2,520
// strip has 2,520 columns, well under 10,240. EmbeddingGemma 2's: 16 layers
// of 12 heads, d = 64, W = 768, F = 3,072, P = 16, k = 3, projection 512 —
// inside each.
//
// Determinism (GDSA.2): run to run on the target — every kernel adds in an
// order fixed by its shape and the row, the streaming attention's keys in
// order. A vision step has no batch: one image, so batch invariance does
// not arise, and an image's rows do not depend on any other image.
//
// The working set (vision_working_set below), at f32, R the plan's rows,
// 2,520: `pixels` R × 3 P², 7.7 MB, which the program writes each image;
// `hidden`, `normed`, `output`, `query`, `key`, `value` and `attention`, R ×
// W, 7.7 MB each; `activation` R × F, 31.0 MB; `soft_tokens` the pooled
// rows R / k² × the projection's width, 573 KB; `range_sums` one float, so no product
// splits along K: none over N patches would, and the projection over M
// rows, which the gate would split at 192 or fewer, is at most 7.5 × 10⁷
// multiply-adds, where splitting would save tens of microseconds for 12.6
// MB of range sums; the buffer's capacity keeps it tiled
// (kernels/matmul/matmul.h);
// `pooled` is `normed`, below. About 94 MB, beside the encoder's 226 MB of
// weights; no cache. The plan offers R rows, the working set's step rows
// (residency/plan.h).
//
// What a step costs, counted, for EmbeddingGemma 2's vision encoder at Q8_0
// and N = 2,520 patches, M = 280:
//   - Launches: the patch embedding and the positions, 2; 8 a layer, 128;
//     the pooled projection's 3: 133, about 0.2 ms at 1.5 µs each
//     (kernels/interface.h). Compiled: each product's decode form and three
//     prefill tiles beside the forms that run, as graph.h's.
//   - Multiply-adds: the patch embedding 1.5 × 10⁹; the layers' products
//     9.44 × 10⁶ a patch a layer — QKV 1,769,472, output 589,824, gate and
//     up 4,718,592, down 2,359,296 — 3.81 × 10¹¹; attention 2 × 12 × N² ×
//     64 a layer, 1.56 × 10¹¹ (block_attention.h); the projection over M
//     rows, 1.1 × 10⁸. 5.4 × 10¹¹ in all: about 0.37 s at the 2.9 TFLOPS
//     Qwen3's 951-token prefill reached on the target, an estimate from a
//     measured rate; attention's rate is its own kernel's, not yet
//     measured.
//   - Bytes: the layers' 160 MB of Q8_0 weights read once a 32-patch tile,
//     79 tiles — from the GPU's caches at best, as matmul.h counts a prefill
//     step; the norms, W × N × 4 bytes a row's read and write, 7.7 MB, 33
//     launches; the activation 31 MB written and read a layer; the pixels
//     7.7 MB written once by the CPU.
// Optimization (practice): the positions summed by one gather into the
// block output the first norm adds anyway, so they cost no pass over X of
// their own and no launch to add them (GDSA.6).
// Optimization (practice): pooling and the norm after it in one launch, a
// workgroup a pooled row: the mean never written and read back, and the √W
// scale folded into ε, so no pass multiplies by it (GDSA.6, GPU.6).
// Optimization (practice): `pooled` is `normed`, whose last reader is the
// last layer's feed-forward and whose rows the pooling writes after it, so
// one 7.7 MB buffer serves both; their lifetimes do not overlap (GPU.9).
// Optimization (practice): the pooling and the projection after it over
// the M pooled rows, not the step's N patches: a row divisor in the
// launch's geometry, where running them over N would compute 8 rows in 9
// that nothing reads — 8.8 × 10⁸ multiply-adds, about 0.6 ms an image
// (GDSA.6).
//
// What it asks of the other contracts:
//   - kernels/interface.h, contract 7: a launch's geometry gains a row
//     divisor, 1 by default: it covers the step's tokens over the divisor
//     — rows, tiles and token ranges all counted in those rows — and its
//     kernel takes the same divisor as an override and bounds its rows by
//     tokens / divisor. The step's tokens are a multiple of every divisor
//     its launches carry, which a vision step's N = M k² is. And the step's
//     head grows from 48 bytes
//     to 64, its padding word becoming `columns`, followed by `media` and
//     `media_token` (kernels/interface.h) and two words of padding; a step's
//     `tokens` may exceed kPrefillBlock where it writes no identifiers — a
//     vision step's patches, up to its plan's rows — and identifiers stay
//     at most kPrefillBlock. A decode step writes 64 bytes where it wrote
//     48.
//   - kernels/gather/gather.h: a position form, override `positions`: no
//     identifier read; row i writes table[i mod columns] + table[positions
//     + i / columns], two F32 groups read and summed a 32-weight group, into
//     the launch's output buffer. Bindings as the token form's, the table
//     one piece (62.9 MB, under a binding).
//   - kernels/rope/rope.h: the encode form's axial variant, override
//     `axial`: invocation i of a head holds dimensions 4j .. 4j + 3 and 4j +
//     d / 4 .. 4j + d / 4 + 3 of half h, h = i / (d / 16), j = i mod (d /
//     16) — the half's NEOX pairs — and turns them by the patch's column
//     for h = 0 and its row for h = 1: x = row mod columns, y = row /
//     columns, from the step. Each half's frequencies are the half's, θ_k =
//     base^(−2k / (d / 2)) for k < d / 4, so the angle tables hold d / 4
//     pairs an axis, d / 2 in all, the size the 1-D form's are. With the
//     value norm and QK-norm as the encode form has them.
//   - kernels/pool/pool.h: a `grid` entry point, its rows M by the
//     divisor: one workgroup of 256 a pooled row; its k² source rows read in order, their
//     mean an invocation's vec4s, then norm.h's reduction and an RMS norm
//     without a gain at the launch's ε. Constants: the grid's pooled
//     columns come from the step (columns / k); k, the width and ε the
//     launch's.
//   - kernels/norm/norm.h: nothing new — the add-only form is encoder.h's.
//   - kernels/matmul/matmul.h: F32 weights, which its unpack composition
//     already covers; and the row divisor, its tiles and token ranges taken
//     over the launch's rows — the projection over M. The GPU
//     test adds both, the divided product's bits equal to the same product
//     over M rows undivided.
//
// Verification the implementation is held to:
//   - On the CPU, EmbeddingGemma 2's vision graph over its mmproj's pinned
//     header (tests/vision_test.cpp): 133 launches at grids of 3 × 3, 6 × 9
//     and 42 × 60, none past 65,535 workgroups; a layer's 8 in the order
//     above; ε / W given to the pooling; the working set and `pooled`
//     bound to `normed`'s range.
//   - Each kernel's new form against f64 on the GPU, as its header states;
//     the whole against llama.cpp (runtime/vision.h).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.129  Distinguish implementation inheritance from interface
//            inheritance — a builder of its own that composes the shared
//            kernels.
//     I.4    Make interfaces precisely and strongly typed — layers by index,
//            buffers by purpose.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — 8 launches a layer; the fusions above.
//     GDSA.6 Count passes over global memory — the bytes above.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            run to run.
//     GPU.9  Suballocate and alias transient memory — `pooled` in `normed`.

// The working buffers a vision encoder's step binds, for the residency plan
// to place: the working set above, R rows, and no cache.
[[nodiscard]] residency::WorkingSet vision_working_set(const model::ModelDescription& model, std::uint32_t rows);

// Appends a vision step's launches to `out`, block by block, in the order a
// vision encoder's graph calls them. Preconditions: `model` is a vision
// encoder's describe's (model_description.h's `vision`); `plan` was made
// from it with vision_working_set, as upload carried it out.
// patch_embedding then positions are called first, in that order;
// pooled_projection last; `out` outlives the builder.
class VisionBuilder {
public:
    VisionBuilder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                  std::vector<kernels::Launch>& out);

    [[nodiscard]] GraphResult patch_embedding();
    [[nodiscard]] GraphResult positions();
    [[nodiscard]] GraphResult attention(std::uint32_t layer);
    [[nodiscard]] GraphResult gated_feed_forward(std::uint32_t layer);
    [[nodiscard]] GraphResult pooled_projection();

private:
    // What `output` holds when the next norm adds it: whether a block has
    // written it, and its post-norm, if any.
    struct Pending {
        bool written = false;
        const residency::WeightView* post_gain = nullptr;
    };

    struct Buffers {
        residency::BufferRange pixels, hidden, normed, query, key, value, attention, output, activation,
            soft_tokens, range_sums;
    };

    [[nodiscard]] const residency::WeightView& view(gguf::TensorId tensor) const;
    [[nodiscard]] const residency::WeightView* role(std::uint32_t layer, model::Role role) const;
    [[nodiscard]] const residency::PlannedGroup* group(std::initializer_list<std::optional<gguf::TensorId>> members) const;
    void product(const residency::WeightView& weight, const residency::BufferRange& input,
                 const residency::BufferRange& output);
    [[nodiscard]] kernels::Launch norm(const residency::WeightView& gain, bool normalize);

    const model::ModelDescription* model_;
    const residency::ResidencyPlan* plan_;
    std::vector<kernels::Launch>* out_;
    std::vector<const residency::WeightView*> views_;
    Buffers buffers_;
    Pending pending_;
};

}  // namespace bllm::graph
