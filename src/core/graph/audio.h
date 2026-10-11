#pragma once

#include <cstdint>
#include <vector>

#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::graph {

// Axis A: changes when an audio encoder needs a block the ones here do not
// cover.
//
// The blocks an audio encoder's graph composes (arch/media_encoder.h), as
// vision.h's are an image encoder's. An audio encoder runs one step over a
// clip's spectrogram and ends in rows of the text model's width — soft
// tokens — which the text model's own step reads in place of tokens
// (runtime/audio.h). A builder of its own, for vision.h's reason: a
// Conformer's blocks begin and end differently from a transformer's and bind
// other buffers (C.129). What they share is the kernels and the residual
// convention: the norm that begins a block adds the last block's result into
// X (graph.h).
//
// Pure, as the other builders are: no GPU object, no fetch.
//
// The step: R rows, the clip's subsampled frames, R = ⌈⌈T / 2⌉ / 2⌉ for T
// spectrogram frames (media/audio.h); its position 0, its token count R,
// and T in the step's `columns` (kernels/interface.h). R is at most the
// plan's rows, the most a clip may make beside the text step's ends and
// markers: 508 at EmbeddingGemma 2's context of 512.
//
// The blocks, each appending its launches in order. Every product below that
// the description gives bounds clamps its input and its output to them
// (kernels/matmul/matmul.h's clipped form); none of the others does.
//   - subsample():
//       1. The first convolution: `mel`, T frames of 128 bins, into
//          `subsampled`, ⌈T / 2⌉ × 64 positions × 128 channels, each
//          position's channels layer-normalized with the first gain and
//          through ReLU (kernels/subsample, its `first` form).
//       2. The second: `subsampled` into `normed`, R rows of 32 positions ×
//          32 channels, flattened position then channel to 1,024, normed and
//          through ReLU likewise (its `second` form).
//       3. The input projection, F32, 1,024 to W: `normed` into `hidden`. One
//          product.
//   - feed_forward(layer, half), half 0 or 1:
//       1. Its norm: the pending result added into `hidden` as the block
//          before left it (below), `hidden` normalized with ffn_norm, or
//          ffn_norm_1, into `normed`. Layer 0's first adds nothing: the
//          input projection wrote X.
//       2. Up, clamped, through SiLU (matmul's activated Write): `normed`
//          into `activation`, R × F.
//       3. Down, clamped: `activation` into `output`. Pending: through
//          ffn_post_norm, or ffn_post_norm_1, times the residual weight, 0.5.
//   - attention(layer):
//       1. Its norm: pending added, normalized with attn_pre_norm.
//       2. Q, K and V, clamped, three products from `normed` into `query`,
//          `key` and `value`. Not the fused QKV product: each has its own
//          bounds, which a fused launch's constants have no room for, and
//          `normed` read three times is 6 MB at 508 rows.
//       3. Local attention (kernels/local_attention): each row's query,
//          scaled, against its own and its 11 earlier rows' keys, plus this
//          layer's relative table's term for each distance, soft-capped,
//          into `attention`.
//       4. The output projection, clamped: `attention` into `output`.
//          Pending: through attn_post_norm, weight 1.
//   - convolution(layer):
//       1. Its norm: pending added, normalized with conv_norm.
//       2. The first pointwise projection, clamped, its GLU in its gated
//          epilogue — the weight's halves its two members, σ(b) × a
//          (kernels/matmul/matmul.h): `normed` into `attention`, R × W,
//          free since the output projection read it.
//       3. The light convolution (kernels/light_conv): the causal depthwise
//          convolution of 5 rows, the norm_conv norm and SiLU, one launch:
//          `attention` into `query`, free since local attention read it.
//       4. The second pointwise projection, clamped: `query` into
//          `output`. Pending: added as it is, no post-norm, weight 1.
//   - The layer's end, in the next block's norm: the second feed-forward's
//     result added through ffn_post_norm_1 times 0.5, then X replaced by its
//     own norm with ln2, written back, then normalized again with the next
//     layer's ffn_norm into `normed` — norm.h's re-norm form, one launch.
//   - output():
//       1. The last layer's end: the add and the re-norm with ln2, X
//          written, nothing normalized after (norm.h's add-only form with
//          re-norm).
//       2. The output projection with its bias: `hidden` into `projected`,
//          R × 1,536 (matmul's biased Write).
//       3. An RMS norm without a gain: `projected` into `activation`, as
//          rows of 1,536 floats (norm.h's unit gain, width 1,536).
//       4. The multimodal projection, 1,536 to 512: `activation` into
//          `soft_tokens`. Its rows are the rows the text step reads.
//   - setup(), the encoder's setup graph, a step of 13 rows run once at
//     load: for each layer, `positions` — the 13 sinusoidal rows, distances
//     12 down to 0, which the audio stage writes — times attn_k_rel into
//     layer l's 13 rows of `relative`. 12 products. The step's local
//     attention reads them, and nothing else writes them.
//
// Names, for a profile: "subsample.first", "subsample.second",
// "subsample.project"; per layer "ffn.norm", "ffn.up", "ffn.down",
// "attention.norm", "attention.q", "attention.k", "attention.v",
// "attention.local", "attention.output", "conv.norm", "conv.in",
// "conv.light", "conv.out", "ffn2.norm", "ffn2.up", "ffn2.down"; and
// "out.renorm", "out.project", "out.norm", "out.embed"; the setup's
// "relative.project".
//
// Shapes the builder checks, refusing the layer and value otherwise: a head
// dimension of 128 (local attention's form); a window of 12 and 13 relative
// positions; a depthwise kernel of 5; the subsampling's 128 mel bins and its
// channels, 128 and 32, as the subsample kernel's forms take them; W, F,
// 2W, 1,536 and 512 whole numbers of 32-weight groups and at most 4,096. An
// encoder of other numbers is refused here, not computed wrongly.
//
// Determinism (GDSA.2): run to run on the target — every kernel adds in an
// order fixed by its shape and the row. A clip has no batch.
//
// The working set (audio_working_set below), at f32, R the plan's rows,
// 508, and T = 4R, 2,032: `mel` T × 128, 1.0 MB, which the program writes
// each clip; `subsampled` 2R × 64 × 128, 33.3 MB; `hidden`, `normed`,
// `output`, `query`, `key`, `value` and `attention`, R × W, 2.1 MB each;
// `activation` R × F, 8.3 MB; `projected` R ×
// 1,536, 3.1 MB; `soft_tokens` R × 512, 1.0 MB; `positions` 13 × W, 53 KB;
// `relative` L × 13 × W, 639 KB; `range_sums` as the gate asks for the step's
// products (kernels/matmul/matmul.h), the most of theirs — a short clip's
// products have few tiles, and split along K as the text encoder's do. About
// 63 MB beside the range sums and the encoder's 328 MB of weights; no cache.
//
// What a step costs, counted, for EmbeddingGemma 2's audio encoder at Q8_0
// and a 20 s clip, T = 1,999 and R = 500:
//   - Launches: the subsampling's 3; 16 a layer, 192; the output's 4: 199,
//     about 0.3 ms at 1.5 µs each (kernels/interface.h), and a combine for
//     each product the gate splits. The setup: 12, once.
//   - Multiply-adds: 1.48 × 10¹¹ (the research note's table), almost all of
//     it the layers' products, 24.1 × 10⁶ a row a layer; attention 3.8 ×
//     10⁴ a row a layer, since a row sees 12 others. About 0.10 s at the 2.9
//     TFLOPS measured on the target for prefill; llama.cpp on Metal measured
//     62 to 65 ms.
//   - Bytes: the layers' 310 MB of Q8_0 weights read once a 32-row tile, 16
//     tiles, from the GPU's caches at best, as matmul.h counts a prefill
//     step; the first convolution's 33.3 MB written once and read about
//     2.25 times (3 × 3 taps at stride 2); the norms, 2 MB a launch.
// Optimization (practice): the residual weight, the post-norms, the layer's
// output norm and the next norm all in the norm that begins the next block —
// one launch between a block's last product and the next's first, as the
// text encoder's (GDSA.6, GPU.6).
// Optimization (practice): the GLU in the first pointwise projection's
// epilogue, so the product writes R × W where it would write R × 2W, and
// each sigmoid is computed once; then the depthwise convolution, its norm
// and SiLU in one launch, a workgroup a row, where three passes would each
// read and write R × W floats (kernels/light_conv, GDSA.6).
// Optimization (practice): each layer's relative table computed once, at
// load: it depends on the weights alone, so a step reads 13 rows a layer
// where computing them would be 12 more products a step (GDSA.17).
// Optimization (practice): `attention` reused for the GLU's rows, `query`
// for the light convolution's, `normed` for the second convolution's,
// `activation` for the output's norm; their lifetimes do not overlap
// (GPU.9).
// Optimization not taken: the two convolutions fused, so the first's 33 MB
// never reaches memory. A workgroup's second-convolution outputs would
// recompute the first's 3 × 3 neighbourhoods, about 1.5 times its work, and
// its 128-channel layer norms in workgroup memory; the unfused pass costs
// about 0.2 ms of a 100 ms step at 400 GB/s.
//
// What it asks of the other contracts:
//   - kernels/matmul/matmul.h: the clipped form, the biased Write, and an
//     activated Write; F32 weights, which it has.
//   - kernels/norm/norm.h: post_scale, re-norm, unit gain; a width of 1,536.
//   - kernels/subsample, kernels/local_attention, kernels/light_conv: new.
//   - residency/plan.h: product spans (arch/gemma4a).
//   - kernels/interface.h: nothing new — `columns` is the clip's frames.
//
// Verification the implementation is held to:
//   - On the CPU, EmbeddingGemma 2's audio graph over its mmproj's pinned
//     header (tests/audio_test.cpp): 199 launches at R of 1, 75 and 508,
//     none past 65,535 workgroups; a layer's 16 in the order above; every
//     clamped product bound to its span; the setup's 12; the working set and
//     the reused buffers bound to their ranges.
//   - Each kernel's new form against f64 on the GPU, as its header states;
//     the whole against llama.cpp (runtime/audio.h).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.129  Distinguish implementation inheritance from interface
//            inheritance — a builder of its own over the shared kernels.
//     I.4    Make interfaces precisely and strongly typed — layers by index,
//            buffers by purpose.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — 16 launches a layer; the fusions above.
//     GDSA.6 Count passes over global memory — the bytes above.
//     GDSA.2 Declare each floating-point reduction's determinism level —
//            run to run.
//     GDSA.17 Plan once per key, reuse the plan — the relative tables at
//            load.
//     GPU.9  Suballocate and alias transient memory — the reused buffers.

// The working buffers an audio encoder's step and setup bind, for the
// residency plan to place: the working set above, R rows, and no cache.
[[nodiscard]] residency::WorkingSet audio_working_set(const model::ModelDescription& model, std::uint32_t rows);

// Appends an audio step's launches to `out`, block by block, in the order an
// audio encoder's graph calls them: subsample; for each layer
// feed_forward(l, 0), attention(l), convolution(l), feed_forward(l, 1);
// output. Or, alone, setup. Preconditions: `model` is an audio encoder's
// describe's (model_description.h's `audio`); `plan` was made from it with
// audio_working_set, as upload carried it out; `out` outlives the builder.
class AudioBuilder {
public:
    AudioBuilder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                 std::vector<kernels::Launch>& out);

    [[nodiscard]] GraphResult subsample();
    [[nodiscard]] GraphResult feed_forward(std::uint32_t layer, std::uint32_t half);
    [[nodiscard]] GraphResult attention(std::uint32_t layer);
    [[nodiscard]] GraphResult convolution(std::uint32_t layer);
    [[nodiscard]] GraphResult output();
    [[nodiscard]] GraphResult setup();

private:
    // What `output` holds when the next norm adds it: whether a block has
    // written it, its post-norm, if any, its weight, and whether the layer
    // ends before the next norm, so X is re-normed with ln2 first.
    struct Pending {
        bool written = false;
        const residency::WeightView* post_gain = nullptr;
        float weight = 1.0f;
        const residency::WeightView* output_norm = nullptr;
    };

    struct Buffers {
        residency::BufferRange mel, subsampled, hidden, normed, query, key, value, attention, output, activation,
            projected, soft_tokens, positions, relative, range_sums;
    };

    [[nodiscard]] const residency::WeightView& view(gguf::TensorId tensor) const;
    // A product's launches, `input` into `output`: clamped where it has
    // bounds, through the activation where `activated`.
    void product(const model::AudioProduct& p, const residency::BufferRange& input,
                 const residency::BufferRange& output, bool activated = false);
    // The norm that begins a block: the pending result added as `pending_`
    // says, X normalized with `gain` into `normed`, or not normalized.
    [[nodiscard]] kernels::Launch norm(const residency::WeightView& gain, bool normalize);
    // Layer l's 13 rows of `relative`.
    [[nodiscard]] residency::BufferRange relative(std::uint32_t layer) const;

    const model::ModelDescription* model_;
    const residency::ResidencyPlan* plan_;
    std::vector<kernels::Launch>* out_;
    std::vector<const residency::WeightView*> views_;
    Buffers buffers_;
    Pending pending_;
};

}  // namespace bllm::graph
