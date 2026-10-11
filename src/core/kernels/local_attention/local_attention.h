#pragma once

#include <cstdint>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// An audio encoder's attention (graph/audio.h): each row's query against
// its own key and the keys of the 11 rows before it — a causal window of
// w = 12 — with a relative position term for each distance, the scores
// soft-capped, then a softmax and the values' weighted sum. llama.cpp's
// gemma4a graph computes it blocked: rows in chunks of 12, each chunk's
// queries against a context of 24 keys, its relative term shifted into
// place as Transformer-XL's appendix B shifts it, and a mask that leaves
// exactly the window. This kernel computes the same sums for the window
// alone, with no block, pad or shift: the mask removes every other score
// before the softmax, so nothing outside the window reaches the output.
//
// For row r and head h, d = 128, heads H = 8, the row's columns 128h to
// 128h + 127 of each buffer:
//   q̂[i] = (q[r][i] × q_scale) × s[i], s the layer's per_dim_scale (the
//     converter applied softplus; graph/audio.h), q_scale = (1 / √d) / ln 2;
//   for each distance j from 0 to w − 1 with r − j ≥ 0:
//     k̂[i] = k[r − j][i] × k_scale, k_scale = ln(1 + e) / ln 2;
//     a_j = Σ_i q̂[i] k̂[i];  b_j = Σ_i q̂[i] P[w − j][i];
//     s_j = c tanh((a_j + b_j) / c), c the softcap, 50;
//   p_j = exp(s_j − max s) / Σ exp(s − max s) over those j;
//   out[r][i] = Σ_j p_j v[r − j][i].
// P is the layer's relative table, 13 rows of the width (graph/audio.h's
// setup): row p is attn_k_rel times the sinusoid of position w − p, so the
// key at distance j reads row w − j — the row llama.cpp's shift leaves
// beside that key, as the shift's arithmetic gives (key slot k of a block,
// query q: the shifted index k − q is w − (q + w − k), and q + w − k is the
// distance). Row 0, distance 12, is outside the window and never read. The
// sums a_j and b_j are taken apart and added, as llama.cpp adds its two
// score matrices; each in i order, an explicit fma a term.
//
//   - One workgroup of 128 a row; the launch's rows are the step's R.
//     Invocation n takes head h = n / 16 and lane l = n mod 16.
//     1. Scores: lanes 0 to 11 each take distance l of their head: a_l and
//        b_l, 2 × 128 multiply-adds, reading q's head (512 bytes, every lane
//        of the head the same), k's at row r − l and P's row w − l; then the
//        cap, its argument held to ±10, past which tanh is ±1 in f32: WGSL
//        states tanh's accuracy as sinh / cosh's, a form that is NaN past
//        about 89. Lanes from the row's distances, min(w, r + 1), to 15
//        compute nothing. Each writes its score to workgroup memory, 8 × 16
//        floats, a head's lanes; one barrier.
//     2. Weights: lane 0 of each head reads its head's scores, at most 12,
//        takes the max, the exponentials and their sum in j order, and
//        writes the weights back in their place; a second barrier. 12
//        exponentials a head a row, once.
//     3. Output: lane l writes dimensions 8l to 8l + 7 of its head, 2 vec4s:
//        Σ_j p_j v[r − j], j in order, the weights read from workgroup
//        memory, the values from the 12 rows' 32 bytes each.
//     Its values are read once a row by one lane each; q, k and P's head
//     slices are read by the lanes of a head together, contiguous along i
//     (GPU.2).
//   - Constants (binding 1): struct LocalAttention { q_scale: f32, k_scale:
//     f32, softcap: f32, window: u32 }.
//   - Bindings: 2 per_dim_scale, F32, and 3 the layer's 13 rows of
//     `relative`, read-only; 4 `query`, 5 `key`, 6 `value`, read-only; 7
//     `attention`, written. Six storage bindings, within WebGPU's default
//     of 8 a stage.
//   - The window's width is the constant's, at most 16, the lanes a head
//     has; d = 128 and H = 8, the forms graph/audio.h checks, are the
//     shader's.
//
// Determinism (GDSA.2): run to run — every sum in the order above, each
// head's softmax one lane's, in j order.
//
// Accuracy, against f64 from the same inputs: a_j and b_j within γ(128) ×
// Σ|terms|; the cap's tanh and the exponentials within WGSL's stated
// accuracy for them; the output within γ(12) × Σ_j p_j |v|. The GPU test
// measures it, a margin checked on the target, not a proof.
//
// What it costs, at R = 500 rows a layer:
//   - Multiply-adds: 8 heads × 12 × 256 for the scores, 12 × 1,024 for the
//     output: 3.7 × 10⁴ a row, 1.8 × 10⁷ a layer.
//   - Bytes: q once a row, 4 KB; k and v at 12 rows each, 96 KB a row from
//     the GPU's caches — each row's keys are its 11 neighbours' too, so
//     about 8 KB a row from memory; P 52 KB a layer, cached.
//   - Exponentials: 12 a head a row, 5.8 × 10⁵ a step at R = 500 —
//     computed once each, where every lane of a head computing its own
//     softmax would be 16 times as many.
//   - 2 barriers a workgroup; 12 launches a step.
// Optimization (practice): the window alone computed, where llama.cpp's
// blocked form computes a chunk's 12 queries against 24 keys and 13
// relative rows, then masks half: 2.4 times the scores, and a padded,
// shifted copy of the relative scores a layer.
// Optimization not taken: a row's 8 heads' scores spread over more lanes —
// the 4 idle lanes in 16 cost a quarter of a step whose work is 0.1% of
// the encoder's.
//
// Verification the implementation is held to, on the GPU against f64: at R
// of 1, 11, 12, 13 and 508 — rows with fewer than 12 predecessors, the first
// full window, the most rows — the f64 reference computing llama.cpp's
// blocked form literally, chunks, the pad and the shift, so the relative
// row a distance reads is checked against the shift, not against this
// header's arithmetic; scores past the cap; equal scores. On the CPU: the
// launch's geometry and bindings.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.20   Prefer return values to out parameters — the launch is returned.
//     I.4    Make interfaces precisely and strongly typed.
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level — run
//            to run.
//     GPU.2  Shape data for coalesced lane access — a head's lanes read its
//            slice together.
//     GPU.8  Synchronize only where data crosses invocations — two
//            barriers, the scores' and the weights'.

struct LocalAttentionLaunch {
    const residency::WeightView* per_dim_scale;   // F32, the head dimension's
    residency::BufferRange relative;              // this layer's 13 rows of the width
    residency::BufferRange query, key, value;     // R rows of the width
    residency::BufferRange attention;             // written
    float q_scale;
    float k_scale;
    float softcap;
    std::uint32_t window;                         // 12
};

// One workgroup of 128 a row, the step's rows. Preconditions: a head
// dimension of 128 and 8 heads; window at most 16; `relative` window + 1
// rows.
[[nodiscard]] Launch local_attention_launch(const LocalAttentionLaunch& attention);

}  // namespace bllm::kernels
