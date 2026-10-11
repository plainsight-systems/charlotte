#pragma once

#include "core/formats/format.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// QK-norm, RoPE and the cache append, one kernel (docs/architecture/
// kernel-fusions.md): between the QKV projection and attention, for every
// token of the step and every head of a layer.
//
// For each token, at position p = step.position + its row:
//   1. Query and key heads, where the file has QK-norm: each head of d values
//      normalized on its own, as the norm kernel normalizes a row —
//      (x × r) × g, r = 1 / sqrt(mean(x²) + ε) — with the layer's query or
//      key gain, d wide and shared by every head, and the file's ε. Gemma 3's
//      gains are stored as 1 + w (arch/architecture.h).
//   2. Query and key heads: rotated by position. Pair k of a head, for k <
//      d / 2, turns by p × θ_k radians, θ_k = base^(−2k / d) — divided by
//      Llama 3's factor for pair k where the file has rope_freqs — as
//        x' = x × cos − y × sin,  y' = x × sin + y × cos
//      where (x, y) are dimensions 2k and 2k + 1 (Adjacent, llama.cpp's NORM)
//      or k and k + d / 2 (Halves, NEOX), as the model description states.
//   3. Queries are written back in place, f32, for attention. Keys and
//      values — values unnormalized and unrotated — are appended to the
//      layer's cache: slot p mod slots (cache/kv.h), stored by the cache
//      format's pack (format.h).
// Attention reads every key from the cache, the step's own included, so a
// key is the same rounded value whether attention meets it in the step that
// wrote it or a later one. Attention's 1 / sqrt(d) is attention's.
//
//   - Geometry: d / 8 invocations a head, each holding two vec4s of it in
//     registers — dimensions 4i .. 4i + 3 and 4i + d / 2 .. 4i + d / 2 + 3
//     for invocation i of its head. Under Halves those are four whole pairs;
//     under Adjacent each vec4 is two. So every pair is in one invocation's
//     registers and no value crosses between invocations. A token's heads
//     are its query heads, its key heads, then its value heads: (H_q + 2
//     H_kv) × d / 8 invocations a row — 512 for Qwen3, 384 for Llama 3.2,
//     192 for Gemma 3. Workgroups of 64, so 512 / d heads a workgroup, and a
//     row is a whole number of workgroups — 8, 6 and 3 — so every invocation
//     of a workgroup works on one token. Where 64 would straddle a row — a
//     vision layer's 36 heads of 64 are 288 invocations — the workgroup is
//     the largest power of two from 64 down to a head's d / 8 that divides
//     both a row and its rotating heads: 32 there, so a workgroup still
//     holds one token's heads, all rotating or all value heads. Head dimensions 64, 128, 256 and
//     512 — the listed models', and EmbeddingGemma 2's global layers, whose
//     head is one whole workgroup: d / 8 must divide the workgroup, and the
//     frequencies fill at most the constants below.
//   - Angles depend on the position and the pair, not the head. So each
//     workgroup that rotates — any whose heads include a query or key head —
//     first computes its token's d / 2 cosines and sines once, its 64
//     invocations sharing them out — one pair each for Qwen3, two for Gemma
//     3, half of them one for Llama 3.2 — into workgroup memory, d / 2 ×
//     8 bytes, 1 KiB at most; the barrier before the norm's tree publishes
//     them, and without QK-norm one barrier of their own does. Every head of
//     the workgroup then reads its pairs' cos and sin there: cosines, then
//     sines, by pair, so under Halves invocation i of a head reads vec4 i of
//     each — adjacent invocations adjacent words, and a workgroup's heads
//     the same ones, which workgroup memory broadcasts. A workgroup of value
//     heads alone computes none. The tables are sized by an override,
//     `pairs_per_head`, d / 2, set on every launch as WebKit requires of an
//     override sizing a workgroup array
//     (docs/research/2026-10-09-webkit-shader-compiler.md), so a head of 128
//     takes the 1 KiB it took. With the norm's 64 partial sums, a workgroup
//     holds 0.75 KiB at d = 64 and 2.25 KiB at d = 512.
//   - The norm reduces each head in workgroup memory: an invocation's own 8
//     squares, then a tree across its head's d / 8 invocations, every head
//     of the workgroup at once — log2(d / 8) levels, 4 for Qwen3, 5 for
//     Gemma 3, a barrier before the tree and after each level, in each
//     workgroup that rotates. Value heads sharing one take part in the
//     barriers, which WGSL requires every invocation reach, and discard their
//     sum; a workgroup of value heads alone — Qwen3's last two a token —
//     reduces nothing. Without QK-norm — Llama — no reduction runs.
//   - Frequencies: the launcher computes each pair's θ_k / 2π in f64,
//     rounded once to f32 — turns a position — into the launch's constants.
//     The kernel forms t = p × turns_k (÷ factor_k), keeps its fraction r = t
//     − round(t) in [−0.5, 0.5], exact in f32, and takes cos and sin of 2πr.
//     Each angle so carries one f32 rounding of p × turns_k, the error
//     Hugging Face's and llama.cpp's f32 angle carries — half a unit of t,
//     under 0.002 radians at Qwen3's last position, 40,960 — and is in
//     [−π, π], where WGSL bounds cos and sin.
//   - Cache: a slot holds every key-value head's d values, head after head
//     (residency/plan.h); invocation i of key head h at slot s writes
//     pack(vec4 i) and pack(vec4 i + d / 8) at words w + 2i and w + 2i + d /
//     4, w = (s × H_kv + h) × d / 2, under F16. A step's tokens take distinct
//     slots, since a ring holds at least a prefill block (residency/plan.h).
//   - Variants are override constants: `halves`, `qk_norm`, `factors`,
//     `value_norm`, the encode form's alone, and the shape —
//     `head_dimension`, `query_heads`, `key_value_heads`, `pairs_per_head` —
//     so every index folds when the pipeline is built. A model has one
//     pairing, and a shape for each kind of layer, so a pipeline each;
//     Gemma 3's two rotary bases are constants.
//   - Constants (binding 1), as WGSL lays them out:
//       struct Rope { slots: u32, epsilon: f32,
//                     turns: array<vec4<f32>, 64> }   // offset 16; d / 2 used
//     1,040 bytes, five slots of the constants buffer.
//   - Bindings: 2 the query gain, 3 the key gain, 4 the factors, read-only; 5
//     the query buffer, read and written; 6 the key and 7 the value buffers,
//     read-only; 8 the layer's keys and 9 its values in the cache, written.
//     Eight storage buffers, WebGPU's default limit and the harness's
//     (gpu/device_requirements.h). Every binding is declared in every
//     variant: a launch without QK-norm or factors binds the key buffer in
//     their place, read-only there and here, so no buffer is bound both
//     writable and read-only (residency/plan.h).
//
// Two forms, each one source: rope.wgsl, what both share — the constants,
// the gains and factors, the queries, the angle tables, the norm's tree and
// the rotation — then the form's entry point, rope_cache.wgsl for the
// decoder's form above and rope_encode.wgsl for the encode form below,
// joined by the launcher at compile time. Neither form's source holds the
// other's: the decoder's packs with the cache format, which an encoder has
// none of, and a change to one form's entry point touches no other.
//
// The encode form, for an encoder (graph/encoder.h), whose step attends to
// its own keys and values and keeps no cache:
//   - The same steps — the angles, QK-norm, the rotation — over the same
//     invocations and workgroups, but every head written back in place, f32:
//     queries to the query buffer, keys to the key buffer, values to the
//     value buffer, nothing packed and no cache bound.
//   - Values normalized on their own, per head, without a gain, where
//     `value_norm` is set — Gemma 4's ggml_rms_norm of each value head, with
//     the file's ε. Every workgroup then reduces, a workgroup of value heads
//     alone among them; without it, as the decoder's form.
//   - QK-norm is required, so its gains are always bound; a launch without
//     factors binds the query gain in their place. Bindings: 2 the query
//     gain, 3 the key gain, 4 the factors, read-only; 5 the queries, 6 the
//     keys and 7 the values, each read and written. Six storage buffers.
//   - Bytes, a token and a layer: queries, keys and values each read and
//     written, 8 d (H_q + 2 H_kv) — 16 KiB for EmbeddingGemma 2's sliding
//     layers, 24 KiB for its global ones; a 512-token step of its 24 layers
//     moves 208 MiB, about 0.55 ms at 400 GB/s.
//   - Axial, override `axial` (model_description.h's RotaryPairing::Axial),
//     a vision encoder's (graph/vision.h): each half of a head is rotated as
//     a head of d / 2 by its own position — the first by the patch's column
//     x = row mod columns, the second by its row y = row / columns, the
//     step's `columns` (kernels/interface.h) — NEOX pairs k and k + d / 4
//     within the half, θ_k = base^(−2k / (d / 2)) for k < d / 4, the
//     constants holding those d / 4 turns. Invocation i of a head holds,
//     within half h = i / (d / 16), vec4 j = i mod (d / 16) of the half and
//     the vec4 a quarter head after it, so each pair is still in one
//     invocation's registers; the angle table holds the column's d / 4 pairs
//     then the row's, d / 2 entries as the 1-D form's. llama.cpp's gemma4v
//     graph: two ggml_rope_ext of d / 2 dimensions, NEOX, the second offset
//     by d / 2. The decoder's form never takes it.
//
// What it asks of the launch contract (kernels/interface.h, program.h): a
// launch names the format whose pack it composes, beside the one whose
// unpack it composes, and the program composes and keys pipelines by both;
// and a launch's constants may run to 1 KiB, each launch's taking a whole
// number of 256-byte slots of the constants buffer, where every launch's
// took one — at most 1,280 bytes, for this kernel's 1,040. Constants stay
// written once, at load.
//
// Determinism (GDSA.2): run to run, and independent of the step's other rows
// and token count — a head is reduced by its own invocations in one fixed
// order, and everything else is per value. A token's query, key and value
// are the same bits whether it was prefilled or decoded.
//
// Accuracy, against the same computation in f64 from the same turns: a
// normalized head within the norm kernel's margin (kernels/norm/norm.h), its
// tree here shorter; each rotated value within (|x| + |y|) × (2⁻¹¹ + 2π ×
// 2⁻²⁴ × |t|) — WGSL bounds cos and sin to 2⁻¹¹ absolute on [−π, π], and the
// fraction carries t's rounding — and, where factors divide t, WGSL's 2.5
// units of a division on top. Keys and values are then rounded once to the
// cache format, which F16 does to nearest on the target (format.h). As the
// norm's, a margin the GPU test checks on the target, not a proof.
//
// What it costs, counted. Launches: one a layer — 28 a pass for Qwen3, 16
// for Llama 3.2, 26 for Gemma 3 — about 42 µs a Qwen3 step at 1.5 µs each
// (interface.h). Bytes, a token and a layer, with d-wide heads in f32 and an
// F16 cache: queries read and written, 8 d H_q; keys and values read, 8 d
// H_kv; both written to the cache, 4 d H_kv — 8 d H_q + 12 d H_kv: 28 KiB for
// Qwen3, 22 KiB for Llama 3.2, 11 KiB for Gemma 3; the gains, 2 d × 4 bytes,
// are read by every head and cached. A 512-row prefill step of Qwen3 moves
// 28 layers × 512 × 28 KiB, 392 MiB, about 1.0 ms at 400 GB/s; a decode
// step, 784 KiB, about 2 µs, far below its 28 launches. Arithmetic: a sin
// and a cos for each pair of each rotating workgroup — for Qwen3 6
// workgroups a token, 384 pairs a token and a layer, where a head at a time
// would compute 1,536; 160 against 1,280 for Llama 3.2, 384 against 640 for
// Gemma 3. A 512-row prefill step of Qwen3 takes 5.5 million pairs, 11
// million sin and cos. An estimate, not a count: if each, reduced to
// [−π, π], costs about 40 instructions — a range check and a polynomial,
// as a precise sin does — that is about 0.4 × 10⁹ instructions, about 0.06
// ms at the 40-core M3 Max's roughly 5,120 lanes issuing one a cycle at
// about 1.4 GHz (third-party figures). Four times that would still be under
// the step's 1.0 ms of bytes, which it overlaps. Unfused — two norms, two rotations and
// two cache writes, each reading and writing its own pass — Qwen3 would move
// 60 KiB a token and a layer, in six launches a layer, 140 more a pass.
// Optimization (practice): the norms, both rotations and the cache append
// ride in one launch, as vLLM's fused_qk_norm_rope with its cache write
// does, rather than six (GPU.6, GDSA.6).
// Optimization (practice): every invocation holds both values of each pair
// it rotates, so the rotation exchanges nothing between invocations, and
// adjacent invocations read and write adjacent vec4s (GPU.2).
// Optimization (practice): a workgroup's angles computed once and shared
// through workgroup memory by its heads — 4 for Qwen3, 8 for Llama 3.2 — not
// once a head, as llama.cpp's kernel does (GPU.5).
// Optimization (practice): the frequencies, fixed by the file, computed once
// at load into the form the kernel reads, turns a position (CDSA.32).
//
// Where WebGPU limits it, and what each limit costs here:
//   - WGSL bounds pow only to 3 + 2 |y log2 x| units in the last place —
//     about 43 for the last pair at a base of 10⁶ — and cos and sin only on
//     [−π, π]. So the turns are computed once in f64, as Hugging Face
//     computes its inverse frequencies once, not with pow per value as
//     llama.cpp's Metal kernel does; and the angle is reduced to [−π, π]
//     before cos and sin. Both cost nothing a step: the turns ride in
//     constants written at load, and the reduction is a round and a
//     subtract.
//   - Subgroup operations are optional in WebGPU, and for cross-browser
//     compatibility the harness requires none (gpu/device_requirements.h).
//     So the head's sum is a tree in workgroup memory, 5 barriers for
//     Qwen3, where vLLM's warp sum needs none; WGSL's zero-filled workgroup
//     memory adds a store and a barrier a workgroup.
//
// Levers not taken:
//   - A prefill step's angles computed once, by a launch of their own, into
//     a working buffer the rope launches read, decode keeping them inline: B
//     × T × d / 2 pairs a step for B distinct bases and factors — 32,768 for
//     Qwen3's 512 rows, 168 times fewer than the workgroups compute. It
//     saves the estimate above, about 0.06 ms and by that estimate under
//     0.25 ms, in a step whose matrix products — 2 × 440 × 10⁶ × 512 f32
//     operations, about 32 ms at about 14 TFLOPS (kernels/matmul/matmul.h)
//     — take over a hundred times longer. It costs a launch of its own and
//     a second form of every rope launch, one for each regime.
//   - A cos and sin table by position, as vLLM's RotaryEmbedding keeps: no
//     sin or cos in the kernel, for a table of context × d / 2 pairs — 20
//     MiB for Qwen3, 64 MiB for Gemma 3's two bases — taken from the cache's
//     budget, 183 tokens of Qwen3's context, to save arithmetic in a kernel
//     its bytes bound.
//   - Attention's 1 / sqrt(d) folded into the queries here: one multiply a
//     query value saved against one a score; attention, which owns the
//     scale, takes it.
//
// Verification the implementation is held to, on the GPU against an f64
// reference from the launcher's turns: each listed shape — Qwen3's d = 128
// under Halves with QK-norm, Llama 3.2's 64 under Adjacent with factors,
// Gemma 3's 256 under Halves with QK-norm and a ring — at positions from 0
// to the last a context offers; values in the cache, the round-to-nearest
// F16 of the input bit for bit; a step whose slots wrap a ring; and the
// same tokens prefilled as one step and decoded one at a time, writing the
// same query and cache bits. On the CPU: each turn within half a unit of its
// f64 value.

// The encode form, likewise: EmbeddingGemma 2's sliding shape, d = 256 with 4
// query and 2 key-value heads, and its global shape, d = 512 with 4 and 1,
// both under Halves with QK-norm and the value norm, at steps of 1, 3 and 512
// tokens from position 0 — queries, keys and values in place against f64,
// within the norm's margin and the rotation's; values left as they were
// without `value_norm`; and the decoder's form at d = 128 the same bits as
// before the tables were sized by an override.//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed — the pairing is an
//            enumeration, the norms and factors views or null.
//     C.12   Don't make data members const or references in a copyable
//            type — RopeLaunch holds its small parts by value and the rest by
//            pointer.
//     F.60   Prefer T* over T& when "no argument" is valid — the norms and
//            factors.
//     F.20   Prefer return values to out parameters — the launch is
//            returned.
//   C++ performance guidelines
//     GDSA.2 Declare each floating-point reduction's determinism level — as
//            above.
//     GDSA.6 Count passes over global memory — the bytes above, and the
//            unfused count against them.
//     GPU.2  Shape data for coalesced lane access — the vec4 mapping above.
//     GPU.5  Use workgroup memory where reuse or reordering pays — each
//            head's partial sums, and the workgroup's angles; its bytes and
//            barriers counted above.
//     GPU.6  Batch tiny GPU work — six steps, one launch.
//     CDSA.32 Transform static data once into the layout its consumer reads —
//            the turns, at load.

// One layer's rope launch.
struct RopeLaunch {
    model::LayerDescription layer;                 // heads, head dimension, rotary base
    model::RotaryPairing pairing;
    const residency::WeightView* query_norm;       // F32, head dimension; null without QK-norm
    const residency::WeightView* key_norm;         // present exactly when query_norm is
    const residency::WeightView* factors;          // F32, head dimension / 2; null without
    float epsilon;
    residency::BufferRange query;                  // the plan's working buffers
    residency::BufferRange key;
    residency::BufferRange value;
    residency::PlannedCacheLayer cache;            // the layer's
    const formats::Format* cache_format;           // never null; its pack writes the cache
};

// The launch for one layer. Preconditions: the head dimension is 64, 128,
// 256 or 512, and (query heads + 2 × key-value heads) × head dimension / 8 a
// multiple of 64; the views are F32 of the widths stated; `cache_format` has a pack and
// is the format the plan sized `cache` in.
[[nodiscard]] Launch rope_launch(const RopeLaunch& rope);

// One layer's rope launch in its encode form (above).
struct RopeEncodeLaunch {
    model::LayerDescription layer;                 // heads, head dimension, rotary base
    model::RotaryPairing pairing;
    const residency::WeightView* query_norm;       // F32, head dimension; never null
    const residency::WeightView* key_norm;         // F32, head dimension; never null
    const residency::WeightView* factors;          // F32, head dimension / 2; null without
    bool value_norm;                               // values normalized per head, without a gain
    float epsilon;
    residency::BufferRange query;                  // the plan's working buffers, each written in place
    residency::BufferRange key;
    residency::BufferRange value;
};

// The encode form's launch for one layer. Preconditions: rope_launch's on
// the shape and the views; the query and key gains both present.
[[nodiscard]] Launch rope_encode_launch(const RopeEncodeLaunch& rope);

}  // namespace bllm::kernels
