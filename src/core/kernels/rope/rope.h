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
//     of a workgroup works on one token. Head dimensions 64, 128 and 256,
//     the listed models': d / 8 must divide the workgroup, and the
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
//     heads alone computes none. With the norm's 64 partial sums, a
//     workgroup holds at most 1.25 KiB of workgroup memory.
//   - The norm reduces each head in workgroup memory: an invocation's own 8
//     squares, then a tree across its head's d / 8 invocations, every head
//     of the workgroup at once — log2(d / 8) levels, 4 for Qwen3, 5 for
//     Gemma 3, a barrier before the tree and after each level. Value heads
//     take part in the barriers, which WGSL requires every invocation reach,
//     and discard their sum. Without QK-norm — Llama — no reduction runs.
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
//   - Variants are override constants: `halves`, `qk_norm`, `factors`, and
//     the shape — `head_dimension`, `query_heads`, `key_value_heads` — so
//     every index folds when the pipeline is built. A model has one shape and
//     one pairing, so one pipeline; Gemma 3's two rotary bases are constants.
//   - Constants (binding 1), as WGSL lays them out:
//       struct Rope { slots: u32, epsilon: f32,
//                     turns: array<vec4<f32>, 32> }   // offset 16; d / 2 used
//     528 bytes.
//   - Bindings: 2 the query gain, 3 the key gain, 4 the factors, read-only; 5
//     the query buffer, read and written; 6 the key and 7 the value buffers,
//     read-only; 8 the layer's keys and 9 its values in the cache, written.
//     Eight storage buffers, WebGPU's default limit and the harness's
//     (gpu/device_requirements.h). Every binding is declared in every
//     variant: a launch without QK-norm or factors binds the key buffer in
//     their place, read-only there and here, so no buffer is bound both
//     writable and read-only (residency/plan.h).
//
// What it asks of the launch contract (kernels/interface.h, program.h): a
// launch names the format whose pack it composes, beside the one whose
// unpack it composes, and the program composes and keys pipelines by both;
// and a launch's constants may run to 1 KiB, each launch's taking a whole
// number of 256-byte slots of the constants buffer, where every launch's
// took one. Constants stay written once, at load.
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
// million sin and cos at tens of instructions each — about 0.4 × 10⁹
// instructions, about 0.05 ms for a GPU of about 14 f32 TFLOPS, as
// third-party measurement puts the 40-core M3 Max — under its bytes' 1.0 ms,
// and overlapped with them. So the bytes bound it. Unfused — two norms, two rotations and
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
//   - The step's angles computed once, by a launch of their own, into a
//     working buffer the rope launches read: B × T × d / 2 pairs a step for
//     B distinct bases and factors — 32,768 for Qwen3's 512 rows, 168 times
//     fewer than the workgroups compute. It saves at most the 0.05 ms above
//     in a prefill step, where the bytes take 1.0 ms anyway, and costs every
//     decode step a launch, 1.5 µs, to save 10,752 pairs spread one to an
//     invocation over 168 workgroups.
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
//
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

// The launch for one layer. Preconditions: the head dimension is 64, 128 or
// 256, and (query heads + 2 × key-value heads) × head dimension / 8 a
// multiple of 64; the views are F32 of the widths stated; `cache_format` has a pack and
// is the format the plan sized `cache` in.
[[nodiscard]] Launch rope_launch(const RopeLaunch& rope);

}  // namespace bllm::kernels
