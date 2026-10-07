#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/kernels/interface.h"

namespace bllm::kernels {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The launches a step can dispatch, planned once at build from every
// launch's geometry, so a step walks those alone rather than every launch
// of the graph. A launch's token range and its last-token rows are fixed by
// its description; only a key split depends on the step's position. So the
// token counts 1 .. kPrefillBlock fall into intervals bounded by every
// launch's least and most + 1, and within one interval, with or without
// logits, the same launches can run: each interval holds two lists of
// launch indices, in the graph's order. A step finds its interval by binary
// search over the bounds and walks its list, and workgroups_for still
// decides each launch's workgroups — 0 for a combine the step does not
// split.
//
//   - Invariant: for every step, the list holds every launch workgroups_for
//     gives workgroups, in order, and only launches whose token range and
//     rows allow the step.
//   - What it costs, for Qwen3 0.6B's 595 launches (graph/graph.h): 4
//     intervals — 1, 2 to 8, 9 to 16, 17 to 512 — each 2 lists, of 253 and
//     259 indices: 2,048 indices, 8 KiB, and the bounds, built once at build
//     from all 673 range ends before duplicates go. A step searches 4 bounds
//     and walks 253 to 259 launches, where it would walk 595; the calls out
//     of the module are unchanged.
// Optimization (practice): the launches a step can run planned once, by
// token count and logits, and reused by every step (GDSA.17).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant — the invariant above.
//     F.24   Use a span to designate a half-open sequence — a list.
//   C++ performance guidelines
//     GDSA.17 Plan once per key, reuse the plan — the key is the step's
//            interval and whether it asks for logits.

class Schedules {
public:
    // Plans from every launch's geometry, in the graph's order.
    explicit Schedules(std::span<const Geometry> launches);

    // The launches a step of `tokens` can dispatch, in order. Preconditions:
    // 1 <= tokens <= kPrefillBlock.
    [[nodiscard]] std::span<const std::uint32_t> of(std::uint32_t tokens, bool logits) const noexcept;

private:
    std::vector<std::uint32_t> bounds_;                // each interval's least token count, ascending
    std::vector<std::vector<std::uint32_t>> lists_;    // interval i without logits at 2i, with at 2i + 1
};

}  // namespace bllm::kernels
