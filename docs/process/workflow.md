# Workflow

This repo adapts the Lite Factory workflow from
`governance/product_memory_workflow.md`. The design of each change lives in
the code it changes, not in a separate packet.

## Default Chain

```text
Coordinator
  -> file headers: the contract, the design, and the guidelines behind it
  -> review of the headers
  -> implementation and tests
  -> review
  -> MEMORY.md and QUEUE.md updates
```

## Local Notes

This repo is C++-dominant and performance-sensitive.

- **The headers are the plan.** Each new or changed file's header states its
  contract and its design, and cites by ID the guidelines from the
  `cpp-guidelines` and `cpp-perf-guidelines` corpora that shape it (for
  example C.2, E.4, CACHE.3, EMB.6). The headers are written and reviewed
  before the implementation.
- **Optimizations are designed in, not deferred.** The harness exists to show
  both the optimizations the browser forces and the ones production systems
  use at scale. Each carries `// Optimization (browser): …` or
  `// Optimization (practice): …`, with its concrete reason and the guideline
  or system it follows, so a reader sees a choice rather than an accident.
  Where it can be measured, it is, and the figures go in the note.
- **A data path's cost is derived before it is built.** Its header walks
  every stage and counts, as functions of the model: the bytes it moves and
  the copies it makes (GDSA.6), the operations in its inner loop and whether
  each is a call the compiler can inline, and the calls it makes across
  every boundary, including the module's calls into browser APIs (WASM.2).
  A count that scales with blocks or fields where it could scale with
  chunks, or a crossing per piece where one per phase would do, is designed
  out before code is written. Once the path runs it is measured against
  that floor (GPU.10, WASM.11) to calibrate per-call costs and overlap; a
  stage well off its own ceiling is fixed or named as the remaining gap
  (`research/2026-10-03-load-performance-audit.md`).
- **Every C++ step is checked against `cpp-guidelines` before its code is
  written, and performance-sensitive work against `cpp-perf-guidelines`.** The
  citations in the headers and optimization notes are the record. If either
  server is unreachable, say so and get explicit agreement before writing C++.
- **Review is per commit.** `scripts/codex-review.sh --post <commit>` has an
  independent reviewer check the commit for two things: whether it is
  correct, and whether it is as fast as it could be — by walking the changed
  hot paths and counting their bytes, copies and boundary crossings — and
  posts its findings as comments on the commit: one with the whole review,
  one on each finding's line. Findings are fixed when they change what the
  program computes or how fast it runs; a P0 or P1 blocks acceptance.
- Inference execution, model load, memory footprint and GPU dispatch are
  performance-sensitive by default.
- Browser and GPU behavior is environment-sensitive. Verification names the
  machine it ran on; a green build is not proof. There is one target, the
  development machine (MEMORY.md); reviews do not ask for a target matrix,
  per-device budgets or regression baselines.
- `packets/` holds the records of BLLM-001 to BLLM-003. New work does not
  start with a packet.

Do not weaken inherited governance without a decision; MEMORY.md records the
ones taken.
