# Project Memory

This file is the canonical entry point for durable project context.

## Product Identity

- **Product:** Charlotte. Named `browser-llm` until 2026-09-30 (BLLM-003).
- **Operating brand:** None. Internal R&D under the parent entity.
- **Parent entity:** Plainsight Systems LLC
- **Repository:** <https://github.com/plainsight-systems/charlotte> (public).
  Deployed: <https://plainsight-systems.github.io/charlotte/>

## Purpose

A self-built inference harness that runs an open-weight model entirely in the
browser: C++ compiled to WebAssembly, with compute executed on WebGPU. No
server-side inference, no remote model execution.

## Inherited Governance

Canonical list, public URLs, and internal/published status:
`inherited.md`.

- `plainsight_policies.md`
- `engineering_philosophies.md`
- `product_memory_workflow.md`
- `repo_creation_runbook.md`

## C++ Governance

This product is C++-dominant and performance-sensitive. All four C++ docs are
inherited and both gates bind:

- `cpp_architecture_playbook.md`
- `cpp_architecture_review.md`
- `cpp_performance_playbook.md`
- `cpp_performance_review.md`

## Locked Decisions

Decided 2026-08-31, on acceptance of BLLM-001:

- **The GPU is reached through the `webgpu.h` C API**, not from JavaScript, so
  the same code can later link native Dawn for deterministic kernel tests.
- **Single-threaded, no pthreads.** GitHub Pages cannot set COOP/COEP, so
  `SharedArrayBuffer` is unavailable. The harness runs in a plain Web Worker.
- ~~The device is asked for the adapter's advertised maxima.~~ Superseded by
  BLLM-002 and the one-target decision below: the device requests WebGPU's
  default limits explicitly (`core/gpu/device.cpp`).
- **Async runs are serialised and generation-stamped** (`core/run_guard`).
  WebGPU cannot be cancelled, so a late callback must be identifiable as late
  rather than allowed to report a second result.

Decided 2026-10-02:

- **The design of a change lives in its file headers, not in a packet.** Each
  header states the contract and design and cites the guidelines behind it by
  ID from the `cpp-guidelines` and `cpp-perf-guidelines` corpora; the headers
  are reviewed before the implementation. With the files and contracts laid
  out, a packet restated them. Supersedes the packet step of the inherited
  workflow (`workflow.md`).
- **Optimizations are designed in, not held until a baseline exists.** The
  harness is meant to show both browser-driven and at-scale optimizations;
  each is labelled `Optimization (browser)` or `Optimization (practice)` with
  its reason, and measured where it can be. Supersedes the inherited rule that
  an optimization needs a baseline before it is accepted.
- **One target: the development machine, at WebGPU's default limits.** This is
  a demonstration of an inference harness and its optimizations, not a
  product, so there is no weakest-device row, no per-device budgets and no
  regression baselines. Every figure is labelled with the machine and build
  it came from, and nothing is claimed for hardware it did not run on.
  Supersedes BLLM-002's target matrix and its owed floor budget.

Decided 2026-10-07:

- **The sampled token stays on the GPU and is read back a step behind.** The
  draw runs on the GPU and the next decode step embeds its token from there,
  so a step is submitted before the last one's token maps
  (`src/core/sampler/sampler.h`). Supersedes the per-step readback decided in
  `research/2026-08-31-gpu-readback-round-trip.md`: that decision assumed a
  token took 20–50 ms. Counted, a Qwen3 decode step is 1.3–2.1 ms, so the
  0.5 ms map would add 24–38% to every token. The cache risk that research
  warned of is stated in the sampler's contract, for the runtime's tests to
  hold.

Decided 2026-10-03:

- **A data path's cost is derived before it is built.** Every stage walked
  and counted as functions of the model — bytes and copies, inner-loop
  operations and whether they inline, calls across each boundary including
  the module's calls into browser APIs — and anything that scales with
  blocks, fields or pieces where it could scale with chunks is designed out
  before code. Measurement afterwards calibrates per-call costs and overlap;
  it does not discover the bottleneck. The load path shipped without the
  walk; an audit asked for after the fact found it 3–4 times its floor, and
  every cause it found was countable on paper
  (`research/2026-10-03-load-performance-audit.md`).

Decided 2026-08-28 during repo bootstrap:

- **Core implementation language is C++ compiled to WebAssembly via Emscripten.**
  Rationale: matches the edge-native/hot-path posture in
  `engineering_philosophies.md` and binds this repo to the C++ architecture and
  performance gates. Alternative considered: Rust via wasm-bindgen, rejected
  because it would leave the formal review gates unbound without authoring
  repo-local equivalents.
- **This repo is treated as C++-dominant and performance-sensitive.** Inference
  execution, model load, memory footprint, and GPU dispatch are
  performance-sensitive by default.
- **No operating brand.** Internal R&D attributed to Plainsight Systems LLC
  directly.
- **Inherited governance is a pinned submodule at
  `docs/process/governance/`.** `plainsight-systems-governance` was published
  (CC BY 4.0) with its internal content moved to the operations repo, so it can
  be checked out in-tree over HTTPS by anyone. Pinning is deliberate: it records
  which version of the review gates a packet was written and reviewed against,
  which a floating link cannot. The pin advances by explicit commit, not
  automatically; a stale pin is a record, not a defect. Deviation from
  `repo_creation_runbook.md`, which prescribes symlinks into a private sibling
  and should grow a public-repo branch.

## Research

`../research/README.md` is the index.

## Active Workflow Pointers

- Queue: `QUEUE.md`
- Packets: `packets/` (BLLM-001 to BLLM-003; new work is designed in file
  headers, `workflow.md`)
- Workflow: `workflow.md`
