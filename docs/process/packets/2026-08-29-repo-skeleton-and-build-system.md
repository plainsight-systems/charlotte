# BLLM-001: Repo skeleton and build system

**Status:** accepted
**Change class:** architectural

> The project was later renamed from browser-llm to Charlotte (BLLM-003, 2026-09-30). This record keeps the name it was written under.

## Intent

- **What is changing:** The repository gains its physical shape and build
  system. A dual-target CMake build produces (a) a native binary and test
  suite from platform-neutral C++ and (b) a WebAssembly module for the
  browser, from the same core sources. A static page loads that module in a
  Web Worker, acquires a WebGPU device through the `webgpu.h` C API, runs one
  trivial compute shader, and verifies the readback. CI builds and tests
  natively on every push and deploys the page to GitHub Pages.

- **Why the change is necessary:** No implementation can begin without a
  toolchain, and the toolchain choices here are not reversible cheaply. The
  boundary between platform-neutral core C++ and the browser wrapper, and the
  decision to reach the GPU through `webgpu.h` rather than JavaScript,
  determine whether inference code can be tested outside a browser at all.
  Establishing them before any model code exists is the difference between a
  testable harness and one that is only observable through a page.

- **Expected behavior changes:** The repository builds where it previously
  contained no source. `make test` produces a native binary that runs unit
  tests without a browser or GPU. `make wasm` produces an ES module and its
  `.wasm`. The deployed page reports the WebGPU adapter, its limits, and the
  result of a vector-add readback. Nothing infers anything; no model is
  loaded; no weights are fetched.

- **Guaranteed invariants/contracts:**
  - `src/core/**` never includes `emscripten.h` and never references anything
    under `src/wasm/` or `web/`. Dependency direction is one-way.
  - `src/wasm/bindings.cpp` is the only Emscripten-aware translation unit.
  - No `#ifdef __EMSCRIPTEN__` anywhere in `src/core`: core is written
    platform-neutral, and `tools/check_boundaries.sh` enforces that it never
    references Emscripten or depends outward on the wrapper.
  - `dispatch_math` and `unique_handle` compile and are tested in the native
    configuration. `device.cpp` and `self_check.cpp` compile **only** under
    Emscripten in this packet, because they need an implementation of
    `webgpu.h` and the native one is Dawn, which is deferred to the first
    model-kernel packet. They are written against `webgpu.h` so that adding
    Dawn requires no source change — but until it lands, native source parity
    is NOT claimed.
  - The build never requires `SharedArrayBuffer`, cross-origin isolation, or
    pthreads. GitHub Pages cannot set COOP/COEP headers, so any dependency on
    them would break deployment.
  - WGSL shader sources are single-sourced. The same file is compiled into
    both targets by build-time codegen; there is no second copy.

## Acceptance Criteria

- [x] `cmake --preset native-debug && cmake --build --preset native-debug`
      succeeds from a clean tree.
- [x] `ctest --preset native-debug` passes, exercising real logic, without a
      browser and without a GPU.
- [x] The wasm configuration builds inside the pinned Emscripten image and
      emits `browser_llm.mjs` and `browser_llm.wasm`.
- [x] Grep check passes: no file under `src/core/` matches `emscripten.h`,
      `EM_JS`, `EMSCRIPTEN_`, or `#include "wasm/`.
- [x] Grep check passes: `src/wasm/bindings.cpp` is the only file under `src/`
      that includes `emscripten.h`.
- [x] Shader embedding is verified: the generated C++ constant is byte-identical
      to `src/shaders/vector_add.wgsl`, asserted by a native test rather than
      by inspection.
- [x] Serving `dist/` over http and opening the page in Chrome reports the
      adapter, the four buffer limits, and `vector_add: OK`. Verified locally
      and against the deployed Pages site. Note: this adapter reports empty
      `description` and `device` strings and populates `vendor`/`architecture`
      instead, so the criterion's "non-empty description" wording was wrong;
      the page renders "(not reported)" rather than implying data it lacks.
- [x] Both CI workflows pass on a push.
- [x] `README.md` documents the layout as built, with planned-but-absent
      directories marked as planned.

## Verification Plan

Native unit tests cover the logic that exists at this stage and no more:

| Test | What it pins |
|---|---|
| `dispatch_math_test` | Workgroup-count arithmetic: exact multiples, non-multiples, zero elements, and sizes near the `maxComputeWorkgroupsPerDimension` bound. Pure function, exhaustively checkable. |
| `shader_embed_test` | The embedded shader constant equals the on-disk `.wgsl` byte for byte, so the single-source invariant cannot silently break. |
| `run_guard_test` | Run serialisation and stale-callback suppression: a second run is refused while one is live, completing twice reports false the second time, and a generation from a timed-out run cannot close a later one. This is the logic that stops a late callback reporting a second result. |
| `unique_handle_test` | `UniqueHandle` ownership: releases exactly once on destruction, move transfers without double-release, self-move is safe, `reset` releases the prior handle, `release` relinquishes without releasing. This is what makes the RAII claim evidence rather than assertion. |
| `test_check_boundaries.sh` | Each boundary rule actually fires against real probe files, including a near-collision filename. |
| `test_codex_review_preflight.sh` | The review script refuses to run when a required MCP server is unreachable. |

Structural invariants are verified by grep in CI rather than by review, because
a reviewer noticing a stray `#include <emscripten.h>` is not a control.

The GPU path is verified in-browser at this stage by readback assertion. A
vector-add is self-evidently correct or not; it needs no oracle. This is
deliberately the last packet where that is true.

**Explicitly not verified here, and why:** there is no CPU-side model logic
yet, so there is nothing further to unit test. Build success is normally not
evidence of correctness, but for a packet whose deliverable *is* the build,
the build succeeding under a pinned toolchain, plus a real GPU readback, is
the contract. Unit-test surface arrives with BLLM-002 and its GGUF parsing.

## C++ Architecture Note

- **Core components touched:** New. `core/gpu` only — device acquisition,
  buffer creation and readback, compute pipeline construction, dispatch
  arithmetic, and a self-check that runs the vector-add end to end.

- **Platform wrappers touched:** New. `src/wasm/bindings.cpp` exposes the
  self-check and device description to JavaScript. `web/` owns the page, the
  worker, and rendering.

- **Public interfaces changed:** None existed. Introduced:
  `gpu::UniqueHandle<Handle, ReleaseFn>` and the RAII aliases in
  `wgpu_handles.h` (`Buffer`, `ShaderModule`, `ComputePipeline`, `BindGroup`,
  `BindGroupLayout`, `CommandEncoder`, `CommandBuffer`, `ComputePassEncoder`,
  `Instance`, `Adapter`, `DeviceHandle`, `Queue`), plus `gpu::Device`,
  `gpu::dispatch_count`, `gpu::DispatchResult` and `gpu::run_self_check`.

- **New files/classes/targets/modules:** Targets `browser_llm_core` (static
  library, both configurations), `browser_llm_tests` (native only),
  `browser_llm` (wasm only).

- **Dependency edges added or removed:** `browser_llm_core` depends on
  `webgpu.h` only. In the wasm configuration that header and its
  implementation come from `--use-port=emdawnwebgpu`; natively they would come
  from Dawn, which is not linked in this packet. `browser_llm_tests` depends
  on core and doctest. `browser_llm` depends on core and Emscripten. No edge
  points from core outward.

- **Ownership/lifetime model:** RAII throughout, via
  `UniqueHandle<Handle, ReleaseFn>`: move-only, non-copyable, releasing
  exactly once in its destructor. Every WebGPU release in the codebase is
  reached through that template — no source file calls `wgpuXRelease` by hand,
  which is what makes early returns safe in a build compiled without
  exceptions (E.25). `Device` holds its instance, adapter, device and queue as
  such handles; member declaration order gives reverse-order release without
  a hand-written destructor. Device ownership is transferred to the caller as
  `std::unique_ptr<Device>`, never as a raw pointer (I.11). Nothing is shared;
  no `shared_ptr` appears.

- **Test surface independent of UI wrappers:** `dispatch_count`, the shader
  embedding, and `UniqueHandle` are tested natively with no GPU and no
  browser. `UniqueHandle` is deliberately free of `webgpu.h` so its ownership
  semantics — release once, move transfers, self-move safe, reset releases the
  old handle — are pinned by tests rather than asserted in prose. `Device` and
  the self-check are written against `webgpu.h` so they can link against
  native Dawn later without modification; that is exercised in the first
  kernel packet.

- **Why this decomposition is appropriate:** The core/wrapper split is the
  only structural decision that is expensive to reverse. Everything else in
  this packet is a build file. Placing WebGPU behind `webgpu.h` inside core,
  rather than in the JS layer, is what keeps the forward pass — when it exists
  — in testable C++ instead of split across the WASM boundary.

- **What is intentionally not abstracted:** There is no rendering, tensor, or
  model abstraction. `UniqueHandle` is not `std::unique_ptr` with a custom
  deleter: these are opaque handles whose release function is not `delete`,
  and the deleter form buys nothing while making every alias noisier. No interface has more than one implementation and none is
  introduced speculatively. `Device` is a concrete type, not an interface;
  substitutability is provided by `webgpu.h` itself, one layer down.

## C++ Performance Note

- **Target hardware/workload:** Chrome and Edge on desktop, discrete and
  integrated GPUs. No inference workload exists yet.

- **Performance budget:** None set. This packet deliberately does not
  establish latency or throughput budgets, because there is no workload to
  budget. Budgets are owed by BLLM-002, which selects the model.

- **Baseline measurement:** Not applicable — no steady-state path exists.
  What this packet does establish is the *measurement surface*: the page
  reports adapter description and the buffer limits that will constrain weight
  residency, so the first real baseline can record the device it was taken on.

- **Hot paths touched:** None. Device acquisition and pipeline creation are
  setup-time and run once.

- **Allocation/copy/serialization behavior:** One upload and one readback of a
  small buffer. Structurally, the design decision that matters is that
  `runtime/weights` will stream chunks to GPU buffers rather than holding a
  whole model in the WASM heap — WASM32's address space makes a resident
  400MB+ model wasteful and eventually impossible. That path is not built
  here, but `Buffer` is shaped to support chunked upload rather than a
  single `load_all`.

- **Concurrency model:** Single-threaded by constraint. No pthreads, because
  GitHub Pages cannot deliver COOP/COEP and therefore cannot provide
  `SharedArrayBuffer`. The harness runs in a plain Web Worker so the main
  thread stays responsive; this needs no shared memory since ownership passes
  by transfer.

- **Edge AI runtime/backend/configuration:** WebGPU via emdawnwebgpu. Compute
  only; no rendering pipeline is created.

- **Before/after measurement plan:** None applicable. Deferred to BLLM-002.

- **Accuracy/product-contract risk from performance optimizations:** None.
  No optimization is performed.

## Review Record: C++ guidelines audit (2026-08-30)

Audited against the `cpp-guidelines` and `cpp-perf-guidelines` MCP servers.
This check was **skipped during implementation** — both servers dropped
mid-session and the C++ was written anyway without the gap being raised. That
process failure is the primary finding; the defects below are its consequence.

| Severity | Finding | Guidelines | Status |
|---|---|---|---|
| P1 | This packet documented `gpu::Buffer` and `gpu::ComputePipeline` and claimed "RAII throughout" when neither type existed and release was manual | §3.6 no-facades | Fixed: the types now exist and the note describes them accurately |
| P1 | 11 hand-written `wgpuXRelease` calls; no RAII. Build has exceptions disabled, so an added early return would leak silently | R.1, E.6, C.31, P.8, E.25 | Fixed: `UniqueHandle`; zero manual releases remain |
| P1 | `wgpuDeviceCreateBuffer` is `WGPU_NULLABLE`; four calls unchecked, null flowing into `wgpuQueueWriteBuffer` | E.25, P.8 | Fixed: checked, with an explicit error |
| P2 | `g_device` non-const owning global, never freed, overwritten on re-entry | R.6, I.2, LIFE.6 | Fixed: construct-on-first-use, re-entry refused explicitly |
| P2 | Device ownership transferred through a raw `Device*` the callee had to `delete` | I.11, R.3, C.149, R.20, F.26 | Fixed: `std::unique_ptr<Device>` |
| P2 | WebGPU validation and OOM errors were silent — no uncaptured-error callback installed | §3.3 observability | Fixed: installed on the device descriptor |
| P3 | Unused `<cstring>`; `std::string_view` included only transitively; `char` passed to `%04x` | Include-what-you-use | Fixed |

Performance corpus: searched, **no material findings**. Top hits were arena
allocators, NUMA and lossy ring buffers — all hot-path concerns. Nothing in
this packet is on a hot path; it is setup-time code run once. `LIFE.6` was the
only applicable rule and is addressed above. Recorded rather than omitted,
because "checked and found nothing, for this reason" is the artifact.

**This audit was self-review.** `product_memory_workflow.md` requires reviewer
≠ implementer for material changes. It should be repeated by a different
identity before acceptance; a self-audit is least likely to catch a conceptual
error its author already made.

## Acceptance (2026-08-31)

Accepted after two independent reviews by a separate identity
(`gpt-5.6-sol` via `scripts/codex-review.sh`), both returning
`changes_requested`, and both sets of findings worked.

| Review | Guideline audit | Findings | State |
|---|---|---|---|
| 1st, 2026-08-30 | **NOT PERFORMED** — `get_guideline` denied by approval policy | 1 P1, 4 P2, 1 P3 | all worked |
| 2nd, 2026-08-30 | Complete — 16 `cpp-guidelines` + 3 `cpp-performance` lookups, no environment failure | 1 P1, 5 P2, 1 P3 | worked, one residual below |

The second review's P1 was the sharpest finding of the packet: `Device::limits()`
reported adapter maxima while acquisition requested none, so dispatch was sized
against capacity the device did not have. It also exposed that a "correction" I
had recorded was itself wrong. Both are in
`docs/research/2026-08-29-webgpu-limits-observed.md`.

**Accepted residual risk:**

- **No native GPU test.** `device.cpp` and `self_check.cpp` compile only under
  Emscripten; the GPU path is verified by the page's readback assertion alone.
  Closing this is BLLM-003, which lands with the first model kernel — the first
  point at which a kernel's correctness is not self-evident.
- **`delete this` in the continuation chains.** Reviewed and kept: ownership
  genuinely moves forward across hops, so reclaiming into a `unique_ptr` at each
  callback entry would destroy the context one hop early. Documented in
  `device.cpp` so it is not re-raised as drift.
- **Second review's fixes were not themselves re-reviewed.** A third pass would
  have diminishing returns against the cost; accepted deliberately rather than
  overlooked.

Verification at acceptance: wasm builds clean, 25 native test cases / 74
assertions pass, four structural guards fire against real probe fixtures, both
CI workflows green, and the deployed page reports 4,096 elements with 0
mismatches.

## Records

- Design discussion preceded this packet and settled: `webgpu.h` over
  JS-side WebGPU, single-threaded WASM, no bundler, weights fetched at runtime
  rather than committed, and Dawn deferred to the first model kernel.
- Deviation from the confirmed plan: doctest is pinned via CMake
  `FetchContent` at **v2.4.11 → v2.4.12** rather than vendored into
  `third_party/`. v2.4.11 declares a `cmake_minimum_required` below CMake 4's
  floor and will not configure at all. Pinning by tag is at least as
  deterministic as vendoring and avoids committing a third-party header.
- Toolchain deviation: Docker's registry access is broken on the development
  machine (the host reaches `auth.docker.io` in 0.2s; the daemon cannot reach
  it at all). The wasm build was produced with a host emsdk pinned to the same
  6.0.8 the image provides. CI continues to build in the container, and did so
  successfully, so the pinned environment is exercised on every push.
- Verified end to end on 2026-08-29: Chrome on Apple silicon (`apple` /
  `metal-3`), 4096 elements, 0 mismatches, deployed from CI to GitHub Pages.
- **Limits: wrong twice, in opposite directions.** The design discussion
  treated 128/256 MiB as hard caps. A first correction read adapter maxima as
  capacity already granted. A second over-corrected back to "the originals
  were right." All three were wrong: those are WebGPU's *defaults*, and
  `requiredLimits` raises them. Measured — asking for the adapter's own maxima
  grants 4096 MiB and 1024 invocations per workgroup, against 128 MiB and 256
  by default. See `docs/research/2026-08-29-webgpu-limits-observed.md`.
- **Scope note (§5.6).** Fixing the P1 required separating device limits from
  adapter maxima. *Also requesting* those maxima via `requiredLimits` is a
  behavior change beyond that fix, taken deliberately: leaving the harness on
  defaults would mean designing weight residency around a limit the hardware
  does not actually impose. Flagged rather than folded in silently. If the
  reviewer disagrees, it belongs in BLLM-002 where the floor is decided.
