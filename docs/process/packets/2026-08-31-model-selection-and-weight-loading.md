# BLLM-002: Model selection and weight residency

**Status:** approved
**Change class:** architectural

## Intent

- **What is changing:** The harness gains a target model and gets its weights
  onto the GPU. GGUF container parsing, Q4_0 block layout, the tokenizer, model
  config — and then the part that matters most: **quantized weights streamed
  into GPU buffers, still quantized, and proven byte-identical by readback.**

- **Why the change is necessary:** Weight residency is the highest-risk unknown
  in the project. ~420 MB of weights must land across many buffers, under a
  storage-binding limit that is granted at runtime and varies per device, on a
  platform where we have chosen a 32-bit address space. Every kernel decision downstream —
  bind group layout, dequant shader shape, attention tiling — depends on how
  the weights are laid out. Deferring it until the kernels exist would mean
  debugging residency and shaders together.

  An earlier draft of this packet scoped the work CPU-only and justified it as
  "natively testable." That was convenience dressed as principle: it deferred
  the riskiest item and kept the easiest. Sequencing follows risk.

- **Expected behavior changes:** The page gains a load phase — fetch with
  progress, SHA-256 verification, OPFS cache — then reports the parsed model
  summary and a **residency map**: how many buffers, their sizes, which limits
  bound the packing, and confirmation that **every resident byte** reads back
  identical to its source. Still no inference and no token generated.

  An earlier draft of this line said "a sampled block". WASM.9 names that
  directly: a sampled verification passes while another chunk is truncated or
  written to the wrong offset. Acceptance criterion 7 always required every
  byte; the intent now matches it rather than promising less.

- **Guaranteed invariants/contracts:**
  - **Weights reach the GPU quantized.** The CPU never materialises a
    dequantized weight. Dequantization happens in a shader, in BLLM-003. A
    CPU-dequantize-then-upload path would quadruple resident bytes and defeat
    the entire reason for quantization.
  - The GGUF parser treats the file as **untrusted input**. Every offset and
    length is bounds-checked, and `offset + length` and shape products use
    **checked arithmetic** — an overflow must not wrap into an in-range value
    (SL.con.3, ES.103).
  - **The parser reads bounded windows, not one span over the file.** An
    earlier draft said it borrows a single `std::span` over the bytes while
    also guaranteeing the 420 MB file is never fully resident. Those
    contradict: a span over the whole file requires the whole file, and a span
    over a prefix cannot validate a tensor region near the end against
    `span::size()`. The reader takes an incremental source plus the
    **authoritative total file length**, validating each region against that
    length rather than against whatever window is currently mapped.
  - The whole model is never resident in the WASM heap. Bounded chunks are
    uploaded and released, so **peak residency is a function of chunk size, not
    of file size** (WASM.9). Linear memory is one contiguous allocation we are
    trying to keep small (WASM.1).

    **Streaming is a choice, not a constraint.** An earlier draft argued it was
    forced: wasm32 caps linear memory at 4 GB, so MEM.7's reserve-and-commit
    strategy was unavailable and streaming was "the only option". That premise
    expired. WebAssembly Memory64 shipped in Chrome M133 and Firefox and was
    standardised in WebAssembly 3.0 on 2026-06-13; the web cap is now ~16 GB,
    and a 420 MB model would fit in linear memory with room to spare.

    We stay on wasm32 deliberately. Memory64 costs roughly 10% throughput, and
    it buys us nothing: the weights' destination is a set of GPU buffers, so
    the heap they would occupy on the way is pure overhead no matter how large
    it is permitted to be. Streaming is what we would choose on a 64-bit heap
    too — which is what the reference implementations do. LlamaWeb, whose
    Emscripten-plus-WGSL structure is close to ours, states the same rule
    independently: weights are downloaded and *never materialised in the
    WebAssembly heap*.

    The distinction is not pedantic. "We had no choice" is a claim a reader can
    falsify in one search; "we knew the option and declined it for a measured
    reason" is the one that holds.
  - **The load path returns to the event loop between chunks** (WASM.3). Fetch,
    hashing and upload are asynchronous browser operations. A blocking C++ loop
    over chunks would require Asyncify, whose cost WASM.3 says not to buy, and
    would freeze the worker for the length of a 420 MB download. The C++ side is
    therefore a step function driven from JS, not a `load()` that runs to
    completion.
  - Buffer packing is planned from the **granted** device limits, read after
    acquisition, and works at the 128 MiB spec-default storage-binding floor.
  - **Weights are de-interleaved at upload into two aligned streams.** This is
    decided, not provisional. On disk a Q4_0 block is 18 bytes — a 2-byte fp16
    scale followed by 16 bytes of nibbles — so block `k` begins at byte `18k`,
    which is 4-byte aligned only for even `k`. Uploaded verbatim, a kernel
    reading `array<u32>` would shift across word boundaries on every load.

    Instead the upload splits each tensor into:
      - **nibbles** as `array<u32>` — 16 bytes is exactly 4 words per block, so
        block `k` starts at word `4k`, and lane L reads word L (GPU.2);
      - **scales** as `array<u32>` — two fp16 packed per word, read in-shader
        with `unpack2x16float`, which is core WGSL and needs no `shader-f16`.

    Total stays **18 bytes per block**, identical to disk: no padding, no
    alignment waste, no optional feature. The de-interleave is one pass during
    the bounded init phase (MEM.9), which is exactly GPU.2's "transpose at load
    time".

    Consequence for the planner: **two bindings per weight tensor**, not one,
    which doubles the binding-count arithmetic against
    `maxStorageBuffersPerShaderStage`.
  - A SHA-256 mismatch fails the load loudly.

## The decision

**Model: Qwen3-0.6B. Quantization: Q4_0. Container: GGUF.**

**Target matrix** (WASM.14 — an unstated matrix defaults to "the machine on my
desk"), weakest row first, because the weakest row constrains every sizing
decision:

| Target | WebGPU | Cross-origin isolated | Heap budget |
|---|---|---|---|
| `desktop-chromium-floor` — Chrome/Edge stable, integrated GPU | at the spec defaults | **no** — GitHub Pages cannot set COOP/COEP | **owed by criterion 9** |
| `desktop-chromium-dev` — this development machine | above the defaults | no | not a budget |

Mobile and Safari are **excluded as a stated decision**, not by omission: a
420 MB download against a mobile heap ceiling is not something this packet can
honestly claim, and Safari's three-tier engine and WebGPU support would each
need their own measurement. WASM.14 permits narrowing the matrix and forbids
leaving it unstated.

The floor row's heap budget is left **owed rather than guessed**. WASM.14 is
explicit that any figure not traceable to a specification default or our own
measurement is unverified, and no vendor publishes a per-device heap ceiling.
Criterion 9 produces the number; `-sINITIAL_HEAP` is then set from it (WASM.1),
not from a round value that happened to work here.

Because the floor row is not cross-origin isolated, `performance.now()` resolves
to **100 µs** and `SharedArrayBuffer` is unavailable — the same constraint that
already forces the single-threaded worker. Both belong in every measurement
record (WASM.11).

Read from the model's `config.json`, not assumed:

| | |
|---|---|
| `hidden_size` | 1024 |
| `intermediate_size` | 3072 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 — GQA, 2:1 |
| `head_dim` | **128** |
| `vocab_size` | 151,936 |
| `rope_theta` | 1,000,000 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | **true** |
| `hidden_act` | silu (SwiGLU) |

**`head_dim` is explicit and is not `hidden_size / num_attention_heads`.**
16 × 128 = 2048 against a hidden size of 1024, so `q_proj` is [1024 → 2048] and
`o_proj` is [2048 → 1024]. Prior in-house work hit this same trap on an
unrelated model. Assert it from the file; never derive it.

**Why this model.** Selected on *kernel surface*, not benchmark quality — the
harness is the demonstration. Qwen3 needs the standard decoder set plus QK-norm.
Rejected `gemma-4-E2B`: a 10.25 GB checkpoint despite the "2B effective" name,
carrying vision and audio towers we would never call. Rejected
`gemma-4-26B-A4B`: MoE optimizes compute, while a browser's binding constraint
is resident bytes — 26B parameters must be resident to route 3.8B of work.
Llama-3.2-1B was runner-up: one fewer kernel, ~280 MB more download. Every
kernel written here transfers unchanged to either.

**Limits floor: the WebGPU defaults, requested explicitly.** The harness states
what it requires and asks for exactly that (WASM.10, and WASM.14's "verify
against the WebGPU defaults, not the limits granted on your hardware"). It does
**not** request
the adapter's advertised maxima — doing so converts an explicit capability
contract into an implicit one, encoding the development machine's GPU into the
design and failing elsewhere untraceably. Adapter maxima are reported as
diagnostics and never planned against.

The defaults cost us nothing, which is why this is the right floor rather than
merely the safe one:

| Limit | Default | What we need |
|---|---|---|
| `maxStorageBufferBindingSize` | 128 MiB | 74.2 MiB — the embedding's nibble stream, the largest single binding |
| `maxBufferSize` | 256 MiB | 320 MiB of Q4_0 weights across two suballocated buffers |
| `maxStorageBuffersPerShaderStage` | 8 | 4 per fused dequant-matmul, so two weight tensors per dispatch |

Every tensor fits; nothing else is within an order of magnitude of the limit.
Raising the floor would exclude conformant devices to buy deeper kernel fusion
we cannot yet measure the value of. If a baseline later shows dispatch count
dominates, raising it becomes a stated decision backed by a number.

Because we require the defaults, the harness runs at them on **every** machine
including capable ones — so the configuration under test is the configuration
some device will grant exactly, rather than one only reachable in CI.

## Scope

**In:** GGUF parse, Q4_0 layout, tokenizer, model config, chunked fetch with
SHA-256 and OPFS cache, weight upload to GPU buffers, residency map, readback
verification.

**Explicitly out**, so it cannot drift in:

- Any shader that computes on the weights. Dequantization on the GPU is
  BLLM-003.
- Native Dawn. No kernel yet needs a reference oracle on the GPU.
- Any decode loop, KV cache, or sampling.
- Generic abstractions over quantization formats, tensor types, or backends.
  There is one model and one format; an abstraction over a single
  implementation is ornament.

## The quantized-embedding hazard

Recorded here because it is the defect most likely to reach output silently.

Qwen3-0.6B has `tie_word_embeddings: true`, and the embedding table is
151,936 × 1024 ≈ 155.6M parameters — about **26% of the model**. In GGUF that
table is quantized like any other tensor.

Two failure modes follow, both of which produce **tensors of the correct shape
containing uninterpretable values** — a §3.1 facade that no shape assertion
catches:

1. A token-embedding gather that is not quantization-aware returns packed
   quantized bytes, which the rest of the pipeline treats as floats.
2. The tied output projection, if rebuilt from the embedding tensor without
   carrying its quantization parameters, dispatches an unquantized matmul over
   packed bytes.

Prior in-house work halted a packet on exactly this pair. The mitigation is
structural and starts here: **the tensor index must carry quantization
parameters with every tensor**, so that no consumer can obtain a weight
reference without also obtaining what it takes to interpret it. A design where
quantization is "handled by the loader" and the rest of the pipeline is
quantization-transparent is the design that fails.

## Acceptance Criteria

Each criterion names the guidelines that constrain it. A citation here is a
constraint on *how* the criterion may be satisfied, not decoration: where a
guideline and the criterion disagree, the criterion is wrong.

- [x] **(1) GGUF parse.** A GGUF file parses to a tensor index and metadata map,
      verified against committed fixtures.
      → `SL.con.3`, `ES.103` (no out-of-bounds, no overflow); `P.11`
      (encapsulate the messy construct — one reader, not offset arithmetic
      scattered across callers); `WASM.9` ("never assume the whole asset is
      addressable — validate against the authoritative total size, not against
      whatever window is currently mapped"), which is the rule the `ByteSource`
      seam exists to satisfy.

- [x] **(2) Malformed inputs** fail with named errors and read nothing out of
      bounds: truncated header, tensor offset past EOF, length overflowing the
      file, bad magic, unknown version, tensor count that cannot fit.
      → `WASM.9` ("use checked arithmetic on offsets and sizes — `offset +
      length` on a 32-bit size type can wrap into an in-range value; treat
      downloaded data as untrusted"). On wasm32 this is not hypothetical, which
      is why `range_within` is written as subtraction. `E.27` governs the error
      contract with exceptions off; `SL.con.3` and `ES.103` the arithmetic.

- [x] **(3) Q4_0 dequantization** matches independently computed values
      **bit-exactly** on a fixture block. This is the test oracle for BLLM-003's
      shader.
      → `TLM.6` (a self-test is a correctness gate, not a benchmark): this
      criterion buys correctness and must never be quoted as a performance
      figure. The CPU dequantizer stays in `tests/support/`, never in `core/`,
      so the no-CPU-dequantization invariant is enforced by location.

- [~] **(4) Quantization parameters are inseparable.** Two clauses with
      different natures, tracked separately because bundling them hid that only
      one was ever testable at this stage.
      → `C.40`/`C.41` (a class with an invariant defines a constructor; a
      constructor creates a *fully* initialized object) and `NR.5` (no two-phase
      initialization). A `TensorEntry` that can exist without its type is the
      quantized-embedding hazard above, expressed as a type.

      - **(4a) Every tensor in the index carries its type — done.**
        `gguf_quant_params_test.cpp` quantifies over the whole index rather than
        spot-checking positions, and fixture `unsupported_tensor_type` closes
        the one path by which an entry could have been admitted with a type the
        harness cannot size. `TensorEntry` now has no default constructor and no
        default `type`, so the invariant holds by construction rather than by
        comment — verified by mutation, and by the old default-then-fill pattern
        becoming a compile error.
      - **(4b) No accessor returns a weight without them — open, and not
        blocked by difficulty.** There is no accessor returning weight bytes
        yet; `residency::upload` does not exist. It is also not a property a
        runtime test can establish by calling something. It is satisfied at the
        consumer seam, by `upload` taking type-and-bytes as one indivisible
        argument, and it is verified there — not here.

### Consulted and deliberately not applied

Recorded so a later reader can tell a considered omission from a gap.

- **`WASM.6` (128-bit wasm SIMD).** The de-interleave is a byte shuffle over
  ~420 MB and is a genuine SIMD candidate. Not taken: this packet establishes
  the baseline and explicitly excludes optimization, so vectorising here would
  be tuning against no number. It becomes a stated decision once criterion 9's
  measurement exists.
- **`WASM.13` (a build per feature set).** One feature set — no `shader-f16`,
  no threads, no SIMD — so there is nothing to select between. The
  de-interleaved layout was chosen precisely so no optional feature is needed.
- **`WASM.5` (cross-origin isolation).** Already decided against the project:
  GitHub Pages cannot set COOP/COEP. Recorded in the target matrix as a
  property of the floor row rather than re-litigated here.
- **`WASM.7`/`WASM.8` (startup budget, module size).** This packet does not
  change the module's size or its instantiation path. They bind BLLM-003, which
  adds shaders.
- **`MEM.7` (reserve address space, commit on demand).** Considered and
  **declined**, which is different from inapplicable. Memory64 makes a 64-bit
  heap available to us; we do not take it, because reserving address space for
  bytes whose destination is a GPU buffer optimises a stage that should not
  exist. The guideline fits a workload that keeps its data in the heap. Ours
  does not.
- **`WASM.4` beyond the tokenizer.** `ByteSource` is virtual and called once per
  chunk, so the indirect-call cost amortises over megabytes. Named here so a
  later reader does not "fix" a seam that is deliberate.

## Verification Plan

**The fixture ladder.** Built smallest-primitive-first, each rung gated before
the next, so a failure surfaces at the operation that caused it:

```
Q4_0 block dequant  →  tensor index  →  config  →  tokenizer  →  residency
```

Each fixture carries provenance sufficient to re-derive it: model repository,
snapshot hash, the tool and version that produced it, and the source file it
was cut from. A fixture without that is a number nobody can reproduce.

**The gate splits by whether the math is exact.** Our reference will differ
from us in accumulation order and precision, so byte equality is unattainable
in general and pursuing it would burn the packet:

| Byte-exact | Bounded divergence |
|---|---|
| Q4_0 dequantization (integer unpack × scale) | *(nothing in this packet)* |
| Tokenizer round trip | |
| Tensor index, offsets, shapes | |
| Uploaded bytes vs source bytes | |

Everything in this packet is on the exact side — which is a reason to do the
loader before the kernels, where derived tolerances start.

| Test | What it pins |
|---|---|
| `gguf_parse_test` | Header, metadata KV types, tensor index against fixtures. |
| `gguf_malformed_test` | Each malformed case above, named error, no out-of-bounds read. The file arrives from a CDN; this is the test that matters. |
| `quant_q4_0_test` | Block layout and dequantization, bit-exact against independently computed values. |
| `quant_params_test` | No tensor accessor yields a weight without its quantization parameters. |
| `tokenizer_test` | Round trip over a fixture corpus. |
| `model_config_test` | Parsed config equals the known table; `head_dim` asserted explicitly. |
| `residency_plan_test` | Buffer packing at the 128 MiB floor, at a large granted limit, and at a limit that forces a tensor to span buffers. Pure planning arithmetic — no GPU. |

CI must not download 420 MB, so fixtures are small synthetic GGUFs. The real
file is exercised manually and its SHA-256 recorded, including the heap-bound
assertion, which fixtures cannot establish.

## C++ Architecture Note

- **Core components touched:** New under `src/core/`: `gguf/` (reader, types),
  `quant/` (Q4_0 block layout and parameters), `tokenizer/`, `model/` (config),
  `residency/` (buffer planning and upload).

- **Platform wrappers touched:** `src/wasm/bindings.cpp` gains a load entry
  point and a summary reporter. `web/` gains fetch-with-progress, SHA-256 and
  the OPFS cache — in JavaScript, because the browser supplies both primitives.

- **Public interfaces changed:** None existing. Introduced: `gguf::Reader`,
  `gguf::TensorIndex`, `quant::Q4_0Block`, `quant::Params`,
  `tokenizer::Tokenizer`, `model::Config`, `residency::Plan`,
  `residency::upload`.

- **New files/classes/targets/modules:** Parsing, quant, tokenizer, config and
  the residency **planner** compile in both configurations — none need
  `webgpu.h`, so native source parity holds and CI compiles and tests them. The
  upload path itself is wasm-only, like `device.cpp`.

- **Dependency edges added or removed:** None outward. No third-party
  dependency: GGUF is simple enough to parse directly, and vendoring an
  existing loader would defeat the premise.

- **Ownership/lifetime model:** The parser borrows a
  `std::span<const std::byte>` and owns nothing; the caller owns the bytes. The
  tensor index holds offsets and lengths, **not pointers**, so it cannot
  outlive its buffer into undefined behaviour. GPU buffers are `UniqueHandle`
  as everywhere else.

  Exceptions are off, so **E.27** governs error propagation: one systematic
  `[[nodiscard]]` result type across parsing, config, upload, cache, hash and
  readback. **E.25 and R.1** govern something different — simulating RAII for
  cleanup, which is `UniqueHandle`'s job. An earlier draft cited E.25 for both,
  leaving the error contract underspecified.

- **Test surface independent of UI wrappers:** Everything except the upload
  itself. The residency *planner* is deliberately separated from the *upload*
  so the packing arithmetic — where the interesting failures are — is a pure
  function testable at limits this machine does not have.

- **Why this decomposition is appropriate:** Separating plan from upload is the
  load-bearing choice. It lets us test packing at the 128 MiB floor, at a
  spanning boundary, and at limits no available device reports, without a GPU.
  The alternative — packing decided inside the upload call — would make the
  riskiest arithmetic in the packet reachable only through a browser.

- **What is intentionally not abstracted:** No tensor library, no graph
  abstraction, no backend interface, no quantization-format abstraction. One
  model, one format. The model seam is **not** typed on `WGPUBuffer`; a
  backend-typed seam is a known portability trap.

## C++ Performance Note

- **Target hardware/workload:** Chrome and Edge desktop. Workload is model load
  only: fetch, verify, parse, pack, upload. Not inference.

- **Performance budget:** Load is performance-sensitive by this repo's default.
  Header and index parsing is an operation over metadata, not over 420 MB, and
  should complete in **under 100 ms** on the blessed target. Upload throughput
  and SHA-256 over 420 MB are **measured, not budgeted**, in this packet —
  there is no baseline to budget against yet.

- **Baseline measurement:** None exists; this packet establishes it. Records
  name device, browser build, granted limits, **build configuration**, and
  network versus OPFS — cold and warm differ by the entire download. Timing
  boundaries must be explicit: a metric measuring the wrong interval produces
  plausible figures that get believed.

  **WASM.11 adds four fields the earlier draft omitted**, and without them a
  browser number means nothing: measured clock resolution (the floor row is not
  cross-origin isolated, so expect 100 µs), `crossOriginIsolated` read at
  runtime rather than assumed, whether DevTools was open (if it was, the figure
  is not quotable at all — Chrome tiers the code down), and how many warm-up
  iterations were discarded. Upload and SHA-256 over 420 MB are far above the
  clock quantum, so the clamp does not distort them — but the resolution is
  still recorded, because the *next* thing measured may not be.

  **Peak-heap instrumentation and clean throughput are separate runs from
  separate build configurations** (TLM.2, TLM.6). Heap tracking perturbs the
  transfer numbers, so one execution cannot honestly produce both. This packet
  adds the `wasm-diag` configuration — same optimisation as `wasm-release`,
  instrumentation compiled in — per
  `research/2026-08-31-measurement-build-configurations.md`. No number from it
  is quotable as throughput.

  **TLM.8** governs the other half: the compile-out claim is an aspiration until
  the linked artifact confirms it. `tools/check_diagnostics_excluded.sh` already
  scans the shipped module; it must be extended to the heap-tracking and
  upload-timing symbols this packet adds, or `wasm-release` silently becomes
  `wasm-diag`.

- **Hot paths touched:** None. This is the bounded init phase; MEM.9's shape
  applies — allocate during load, run the steady state with a fixed footprint.
  Weight residency established here *is* that fixed footprint.

- **Allocation/copy/serialization behaviour:** The binding constraint. A 420 MB
  model must not be resident in the WASM heap — not because it could not fit
  (Memory64 would permit it) but because the heap is a waypoint on the route to
  a GPU buffer, and a waypoint that holds the whole payload is the residency
  failure WASM.9 describes. Chunks are uploaded and released; peak heap is
  asserted by test against the real file. Weights are **never** copied into a
  dequantized form on the CPU.

  GPU.1 applies to the upload itself: batch unavoidable transfers, prefer one
  large copy over many small ones. Chunk size is therefore a measured trade-off
  between peak heap and transfer efficiency, not an arbitrary constant — and
  the chosen size must be recorded with the number that justified it.

- **JS/wasm boundary budget (WASM.2):** Crossings during load are
  `ceil(model_bytes / chunk_bytes)` — proportional to model size, which WASM.2
  says needs a written reason. The reason is WASM.9: a single crossing carrying
  the whole model is the residency failure this packet exists to prevent. Each
  crossing moves a **phase** — one whole chunk as a `(pointer, length)` pair —
  never a block, a tensor or a field. Chunk size is therefore the single knob
  trading crossing count against peak heap, and criterion 9 measures both ends
  of it. Reporting (criterion 11) is a constant number of crossings and must
  stay so.

- **Concurrency model:** Single-threaded, unchanged. Fetch and hashing are the
  browser's, in the worker, and the C++ side returns to the event loop between
  chunks rather than blocking (WASM.3).

- **Edge AI runtime/backend/configuration:** None. No inference runtime exists
  and none will be adopted.

- **Before/after measurement plan:** No "after" — nothing is optimized here.
  These measurements become the baseline BLLM-003 is compared against.

- **Accuracy/product-contract risk from performance optimizations:** None; no
  optimization is performed. The accuracy risk that exists is Q4_0
  dequantization correctness and quantization-parameter propagation, which is
  why the first is pinned bit-exactly and the second has its own test.

## Records

- Guidelines consulted before design, per `workflow.md`. `cpp-guidelines`:
  SL.con.3 (bounds errors), SL.4 (type-safe standard library), P.11
  (encapsulate messy constructs) shaped the span-based, offsets-not-pointers
  parser contract, since the file arrives from a CDN; E.25 governs error
  returns with exceptions off. `cpp-perf-guidelines`: MEM.9 (allocate at init,
  fixed steady state) frames residency as the init-phase footprint; MEM.7
  (reserve address space, commit on demand) was considered and **declined** —
  see the invariant above for why the original "wasm32 makes it impossible"
  reasoning expired and what replaced it. GPU.1 (keep
  data on device; batch transfers) governs upload chunking, and GPU.2
  (coalesced lane access; transpose at load time) makes the on-device weight
  layout a decision owed jointly with BLLM-003 rather than a file-order
  default.
- **Full-corpus audit, 2026-09-06** — after the `wasm` category landed, both
  servers were re-queried against every step of this packet and the citations
  above were added per criterion. Four changes were substantive, not
  annotation:

  | Finding | Guideline | Change |
  |---|---|---|
  | Intent promised a **sampled** readback while criterion 7 required every byte | `WASM.9` caveat | Intent corrected; the packet no longer promises less than it tests |
  | No target matrix and no minimum device — the heap bound would have come from this machine | `WASM.14` | Matrix added, weakest row first, with mobile/Safari excluded as a stated decision and the floor row's heap budget marked *owed* rather than guessed |
  | The load path's obligation to yield to the event loop was unstated, leaving Asyncify as an unexamined option | `WASM.3` | Added as an invariant: the C++ side is a step function driven from JS |
  | Measurement records named five conditions but not the four that make a browser number meaningful | `WASM.11` | Clock resolution, `crossOriginIsolated`, DevTools state and discarded warm-up added |

  Two smaller gaps were closed the same way: the JS/wasm crossing budget is now
  written down (`WASM.2` requires a reason for a content-proportional crossing
  count, and this one has one), and `TLM.8`'s artifact scan is named as the
  proof obligation for the new `wasm-diag` split rather than assumed.

  `WASM.6` (SIMD for the de-interleave) is the one live candidate the audit
  found and deliberately did not take — recorded under "consulted and
  deliberately not applied" so it is a decision rather than an oversight.

- **Premise correction, 2026-09-23.** This packet asserted in four places that
  wasm32's 32-bit address space made streaming mandatory. WebAssembly Memory64
  shipped in Chrome M133 and Firefox and was standardised in WebAssembly 3.0 on
  2026-06-13 (~16 GB cap on the web, ~10% throughput cost), so the premise was
  false by the time the packet was approved and none of us checked. The design
  is unchanged and the conclusion is unchanged; what changed is that streaming
  is now recorded as a decision with a reason rather than a constraint with no
  alternative. One use of wasm32 survives untouched — criterion 2's note that
  offset overflow "is not hypothetical on wasm32" is still true, because we are
  still building wasm32; we simply are no longer forced to.

  Found while researching published browser-inference throughput, not by
  reviewing the packet. A premise stated five times and checked zero is the
  shape of error this repo should expect to make again.

- Process note: an earlier draft of BLLM-003 recorded that the performance
  corpus had no GPU material. That was wrong. The MCP server was reading a
  second checkout two commits behind the one where the GPU category landed, so
  a refresh re-indexed a repo that had not moved. Both packets were drafted
  without guidance that existed and was two commands away.
- Model sizes measured from the Hugging Face API, not estimated: Qwen3-0.6B
  1.50 GB, Llama-3.2-1B 2.47 GB, Qwen3-1.7B 4.06 GB, gemma-4-E2B 10.25 GB
  (bf16 checkpoints).
- Method notes from prior in-house inference work:
  `research/2026-08-31-model-port-methodology.md`. The readback measurement
  that settles the decode-loop shape: `research/2026-08-31-gpu-readback-round-trip.md`.
- BLLM-003 follows: native Dawn plus the first GPU kernels, using this packet's
  bit-exact dequantization oracle as the correctness reference.
