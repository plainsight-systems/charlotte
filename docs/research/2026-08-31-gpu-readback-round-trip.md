# GPU readback round trip: what a per-token read actually costs

**Date:** 2026-08-31
**Device:** Chrome on Apple silicon (`apple` / `metal-3`), WebGPU via emdawnwebgpu.
**Code:** `src/core/gpu/readback_bench.{h,cpp}`, run from the page with `?bench`.

> **Superseded 2026-10-07.** The sampled token now stays on the GPU and is
> read back a step behind (`src/core/sampler/sampler.h`): a Qwen3 decode step
> counts 1.3 to 2.1 ms, so the 0.5 ms map would add 24 to 38% to every
> token. The measurement stands; the decision it settled here does not.

## Why this was measured before designing the decode loop

Experience from prior in-house inference work says the throughput gap in a
harness is often in the **orchestration around the kernels rather than the
kernels themselves**, and that a per-token device→host synchronisation is a
prime suspect — even on hardware where host and device share memory and there
is no bus to cross.

WebGPU's equivalent is worse on paper: reading a sampled token back means
`wgpuBufferMapAsync`, an async round trip through the JavaScript event loop. If
that cost is large, a decode loop that reads its token back each step is capped
by it regardless of kernel quality, and the architecture has to change to hide
it.

So the number was measured rather than assumed.

## Method

200 iterations of each phase, against a deliberately trivial dispatch (64
elements, one workgroup) so the measurement isolates round-trip overhead rather
than compute:

- **sequential** — submit, map, read, unmap; each iteration waits for the last.
  This is exactly the per-token decode shape.
- **batched** — 200 dispatches submitted, then a single readback.

Timing comes from an injected clock (`emscripten_get_now`, supplied by the
wrapper), so `core/` keeps no platform dependency. The mapped value is read
rather than discarded.

## Result

Two consecutive runs, same machine:

| | run 1 | run 2 |
|---|---|---|
| sequential min | 0.300 ms | 0.300 ms |
| **sequential median** | **0.500 ms** | **0.500 ms** |
| sequential p95 | 0.800 ms | 0.800 ms |
| sequential max | 6.500 ms | 6.300 ms |
| sequential mean | 0.594 ms | 0.583 ms |
| batched, 200 dispatches + 1 readback | 8.70 ms | 9.90 ms |
| batched, per dispatch | 0.0435 ms | 0.0495 ms |

Stable to the reported precision across runs.

## What it means

**A serialized GPU round trip costs ~0.5 ms median, ~0.8 ms at p95.** Submit
cost alone is ~0.045 ms, so nearly all of it is the map and the event-loop hop.

That caps a read-back-per-token design at roughly **2000 tokens/s** — an order
of magnitude above any rate a 0.6B model will reach in a browser. At a plausible
20–50 tok/s the per-token round trip is **1–3% of the token budget**, not a
ceiling.

**So the per-token-sync concern does not transfer as a blocker.** It bites when
the sync costs a meaningful fraction of a short token time *and* prevents
compute/host overlap. Here the absolute cost is 0.5 ms against a token that will
take far longer.

The 10–11× "cost of serializing" shown on the page is measured against a trivial
dispatch and **overstates the case for real work**. The transferable figure is
the absolute 0.5 ms, not the ratio: once a token costs tens of milliseconds of
real compute, the round trip is additive overhead, not a multiplier.

## The decision this settles

**Read the sampled token back per step. Do not build a speculative pipeline.**

This is the inversion worth recording. In prior in-house work, an async decode
pipeline added specifically to recover throughput introduced a correctness
defect in cache state that survived multiple review rounds — and the regression
suite could not catch it, because the test configuration's token limit meant the
defective path never executed.

We have measured that we do not need that complexity here. Taking it anyway
would import the risk without the reason.

If a future model or a faster device changes the arithmetic, this measurement is
cheap to re-run and the decision can be revisited on evidence.

## Caveats

- One device, one browser. The blessed targets include Edge and other GPUs; the
  absolute number will differ, though the order of magnitude is unlikely to
  change enough to matter.
- The `max` of ~6.5 ms shows a real tail — scheduling or GC. Irrelevant at 1–3%
  of a token budget, but it is not a flat 0.5 ms.
- Measured with a trivial dispatch. This establishes the *overhead floor*, not
  behaviour when a real forward pass is in flight; overlap under load is still
  unmeasured.

## Related lessons this does not settle

Owed by the packets that touch them. See
`2026-08-31-model-port-methodology.md` for the porting-method notes.

- **Greedy sampling must use `argmax` directly.** Converting logits to
  log-probabilities on a greedy path costs a full-vocabulary reduction every
  token, when subtracting a per-row constant cannot change an argmax.
  Qwen3-0.6B's vocabulary is 151,936.
- **Detokenization must be incremental.** Re-decoding the whole sequence each
  token is O(N²) and the cost is invisible until the sequence is long.
- **There must be exactly one decode loop.** Duplicated decode paths mean every
  fix lands several times or drifts.
- **Quantized embeddings with a tied output projection break any "the loader
  handles quantization" assumption.** A gather over packed weights is not
  quantization-aware, and a tied head rebuilt from the embedding can silently
  discard quantization state — producing tensors of the right shape with
  uninterpretable contents. Qwen3-0.6B has `tie_word_embeddings: true` and keeps
  ~26% of its parameters in the embedding, so this applies directly.
- **Timing boundaries must be explicit.** A throughput metric that measures the
  wrong interval produces plausible numbers that get believed. You also cannot
  pipeline phases you cannot delimit.
