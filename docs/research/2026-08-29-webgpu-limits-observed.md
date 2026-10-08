# WebGPU limits: defaults, adapter maxima, and what you get if you ask

**Date:** 2026-08-29, **revised twice** — 2026-08-30.
**Context:** BLLM-001 self-check, Chrome on Apple silicon (`apple` / `metal-3`).

> **Superseded 2026-08-31.** The harness no longer requests the adapter's
> advertised maxima: WASM.10 names that as the standard error, since it makes
> the contract implicit and dependent on the development machine. It requires
> the WebGPU defaults explicitly and is granted 8 bindings and 128 MiB bindings
> on every device. The values measured while it requested the maxima remain
> valid as adapter diagnostics; they are no longer what the harness plans
> against.

## What is actually true

WebGPU limits work in three layers, and conflating any two of them produces a
wrong design conclusion. All three measured on the same machine:

| Limit | Spec default | Device, asking nothing | Adapter advertises | Device, asking for adapter maxima |
|---|---|---|---|---|
| `maxBufferSize` | 256 MiB | 256 MiB | 4096 MiB | **4096 MiB** |
| `maxStorageBufferBindingSize` | 128 MiB | 128 MiB | 4096 MiB | **4096 MiB** |
| `maxComputeInvocationsPerWorkgroup` | 256 | 256 | 1024 | **1024** |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 | 65,535 | 65,535 |

- `wgpuAdapterGetLimits` reports what the adapter **could** grant.
- A device gets the **defaults** unless `requiredLimits` asks for more.
- `wgpuDeviceGetLimits` reports what validation actually enforces.

**128 MiB is not a cap. It is the price of not asking.**

## Strategy, and why this one

Requesting an adapter's *own advertised maxima* is always satisfiable — every
adapter can grant what it just said it supports. So it raises the ceiling on
capable hardware at **no portability cost**: a weaker adapter advertises less
and is granted less, and the harness must cope with varying limits either way.

That is what the harness now does. The alternatives:

- **Ask for nothing** — uniform 128 MiB everywhere, simplest to reason about,
  and needlessly discards an order of magnitude on capable hardware.
- **Ask for a fixed floor** — uniform above the floor, but acquisition *fails*
  on anything below it. That is a declared portability floor and a real
  product decision, not a default to drift into.
- **Ask for the adapter's maxima** (chosen) — best available per device, no
  acquisition failures, at the cost of limits that differ between machines.

## Two more limits, measured 2026-08-31

The four limits first recorded here were not the whole surface. Two more
constrain the residency plan, and both are small enough to shape kernel design:

| Limit | Spec default | Granted here |
|---|---|---|
| `maxStorageBuffersPerShaderStage` | 8 | **10** |
| `minStorageBufferOffsetAlignment` | 256 | **256** |

**Ten bindings is a hard, small budget.** Weights are uploaded de-interleaved —
a nibble stream and a scale stream — so **each weight tensor costs two
bindings**. A single shader can therefore see at most four weight tensors once
activations, output and a params buffer are bound. One kernel cannot hold a
whole decoder layer's projections; the forward pass must be several dispatches,
and that is a structural consequence rather than a tuning choice. On a
minimum-spec device the budget is 8, so three weight tensors.

**256-byte binding alignment does not divide an 18-byte block.** If several
tensors share one physical buffer and are bound as sub-ranges, every sub-range
must begin on a 256-byte boundary, so padding between tensors is unavoidable.
`lcm(18, 256)` is 2304 bytes — 128 blocks — so a tensor whose block count is
not a multiple of 128 leaves a gap. The planner must budget that padding rather
than discover it when a binding is rejected.

Note the asymmetry: `maxBufferSize` was granted at 16× the default, while
`maxStorageBuffersPerShaderStage` came back at 10 against a default of 8.
Requesting the adapter's maxima buys enormous headroom on some limits and
almost none on others.

## Consequence for BLLM-002

Weight residency is sized against a number that **varies per device** and is
known only after acquisition. The harness must plan buffer packing from the
device's actual granted limits, not from a compile-time constant, and must
still degrade sensibly on an adapter that advertises only the defaults.

## Correction history, kept deliberately

This note was wrong twice, in opposite directions:

1. Design discussion: "128 MiB and 256 MiB are hard caps; weights must split
   across many buffers regardless of device." Wrong — they are defaults.
2. First revision: "observed limits are 4096 MiB, so residency is a
   portability decision." Right conclusion, wrong mechanism — it read adapter
   capability as capacity already granted, when nothing had been requested.
3. Second revision: "the original figures were right; weights must split."
   Wrong again, over-corrected off the back of finding (2)'s error.

The lesson worth keeping is not any of the three numbers. It is that
*capability advertised*, *capacity granted*, and *capacity requested* are three
different quantities, and every wrong version above came from treating two of
them as one.
