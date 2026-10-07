# Decode in Chrome, step by step: the GPU is kept fed

**Date:** 2026-10-07
**Device:** Apple M3 Max; the Claude desktop app's Chromium browser pane, WebGPU on Metal.
**Model:** Qwen3 0.6B Q4_0, the same file as the forward-pass profile; context offered 15,181.
**Code:** commit `980df48`: the step profile (`web/dev/step_profile.js`, `runtime/runtime.h`) over the kernels of `479b2d9`, `5d2dc0f` and `e465632`.

**DIAGNOSTIC.** The profiled turn ran every step with timestamp queries on the
diagnostic site (`make serve-diag`, `?profile`); its numbers say where a
token's time goes, not how fast the page runs (TLM.6). The one throughput
figure here is the clean build's, from the page's own count.

## Why

The forward-pass profile (`2026-10-07-forward-pass-profile.md`) put a
pipelined decode step at 4.31 ms of GPU time natively while the page
decoded a long reply at about 8 ms a token, and nothing said whether
Chrome's step was slower on the GPU or the GPU waited between steps.

## Method

One turn: "Write a 400-word story about a lighthouse keeper who finds a
message in a bottle.", Thinking on, the model's sampling. 29 prompt tokens
in one prefill step, then 623 decode steps to position 652. The summary
leaves out the first two decode steps, which fill the pipeline, and gives
the in-pass time by position. The native reference is `make profile`'s
pipelined run over the same kernels: 128 steps from position 64, 3.228 ms
a step in its pass on average.

Chrome rounds each timestamp to 65.5 µs (2^16 ns, from the values seen),
so a step's in-pass time is good to about 2%. One pair of steps was left
out of the cross-step figures: its GPU times ran backward, the conversion of
ticks to nanoseconds having changed between them, as natively (below).

## Results

| | Chrome | Native, same kernels |
|---|---|---|
| In-pass time, positions 64–191 | 3.54 ms | 3.23 ms |
| GPU period between steps | 3.93 ms | — |
| GPU time outside passes | 8.0% | 7.1% |
| Interval between reports, CPU | 3.90 ms | 3.48 ms |

In-pass time by position: 3.54 ms from 0 to 255, 3.60 ms from 256 to 511,
3.67 ms from 512 to 652.

The interval between reports equals the GPU's period, and the GPU spends
8% of it outside the passes — each step's query resolve and copies, the
next step's write and any idle, an upper bound on the idle. Chrome keeps
the GPU fed: a token costs its step's GPU time and 8%. Its step's pass is
about 10% longer than native Dawn's; the kernels, model and positions
being the same, that is Chrome's Dawn and Tint against the release the
native build pins.

The clean build, unprofiled, on the same prompt: 552 tokens at 258.3 tok/s,
counted by the page from the turn's start, prefill included — about
3.9 ms a token. The page decoded at 125 to 136 tok/s before the prefill
micro-tile, the decode rows a set and the 64-key chunks; there is no step
profile from then, so how the earlier 8 ms divided between the GPU and the
loop is not known.

### Dawn's conversion of GPU ticks changes within a process

Natively, a turn's last step reported a pass that began 341.6 s before the
step ahead of it, its own end less its beginning a normal 0.68 ms. It
happened in a few of the GPU test's runs; the two whose times were recorded
were both 341.6 s apart, which at the timestamps' size is a factor
1.07 × 10⁻⁴ apart. The two times of one step are converted alike;
those of two steps may not be. `runtime/runtime.h`'s StepTimes says so, and
the summary leaves such pairs out and counts them.

## What follows

1. The combine at depth: at 8,192 positions it takes 0.039 ms a layer,
   12% of a native step, each invocation folding 129 partials in a chain
   of loads from device memory.
2. Prefill attention: 20.7% of a 512-token step natively.
3. The decode products of few rows: 150 to 210 GB/s against the larger
   products' 300 and more, with 512 workgroups each.
