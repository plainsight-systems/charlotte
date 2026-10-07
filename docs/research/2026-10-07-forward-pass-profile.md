# The forward pass's first profile: where Qwen3's step time goes

**Date:** 2026-10-07
**Device:** Apple M3 Max, Metal (`Metal driver on macOS Version 26.6.2 (Build 25G83)`), native Dawn `31e25af254ab572c77054edec4946d2244e184dd`.
**Model:** Qwen3 0.6B Q4_0, SHA-256 `33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb`; context offered 15,181 under the default 2 GiB budget.
**Code:** `bench/forward_profile.cpp` at commit `2cb8a1d`, run by `make profile`; the profiled step in `src/core/kernels/program.h`.

**DIAGNOSTIC.** These are a diagnostic build's numbers, taken natively, and
say where a step's time goes. They are not throughput figures for the page,
which runs in Chrome's own Dawn and Tint (TLM.6). Each finding below that a
change is built on is confirmed there by the page's whole-step rates first.

## Why

Once generation ran end to end in the browser, a 24-token prefill took about
1.2 s natively and a long reply decoded at about 8 ms a token in Chrome,
against `graph/graph.h`'s floors of 32 ms for a 512-token prefill step and
0.94 ms + 0.29 µs × p for a decode step at position p. The profile names the
launches that take the difference.

## Method

The target's Metal adapter offers no timestamps inside a pass, so a launch's
time is what it adds to a prefix of the step, in the step's one pass:
T(k) − T(k − 1), over the embedding, the first layer and the output block.
Every time is the median of 5 runs after 2 warm-ups, or 3 after 1 for a step
over 500 ms, the runs submitted back to back, two outstanding. The first
version waited for each run before the next; the GPU clocked down between
them, and one shape timed 2.99 ms in one workload and 20.9 ms in another.

A repetition check compares the step's layers, timed whole, with its first
layer's launches times its layers. A step that misses by more than 10% has
its per-launch figures discarded; its whole-step time stands.

The smallest interval observed was 5.3 µs, a bound on the timestamps'
resolution. A decode step's launches take 3 to 15 µs each, so their
per-launch figures are coarse; whole steps and prefill launches are not.

## Results

### Whole steps

| Step | GPU time | Spread | Check |
|---|---|---|---|
| Decode at position 0 | 2.77 ms | 2.62–3.26 | 6.4% |
| Decode at 1,024 | 6.16 ms | 6.15–6.27 | 6.7% |
| Decode at 8,192 | 10.93 ms | 10.88–11.02 | 2.0% |
| Prefill, 8 tokens | 20.4 ms | 20.42–20.44 | 1.7% |
| Prefill, 16 tokens | 24.0 ms | 23.96–23.98 | 2.2% |
| Prefill, 32 tokens | 353 ms | 353.0–353.4 | 1.4% |
| Prefill, 64 tokens | 882 ms | 870–890 | failed, 59.8% |
| Prefill, 128 tokens | 1,194 ms | 1,170–1,200 | failed, 37.0% |
| Prefill, 512 tokens | 3,128 ms | 2,430–3,267 | 7.4% |

### Prefill: the 32-token tile

From 16 to 32 tokens a prefill step goes from 24 ms to 353 ms. Each prefill
product launch — `ffn.down`, `ffn.gate_up`, `attention.qkv`,
`attention.output` — goes from about 0.31 ms to about 5.0 ms. 32 tokens is
where `kernels/matmul/matmul.h` switches from its 16-token tile to its
32-token one.

Routing every step over 16 tokens to the 16-token tile instead, everything
else unchanged (CPU wall time a step, median of 4, release build):

| Step | 32-token tile from 17 | 16-token tile from 9 | |
|---|---|---|---|
| 16 tokens | 26.0 ms | 26.0 ms | |
| 32 tokens | 727 ms | 27.5 ms | 26× |
| 128 tokens | 1,134 ms | 60.7 ms | 19× |
| 512 tokens | 2,926 ms | 510 ms | 5.7× |

The 32-token tile decodes each weight half as often as two 16-token tiles,
and is still 26 times slower at 32 tokens. Its invocations each hold a
4-token × 8-output micro-tile, twice the 16-token tile's, its accumulators in
function-scope arrays indexed by loop variables. A collapse of this size at
the switch fits those arrays leaving registers for memory; that is the
hypothesis for the tile's design to test.

Without the 32-token tile, a 512-token step still takes 510 ms against the
32 ms arithmetic floor, and grows faster than its tokens: 61 ms at 128
tokens, 510 at 512.

### Decode: attention grows with position

| Position | `attention.scores` a layer | Its share of the step |
|---|---|---|
| 0 | 0.015 ms | 15.5% |
| 1,024 | 0.131 ms | 59.5% |
| 8,192 | 0.297 ms | 76.2% |

At 8,192 a layer's keys and values are 32 MiB, read in 0.297 ms: about 113
GB/s against the device's 400. At 1,024 they are 4 MiB in 0.131 ms, about 32
GB/s, so attention also carries a large cost that does not scale with keys.
The products run near the weights' bandwidth: the head at 336 to 383 GB/s,
the layers' products at 95 to 330.

### Decode, pipelined and fed: unexplained

128 decode steps fed on the GPU, two outstanding, from position 64, kept the
GPU busy — 2.8% idle — at 40.0 ms a step; a run an hour earlier gave 19.1 ms.
The same decode step repeated back to back at position 0 takes 2.77 ms, and
Chrome decoded a long reply at about 8 ms a token. What makes fed,
successive steps slow natively, and why it varies between runs, is not yet
known.

## What follows

1. The 32-token tile: why it collapses, and whether a wider tile can earn its
   place, or prefill takes the 16-token tile from 9 tokens.
2. Prefill past the tile: 510 ms at 512 tokens, growing faster than its
   tokens.
3. Decode attention's cost at depth, and the part of it that does not scale
   with keys.
4. The pipelined, fed decode anomaly.
