# The forward pass's first profile: where Qwen3's step time goes

**Date:** 2026-10-07
**Device:** Apple M3 Max, Metal (`Metal driver on macOS Version 26.6.2 (Build 25G83)`), native Dawn `31e25af254ab572c77054edec4946d2244e184dd`.
**Model:** Qwen3 0.6B Q4_0, SHA-256 `33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb`; context offered 15,181 under the default 2 GiB budget.
**Code:** `bench/forward_profile.cpp` at commit `4289903`, run by `make profile`; the profiled step in `src/core/kernels/program.h`.

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

A launch's time is a difference of two medians of independent runs, so it
carries the two prefixes' spreads as its uncertainty; a launch under it,
negative ones among them, is below the noise and its figure is not used
here. A first layer's prefix reads only that layer's weights, so it does not
reproduce the whole step's traffic through the GPU's caches. A repetition
check compares the step's layers, timed whole, with its first layer's
launches times its layers; every step below passed it, within 7.3%.

The smallest interval observed was 5.25 µs, a bound on the timestamps'
resolution. In a decode step at position 0 the layers' products take 9 to
13 µs each, close to it; the head, 351 µs, and prefill's launches are far
above it.

## Results

### Whole steps

| Step | GPU time | Spread |
|---|---|---|
| Decode at position 0 | 2.73 ms | 2.54–3.30 |
| Decode at 1,024 | 6.11 ms | 6.10–6.29 |
| Decode at 8,192 | 10.71 ms | 10.70–10.86 |
| Prefill, 8 tokens | 20.3 ms | 20.27–20.54 |
| Prefill, 16 tokens | 24.0 ms | 23.98–24.22 |
| Prefill, 32 tokens | 353 ms | 353.3–353.6 |
| Prefill, 64 tokens | 412 ms | 411.8–412.3 |
| Prefill, 128 tokens | 560 ms | 559.6–559.8 |
| Prefill, 512 tokens | 1,851 ms | 1,843–1,854 |

Between this run and two earlier ones the same day, decode and prefill of up
to 32 tokens agreed within 2%; prefill of 64 to 512 tokens took up to 2.1
times as long in the earlier runs (882, 1,194 and 3,128 ms), each run's own
spread under 1%. What differed between runs is not yet known; a finding
below rests only on what held across all three.

### Prefill: the 32-token tile

From 16 to 32 tokens a prefill step goes from 24 ms to 353 ms, in all three
runs. Each prefill product launch slows 13.7 to 16.0 times: `attention.qkv`
0.124 → 1.703 ms, `attention.output` 0.222 → 3.358, `ffn.gate_up` 0.133 →
2.288, `ffn.down` 0.321 → 5.107. 32 tokens is where
`kernels/matmul/matmul.h` switches from its 16-token tile to its 32-token
one.

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
| 0 | 0.0155 ms | 15.8% |
| 1,024 | 0.130 ms | 59.4% |
| 8,192 | 0.281 ms | 73.5% |

At 8,192 a layer's keys and values are 32 MiB, read in 0.281 ms: about 119
GB/s against the device's 400. At 1,024 they are 4 MiB in 0.130 ms, about 32
GB/s, so attention also carries a large cost that does not scale with keys.
The decode products read their weights at 107 to 383 GB/s: the head at 349
to 383, `attention.qkv` and `ffn.gate_up` near 260 to 340, `ffn.down` and
`attention.output` 107 to 180.

### Decode, pipelined

128 decode steps fed on the GPU, two outstanding, from position 64, took
4.31 ms a step in their passes, with 5.5% of the run's span outside the
passes — each step's query resolve and copies, the next step's write, and
any idle, so the GPU idled at most that. Two earlier runs gave 19.1 and
40.0 ms a step; their steps drew with the sampler's settings unset, top_k 0,
a different draw from a turn's, and are not comparable. Chrome decoded a
long reply at about 8 ms a token.

## What follows

1. The 32-token tile: why it collapses, and whether a wider tile can earn its
   place, or prefill takes the 16-token tile from 9 tokens.
2. Prefill past the tile: 510 ms at 512 tokens, growing faster than its
   tokens; and why long prefill steps vary up to 2.1 times between runs.
3. Decode attention's cost at depth, and the part of it that does not scale
   with keys.
