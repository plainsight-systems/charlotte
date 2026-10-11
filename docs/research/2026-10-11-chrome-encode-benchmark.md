# The encode in Chrome: 13.9 ms for a query, 11 to 13 ms of it fixed

**Date:** 2026-10-11
**Device:** Apple M3 Max, 16 cores, macOS 26.6.2, on mains power. Google
Chrome 151.0.7922.174, WebGPU on Metal.
**Model:** EmbeddingGemma 2, Q8_0; context offered 512.
**Code:** commit `bd9ce6f`, the development site (`tools/assemble_site.sh
--dev`, `?benchmark`): the release module, the clean build's, with
`web/dev/embed_benchmark.js` beside it.

## Why

`runtime/embedder.h` and `web/search.js` estimated the encode's cost before
it ran: about 2 ms for a 16-token query and 60 ms for 512 tokens. The search
panel's header also left one per-call cost without a figure: the worker's
message hops, which decide whether the indexing loop belongs in the page or
in the worker. This is the page's time for each, as the panel's requests
meet it.

## Method

`web/dev/embed_benchmark.js`, as committed: each scenario's text written
into the model's task prompt, as the panel writes it, and sent as one EMBED.
The page's clock runs from the request to its reply. Each scenario discards
one run and keeps twenty.

- **query:** a search query, 21 tokens with its ends.
- **paragraph:** one of the search fixture's paragraphs as a document, 70
  tokens.
- **long:** five chapters of the benchmark's text as a document, 418 tokens.
- **hop:** a CANCEL naming no request, which the worker answers without
  calling the module: two messages and nothing else.

Conditions:

- The page was visible for every run: none was left out hidden.
- The page is not cross-origin isolated, so `performance.now()` is rounded
  to 100 µs.
- DevTools were not open. The page was driven by the Claude in Chrome
  extension.
- The load average was 3.2 at the start. A first run overlapped another
  program's GPU test suite and gave times 1.4 to 2 times these; it was
  discarded, and the suite had finished before this run.

## Results

| Scenario | Tokens | Median | Least–most |
|---|---|---|---|
| query | 21 | 13.9 ms | 13.7–14.4 |
| paragraph | 70 | 15.6 ms | 15.3–15.8 |
| long | 418 | 38.5 ms | 38.3–38.6 |
| hop | — | under 0.1 ms | 0–0.1 |

The discarded runs: 21.5 ms for the first query, the module tiering up and
the pipelines warming; the others within 2% of their medians.

## What it says

- **The worker's hops are negligible.** Two messages take less than the
  clock resolves, under 1% of the cheapest encode. The indexing loop stays
  in the page, as `web/search.js` decided pending this figure.
- **The long text meets its estimate; the short ones do not.** 418 tokens
  took 38.5 ms, against the 39 ms its products alone come to at the rate
  `embedder.h` estimates from. The query took 13.9 ms against an estimate of
  about 2 ms.
- **Most of a short encode is a cost that does not scale with its tokens.**
  Extrapolated to no tokens, the paragraph and long figures give 11 ms and
  the query and paragraph figures 13 ms: 11 to 13 ms an encode. The cost
  walk counted about 1 ms that does not scale: the dispatches' 0.44 ms and
  the readback's 0.5 ms mapping. The hops are not it. What the rest is, this
  measurement cannot say. It needs the encoder's step timed launch by launch,
  natively, as `2026-10-07-forward-pass-profile.md` did for the decoder.
- **It changes the batching argument.** `web/search.js` declined putting
  several passages in one step because the step's fixed cost was about
  0.5 ms of 6.1. Measured, a 70-token passage takes 15.6 ms, 11 to 13 of
  them not scaling with its tokens. If that cost is the step's, as the
  hops' absence suggests, batching would save most of it: a document of a
  thousand such paragraphs takes about 16 s to index, 11 to 13 s of it that
  cost. Whether it is the step's is what the profile above would show.

## The fixed cost found, and the split along K

**Code:** commit after `b609405`, the development site's release module,
`web/dev/embed_benchmark.js` with texts of 2, 3, 4 and 6 chapters beside the
first three, 10 runs a text after one discarded.

### Where the 11 to 13 ms went

Natively, Qwen3's prompt steps of 8, 16 and 32 tokens each took 15 to 17 ms
of GPU time against floors near 1 ms (`make profile`). The time was in the
tiled matrix products: each tile's workgroup walks the whole of K alone,
and a product of few output tiles, such as an output or down projection,
ran 8 to 16 workgroups on the GPU's 40 cores. EmbeddingGemma 2's down,
output and per-layer products, timed alone in Chrome, came to about 9.8 ms
of a 21-token encode's 14.3: the fixed cost above.

`kernels/matmul/matmul.h` now splits such a product along K, a workgroup
for each of the 32 ranges its fixed order of addition already has, and adds
each output's range sums in that order in a second launch. The bits are the
tiled form's, which the GPU tests hold it to. Which steps split is a gate of
two values: the ratio is fitted to every listed model's product shapes timed
alone in Chrome; the cap to the whole encodes below.

### Method

Four builds of the release module, each served at a path of its own on one
origin, so they share the cached model and no build is cached for another:

- **before:** the commit before the split.
- **never:** the split built in, its gate set never to split.
- **cap 128** and **cap 64:** the gate at a ratio of 2, capped at 128 and at
  64 workgroups.

They ran in turn, then in turn reversed, so drift through a session falls on
each alike.

Conditions:

- Chrome 151 on the M3 Max, on mains power; the page visible throughout,
  none left out hidden; not cross-origin isolated, so `performance.now()`
  rounds to 100 µs.
- Another program's GPU test suite ran repeatedly through the first
  attempts and slowed them by up to 1.8 times, the runs drifting slower
  from one to the next. Those runs are discarded; the suite was paused for
  the runs below, and none started during them.

### Results

The medians of the last round, run cap 64, never, before. The round
before it, run before, never, cap 64, agreed within 1% for never and cap
64; before, first in it, ran 3 to 5% slower than here:

| Text | Tokens | before | never | cap 64 |
|---|---|---|---|---|
| query | 21 | 14.3 ms | 14.5 | 7.0 |
| paragraph | 70 | 16.0 | 16.0 | 11.0 |
| 2 chapters | 172 | 22.1 | 22.3 | 19.5 |
| 3 chapters | 254 | 26.7 | 26.7 | 26.0 |
| 4 chapters | 336 | 33.5 | 33.6 | 33.5 |
| long | 418 | 39.1 | 39.1 | 39.2 |
| 6 chapters | 500 | 47.0 | 47.3 | 47.1 |

- **The split halves a short encode.** A query takes 7.0 ms, 2.0 times
  faster; a paragraph 11.0, 1.45 times; 172 tokens 1.13 times. Past 254
  tokens the gate leaves the products tiled and the times are the tiled
  form's.
- **The cap of 64, not 128.** Under 128 the down and global output
  projections split at 418 and 500 tokens, 27 to 33 MB of range sums each
  way a launch, and those encodes ran 1.0 to 1.4 ms slower than never
  splitting, in two rounds; at 336 tokens 128 was 2% faster than 64. A
  launch timed alone keeps its range sums in the GPU's caches, which is why
  the shapes timed alone had favored 128.
- **The tiled form's own cost.** The first build with the split ran the
  tiled form 2 to 3% slower than before at every length, in Chrome; natively
  0.3%. Each invocation's place in its tile was recomputed for each of the
  32 ranges. Worked out once a tile, never matches before within 1%.
- **What remains.** A query is still 7.0 ms against the 2 ms
  `runtime/embedder.h` estimated. The products the gate leaves are now the
  largest share: natively, Qwen3's gate and up take 35% of a 16-token step,
  a product of many tiles over a short K, where splitting lost in every
  measurement.

Natively, Qwen3's prompt steps, GPU time a step, before and after (`make
profile`, DIAGNOSTIC, the final code): 8 tokens 16.8 ms and 6.5; 16 tokens
15.0 and 6.5; 32 tokens 16.9 and 10.1; 64 tokens 20.6 and 15.7; 128 tokens
33.0 and 31.1; 512 tokens 129.1 and 128.7. Its repetition check on decode at
position 1,024, a kernel the split does not touch, read 10.6% against its
10% bound, and 7.1% and 11.5% on other runs: a bound that run-to-run
variation reaches.
