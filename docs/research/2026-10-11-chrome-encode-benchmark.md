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
