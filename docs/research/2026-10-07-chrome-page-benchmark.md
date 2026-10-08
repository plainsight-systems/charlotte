# The page in Chrome: 21 ms to a short prompt's first token, 270 tok/s

**Date:** 2026-10-07
**Device:** Apple M3 Max (Mac15,9), 128 GB, macOS 26.6.2, on mains power. Google Chrome 151.0.7922.174, WebGPU on Metal.
**Model:** Qwen3 0.6B Q4_0; context offered 15,172; the model's sampling, greedy, seed 1234.
**Code:** commit `8f0bbca`, the development site (`make serve-dev`, `?benchmark`): the release module, the clean build's, with `web/dev/benchmark.js` beside it.

## Why

Every throughput figure before this one was taken in the Claude desktop
app's browser pane, whose Chromium, Dawn and Tint are not Chrome's
(`2026-10-07-app-chromium-decode-profile.md`). This is the page's
throughput in Chrome itself, under the conditions WASM.11 asks a browser
figure to carry.

## Method

`web/dev/benchmark.js`, unchanged: each scenario's prompt sent to the
runtime as raw text, each run opening with its own word so nothing of it is
cached and every run prefills its whole prompt; one run discarded, then
five kept. Time to first token is the page's clock from the request to the
first token's reply; the decode rate is the tokens after the first over the
time from the first to the reply's end.

- **short:** 19 to 21 prompt tokens, 256 drawn.
- **long:** 950 or 951 prompt tokens, two prefill blocks, 32 drawn.

Conditions, as recorded and as observed:

- The page was visible throughout every run: none was left out hidden.
- The page is not cross-origin isolated, so `performance.now()` is rounded
  to 100 µs, under 0.5% of the shortest time here.
- DevTools were not open. The page was driven by the Claude in Chrome
  extension, which pressed Load and then polled once a second, from the
  page's main thread, for the result.
- The machine was not idle: the load average was 3.3 at the end, on 16
  cores. Each of the ten kept runs fell within 2% of its scenario's median
  on both figures, so the load did not reach the runs unevenly.

## Results

| Scenario | Time to first token, median (least–most) | Decode, median (least–most) |
|---|---|---|
| short, ~20 prompt tokens | 21.4 ms (21.0–21.6) | 269.9 tok/s (267.4–271.9) |
| long, ~950 prompt tokens | 296.1 ms (295.6–296.4) | 251.0 tok/s (250.2–252.0) |

The discarded runs: 73 ms to the short prompt's first token, the module
tiering up and the pipelines warming; 297 ms to the long one's, by then
warm.

- **Decode:** 3.7 ms a token near the start of a reply, 4.0 ms at about
  1,000 positions, attention growing with the cache as the step profile
  found in the app's pane (3.54 ms in-pass at positions 0–255, 3.67 ms at
  512–652).
- **Long prompt:** 951 tokens to their first drawn token in 296 ms, about
  3,200 prompt tokens a second, the first token's draw included.
- **Against the app's pane:** that pane decoded a 552-token reply at 258.3
  tok/s, counted from the turn's start, prefill included, on another
  prompt. The figures are not matched, and say no more than that Chrome is
  not slower.
