// DEVELOPMENT ONLY — DIAGNOSTIC (TLM.6). A turn's steps as the page ran them,
// summarized: where Chrome's time a token goes, on the GPU and between its
// steps. Never quoted as throughput: every step it sees ran profiled.
//
// Served only by the diagnostic site — tools/assemble_site.sh --diag, `make
// serve-diag`, its wasm the wasm-diag build — and selected by ?profile. The
// worker then checks the device with bllm_run_profile_check, asking for
// timestamp queries, and after each load sets the runtime's step observer
// (bllm_observe_steps, src/wasm/bindings.cpp). Each step's report crosses as
// { prefill, position, tokens, beginNs, endNs }, the worker adding its own
// performance.now() at the report, reportMs; at the turn's end it posts the
// turn's steps to the page, which passes them to summarize() and logs the
// result with console.table, keeping each on window.bllmProfiles. A deployed
// page asked for ?profile fails by name: the check is not compiled in.
//
// summarize(steps) — for the turn's decode steps, all but the first two,
// which fill the pipeline:
//   decodeSteps  how many were summarized
//   inPassMs     median of endNs − beginNs: a step's GPU time in its pass
//   periodMs     median of beginNs_k − beginNs_(k−1): a step's share of the
//                GPU's clock
//   outsidePct   the GPU's time outside the passes, 1 − Σ in-pass ÷ (last
//                endNs − first beginNs): each step's query resolve and
//                copies, the next step's write, and any idle — a bound on
//                the GPU's idle from above
//   reportMs     median interval between reports, on the CPU's clock
// and for each prefill step its tokens and in-pass time. The GPU's clock and
// the CPU's are never subtracted from each other (TLM.11).
//
// Conditions the numbers carry (WASM.11): Chrome rounds timestamps to
// 100 µs, so a decode step's in-pass time, about 4 ms, is good to about
// 2.5%, and a turn's GPU span to 0.1 ms; performance.now() is rounded to
// 100 µs without cross-origin isolation, or 5 µs with it; DevTools left
// closed, since opening it de-optimizes the code measured.
//
// What the numbers decide (GPU.10): periodMs near inPassMs, outsidePct small
// and reportMs near periodMs says the GPU is kept fed, and a token costs its
// step's GPU time — Chrome's code then slower than native's 4.31 ms a step
// at position 64 (docs/research/2026-10-07-forward-pass-profile.md) by
// inPassMs's excess. A large outsidePct says the GPU waits between steps, on
// the turn loop or the browser.
//
// Pure (F.8): no DOM and no clock; tested in tests/web/step_profile.test.js.
