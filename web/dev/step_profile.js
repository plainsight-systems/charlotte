// DEVELOPMENT ONLY — DIAGNOSTIC (TLM.6). A turn's steps as the page ran them,
// summarized: where Chrome's time a token goes, on the GPU and between its
// steps. Never quoted as throughput: every step it sees ran profiled.
//
// Served only by the diagnostic site — tools/assemble_site.sh --diag, `make
// serve-diag`, its wasm the wasm-diag build — and selected by ?profile. The
// worker then checks the device with bllm_run_profile_check, asking for
// timestamp queries, and after each load sets the runtime's step observer
// (bllm_observe_steps, src/wasm/bindings.cpp). A turn's steps cross once, at
// its end, each { prefill, position, tokens, beginNs, endNs, reportMs } —
// reportMs the module's clock at the report, read after the turn has run
// its next steps; the page passes them to summarize() and logs the result
// with console.table, keeping each on window.bllmProfiles. A deployed page
// asked for ?profile fails by name: the check is not compiled in.
//
// summarize(steps) — for the turn's decode steps, all but the first two,
// which fill the pipeline. A decode step's time grows with its position, so
// its in-pass time is given by position, never as one median over a turn:
//   reference    the median in-pass time of steps at positions 64 to 191,
//                the native profiler's pipelined run's — 128 steps from 64,
//                a step's sampling as the runtime's — or null when the turn
//                did not reach them
//   byPosition   for each 128-position bucket reached, its steps and their
//                median in-pass time
//   decodeSteps  how many were summarized
// and across steps, from consecutive pairs of them:
//   periodMs     median of beginNs_k − beginNs_(k−1): a step's share of the
//                GPU's clock
//   outsidePct   the GPU's time outside the passes, 1 − Σ in-pass of each
//                pair's first step ÷ Σ the pairs' periods: each step's query
//                resolve and copies, the next step's write, and any idle — a
//                bound on the GPU's idle from above
//   reportMs     median interval between reports, on the module's clock
//   pairsLeftOut pairs whose period ran backward or past a second: the two
//                steps' ticks converted to nanoseconds by different factors
//                (runtime/runtime.h's StepTimes), so their difference is not
//                a time
// and for each prefill step its tokens and in-pass time. The GPU's clock and
// the CPU's are never subtracted from each other (TLM.11).
//
// Conditions the numbers carry (WASM.11): Chrome rounds timestamps to
// 100 µs, so a decode step's in-pass time, a few ms, is good to a few
// percent, and a turn's sum of periods to 0.1 ms a pair; performance.now()
// is rounded to 100 µs without cross-origin isolation, or 5 µs with it;
// DevTools left closed, since opening it de-optimizes the code measured.
//
// What the numbers decide (GPU.10): periodMs near the in-pass time,
// outsidePct small and reportMs near periodMs says the GPU is kept fed, and
// a token costs its step's GPU time. Chrome's code is then slower than
// native's by reference's excess over the native run of the same positions
// and build, made by `make profile` at the same commit
// (bench/forward_profile.cpp) — never against a figure from another commit.
// A large outsidePct says the GPU waits between steps, on the turn loop or
// the browser.
//
// Pure (F.8): no DOM and no clock; tested in tests/web/step_profile.test.js.

const REFERENCE_FIRST = 64;
const REFERENCE_LAST = 191;
const BUCKET = 128;
const FILLING = 2;              // decode steps that fill the pipeline
const LONGEST_PERIOD_MS = 1000; // past this, a pair's ticks were converted apart

const median = (values) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const inPassMs = (step) => (step.endNs - step.beginNs) / 1e6;

export function summarize(steps) {
  const decode = steps.filter((s) => !s.prefill).slice(FILLING);

  const reached = decode.filter((s) => s.position >= REFERENCE_FIRST && s.position <= REFERENCE_LAST);
  const buckets = new Map();
  for (const step of decode) {
    const from = Math.floor(step.position / BUCKET) * BUCKET;
    if (!buckets.has(from)) buckets.set(from, []);
    buckets.get(from).push(inPassMs(step));
  }
  const byPosition = [...buckets.entries()]
    .sort(([a], [b]) => a - b)
    .map(([from, times]) => ({ from, steps: times.length, inPassMs: median(times) }));

  const periods = [];
  const reports = [];
  let pairsLeftOut = 0;
  let pairedInPass = 0;
  let pairedPeriods = 0;
  for (let k = 1; k < decode.length; k++) {
    reports.push(decode[k].reportMs - decode[k - 1].reportMs);
    const period = (decode[k].beginNs - decode[k - 1].beginNs) / 1e6;
    if (period < 0 || period > LONGEST_PERIOD_MS) {
      pairsLeftOut++;
      continue;
    }
    periods.push(period);
    pairedInPass += inPassMs(decode[k - 1]);
    pairedPeriods += period;
  }

  return {
    decodeSteps: decode.length,
    reference: median(reached.map(inPassMs)),
    byPosition,
    periodMs: median(periods),
    outsidePct: pairedPeriods > 0 ? 100 * (1 - pairedInPass / pairedPeriods) : null,
    reportMs: median(reports),
    pairsLeftOut,
    prefill: steps.filter((s) => s.prefill).map((s) => ({ tokens: s.tokens, inPassMs: inPassMs(s) })),
  };
}
