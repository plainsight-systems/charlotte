// DEVELOPMENT ONLY — BENCHMARK. The benchmark's runner (benchmark.js): the
// page's clock, the worker and the page's visibility, around the pure
// scenarios and summary there.

import { Request } from '../protocol.js';
import { RUNS, SCENARIOS, SEED, promptFor, summarize } from './benchmark.js';

// The conditions a browser figure needs (WASM.11). DevTools' being closed is
// the runner's to state; the page cannot see it.
async function conditionsOf(model, verdict, sampling) {
  let version = null;
  try {
    const high = await navigator.userAgentData?.getHighEntropyValues(['fullVersionList', 'platformVersion']);
    version = high ? { brands: high.fullVersionList, platform: high.platformVersion } : null;
  } catch {
    version = null;
  }
  let power = null;
  try {
    const battery = await navigator.getBattery?.();
    power = battery ? { charging: battery.charging, level: battery.level } : null;
  } catch {
    power = null;
  }
  return {
    userAgent: navigator.userAgent,
    version,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    power,
    model: model.id,
    contextOffered: verdict.fit?.contextOffered ?? null,
    sampling: sampling ?? 'greedy',
    seed: SEED,
    devTools: 'closed, as the runner states',
  };
}

// One run: the request to the first TOKEN reply, then to the reply's end,
// and the reply's text: drawn greedily from a fixed seed, the same in every
// browser whose kernels compute the same, so two browsers' texts can be
// compared token for token.
async function timedRun(client, scenario, n, sampling, warmup) {
  let hidden = document.visibilityState !== 'visible';
  const onVisibility = () => {
    if (document.visibilityState !== 'visible') hidden = true;
  };
  document.addEventListener('visibilitychange', onVisibility);
  let first = null;
  let text = '';
  const t0 = performance.now();
  try {
    const { reply } = client.send(Request.GENERATE,
      { prompt: promptFor(scenario, n), sampling, seed: SEED, maxTokens: scenario.maxTokens },
      { onToken: (piece) => { first ??= performance.now(); text += piece; } });
    const result = await reply;
    const end = performance.now();
    return { scenario: scenario.name, warmup, hidden, ttftMs: (first ?? end) - t0, decodeMs: end - (first ?? end),
      tokens: result.tokens, promptTokens: result.promptTokens, reusedTokens: result.reusedTokens, text };
  } finally {
    document.removeEventListener('visibilitychange', onVisibility);
  }
}

// Runs every scenario, one run discarded then RUNS, and logs and keeps the
// summary on window.bllmBenchmark.
export async function runBenchmark({ client, model, verdict }) {
  const sampling = model.policy?.sampling?.default;
  const runs = [];
  let n = 0;
  for (const scenario of SCENARIOS) {
    for (let i = 0; i <= RUNS; i++) runs.push(await timedRun(client, scenario, n++, sampling, i === 0));
  }
  const summary = summarize(runs, await conditionsOf(model, verdict, sampling));
  globalThis.bllmBenchmark = { runs, summary };
  console.table(Object.fromEntries(Object.entries(summary.scenarios).map(([name, s]) => [name, {
    runs: s.runs, hidden: s.leftOutHidden, promptTokens: s.promptTokens, reused: s.reusedTokensMost,
    tokens: s.tokens, ttftMs: s.ttftMs.median, decodeTokS: s.decodeTokS.median }])));
  console.log('benchmark', JSON.stringify(summary));
  return summary;
}
