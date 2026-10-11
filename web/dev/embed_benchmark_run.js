// DEVELOPMENT ONLY — BENCHMARK. The encode benchmark's runner
// (embed_benchmark.js): the page's clock, the worker and the page's
// visibility, around the pure scenarios and summary there.

import { Request } from '../protocol.js';
import { taskPrompt } from '../task_prompt.js';
import { conditionsOf } from './benchmark_run.js';
import { EMBED_RUNS, EMBED_SCENARIOS, summarizeEmbeds } from './embed_benchmark.js';

async function timedRun(client, model, scenario, warmup) {
  let hidden = document.visibilityState !== 'visible';
  const onVisibility = () => {
    if (document.visibilityState !== 'visible') hidden = true;
  };
  document.addEventListener('visibilitychange', onVisibility);
  try {
    const t0 = performance.now();
    if (scenario.task === null) {
      await client.request(Request.CANCEL, { target: 0 });
      return { scenario: scenario.name, warmup, hidden, ms: performance.now() - t0, tokens: null };
    }
    const { tokens } = await client.request(Request.EMBED,
      { text: taskPrompt(model.prompts ?? null, scenario.task, scenario.text) });
    return { scenario: scenario.name, warmup, hidden, ms: performance.now() - t0, tokens };
  } finally {
    document.removeEventListener('visibilitychange', onVisibility);
  }
}

// Runs every scenario, one run discarded then EMBED_RUNS, and logs and
// keeps the summary on window.bllmBenchmark.
export async function runEmbedBenchmark({ client, model, verdict }) {
  const runs = [];
  for (const scenario of EMBED_SCENARIOS) {
    for (let i = 0; i <= EMBED_RUNS; i++) runs.push(await timedRun(client, model, scenario, i === 0));
  }
  // The generator's conditions, less its sampling: an encode draws nothing.
  const { sampling, seed, ...conditions } = await conditionsOf(model, verdict, null);
  const summary = summarizeEmbeds(runs, conditions);
  globalThis.bllmBenchmark = { runs, summary };
  console.table(Object.fromEntries(Object.entries(summary.scenarios).map(([name, s]) => [name, {
    runs: s.runs, hidden: s.leftOutHidden, tokens: s.tokens, ms: s.ms.median, least: s.ms.least, most: s.ms.most }])));
  console.log('embed benchmark', JSON.stringify(summary));
  return summary;
}
