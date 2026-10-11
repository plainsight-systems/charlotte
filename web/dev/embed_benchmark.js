// DEVELOPMENT ONLY — BENCHMARK (TLM.6). The page's encode times for a model
// that embeds, as benchmark.js measures a generator's throughput: on the
// clean module, from the development site, selected by ?benchmark once the
// model loads (embed_benchmark_run.js runs it).
//
// Each scenario is one request the search panel makes (web/search.js),
// written into the model's task prompt as the panel writes it, timed on the
// page's clock from the request to its reply:
//   - query: a search query, about 20 tokens.
//   - paragraph: one of the search fixture's paragraphs as a document,
//     about 70 tokens: what indexing sends a passage at a time.
//   - long: five chapters of the benchmark's text as a document, about 450
//     tokens, near the most one encode holds; and two, three, four and six
//     chapters, about 170 to 500, the lengths a split along K's gate decides
//     between (kernels/matmul/matmul.h).
//   - hop: a CANCEL naming no request, which the worker answers without
//     calling the module (wasm_runtime.js): the page to the worker and
//     back, two messages and nothing else. Not an encode: it calibrates the
//     message hops search.js's cost walk could not count, so a paragraph's
//     time splits into its hops and the rest.
// The encode keeps nothing between requests, so the same text each run is
// a fresh encode. Each scenario: one run discarded, the module tiering up
// and the GPU's clocks rising (WASM.11), then RUNS. A run during which the
// page was ever hidden is left out and counted.
//
// summarize(runs, conditions) — pure, tested in
// tests/web/embed_benchmark.test.js — for each scenario the runs kept and
// left out, the tokens its text made, and the median, least and most of
// its time; the conditions as benchmark.js records them.

import { spread } from './benchmark.js';

const CHAPTER = 'The lighthouse keeper climbed the spiral stairs each evening at dusk, counting the steps as he ' +
  'went. There were one hundred and twelve of them, worn smooth in the middle by a century of boots. At the top ' +
  'he trimmed the wick, polished the great lens until it shone, and watched the first ships of the night turn ' +
  'toward the harbour, their lanterns small and steady against the dark water.';

const PARAGRAPH = 'Sourdough rises without any yeast from a packet. A starter of flour and water, fed every day, ' +
  'keeps wild yeasts and lactic acid bacteria alive; mixed into the dough, they give off carbon dioxide, which ' +
  'the gluten traps in small pockets, and the loaf swells over several hours before it is baked.';

export const EMBED_SCENARIOS = [
  { name: 'query', task: 'query', text: 'how many steps does the keeper climb to reach the lamp?' },
  { name: 'paragraph', task: 'document', text: PARAGRAPH },
  { name: 'long', task: 'document',
    text: Array.from({ length: 5 }, (_, i) => `Chapter ${i + 1}. ${CHAPTER}`).join(' ') },
  ...[2, 3, 4, 6].map((n) => ({ name: `chapters${n}`, task: 'document',
    text: Array.from({ length: n }, (_, i) => `Chapter ${i + 1}. ${CHAPTER}`).join(' ') })),
  { name: 'hop', task: null, text: null },
];

export const EMBED_RUNS = 10;   // measured, after one discarded

const median = (values) => spread(values).median;

// runs: { scenario, warmup, hidden, ms, tokens }, tokens null for the hop.
export function summarizeEmbeds(runs, conditions) {
  const scenarios = {};
  for (const name of [...new Set(runs.map((r) => r.scenario))]) {
    const mine = runs.filter((r) => r.scenario === name && !r.warmup);
    const kept = mine.filter((r) => !r.hidden);
    scenarios[name] = {
      runs: kept.length,
      leftOutHidden: mine.length - kept.length,
      tokens: median(kept.filter((r) => r.tokens !== null).map((r) => r.tokens)),
      ms: spread(kept.map((r) => r.ms)),
    };
  }
  return { scenarios, conditions };
}
