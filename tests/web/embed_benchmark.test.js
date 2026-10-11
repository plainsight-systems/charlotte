import { test } from 'node:test';
import assert from 'node:assert/strict';

import { EMBED_SCENARIOS, summarizeEmbeds } from '../../web/dev/embed_benchmark.js';

test('the scenarios run from a short query to near the most one encode holds, and the hop encodes nothing', () => {
  const words = (name) => EMBED_SCENARIOS.find((s) => s.name === name).text.split(/\s+/).length;
  assert.ok(words('query') < 20);
  assert.ok(words('paragraph') > 40 && words('paragraph') < 80);
  assert.ok(words('long') > 300 && words('long') < 420);
  assert.equal(EMBED_SCENARIOS.find((s) => s.name === 'hop').task, null);
});

test('the summary keeps measured, visible runs, with each scenario its tokens and its median, least and most', () => {
  const run = (over) => ({ scenario: 'query', warmup: false, hidden: false, ms: 2, tokens: 20, ...over });
  const runs = [run({ warmup: true, ms: 50 }), run({ ms: 1 }), run({ ms: 3 }), run({ ms: 2 }), run({ hidden: true, ms: 9 }),
    run({ scenario: 'hop', ms: 0.2, tokens: null })];
  const { scenarios, conditions } = summarizeEmbeds(runs, { browser: 'x' });
  assert.deepEqual(scenarios.query, { runs: 3, leftOutHidden: 1, tokens: 20, ms: { median: 2, least: 1, most: 3 } });
  assert.equal(scenarios.hop.tokens, null);
  assert.deepEqual(conditions, { browser: 'x' });
});
