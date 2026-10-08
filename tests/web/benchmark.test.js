import { test } from 'node:test';
import assert from 'node:assert/strict';

import { RUNS, SCENARIOS, promptFor, summarize } from '../../web/dev/benchmark.js';

test('every run of the benchmark opens with a word of its own', () => {
  const firsts = [];
  for (let n = 0; n < SCENARIOS.length * (RUNS + 1); n++) firsts.push(promptFor(SCENARIOS[0], n).split('.')[0]);
  assert.equal(new Set(firsts).size, firsts.length);
});

test('the long scenario is about a thousand words of prompt, the short a sentence', () => {
  const words = (s) => s.text.split(/\s+/).length;
  assert.ok(words(SCENARIOS.find((s) => s.name === 'long')) > 700);
  assert.ok(words(SCENARIOS.find((s) => s.name === 'short')) < 30);
});

test('the summary keeps measured, visible runs, and gives each scenario its median, least and most', () => {
  const run = (over) => ({ scenario: 'short', warmup: false, hidden: false, ttftMs: 100, decodeMs: 1000, tokens: 101,
    promptTokens: 20, reusedTokens: 0, ...over });
  const runs = [run({ warmup: true, ttftMs: 999 }), run({ ttftMs: 90 }), run({ ttftMs: 110, decodeMs: 500 }),
    run({ ttftMs: 100, decodeMs: 2000 }), run({ hidden: true, ttftMs: 5000 })];
  const { scenarios, conditions } = summarize(runs, { browser: 'x' });
  const s = scenarios.short;
  assert.equal(s.runs, 3);
  assert.equal(s.leftOutHidden, 1);
  assert.deepEqual(s.ttftMs, { median: 100, least: 90, most: 110 });
  // 100 tokens after the first in 1, 0.5 and 2 seconds.
  assert.deepEqual(s.decodeTokS, { median: 100, least: 50, most: 200 });
  assert.equal(s.reusedTokensMost, 0);
  assert.deepEqual(conditions, { browser: 'x' });
});
