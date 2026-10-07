import { test } from 'node:test';
import assert from 'node:assert/strict';

import { summarize } from '../../web/dev/step_profile.js';

// A turn: one prefill step of `prompt` tokens, then decode steps from that
// position, each `pass` ms in its pass and beginning every `period` ms,
// reported every `report` ms.
function turn({ prompt, decodes, pass, period, report }) {
  const steps = [{ prefill: true, position: 0, tokens: prompt, beginNs: 0, endNs: 20e6, reportMs: 25 }];
  for (let i = 0; i < decodes; i++) {
    const beginNs = 30e6 + i * period * 1e6;
    steps.push({ prefill: false, position: prompt + i, tokens: 1, beginNs, endNs: beginNs + pass(prompt + i) * 1e6,
      reportMs: 40 + i * report });
  }
  return steps;
}

test('the reference is the in-pass mean of positions 64 to 191, the pipeline\'s first two steps left out', () => {
  // In-pass time grows with position; only 64 to 191 count.
  const steps = turn({ prompt: 20, decodes: 300, pass: (p) => 2 + p / 1000, period: 4, report: 4 });
  const s = summarize(steps);
  assert.equal(s.decodeSteps, 298);
  assert.ok(Math.abs(s.reference - (2 + 127.5 / 1000)) < 1e-12);
  assert.deepEqual(s.byPosition.map((b) => [b.from, b.steps]), [[0, 106], [128, 128], [256, 64]]);
  assert.deepEqual(s.prefill, [{ tokens: 20, inPassMs: 20 }]);
});

test('a turn whose decode steps do not cover all of 64 to 191 has no reference', () => {
  const s = summarize(turn({ prompt: 10, decodes: 20, pass: () => 3, period: 4, report: 4 }));
  assert.equal(s.reference, null);
  assert.equal(s.decodeSteps, 18);
  // A prompt of 63 leaves 64 and 65 to fill the pipeline: not covered.
  assert.equal(summarize(turn({ prompt: 63, decodes: 200, pass: () => 3, period: 4, report: 4 })).reference, null);
  // A reply that stops at 190: not covered.
  assert.equal(summarize(turn({ prompt: 20, decodes: 171, pass: () => 3, period: 4, report: 4 })).reference, null);
  assert.equal(summarize(turn({ prompt: 20, decodes: 172, pass: () => 3, period: 4, report: 4 })).reference, 3);
});

test('period, the GPU\'s share outside passes, and the reports\' interval', () => {
  const s = summarize(turn({ prompt: 100, decodes: 50, pass: () => 3, period: 4, report: 5 }));
  assert.ok(Math.abs(s.periodMs - 4) < 1e-9);
  assert.ok(Math.abs(s.outsidePct - 25) < 1e-9);
  assert.ok(Math.abs(s.reportMs - 5) < 1e-9);
  assert.equal(s.pairsLeftOut, 0);
});

test('a pair whose ticks were converted apart is left out, counted, not read as a time', () => {
  const steps = turn({ prompt: 100, decodes: 20, pass: () => 3, period: 4, report: 4 });
  // The last step's conversion moved by 341.6 s, as Dawn's did natively.
  const last = steps.at(-1);
  last.beginNs -= 341.6e9;
  last.endNs -= 341.6e9;
  const s = summarize(steps);
  assert.equal(s.pairsLeftOut, 1);
  assert.ok(Math.abs(s.periodMs - 4) < 1e-9);
  assert.ok(Math.abs(s.outsidePct - 25) < 1e-9);
});
