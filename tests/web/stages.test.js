import { test } from 'node:test';
import assert from 'node:assert/strict';

import { STAGES, nextStage, offersDownload, reaches } from '../../web/stages.js';

const verdict = {
  reached: 'download',
  blockers: [
    { stage: 'describe', detail: 'architecture "qwen3" is not supported' },
    { stage: 'run', detail: 'format Q4_0 is not supported' },
    { stage: 'fit', detail: 'the fit stage is not implemented in this build' },
  ],
};

test('the stages are in the order preflight judges them', () => {
  assert.deepEqual(STAGES, ['read', 'download', 'describe', 'fit', 'upload', 'run']);
});

test('a verdict reaches its stage and every stage before it, and nothing after', () => {
  assert.equal(reaches(verdict, 'read'), true);
  assert.equal(reaches(verdict, 'download'), true);
  assert.equal(reaches(verdict, 'describe'), false);
  assert.equal(reaches(verdict, 'upload'), false);
});

test('the blockers of the next stage are separated from the rest', () => {
  const { next, blocking, later } = nextStage(verdict);
  assert.equal(next, 'describe');
  assert.deepEqual(blocking.map((b) => b.stage), ['describe']);
  assert.deepEqual(later.map((b) => b.stage), ['run', 'fit']);
});

test('a model that runs has no next stage', () => {
  assert.deepEqual(nextStage({ reached: 'run', blockers: [] }), { next: undefined, blocking: [], later: [] });
});

test('a download is offered only for a model nothing blocks from running here', () => {
  assert.equal(offersDownload({ reached: 'run', blockers: [] }), true);
  assert.equal(offersDownload(verdict), false);
  assert.equal(offersDownload({ reached: 'describe', blockers: [{ stage: 'fit', detail: 'needs 801 MiB' }] }), false);
});
