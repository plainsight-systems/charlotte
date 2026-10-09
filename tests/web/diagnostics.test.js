import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  describe, kStallMs, phaseOfModelState, report, stalledFor, startSession, withEvent, withModel, withPhase,
  withProgress, worthReporting,
} from '../../web/diagnostics.js';
import { createRecorder } from '../../web/diagnostics_recorder.js';

const qwen = { id: 'qwen3', name: 'Qwen3 0.6B', sizeBytes: 382 };
const begun = () => startSession({ startedAt: 1000, userAgent: 'test', platform: 'mobile' });

test('a load names its phase from the model panel: buffers, then weights, then kernels', () => {
  assert.equal(phaseOfModelState({ phase: 'loading', done: 0, total: null }).phase, 'creating buffers');
  const uploading = phaseOfModelState({ phase: 'loading', done: 10, total: 40 });
  assert.equal(uploading.phase, 'uploading weights');
  assert.equal(uploading.detail, '10 B of 40 B');
  assert.equal(phaseOfModelState({ phase: 'loading', done: 40, total: 40 }).phase, 'building kernels');
  assert.equal(phaseOfModelState({ phase: 'loaded' }).busy, false);
  assert.equal(phaseOfModelState({ phase: 'failed', action: 'load', error: new Error('no') }).error, 'load: no');
});

test('a record left busy, failed, or holding a device event is worth reporting; one at rest is not', () => {
  const rested = withPhase(begun(), { phase: 'loaded', busy: false }, 2000);
  assert.equal(worthReporting(rested), false);
  assert.equal(worthReporting(null), false);
  assert.equal(worthReporting(withPhase(rested, { phase: 'uploading weights', busy: true }, 3000)), true);
  assert.equal(worthReporting(withPhase(rested, { phase: 'failed', busy: false, error: 'load: no' }, 3000)), true);
  assert.equal(worthReporting(withEvent(rested, { kind: 'lost', reason: 'unknown', message: '' }, 3000)), true);
});

test('a later phase clears an earlier failure; the log keeps both', () => {
  const failed = withPhase(begun(), { phase: 'failed', busy: false, error: 'load: no' }, 2000);
  const retried = withPhase(failed, { phase: 'loaded', busy: false }, 3000);
  assert.equal(retried.error, null);
  assert.deepEqual(retried.log.map((e) => e.phase), ['failed', 'loaded']);
  assert.equal(retried.log[0].at, 1000);
});

test('progress moves a phase on without adding to the log, and a busy phase without it is stalled', () => {
  const busy = withPhase(begun(), { phase: 'uploading weights', busy: true }, 2000);
  assert.equal(stalledFor(busy, 2000 + kStallMs), kStallMs);
  const moved = withProgress(busy, '20 of 40 bytes', 15_000);
  assert.equal(moved.log.length, 1);
  assert.equal(stalledFor(moved, 2000 + kStallMs), 2000 + kStallMs - 15_000);
  assert.equal(stalledFor(withPhase(moved, { phase: 'loaded', busy: false }, 16_000), 99_000), 0);
});

test('the log and the events keep only their latest entries', () => {
  let session = begun();
  for (let i = 0; i < 100; ++i) session = withPhase(session, { phase: `p${i}`, busy: false }, 1000 + i);
  for (let i = 0; i < 100; ++i) session = withEvent(session, { kind: 'uncaptured', type: 'validation', message: `${i}` }, 1000);
  assert.equal(session.log.length, 40);
  assert.equal(session.log.at(-1).phase, 'p99');
  assert.equal(session.events.length, 20);
  assert.equal(session.events.at(-1).message, '99');
});

test('a record describes itself for a sentence, and reports as JSON', () => {
  const session = withPhase(withModel(begun(), qwen), { phase: 'uploading weights', busy: true, detail: '1 of 2 bytes' }, 2000);
  assert.equal(describe(session), 'uploading weights (1 of 2 bytes) for Qwen3 0.6B');
  assert.equal(JSON.parse(report(session)).platform, 'mobile');
});

// A Storage as the browser's, kept in a map, counting writes.
function memoryStorage(initial = null) {
  const items = new Map(initial === null ? [] : [['charlotte.diagnostics.v1', initial]]);
  return {
    writes: 0,
    getItem: (key) => items.get(key) ?? null,
    setItem(key, value) { this.writes += 1; items.set(key, value); },
    read: () => JSON.parse(items.get('charlotte.diagnostics.v1')),
  };
}

test('the recorder hands over the last visit, then writes a phase change at once, before its step runs', () => {
  const last = withPhase(begun(), { phase: 'creating buffers', busy: true }, 2000);
  const storage = memoryStorage(JSON.stringify(last));
  let clock = 5000;
  const recorder = createRecorder({ storage, now: () => clock, userAgent: 'ua', platform: null });
  assert.equal(recorder.previous.phase, 'creating buffers');
  assert.equal(storage.read().phase, 'started');   // this visit's record replaced it

  recorder.modelState({ phase: 'loading', model: qwen, done: 0, total: null });
  assert.equal(storage.read().phase, 'creating buffers');
  assert.equal(storage.read().model.name, 'Qwen3 0.6B');
});

test('progress is written at most once a second; phase changes and device events always', () => {
  const storage = memoryStorage();
  let clock = 0;
  const recorder = createRecorder({ storage, now: () => clock, userAgent: 'ua', platform: null });
  recorder.modelState({ phase: 'loading', model: qwen, done: 1, total: 100 });
  const after = storage.writes;
  for (let i = 2; i < 50; ++i) {
    clock += 10;
    recorder.modelState({ phase: 'loading', model: qwen, done: i, total: 100 });
  }
  assert.equal(storage.writes, after);                       // 490 ms of progress, unwritten
  assert.equal(recorder.current().detail, '49 B of 100 B');  // but known
  clock += 600;
  recorder.modelState({ phase: 'loading', model: qwen, done: 60, total: 100 });
  assert.equal(storage.writes, after + 1);
  recorder.event({ kind: 'lost', reason: 'unknown', message: 'gone' });
  assert.equal(storage.read().events[0].message, 'gone');
});

test('a turn is recorded from the prompt to the reply, and its failure kept', () => {
  const storage = memoryStorage();
  const recorder = createRecorder({ storage, now: () => 0, userAgent: 'ua', platform: null });
  recorder.turn('start');
  assert.equal(storage.read().phase, 'reading the prompt');
  recorder.turn('text');
  recorder.turn('text');
  assert.equal(storage.read().phase, 'writing the reply');
  recorder.turn('failed', 'the GPU device was lost');
  assert.equal(storage.read().error, 'turn: the GPU device was lost');
  assert.deepEqual(storage.read().log.map((e) => e.phase), ['reading the prompt', 'writing the reply', 'failed']);
});

test('storage the browser refuses leaves the recorder working, without a record to survive', () => {
  const refusing = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
  const recorder = createRecorder({ storage: refusing, now: () => 0, userAgent: 'ua', platform: null });
  assert.equal(recorder.previous, null);
  recorder.turn('start');
  assert.equal(recorder.current().phase, 'reading the prompt');
  const without = createRecorder({ storage: null, now: () => 0, userAgent: 'ua', platform: null });
  without.turn('start');
  assert.equal(without.current().busy, true);
});
