import { test } from 'node:test';
import assert from 'node:assert/strict';

import { streamFile } from '../../web/load.js';

// A runtime that records every call in order, reading from `bytes`.
function runtime(bytes, { refuse } = {}) {
  const calls = [];
  let buffer = null;
  return {
    calls,
    size: bytes.length,
    begin: async ({ maxChunk }) => { calls.push(['begin', maxChunk]); },
    read: async ({ offset, length }) => { buffer = bytes.slice(offset, offset + length); calls.push(['read', offset, length]); },
    send: async ({ offset, length }) => {
      calls.push(['send', offset, buffer]);
      assert.equal(buffer.length, length);
      if (refuse !== undefined && offset >= refuse) throw new Error('the device refused it');
    },
    finish: async () => { calls.push(['finish']); return 'finished'; },
  };
}

test('the file is read and sent in order, in chunks that cover it exactly, between begin and finish', async () => {
  const bytes = Uint8Array.from({ length: 1000 }, (_, i) => i % 251);
  const r = runtime(bytes);
  assert.equal(await streamFile({ ...r, chunkBytes: 300 }), 'finished');
  assert.deepEqual(r.calls[0], ['begin', 300]);
  assert.deepEqual(r.calls.at(-1), ['finish']);
  const steps = r.calls.slice(1, -1);
  assert.deepEqual(steps.map(([kind, offset]) => [kind, offset]),
    [['read', 0], ['send', 0], ['read', 300], ['send', 300], ['read', 600], ['send', 600], ['read', 900], ['send', 900]]);
  const joined = new Uint8Array(1000);
  for (const [kind, offset, chunk] of steps) if (kind === 'send') joined.set(chunk, offset);
  assert.deepEqual(joined, bytes);
});

test('the next chunk is not read until the runtime has taken the one before', async () => {
  let inFlight = 0;
  let most = 0;
  const r = runtime(new Uint8Array(1000));
  await streamFile({ ...r, chunkBytes: 100,
    read: async () => { most = Math.max(most, ++inFlight); },
    send: async () => { await new Promise((resolve) => setTimeout(resolve, 1)); inFlight--; } });
  assert.equal(most, 1);
});

test('progress is reported after each chunk is taken, up to the file\'s size', async () => {
  const seen = [];
  await streamFile({ ...runtime(new Uint8Array(250)), chunkBytes: 100, onProgress: (done, total) => seen.push([done, total]) });
  assert.deepEqual(seen, [[100, 250], [200, 250], [250, 250]]);
});

test('a runtime that refuses a chunk stops the pass with its reason, and is not asked to finish', async () => {
  const r = runtime(new Uint8Array(1000), { refuse: 200 });
  await assert.rejects(streamFile({ ...r, chunkBytes: 100 }), /refused it/);
  assert.notDeepEqual(r.calls.at(-1), ['finish']);
});

test('a pass the caller aborts is still finished, so the runtime takes the next', async () => {
  const controller = new AbortController();
  const r = runtime(new Uint8Array(1000));
  const send = r.send;
  r.send = async (chunk) => {
    await send(chunk);
    if (chunk.offset === 200) controller.abort();
  };
  await assert.rejects(streamFile({ ...r, chunkBytes: 100, signal: controller.signal }), { name: 'AbortError' });
  assert.deepEqual(r.calls.at(-1), ['finish']);
});

test('a begin the runtime refuses reads nothing; a skipped pass streams nothing and resolves null', async () => {
  const refused = runtime(new Uint8Array(10));
  refused.begin = async () => { throw new Error('no checked GPU device to load onto'); };
  await assert.rejects(streamFile(refused), /no checked GPU/);
  assert.equal(refused.calls.length, 0);

  const skipped = runtime(new Uint8Array(10));
  skipped.begin = async () => ({ skipped: true });
  assert.equal(await streamFile(skipped), null);
  assert.equal(skipped.calls.length, 0);
});

test("a file stored as ranges is sent at each chunk's published offset, none crossing a gap", async () => {
  const ranges = [[0, 250], [1000, 1120]];
  const r = runtime(new Uint8Array(370));
  assert.equal(await streamFile({ ...r, ranges, chunkBytes: 100 }), 'finished');
  const moved = r.calls.filter(([kind]) => kind === 'read' || kind === 'send')
    .map(([kind, offset, n]) => [kind, offset, kind === 'send' ? n.length : n]);
  assert.deepEqual(moved, [
    ['read', 0, 100], ['send', 0, 100], ['read', 100, 100], ['send', 100, 100], ['read', 200, 50], ['send', 200, 50],
    ['read', 250, 100], ['send', 1000, 100], ['read', 350, 20], ['send', 1100, 20]]);
});

test('ranges that do not add up to the stored file are refused before begin', async () => {
  const r = runtime(new Uint8Array(100));
  await assert.rejects(streamFile({ ...r, ranges: [[0, 50], [80, 120]], chunkBytes: 64 }), /hold 90/);
  assert.deepEqual(r.calls, []);
});
