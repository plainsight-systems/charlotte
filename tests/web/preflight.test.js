import { test } from 'node:test';
import assert from 'node:assert/strict';

import { FIRST_FETCH_BYTES, MAX_INDEX_BYTES, PreflightError, preflight, rangesOfFile, withinFirstRange }
  from '../../web/preflight.js';

// A file of `totalSize` bytes whose index ends at `indexEnd`, served by a
// fake fetchRange that records every range asked for, and read by a fake
// reader that behaves like the real one: it reads, or asks for `indexEnd`.
function fakeFile({ totalSize, indexEnd }) {
  const fetched = [];
  return {
    fetched,
    fetchRange: async (start, end) => {
      fetched.push([start, end]);
      return { bytes: new ArrayBuffer(Math.min(end, totalSize) - start), totalSize };
    },
    readIndex: async (bytes) => (bytes.length >= indexEnd
      ? { status: 'read', accepted: false, rejections: [] }
      : { status: 'need-bytes', bytesNeeded: indexEnd }),
  };
}

test('a small index is read from the first fetch alone', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: 6e6 });
  const verdict = await preflight(file);
  assert.equal(verdict.status, 'read');
  assert.equal(verdict.totalSize, 500e6);
  assert.deepEqual(file.fetched, [[0, FIRST_FETCH_BYTES]]);
});

test('a larger index fetches only the bytes it lacks, contiguously', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: 20e6 });
  await preflight(file);
  assert.equal(file.fetched[0][0], 0);
  for (let i = 1; i < file.fetched.length; i++) {
    assert.equal(file.fetched[i][0], file.fetched[i - 1][1], 'each fetch starts where the last ended');
  }
});

test('a file smaller than the first fetch is read whole, without asking past its end', async () => {
  const file = fakeFile({ totalSize: 1000, indexEnd: 900 });
  await preflight(file);
  assert.deepEqual(file.fetched, [[0, FIRST_FETCH_BYTES]]);
});

test('an index larger than the limit is refused by name', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: MAX_INDEX_BYTES + 1 });
  await assert.rejects(preflight(file), (error) =>
    error instanceof PreflightError && /index is larger than/.test(error.message));
});

test('an unreadable file is reported with the reader\'s reason', async () => {
  await assert.rejects(preflight({
    fetchRange: async () => ({ bytes: new ArrayBuffer(4), totalSize: 4 }),
    readIndex: async () => ({ status: 'unreadable', error: 'not a GGUF file: magic mismatch' }),
  }), /magic mismatch/);
});

test('a server that sends nothing ends the check instead of looping', async () => {
  await assert.rejects(preflight({
    fetchRange: async () => ({ bytes: new ArrayBuffer(0), totalSize: 100 }),
    readIndex: async () => assert.fail('nothing to read'),
  }), /sent no bytes/);
});

test('a file on this device serves its own ranges, with its own size', async () => {
  const file = new Blob([Uint8Array.from({ length: 100 }, (_, i) => i)]);
  const { bytes, totalSize } = await rangesOfFile(file)(10, 20);
  assert.equal(totalSize, 100);
  assert.deepEqual([...new Uint8Array(bytes)], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
});

test('the verdict says how many of the first bytes the index needed, to be sent again at load', async () => {
  const small = await preflight(fakeFile({ totalSize: 500e6, indexEnd: 6e6 }));
  assert.equal(small.indexBytes, FIRST_FETCH_BYTES);   // read from the first fetch, whole
  const file = fakeFile({ totalSize: 500e6, indexEnd: 20e6 });
  const large = await preflight(file);
  assert.equal(large.indexBytes, file.fetched.at(-1)[1]);   // every byte fetched
  assert.ok(large.indexBytes >= 20e6);
});

test("a companion's index is read from its first range alone, sized as the published file", async () => {
  // The cached ranges: 60,000 bytes of header, then the vision span.
  const cached = fakeFile({ totalSize: 226e6, indexEnd: 58_656 });
  const verdict = await preflight({ fetchRange: withinFirstRange(cached.fetchRange, 60_000, 554_821_024),
    readIndex: cached.readIndex });
  assert.equal(verdict.totalSize, 554_821_024);
  assert.equal(verdict.indexBytes, 60_000);
  assert.deepEqual(cached.fetched, [[0, 60_000]]);
});

test('an index past the first range is refused, by name, without fetching past it', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: 70_000 });
  await assert.rejects(preflight({ fetchRange: withinFirstRange(file.fetchRange, 60_000, 500e6),
    readIndex: file.readIndex }), (error) => error instanceof PreflightError && /first range/.test(error.message));
  assert.deepEqual(file.fetched, [[0, 60_000]]);
});
