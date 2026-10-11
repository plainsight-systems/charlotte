import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';

import { DownloadError, cacheKey, downloadModel, downloadRanges, rangesCacheKey } from '../../web/download.js';

// A cache that records what a download did with its writer. It stands in for
// OPFS, which Node does not have; the behaviour under test is the download's
// — what it writes, and when it commits or discards.
function recordingCache() {
  const record = { written: [], committed: null, discarded: false, key: null };
  return {
    record,
    writer: async (key) => {
      record.key = key;
      return {
        write: async (bytes) => { record.written.push(bytes); },
        commit: async (description) => { record.committed = description; },
        discard: async () => { record.discarded = true; },
      };
    },
  };
}

function serving(bytes, { chunk = 1000, totalSize = bytes.length } = {}) {
  return async () => ({
    totalSize,
    chunks: (async function* () {
      for (let i = 0; i < bytes.length; i += chunk) yield bytes.subarray(i, i + chunk);
    })(),
  });
}

const FILE = randomBytes(10_000);
const SHA256 = createHash('sha256').update(FILE).digest('hex');
const MODEL = { name: 'M', url: 'https://h/m.gguf', sizeBytes: FILE.length, sha256: SHA256 };

test('a complete, matching download is committed with its hash', async () => {
  const cache = recordingCache();
  const progress = [];
  await downloadModel({ model: MODEL, cache, openDownload: serving(FILE),
    onProgress: (received, total) => progress.push([received, total]) });
  assert.equal(cache.record.key, `sha256-${SHA256}`);
  assert.deepEqual(Buffer.concat(cache.record.written), FILE);
  assert.equal(cache.record.committed.sha256, SHA256);
  assert.equal(cache.record.discarded, false);
  assert.deepEqual(progress.at(-1), [FILE.length, FILE.length]);
});

test('a file that does not match its published hash is discarded, never committed', async () => {
  const cache = recordingCache();
  const corrupt = Buffer.from(FILE);
  corrupt[5000] ^= 1;
  await assert.rejects(downloadModel({ model: MODEL, cache, openDownload: serving(corrupt) }),
    (error) => error instanceof DownloadError && /SHA-256/.test(error.message));
  assert.equal(cache.record.committed, null);
  assert.equal(cache.record.discarded, true);
});

test('a download that ends early is discarded', async () => {
  const cache = recordingCache();
  await assert.rejects(downloadModel({ model: MODEL, cache,
    openDownload: serving(FILE.subarray(0, 4000), { totalSize: FILE.length }) }), /ended after/);
  assert.equal(cache.record.discarded, true);
});

test('a server whose file size differs from the list is refused before writing', async () => {
  const cache = recordingCache();
  await assert.rejects(downloadModel({ model: MODEL, cache,
    openDownload: serving(FILE, { totalSize: FILE.length + 1 }) }), /model list says/);
  assert.equal(cache.record.key, null);
});

test('cancelling discards what was written', async () => {
  const cache = recordingCache();
  const controller = new AbortController();
  await assert.rejects(downloadModel({ model: MODEL, cache, openDownload: serving(FILE),
    signal: controller.signal,
    onProgress: (received) => { if (received >= 3000) controller.abort(); } }),
  (error) => error.name === 'AbortError');
  assert.equal(cache.record.discarded, true);
  assert.equal(cache.record.committed, null);
});

test('a pasted model, with no published hash, is keyed by its URL', () => {
  const key = cacheKey({ url: 'https://h/pasted.gguf' });
  assert.match(key, /^url-[0-9a-f]{32}$/);
  assert.equal(cacheKey({ url: 'https://h/pasted.gguf' }), key);
  assert.notEqual(cacheKey({ url: 'https://h/other.gguf' }), key);
});

// A companion's file, of which the download reads two ranges.
const COMPANION = randomBytes(5_000);
const RANGES = [[0, 600], [3_000, 5_000]];
const RANGED = Buffer.concat(RANGES.map(([begin, end]) => COMPANION.subarray(begin, end)));
const VISION = { url: 'https://h/mmproj.gguf', sizeBytes: COMPANION.length, ranges: RANGES,
  rangesSha256: createHash('sha256').update(RANGED).digest('hex'), tokens: { least: 70, most: 280 } };

function servingRanges(bytes, { chunk = 700, totalSize = bytes.length, short = 0 } = {}) {
  const asked = [];
  const open = async (url, begin, end) => {
    asked.push([url, begin, end]);
    const part = bytes.subarray(begin, end - short);
    return { totalSize, chunks: (async function* () {
      for (let i = 0; i < part.length; i += chunk) yield part.subarray(i, i + chunk);
    })() };
  };
  return { asked, open };
}

test("a companion's ranges are fetched in order into one file, committed under their hash", async () => {
  const cache = recordingCache();
  const server = servingRanges(COMPANION);
  const progress = [];
  await downloadRanges({ name: 'E images', entry: VISION, openRange: server.open, cache,
    onProgress: (received, total) => progress.push([received, total]) });
  assert.deepEqual(server.asked, [[VISION.url, 0, 600], [VISION.url, 3_000, 5_000]]);
  assert.equal(cache.record.key, rangesCacheKey(VISION));
  assert.equal(cache.record.key, `ranges-${VISION.rangesSha256}`);
  assert.deepEqual(Buffer.concat(cache.record.written), RANGED);
  assert.deepEqual(cache.record.committed, { name: 'E images', url: VISION.url, sizeBytes: RANGED.length,
    ranges: RANGES, sha256: VISION.rangesSha256 });
  assert.deepEqual(progress.at(-1), [RANGED.length, RANGED.length]);
});

test('ranges whose bytes do not match their hash are discarded, never committed', async () => {
  const cache = recordingCache();
  const changed = Buffer.from(COMPANION);
  changed[3_100] ^= 1;
  await assert.rejects(downloadRanges({ name: 'E', entry: VISION, openRange: servingRanges(changed).open, cache }),
    (error) => error instanceof DownloadError && /SHA-256/.test(error.message));
  assert.equal(cache.record.committed, null);
  assert.equal(cache.record.discarded, true);
});

test('a range that ends early, or a server whose file differs in size, is discarded', async () => {
  for (const [server, pattern] of [[servingRanges(COMPANION, { short: 1 }), /range ended/],
    [servingRanges(COMPANION, { totalSize: 4_999 }), /model list says/]]) {
    const cache = recordingCache();
    await assert.rejects(downloadRanges({ name: 'E', entry: VISION, openRange: server.open, cache }), pattern);
    assert.equal(cache.record.committed, null);
    assert.equal(cache.record.discarded, true);
  }
});
