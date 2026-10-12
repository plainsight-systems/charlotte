import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';

import { CompanionError, createCompanion, describeCompanion } from '../../web/companion.js';
import { Request } from '../../web/protocol.js';

// The companion's published file: 5,000 bytes, a header of 600 in its first
// range and the span the page needs at its end.
const FILE = randomBytes(5_000);
const RANGES = [[0, 600], [3_000, 5_000]];
const RANGED = Buffer.concat(RANGES.map(([begin, end]) => FILE.subarray(begin, end)));
const VISION = { url: 'https://h/mmproj.gguf', sizeBytes: FILE.length, ranges: RANGES,
  rangesSha256: createHash('sha256').update(RANGED).digest('hex'), tokens: { least: 70, most: 280 } };
const MODEL = { id: 'e', name: 'E', vision: VISION, policy: { load: { memoryBudget: 1e9 } } };

const RUNS = { status: 'read', reached: 'run', fit: null, blockers: [] };

// A cache in memory, as OPFS stands in the page: a file is listed only once
// committed.
function memoryCache() {
  const files = new Map();
  return {
    files,
    file: async (key) => files.get(key) ?? null,
    writer: async (key) => {
      const parts = [];
      return {
        write: async (bytes) => { parts.push(Buffer.from(bytes)); },
        commit: async () => { files.set(key, new Blob([Buffer.concat(parts)])); },
        discard: async () => {},
      };
    },
  };
}

// A worker that answers each request kind as the runtime would, recording
// what it was asked; `load` answers LOAD_COMPANION.
function fakeClient({ verdict = RUNS, load = async () => ({ imageTokens: 280 }) } = {}) {
  const asked = [];
  const answer = async (kind, payload, options) => {
    asked.push([kind, payload]);
    if (kind === Request.PREFLIGHT_COMPANION) return verdict;
    if (kind === Request.LOAD_COMPANION) return load(payload, options);
    if (kind === Request.CANCEL) return true;
    throw new Error(`unexpected ${kind}`);
  };
  return {
    asked,
    request: (kind, payload, options) => answer(kind, payload, options),
    send: (kind, payload, options) => ({ id: asked.length + 1, reply: answer(kind, payload, options) }),
  };
}

function network(bytes = FILE) {
  const fetched = [];
  return {
    fetched,
    fetchRange: async (url, start, end) => {
      fetched.push(['range', start, end]);
      return { bytes: bytes.subarray(start, end).buffer.slice(bytes.byteOffset + start, bytes.byteOffset + end),
        totalSize: bytes.length };
    },
    openRange: async (url, begin, end) => {
      fetched.push(['open', begin, end]);
      return { totalSize: bytes.length, chunks: (async function* () { yield bytes.subarray(begin, end); })() };
    },
  };
}

test('a first ensure checks, downloads the ranges and loads them, in that order', async () => {
  const cache = memoryCache();
  const client = fakeClient();
  const net = network();
  const phases = [];
  const companion = createCompanion({ kind: 'image', client, cache, ...net, onChange: (s) => phases.push(s?.phase) });
  assert.deepEqual(await companion.ensure(MODEL), { imageTokens: 280 });
  assert.deepEqual([...new Set(phases)], ['checking', 'downloading', 'loading', 'loaded']);
  assert.equal(companion.state.encoder, 'image encoder');
  // The header read from the first range alone, sized as the published file.
  assert.deepEqual(net.fetched, [['range', 0, 600], ['open', 0, 600], ['open', 3_000, 5_000]]);
  const [, preflight] = client.asked.find(([kind]) => kind === Request.PREFLIGHT_COMPANION);
  assert.equal(preflight.totalSize, FILE.length);
  assert.equal(preflight.memoryBudget, 1e9);
  const [, load] = client.asked.find(([kind]) => kind === Request.LOAD_COMPANION);
  assert.equal(load.name, `ranges-${VISION.rangesSha256}.gguf`);
  assert.equal(load.media, 'image');
  assert.equal(preflight.media, 'image');
  assert.deepEqual(load.entry.ranges, RANGES);
  assert.deepEqual(Buffer.from(await (await cache.file(`ranges-${VISION.rangesSha256}`)).arrayBuffer()), RANGED);
  // Loaded: a second ensure asks nothing, and answers as the load did.
  const before = client.asked.length;
  assert.deepEqual(await companion.ensure(MODEL), { imageTokens: 280 });
  assert.equal(client.asked.length, before);
});

test('a cached companion is checked from its copy and loaded without a fetch', async () => {
  const cache = memoryCache();
  cache.files.set(`ranges-${VISION.rangesSha256}`, new Blob([RANGED]));
  const net = network();
  const companion = createCompanion({ kind: 'image', client: fakeClient(), cache, ...net });
  await companion.ensure(MODEL);
  assert.deepEqual(net.fetched, []);
  assert.equal(companion.state.phase, 'loaded');
});

test('a verdict short of run is refused, by name, with nothing downloaded', async () => {
  const verdict = { status: 'read', reached: 'describe', fit: null,
    blockers: [{ stage: 'fit', detail: 'needs 330 MB beside the model; 120 MB are left' }] };
  const net = network();
  const companion = createCompanion({ kind: 'image', client: fakeClient({ verdict }), cache: memoryCache(), ...net });
  await assert.rejects(companion.ensure(MODEL), (error) => error instanceof CompanionError && /120 MB/.test(error.message));
  assert.equal(companion.state.phase, 'failed');
  assert.deepEqual(companion.state.verdict.blockers, verdict.blockers);
  assert.deepEqual(net.fetched, [['range', 0, 600]]);
});

test('ranges that do not match their hash are refused and nothing is cached', async () => {
  const changed = Buffer.from(FILE);
  changed[4_000] ^= 1;
  const cache = memoryCache();
  const client = fakeClient();
  const companion = createCompanion({ kind: 'image', client, cache, ...network(changed) });
  await assert.rejects(companion.ensure(MODEL), /SHA-256/);
  assert.equal(cache.files.size, 0);
  assert.ok(!client.asked.some(([kind]) => kind === Request.LOAD_COMPANION));
});

test('release mid-load cancels the load, and its late answer changes nothing', async () => {
  let finishLoad;
  const client = fakeClient({ load: () => new Promise((resolve) => { finishLoad = resolve; }) });
  const states = [];
  const companion = createCompanion({ kind: 'image', client, cache: memoryCache(), ...network(), onChange: (s) => states.push(s) });
  const ensured = companion.ensure(MODEL);
  while (finishLoad === undefined) await new Promise((resolve) => setTimeout(resolve, 0));
  companion.release();
  finishLoad({ imageTokens: 280 });
  await assert.rejects(ensured, { name: 'AbortError' });
  assert.equal(companion.state, null);
  assert.equal(states.at(-1), null);
  assert.ok(client.asked.some(([kind]) => kind === Request.CANCEL));
  // A later ensure loads it again rather than taking the abandoned load as done.
  finishLoad = undefined;
  const again = companion.ensure(MODEL);
  while (finishLoad === undefined) await new Promise((resolve) => setTimeout(resolve, 0));
  finishLoad({ imageTokens: 280 });
  await again;
  assert.equal(companion.state.phase, 'loaded');
});

test('each state reads as the search panel shows it', () => {
  assert.equal(describeCompanion(null), '');
  const image = { encoder: 'image encoder' };
  assert.equal(describeCompanion({ ...image, phase: 'checking' }), 'Checking the image encoder');
  assert.equal(describeCompanion({ ...image, phase: 'downloading', received: 81e6, total: 226e6 }),
    'Downloading the image encoder: 81 MB of 226 MB');
  assert.equal(describeCompanion({ ...image, phase: 'loading', received: 0, total: null }), 'Loading the image encoder');
  assert.equal(describeCompanion({ ...image, phase: 'loaded' }), 'Image encoder loaded');
  assert.equal(describeCompanion({ ...image, phase: 'failed', message: 'x' }), 'The image encoder did not load');
  assert.equal(describeCompanion({ encoder: 'audio encoder', phase: 'loaded' }), 'Audio encoder loaded');
});

test("an audio companion reads the entry's `audio` and is listed as the model's audio", async () => {
  const cache = memoryCache();
  const client = fakeClient();
  const listed = [];
  const recording = { ...cache, writer: async (key) => {
    const w = await cache.writer(key);
    return { ...w, commit: async (d) => { listed.push(d.name); await w.commit(d); } };
  } };
  const companion = createCompanion({ kind: 'audio', client, cache: recording, ...network() });
  await companion.ensure({ ...MODEL, vision: undefined, audio: VISION });
  assert.deepEqual(listed, ['E audio']);
  const [, load] = client.asked.find(([kind]) => kind === Request.LOAD_COMPANION);
  assert.equal(load.media, 'audio');
  assert.equal(companion.state.encoder, 'audio encoder');
  assert.throws(() => createCompanion({ kind: 'video', client, cache }), /no companion of kind/);
  // Refused, it names what it would have embedded.
  const verdict = { status: 'read', reached: 'describe', fit: null, blockers: [{ stage: 'fit', detail: 'no room' }] };
  const refused = createCompanion({ kind: 'audio', client: fakeClient({ verdict }), cache: memoryCache(), ...network() });
  await assert.rejects(refused.ensure({ ...MODEL, vision: undefined, audio: VISION }),
    /^CompanionError: recordings cannot be embedded here: no room$/);
});
