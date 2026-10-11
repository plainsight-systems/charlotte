// Axis H. A loaded model's companion file — the vision encoder EmbeddingGemma
// 2's images need (runtime/vision.h) — checked, downloaded and loaded the
// first time the page needs it, beside the model already loaded. The model
// controller's lifecycle (model_controller.js), for a file that is never
// chosen alone and is read in part.
//
// createCompanion({ client, cache, onChange, onCacheChanged, loadPolicyOf,
// fetchRange, openRange }) returns { ensure(model) → Promise, state,
// release() }; the two fetches are fetch.js's unless given, and
// loadPolicyOf(model) gives the model's load policy as the model controller
// has it, whose memory budget the companion is judged and planned within:
//   - ensure(model) with the loaded model's catalog entry, whose `vision`
//     names the companion (catalog.js). Resolves once the companion is
//     loaded; at once if it already is; with the same promise while it
//     loads; rejects with what stopped it. Each step changes `state`, which
//     onChange renders where the search panel shows it: { phase: 'checking'
//     | 'downloading' | 'loading' | 'loaded' | 'failed', received, total,
//     verdict, message }.
//   - Checking: its header from the first range — from the cache if it is
//     there, else the network (fetch.js's range fetch) — sent as
//     PREFLIGHT_COMPANION, the verdict naming each stage it cannot reach, as
//     a model's verdict does (stages.js): the projector type unsupported,
//     the plan past what the budget leaves, the text model not one it
//     serves.
//   - Downloading: each range fetched in order and streamed into one cache
//     file, its SHA-256 computed over the ranges' bytes in order and
//     checked against rangesSha256 before the file is listed (download.js,
//     opfs.js, sha256.js) — a part of a file verified as a whole file is,
//     against a pinned value. Cached under its own key,
//     `ranges-<rangesSha256>`, beside the model's.
//   - Loading: LOAD_COMPANION with the cache file's name and the ranges; the
//     worker streams the cache file in chunks and gives each chunk its
//     offset in the published file: a chunk at place p of the cache file
//     lies at range r's begin + (p − the bytes of ranges before r)
//     (load.js), never across a range's end.
//   - release() on a new model, as the model controller abandons a step:
//     work aborted, a late answer ignored. The runtime releases the
//     companion with the model it serves (runtime/vision.h).
//
// What it asks of the other files:
//   - download.js: downloading a list of ranges by range fetch into one
//     cache file, hashing as the bytes arrive, the whole file's path being
//     its one range; cacheKey for a companion's ranges.
//   - load.js: streaming a cache file of ranges with each chunk's published
//     offset.
//   - preflight.js: the prefix read from the first range, its total size
//     the published file's, so the reader validates the index as for the
//     whole file.
//   - protocol.js and worker.js: PREFLIGHT_COMPANION and LOAD_COMPANION,
//     their payloads the model's plus the ranges, the runtime's crossings
//     (runtime/vision.h).
//
// What it costs: one range fetch of the header, 58,656 bytes, on a first
// check; then the ranges, 226.4 MB, streamed once into the cache and hashed
// as they arrive; on each load the cache file read once in 16 MiB chunks, as
// a model's (load.js). Nothing per encode.
//
// Verification: tests/web/companion.test.js, with a fake client and cache:
// the phases in order; a cached companion loaded without a fetch; a
// mismatched hash refused and nothing listed; release() mid-download leaving
// nothing listed and ignoring the late answer; the chunk offsets across a
// gap, none crossing it.

import { downloadRanges, rangesCacheKey } from './download.js';
import { fetchRange as fetchRangeOf, openRange as openRangeOf } from './fetch.js';
import { LOAD_CHUNK_BYTES } from './load.js';
import { cachedFileName } from './opfs.js';
import { preflight, rangesOfFile, withinFirstRange } from './preflight.js';
import { Request } from './protocol.js';
import { nextStage, offersDownload } from './stages.js';

// A companion this build or device cannot run, with the verdict that says why.
export class CompanionError extends Error {
  constructor(verdict) {
    const { next, blocking } = nextStage(verdict);
    super(`images cannot be embedded here: ${blocking.map((b) => b.detail).join('; ') || `stopped before ${next}`}`);
    this.name = 'CompanionError';
    this.verdict = verdict;
  }
}

export function createCompanion({
  client, cache, onChange, onCacheChanged, loadPolicyOf = (model) => model.policy?.load,
  fetchRange = fetchRangeOf, openRange = openRangeOf,
}) {
  let state = null;
  let step = null;
  let loaded = null;    // the model whose companion is loaded, by id
  let pending = null;   // { id, promise } while one loads

  const show = (next) => {
    state = next;
    onChange?.(state);
  };

  async function run(model, signal) {
    const { vision } = model;
    const key = rangesCacheKey(vision);
    const memoryBudget = loadPolicyOf(model)?.memoryBudget;
    show({ phase: 'checking' });
    const file = await cache.file(key);
    const source = file !== null ? rangesOfFile(file) : (start, end) => fetchRange(vision.url, start, end, { signal });
    const verdict = await preflight({
      fetchRange: withinFirstRange(source, vision.ranges[0][1], vision.sizeBytes),
      readIndex: (bytes, totalSize) => {
        const copy = bytes.slice().buffer;
        return client.request(Request.PREFLIGHT_COMPANION, { bytes: copy, totalSize, vision, memoryBudget },
          { transfer: [copy] });
      },
    });
    signal.throwIfAborted();
    if (!offersDownload(verdict)) throw new CompanionError(verdict);

    if (file === null) {
      const total = vision.ranges.reduce((sum, [begin, end]) => sum + (end - begin), 0);
      show({ phase: 'downloading', verdict, received: 0, total });
      await downloadRanges({ name: `${model.name} images`, vision, openRange, cache, signal,
        onProgress: (received) => { if (!signal.aborted) show({ phase: 'downloading', verdict, received, total }); } });
      onCacheChanged?.();
      signal.throwIfAborted();
    }

    show({ phase: 'loading', verdict, received: 0, total: null });
    const { id, reply } = client.send(Request.LOAD_COMPANION,
      { name: cachedFileName(key), vision, memoryBudget, maxChunk: LOAD_CHUNK_BYTES },
      { onProgress: ({ done, total }) => { if (!signal.aborted) show({ phase: 'loading', verdict, received: done, total }); } });
    const cancel = () => client.request(Request.CANCEL, { target: id }).catch(() => {});
    signal.addEventListener('abort', cancel, { once: true });
    try {
      await reply;
    } finally {
      signal.removeEventListener('abort', cancel);
    }
    signal.throwIfAborted();
    loaded = model.id;
    show({ phase: 'loaded', verdict });
  }

  function ensure(model) {
    if (loaded === model.id) return Promise.resolve();
    if (pending?.id === model.id) return pending.promise;
    step?.abort();
    step = new AbortController();
    const { signal } = step;
    const promise = run(model, signal).catch((error) => {
      if (!signal.aborted) show({ phase: 'failed', verdict: error.verdict ?? null, message: error.message });
      throw error;
    }).finally(() => {
      if (pending?.promise === promise) pending = null;
    });
    pending = { id: model.id, promise };
    return promise;
  }

  return {
    ensure,
    get state() { return state; },
    release() {
      step?.abort();
      step = null;
      loaded = null;
      pending = null;
      show(null);
    },
  };
}
