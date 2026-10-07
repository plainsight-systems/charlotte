// The chosen model's lifecycle: check it, download it, load it. Holds the one
// state object the model panel renders, and re-renders on every change.
// Choosing another model, or cancelling, abandons the step in progress: its
// work is aborted and any late answer is ignored.
//
// A model already in the cache is checked from its copy, not the network,
// and a fresh download is read back from the cache before it is offered.
// Preflight and load carry the model's load policy, the same at both
// (web/protocol.js).

import { cacheKey, downloadModel } from './download.js';
import { confirmDuplicates } from './duplicates.js';
import { fetchRange, openDownload } from './fetch.js';
import { LOAD_CHUNK_BYTES } from './load.js';
import { renderModel } from './model_view.js';
import { cachedFileName, requestPersistence } from './opfs.js';
import { preflight, rangesOfFile } from './preflight.js';
import { Request } from './protocol.js';

export function createModelController({ element, client, cache, onLoaded, onCacheChanged }) {
  let state = null;
  let step = null;

  const show = (next) => {
    state = next;
    renderModel(element, state, actions);
  };

  // Starts a step, abandoning the one before, and returns its signal.
  const begin = () => {
    step?.abort();
    step = new AbortController();
    return step.signal;
  };

  // Preflight over any range source: the network, or a cached file.
  const readVerdict = (fetchRangeOf) => preflight({
    fetchRange: fetchRangeOf,
    readIndex: (bytes, totalSize) => {
      const copy = bytes.slice().buffer;
      return client.request(Request.PREFLIGHT, { bytes: copy, totalSize }, { transfer: [copy] });
    },
  });

  async function choose(model) {
    const signal = begin();
    show({ phase: 'checking', model });
    try {
      const file = await cache.file(cacheKey(model));
      const verdict = await readVerdict(file !== null
        ? rangesOfFile(file)
        : (start, end) => fetchRange(model.url, start, end, { signal }));
      if (!signal.aborted) show({ phase: file !== null ? 'cached' : 'downloadable', model, verdict });
    } catch (error) {
      if (!signal.aborted) show({ phase: 'failed', model, action: 'check', error });
    }
  }

  async function download() {
    const { model, verdict } = state;
    const signal = begin();
    show({ phase: 'downloading', model, verdict, received: 0, total: model.sizeBytes ?? null });
    requestPersistence();
    try {
      await downloadModel({
        model, cache, signal, openDownload,
        onProgress: (received, total) => {
          if (!signal.aborted) show({ ...state, received, total });
        },
      });
      onCacheChanged();
      const copy = await readVerdict(rangesOfFile(await cache.file(cacheKey(model))));
      if (copy.tensorCount !== verdict.tensorCount || copy.architecture !== verdict.architecture) {
        throw new Error('the downloaded copy does not read the same as the file on the server');
      }
      if (!signal.aborted) show({ phase: 'cached', model, verdict: copy });
    } catch (error) {
      if (!signal.aborted) show({ phase: 'failed', model, action: 'download', error });
    }
  }

  async function load() {
    const { model, verdict } = state;
    const signal = begin();
    show({ phase: 'loading', model, verdict, done: 0, total: null });
    try {
      const file = await cache.file(cacheKey(model));
      if (file === null) throw new Error('the file is no longer in the cache');
      // The duplicates whose bytes really match, compared before the worker
      // opens the file, which locks it while it reads.
      const confirmed = await confirmDuplicates({ file, candidates: verdict.fit?.duplicates ?? [], signal });
      const { id, reply } = client.send(Request.LOAD,
        { name: cachedFileName(cacheKey(model)), indexBytes: verdict.indexBytes, confirmed, maxChunk: LOAD_CHUNK_BYTES },
        {
          // A diagnostic build reads every byte back after loading them.
          onProgress: ({ phase, done, total }) => {
            if (!signal.aborted) show({ phase: phase === 'check' ? 'verifying' : 'loading', model, verdict, done, total });
          },
        });
      const cancel = () => client.request(Request.CANCEL, { target: id }).catch(() => {});
      signal.addEventListener('abort', cancel, { once: true });
      const { check } = await reply;
      signal.removeEventListener('abort', cancel);
      if (signal.aborted) return;
      show({ phase: 'loaded', model, verdict, check });
      onLoaded(model, verdict);
    } catch (error) {
      if (!signal.aborted) show({ phase: 'failed', model, action: 'load', error });
    }
  }

  async function remove() {
    const { model, verdict } = state;
    await cache.remove(cacheKey(model));
    onCacheChanged();
    show({ phase: 'downloadable', model, verdict });
  }

  const actions = {
    download,
    load,
    remove,
    cancel: () => {
      step?.abort();
      show({ phase: 'downloadable', model: state.model, verdict: state.verdict });
    },
    retry: () => choose(state.model),
  };

  return { choose };
}
