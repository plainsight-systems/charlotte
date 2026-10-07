// Owns the runtime and answers the page's requests. Runs off the main thread
// so the page stays responsive.
//
// Contract 11, the boundary: this file and src/wasm/bindings.cpp are the only
// two places JavaScript and C++ meet. The crossings are preflight a header
// prefix; load a cached model, which the worker streams through the module
// itself (load.js); generate from a rendered prompt and the turn's policy;
// and cancel, which stops a load or a generation by naming its request. Text
// comes back one message per piece the reply completes, and a load's progress
// one message a chunk (WASM.2). A failure's code, where the runtime gives one,
// crosses with its message, so the page can act on it by name.
//
// A plain Web Worker, deliberately: it needs no SharedArrayBuffer, so it works
// on GitHub Pages, which cannot set the COOP/COEP headers that cross-origin
// isolation requires.

import { Notice, Reply, Request } from './protocol.js';

const flags = new URLSearchParams(self.location.search);
const postDevice = (device) => self.postMessage({ kind: Notice.DEVICE, device });

// ?fake-runtime selects the development stand-in, which is absent from the
// deployed site; asking for it there fails at this import, by name.
const runtimeModule = flags.has('fake-runtime') ? './dev/fake_runtime.js' : './wasm_runtime.js';
const runtimePromise = import(runtimeModule)
  .then(({ createRuntime }) => createRuntime({ onDevice: postDevice, runBench: flags.has('bench') }));
runtimePromise.catch((error) =>
  postDevice({ ok: false, stage: 'runtime', error: String(error?.message ?? error) }));

// Loads in progress, by request id, so a CANCEL naming one stops it.
const loads = new Map();

// One handler per request kind. Each receives the runtime, the request, and
// the ways to answer before it is done — `token` streams text, `progress`
// reports a load's progress; it returns the request's result or throws.
const handlers = {
  [Request.PREFLIGHT]: (runtime, { bytes, totalSize }) => runtime.preflight(bytes, totalSize),
  [Request.LOAD]: async (runtime, { id, name, indexBytes, confirmed, maxChunk }, { progress }) => {
    const controller = new AbortController();
    loads.set(id, controller);
    try {
      return await runtime.loadFromCache({ name, indexBytes, confirmed, maxChunk, onProgress: progress,
        signal: controller.signal });
    } finally {
      loads.delete(id);
    }
  },
  [Request.GENERATE]: (runtime, { id, prompt, sampling, seed }, { token }) =>
    runtime.generate({ id, prompt, sampling, seed, onText: token }),
  [Request.CANCEL]: (runtime, { target }) => {
    const load = loads.get(target);
    if (load === undefined) return runtime.cancel(target);
    load.abort();
    return true;
  },
};

self.addEventListener('message', async ({ data: request }) => {
  const reply = (message) => self.postMessage({ id: request.id, ...message });
  const handler = handlers[request.kind];
  if (handler === undefined) {
    reply({ kind: Reply.FAILED, error: { stage: 'protocol', message: `unknown request "${request.kind}"` } });
    return;
  }
  try {
    const runtime = await runtimePromise;
    const token = (text) => reply({ kind: Reply.TOKEN, text });
    const progress = (value) => reply({ kind: Reply.PROGRESS, progress: value });
    reply({ kind: Reply.DONE, value: await handler(runtime, request, { token, progress }) });
  } catch (error) {
    reply({ kind: Reply.FAILED, error: { stage: request.kind, message: String(error?.message ?? error) } });
  }
});
