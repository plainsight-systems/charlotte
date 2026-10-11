// The runtime: the C++ module, behind the operations the worker offers the
// page. Every call into C++ goes through here.
//
// A load is one operation, loadFromCache: it opens the cached file with a
// FileSystemSyncAccessHandle, reads the index prefix into the heap, begins,
// reads each chunk straight into the chunk buffer begin reported through a
// fresh view of the heap each time, since growing the heap detaches any
// view kept (WASM.1), and finishes. A diagnostic module then checks every
// byte the same way, through the same handle; the clean module has no check
// (`canCheck`). The handle is closed however the load ends, so the file is
// unlocked for the page to read again. Preflight and load carry the model's
// load policy, written as the module reads it (src/wasm/bindings.cpp): a
// field unset as its stand-in, the stop texts' bytes in one buffer with an
// array of their lengths.
//
// generate encodes the prompt as UTF-8 into the module's memory, calls
// bllm_generate, and frees the bytes once the call returns, the module
// having encoded them; each piece of text arrives through bllmOnText with the
// call's id, read from the module's memory at once, and goes to onText; the
// turn's answer resolves the promise, or its failure rejects it, carrying the
// module's code. One turn at a time, as the module allows. cancel names a
// generate's request: the runtime keeps the call id each request was given,
// and asks the module to end that turn; a request that is not running is
// answered false.
//
// embed, for a model that encodes, writes the text into the module's memory
// as generate does and calls bllm_embed; the vector arrives once through
// bllmOnVector with the call's id, a view of the module's memory copied at
// once, and resolves with the tokens the text made; a failure rejects,
// carrying the module's code — "text-too-long" with both counts. Load's
// answer carries the embedding's length for such a model. With `image`, the
// module places the image its slot holds where the text writes <|image|>.
//
// A companion (companion.js) is judged and loaded as a model is, over the
// ranges of its file the cache holds back to back: its ranges cross as
// pairs of doubles, and each chunk at its offset in the published file
// (load.js's chunksOf). prepareImage decodes a photo here, in the worker
// (image_input.js), starts it in the module's slot, then hands it over a
// band of rows a call, each band copied into the module's memory and freed
// once the call returns.

import createModule from './charlotte.mjs';
import { decode, bands } from './image_input.js';
import { streamFile } from './load.js';

// Starts the module and the device check, which reports through `onDevice`.
// Resolves with the runtime once the module is instantiated.
// With `profileSteps` (?profile, a diagnostic build's alone), the device is
// asked for timestamp queries, each load sets the runtime's step observer,
// and a turn's result carries its steps (web/dev/step_profile.js).
export async function createRuntime({ onDevice, onDeviceEvent, runBench, profileSteps = false }) {
  // The device's granted limits, once the device check reports: what preflight
  // judges fit against. Null when no device was acquired.
  let reportLimits;
  const deviceLimits = new Promise((resolve) => { reportLimits = resolve; });

  // The module reports the device check by calling this by name.
  globalThis.bllmOnResult = (device) => {
    reportLimits(device.ok && device.limits !== undefined ? device.limits : null);
    onDevice(device);
  };
  // The device's uncaptured errors and its loss, as the module reports them.
  globalThis.bllmOnDeviceEvent = (event) => onDeviceEvent?.(event);
  globalThis.bllmOnReply = answer;
  globalThis.bllmOnText = (call, text) => streams.get(call)?.(text);
  globalThis.bllmOnVector = (call, view) => vectors.set(call, view.slice());

  const module = await createModule();
  startDeviceCheck(module, { runBench, profileSteps });
  // Only a diagnostic module checks a load.
  const canCheck = typeof module._bllm_check_begin === 'function';

  return {
    preflight: async (bytes, totalSize, policy) => {
      const limits = await deviceLimits;
      return withBytesInModule(module, bytes, (pointer, length) =>
        withPolicy(module, policy, (fields) =>
          callModule((call) => module._bllm_preflight(call, pointer, length, totalSize,
            limits?.maxBufferSize ?? 0, limits?.maxStorageBufferBindingSize ?? 0,
            limits?.minStorageBufferOffsetAlignment ?? 0, ...fields))));
    },

    canCheck,

    loadFromCache: async ({ name, indexBytes, confirmed, maxChunk, policy, onProgress, signal }) => {
      const root = await navigator.storage.getDirectory();
      const handle = await (await root.getFileHandle(name)).createSyncAccessHandle();
      try {
        const size = handle.getSize();
        // Reads straight into the heap, through a view made now: growing the
        // heap detaches any view kept.
        const readInto = (pointer, offset, length) => {
          const read = handle.read(module.HEAPU8.subarray(pointer, pointer + length), { at: offset });
          if (read !== length) throw new Error(`the cached file ended early, at byte ${offset + read}`);
        };
        let chunkPointer = 0;
        const pass = (phase, calls) => streamFile({
          size, chunkBytes: maxChunk, signal,
          begin: calls.begin,
          read: ({ offset, length }) => readInto(chunkPointer, offset, length),
          send: async ({ offset, length }) => settled(await callModule((call) => calls.chunk(call, offset, length))),
          finish: async () => settled(await callModule((call) => calls.finish(call))),
          onProgress: (done, total) => onProgress?.({ phase, done, total }),
        });

        const loaded = await pass('load', {
          begin: async ({ maxChunk: chunkBytes }) => {
            const prefix = module._malloc(indexBytes);
            const ids = module._malloc(Math.max(4, confirmed.length * 4));
            try {
              readInto(prefix, 0, indexBytes);
              module.HEAPU8.set(new Uint8Array(u32s(confirmed)), ids);
              chunkPointer = settled(await withPolicy(module, policy, (fields) => callModule((call) =>
                module._bllm_load_begin(call, prefix, indexBytes, size, ids, confirmed.length, chunkBytes,
                  ...fields)))).chunkPointer;
            } finally {
              module._free(prefix);
              module._free(ids);
            }
          },
          chunk: (call, offset, length) => module._bllm_load_chunk(call, offset, length),
          finish: (call) => module._bllm_load_finish(call),
        });
        const { contextOffered } = loaded;
        const embeddingLength = loaded.embeddingLength ?? null;
        if (profileSteps && module._bllm_observe_steps(1) !== 1) {
          throw new Error('the step observer could not be set: the device grants no timestamp queries');
        }
        if (!canCheck) return { check: null, contextOffered, embeddingLength };
        const checked = await pass('check', {
          begin: async ({ maxChunk: chunkBytes }) => {
            chunkPointer = settled(await callModule((call) => module._bllm_check_begin(call, chunkBytes))).chunkPointer;
          },
          chunk: (call, offset, length) => module._bllm_check_chunk(call, offset, length),
          finish: (call) => module._bllm_check_finish(call),
        });
        return { check: { mismatches: checked.mismatches }, contextOffered, embeddingLength };
      } finally {
        handle.close();
      }
    },

    generate: async ({ id, prompt, sampling, seed, maxTokens, onText }) => {
      const bytes = new TextEncoder().encode(prompt);
      const pointer = module._malloc(Math.max(1, bytes.length));
      if (pointer === 0) throw new Error(`could not allocate ${bytes.length} bytes in the module`);
      let call;
      let answered;
      try {
        module.HEAPU8.set(bytes, pointer);
        answered = callModule((c) => {
          call = c;
          streams.set(c, onText);
          turns.set(id, c);
          module._bllm_generate(c, pointer, bytes.length, sampling === undefined ? 0 : 1,
            sampling?.temperature ?? 0, sampling?.topK ?? 0, sampling?.topP ?? 0, sampling?.minP ?? 0,
            seed, maxTokens ?? 0xFFFFFFFF);
        });
      } finally {
        // The module encoded the prompt during the call.
        module._free(pointer);
      }
      try {
        const answer = await answered;
        if (!answer.ok) throw new ModuleError(answer);
        const { stopReason, tokens, promptTokens, reusedTokens, steps } = answer;
        const result = { stopReason, tokens, promptTokens, reusedTokens };
        if (steps !== undefined) {
          result.steps = steps.map(([prefill, position, count, beginNs, endNs, reportMs]) =>
            ({ prefill: prefill === 1, position, tokens: count, beginNs, endNs, reportMs }));
        }
        return result;
      } finally {
        streams.delete(call);
        turns.delete(id);
      }
    },

    // A companion's verdict beside the loaded model, from its header.
    companionPreflight: async (kind, bytes, totalSize, entry, memoryBudget) => {
      const limits = await deviceLimits;
      return withBytesInModule(module, bytes, (pointer, length) =>
        withRanges(module, entry.ranges, (ranges, count) => callModule((call) =>
          module._bllm_companion_preflight(call, kindIndex(kind), pointer, length, totalSize, ranges, count,
            entry.tokens?.least ?? 0, entry.tokens?.most ?? 0, limits?.maxBufferSize ?? 0,
            limits?.maxStorageBufferBindingSize ?? 0, limits?.minStorageBufferOffsetAlignment ?? 0,
            memoryBudget ?? NaN))));
    },

    // Loads the cached ranges of a companion beside the loaded model.
    loadCompanion: async ({ kind, name, entry, memoryBudget, maxChunk, onProgress, signal }) => {
      const root = await navigator.storage.getDirectory();
      const handle = await (await root.getFileHandle(name)).createSyncAccessHandle();
      try {
        const readInto = (pointer, offset, length) => {
          const read = handle.read(module.HEAPU8.subarray(pointer, pointer + length), { at: offset });
          if (read !== length) throw new Error(`the cached file ended early, at byte ${offset + read}`);
        };
        const headerBytes = entry.ranges[0][1];
        let chunkPointer = 0;
        return await streamFile({
          size: handle.getSize(), ranges: entry.ranges, chunkBytes: maxChunk, signal,
          begin: async ({ maxChunk: chunkBytes }) => {
            const prefix = module._malloc(headerBytes);
            try {
              readInto(prefix, 0, headerBytes);
              chunkPointer = settled(await withRanges(module, entry.ranges, (ranges, count) =>
                callModule((call) => module._bllm_companion_load_begin(call, kindIndex(kind), prefix, headerBytes,
                  entry.sizeBytes, ranges, count, chunkBytes, entry.tokens?.least ?? 0, entry.tokens?.most ?? 0,
                  memoryBudget ?? NaN))))
                .chunkPointer;
            } finally {
              module._free(prefix);
            }
          },
          read: ({ offset, length }) => readInto(chunkPointer, offset, length),
          send: async ({ offset, length }) =>
            settled(await callModule((call) => module._bllm_companion_load_chunk(call, offset, length))),
          finish: async () => {
            const { ok, ...answer } = settled(await callModule((call) => module._bllm_companion_load_finish(call)));
            return answer;
          },
          onProgress: (done, total) => onProgress?.({ phase: 'load', done, total }),
        });
      } finally {
        handle.close();
      }
    },

    // Decodes a photo and prepares it in the module's image slot: the tokens
    // it becomes and the size it is resampled to.
    prepareImage: async (blob) => {
      const { bitmap, width, height } = await decode(blob);
      let grid;
      try {
        grid = settled(await callModule((call) => module._bllm_image_begin(call, width, height)));
      } catch (error) {
        bitmap.close();
        throw error;
      }
      let ready = false;
      for (const { first, pixels } of bands(bitmap)) {
        const pointer = module._malloc(pixels.length);
        if (pointer === 0) throw new Error(`could not allocate ${pixels.length} bytes in the module`);
        try {
          module.HEAPU8.set(pixels, pointer);
          ready = settled(await callModule((call) =>
            module._bllm_image_rows(call, first, pointer, pixels.length))).ready;
        } finally {
          module._free(pointer);
        }
      }
      if (!ready) throw new Error('the image is not ready after its last band');
      return { tokens: grid.tokens, width: grid.width, height: grid.height };
    },

    // The text's unit-length embedding, and the tokens it made; with
    // `image`, the prepared image in the text's <|image|>.
    embed: async (text, image = false) => {
      const bytes = new TextEncoder().encode(text);
      const pointer = module._malloc(Math.max(1, bytes.length));
      if (pointer === 0) throw new Error(`could not allocate ${bytes.length} bytes in the module`);
      let call;
      let answered;
      try {
        module.HEAPU8.set(bytes, pointer);
        answered = callModule((c) => {
          call = c;
          module._bllm_embed(c, pointer, bytes.length, image ? 1 : 0);
        });
      } finally {
        // The module encoded the text during the call.
        module._free(pointer);
      }
      try {
        const answer = await answered;
        if (!answer.ok) throw new ModuleError(answer);
        return { vector: vectors.get(call), tokens: answer.tokens };
      } finally {
        vectors.delete(call);
      }
    },

    // Ends a generate's turn, named by its request; false when none runs.
    cancel: async (target) => {
      const call = turns.get(target);
      if (call === undefined) return false;
      return (await callModule((c) => module._bllm_cancel(c, call))).running;
    },
  };
}

// Starts the device check the page asked for. Every outcome is reported as
// the module reports one, through bllmOnResult, so preflight's wait for the
// device's limits ends whichever way the check went.
function startDeviceCheck(module, { runBench, profileSteps }) {
  const report = globalThis.bllmOnResult;
  if (profileSteps) {
    if (typeof module._bllm_run_profile_check === 'function') {
      // Present only in a diagnostic build.
      module._bllm_run_profile_check();
    } else {
      report({
        ok: false,
        stage: 'request',
        error: 'the step profile is not compiled into this build; serve the diagnostic site (make serve-diag)',
      });
    }
  } else if (!runBench) {
    module._bllm_run_self_check();
  } else if (typeof module._bllm_run_readback_bench === 'function') {
    // Present only in a diagnostic build.
    module._bllm_run_readback_bench();
  } else {
    report({
      ok: false,
      stage: 'request',
      error: 'the readback benchmark is not compiled into this build; ' +
             'configure the wasm-diag preset to run it',
    });
  }
}

// C++ answers a call through bllmOnReply, either during the call or later from
// a callback. Each call gets an id so its answer finds it either way.
const pendingCalls = new Map();
let nextCall = 1;

function answer(call, value) {
  pendingCalls.get(call)(value);
  pendingCalls.delete(call);
}

function callModule(start) {
  const call = nextCall++;
  return new Promise((resolve) => {
    pendingCalls.set(call, resolve);
    start(call);
  });
}

// Each running generate's text callback, by its call id; and its call id, by
// the worker's request id, for cancel.
const streams = new Map();
const turns = new Map();
// Each answered embed's vector, by its call id, until its answer is read.
const vectors = new Map();

// A failure the module answered, with its code where it gave one.
class ModuleError extends Error {
  constructor({ error, subject, code, promptTokens, textTokens, contextTokens }) {
    super(subject ? `${error} (${subject})` : error);
    this.code = code;
    if (code === 'prompt-too-long') this.counts = { promptTokens, contextTokens };
    if (code === 'text-too-long') this.counts = { textTokens, contextTokens };
  }
}

// The cache precisions, in CachePrecision's order (core/policy/policy.h).
const PRECISIONS = ['F16', 'BF16', 'Q8_0'];

// The model's load policy as the module reads it (src/wasm/bindings.cpp),
// for the duration of `use`: each field, or its stand-in for unset; the stop
// texts' UTF-8 bytes in one buffer, and their lengths.
async function withPolicy(module, policy = {}, use) {
  const precision = policy.cachePrecision === undefined ? -1 : PRECISIONS.indexOf(policy.cachePrecision);
  if (precision === -1 && policy.cachePrecision !== undefined) {
    throw new Error(`cache precision "${policy.cachePrecision}" is not one this build names`);
  }
  const encoder = new TextEncoder();
  const stops = (policy.stop ?? []).map((text) => encoder.encode(text));
  const total = stops.reduce((sum, bytes) => sum + bytes.length, 0);
  const bytes = module._malloc(Math.max(1, total));
  const lengths = module._malloc(Math.max(4, stops.length * 4));
  try {
    let at = 0;
    for (const stop of stops) {
      module.HEAPU8.set(stop, bytes + at);
      at += stop.length;
    }
    module.HEAPU8.set(new Uint8Array(u32s(stops.map((stop) => stop.length))), lengths);
    return await use([precision, policy.memoryBudget ?? NaN, policy.rollbackReserve ?? 0xFFFFFFFF,
      bytes, lengths, stops.length]);
  } finally {
    module._free(bytes);
    module._free(lengths);
  }
}

// A load answer, or its failure thrown, named.
function settled(answer) {
  if (!answer.ok) throw new Error(answer.subject ? `${answer.error} (${answer.subject})` : answer.error);
  return answer;
}

// Tensor ids as little-endian 32-bit words, for the module.
function u32s(values) {
  const words = new DataView(new ArrayBuffer(Math.max(4, values.length * 4)));
  values.forEach((v, i) => words.setUint32(i * 4, v, true));
  return words.buffer;
}

// A companion's kind as the module reads it (src/wasm/bindings.cpp).
function kindIndex(kind) {
  const index = ['image', 'audio'].indexOf(kind);
  if (index === -1) throw new Error(`no companion of kind "${kind}"`);
  return index;
}

// A list of [begin, end) ranges as pairs of doubles in the module's memory,
// for the duration of `use`.
async function withRanges(module, ranges, use) {
  const pointer = module._malloc(16 * ranges.length);
  if (pointer === 0) throw new Error('could not allocate the ranges in the module');
  try {
    new Float64Array(module.HEAPU8.buffer, pointer, 2 * ranges.length).set(ranges.flat());
    return await use(pointer, ranges.length);
  } finally {
    module._free(pointer);
  }
}

// Copies `bytes` into the module's memory for the duration of `use`.
async function withBytesInModule(module, bytes, use) {
  const pointer = module._malloc(bytes.byteLength);
  if (pointer === 0) throw new Error(`could not allocate ${bytes.byteLength} bytes in the module`);
  try {
    module.HEAPU8.set(new Uint8Array(bytes), pointer);
    return await use(pointer, bytes.byteLength);
  } finally {
    module._free(pointer);
  }
}
