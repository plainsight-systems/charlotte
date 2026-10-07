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
// field unset as its stand-in, the stop texts NUL-terminated in one buffer.
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

import createModule from './charlotte.mjs';
import { streamFile } from './load.js';

// Starts the module and the device check, which reports through `onDevice`.
// Resolves with the runtime once the module is instantiated.
export async function createRuntime({ onDevice, runBench }) {
  // The device's granted limits, once the device check reports: what preflight
  // judges fit against. Null when no device was acquired.
  let reportLimits;
  const deviceLimits = new Promise((resolve) => { reportLimits = resolve; });

  // The module reports the device check by calling this by name.
  globalThis.bllmOnResult = (device) => {
    reportLimits(device.ok && device.limits !== undefined ? device.limits : null);
    onDevice(device);
  };
  globalThis.bllmOnReply = answer;

  const module = await createModule();
  startDeviceCheck(module, { onDevice, runBench });
  // Only a diagnostic module checks a load.
  const canCheck = typeof module._bllm_check_begin === 'function';

  return {
    preflight: async (bytes, totalSize) => {
      const limits = await deviceLimits;
      return withBytesInModule(module, bytes, (pointer, length) =>
        callModule((call) => module._bllm_preflight(call, pointer, length, totalSize,
          limits?.maxBufferSize ?? 0, limits?.maxStorageBufferBindingSize ?? 0,
          limits?.minStorageBufferOffsetAlignment ?? 0)));
    },

    canCheck,

    loadFromCache: async ({ name, indexBytes, confirmed, maxChunk, onProgress, signal }) => {
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

        await pass('load', {
          begin: async ({ maxChunk: chunkBytes }) => {
            const prefix = module._malloc(indexBytes);
            const ids = module._malloc(Math.max(4, confirmed.length * 4));
            try {
              readInto(prefix, 0, indexBytes);
              module.HEAPU8.set(new Uint8Array(u32s(confirmed)), ids);
              chunkPointer = settled(await callModule((call) => module._bllm_load_begin(
                call, prefix, indexBytes, size, ids, confirmed.length, chunkBytes))).chunkPointer;
            } finally {
              module._free(prefix);
              module._free(ids);
            }
          },
          chunk: (call, offset, length) => module._bllm_load_chunk(call, offset, length),
          finish: (call) => module._bllm_load_finish(call),
        });
        if (!canCheck) return { check: null };
        const checked = await pass('check', {
          begin: async ({ maxChunk: chunkBytes }) => {
            chunkPointer = settled(await callModule((call) => module._bllm_check_begin(call, chunkBytes))).chunkPointer;
          },
          chunk: (call, offset, length) => module._bllm_check_chunk(call, offset, length),
          finish: (call) => module._bllm_check_finish(call),
        });
        return { check: { mismatches: checked.mismatches } };
      } finally {
        handle.close();
      }
    },

    generate: () => {
      throw new Error('generating text is not implemented in this build');
    },

    // Nothing generates, so there is nothing to stop.
    cancel: () => false,
  };
}

function startDeviceCheck(module, { onDevice, runBench }) {
  if (!runBench) {
    module._bllm_run_self_check();
  } else if (typeof module._bllm_run_readback_bench === 'function') {
    // Present only in a diagnostic build.
    module._bllm_run_readback_bench();
  } else {
    onDevice({
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
