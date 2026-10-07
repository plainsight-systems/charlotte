import { Notice, Reply } from './protocol.js';

// A request the worker could not complete. `stage` names where it failed;
// `code`, where the runtime gives one, what the page can act on, as
// "prompt-too-long".
export class WorkerError extends Error {
  constructor({ stage, message, code, counts }) {
    super(message);
    this.name = 'WorkerError';
    this.stage = stage;
    this.code = code;
    // For "prompt-too-long": { promptTokens, contextTokens }.
    this.counts = counts;
  }
}

// The page's side of the boundary. Turns request and reply messages into
// promises, and routes streamed text and notices to callbacks.
//
// `port` is a Worker in the page and a MessagePort in tests; both deliver
// messages the same way.
export class WorkerClient {
  #port;
  #onDevice;
  #nextId = 1;
  #pending = new Map();

  constructor(port, { onDevice }) {
    this.#port = port;
    this.#onDevice = onDevice;
    port.addEventListener('message', (event) => this.#receive(event.data));
    port.start?.();
  }

  // Resolves with the request's result, or rejects with a WorkerError.
  // `onToken` receives each piece of streamed text, in order; `onProgress`
  // each progress report.
  request(kind, payload = {}, options = {}) {
    return this.send(kind, payload, options).reply;
  }

  // As request(), but also returns the request's id, for a later request
  // that refers to it.
  send(kind, payload = {}, { onToken, onProgress, transfer = [] } = {}) {
    const id = this.#nextId++;
    const reply = new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onToken, onProgress });
      this.#port.postMessage({ id, kind, ...payload }, transfer);
    });
    return { id, reply };
  }

  // Fails every request still waiting, when the worker itself has died.
  failAll(error) {
    for (const { reject } of this.#pending.values()) reject(error);
    this.#pending.clear();
  }

  #receive(message) {
    if (message.kind === Notice.DEVICE) {
      this.#onDevice(message.device);
      return;
    }
    const pending = this.#pending.get(message.id);
    if (pending === undefined) {
      throw new Error(`worker replied to unknown request ${message.id}`);
    }
    switch (message.kind) {
      case Reply.TOKEN:
        pending.onToken?.(message.text);
        return;
      case Reply.PROGRESS:
        pending.onProgress?.(message.progress);
        return;
      case Reply.DONE:
        this.#pending.delete(message.id);
        pending.resolve(message.value);
        return;
      case Reply.FAILED:
        this.#pending.delete(message.id);
        pending.reject(new WorkerError(message.error));
        return;
      default:
        throw new Error(`worker sent unknown reply kind "${message.kind}"`);
    }
  }
}
