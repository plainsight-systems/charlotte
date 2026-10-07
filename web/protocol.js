// The messages that cross between the page and the worker. Both sides import
// this file, so every kind is spelled once.
//
// A request carries an id. Every reply to it carries the same id, and the
// replies end with exactly one DONE or FAILED; TOKEN and PROGRESS replies may
// come before.
// Notices carry no id: the worker sends them unprompted.

export const Request = Object.freeze({
  // Read a model file's index from its first bytes and judge it, under the
  // model's load policy (`policy`, the entry's policy.load; absent for an
  // unmeasured model).
  PREFLIGHT: 'preflight',
  // Load a cached model, by its file's name, with the index prefix length,
  // the confirmed duplicates and the same load policy; progress comes back as
  // PROGRESS replies. DONE carries the context offered.
  LOAD: 'load',
  // Generate a reply to a rendered prompt — { prompt, sampling, seed,
  // maxTokens }, sampling and maxTokens optional — streaming its text as
  // TOKEN replies. DONE carries { stopReason, tokens, promptTokens,
  // reusedTokens }; FAILED's error carries code "prompt-too-long" when the
  // prompt does not fit the context.
  GENERATE: 'generate',
  // Stop a load or a generation early; `target` is its request id.
  CANCEL: 'cancel',
});

export const Reply = Object.freeze({
  TOKEN: 'token',
  // How far a load has come: { phase, done, total }.
  PROGRESS: 'progress',
  DONE: 'done',
  FAILED: 'failed',
});

export const Notice = Object.freeze({
  // The GPU device was acquired and checked, or could not be.
  DEVICE: 'device',
});
