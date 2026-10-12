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
  // PROGRESS replies. DONE carries the context offered, and the embedding's
  // length for a model that encodes, else null.
  LOAD: 'load',
  // Generate a reply to a rendered prompt — { prompt, sampling, seed,
  // maxTokens }, sampling and maxTokens optional — streaming its text as
  // TOKEN replies. DONE carries { stopReason, tokens, promptTokens,
  // reusedTokens }; FAILED's error carries code "prompt-too-long" when the
  // prompt does not fit the context, and "prompt-too-large", shown by name,
  // when it is past the bytes one encode takes.
  GENERATE: 'generate',
  // Embed a text, for a model that encodes — { text, media }, the task's
  // prompt written around it; with `media` 'image', the image PREPARE_IMAGE
  // prepared goes where the text writes <|image|>, and with 'audio', the clip
  // PREPARE_AUDIO prepared where it writes <|audio|>. DONE carries { vector,
  // tokens }: the unit-length Float32Array and the tokens the text made, ends
  // included; FAILED's error carries code "text-too-long", with both counts,
  // when the text does not fit the context.
  EMBED: 'embed',
  // Judge a loaded model's companion file (companion.js) from its header,
  // read from its first range — { media, bytes, totalSize, entry,
  // memoryBudget }, `media` 'image' or 'audio', `entry` the catalog's for
  // that kind and memoryBudget the model's load policy's, or absent. DONE
  // carries the verdict, as PREFLIGHT's.
  PREFLIGHT_COMPANION: 'preflight-companion',
  // Load a cached companion beside the loaded model, by its file's name —
  // { media, name, entry, memoryBudget, maxChunk }; progress comes back as
  // PROGRESS replies. DONE carries, for an image companion, { imageTokens },
  // the most an image becomes; for an audio companion, { mostSamples }, the
  // most a clip holds.
  LOAD_COMPANION: 'load-companion',
  // Decode a photo in the worker and prepare it in the module's one image
  // slot — { image }, a Blob. DONE carries { tokens, width, height }: the
  // tokens it becomes and the size it is resampled to.
  PREPARE_IMAGE: 'prepare-image',
  // Prepare a clip in the module's one audio slot — { samples }, a
  // Float32Array of at most mostSamples at 16 kHz mono, its buffer
  // transferred. DONE carries { frames, rows }: its spectrogram's frames and
  // the rows it becomes.
  PREPARE_AUDIO: 'prepare-audio',
  // Stop a load, a companion's load or a generation early; `target` is its
  // request id.
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
  // The device reported, unprompted, an error no scope caught or its loss:
  // { kind: 'uncaptured', type, message } or { kind: 'lost', reason, message }
  // (src/wasm/bindings.cpp). What no reply carries — a loss between turns.
  DEVICE_EVENT: 'device-event',
});
