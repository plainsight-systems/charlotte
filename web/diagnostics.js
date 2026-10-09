// Axis H. The session's diagnostics: what the page was doing, what the GPU
// said, and how far it got, kept so a failure no message reports can still
// be told. Pure functions over a plain record; diagnostics_recorder.js keeps
// it in the browser's storage, and diagnostics_view.js shows it.
//
// A phone may kill the page outright for its memory, and a GPU may stop
// answering without an error; either leaves the page with nothing to say.
// So the record names the phase the page is in whenever it changes, and is
// written down before each step that could kill it: the next visit reads a
// record left busy and says where the last one stopped.
//
//   - A phase is busy (checking, downloading, creating buffers, uploading
//     weights, building kernels, verifying, reading the prompt, writing the
//     reply) or at rest (chosen, loaded, replied, failed). A record left
//     busy is where a visit stopped without a word.
//   - The log keeps the phases entered, the last kLogEntries; progress within
//     a phase updates its detail and its time, and adds no entry.
//   - Device events (Notice.DEVICE_EVENT) are kept, the last kEvents, and
//     always reported: an uncaptured error or a loss is worth showing even
//     where the visit went on.
//   - A busy phase with no progress for kStallMs is stalled: the GPU may have
//     stopped answering, which WebGPU does not always report.

import { formatBytes } from './units.js';

export const kStallMs = 20_000;
const kLogEntries = 40;
const kEvents = 20;

export function startSession({ startedAt, userAgent, platform }) {
  return {
    version: 1, startedAt, userAgent, platform,
    device: null, model: null,
    phase: 'started', detail: '', busy: false, phaseAt: startedAt,
    error: null, log: [], events: [],
  };
}

// The device notice (Notice.DEVICE): the adapter and limits, or why none.
export function withDevice(session, device) {
  const summary = device.ok
    ? { ok: true, adapter: device.adapter, limits: device.limits, adapterMaxima: device.adapterMaxima }
    : { ok: false, stage: device.stage, error: device.error };
  return { ...session, device: summary };
}

export function withModel(session, model) {
  return { ...session, model };
}

// Enters a phase. `error`, for a failure, is its message.
export function withPhase(session, { phase, busy, detail = '', error = null }, now) {
  const entry = { at: now - session.startedAt, phase, ...(detail && { detail }), ...(error && { error }) };
  return {
    ...session, phase, busy, detail, phaseAt: now,
    error,
    log: [...session.log, entry].slice(-kLogEntries),
  };
}

// Progress within the current phase.
export function withProgress(session, detail, now) {
  return { ...session, detail, phaseAt: now };
}

export function withEvent(session, event, now) {
  return { ...session, events: [...session.events, { at: now - session.startedAt, ...event }].slice(-kEvents) };
}

// The model panel's state (model_controller.js) as a phase, or null where it
// names none.
export function phaseOfModelState(state) {
  switch (state.phase) {
    case 'checking': return { phase: 'checking the file', busy: true };
    case 'downloadable': case 'cached': return { phase: 'chosen', busy: false };
    case 'downloading': return { phase: 'downloading', busy: true, detail: fraction(state.received, state.total) };
    case 'loading':
      if (!state.total || state.done === 0) return { phase: 'creating buffers', busy: true };
      if (state.done < state.total) {
        return { phase: 'uploading weights', busy: true, detail: fraction(state.done, state.total) };
      }
      return { phase: 'building kernels', busy: true };
    case 'verifying': return { phase: 'verifying', busy: true, detail: fraction(state.done, state.total) };
    case 'loaded': return { phase: 'loaded', busy: false };
    case 'failed': return { phase: 'failed', busy: false, error: `${state.action}: ${state.error?.message ?? state.error}` };
    default: return null;
  }
}

const fraction = (done, total) => (total ? `${formatBytes(done)} of ${formatBytes(total)}` : formatBytes(done));

// Whether a record has something to report: it was left busy, or failed, or
// the device spoke.
export function worthReporting(session) {
  return session !== null && (session.busy || session.error !== null || session.events.length > 0);
}

// How long a busy phase has gone without progress; 0 at rest.
export function stalledFor(session, now) {
  return session.busy ? now - session.phaseAt : 0;
}

// "uploading weights (… bytes) for Qwen3 0.6B", for a sentence.
export function describe(session) {
  const detail = session.detail ? ` (${session.detail})` : '';
  const model = session.model?.name ? ` for ${session.model.name}` : '';
  return `${session.phase}${detail}${model}`;
}

// The record as text to paste into a report.
export function report(session) {
  return JSON.stringify(session, null, 2);
}
