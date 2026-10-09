// Axis L: browser storage. Keeps this visit's diagnostics (diagnostics.js)
// in localStorage, so a page the browser kills leaves its record behind, and
// hands the last visit's record over once, at start.
//
//   - A phase change is written at once, before the step it names runs, so
//     a kill during that step finds it stored. Progress is written at most
//     once a second (kSaveEveryMs): a download reports progress per network
//     chunk, and a synchronous write each would cost more than it tells.
//   - The last visit's record is read before this visit's first write
//     replaces it, and handed over as `previous`; the view decides whether
//     to show it.
//   - Storage may be refused (a private tab, a full quota): every read and
//     write is guarded, and the page works without it, only without the
//     record surviving a reload.
//   - `onChange(session)` follows every change, for the view.

import {
  startSession, withDevice, withEvent, withModel, withPhase, withProgress, phaseOfModelState,
} from './diagnostics.js';

const KEY = 'charlotte.diagnostics.v1';
const kSaveEveryMs = 1000;

export function createRecorder({ storage, now, userAgent, platform, onChange }) {
  const previous = read(storage);
  let session = startSession({ startedAt: now(), userAgent, platform });
  let savedAt = -Infinity;

  const save = () => {
    savedAt = now();
    try {
      storage?.setItem(KEY, JSON.stringify(session));
    } catch {
      // Storage refused: the record lives only as long as the page.
    }
  };
  const change = (next, { urgent }) => {
    session = next;
    if (urgent || now() - savedAt >= kSaveEveryMs) save();
    onChange?.(session);
  };
  save();

  return {
    previous,
    current: () => session,
    device: (device) => change(withDevice(session, device), { urgent: true }),
    event: (event) => change(withEvent(session, event, now()), { urgent: true }),
    // The model panel's state (model_controller.js).
    modelState: (state) => {
      const model = state.model === undefined ? session.model
        : { id: state.model.id, name: state.model.name, sizeBytes: state.model.sizeBytes,
            contextOffered: state.verdict?.fit?.contextOffered ?? null };
      const entered = phaseOfModelState(state);
      if (entered === null) return;
      const next = withModel(session, model);
      if (entered.phase === session.phase && entered.busy) {
        change(withProgress(next, entered.detail ?? '', now()), { urgent: false });
      } else {
        change(withPhase(next, entered, now()), { urgent: true });
      }
    },
    // A chat turn (chat.js): 'start', 'text' as the reply streams, 'done' or
    // 'failed' with its message.
    turn: (step, message = '') => {
      switch (step) {
        case 'start': return change(withPhase(session, { phase: 'reading the prompt', busy: true }, now()), { urgent: true });
        case 'text':
          if (session.phase === 'writing the reply') return change(withProgress(session, '', now()), { urgent: false });
          return change(withPhase(session, { phase: 'writing the reply', busy: true }, now()), { urgent: true });
        case 'done': return change(withPhase(session, { phase: 'replied', busy: false }, now()), { urgent: true });
        case 'failed':
          return change(withPhase(session, { phase: 'failed', busy: false, error: `turn: ${message}` }, now()),
            { urgent: true });
        default: return undefined;
      }
    },
  };
}

function read(storage) {
  try {
    const text = storage?.getItem(KEY);
    return text ? JSON.parse(text) : null;
  } catch {
    return null;
  }
}
