// Axis H. The diagnostics box above the page (diagnostics.js): where the
// last visit stopped, when it stopped without a word or failed, and what
// the GPU reported or failed to answer on this one. Each part copies its
// record for a report; where the clipboard is refused, the record is shown
// to select by hand. Hidden when there is nothing to say.
//
// render() is called on every change and every two seconds (a stall is the
// absence of change), but rebuilds only when what it would say changes, so
// a reader's copy, the record shown and any selection in it survive.

import { describe, kStallMs, report, stalledFor, worthReporting } from './diagnostics.js';
import { h } from './dom.js';

export function createDiagnosticsView(element) {
  let previousDismissed = false;
  let last = null;
  let said = null;

  const render = ({ previous, session, now }) => {
    last = { previous, session, now };
    const showPrevious = !previousDismissed && worthReporting(previous);
    const stalled = stalledFor(session, now) >= kStallMs;
    const saying = JSON.stringify([showPrevious, session.events.length, stalled && describe(session)]);
    if (saying === said) return;
    said = saying;
    const parts = [];
    if (showPrevious) {
      parts.push(part(lastVisit(previous), () => previous, () => {
        previousDismissed = true;
        render(last);
      }));
    }
    const live = thisVisit(session, stalled);
    if (live.length > 0) parts.push(part(live, () => last.session, null));
    element.replaceChildren(...parts);
    element.hidden = parts.length === 0;
  };
  return { render };
}

// What to say of the last visit's record.
function lastVisit(previous) {
  const lines = [];
  if (previous.busy) {
    lines.push(heading(`The last visit stopped while ${describe(previous)}.`));
    lines.push(text('Nothing said why. If the browser closed or reloaded the page, this is where it was.'));
  } else if (previous.error !== null) {
    lines.push(heading(`The last visit failed: ${previous.error}`));
  } else {
    lines.push(heading('The GPU reported a problem on the last visit.'));
  }
  return [...lines, ...previous.events.map(eventLine)];
}

// What to say of this visit: the device's events, and a stall.
function thisVisit(session, stalled) {
  const lines = session.events.map(eventLine);
  if (stalled) {
    lines.push(text(`No progress for ${kStallMs / 1000} s or more while ${describe(session)}. ` +
                    'The GPU may have stopped answering.'));
  }
  return lines.length === 0 ? [] : [heading('Something went wrong on the GPU.'), ...lines];
}

function eventLine(event) {
  const said = event.message ? `: ${event.message}` : '';
  return text(event.kind === 'lost'
    ? `The GPU device was lost (${event.reason})${said}`
    : `The GPU reported an error no step caught (${event.type})${said}`);
}

// `record()` is the record to copy as it is when copied.
function part(lines, record, onDismiss) {
  const shown = h('pre', { className: 'diagnostics-record' });
  shown.hidden = true;
  const copy = h('button', { type: 'button', className: 'action', text: 'Copy diagnostics' });
  copy.onclick = async () => {
    const copied = report(record());
    try {
      await navigator.clipboard.writeText(copied);
      copy.textContent = 'Copied';
    } catch {
      shown.textContent = copied;
      shown.hidden = false;
    }
  };
  const actions = h('div', { className: 'actions' }, copy,
    onDismiss === null ? null : h('button', { type: 'button', className: 'action', text: 'Dismiss', onclick: onDismiss }));
  return h('div', { className: 'diagnostics-part' }, ...lines, actions, shown);
}

const heading = (value) => h('p', { className: 'diagnostics-heading', text: value });
const text = (value) => h('p', { text: value });
