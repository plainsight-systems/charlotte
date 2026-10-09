// Axis H. The chosen model: how far this build can take it, what stops the
// next stage, and what can be done now — download it, load it, remove it —
// with progress while that happens.
//
// Renders one state object; it holds no state of its own.
//   checking      preflight is running
//   failed        a step failed; `action` names which
//   downloadable  the file reads and is not cached; offered for download
//                 only where it can run (stages.js's offersDownload)
//   downloading   `received` of `total` bytes so far
//   cached        in the cache, and read from that copy
//   loading       `done` of `total` bytes handed to the runtime
//   loaded        ready to chat

import { h } from './dom.js';
import { nextStage, offersDownload, reaches } from './stages.js';
import { formatBytes, formatMemory } from './units.js';

export function renderModel(element, state, actions) {
  element.dataset.state = TONES[state.phase];
  element.replaceChildren(...CONTENT[state.phase](state, actions));
}

const TONES = {
  checking: 'pending', failed: 'bad', downloadable: 'ok',
  downloading: 'pending', cached: 'ok', loading: 'pending', verifying: 'pending', loaded: 'ok',
};

const CONTENT = {
  checking: ({ model }) => [heading(`Checking ${model.name}…`)],

  failed: ({ model, action, error }, { retry }) => [
    heading(`Could not ${action} ${model.name}`),
    h('p', { text: error.message }),
    buttons(['Try again', retry, 'primary']),
  ],

  downloadable: ({ model, verdict }, { download }) => offersDownload(verdict)
    ? [
      heading(`${model.name} reads, and can be downloaded`),
      ...stages(verdict),
      buttons([`Download ${formatBytes(verdict.totalSize)}`, download, 'primary']),
    ]
    : [heading(`${model.name} reads, but cannot run here, so it is not offered for download`), ...stages(verdict)],

  downloading: ({ model, received, total }, { cancel }) => [
    heading(`Downloading ${model.name}`),
    progress(received, total),
    buttons(['Cancel', cancel]),
  ],

  cached: ({ model, verdict }, { load, remove }) => [
    heading(`${model.name} is downloaded, and its copy reads`),
    ...stages(verdict),
    reaches(verdict, 'upload')
      ? buttons(['Load', load, 'primary'], ['Remove', remove])
      : buttons(['Remove', remove]),
  ],

  loading: ({ model, done, total }) => [heading(`Loading ${model.name}`), progress(done, total)],

  verifying: ({ model, done, total }) => [
    heading(`Checking every byte of ${model.name} on the GPU`),
    progress(done, total),
  ],

  // `check` is the diagnostic build's byte-for-byte check, null in the clean
  // build, which does not run one.
  loaded: ({ model, check }) => [
    heading(`${model.name} is loaded`),
    ...(check ? [h('p', { text: checked(check.mismatches) })] : []),
  ],
};

function checked(mismatches) {
  if (mismatches.length === 0) return 'Every byte read back from the GPU matches the file.';
  const first = mismatches[0];
  return `${mismatches.length} ranges on the GPU differ from the file; the first is in ` +
    `${first.tensor}, buffer ${first.buffer} at byte ${first.offset}.`;
}

const NEXT = {
  describe: 'describe it',
  fit: 'fit it on this device',
  upload: 'upload it to the GPU',
  run: 'run it',
};

// What the file is, how it fits this device, what stops the next stage, and —
// folded away — what else running it needs.
function stages(verdict) {
  const { next, blocking, later } = nextStage(verdict);
  const parts = [facts(verdict)];
  if (verdict.fit) parts.push(fits(verdict.fit));
  if (next === undefined) return [...parts, h('p', { text: 'This build can run it.' })];
  parts.push(h('p', { text: `Cannot ${NEXT[next]} yet:` }), reasons(blocking));
  if (later.length > 0) {
    parts.push(h('details', { className: 'later' },
      h('summary', { text: `Also needed to run it (${later.length})` }),
      reasons(later)));
  }
  return parts;
}

const fits = (fit) => h('p', { className: 'model-fit',
  text: `Fits this device: ${fit.contextOffered.toLocaleString()} of ` +
        `${fit.trainedContext.toLocaleString()} tokens of context, ` +
        `${formatMemory(fit.totalBytes)} of a ${formatMemory(fit.memoryBudget)} budget ` +
        `(${formatMemory(fit.weightBytes)} weights, ${formatMemory(fit.cacheBytes)} cache)` });

const reasons = (blockers) => h('ul', {}, blockers.map((b) => h('li', { text: b.detail })));

const heading = (text) => h('p', { className: 'model-heading', text });

const facts = (verdict) => h('p', { className: 'model-facts',
  text: `${verdict.architecture ?? 'unnamed architecture'} · ${verdict.tensorCount} tensors` });

function progress(done, total) {
  const label = total === null
    ? formatBytes(done)
    : `${formatBytes(done)} of ${formatBytes(total)}`;
  return h('div', { className: 'progress' },
    h('progress', { value: done, max: total ?? undefined, 'aria-label': label }),
    h('span', { text: label }));
}

// Each action is [text, onclick] or [text, onclick, 'primary'] for the step
// the state leads to.
const buttons = (...actions) => h('div', { className: 'actions' },
  actions.map(([text, onclick, kind]) =>
    h('button', { type: 'button', className: kind === 'primary' ? 'action primary' : 'action', text, onclick })));
