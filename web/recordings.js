// Axis H. The recordings chosen for the search panel's index (search.js),
// for a model whose catalog entry names an audio companion: an "Add
// recordings" button and the panel as a drop target for audio files, each
// listed with its name and length and a button to remove it. The photo
// picker's twin (photos.js), for its own kind.
//
// createRecordings({ onChange, decode }) returns { element, recordings(),
// dropOnto(target), release() }; decode is audio_input.js's unless given.
//   - A recording is checked as it is added: decoded on the page and its
//     length kept, its samples dropped, so one the browser can't decode, or
//     longer than 10 minutes, is named and not added, rather than ending an
//     indexing later. It is decoded again when it is indexed: holding every
//     added recording's samples would hold up to 38 MB each.
//   - Each entry keeps its file and an object URL for it, which the search
//     panel's results play from at the clip's start (an <audio> element,
//     the URL's media fragment #t=start,end), kept until the recording is
//     removed or release() is called.
//   - onChange() follows every addition and removal: the index no longer
//     holds the recordings shown. A file refused adds nothing, so changes
//     nothing.
//   - The panel's drop target takes images to photos.js and audio to this
//     file, each by its MIME type.

import { decode as decodeAudio, kRate } from './audio_input.js';
import { h } from './dom.js';

// A length in seconds as m:ss.
export function clock(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

export function createRecordings({ onChange, decode = decodeAudio }) {
  const recordings = [];   // { file, url, samples }, in the order added
  const list = h('ul', { className: 'recording-list' });
  const refusals = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const input = h('input', { type: 'file', accept: 'audio/*', multiple: true, hidden: true,
    'aria-label': 'Recordings' });
  const button = h('button', { type: 'button', className: 'action', text: 'Add recordings',
    onclick: () => input.click() });

  const render = () => list.replaceChildren(...recordings.map((recording) =>
    h('li', {},
      h('span', { text: `${recording.file.name} · ${clock(recording.samples / kRate)}` }),
      h('button', { type: 'button', className: 'action', text: '×', 'aria-label': `Remove ${recording.file.name}`,
        onclick: () => remove(recording) }))));

  async function add(files) {
    const refused = [];
    for (const file of files) {
      let samples;
      try {
        samples = (await decode(file)).length;
      } catch (error) {
        refused.push(`${file.name}: ${error.message}`);
        continue;
      }
      recordings.push({ file, url: URL.createObjectURL(file), samples });
    }
    refusals.textContent = refused.join(' · ');
    render();
    if (refused.length < files.length) onChange?.();
  }

  function remove(recording) {
    recordings.splice(recordings.indexOf(recording), 1);
    URL.revokeObjectURL(recording.url);
    render();
    onChange?.();
  }

  input.addEventListener('change', () => {
    const files = [...input.files];
    input.value = '';
    add(files);
  });

  return {
    element: h('div', { className: 'recordings' }, h('div', { className: 'composer-row' }, button, input), list,
      refusals),
    // The recordings to index, in order: { file, url, samples }, samples
    // their length at 16 kHz.
    recordings: () => [...recordings],
    dropOnto(target) {
      target.addEventListener('dragover', (event) => event.preventDefault());
      target.addEventListener('drop', (event) => {
        event.preventDefault();
        add([...event.dataTransfer.files].filter((file) => file.type.startsWith('audio/')));
      });
    },
    release() {
      for (const recording of recordings) URL.revokeObjectURL(recording.url);
      recordings.length = 0;
    },
  };
}
