// Axis H. The videos chosen for the search panel's index (search.js), for a
// model whose catalog entry names a vision companion whose encoder reads
// video (runtime/video.h): an "Add videos" button and the panel as a drop
// target for video files, each listed with its name and length and a button
// to remove it. The recordings' twin (recordings.js), for its own kind.
//
// createVideos({ onChange, open }) returns { element, videos(),
// dropOnto(target), release() }; open is video_input.js's unless given.
//   - A video is checked as it is added: opened on the page and its size and
//     duration kept, the element released, so one the browser can't play,
//     with no picture, or longer than 10 minutes is named and not added,
//     rather than ending an indexing later. It is opened again when it is
//     indexed.
//   - Each entry keeps its file and an object URL for it, which the search
//     panel's results play from at the clip's start (a <video> element, the
//     URL's media fragment #t=start,end), kept until the video is removed or
//     release() is called.
//   - onChange() follows every addition and removal: the index no longer
//     holds the videos shown. A file refused adds nothing, so changes
//     nothing.
//   - The panel's drop target takes images to photos.js, audio to
//     recordings.js and video to this file, each by its MIME type.

import { h } from './dom.js';
import { clock } from './recordings.js';
import { open as openVideo } from './video_input.js';

export function createVideos({ onChange, open = openVideo }) {
  const videos = [];   // { file, url, width, height, duration }, in the order added
  const list = h('ul', { className: 'recording-list' });
  const refusals = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const input = h('input', { type: 'file', accept: 'video/*', multiple: true, hidden: true, 'aria-label': 'Videos' });
  const button = h('button', { type: 'button', className: 'action', text: 'Add videos',
    onclick: () => input.click() });

  const render = () => list.replaceChildren(...videos.map((video) =>
    h('li', {},
      h('span', { text: `${video.file.name} · ${clock(video.duration)}` }),
      h('button', { type: 'button', className: 'action', text: '×', 'aria-label': `Remove ${video.file.name}`,
        onclick: () => remove(video) }))));

  async function add(files) {
    const refused = [];
    for (const file of files) {
      let opened;
      try {
        opened = await open(file);
      } catch (error) {
        refused.push(`${file.name}: ${error.message}`);
        continue;
      }
      const { width, height, duration } = opened;
      opened.release();
      videos.push({ file, url: URL.createObjectURL(file), width, height, duration });
    }
    refusals.textContent = refused.join(' · ');
    render();
    if (refused.length < files.length) onChange?.();
  }

  function remove(video) {
    videos.splice(videos.indexOf(video), 1);
    URL.revokeObjectURL(video.url);
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
    // The videos to index, in order: { file, url, width, height, duration }.
    videos: () => [...videos],
    dropOnto(target) {
      target.addEventListener('dragover', (event) => event.preventDefault());
      target.addEventListener('drop', (event) => {
        event.preventDefault();
        add([...event.dataTransfer.files].filter((file) => file.type.startsWith('video/')));
      });
    },
    release() {
      for (const video of videos) URL.revokeObjectURL(video.url);
      videos.length = 0;
    },
  };
}
