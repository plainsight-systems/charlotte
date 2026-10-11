// Axis H. The photos chosen for the search panel's index (search.js), for a
// model whose catalog entry names a vision companion: an "Add photos" button
// and the panel as a drop target, each photo shown as a thumbnail with a
// button to remove it.
//
// createPhotos({ onChange, decode }) returns { element, photos(),
// dropOnto(target), release() }; decode is image_input.js's unless given.
//   - A photo is checked as it is added: decoded on the page (createImageBitmap
//     decodes off the main thread) and its bitmap closed at once, so one the
//     browser cannot decode, or with a side past 16,384, is named and not
//     added, rather than ending an indexing later. The worker decodes it
//     again when it is indexed (image_input.js): the bitmap is not kept,
//     since a hundred photos' pixels would be gigabytes.
//   - Each thumbnail is the file itself, through URL.createObjectURL, kept
//     until the photo is removed or release() is called; the search panel's
//     results show the same URL.
//   - onChange() follows every addition and removal: the index no longer
//     holds the photos shown.

import { h } from './dom.js';
import { decode as decodeImage } from './image_input.js';

export function createPhotos({ onChange, decode = decodeImage }) {
  const photos = [];   // { file, url }, in the order added
  const strip = h('ul', { className: 'photo-strip' });
  const refusals = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const input = h('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true,
    'aria-label': 'Photos' });
  const button = h('button', { type: 'button', className: 'action', text: 'Add photos',
    onclick: () => input.click() });

  const render = () => strip.replaceChildren(...photos.map((photo) =>
    h('li', {},
      h('img', { src: photo.url, alt: photo.file.name }),
      h('button', { type: 'button', className: 'action', text: '×', 'aria-label': `Remove ${photo.file.name}`,
        onclick: () => remove(photo) }))));

  async function add(files) {
    const refused = [];
    for (const file of files) {
      try {
        const { bitmap } = await decode(file);
        bitmap.close();
      } catch (error) {
        refused.push(`${file.name}: ${error.message}`);
        continue;
      }
      photos.push({ file, url: URL.createObjectURL(file) });
    }
    refusals.textContent = refused.join(' · ');
    render();
    onChange?.();
  }

  function remove(photo) {
    photos.splice(photos.indexOf(photo), 1);
    URL.revokeObjectURL(photo.url);
    render();
    onChange?.();
  }

  input.addEventListener('change', () => {
    const files = [...input.files];
    input.value = '';
    add(files);
  });

  return {
    element: h('div', { className: 'photos' }, h('div', { className: 'composer-row' }, button, input), strip, refusals),
    // The photos to index, in order: { file, url }.
    photos: () => [...photos],
    dropOnto(target) {
      target.addEventListener('dragover', (event) => event.preventDefault());
      target.addEventListener('drop', (event) => {
        event.preventDefault();
        add([...event.dataTransfer.files].filter((file) => file.type.startsWith('image/')));
      });
    },
    release() {
      for (const photo of photos) URL.revokeObjectURL(photo.url);
      photos.length = 0;
    },
  };
}
