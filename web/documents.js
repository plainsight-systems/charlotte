// Axis H. The text files chosen for the search panel's index (search.js): an
// "Add documents" button and the panel as a drop target for them, each listed
// with its name and its passages' count and a button to remove it. The
// photos', recordings' and videos' twin (photos.js, recordings.js,
// videos.js), for text.
//
// createDocuments({ onChange }) returns { element, documents(),
// dropOnto(target), release() }.
//   - Plain text and Markdown only (isTextFile), by extension, since a
//     Markdown file often reaches the page with no MIME type, or by type. A
//     file the page would have to parse to find its text, as a PDF or a Word
//     file is, is named and not added.
//   - A file is read as UTF-8 as it is added and cut into passages as a
//     pasted document is (readDocument, passages.js), so one with no
//     paragraphs is named and not added rather than ending an indexing
//     later. Markdown is read as written: its marks stay in its passages, as
//     they would pasted.
//   - onChange() follows every addition and removal: the index no longer
//     holds the documents shown. A file refused adds nothing, so changes
//     nothing.
//   - The panel's drop target takes text files to this file, images to
//     photos.js, audio to recordings.js and video to videos.js.
//
// Verification: tests/web/documents.test.js — which names and types are text
// files; a file read into its passages; a file with none refused by name.

import { h } from './dom.js';
import { passagesOf } from './passages.js';

const kExtensions = /\.(txt|text|md|markdown)$/i;
const kTypes = new Set(['text/plain', 'text/markdown', 'text/x-markdown']);

// Whether a file is one this page reads as text: by its name's extension, or
// by its MIME type.
export function isTextFile(file) {
  return kExtensions.test(file.name) || kTypes.has(file.type);
}

// A text file read into the passages a search embeds: { file, passages }.
// Throws, naming why, for a file that is not text or holds no paragraph.
export async function readDocument(file) {
  if (!isTextFile(file)) throw new Error('only text and Markdown files are read');
  const passages = passagesOf(await file.text());
  if (passages.length === 0) throw new Error('no paragraphs to search');
  return { file, passages };
}

export function createDocuments({ onChange }) {
  const documents = [];   // { file, passages }, in the order added
  const list = h('ul', { className: 'recording-list' });
  const refusals = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const input = h('input', { type: 'file', accept: '.txt,.text,.md,.markdown,text/plain,text/markdown',
    multiple: true, hidden: true, 'aria-label': 'Documents' });
  const button = h('button', { type: 'button', className: 'action', text: 'Add documents',
    onclick: () => input.click() });

  const render = () => list.replaceChildren(...documents.map((entry) =>
    h('li', {},
      h('span', { text: `${entry.file.name} · ${entry.passages.length} ` +
        `passage${entry.passages.length === 1 ? '' : 's'}` }),
      h('button', { type: 'button', className: 'action', text: '×', 'aria-label': `Remove ${entry.file.name}`,
        onclick: () => remove(entry) }))));

  async function add(files) {
    const refused = [];
    for (const file of files) {
      try {
        documents.push(await readDocument(file));
      } catch (error) {
        refused.push(`${file.name}: ${error.message}`);
      }
    }
    refusals.textContent = refused.join(' · ');
    render();
    if (refused.length < files.length) onChange?.();
  }

  function remove(entry) {
    documents.splice(documents.indexOf(entry), 1);
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
    // The documents to index, in order: { file, passages }.
    documents: () => [...documents],
    dropOnto(target) {
      target.addEventListener('dragover', (event) => event.preventDefault());
      target.addEventListener('drop', (event) => {
        event.preventDefault();
        const files = [...event.dataTransfer.files].filter(isTextFile);
        if (files.length > 0) add(files);
      });
    },
    release() {
      documents.length = 0;
    },
  };
}
