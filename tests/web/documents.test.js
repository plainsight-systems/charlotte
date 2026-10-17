import assert from 'node:assert/strict';
import { test } from 'node:test';

import { isTextFile, readDocument } from '../../web/documents.js';

const file = (name, text, type = '') => new File([text], name, { type });

test('text and Markdown files are text files, by extension or by type', () => {
  for (const name of ['notes.txt', 'NOTES.TXT', 'readme.md', 'guide.markdown', 'plain.text']) {
    assert.ok(isTextFile(file(name, '')), name);
  }
  assert.ok(isTextFile(file('untitled', '', 'text/plain')));
  assert.ok(isTextFile(file('untitled', '', 'text/markdown')));
  for (const name of ['report.pdf', 'letter.docx', 'photo.png', 'notes.txt.zip', 'md']) {
    assert.ok(!isTextFile(file(name, '', name.endsWith('.pdf') ? 'application/pdf' : '')), name);
  }
});

test('a text file is read into its paragraphs, as a pasted document is', async () => {
  const read = await readDocument(file('notes.md', '# Rockets\n\nThey lift off.\n\n\n  Then they land.  \n'));
  assert.equal(read.file.name, 'notes.md');
  assert.deepEqual(read.passages, ['# Rockets', 'They lift off.', 'Then they land.']);
});

test('a file that is not text, or holds no paragraph, is refused by name', async () => {
  await assert.rejects(readDocument(file('report.pdf', '%PDF-1.7', 'application/pdf')),
    /only text and Markdown files are read/);
  await assert.rejects(readDocument(file('empty.txt', ' \n\n \n')), /no paragraphs to search/);
});
