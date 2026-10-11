import { test } from 'node:test';
import assert from 'node:assert/strict';

import { passagesOf, piecesOf } from '../../web/passages.js';

const refused = (textTokens) => ({ textTokens, contextTokens: 512 });

// The pieces, in order, are the passage less the whitespace at the cuts.
function assertCovers(pieces, passage) {
  let from = 0;
  for (const piece of pieces) {
    const at = passage.indexOf(piece, from);
    assert.ok(at >= from, `"${piece.slice(0, 20)}…" is not next in the passage`);
    assert.match(passage.slice(from, at), /^\s*$/);
    from = at + piece.length;
  }
  assert.match(passage.slice(from), /^\s*$/);
}

test('a document is its paragraphs, trimmed, across runs of blank lines', () => {
  const document = '\n  First, one line.\n\n\n  Second,\nover two lines. \n \t\nThird\r\n\r\nFourth  \n\n  ';
  assert.deepEqual(passagesOf(document), ['First, one line.', 'Second,\nover two lines.', 'Third', 'Fourth']);
  assert.deepEqual(passagesOf(' \n\n '), []);
});

test('a refused passage is cut into as many pieces as its tokens over three quarters of the context', () => {
  const passage = Array.from({ length: 30 }, (_, i) => `Sentence ${i} says a little.`).join(' ');
  // 384 tokens a piece: 1,000 make three, 385 two, and a count under the
  // target still two, since the runtime refused it.
  assert.equal(piecesOf(passage, refused(1000)).length, 3);
  assert.equal(piecesOf(passage, refused(385)).length, 2);
  assert.equal(piecesOf(passage, refused(300)).length, 2);
});

test('cuts fall at sentence ends where there are some', () => {
  const passage = Array.from({ length: 30 }, (_, i) => `Sentence ${i} says “a little.”`).join(' ') + ' 灯台。階段！';
  const pieces = piecesOf(passage, refused(2000));
  assert.equal(pieces.length, 6);
  for (const piece of pieces.slice(0, -1)) assert.match(piece, /\.”$/);
  assertCovers(pieces, passage);
});

test('then at spaces, then anywhere', () => {
  const words = Array.from({ length: 200 }, (_, i) => `word${i}`).join(' ');
  const byWord = piecesOf(words, refused(1000));
  assert.equal(byWord.length, 3);
  for (const piece of byWord) assert.match(piece, /^word\d+( word\d+)*$/);
  assertCovers(byWord, words);

  const run = 'a'.repeat(5000);
  const anywhere = piecesOf(run, refused(5000));
  assert.equal(anywhere.length, 14);
  assert.equal(anywhere.join(''), run);
});

test('a cut never splits a surrogate pair', () => {
  const run = '😀'.repeat(301);
  const pieces = piecesOf(run, refused(1000));
  assert.equal(pieces.join(''), run);
  for (const piece of pieces) assert.equal(piece, '😀'.repeat([...piece].length));
});

test('a passage of one code point is left to its refusal', () => {
  assert.equal(piecesOf('😀', refused(600)), null);
  assert.equal(piecesOf('  x ', refused(600)), null);
});

test('every piece is shorter than its passage, so cutting again ends', () => {
  let pieces = ['b'.repeat(40)];
  for (let round = 0; round < 10 && pieces.length < 40; round++) {
    pieces = pieces.flatMap((piece) => piecesOf(piece, refused(600)) ?? [piece]);
  }
  assert.equal(pieces.length, 40);
});
