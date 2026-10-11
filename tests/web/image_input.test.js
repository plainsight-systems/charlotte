import { test } from 'node:test';
import assert from 'node:assert/strict';

import { BAND_BYTES, bandsOf } from '../../web/image_input.js';

// Every row once, in order, each band within BAND_BYTES unless a row alone is
// larger.
function assertCovers(width, height) {
  const bands = bandsOf(width, height);
  let next = 0;
  for (const { first, rows } of bands) {
    assert.equal(first, next);
    assert.ok(rows >= 1);
    assert.ok(rows === 1 || 4 * width * rows <= BAND_BYTES);
    next += rows;
  }
  assert.equal(next, height);
  return bands;
}

test('a one-pixel-wide image is one band of every row up to the band size', () => {
  assert.deepEqual(assertCovers(1, 300), [{ first: 0, rows: 300 }]);
  assert.equal(assertCovers(1, 16384).length, 1);
});

test('a 4,032 × 3,024 photo is twelve bands of 260 rows, the last the rest', () => {
  const bands = assertCovers(4032, 3024);
  assert.equal(bands.length, 12);
  assert.equal(bands[0].rows, 260);
  assert.deepEqual(bands.at(-1), { first: 2860, rows: 164 });
});

test('a 16,384-wide image is a band of 64 rows', () => {
  assert.equal(assertCovers(16384, 100)[0].rows, 64);
});

test('a row wider than a band is a band alone', () => {
  // Past kMaxSide, which decode refuses; the band count still holds.
  assert.equal(assertCovers(2 ** 21, 3).length, 3);
});

test('an image with no rows has no bands', () => {
  assert.deepEqual(bandsOf(10, 0), []);
});
