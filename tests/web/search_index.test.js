import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createIndex } from '../../web/search_index.js';

const unit = (...values) => {
  const norm = Math.hypot(...values);
  return Float32Array.from(values, (v) => v / norm);
};
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);

test('passages rank best first, each scored by its dot product with the query', () => {
  const index = createIndex(3);
  const vectors = { a: unit(1, 0, 0), b: unit(1, 1, 0), c: unit(0, 0, 1), d: unit(-1, 0, 0) };
  for (const [passage, vector] of Object.entries(vectors)) index.add(passage, vector);
  const query = unit(1, 0.2, 0);
  const ranked = index.rank(query, 3);
  assert.deepEqual(ranked.map((r) => r.passage), ['a', 'b', 'c']);
  for (const { passage, score } of ranked) assert.equal(score, dot(vectors[passage], query));
  assert.equal(index.size(), 4);
});

test('equal scores keep document order', () => {
  const index = createIndex(2);
  for (const passage of ['first', 'second', 'third']) index.add(passage, unit(0, 1));
  index.add('better', unit(1, 1));
  assert.deepEqual(index.rank(unit(0, 1), 3).map((r) => r.passage), ['first', 'second', 'third']);
  assert.deepEqual(index.rank(unit(1, 1), 2).map((r) => r.passage), ['better', 'first']);
});

test('asking for more than the index holds gives every passage', () => {
  const index = createIndex(2);
  index.add('only', unit(1, 0));
  assert.equal(index.rank(unit(1, 0), 5).length, 1);
  assert.deepEqual(createIndex(2).rank(unit(1, 0), 5), []);
});

test('a vector of another length is refused, naming both', () => {
  const index = createIndex(3);
  assert.throws(() => index.add('x', new Float32Array(2)), /2 numbers .* index of 3/);
  assert.throws(() => index.rank(new Float32Array(4), 1), /4 numbers .* index of 3/);
});
