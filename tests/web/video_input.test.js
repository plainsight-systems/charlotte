import { test } from 'node:test';
import assert from 'node:assert/strict';

import { clipsOf } from '../../web/video_input.js';

// Every frame once, in order, each clip at most `k`, its span its frames'
// seconds.
function assertCovers(frames, k) {
  const clips = clipsOf(frames, k);
  let next = 0;
  for (const clip of clips) {
    assert.equal(clip.first, next);
    assert.ok(clip.count >= 1 && clip.count <= k);
    assert.equal(clip.start, clip.first);
    assert.equal(clip.end, clip.first + clip.count);
    next += clip.count;
  }
  assert.equal(next, frames);
  return clips.map((c) => c.count);
}

test('a video of fewer frames than a clip holds is one clip', () => {
  assert.deepEqual(assertCovers(1, 4), [1]);
  assert.deepEqual(assertCovers(3, 4), [3]);
  assert.deepEqual(assertCovers(4, 4), [4]);
});

test('the last clip holds what remains', () => {
  assert.deepEqual(assertCovers(5, 4), [4, 1]);
  assert.deepEqual(assertCovers(5, 3), [3, 2]);
});

test('five minutes at 1 a second is 75 clips of 4, or 100 of 3', () => {
  const four = assertCovers(300, 4);
  assert.equal(four.length, 75);
  assert.ok(four.every((n) => n === 4));
  assert.equal(assertCovers(300, 3).length, 100);
});

test('no frames, or no room for one, is refused', () => {
  assert.throws(() => clipsOf(0, 4), RangeError);
  assert.throws(() => clipsOf(4, 0), RangeError);
});
