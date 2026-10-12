import { test } from 'node:test';
import assert from 'node:assert/strict';

import { clipsOf, mono, pcmWav } from '../../web/audio_input.js';

const kMost = 325280;

// Every sample once, in order, each clip at most `most` and the clips' lengths
// within one of each other.
function assertCovers(n, most) {
  const samples = Float32Array.from({ length: n }, (_, i) => i);
  const clips = clipsOf(samples, most);
  let next = 0;
  for (const clip of clips) {
    assert.equal(clip.samples[0] ?? next, next);
    assert.ok(clip.samples.length <= most);
    assert.equal(clip.start, next / 16000);
    next += clip.samples.length;
    assert.equal(clip.end, next / 16000);
  }
  assert.equal(next, n);
  const lengths = clips.map((c) => c.samples.length);
  assert.ok(Math.max(...lengths) - Math.min(...lengths) <= 1);
  return lengths;
}

test('a recording no longer than the most is one clip, however short', () => {
  assert.deepEqual(assertCovers(160, kMost), [160]);
  assert.deepEqual(assertCovers(161, kMost), [161]);
  assert.deepEqual(assertCovers(kMost, kMost), [kMost]);
});

test('one sample past the most makes two near-equal clips, not a sliver', () => {
  assert.deepEqual(assertCovers(kMost + 1, kMost), [162640, 162641]);
});

test('five minutes is 15 clips of 20 s', () => {
  const lengths = assertCovers(4_800_000, kMost);
  assert.equal(lengths.length, 15);
  assert.ok(lengths.every((n) => n === 320000));
});

test('a clip is a view of the recording, not a copy', () => {
  const samples = new Float32Array(kMost + 1);
  const [first, second] = clipsOf(samples, kMost);
  assert.equal(first.samples.buffer, samples.buffer);
  assert.equal(second.samples.byteOffset, 162640 * 4);
});

test('mono keeps one channel as it is and means two, sample by sample', () => {
  const left = Float32Array.from([0.5, -1, 0.25]);
  const right = Float32Array.from([0.25, 1, -0.75]);
  const one = mono({ numberOfChannels: 1, length: 3, getChannelData: () => left });
  assert.deepEqual([...one], [...left]);
  assert.notEqual(one.buffer, left.buffer);
  const two = mono({ numberOfChannels: 2, length: 3, getChannelData: (c) => (c === 0 ? left : right) });
  assert.deepEqual([...two], [0.375, 0, -0.25]);
});

// A WAV file's bytes: a RIFF header, `before` chunks, then fmt and data.
function wav({ channels = 1, rate = 16000, bits = 16, tag = 1, samples = [], before = [] }) {
  const chunks = [];
  const chunk = (id, body) => {
    const head = new DataView(new ArrayBuffer(8));
    [...id].forEach((c, i) => head.setUint8(i, c.charCodeAt(0)));
    head.setUint32(4, body.byteLength, true);
    chunks.push(new Uint8Array(head.buffer), new Uint8Array(body), new Uint8Array(body.byteLength % 2));
  };
  for (const [id, body] of before) chunk(id, body);
  const fmt = new DataView(new ArrayBuffer(16));
  fmt.setUint16(0, tag, true);
  fmt.setUint16(2, channels, true);
  fmt.setUint32(4, rate, true);
  fmt.setUint32(8, (rate * channels * bits) / 8, true);
  fmt.setUint16(12, (channels * bits) / 8, true);
  fmt.setUint16(14, bits, true);
  chunk('fmt ', fmt.buffer);
  const data = new DataView(new ArrayBuffer(samples.length * 2));
  samples.forEach((v, i) => data.setInt16(2 * i, v, true));
  chunk('data', data.buffer);
  const body = chunks.reduce((n, c) => n + c.byteLength, 4);
  const out = new Uint8Array(8 + body);
  out.set([...'RIFF'].map((c) => c.charCodeAt(0)), 0);
  new DataView(out.buffer).setUint32(4, body, true);
  out.set([...'WAVE'].map((c) => c.charCodeAt(0)), 8);
  let at = 12;
  for (const c of chunks) { out.set(c, at); at += c.byteLength; }
  return out.buffer;
}

test('a 16-bit PCM WAV at 16 kHz is read as each integer over 32,768, the extremes included', () => {
  const samples = [0, 1, -1, 32767, -32768, 12345];
  const buffer = pcmWav(wav({ samples, before: [['LIST', new Uint8Array(3).buffer]] }));
  assert.equal(buffer.numberOfChannels, 1);
  assert.equal(buffer.length, samples.length);
  assert.deepEqual([...buffer.getChannelData(0)], samples.map((v) => Math.fround(v / 32768)));
});

test('a stereo WAV is read channel by channel', () => {
  const buffer = pcmWav(wav({ channels: 2, samples: [1, -2, 3, -4] }));
  assert.equal(buffer.numberOfChannels, 2);
  assert.deepEqual([...buffer.getChannelData(0)], [1 / 32768, 3 / 32768]);
  assert.deepEqual([...buffer.getChannelData(1)], [-2 / 32768, -4 / 32768]);
});

test('anything else is left to the browser: 8 bits, 44.1 kHz, floats, a file cut short, not a WAV', () => {
  assert.equal(pcmWav(wav({ bits: 8, samples: [] })), null);
  assert.equal(pcmWav(wav({ rate: 44100, samples: [1] })), null);
  assert.equal(pcmWav(wav({ tag: 3, samples: [1] })), null);
  assert.equal(pcmWav(wav({ samples: [1, 2, 3] }).slice(0, 48)), null);
  assert.equal(pcmWav(new TextEncoder().encode('not audio at all').buffer), null);
});
