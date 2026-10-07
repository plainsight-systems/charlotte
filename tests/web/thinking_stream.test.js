import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createThinkingStream } from '../../web/thinking_stream.js';

// The whole reply split at once: what the stream must reach however the
// reply arrives. Reasoning trimmed; the answer's leading whitespace dropped;
// a reply not opening with <think> all answer, untrimmed.
function whole(text) {
  const open = /^\s*<think>/.exec(text);
  if (open === null) return { reasoning: false, thinking: '', closed: false, answer: text };
  const rest = text.slice(open[0].length);
  const close = rest.indexOf('</think>');
  if (close === -1) return { reasoning: true, thinking: rest.trim(), closed: false, answer: '' };
  return {
    reasoning: true,
    thinking: rest.slice(0, close).trim(),
    closed: true,
    answer: rest.slice(close + '</think>'.length).trimStart(),
  };
}

// The stream's parts, the pieces pushed in turn and then finished.
function streamed(pieces) {
  const stream = createThinkingStream();
  const parts = { reasoning: false, thinking: '', closed: false, answer: '' };
  for (const out of [...pieces.map((p) => stream.push(p)), stream.finish()]) {
    parts.reasoning ||= out.reasoning;
    parts.closed ||= out.closed;
    parts.thinking += out.thinking;
    parts.answer += out.answer;
  }
  return parts;
}

const replies = [
  'Four.',
  '  Four, with leading space.',
  '<think>\nAdd th',
  '<think>\nAdd them.\n</think>\n\n4',
  '<think>\n\n</think>\n\nHi',
  '\n <think> a  b \n</think>  answer  ',
  '<think>x </th',
  '<thinking is not a tag',
  '<thi',
  '   ',
  '',
  '<think>one </think and more</think>two',
];

test('one piece reaches the whole reply split at once', () => {
  for (const text of replies) assert.deepEqual(streamed([text]), whole(text), JSON.stringify(text));
});

test('every split into two pieces, and a character at a time, reach the same', () => {
  for (const text of replies) {
    for (let i = 0; i <= text.length; i++) {
      assert.deepEqual(streamed([text.slice(0, i), text.slice(i)]), whole(text), `${JSON.stringify(text)} at ${i}`);
    }
    assert.deepEqual(streamed([...text]), whole(text), `${JSON.stringify(text)} a character at a time`);
  }
});

test('reasoning is given out as it streams, its trailing whitespace and a possible tag held', () => {
  const stream = createThinkingStream();
  assert.deepEqual(stream.push('<thi'), { reasoning: false, thinking: '', closed: false, answer: '' });
  assert.deepEqual(stream.push('nk>\nAdd'), { reasoning: true, thinking: 'Add', closed: false, answer: '' });
  assert.deepEqual(stream.push(' them. </th'), { reasoning: true, thinking: ' them.', closed: false, answer: '' });
  assert.deepEqual(stream.push('ink>\n\n4'), { reasoning: true, thinking: '', closed: true, answer: '4' });
  assert.deepEqual(stream.push(' more'), { reasoning: true, thinking: '', closed: false, answer: ' more' });
  assert.deepEqual(stream.finish(), { reasoning: true, thinking: '', closed: false, answer: '' });
});

test('a reply without reasoning is given out as answer once its opening rules out <think>', () => {
  const stream = createThinkingStream();
  assert.equal(stream.push(' <').answer, '');
  assert.equal(stream.push('b>').answer, ' <b>');
  assert.equal(stream.push('old').answer, 'old');
});
