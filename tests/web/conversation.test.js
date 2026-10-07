import { test } from 'node:test';
import assert from 'node:assert/strict';

import { shortened, withAssistant, withUser } from '../../web/conversation.js';

test('messages are added without changing the history they extend', () => {
  const first = withUser([], 'hi');
  const second = withAssistant(first, 'hello');
  assert.deepEqual(first, [{ role: 'user', content: 'hi' }]);
  assert.deepEqual(second.at(-1), { role: 'assistant', content: 'hello' });
});

// Four exchanges of 100 characters a message, then the new user message.
const long = [
  { role: 'system', content: 's'.repeat(40) },
  ...[1, 2, 3, 4].flatMap((n) => [
    { role: 'user', content: `${n}`.repeat(100) },
    { role: 'assistant', content: `${n}`.repeat(100) },
  ]),
  { role: 'user', content: 'now' },
];

test('a conversation too long drops as many oldest exchanges as the excess asks, whole', () => {
  // 1,000 tokens against a context of 1,000: three quarters is 750, so a
  // quarter of the text goes — 210 of 843 characters, two exchanges' worth
  // reached only by the second.
  const { messages, dropped } = shortened(long, { promptTokens: 1000, contextTokens: 1000 });
  assert.equal(dropped, 4);
  assert.deepEqual(messages.map((m) => m.content[0]), ['s', '3', '3', '4', '4', 'n']);
});

test('a small excess still drops one whole exchange, and keeps the system message', () => {
  const { messages, dropped } = shortened(long, { promptTokens: 751, contextTokens: 1000 });
  assert.equal(dropped, 2);
  assert.equal(messages[0].role, 'system');
  assert.equal(messages[1].content[0], '2');
  assert.equal(messages.at(-1).content, 'now');
});

test('a huge excess drops every earlier exchange, and then there is nothing left to drop', () => {
  const { messages, dropped } = shortened(long, { promptTokens: 100000, contextTokens: 1000 });
  assert.equal(dropped, 8);
  assert.deepEqual(messages.map((m) => m.role), ['system', 'user']);
  assert.equal(shortened(messages, { promptTokens: 100000, contextTokens: 1000 }), null);
});
