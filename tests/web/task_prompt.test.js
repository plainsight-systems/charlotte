import { test } from 'node:test';
import assert from 'node:assert/strict';

import { taskPrompt } from '../../web/task_prompt.js';

// EmbeddingGemma 2's, as its catalog entry carries them.
const prompts = { query: 'task: search result | query: ', document: 'title: none | text: ' };

test("each task's prompt is written before the text, the text unchanged", () => {
  assert.equal(taskPrompt(prompts, 'query', 'how tall is it?'), 'task: search result | query: how tall is it?');
  assert.equal(taskPrompt(prompts, 'document', '  Two\n\nlines '), 'title: none | text:   Two\n\nlines ');
});

test('a model with no prompts has its text embedded as written', () => {
  assert.equal(taskPrompt(null, 'query', 'as written'), 'as written');
});

test('an unknown task is refused, naming it', () => {
  assert.throws(() => taskPrompt(prompts, 'clustering', 'x'), /"clustering" is not a task/);
});
