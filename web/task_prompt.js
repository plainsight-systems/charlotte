// Axis H. A text written into the task prompt an embedding model was trained
// with, as the chat writes a conversation into a generator's chat template
// (template.js). Pure.
//
// taskPrompt(prompts, task, text): `task` is 'query' or 'document', and any
// other is an error, by name. `prompts` is the model's { query, document },
// from its entry in models.json (catalog.js), as its card gives them — for
// EmbeddingGemma 2, "task: search result | query: " and "title: none |
// text: ", the card's search query and its document with no title, the
// prompts sentence-transformers names "query" and "document" in the model's
// config_sentence_transformers.json. The prompt is written before the text,
// nothing after it: the file's ends are the runtime's (runtime/ends.h).
//
// A model with no prompts — one named by a pasted URL, whose card the page
// has not read — has its text embedded as written, `prompts` null; the
// search panel says so (search.js), and the model is labelled unmeasured.
//
// Verification: tests/web/task_prompt.test.js — each task's prompt before
// the text, the text unchanged; null prompts give the text as written; an
// unknown task is refused, naming it.

const TASKS = ['query', 'document'];

export function taskPrompt(prompts, task, text) {
  if (!TASKS.includes(task)) throw new Error(`"${task}" is not a task; the tasks are ${TASKS.join(' and ')}`);
  return prompts === null ? text : prompts[task] + text;
}
