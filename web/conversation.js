// Axis H. The conversation as the chat template sees it — whole
// { role, content } messages — the split of a reply into its reasoning and
// its answer, and the conversation shortened to fit the context: its oldest
// exchange dropped, a user message and the replies before the next, any
// system message kept; null when only the newest user message is left to
// drop. Pure functions; the chat view holds the state.

export const withUser = (messages, content) => [...messages, { role: 'user', content }];

export const withAssistant = (messages, content) => [...messages, { role: 'assistant', content }];

// Splits a reply, possibly still streaming, into the reasoning a model wrote
// inside <think>…</think> and the answer after it. `thinking` is null when the
// reply has no reasoning; `thinkingDone` is false while it is still open.
export function splitThinking(text) {
  const open = /^\s*<think>/.exec(text);
  if (open === null) return { thinking: null, thinkingDone: true, answer: text };
  const rest = text.slice(open[0].length);
  const close = rest.indexOf('</think>');
  if (close === -1) return { thinking: rest.trim(), thinkingDone: false, answer: '' };
  return {
    thinking: rest.slice(0, close).trim(),
    thinkingDone: true,
    answer: rest.slice(close + '</think>'.length).trimStart(),
  };
}
