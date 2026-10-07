// Axis H. The conversation as the chat template sees it — whole
// { role, content } messages — the split of a reply into its reasoning and
// its answer, and the conversation shortened to fit the context: the oldest
// exchanges dropped — each a user message and the replies before the next,
// any system message kept — as many as the excess asks, by their share of
// the rendered text, or one more where that fell short; null when only the
// newest user message would be left. Pure functions; the chat view holds the state.

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

// The conversation shortened to fit the context, after the runtime refused it
// with `counts`, { promptTokens, contextTokens }. Drops the oldest exchanges —
// each a user message and the replies before the next user message — whose
// share of the messages' text covers the prompt's excess over three quarters
// of the context, leaving a quarter for the reply; at least one. System
// messages and the newest message, the user's turn, are kept. Returns
// { messages, dropped }, or null when there is nothing left to drop.
export function shortened(messages, { promptTokens, contextTokens }) {
  const newest = messages.length - 1;
  const total = messages.reduce((sum, m) => sum + m.content.length, 0);
  const target = Math.floor((contextTokens * 3) / 4);
  const need = promptTokens > target ? ((promptTokens - target) / promptTokens) * total : 0;
  const kept = [];
  let dropped = 0;
  let droppedText = 0;
  let dropping = true;
  for (let i = 0; i < newest; i++) {
    const m = messages[i];
    if (m.role === 'system') {
      kept.push(m);
      continue;
    }
    // Stop only where an exchange begins, once enough has gone.
    if (dropping && m.role === 'user' && dropped > 0 && droppedText >= need) dropping = false;
    if (dropping) {
      dropped++;
      droppedText += m.content.length;
    } else {
      kept.push(m);
    }
  }
  if (dropped === 0) return null;
  return { messages: [...kept, messages[newest]], dropped };
}
