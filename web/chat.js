// Axis H: product and interface.
//
// The conversation panel. Holds the messages; renders the whole conversation
// through the model's chat template each turn; sends it with the turn's
// sampling settings and a fresh seed; and renders the reply as it streams.
//
// It holds no cache state. The runtime diffs each turn against what its cache
// holds, so nothing here changes when earlier turns are rewritten.
//
// A conversation longer than the context is refused by the runtime with code
// "prompt-too-long", before anything runs, carrying the prompt's tokens and
// the context's. The panel then drops the oldest exchanges — a user message
// and the reply to it, each — whose share of the rendered text covers the
// excess, estimated from those counts, renders again and resends; should the
// estimate fall short, one more exchange at a time, until the turn fits or
// only the new message is left, which fails as any refusal does. So a
// shortening costs one render and one encode, a few milliseconds, and
// another only where the estimate missed, rather than one an exchange
// dropped. The dropped messages stay dropped, and the log says how many went
// (conversation.js's shortening, tested on its own). The runtime's diff
// finds what of the shortened conversation it still holds
// (logical-overview.md).
//
// A reply's text is drawn at most once a frame: pieces arriving between
// frames are joined, split into reasoning and answer by thinking_stream.js,
// each piece read once, and appended to the text nodes the reply keeps, so
// drawing a reply costs Θ(its length), and the reasoning's open or closed
// state and any selection in it are the reader's while it streams. The
// reasoning is collapsed once, when it finishes.
//
// The log scrolls within the panel. Sending a message brings the log to its
// end, and a streaming reply keeps it there only while the reader is at the
// end, so scrolling up to read an earlier turn is not undone by the next
// frame.

import { shortened, withAssistant, withUser } from './conversation.js';
import { createThinkingStream } from './thinking_stream.js';
import { h } from './dom.js';
import { compileTemplate, offersThinking } from './template.js';

const PLACEHOLDER = 'Choose a model to start.';

// `generate(prompt, { sampling, seed, onText })` starts a reply and returns
// { id, reply }; `cancel(id)` stops it early.
export function createChat(root, { generate, cancel }) {
  let active = null;

  const close = () => {
    if (active !== null && active.generating !== null) cancel(active.generating);
    active = null;
    root.replaceChildren(h('p', { className: 'empty', text: PLACEHOLDER }));
  };

  const open = (model, chat) => {
    close();
    let render;
    try {
      render = compileTemplate(chat);
    } catch (error) {
      root.replaceChildren(h('p', { className: 'chat-error', text: error.message }));
      return;
    }
    active = { generating: null };
    root.replaceChildren(conversationPanel(model, chat, render, active, { generate, cancel }));
  };

  close();
  return { open, close };
}

// Whether a scrolled element shows its end, within a line's slack for
// fractional pixels.
const atEnd = (element) => element.scrollHeight - element.scrollTop - element.clientHeight < 24;

function conversationPanel(model, chat, render, session, { generate, cancel }) {
  let messages = [];

  const log = h('div', { className: 'chat-log', role: 'log', 'aria-live': 'polite' });
  const error = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const input = h('textarea', { rows: 2, placeholder: `Message ${model.name}`, 'aria-label': 'Message' });
  const thinking = offersThinking(chat)
    ? h('input', { type: 'checkbox', checked: true, 'aria-label': 'Thinking' })
    : null;
  const button = h('button', { type: 'submit', className: 'action primary', text: 'Send' });

  const send = async () => {
    const text = input.value.trim();
    if (text === '' || session.generating !== null) return;
    error.textContent = '';

    const mode = thinking?.checked ? 'thinking' : 'default';
    const variables = thinking === null ? {} : { enable_thinking: thinking.checked };
    let turn = withUser(messages, text);

    const userBubble = h('div', { className: 'message user', text });
    // The reply's text in a part of its own, so a redraw leaves the facts
    // appended after it.
    const replyText = h('div');
    const replyBubble = h('div', { className: 'message assistant' }, replyText);
    log.append(userBubble, replyBubble);
    log.scrollTop = log.scrollHeight;
    input.value = '';

    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const started = performance.now();
    // The reply drawn at most once a frame (see the top of this file).
    let reply = '';
    let undrawn = '';
    let frame = null;
    const view = replyView(replyText);
    const draw = () => {
      frame = null;
      const following = atEnd(log);
      view.append(undrawn);
      undrawn = '';
      if (following) log.scrollTop = log.scrollHeight;
    };
    const onText = (piece) => {
      reply += piece;
      undrawn += piece;
      if (frame === null) frame = requestAnimationFrame(draw);
    };
    button.textContent = 'Stop';

    let dropped = 0;
    try {
      let result;
      for (;;) {
        const prompt = render(turn, { variables, now: new Date() });
        const { id, reply: done } = generate(prompt, { sampling: model.policy?.sampling?.[mode], seed, onText });
        session.generating = id;
        try {
          result = await done;
          break;
        } catch (failure) {
          // Longer than the context: drop the oldest exchanges and resend.
          const shorter = failure.code === 'prompt-too-long' ? shortened(turn, failure.counts) : null;
          if (shorter === null) throw failure;
          turn = shorter.messages;
          dropped += shorter.dropped;
        }
      }
      if (frame !== null) cancelAnimationFrame(frame);
      draw();
      view.finish();
      messages = withAssistant(turn, reply);
      if (dropped > 0) {
        log.insertBefore(h('p', { className: 'message-facts',
          text: `${dropped} earlier messages dropped to fit the context` }), userBubble);
      }
      const { stopReason, tokens, promptTokens, reusedTokens } = result;
      const seconds = (performance.now() - started) / 1000;
      const ended = { cancelled: ' · stopped', context: ' · context full', limit: ' · limit reached' }[stopReason] ?? '';
      const following = atEnd(log);
      replyBubble.append(h('p', { className: 'message-facts',
        text: `${tokens} tokens · ${(tokens / seconds).toFixed(1)} tok/s · ` +
              `${reusedTokens} of ${promptTokens} prompt tokens cached · seed ${seed}${ended}` }));
      if (following) log.scrollTop = log.scrollHeight;
    } catch (failure) {
      // The turn did not happen: take it back out, and return the text.
      userBubble.remove();
      replyBubble.remove();
      input.value = text;
      error.textContent = failure.message;
    } finally {
      session.generating = null;
      button.textContent = 'Send';
    }
  };

  const form = h('form', { className: 'composer', onsubmit: (event) => {
    event.preventDefault();
    if (session.generating !== null) cancel(session.generating);
    else send();
  } },
    input,
    h('div', { className: 'composer-row' },
      thinking === null ? h('span') : h('label', { className: 'toggle' }, thinking, ' Thinking'),
      button));

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  return h('div', { className: 'conversation' },
    h('p', { className: 'chat-heading', text: model.measured ? model.name : `${model.name} · unmeasured` }),
    log, error, form);
}

// A reply's reasoning, open while it streams and collapsed once it finishes,
// then its answer, each a text node appended to as the stream gives them.
function replyView(bubble) {
  const stream = createThinkingStream();
  const answer = document.createTextNode('');
  let thought = null;
  let summary = null;
  bubble.append(h('p', { className: 'answer' }, answer));
  const apply = (out) => {
    if (out.reasoning && thought === null) {
      thought = document.createTextNode('');
      summary = h('summary', { text: 'Thinking…' });
      bubble.prepend(h('details', { className: 'thinking', open: true }, summary, h('p', {}, thought)));
    }
    if (out.thinking) thought.appendData(out.thinking);
    if (out.closed) {
      summary.textContent = 'Thought';
      summary.parentElement.open = false;
    }
    if (out.answer) answer.appendData(out.answer);
  };
  return {
    append: (text) => apply(stream.push(text)),
    finish: () => apply(stream.finish()),
  };
}
