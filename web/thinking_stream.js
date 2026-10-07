// Axis H. A reply, as it streams, split into the reasoning a model writes
// inside <think>…</think> and the answer after it, a piece at a time: each
// piece is read once, so a reply of B bytes costs Θ(B) however it is split.
// Pure; the chat view appends what it returns.
//
// `push(piece)` and `finish()` each return what the text so far adds:
//   { reasoning, thinking, closed, answer }
// `reasoning` is true from the push that finds the reply opens with <think>
// (after any whitespace) — once known, it does not change; `thinking` and
// `answer` are text to append to each part; `closed` is true on the push
// that finds </think>.
//
// The parts, once finished, are the reasoning trimmed and the answer with its
// leading whitespace dropped, as a reader expects them: text that could still
// turn out to be trimmed — the reasoning's trailing whitespace, a tail that
// could begin </think>, a reply's opening that could still be <think> — is
// held until a later piece settles it, or `finish()` gives it out. A reply
// that does not open with <think> is all answer, untrimmed.

const OPEN = '<think>';
const CLOSE = '</think>';

// The length of the longest suffix of `text` that is a proper prefix of
// `word`: text that could still become `word`.
function partialSuffix(text, word) {
  for (let n = Math.min(word.length - 1, text.length); n > 0; n--) {
    if (text.endsWith(word.slice(0, n))) return n;
  }
  return 0;
}

const isSpace = (c) => /\s/.test(c);

export function createThinkingStream() {
  let state = 'opening';   // 'opening', 'thinking', 'answer-lead', 'answer'
  let reasoning = false;
  let held = '';
  let thoughtYet = false;  // whether any reasoning text has been given out

  const result = () => ({ reasoning, thinking: '', closed: false, answer: '' });

  // Reasoning text, its leading whitespace dropped until the first other
  // character.
  const think = (out, text) => {
    const given = thoughtYet ? text : text.trimStart();
    if (given.length > 0) thoughtYet = true;
    out.thinking += given;
  };

  const answerLead = (out, text) => {
    const given = text.trimStart();
    if (given.length > 0) {
      state = 'answer';
      out.answer += given;
    }
  };

  const thinkingPiece = (out, text) => {
    const close = text.indexOf(CLOSE);
    if (close !== -1) {
      think(out, text.slice(0, close).trimEnd());
      out.closed = true;
      state = 'answer-lead';
      held = '';
      answerLead(out, text.slice(close + CLOSE.length));
      return;
    }
    // Hold a tail that could begin </think>, and the whitespace before it.
    let keep = text.length - partialSuffix(text, CLOSE);
    while (keep > 0 && isSpace(text[keep - 1])) keep--;
    think(out, text.slice(0, keep));
    held = text.slice(keep);
  };

  const push = (piece) => {
    const out = result();
    if (state === 'opening') {
      held += piece;
      const start = held.trimStart();
      if (start.startsWith(OPEN)) {
        reasoning = true;
        state = 'thinking';
        const rest = start.slice(OPEN.length);
        held = '';
        out.reasoning = true;
        thinkingPiece(out, rest);
      } else if (start.length === 0 || OPEN.startsWith(start)) {
        // Whitespace, or the beginning of <think>: not yet known.
      } else {
        state = 'answer';
        out.answer = held;
        held = '';
      }
      return out;
    }
    if (state === 'thinking') thinkingPiece(out, held + piece);
    else if (state === 'answer-lead') answerLead(out, piece);
    else out.answer = piece;
    return out;
  };

  // The end of the reply: what was held is given out, trimmed as the part it
  // belongs to.
  const finish = () => {
    const out = result();
    if (state === 'opening') {
      out.answer = held;
      state = 'answer';
    } else if (state === 'thinking') {
      think(out, held.trimEnd());
    }
    held = '';
    return out;
  };

  return { push, finish };
}
