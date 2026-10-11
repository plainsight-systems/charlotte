// Axis H. A document cut into the passages the search panel embeds, and a
// passage the runtime refused as longer than the context cut into pieces
// that fit. Pure functions; the panel holds the state (search.js).
//
//   - passagesOf(document): its paragraphs, the text between blank lines,
//     each trimmed, the empty dropped, in order. A paragraph is what a
//     reader recognizes in a result, and a size retrieval commonly embeds.
//   - piecesOf(passage, { textTokens, contextTokens }): the passage the
//     runtime refused with those counts, "text-too-long"
//     (runtime/embedder.h), cut into ceil(textTokens / target) pieces of
//     about equal length, the target three quarters of the context: room
//     for the estimate's error, since tokens do not spread evenly over
//     characters, and for the task prompt's tokens, which the counts
//     include. Each cut moves to the nearest sentence end, else the nearest
//     space, else stays where it falls, never inside a surrogate pair; the
//     pieces are trimmed, and none is empty.
//   - The counts are an estimate's input, not a promise: a refusal before
//     the encode carries the text's least count (tokenizer/tokenizer.h), a
//     lower bound. So a piece can be refused in turn, and is cut the same
//     way. That ends: every piece is shorter than the passage it came from,
//     and a passage of one code point returns null, nothing left to cut,
//     which the panel reports as the refusal it was.
//
// Verification: tests/web/passages.test.js — paragraphs across blank runs
// and leading and trailing space; the piece count for given counts; cuts at
// sentence ends, then spaces, then anywhere, a surrogate pair kept whole;
// the pieces, joined, are the passage less the whitespace at the cuts; a
// passage of one code point gives null.

export function passagesOf(document) {
  return document.split(/\r?\n\s*\r?\n/).map((paragraph) => paragraph.trim()).filter((p) => p !== '');
}

// A sentence's end: a stop, any closing quotes or brackets, then a space —
// or a full-width stop, which needs none.
const SENTENCE_END = /[.!?]["'”’)\]]*(?=\s)|[。！？]/g;

export function piecesOf(passage, { textTokens, contextTokens }) {
  const text = passage.trim();
  const codePoints = [...text].length;
  if (codePoints <= 1) return null;
  const target = Math.max(1, Math.floor((contextTokens * 3) / 4));
  const count = Math.min(codePoints, Math.max(2, Math.ceil(textTokens / target)));

  const sentenceEnds = [...text.matchAll(SENTENCE_END)].map((m) => m.index + m[0].length);
  const spaces = [...text.matchAll(/\s/g)].map((m) => m.index);
  // A cut moves no further than half a piece, so the cuts stay in order.
  const reach = text.length / (2 * count);

  const cuts = [0];
  for (let i = 1; i < count; i++) {
    const ideal = Math.round((i * text.length) / count);
    const cut = nearest(sentenceEnds, ideal, reach) ?? nearest(spaces, ideal, reach) ?? wholeCodePoint(text, ideal);
    if (cut > cuts.at(-1) && cut < text.length) cuts.push(cut);
  }
  cuts.push(text.length);
  const pieces = [];
  for (let i = 1; i < cuts.length; i++) {
    const piece = text.slice(cuts[i - 1], cuts[i]).trim();
    if (piece !== '') pieces.push(piece);
  }
  return pieces;
}

// The position in `positions` nearest `ideal` within `reach`, or null.
function nearest(positions, ideal, reach) {
  let best = null;
  for (const position of positions) {
    if (Math.abs(position - ideal) > reach) continue;
    if (best === null || Math.abs(position - ideal) < Math.abs(best - ideal)) best = position;
  }
  return best;
}

// `at`, or one past it where it falls inside a surrogate pair.
function wholeCodePoint(text, at) {
  const before = text.charCodeAt(at - 1);
  return before >= 0xd800 && before <= 0xdbff ? at + 1 : at;
}
