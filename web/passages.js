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
