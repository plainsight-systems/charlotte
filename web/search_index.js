// Axis H. The vectors of a document's passages, and a query ranked against
// them. Holds the passages and their vectors; no DOM, no I/O.
//
// createIndex(dimensions) returns { add, rank, size }:
//   - add(passage, vector) keeps the passage and its vector as the worker's
//     message delivered it, a Float32Array, not copied. A vector of another
//     length is refused, naming both lengths: it came from another model.
//   - rank(query, k) scores every passage by its vector's dot product with
//     the query's, which is their cosine, since the embedder delivers unit
//     vectors (runtime/embedder.h), and returns the best k as
//     { passage, score }, best first. Equal scores keep document order, so
//     a ranking is the same every time it is made.
//   - One pass: P × dimensions multiply-adds for P passages, the best k
//     kept as it goes, Θ(P · k) with the panel's k of 5, rather than all P
//     scores sorted.
//   - Each vector stays its own array rather than a row copied into one
//     matrix: a ranking reads every vector once, in order, either way, and
//     a matrix would copy each in and grow with the document.
//
// Verification: tests/web/search_index.test.js — a ranking over hand-made
// unit vectors, best first, its scores their dot products; ties in document
// order; k past P gives P; a vector of another length refused by name.
