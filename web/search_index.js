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

export function createIndex(dimensions) {
  const passages = [];
  const vectors = [];

  return {
    size: () => passages.length,

    add(passage, vector) {
      if (vector.length !== dimensions) {
        throw new Error(`a vector of ${vector.length} numbers cannot join an index of ${dimensions}`);
      }
      passages.push(passage);
      vectors.push(vector);
    },

    rank(query, k) {
      if (query.length !== dimensions) {
        throw new Error(`a query of ${query.length} numbers cannot rank an index of ${dimensions}`);
      }
      // The best so far, best first; a later passage enters only by beating
      // one, so an equal score keeps document order.
      const best = [];
      for (let p = 0; p < vectors.length; p++) {
        const vector = vectors[p];
        let score = 0;
        for (let i = 0; i < dimensions; i++) score += vector[i] * query[i];
        if (best.length === k && score <= best[k - 1].score) continue;
        let at = best.length;
        while (at > 0 && best[at - 1].score < score) at--;
        best.splice(at, 0, { passage: passages[p], score });
        if (best.length > k) best.pop();
      }
      return best;
    },
  };
}
