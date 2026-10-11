// Axis H: product and interface.
//
// The search panel, for a model that embeds (runtime/embedder.h), as the
// chat is for one that generates. A document pasted in is cut into passages,
// each embedded as a document; a query is embedded as a query; and the
// passages are ranked by their vectors' dot product with the query's, the
// best five shown with their scores.
//
// createSearch(root, { embed, onSearch, onChooseModel }) returns
// { open(model, loaded), close() }. `embed(text)` sends one EMBED and
// returns its reply, { vector, tokens }; `loaded` is the load's answer,
// { contextOffered, embeddingLength }, which the panel's heading states:
// "768 numbers a text, up to 512 tokens". `onSearch(step, detail)`, if
// given, follows the indexing and each query for the diagnostics
// (diagnostics_recorder.js).
//
//   - Passages are the document's paragraphs (passages.js), each written
//     into the model's document prompt (task_prompt.js) and sent as one
//     EMBED, in order, one at a time: the embedder runs one encode at a time
//     and refuses a second, Busy, so the panel waits for each answer before
//     sending the next. A passage refused as "text-too-long" is cut from the
//     counts its refusal carries (passages.js) and its pieces sent in its
//     place; a refusal costs an encode on the CPU and no step.
//   - The index is built whole or not at all. Any other failure, or Stop,
//     ends the indexing and discards what was embedded: a ranking over part
//     of a document would show the rest as not matching. Editing the
//     document discards the index, and the results with it, since they rank
//     a text no longer shown.
//   - A query is embedded once the index is built, never while it builds,
//     written into the model's query prompt. A query longer than the context
//     is refused, by name, not cut: its pieces would be other questions.
//   - Each result is a passage as it was embedded, a piece where one was
//     cut, with its score to three places. Facts follow the results: the
//     query's tokens, its round trip in milliseconds and the passages
//     ranked; and the document: the passages embedded, their tokens and the
//     indexing's time. The times are the page's round trips, the worker's
//     hops included, under performance.now()'s 100 µs clamp: what a reader
//     waited, not a measurement of the encode.
//   - A model with no task prompts, named by a pasted URL, has its texts
//     embedded as written, and the panel says so beside the heading's
//     "unmeasured", as the chat labels such a model.
//   - Closing the panel, as choosing another model does, abandons an
//     indexing in progress: no further passage is sent, and an answer still
//     on its way is ignored. The next load cancels an encode still running
//     (src/wasm/bindings.cpp).
//
// What it costs, counted, for EmbeddingGemma 2 on the target; a passage of
// B bytes making N tokens:
//   - Crossings: a passage is one request, two messages — the page to the
//     worker and back — and one call into the module and one out of it, the
//     vector's (wasm_runtime.js). A document of P passages is P of each, and
//     P steps. WASM.2 asks a crossing that scales with content for its
//     reason: the embedder encodes one text a step, so each crossing carries
//     a step's whole work — about 1 ms fixed, the dispatches, the weights'
//     one read and the mapping, and 5.6 ms of products for a 60-token
//     paragraph at the rate embedder.h estimates from — beside which a
//     message's hop is small.
//   - Bytes: the passage, B bytes, is copied three times on its way in —
//     written into the prompt, cloned into the worker's message, encoded as
//     UTF-8 into the module's memory; the vector, 3 KiB, twice on its way
//     out — sliced from the module's memory (wasm_runtime.js) and cloned
//     into the reply — and kept as it arrives (search_index.js). A document
//     of a thousand passages holds 3 MB of vectors (GDSA.6).
//   - A query: the same, once, then the ranking's P × 768 multiply-adds
//     (search_index.js), about a millisecond for a thousand passages by an
//     estimate, not a measurement, beside the query's own encode.
//   - One passage a step, as the embedder takes them. Several in one step
//     would share only the step's fixed cost, about 1 ms of the 6.6 a
//     60-token paragraph takes by the estimates above, since every
//     passage's products scale with its own tokens either way; and the
//     embedder would have to mask attention and pool by passage (GPU.6), a
//     change to the encode path's design, not to this page.
//
// Verification: in the app's Chromium with EmbeddingGemma 2, the document
// in tests/fixtures/search/ — paragraphs on unrelated subjects, one of them
// past the context — is indexed, the long paragraph cut and its pieces
// indexed, and each of the fixture's queries ranks the paragraph written to
// answer it first; Stop and an edit each discard the index. On a phone
// screen, in iOS Safari under
// ?fake-runtime=embedder (dev/fake_runtime.js), the panel with the
// on-screen keyboard up.
