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
// B bytes making N tokens, sent A times for P passages kept — A less P the
// refusals that were cut:
//   - Crossings, each send: two messages, the page to the worker and back;
//     three calls into the module — malloc, bllm_embed, free — and out of
//     it one call for a refusal, two for a vector: bllm_vector, then
//     bllm_reply (wasm_runtime.js). Each kept passage is one step. WASM.2
//     asks a crossing that scales with content for its reason: the embedder
//     encodes one text a step, so each send carries a step's whole work.
//   - The step, by embedder.h's estimates: 1.36 × 10⁸ N multiply-adds,
//     5.6 ms for a 60-token paragraph at the rate it estimates from; the
//     weights, 151 MB, read once a 32-token tile, ceil(N / 32) times — twice
//     for 60 tokens, 0.76 ms at 400 GB/s, under the products; the 296
//     dispatches' 0.44 ms, overlapped; then the 0.5 ms mapping. About 6.1 ms.
//   - Bytes in, B each time: cloned into the worker's message, encoded as
//     UTF-8 (TextEncoder.encode), copied into the module's memory
//     (HEAPU8.set), then normalized by the tokenizer (embedder.h). Writing
//     the prompt before it may or may not copy, as the engine joins strings.
//   - Bytes out, 3 KiB a vector, three copies after the mapping: into the
//     embedder's vector (embedder.h), sliced from the module's memory
//     (wasm_runtime.js), cloned into the reply; then kept as it arrives
//     (search_index.js). A thousand passages hold 3 MB of vectors (GDSA.6).
//   - A query: the same, once, then the ranking's P × 768 multiply-adds
//     (search_index.js), about a millisecond for a thousand passages by an
//     estimate, not a measurement, beside the query's own encode.
//   - Between steps the GPU waits on two message hops and the next
//     passage's encode on the CPU. The loop stays in the page, and the index
//     with it: in the worker, it would save the hops and the vector's clone
//     each passage, but the worker would hold an index across requests, a
//     second owner of state beside the module in the file contract 11 keeps
//     to crossings. The hops are the one per-call cost here with no figure
//     on the target; the page measurement of indexing calibrates them
//     against the steps' estimates, and a share of the indexing that
//     matters moves the loop.
//   - The copies are kept: encoding into the module's memory directly
//     (encodeInto) saves B bytes of a send, a few KiB, and is the
//     boundary's choice for generate as for embed; transferring the vector
//     instead of cloning it saves 3 KiB. Beside a 6 ms step, neither is
//     worth a change to the boundary.
//   - One passage a step, as the embedder takes them. Several in one step
//     would share the mapping, about 0.5 ms of the 6.1 a 60-token paragraph
//     takes, and the padding of each passage's last tile; the products and
//     the weights' reads scale with tokens either way. The embedder would
//     have to mask attention and pool by passage (GPU.6), a change to the
//     encode path's design, not to this page.
//
// Verification: in the app's Chromium with EmbeddingGemma 2, the document
// in tests/fixtures/search/ — paragraphs on unrelated subjects, one of them
// past the context — is indexed, the long paragraph cut and its pieces
// indexed, and each of the fixture's queries ranks the paragraph written to
// answer it first; Stop and an edit each discard the index. On a phone
// screen, in iOS Safari under
// ?fake-runtime=embedder (dev/fake_runtime.js), the panel with the
// on-screen keyboard up.
