// Axis H: product and interface.
//
// The search panel, for a model that embeds (runtime/embedder.h), as the
// chat is for one that generates. A document pasted in is cut into passages,
// each embedded as a document; a query is embedded as a query; and the
// passages are ranked by their vectors' dot product with the query's, the
// best five shown with their scores.
//
// createSearch(root, { embed, onSearch }) returns
// { open(model, loaded), close() }. `embed(text)` sends one EMBED and
// returns its reply, { vector, tokens }; `loaded` is the load's answer,
// { contextOffered, embeddingLength }, which the panel's heading states:
// "768 numbers a text, up to 512 tokens". `onSearch(step, detail)`, if
// given, follows the indexing and each query for the diagnostics
// (diagnostics_recorder.js): 'indexing', 'progress' with "12 of 40",
// 'querying', then 'done', 'stopped', or 'failed' with its message. close()
// leaves the panel empty; the page shows the chat's empty state (app.js).
//
//   - Passages are the document's paragraphs (passages.js), each written
//     into the model's document prompt (task_prompt.js) and sent as one
//     EMBED, in order, one at a time: the embedder runs one encode at a time
//     and refuses a second, Busy, so the panel waits for each answer before
//     sending the next — after Stop too, so embedding again never meets the
//     stopped run's last encode still running. A passage refused as "text-too-long" is cut from the
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
//   - Photos, for a model whose catalog entry names a vision companion
//     (catalog.js, runtime/vision.h): photos chosen or dropped beside the
//     document join the same index, each embedded alone as `<|image|>`
//     with no prompt, as the card says, so one query ranks paragraphs and
//     photos together — the point of a shared space. Each photo is decoded
//     in the worker as it is added (image_input.js): one the browser cannot
//     decode, or larger than a canvas holds, is named and not added. The
//     first photo indexed loads the companion (companion.js), with its own
//     progress and its own refusals by stage, as a model's are: 226 MB the
//     first time, then from the cache. Indexing sends each photo's bands
//     and its EMBED, and begins the next photo's bands as soon as that
//     EMBED is accepted, while its passes run (runtime/vision.h); a photo's
//     failure ends the indexing as a passage's does, whole or not at all. A
//     photo's result is its thumbnail, the file shown by the page
//     (URL.createObjectURL, released with the index), and its score.
//     Under ?fake-runtime=embedder a photo's vector is a hash of its pixels
//     (dev/fake_runtime.js), never shown as a model's.
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
//     to crossings. Measured in Chrome, the two hops take under 0.1 ms,
//     under 1% of the cheapest encode
//     (docs/research/2026-10-11-chrome-encode-benchmark.md).
//   - The copies are kept: encoding into the module's memory directly
//     (encodeInto) saves B bytes of a send, a few KiB, and is the
//     boundary's choice for generate as for embed; transferring the vector
//     instead of cloning it saves 3 KiB. Beside a 6 ms step, neither is
//     worth a change to the boundary.
//   - One passage a step, as the embedder takes them. By the estimates
//     above, several in one step would share only the mapping, about 0.5 ms
//     of 6.1. Measured in Chrome, a 70-token paragraph took 15.6 ms, most of
//     it matrix products of too few tiles to fill the GPU, and takes 11.0
//     with them split along K (kernels/matmul/matmul.h; the same note); what
//     a batch would share is again the step's fixed cost, and the products
//     the split leaves. The embedder would have to mask attention and pool
//     by passage (GPU.6), a change to the encode path's design, not to this
//     page.
//
// A photo, for EmbeddingGemma 2 on the target, by runtime/vision.h's
// estimates: decoding by the browser, its pixels a band a crossing — 4 MiB
// bands, 12 for a 12-megapixel photo — the module's resample, 0.1 to 0.2 s,
// overlapped with the last photo's encode; the vision step, 0.4 to 0.5 s;
// the text step, about 29 ms; the vector's 3 KiB as a passage's. About half
// a second a photo once preparing overlaps encoding, so a hundred photos
// index in under a minute.
//
// Verification: in the app's Chromium with EmbeddingGemma 2, the document
// in tests/fixtures/search/ — paragraphs on unrelated subjects, one of them
// past the context — is indexed, the long paragraph cut and its pieces
// indexed, and each of the fixture's queries ranks the paragraph written to
// answer it first; Stop and an edit each discard the index. With the four
// photographs of tests/fixtures/vision/ added, each of its queries ranks
// its own photograph first, among the photos and the paragraphs. On a phone
// screen, in iOS Safari under
// ?fake-runtime=embedder (dev/fake_runtime.js), the panel with the
// on-screen keyboard up.

import { h } from './dom.js';
import { passagesOf, piecesOf } from './passages.js';
import { createIndex } from './search_index.js';
import { taskPrompt } from './task_prompt.js';

const kResults = 5;

export function createSearch(root, { embed, onSearch }) {
  let active = null;

  const close = () => {
    if (active !== null) active.closed = true;
    active = null;
    root.replaceChildren();
  };

  const open = (model, loaded) => {
    close();
    active = { closed: false };
    root.replaceChildren(searchPanel(model, loaded, active, { embed, onSearch }));
  };

  return { open, close };
}

function searchPanel(model, { contextOffered, embeddingLength }, session, { embed: request, onSearch }) {
  const prompts = model.prompts ?? null;
  // Each EMBED waits for the one before to answer, even one whose run was
  // stopped: the embedder refuses a second while one runs.
  let previous = Promise.resolve();
  const embed = (text) => {
    const reply = previous.then(() => request(text));
    previous = reply.catch(() => {});
    return reply;
  };
  // The index once built; the indexing running, if one is; a query running.
  let ready = null;
  let indexing = null;
  let querying = false;

  const documentInput = h('textarea', { rows: 5, 'aria-label': 'Document',
    placeholder: 'Paste a document. Each paragraph, between blank lines, becomes a passage.' });
  const documentFacts = h('span', { className: 'message-facts' });
  const indexButton = h('button', { type: 'submit', className: 'action primary', text: 'Embed passages' });
  const queryInput = h('input', { type: 'search', 'aria-label': 'Query', placeholder: 'Search the passages' });
  const queryButton = h('button', { type: 'submit', className: 'action primary', text: 'Search' });
  const error = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const results = h('ol', { className: 'search-results', 'aria-live': 'polite' });
  const queryFacts = h('p', { className: 'message-facts' });

  const settle = () => {
    indexButton.textContent = indexing === null ? 'Embed passages' : 'Stop';
    indexButton.disabled = querying;
    queryButton.disabled = ready === null || indexing !== null || querying;
  };

  const clearResults = () => {
    results.replaceChildren();
    queryFacts.textContent = '';
  };

  // Whether a run's answer still counts: the panel open, the run current.
  const current = (run) => !session.closed && indexing === run;

  const index = async () => {
    const passages = passagesOf(documentInput.value);
    error.textContent = '';
    if (passages.length === 0) {
      error.textContent = 'The document has no paragraphs to embed.';
      return;
    }
    const run = {};
    indexing = run;
    ready = null;
    clearResults();
    documentFacts.textContent = `0 of ${passages.length} passages embedded`;
    settle();
    const built = createIndex(embeddingLength);
    const queue = [...passages];
    let tokens = 0;
    const started = performance.now();
    onSearch?.('indexing');
    try {
      while (queue.length > 0) {
        const passage = queue.shift();
        let reply;
        try {
          reply = await embed(taskPrompt(prompts, 'document', passage));
        } catch (failure) {
          if (!current(run)) return;
          // Longer than the context: its pieces go in its place.
          const pieces = failure.code === 'text-too-long' ? piecesOf(passage, failure.counts) : null;
          if (pieces === null) throw failure;
          queue.unshift(...pieces);
          continue;
        }
        if (!current(run)) return;
        built.add(passage, reply.vector);
        tokens += reply.tokens;
        const progress = `${built.size()} of ${built.size() + queue.length}`;
        documentFacts.textContent = `${progress} passages embedded`;
        onSearch?.('progress', progress);
      }
      ready = built;
      const seconds = (performance.now() - started) / 1000;
      documentFacts.textContent =
        `${built.size()} passages embedded · ${tokens} tokens · ${seconds.toFixed(2)} s`;
      onSearch?.('done');
    } catch (failure) {
      if (!current(run)) return;
      documentFacts.textContent = 'Nothing kept.';
      error.textContent = failure.message;
      onSearch?.('failed', failure.message);
    } finally {
      if (indexing === run) indexing = null;
      if (!session.closed) settle();
    }
  };

  // Stop, or an edit: whatever was embedded goes, and an answer still on
  // its way finds its run gone.
  const discard = (fact) => {
    if (indexing !== null) onSearch?.('stopped');
    indexing = null;
    ready = null;
    clearResults();
    documentFacts.textContent = fact;
    settle();
  };

  const search = async () => {
    const text = queryInput.value.trim();
    if (text === '' || ready === null || indexing !== null || querying) return;
    const ranking = ready;
    querying = true;
    error.textContent = '';
    settle();
    const started = performance.now();
    onSearch?.('querying');
    try {
      const { vector, tokens } = await embed(taskPrompt(prompts, 'query', text));
      if (session.closed || ready !== ranking) return;
      const milliseconds = performance.now() - started;
      results.replaceChildren(...ranking.rank(vector, kResults).map(({ passage, score }) =>
        h('li', {}, h('span', { className: 'score', text: score.toFixed(3) }), h('p', { text: passage }))));
      queryFacts.textContent = `${tokens} tokens · ${milliseconds.toFixed(1)} ms · ${ranking.size()} passages ranked`;
      onSearch?.('done');
    } catch (failure) {
      if (session.closed) return;
      const message = failure.code === 'text-too-long'
        ? `The query is ${failure.counts.textTokens} tokens, past the context's ${failure.counts.contextTokens}; ` +
          'a query is not cut.'
        : failure.message;
      error.textContent = message;
      onSearch?.('failed', message);
    } finally {
      querying = false;
      if (!session.closed) settle();
    }
  };

  documentInput.addEventListener('input', () => {
    if (indexing !== null || ready !== null) discard('Edited: embed the passages again.');
  });

  const documentForm = h('form', { className: 'composer', onsubmit: (event) => {
    event.preventDefault();
    if (indexing !== null) discard('Stopped; nothing kept.');
    else index();
  } },
    documentInput,
    h('div', { className: 'composer-row' }, documentFacts, indexButton));

  const queryForm = h('form', { className: 'composer search-query', onsubmit: (event) => {
    event.preventDefault();
    search();
  } },
    h('div', { className: 'composer-row' }, queryInput, queryButton));

  const about = `${embeddingLength} numbers a text, up to ${contextOffered} tokens` +
    (prompts === null ? ' · no task prompts are known for this model: texts are embedded as written' : '');
  settle();
  return h('div', { className: 'conversation search-panel' },
    h('p', { className: 'chat-heading', text: model.measured ? model.name : `${model.name} · unmeasured` }),
    h('p', { className: 'message-facts', text: about }),
    documentForm, queryForm, error, results, queryFacts);
}
