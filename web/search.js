// Axis H: product and interface.
//
// The search panel, for a model that embeds (runtime/embedder.h), as the
// chat is for one that generates. A document pasted in, and each text file
// added, is cut into passages, each embedded as a document; a query is
// embedded as a query; and the passages are ranked by their vectors' dot
// product with the query's, the best five shown with their scores.
//
// createSearch(root, { embed, prepareImage, prepareAudio, planVideo,
// prepareVideo, companion, audioCompanion, onSearch }) returns { open(model,
// loaded), close(), showCompanion(state) }. `embed(text, media)` sends one
// EMBED, `media` null, 'image', 'audio' or 'video', and returns its reply,
// { vector, tokens }; `prepareImage(file)` sends one PREPARE_IMAGE,
// `prepareAudio(samples)` one PREPARE_AUDIO, `planVideo(width, height,
// duration)` one PLAN_VIDEO and `prepareVideo(width, height, frames)` one
// PREPARE_VIDEO; `companion` and `audioCompanion` are companion.js's,
// for images and for audio, whose onChange the page points at
// showCompanion; `loaded`
// is the load's answer,
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
//     a text no longer shown, and the panel says to search again.
//   - Search is the panel's one action. Pressed with no index current, it
//     builds one first, its progress shown and the button turned to Stop,
//     then embeds the query; pressed again with nothing changed, it embeds
//     only the query. A press is never ignored: with no query, or nothing
//     to search, the panel says what to add. The query is written into the
//     model's query prompt. A query longer than the context is refused, by
//     name, not cut: its pieces would be other questions.
//   - Each result is a passage as it was embedded, a piece where one was
//     cut, with its score to three places. Facts follow the results: the
//     query's tokens, its round trip in milliseconds and the passages
//     ranked; and the document: the passages embedded, their tokens and the
//     indexing's time. The times are the page's round trips, the worker's
//     hops included, under performance.now()'s 100 µs clamp: what a reader
//     waited, not a measurement of the encode.
//   - Documents: text and Markdown files added beside the pasted document
//     (documents.js), each cut into passages as it is. Their passages follow
//     the pasted document's, in the order the files were added, and are
//     embedded and cut the same way; a result from a file names it above
//     its passage. Adding or removing one discards the index, as an edit
//     does.
//   - A model with no task prompts, named by a pasted URL, has its texts
//     embedded as written, and the panel says so beside the heading's
//     "unmeasured", as the chat labels such a model.
//   - Photos, for a model whose catalog entry names a vision companion
//     (catalog.js, runtime/vision.h): photos chosen or dropped beside the
//     document join the same index, each embedded alone as `<|image|>`
//     with no prompt, as the card says, so one query ranks paragraphs and
//     photos together — the point of a shared space. Each photo is checked
//     as it is added (photos.js): one the browser cannot decode, or larger
//     than a canvas holds, is named and not added. Adding or removing one
//     discards the index, as an edit does. The
//     first photo indexed loads the companion (companion.js), with its own
//     progress and its own refusals by stage, as a model's are: 226 MB the
//     first time, then from the cache. Indexing sends each photo's bands
//     and its EMBED, and begins the next photo's bands as soon as that
//     EMBED is sent, while its passes run (runtime/vision.h): the worker
//     reads its messages in order and an EMBED reaches the module before the
//     worker reads the next, and the module copies the slot's pixels when it
//     submits the step, so the next photo may fill the slot; a photo's
//     failure ends the indexing as a passage's does, whole or not at all. A
//     photo's result is its thumbnail, the file shown by the page
//     (URL.createObjectURL, released with the photo), and its score.
//     Stopping leaves a companion's download or load running, so the next
//     indexing finds it further on.
//     Under ?fake-runtime=embedder a photo's vector is a hash of its pixels
//     (dev/fake_runtime.js), never shown as a model's.
//   - Recordings, for a model whose catalog entry names an audio companion
//     (catalog.js, runtime/audio.h), likewise: chosen or dropped beside the
//     document and the photos (recordings.js), joining the same index after
//     the photos. The first recording indexed loads the audio companion, 328
//     MB the first time, once the image companion's ensure, if any, has
//     settled. Each recording is decoded on the page and cut into clips of
//     at most the module's mostSamples, equal in length (audio_input.js),
//     and each clip is one entry of the index, embedded alone as
//     `<|audio|>` with no prompt, as the card says: a query finds where in a
//     recording it is answered, as it finds a passage of a document.
//     Indexing sends each clip's PREPARE_AUDIO and its EMBED, the next
//     clip's PREPARE_AUDIO sent once that EMBED is, as a photo's bands are;
//     a clip's failure ends the indexing, whole or not at all. A clip's
//     result is the recording's name, its span — "0:20–0:40" — a player for
//     that span, and its score. Under ?fake-runtime=embedder a clip's vector
//     is its loudness over time, binned (dev/fake_runtime.js), never shown
//     as a model's.
//   - Videos, for a model whose vision companion reads video
//     (runtime/video.h; the catalog's vision.video), likewise: chosen or
//     dropped beside the rest (videos.js), joining the index after the
//     recordings, through the image companion — no download of their own;
//     a companion whose load says it attached no video stage refuses the
//     indexing by name. Each video is opened on
//     the page, planned by the module, and cut into clips of k frames at 1
//     a second, 4 for a 16:9 video (video_input.js); each clip is one
//     entry, embedded alone as `<|video|>` with no prompt, as the card
//     says. Indexing sends each clip's PREPARE_VIDEO and its EMBED, the
//     next clip's frames seeked and sent once that EMBED is, as a photo's
//     bands are; a clip's failure ends the indexing, whole or not at all.
//     A clip's result is the video's name, its span — "0:04–0:08" — a
//     player for that span, and its score. Under ?fake-runtime=embedder a
//     clip's vector is its frames' colour histogram, summed
//     (dev/fake_runtime.js), never shown as a model's.
//   - Closing the panel, as choosing another model does, abandons an
//     indexing in progress: no further passage is sent, and an answer still
//     on its way is ignored. The companion is released with it. The next load cancels an encode still running
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
// A recording, for EmbeddingGemma 2 on the target, by runtime/audio.h's
// estimates: the browser's decode; then a clip of 20 s a crossing of 1.28
// MB, its spectrogram a few milliseconds, overlapped with the last clip's
// encode; the audio step about 0.1 s; the text step about 45 ms. About
// 0.15 s a 20 s clip: a 5 minute recording, 15 clips, in about 2.3 s.
// Measured in the app's Chromium on an Apple M3 Max: that recording's 15
// clips indexed in about 2.0 s.
//
// A video, for EmbeddingGemma 2 on the target, by runtime/video.h's
// estimates from a measured frame: a seek and a bitmap a second; a 4-frame
// clip's four vision steps, about 0.40 s, and its text step, about 45 ms;
// about 0.45 s for 4 s of video, a 5 minute video's 75 clips in about 34 s.
// Measured in the app's Chromium on an Apple M3 Max: the four NASA videos,
// 31 s in 8 clips, indexed in 3.85 s, about 0.48 s a clip; their 31 seeks
// and bitmaps alone 0.59 s; 720 × 384 photos on the same build 0.125 s each.
//
// A photo, for EmbeddingGemma 2 on the target, by runtime/vision.h's
// estimates: decoding by the browser, its pixels a band a crossing — 4 MiB
// bands, 12 for a 12-megapixel photo — the module's resample, 0.1 to 0.2 s,
// overlapped with the last photo's encode; the vision step, 0.4 to 0.5 s;
// the text step, about 29 ms; the vector's 3 KiB as a passage's. About half
// a second a photo once preparing overlaps encoding, so a hundred photos
// index in under a minute. Measured in the app's Chromium on an Apple M3
// Max: a 4,032 × 3,024 JPEG alone, 270 tokens, 0.49 s with nothing to
// overlap; 300 × 300 drawings of 80 tokens about 0.09 s each.
//
// Verification: in the app's Chromium with EmbeddingGemma 2, the document
// in tests/fixtures/search/ — paragraphs on unrelated subjects, one of them
// past the context — is indexed, the long paragraph cut and its pieces
// indexed, and each of the fixture's queries ranks the paragraph written to
// answer it first; Stop and an edit each discard the index. With the four
// scenes of tests/fixtures/vision/ added as PNGs
// (tools/make_vision_photos.py), each query of
// tests/fixtures/reference/vision_embeddings.inc ranks its own scene first,
// among the photos and the paragraphs, each score llama.cpp's to three
// places; a file that is not an image is refused by name. With the four
// recordings of tests/fixtures/audio/ added, each of their queries ranks its
// own recording first among the recordings, the photos and the paragraphs,
// each score llama.cpp's to three places; a 5 minute recording is indexed
// as 15 clips, the one a query names ranked with its span. With the four
// NASA videos of tests/fixtures/video/ added, each of their queries ranks
// its own video's clips first among everything indexed, each score within
// 0.005 of llama.cpp's on FFmpeg's frames (tests/fixtures/video/
// page_scores.json), the two decoders' difference. On a phone
// screen, in iOS Safari under
// ?fake-runtime=embedder (dev/fake_runtime.js), the panel with the
// on-screen keyboard up.

import { clipsOf, decode as decodeAudio, mono } from './audio_input.js';
import { describeCompanion } from './companion.js';
import { h } from './dom.js';
import { passagesOf, piecesOf } from './passages.js';
import { createDocuments } from './documents.js';
import { createPhotos } from './photos.js';
import { clock, createRecordings } from './recordings.js';
import { createIndex } from './search_index.js';
import { taskPrompt } from './task_prompt.js';
import { clipsOf as videoClipsOf, frameAt, open as openVideo } from './video_input.js';
import { createVideos } from './videos.js';

const kResults = 5;
// What a photo is embedded as: the image alone, with no task prompt, as the
// model card gives it.
const kPhotoText = '<|image|>';
// And a clip: the clip alone.
const kClipText = '<|audio|>';
// And a video's clip.
const kVideoText = '<|video|>';

// Words joined as a list: "a", "a and b", "a, b and c", or with "or".
function listed(words, conjunction = 'and') {
  return words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} ${conjunction} ${words.at(-1)}`;
}

export function createSearch(root,
  { embed, prepareImage, prepareAudio, planVideo, prepareVideo, companion, audioCompanion, onSearch }) {
  let active = null;

  const close = () => {
    if (active !== null) {
      active.closed = true;
      active.release();
      companion?.release();
      audioCompanion?.release();
    }
    active = null;
    root.replaceChildren();
  };

  const open = (model, loaded) => {
    close();
    active = { closed: false, release: () => {}, showCompanion: () => {} };
    root.replaceChildren(searchPanel(model, loaded, active,
      { embed, prepareImage, prepareAudio, planVideo, prepareVideo, companion, audioCompanion, onSearch }));
  };

  return { open, close, showCompanion: (state) => active?.showCompanion(state) };
}

function searchPanel(model, { contextOffered, embeddingLength }, session,
  { embed: request, prepareImage, prepareAudio, planVideo, prepareVideo, companion, audioCompanion, onSearch }) {
  const prompts = model.prompts ?? null;
  const sees = model.vision !== undefined && companion !== undefined;
  const hears = model.audio !== undefined && audioCompanion !== undefined;
  const watches = sees && model.vision.video === true && prepareVideo !== undefined;
  // Each EMBED waits for the one before to answer, even one whose run was
  // stopped: the embedder refuses a second while one runs.
  let previous = Promise.resolve();
  const embed = (text) => {
    const reply = previous.then(() => request(text, null));
    previous = reply.catch(() => {});
    return reply;
  };
  // A photo's EMBED, sent now, once every EMBED before has answered, so the
  // caller knows it has reached the worker before it prepares the next.
  const embedPhoto = async () => {
    await previous;
    const reply = request(kPhotoText, 'image');
    previous = reply.catch(() => {});
    return { reply };
  };
  // And a clip's, of a recording or of a video.
  const embedClip = async (text = kClipText, media = 'audio') => {
    await previous;
    const reply = request(text, media);
    previous = reply.catch(() => {});
    return { reply };
  };
  // The index once built; the indexing running, if one is; a query running.
  let ready = null;
  let indexing = null;
  let querying = false;

  const documentInput = h('textarea', { rows: 5, 'aria-label': 'Document',
    placeholder: 'Paste a document. Each paragraph, between blank lines, becomes a passage.' });
  const documentFacts = h('span', { className: 'message-facts' });
  const queryInput = h('input', { type: 'search', 'aria-label': 'Query',
    placeholder: `Search the ${listed(['passages', sees ? 'photos' : null, hears ? 'recordings' : null,
      watches ? 'videos' : null].filter(Boolean))}` });
  const queryButton = h('button', { type: 'submit', className: 'action primary', text: 'Search' });
  const error = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const results = h('ol', { className: 'search-results', 'aria-live': 'polite' });
  const queryFacts = h('p', { className: 'message-facts' });
  const companionFacts = h('p', { className: 'message-facts', 'aria-live': 'polite' });
  session.showCompanion = (state) => { companionFacts.textContent = describeCompanion(state); };
  const documents = createDocuments({ onChange: () => {
    if (indexing !== null || ready !== null) discard('Changed: search again to include it.');
  } });
  const photos = sees ? createPhotos({ onChange: () => {
    if (indexing !== null || ready !== null) discard('Changed: search again to include it.');
  } }) : null;
  const recordings = hears ? createRecordings({ onChange: () => {
    if (indexing !== null || ready !== null) discard('Changed: search again to include it.');
  } }) : null;
  const videos = watches ? createVideos({ onChange: () => {
    if (indexing !== null || ready !== null) discard('Changed: search again to include it.');
  } }) : null;
  session.release = () => {
    documents.release();
    photos?.release();
    recordings?.release();
    videos?.release();
  };

  const settle = () => {
    queryButton.textContent = indexing === null ? 'Search' : 'Stop';
    queryButton.disabled = querying;
  };

  const clearResults = () => {
    results.replaceChildren();
    queryFacts.textContent = '';
  };

  // Whether a run's answer still counts: the panel open, the run current.
  const current = (run) => !session.closed && indexing === run;

  const index = async () => {
    // The pasted document's passages, then each added file's, a file's
    // passages naming it: { text, from }, from null for the pasted one.
    const passages = [...passagesOf(documentInput.value).map((text) => ({ text, from: null })),
      ...documents.documents().flatMap(({ file, passages: texts }) =>
        texts.map((text) => ({ text, from: file.name })))];
    const pictures = photos?.photos() ?? [];
    const sounds = recordings?.recordings() ?? [];
    const films = videos?.videos() ?? [];
    error.textContent = '';
    if (passages.length === 0 && pictures.length === 0 && sounds.length === 0 && films.length === 0) {
      error.textContent = sees || hears
        ? `Add ${listed(['a document', sees ? 'photos' : null, hears ? 'recordings' : null,
          watches ? 'videos' : null].filter(Boolean), 'or')} to search.`
        : 'Paste or add a document to search.';
      return;
    }
    const run = {};
    indexing = run;
    ready = null;
    clearResults();
    // A recording's clips are counted once the audio encoder says how long a
    // clip may be, and a video's once the module plans it; until then, one
    // each.
    let clipCount = sounds.length;
    let videoClipCount = films.length;
    const left = () => pictures.length + clipCount + videoClipCount;
    documentFacts.textContent = `0 of ${passages.length + left()} embedded`;
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
          reply = await embed(taskPrompt(prompts, 'document', passage.text));
        } catch (failure) {
          if (!current(run)) return;
          // Longer than the context: its pieces go in its place.
          const pieces = failure.code === 'text-too-long' ? piecesOf(passage.text, failure.counts) : null;
          if (pieces === null) throw failure;
          queue.unshift(...pieces.map((text) => ({ text, from: passage.from })));
          continue;
        }
        if (!current(run)) return;
        built.add(passage.from === null ? passage.text : passage, reply.vector);
        tokens += reply.tokens;
        const progress = `${built.size()} of ${built.size() + queue.length + left()}`;
        documentFacts.textContent = `${progress} embedded`;
        onSearch?.('progress', progress);
      }
      const kept = built.size();
      if (pictures.length > 0) {
        await companion.ensure(model);
        if (!current(run)) return;
        // Each photo's bands go in while the one before is encoded.
        let prepared = prepareImage(pictures[0].file);
        for (let i = 0; i < pictures.length; i++) {
          await prepared;
          if (!current(run)) return;
          const { reply } = await embedPhoto();
          prepared = i + 1 < pictures.length ? prepareImage(pictures[i + 1].file) : null;
          prepared?.catch(() => {});   // its failure is met when it is awaited
          const { vector, tokens: made } = await reply;
          if (!current(run)) return;
          built.add(pictures[i], vector);
          tokens += made;
          const progress = `${built.size()} of ${kept + left()}`;
          documentFacts.textContent = `${progress} embedded`;
          onSearch?.('progress', progress);
        }
      }
      if (sounds.length > 0) {
        const { mostSamples } = await audioCompanion.ensure(model);
        if (!current(run)) return;
        clipCount = sounds.reduce((n, r) => n + Math.max(1, Math.ceil(r.samples / mostSamples)), 0);
        // The recordings' clips in order, each recording decoded when its
        // first clip is reached and dropped after its last.
        const clips = (async function* () {
          for (const recording of sounds) {
            for (const clip of clipsOf(mono(await decodeAudio(recording.file)), mostSamples)) {
              yield { recording, ...clip };
            }
          }
        })();
        // A clip's samples copied out of its recording, so the worker can
        // take their buffer, and prepared.
        const prepare = (clip) => (clip === null ? null : prepareAudio(clip.samples.slice()).then(() => clip));
        const next = async () => {
          const { value, done } = await clips.next();
          return done ? null : value;
        };
        // Each clip's samples go in while the one before is encoded.
        let prepared = prepare(await next());
        while (prepared !== null) {
          const clip = await prepared;
          if (!current(run)) return;
          const { reply } = await embedClip();
          prepared = prepare(await next());
          prepared?.catch(() => {});   // its failure is met when it is awaited
          const { vector, tokens: made } = await reply;
          if (!current(run)) return;
          built.add({ recording: clip.recording, start: clip.start, end: clip.end }, vector);
          tokens += made;
          const progress = `${built.size()} of ${kept + left()}`;
          documentFacts.textContent = `${progress} embedded`;
          onSearch?.('progress', progress);
        }
      }
      if (films.length > 0) {
        const { video: reads } = await companion.ensure(model);
        if (!current(run)) return;
        if (!reads) throw new Error('the image encoder loaded does not read video');
        const plans = [];
        for (const film of films) plans.push(await planVideo(film.width, film.height, film.duration));
        if (!current(run)) return;
        videoClipCount = plans.reduce((n, plan) => n + Math.ceil(plan.frames / plan.clipFrames), 0);
        // The videos' clips in order, each video opened when its first clip
        // is reached and released after its last, each clip's frames seeked
        // as it is reached.
        const clips = (async function* () {
          for (let v = 0; v < films.length; v++) {
            const opened = await openVideo(films[v].file);
            try {
              for (const clip of videoClipsOf(plans[v].frames, plans[v].clipFrames)) {
                const frames = [];
                for (let i = clip.first; i < clip.first + clip.count; i++) frames.push(await frameAt(opened.video, i));
                yield { video: films[v], frames, start: clip.start, end: clip.end };
              }
            } finally {
              opened.release();
            }
          }
        })();
        const prepare = (clip) => (clip === null ? null
          : prepareVideo(clip.video.width, clip.video.height, clip.frames).then(() => clip));
        const next = async () => {
          const { value, done } = await clips.next();
          return done ? null : value;
        };
        try {
          // Each clip's frames go in while the one before is encoded.
          let prepared = prepare(await next());
          while (prepared !== null) {
            const clip = await prepared;
            if (!current(run)) return;
            const { reply } = await embedClip(kVideoText, 'video');
            prepared = prepare(await next());
            prepared?.catch(() => {});   // its failure is met when it is awaited
            const { vector, tokens: made } = await reply;
            if (!current(run)) return;
            built.add({ video: clip.video, start: clip.start, end: clip.end }, vector);
            tokens += made;
            const progress = `${built.size()} of ${kept + left()}`;
            documentFacts.textContent = `${progress} embedded`;
            onSearch?.('progress', progress);
          }
        } finally {
          await clips.return();   // the video open, if the run ended within it, released
        }
      }
      ready = built;
      const seconds = (performance.now() - started) / 1000;
      const count = (n, one) => `${n} ${one}${n === 1 ? '' : 's'}`;
      const onlyText = pictures.length === 0 && sounds.length === 0 && films.length === 0;
      const counted = listed([kept > 0 || onlyText ? count(kept, 'passage') : null,
        pictures.length > 0 ? count(pictures.length, 'photo') : null,
        sounds.length > 0 ? count(clipCount, 'clip') : null,
        films.length > 0 ? count(videoClipCount, 'video clip') : null].filter(Boolean));
      documentFacts.textContent = `${counted} embedded · ${tokens} tokens · ${seconds.toFixed(2)} s`;
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
    if (querying) return;
    if (text === '') {
      error.textContent = 'Type a query to search.';
      queryInput.focus();
      return;
    }
    if (ready === null) {
      clearResults();
      await index();
      if (session.closed || ready === null) return;
    }
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
        h('li', {}, h('span', { className: 'score', text: score.toFixed(3) }), resultOf(passage))));
      queryFacts.textContent = `${tokens} tokens · ${milliseconds.toFixed(1)} ms · ${ranking.size()} ranked`;
      // Answered where it can be seen: the results may sit below the fold.
      results.firstElementChild?.scrollIntoView({ block: 'nearest' });
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
    if (indexing !== null || ready !== null) discard('Changed: search again to include it.');
  });

  const documentBox = h('div', { className: 'composer' },
    documentInput,
    h('div', { className: 'composer-row' }, documentFacts));

  // Search is the panel's one action; while the index builds, it stops it.
  const queryForm = h('form', { className: 'composer search-query', onsubmit: (event) => {
    event.preventDefault();
    if (indexing !== null) discard('Stopped; nothing kept.');
    else search();
  } },
    h('div', { className: 'composer-row' }, queryInput, queryButton));

  const about = `${embeddingLength} numbers a text, up to ${contextOffered} tokens` +
    (prompts === null ? ' · no task prompts are known for this model: texts are embedded as written' : '');
  settle();
  const panel = h('div', { className: 'conversation search-panel' },
    h('p', { className: 'chat-heading', text: model.measured ? model.name : `${model.name} · unmeasured` }),
    h('p', { className: 'message-facts', text: about }),
    documentBox, documents.element, photos?.element, recordings?.element, videos?.element, companionFacts, queryForm, error, results,
    queryFacts);
  documents.dropOnto(panel);
  photos?.dropOnto(panel);
  recordings?.dropOnto(panel);
  videos?.dropOnto(panel);
  return panel;
}

// A result's entry: a passage's text; a photo's thumbnail; or a clip's
// recording or video, its span and a player for that span.
function resultOf(entry) {
  if (typeof entry === 'string') return h('p', { text: entry });
  if (entry.from !== undefined) {
    return h('div', { className: 'result-clip' }, h('p', { text: entry.from }), h('p', { text: entry.text }));
  }
  if (entry.file !== undefined) return h('img', { className: 'result-photo', src: entry.url, alt: entry.file.name });
  if (entry.video !== undefined) {
    const { video, start, end } = entry;
    const span = `${clock(start)}–${clock(end)}`;
    return h('div', { className: 'result-clip' },
      h('p', { text: `${video.file.name} · ${span}` }),
      h('video', { className: 'result-video', controls: true, muted: true, playsinline: true, preload: 'metadata',
        src: `${video.url}#t=${start},${end}`, 'aria-label': `${video.file.name}, ${span}` }));
  }
  const { recording, start, end } = entry;
  const span = `${clock(start)}–${clock(end)}`;
  return h('div', { className: 'result-clip' },
    h('p', { text: `${recording.file.name} · ${span}` }),
    h('audio', { controls: true, preload: 'none', src: `${recording.url}#t=${start},${end}`,
      'aria-label': `${recording.file.name}, ${span}` }));
}
