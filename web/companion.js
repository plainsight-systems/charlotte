// Axis H. A loaded model's companion file — the vision encoder EmbeddingGemma
// 2's images need (runtime/vision.h) — checked, downloaded and loaded the
// first time the page needs it, beside the model already loaded. The model
// controller's lifecycle (model_controller.js), for a file that is never
// chosen alone and is read in part.
//
// createCompanion({ client, cache, onChange }) returns
// { ensure(model) → Promise<loaded>, state, release() }:
//   - ensure(model) with the loaded model's catalog entry, whose `vision`
//     names the companion (catalog.js). Resolves once the companion is
//     loaded; at once if it already is. Each step changes `state`, which
//     onChange renders where the search panel shows it: { phase: 'checking'
//     | 'downloading' | 'loading' | 'loaded' | 'failed', received, total,
//     verdict, message }.
//   - Checking: its header from the first range — from the cache if it is
//     there, else the network (fetch.js's range fetch) — sent as
//     PREFLIGHT_COMPANION, the verdict naming each stage it cannot reach, as
//     a model's verdict does (stages.js): the projector type unsupported,
//     the plan past what the budget leaves, the text model not one it
//     serves.
//   - Downloading: each range fetched in order and streamed into one cache
//     file, its SHA-256 computed over the ranges' bytes in order and
//     checked against rangesSha256 before the file is listed (download.js,
//     opfs.js, sha256.js) — a part of a file verified as a whole file is,
//     against a pinned value. Cached under its own key,
//     `ranges-<rangesSha256>`, beside the model's.
//   - Loading: LOAD_COMPANION with the cache file's name and the ranges; the
//     worker streams the cache file in chunks and gives each chunk its
//     offset in the published file: a chunk at place p of the cache file
//     lies at range r's begin + (p − the bytes of ranges before r)
//     (load.js), never across a range's end.
//   - release() on a new model, as the model controller abandons a step:
//     work aborted, a late answer ignored. The runtime releases the
//     companion with the model it serves (runtime/vision.h).
//
// What it asks of the other files:
//   - download.js: downloading a list of ranges by range fetch into one
//     cache file, hashing as the bytes arrive, the whole file's path being
//     its one range; cacheKey for a companion's ranges.
//   - load.js: streaming a cache file of ranges with each chunk's published
//     offset.
//   - preflight.js: the prefix read from the first range, its total size
//     the published file's, so the reader validates the index as for the
//     whole file.
//   - protocol.js and worker.js: PREFLIGHT_COMPANION and LOAD_COMPANION,
//     their payloads the model's plus the ranges, the runtime's crossings
//     (runtime/vision.h).
//
// What it costs: one range fetch of the header, 58,656 bytes, on a first
// check; then the ranges, 226.4 MB, streamed once into the cache and hashed
// as they arrive; on each load the cache file read once in 16 MiB chunks, as
// a model's (load.js). Nothing per encode.
//
// Verification: tests/web/companion.test.js, with a fake client and cache:
// the phases in order; a cached companion loaded without a fetch; a
// mismatched hash refused and nothing listed; release() mid-download leaving
// nothing listed and ignoring the late answer; the chunk offsets across a
// gap, none crossing it.
