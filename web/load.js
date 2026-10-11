// Axis L. Streams a cached model file through the module, in order, one
// chunk at a time. It runs in the worker: the worker reads the cache itself,
// straight into the module's chunk buffer, so a chunk is never held by the
// page, never crosses between page and worker, and is never copied into the
// heap after being read.
//
// A load crosses into the module in three kinds of call — begin and finish
// once each, and one per chunk: ceil(size / LOAD_CHUNK_BYTES) + 2 crossings
// in all, 46 for Gemma 3's 722 MB at 16 MiB (WASM.2: a phase per crossing):
//
//   1. begin: the index prefix preflight read, the file's size, and the
//      duplicates confirmDuplicates confirmed (duplicates.js). The module
//      plans the routes and creates the buffers, and answers once the device
//      has confirmed it holds them, or names what it could not hold.
//   2. a chunk, LOAD_CHUNK_BYTES at a time, in order: read from the cache
//      directly into the chunk buffer the module allocated once at begin,
//      through a view over the heap made afresh for each chunk and never
//      kept, since growing the heap detaches it (WASM.1). The module answers
//      when it may take the next — after the queue has finished the chunk
//      before (src/core/residency/upload.h).
//   3. finish: answered once every write has completed, or with the failure
//      that stopped them.
//
// The page makes one request for the whole load and is told its progress;
// it cancels by naming that request (worker.js).
//
// A diagnostic build then streams the file a second time, in the same
// chunks, through a check that compares every byte upload wrote with what
// the device holds (src/core/residency/upload_check.h); the same count of
// crossings again. streamFile runs both passes; a pass whose begin answers
// { skipped: true }, as a build without the check does, streams nothing.
//
// Optimization (browser): the worker reads the cache through a
// FileSystemSyncAccessHandle, which only a worker may open, into the module's
// own memory: the page's main thread reads nothing, and a copy into the heap
// and two messages a chunk between page and worker are gone. On an Apple M3
// Max in the desktop app's Chromium 152, warm, Qwen3 0.6B's 382 MB read in about 24 ms this way,
// against 125 ms by blob reads on the page one at a time. The load itself
// gained less, 257 to 241 ms median of five: the page's reads had overlapped
// the GPU process working through the previous chunk's writes, and those
// writes, not the reads, bound the load.
//
// Guidelines (the C++ performance guidelines; their WASM and GPU entries
// govern the page as much as the module):
//   WASM.1 Size linear memory to the real high-water mark; never hold a view
//          across a call that can grow the heap — one chunk buffer, a fresh
//          view per chunk.
//   WASM.2 Batch work across the JS boundary — ceil(size / chunk) + 2
//          crossings, one chunk per crossing, by pointer and length.
//   WASM.9 Stream in bounded chunks — one chunk is held, in the heap.
//   GDSA.6 Count the bytes each stage moves — the read now writes each byte
//          once, into the buffer the module reads it from.

export const LOAD_CHUNK_BYTES = 16 * 2 ** 20;

// The chunks a stored file of `size` bytes is streamed in: { stored,
// published, length }, where `stored` is the chunk's place in the stored
// file and `published` its offset in the file it was downloaded from. A file
// stored whole has none of `ranges`, and the two are equal. A file stored as
// ranges of the published one ([begin, end) pairs in order, catalog.js's
// `vision`) holds them back to back: a chunk at place p lies at range r's
// begin + (p − the bytes of the ranges before r), and never crosses a
// range's end. Throws when the ranges do not add up to `size`.
export function* chunksOf({ size, chunkBytes, ranges }) {
  const spans = ranges ?? [[0, size]];
  const total = spans.reduce((sum, [begin, end]) => sum + (end - begin), 0);
  if (total !== size) throw new Error(`the stored file is ${size} bytes, but its ranges hold ${total}`);
  let stored = 0;
  for (const [begin, end] of spans) {
    for (let published = begin; published < end; published += chunkBytes) {
      const length = Math.min(chunkBytes, end - published);
      yield { stored, published, length };
      stored += length;
    }
  }
}

// Streams a stored file of `size` bytes through one pass: `begin({ maxChunk
// })` starts it, and may answer { skipped: true }, when nothing is streamed
// and this resolves null; `read({ offset, length })` puts that part of the
// stored file where the runtime takes it from; `send({ offset, length })`,
// the offset the chunk's in the published file (chunksOf), resolves when the
// runtime has taken it; `finish()` resolves with the pass's answer. A pass
// the caller aborts is still finished, so the runtime settles it; one the
// runtime refuses is already settled there.
export async function streamFile({ size, ranges, begin, read, send, finish, onProgress, signal,
  chunkBytes = LOAD_CHUNK_BYTES }) {
  const chunks = [...chunksOf({ size, chunkBytes, ranges })];
  const started = await begin({ maxChunk: chunkBytes });
  if (started?.skipped) return null;
  try {
    for (const { stored, published, length } of chunks) {
      signal?.throwIfAborted();
      await read({ offset: stored, length });
      await send({ offset: published, length });
      onProgress?.(stored + length, size);
    }
  } catch (error) {
    if (signal?.aborted) await finish().catch(() => {});
    throw error;
  }
  return finish();
}
