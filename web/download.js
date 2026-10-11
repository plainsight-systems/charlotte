// Axis L: how we download. Streams a model file into the cache, hashing it
// as it arrives, and publishes it only when it is complete and matches.

import { Sha256 } from './sha256.js';
import { formatBytes } from './units.js';

export class DownloadError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DownloadError';
  }
}

// Where a model lives in the cache. A curated model is named by its hash, so
// the same file is cached once whichever URL served it; a pasted URL is
// named by the URL.
export function cacheKey(model) {
  if (model.sha256 !== undefined) return `sha256-${model.sha256}`;
  return `url-${new Sha256().update(new TextEncoder().encode(model.url)).hex().slice(0, 32)}`;
}

// Where a companion's ranges live in the cache: named by their hash, as a
// curated model is by its file's.
export const rangesCacheKey = (entry) => `ranges-${entry.rangesSha256}`;

// `openDownload(url, { signal })` resolves with { totalSize, chunks };
// `cache.writer(key)` with a writer. `onProgress(received, total)` follows the
// download; `total` is null when neither the list nor the server says.
export async function downloadModel({ model, openDownload, cache, onProgress, signal }) {
  const { totalSize, chunks } = await openDownload(model.url, { signal });
  if (model.sizeBytes !== undefined && totalSize !== null && totalSize !== model.sizeBytes) {
    throw new DownloadError(`the server's file is ${formatBytes(totalSize)}, ` +
                            `but the model list says ${formatBytes(model.sizeBytes)}`);
  }
  const expected = model.sizeBytes ?? totalSize;

  const writer = await cache.writer(cacheKey(model));
  try {
    const hash = new Sha256();
    let received = 0;
    for await (const chunk of chunks) {
      signal?.throwIfAborted();
      hash.update(chunk);
      await writer.write(chunk);
      received += chunk.length;
      onProgress?.(received, expected);
    }
    if (expected !== null && received !== expected) {
      throw new DownloadError(`the download ended after ${formatBytes(received)} of ${formatBytes(expected)}`);
    }
    const sha256 = hash.hex();
    if (model.sha256 !== undefined && sha256 !== model.sha256) {
      throw new DownloadError('the downloaded file does not match its published SHA-256');
    }
    await writer.commit({ name: model.name, url: model.url, sizeBytes: received, sha256 });
  } catch (error) {
    await writer.discard();
    throw error;
  }
}

// Streams a companion's ranges (catalog.js's `vision` or `audio`, `entry`)
// into one cache file,
// in order, back to back, hashing the bytes as they arrive, and publishes it
// only when every range is whole and their hash matches rangesSha256: a part
// of a file verified as a whole file is, against a pinned value.
// `openRange(url, begin, end, { signal })` resolves with { totalSize, chunks
// } for that range. `onProgress(received, total)` follows the download over
// the ranges' bytes. `name` is what the cache lists it as.
export async function downloadRanges({ name, entry, openRange, cache, onProgress, signal }) {
  const total = entry.ranges.reduce((sum, [begin, end]) => sum + (end - begin), 0);
  const writer = await cache.writer(rangesCacheKey(entry));
  try {
    const hash = new Sha256();
    let received = 0;
    for (const [begin, end] of entry.ranges) {
      const { totalSize, chunks } = await openRange(entry.url, begin, end, { signal });
      if (totalSize !== entry.sizeBytes) {
        throw new DownloadError(`the server's file is ${formatBytes(totalSize)}, ` +
                                `but the model list says ${formatBytes(entry.sizeBytes)}`);
      }
      let got = 0;
      for await (const chunk of chunks) {
        signal?.throwIfAborted();
        if (got + chunk.length > end - begin) throw new DownloadError('the server sent more than the range asked for');
        hash.update(chunk);
        await writer.write(chunk);
        got += chunk.length;
        received += chunk.length;
        onProgress?.(received, total);
      }
      if (got !== end - begin) {
        throw new DownloadError(`a range ended after ${formatBytes(got)} of ${formatBytes(end - begin)}`);
      }
    }
    const sha256 = hash.hex();
    if (sha256 !== entry.rangesSha256) {
      throw new DownloadError('the downloaded ranges do not match their published SHA-256');
    }
    await writer.commit({ name, url: entry.url, sizeBytes: received, ranges: entry.ranges, sha256 });
  } catch (error) {
    await writer.discard();
    throw error;
  }
}
