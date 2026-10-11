// Axis L: how we download. Reads a model file's index before any weight is
// downloaded.
//
// Fetches the front of the file and hands it to the reader. While the reader
// answers that it needs more, fetches only the bytes it lacks and asks again.

import { formatBytes } from './units.js';

// Enough for the index of every model the harness targets in one fetch.
export const FIRST_FETCH_BYTES = 8 * 2 ** 20;
// An index larger than this is not a model this harness can hold.
export const MAX_INDEX_BYTES = 64 * 2 ** 20;

export class PreflightError extends Error {
  constructor(message) {
    super(message);
    this.name = 'PreflightError';
  }
}

// `fetchRange(start, end)` resolves with { bytes, totalSize } for that part of
// the file. `readIndex(bytes, totalSize)` resolves with the reader's answer
// for the file's first bytes. Resolves with the reader's verdict, the file's
// size, and `indexBytes`: how many of its first bytes the index needed, which
// a load sends again so the module reads the same index.
export async function preflight({ fetchRange, readIndex }) {
  let resident = new Uint8Array(0);
  let totalSize = Infinity;
  let wanted = FIRST_FETCH_BYTES;

  for (;;) {
    const piece = await fetchRange(resident.length, Math.min(wanted, totalSize));
    if (piece.bytes.byteLength === 0) throw new PreflightError('the server sent no bytes');
    totalSize = piece.totalSize;
    resident = concat(resident, new Uint8Array(piece.bytes));

    const answer = await readIndex(resident, totalSize);
    switch (answer.status) {
      case 'read':
        return { ...answer, totalSize, indexBytes: resident.length };
      case 'unreadable':
        throw new PreflightError(`not a readable GGUF file: ${answer.error}`);
      case 'need-bytes':
        if (answer.bytesNeeded > MAX_INDEX_BYTES) {
          throw new PreflightError(`the file's index is larger than ${formatBytes(MAX_INDEX_BYTES)}`);
        }
        wanted = Math.max(answer.bytesNeeded, resident.length * 2);
        break;
      default:
        throw new PreflightError(`unexpected answer from the reader: ${answer.status}`);
    }
  }
}

// A range source over a file already on this device, so a cached model is
// read from its copy rather than the network.
export const rangesOfFile = (file) => async (start, end) =>
  ({ bytes: await file.slice(start, end).arrayBuffer(), totalSize: file.size });

// A range source over a file read in part: only its first range, [0, end),
// which holds its header (catalog.js's `vision`), answering with the
// published file's size `totalSize`, so the reader validates the index as
// for the whole file. `fetchRange` is the network's or the cached ranges'
// own, whose first bytes are the first range's.
export const withinFirstRange = (fetchRange, end, totalSize) => async (start, stop) => {
  if (start >= end) throw new PreflightError("the file's index runs past its first range");
  const piece = await fetchRange(start, Math.min(stop, end));
  return { bytes: piece.bytes, totalSize };
};

function concat(a, b) {
  const joined = new Uint8Array(a.length + b.length);
  joined.set(a);
  joined.set(b, a.length);
  return joined;
}
