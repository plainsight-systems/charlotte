// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// A photo decoded in the worker and handed to the module a band of rows at a
// time (runtime/vision.h, media/image.h). The page's half of an image's
// preparation: the browser decodes, the module resamples.
//
//   - decode(blob): createImageBitmap(blob, { colorSpaceConversion: 'none',
//     premultiplyAlpha: 'none', imageOrientation: 'from-image' }) — no
//     colour management, as stb_image, llama.cpp's decoder, applies none;
//     the file's orientation applied, so a photo is embedded as it is seen,
//     which stb_image would not do. Refused, by name, where the browser
//     cannot decode the file (a format it lacks) or a side is over 16,384
//     (media/image.h's kMaxSide).
//   - bands(bitmap): the bitmap drawn once onto an OffscreenCanvas of its
//     size, then getImageData a band at a time: RGBA, rows of 4 × width
//     bytes, each band ⌊4 MiB / (4 × width)⌋ rows, at least one, in order
//     from row 0. A canvas keeps its pixels premultiplied, so a pixel that
//     is not opaque comes back rounded; a photo is opaque, and alpha is
//     dropped by the module in any case.
//   - Each band crosses into the module once, copied into its memory and
//     handed to bllm_image_rows (wasm_runtime.js); the band's ImageData is
//     released after. The bitmap and canvas are closed once the last band
//     is in.
//
// What it costs, for a W × H photo: the browser's decode, its own; one draw;
// ceil(4 W H / 4 MiB) bands, 12 for 4,032 × 3,024, each one getImageData,
// one copy into the module's memory and one call (WASM.2: the crossing
// carries a band's work, the resample of its rows). The canvas holds the
// photo, 4 W H bytes, 49 MB at 12 megapixels, outside the module's heap,
// for as long as the bands take; the module never holds more than a band
// (WASM.9).
//
// Verification: tests/web/image_input.test.js — band heights for widths of
// 1, 4,032 and 16,384; the bands covering every row once, in order. In the
// app's Chromium, a PNG fixture's bands equal its pixels as Pillow reads
// them, and a JPEG with an orientation tag arrives turned.

// The largest side an image may have (media/image.h's kMaxSide).
export const MAX_SIDE = 16384;
// The bytes of one band, at most, where a row fits.
export const BAND_BYTES = 4 * 2 ** 20;

export class ImageInputError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ImageInputError';
  }
}

// The rows of each band of a `width` × `height` image, in order: { first,
// rows }, each ⌊BAND_BYTES / (4 × width)⌋ rows, at least one, the last what
// remains.
export function bandsOf(width, height) {
  const rows = Math.max(1, Math.floor(BAND_BYTES / (4 * width)));
  const bands = [];
  for (let first = 0; first < height; first += rows) bands.push({ first, rows: Math.min(rows, height - first) });
  return bands;
}

// The photo in `blob`, decoded with no colour management and its
// orientation applied: { bitmap, width, height }, or a refusal, named.
export async function decode(blob) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob,
      { colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image' });
  } catch {
    throw new ImageInputError('the browser cannot decode this file as an image');
  }
  const { width, height } = bitmap;
  if (width > MAX_SIDE || height > MAX_SIDE) {
    bitmap.close();
    throw new ImageInputError(`a side of the image is over ${MAX_SIDE.toLocaleString('en-US')} pixels`);
  }
  return { bitmap, width, height };
}

// The bitmap's RGBA rows, a band at a time: { first, rows, pixels }, the
// pixels a Uint8ClampedArray of 4 × width × rows bytes. The bitmap is closed
// once the last band is read, or the reading stops.
export function* bands(bitmap) {
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  try {
    context.drawImage(bitmap, 0, 0);
    for (const { first, rows } of bandsOf(width, height)) {
      yield { first, rows, pixels: context.getImageData(0, first, width, rows).data };
    }
  } finally {
    bitmap.close();
    canvas.width = 0;
    canvas.height = 0;
  }
}
