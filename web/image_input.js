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
