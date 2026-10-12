// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// A video decoded on the page, a frame a second, and cut into clips the
// module embeds one at a time (runtime/video.h, media/video.h). The page's
// half of a clip's preparation: the browser decodes and converts each
// frame to RGB, the module resamples it.
//
//   - open(file): a <video> element, muted, preload "auto", its src an
//     object URL of the file, awaited to `loadedmetadata`: its videoWidth,
//     videoHeight and duration. On the page, not in the worker: a worker
//     has no media element. Refused, by name, where the browser cannot
//     play the file, where it has no picture, or where it runs past 10
//     minutes — 600 frames, rather than an indexing that takes minutes the
//     page cannot show.
//   - plan: the module's, from the size and duration (bllm_video_plan,
//     runtime/video.h): the frames, 1 a second, and how many a clip holds,
//     k — 4 for a 16:9 video. The page asks once a video; the rule is the
//     module's alone (media/video.h), so nothing here recomputes it.
//   - clipsOf(frames, k): the frames in order, k to a clip, the last clip
//     what remains; clip c the frames from c k, its span c k to
//     min((c + 1) k, frames) seconds.
//   - frameAt(video, i): the element seeked to i s, awaited to `seeked`,
//     then createImageBitmap(video, { colorSpaceConversion: 'none' }): the
//     frame shown at time i s, which at a constant rate is native frame
//     ⌊i × rate⌋, the one EmbeddingGemma 2's processor picks
//     (docs/research/2026-10-11-embeddinggemma2-video.md). The browser's
//     decoder and its YUV-to-RGB conversion are its own, not ffmpeg's, so a
//     frame is near the reference's, not equal. Frames are seeked in order,
//     each once.
//   - A clip crosses to the worker as one PREPARE_VIDEO, its k bitmaps
//     transferred, not copied; the worker begins the clip in the module
//     (bllm_video_begin) and hands each bitmap over in bands as a photo's
//     (image_input.js's bands; bllm_video_rows), closing each bitmap once
//     its last band is in.
//   - The element and its URL are released once the video's last clip is
//     sent.
//
// What it costs, for a video of n seconds: n seeks and n bitmaps, the
// browser's decode of the frames around each seek; one message a clip; each
// frame's bands, 4 W H bytes, 3.7 MB at 1,280 × 720, drawn, read back by
// getImageData and copied once into the module's memory — three copies of
// the frame — each band crossing as a photo's: _malloc, the copy, the rows
// call and _free, three calls into the module and a reply (WASM.2). The
// copies are the cost; the calls around them are microseconds. The page
// holds a clip's k bitmaps at most, 15 MB at 1,280 × 720, between their
// decode and their transfer.
//
// Verification: tests/web/video_input.test.js — clipsOf at 1, 3, 4, 5 and
// 300 frames with k of 3 and 4, covering every frame once, in order, each
// clip at most k, its span its frames' seconds. In the app's Chromium, each
// fixture video's frames (tests/fixtures/video/) within a few units a
// channel of the reference's ffmpeg frames, and its clips the reference's.
