# EmbeddingGemma 2's video: what it is, what llama.cpp runs, what it costs

**Date:** 2026-10-11
**Method:** Google's `processor_config.json` and model card for
`google/embeddinggemma-2`; Hugging Face's `EmbeddingGemma2Processor`,
`EmbeddingGemma2VideoProcessor` and `EmbeddingGemma2Model` read from source
(`transformers/models/embedding_gemma2`); llama.cpp read from source at the
embeddings' pin, `a9d27ac693ff0a1d3cba449c912c2b891904a4c7`; the text model's
vocabulary read from the pinned GGUF; a frame's encode timed in the app's
Chromium on the target (Apple M3 Max, Metal). Costs are derived from shapes;
the rates used are measured ones, named.

## Why

Images and audio are in (`2026-10-11-embeddinggemma2-vision.md`,
`2026-10-11-embeddinggemma2-audio.md`). Video is the card's last modality,
and EmbeddingGemma 2 isn't multimodal here until it takes it. This note is
what the video path needs: what a video becomes before the text model reads
it, what llama.cpp does with video, and what fits in the context this
harness offers.

## What Google's processor does

The card says video is "sampled frames" through the vision encoder, 1 frame
a second by default, 140 soft tokens a frame by default, passed with no task
prompt. `processor_config.json` names the processor that does it,
`EmbeddingGemma2VideoProcessor`, with: `fps` 1, `max_frames` 32,
`overflow_strategy` "uniform", `max_soft_tokens` 140, `pooling_kernel_size`
3, `patch_size` 16, `add_timestamps` false, `do_normalize` false, `resample`
3 (bicubic). It is not Gemma 4's video processor, whose defaults differ (70
tokens, 32 frames sampled by count, timestamps on).

1. **Frames.** For a video of `duration` seconds at native rate `f`:
   `max(1, ⌊duration × 1⌋)` frames, frame i the native frame
   `min(total − 1, ⌊i × f⌋)`, so the frame at time i s. Over 32, the list is
   cut to 32 by `linspace(0, n − 1, 32)`.
2. **Size.** Each frame resized as an image is, by
   `get_aspect_ratio_preserving_size` with at most 140 × 9 = 1,260 patches:
   both sides scaled by √(1,260 × 256 / area) — up or down — and floored to
   a multiple of 48. A 1,280 × 720 frame and a 640 × 360 one both become
   720 × 384: 45 × 24 patches, 120 soft tokens. A square frame becomes
   528 × 528, 121 tokens; a 4:3 one 624 × 480, 130 tokens. Then bicubic
   with antialiasing (torchvision), x / 255.
3. **The text.** `<|video|>` in the text expands to, for each frame,
   `<|image>`, the frame's soft tokens, `<image|>`, frames back to back with
   nothing between them: no timestamps, no "Video:" label.
4. **The model.** `get_video_features` runs every frame through the same
   vision tower and `embed_vision` projection as an image, each frame on its
   own; the rows replace the `<|video|>` placeholders in order. The audio
   track is not read.

The text model's vocabulary holds the tokens: `<|image>` 255999,
`<image|>` 258882, `<|video|>` 258884 (the pinned GGUF's
`tokenizer.ggml.tokens`).

So a video, here, is a sequence of frames through the vision encoder
already built for images (`graph/vision.h`), at 140 tokens a frame where an
image takes 280. No new kernel, no new weights: the image companion's range
holds everything video needs.

## What llama.cpp does

llama.cpp's video support at the pin is a helper over its core library
(`mtmd-helper.cpp`, built with `MTMD_VIDEO`): ffmpeg decodes at 4 fps by
default (its `fps` filter), the helper emits the text "Video:" first and a
timestamp "[0m5.00s]" every 5 s, and each frame goes through the image
preprocessor at the image token budget. None of that is EmbeddingGemma 2's
processor: the rate, the text and the frame budget all differ.

What llama.cpp does compute as Google does is each frame's rows: the vision
encoder per frame, with `clip.use_gelu` set on a copy of the mmproj as the
image path's oracle sets it (`2026-10-11-embeddinggemma2-vision.md`). So the
oracle here is llama.cpp's encoder and text model, given the token sequence
and the frames Google's processor makes, built by the reference tool itself,
as the image reference builds its sequence.

## Frame size against the image path

Charlotte's image path sizes as llama.cpp does (`media/image.h`): a target
between the budget's least and most, sides rounded to 48. Google's rule
always fills toward the most. The two agree whenever a frame is scaled down
— any frame larger than 1,260 patches' area, 322,560 pixels, so 1,280 × 720
and up — and differ when it is scaled up: 640 × 360 becomes 720 × 384 by
Google's rule, 624 × 336 by llama.cpp's with a least of 70.

Video follows Google's rule, since EmbeddingGemma 2's processor is the
definition of what a frame is. The resampler stays the image path's,
llama.cpp's Pillow-compatible bicubic, so frames and images are resampled
alike; Google's torchvision resampler differs from it as it does for images.

The image path's own rule differs from Google's the same way for images
smaller than 2,520 patches' area: Google enlarges a 320 × 240 image to 280
tokens, llama.cpp's rule to 70. That is a finding about images, outside
this design.

## What fits: 512 rows a step

The embedder holds one step of at most 512 rows (`runtime/embedder.h`).
Google's default video, 32 frames of 120 tokens, is 32 × 122 + 2 = 3,906
rows: it needs the longer encode that design names as another. At 512, a
16:9 frame's 122 rows (its 120 and its two tokens) fit four times beside
the file's two ends: 4 × 122 + 2 = 490.

So a video here is cut into clips of k consecutive frames, k the most that
fit — ⌊(512 − 2) / (T + 2)⌋ for T tokens a frame: 4 at 16:9 or square, 3 at
4:3 — each clip one embedding, as a recording is cut into clips of 20 s. At
1 fps a clip is k seconds of video, and a query finds where in a video it is
answered. A clip's embedding is Google's computation for a video of those k
frames; it is not the whole video's embedding, which needs the longer
encode.

Fewer tokens a frame would make longer clips: at 70, a 16:9 frame is
528 × 288, 66 tokens, seven to a clip. Measured, it saves little (below),
and it departs from the card's default, so frames take 140.

## Decoding video in a browser

The page decodes: a `<video>` element seeked to i s for frame i, and
`createImageBitmap(video)` for its pixels, which Chromium and Safari both
support; a worker has no media element. The bitmap is transferred to the
worker, which prepares it as an image's bitmap is prepared
(`web/image_input.js`). The browser decodes and converts YUV to RGB with its
own code, not ffmpeg's, so its frames are near the reference's, not equal,
as a resampled recording's samples are. The duration is the element's.

For a video at a constant rate, the frame shown at time i s is native frame
⌊i × f⌋, the one Google's rule picks.

## What a clip costs, measured and derived

Timed in the app's Chromium on the target, through the image path, which is
the frame's path: images of 720 × 384, 120 tokens each, a vision step of
1,080 patches and a text step of 124 rows. Eight indexed in 1.02 s, so
about 0.13 s each with its text step, about 0.10 s for the vision step
alone. At 70 tokens, 528 × 288, eight took 0.73 s: fixed costs a step
dominate, so halving the tokens saves under a third.

- A 4-frame clip: four vision steps, about 0.40 s; one text step of 490
  rows, about 45 ms (`2026-10-11-chrome-encode-benchmark.md`). About 0.45 s
  for 4 s of video.
- A 5 minute video: 75 clips, about 34 s, beside the page's decode and
  seeks.
- The page: a seek and a bitmap a second of video; one transfer a frame to
  the worker; the frame's resample in the module, as an image's.

## What follows for Charlotte

- Video is a medium of its own, `<|video|>`, read by the vision encoder the
  image companion already loads: no new download.
- A clip is k frames, each `<|image>`, its rows, `<image|>`, with the ends
  around them and no text, as Google's processor writes it.
- Frames are sized by Google's rule at 140 tokens and resampled as images
  are.
- A text step then holds several media spans, one a frame, and each frame's
  rows are written by its own vision step before the next frame's
  overwrites them: the runtime's medium becomes a list of segments.
- The oracle is llama.cpp's encoder per frame and its text model over
  Google's sequence, on frames prepared as Charlotte prepares them.
