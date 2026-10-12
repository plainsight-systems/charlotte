#pragma once

#include <cstdint>

#include "core/media/image.h"

namespace bllm::media {

// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// A video made into what a vision encoder reads: frames sampled from it,
// each sized for the frame budget, and the frames grouped into clips the
// text encoder takes one step each. Pure: no GPU, no browser, no file. The
// page decodes the video and hands over each frame's pixels as an image's
// are handed over (runtime/video.h, web/video_input.js); this file decides
// which frames, at what size, and how many to a clip.
//
// It follows EmbeddingGemma 2's own video processor, as Google's
// processor_config.json configures it and Hugging Face's
// EmbeddingGemma2VideoProcessor computes it
// (docs/research/2026-10-11-embeddinggemma2-video.md), not llama.cpp's
// video helper, whose rate, text and frame budget all differ:
//
//   1. The frames: at kFramesPerSecond, 1 — frame i at time i s, for i from
//      0 to max(1, ⌊duration⌋) − 1 (frame_count). Google's rule picks native
//      frame min(total − 1, ⌊i × f⌋) of a video at f frames a second; at a
//      constant rate that is the frame shown at time i s, which the page
//      seeks to. Google caps a video at 32 frames, spread uniformly; a clip
//      here never reaches that cap (4 below).
//   2. The size: frame_grid, Google's get_aspect_ratio_preserving_size at
//      kFrameTokens, 140, the card's default: at most 140 × 9 = 1,260
//      patches; both sides scaled by √(1,260 × patch² / area), up or down,
//      in f64 as Python computes it, then floored to a multiple of patch ×
//      pooling, 48. A side floored to 0 takes 48 and the other ⌊its ratio⌋ ×
//      48, at most the budget's side, as Google's edge case does; both
//      floored to 0 is refused, Empty — a frame under 48 px both ways
//      cannot be one. A 1,280 × 720 frame and a 640 × 360 one both become
//      720 × 384, 45 × 24 patches, 120 tokens.
//      This is not image_grid's rule (media/image.h), which sizes as
//      llama.cpp does: the two agree when a frame is scaled down and differ
//      when it is scaled up, and EmbeddingGemma 2's processor is the
//      definition of a frame.
//   3. The fit: Google resizes a frame to its grid exactly, its aspect
//      changed by the flooring by under 48 px a side, with no padding:
//      frame_fit fills the grid, where image_fit letterboxes as llama.cpp
//      does for gemma4v. The resampler is the image path's, Pillow's
//      bicubic as llama.cpp adapts it (media/image.h, ImageResampler), so a
//      frame and an image are resampled alike; Google's torchvision
//      resampler differs from it as it does for images.
//   4. The patches: as an image's, (u / 255) × 2 − 1 (media/image.h).
//   5. The clip: the frames grouped in order, frames_per_clip to a clip, the
//      last clip what remains. A clip's step holds, for each frame,
//      `<|image>`, its tokens and `<image|>`, beside the text file's ends:
//      frames_per_clip is the most frames of T tokens that fit in `room` —
//      the step's context less its ends, 510 for EmbeddingGemma 2 —
//      ⌊room / (T + 2)⌋, and at most kClipFrames, 4. A 16:9 or square
//      frame's 120 or 121 tokens give 4, 4 s a clip at 1 fps; a 4:3 frame's
//      130 give 3. A clip's embedding is Google's computation for a video of
//      those frames: the whole video's, up to 32 frames, needs a step of
//      3,906 rows, which runtime/embedder.h names as another design.
//
//   - kClipFrames bounds the frame slots allocated at load (runtime/
//     video.h): 4 slots of 1,260 patches, 3.9 MB each. Without it a
//     panorama's few tokens a frame would make clips of a dozen frames and
//     slots to match; 4 is what a 16:9 frame fits, so the cap costs no
//     ordinary video anything.
//   - All arithmetic on integers but the scale, which is f64 as Python's
//     math.sqrt is; floored as int(math.floor(x)).
//
// What it costs: nothing per frame beyond the image path's resample
// (media/image.h): a few comparisons and one square root per video.
//
// Verification the implementation is held to (tests/media_video_test.cpp):
// frame_grid against Google's rule, computed in the test from its
// definition, over 1,280 × 720, 640 × 360, 1,920 × 1,080, 720 × 1,280,
// 1,024 × 1,024, 640 × 480, 3,840 × 2,160, 48 × 48, 47 × 47 (refused), and
// 4,000 × 60 and 60 × 4,000 (an edge case each); every grid at most 1,260
// patches; frame_fit filling its grid; frame_count over durations of 0,
// 0.5, 1, 4.99, 5, 300; frames_per_clip at 120, 121, 130, 140 and 10 tokens
// against a room of 510 (4, 4, 3, 3, 4).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.20   For "out" output values, prefer return values to output
//            parameters — the grid and the counts are returned.
//     I.4    Make interfaces precisely and strongly typed — a grid, not four
//            integers.
//     E.27   Use error codes systematically — ImageError's Empty.
//   C++ performance guidelines
//     MEM.9  Allocate at init — kClipFrames bounds the slots made at load.

inline constexpr std::uint32_t kFramesPerSecond = 1;
inline constexpr std::uint32_t kFrameTokens = 140;
inline constexpr std::uint32_t kClipFrames = 4;

// The frames a video of `duration` seconds gives: max(1, ⌊duration⌋) at 1 a
// second; 0 for a duration that is not positive and finite.
[[nodiscard]] std::uint32_t frame_count(double duration) noexcept;

// A frame's grid by Google's rule (2 above), or ImageError::Empty for one
// with no side at least patch × pooling once scaled. Preconditions: both
// sides at least 1 and at most kMaxSide.
[[nodiscard]] ImageError frame_grid(std::uint32_t width, std::uint32_t height, std::uint32_t patch,
                                    std::uint32_t pooling, std::uint32_t tokens, ImageGrid& out) noexcept;

// Where a frame lies in its grid: the whole grid, stretched (3 above).
[[nodiscard]] ImageFit frame_fit(const ImageGrid& grid) noexcept;

// The frames of `tokens` each a clip holds in `room` rows (5 above), at
// least 1 when one frame fits, else 0.
[[nodiscard]] std::uint32_t frames_per_clip(std::uint32_t tokens, std::uint32_t room) noexcept;

}  // namespace bllm::media
