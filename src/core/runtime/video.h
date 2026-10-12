#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/kernels/program.h"
#include "core/media/image.h"
#include "core/media/video.h"
#include "core/runtime/media_stage.h"

namespace bllm::runtime {

// Axis M (encode): changes with a new stage in the encode path.
//
// The video stage of an encode: a clip of a video — up to 4 frames, 4 s at
// 1 fps — becomes rows of the text model's width, one run of rows a frame,
// and those runs stand in the text step where the clip's placeholder stood,
// so one embedding represents the text and the clip together, comparable
// with any other (docs/research/2026-10-11-embeddinggemma2-video.md). A
// video's frames are images to the vision encoder: this stage runs the
// image stage's encoder (runtime/vision.h) once a frame, and differs from
// it in what it prepares and how many runs of rows it hands the embedder.
//
// An encode with a clip, end to end:
//   1. Plan (once a video): the page reads the video's size and duration and
//      asks plan() for its frames — frame_count, 1 a second — the grid every
//      frame takes, frame_grid at 140 tokens, and the frames a clip holds,
//      frames_per_clip in the room the embedder leaves (media/video.h). The
//      page cuts the video into clips of that many consecutive frames.
//   2. Prepare (before the encode, and while the clip before it runs): for
//      each clip, begin() with its frame count, then each frame's pixels a
//      band at a time into its slot, as an image's (web/video_input.js);
//      each slot resamples to the frame grid, filled, not letterboxed
//      (media/video.h's frame_fit), and cuts the patches.
//   3. The text, UTF-8, with `<|video|>` exactly once — the card's
//      placeholder for a video, written alone, as the card says video takes
//      no task prompt. The embedder replaces it with, for each frame,
//      `<|image>`, the frame's T rows, `<image|>`, frames back to back, as
//      EmbeddingGemma 2's processor writes a video: no timestamps, no text
//      between frames. 4 frames of 120 rows and the file's ends are 490
//      rows, within the context of 512.
//   4. The vision steps, one a frame, in order: each the image stage's step
//      over that frame's patches — position 0, the frame's patches,
//      `columns` the grid's — its patches written into the vision program's
//      `pixels` in one write, its T rows of `soft_tokens` copied after its
//      pass into the text program's `hidden` at the frame's first row, so
//      the next frame's step may overwrite `soft_tokens`
//      (kernels/program.h's copy after a pass). Two in flight at most, the
//      program's limit: the embedder submits the next frame as each
//      completes (runtime/media_stage.h).
//   5. The text step, once the last frame is submitted, on the queue behind
//      it: gather writes every row but the frames', whose identifiers are
//      the placeholder's (kernels/gather/gather.h); then the text encoder's
//      step as for an image's rows.
//   6. The vector read back, as embedder.h reads it.
//   - Refusals before the GPU runs, each named, start() returning at once:
//     MediaNotReady, the clip's frames not all prepared; Placeholder, as
//     media_stage.h states; TooManyTokens, the text's tokens and the
//     frames' rows past the context. And the clip's own as it is prepared:
//     a frame size Empty or TooLarge (media/image.h); TooManyFrames, a
//     clip of more frames than plan() allowed or none; BadBand.
//
// The stage's state:
//   - Two sets of kClipFrames slots, each slot an ImageResampler and the
//     patches of one frame, 1,260 × 3 P² floats, 3.9 MB: 8 slots, 31 MB of
//     the module's heap, allocated at load (MEM.9). begin() fills the set
//     the last encode did not use, so the page prepares clip c + 1 while
//     clip c's frames are submitted one by one from the other set: a set
//     is in use from the start() that takes it until that encode's last
//     frame is submitted, and begin() refuses Busy for a set still in use,
//     which the page's order — prepare, embed, prepare the next once the
//     embed is sent — never meets.
// Optimization (practice): staging double-buffered, so the CPU's resample
// of the next clip overlaps the GPU's passes over this one, where one set
// would make the page wait for an encode before preparing (GPU.7, MEM.4).
//   - The vision encoder's program, shared with the image stage: both hold
//     it, std::shared_ptr, since either may be attached alone and the
//     embedder runs one encode at a time, so their steps never interleave
//     (F.27, R.20). Its plan's `pixels` holds 2,520 patches, the image
//     budget's most; a frame's 1,260 or fewer fit it.
//   - Its own link, `soft_tokens` into the text program's `hidden`, as the
//     image stage's: two links over the same two buffers, so neither stage
//     borrows the other's.
//
// The companion: none of its own. The image companion's load (runtime/
// vision.h) attaches both stages, image and video, from the one program,
// when the encoder it loads names video tokens (arch/media_encoder.h), and
// releases both together. The video's `<|video|>`, `<|image>` and
// `<image|>` resolved from the text model's vocabulary by text at load.
//
// What it costs, counted, for EmbeddingGemma 2 at Q8_0 and a 16:9 clip of 4
// frames, each 720 × 384, 1,080 patches, 120 rows:
//   - Preparing, on the CPU: each frame's resample, as an image's
//     (media/image.h): a 1,280 × 720 frame to 720 × 384, 9 taps an axis,
//     1.4 × 10⁷ integer multiply-adds horizontally and 7.5 × 10⁶
//     vertically, a few milliseconds; overlapping the last clip's GPU
//     work.
//   - Crossings: the frames' bands, 3.7 MB a 1,280 × 720 frame, one call a
//     band; begin() once a clip; plan() once a video.
//   - The vision steps: 4 × 1,080 patches, about 0.10 s each measured in the
//     app's Chromium on the target through the image path, 0.40 s.
//   - The copies: 4 × 120 × 512 × 4 bytes, 983 KB, after each pass.
//   - The text step: 490 rows, about 45 ms (runtime/embedder.h).
//   - About 0.45 s for 4 s of video: 75 clips, a 5 minute video, in about
//     34 s.
//   - Submissions: 5 an encode, against 2 for an image; each frame's
//     completion is a callback that submits the next, a CPU round trip
//     the second frame in flight hides.
//
// What it asks of the other contracts:
//   - media/video.h: frame_count, frame_grid, frame_fit, frames_per_clip.
//   - media/image.h: the resampler taking its fit.
//   - runtime/media_stage.h: segments, each run with its copy out.
//   - kernels/program.h: a run's copies after its pass.
//   - arch/media_encoder.h: MediaKind::Video; an image encoder's video
//     tokens.
//   - runtime/embedder.h: three kinds; the placeholder replaced by every
//     segment's rows between its kind's tokens.
//   - The boundary, contract 11 (src/wasm/bindings.cpp, web/worker.js):
//       bllm_video_plan(request, width, height, duration) → the frames, the
//         grid's tokens and size, and the frames a clip holds, or the
//         refusal; no state changes;
//       bllm_video_begin(request, width, height, frames) → the clip's slot
//         set, or the refusal;
//       bllm_video_rows(request, frame, first_row, pixels, length) → whether
//         the clip is ready, as bllm_image_rows answers for an image;
//       bllm_embed's `media` 3, the video stage's.
//
// Verification the implementation is held to:
//   - Against the reference: tools/reference_vision embeds, through
//     llama.cpp at the embeddings' pin on the mmproj with clip.use_gelu set,
//     on its CPU and on Metal, clips of computed frames
//     (tests/fixtures/video/scenes.h: a red ball crossing, a blue square
//     growing, green stripes scrolling, a black line sweeping, each 4 s at
//     1,280 × 720) — each frame sized by Google's rule and resampled by
//     llama.cpp's resampler, encoded by llama.cpp's vision encoder at that
//     size, the token sequence built as EmbeddingGemma 2's processor builds
//     it — alone and one with a caption, and four text queries. Each of
//     Charlotte's embeddings within llama.cpp's own backends' spread
//     (tests/gpu/video_embed_test.cpp), its tokens and rows llama.cpp's.
//     The check discriminates: the frames letterboxed as an image is, or
//     given in reverse order, fall outside.
//   - Each query ranks its own scene's clip first among the four.
//   - On the GPU: a clip embedded twice gives the same bits, and the same
//     when the next clip is prepared during its encode; an image's encode
//     and a recording's around a clip's give their own bits; each refusal
//     named.
//   - In the app's Chromium, the page: four public-domain NASA videos, each
//     cut to a few seconds and committed with its source
//     (tests/fixtures/video/SOURCES.md); each frame decoded by the browser
//     near, not equal to, the reference's ffmpeg decode; each video's query
//     ranking its own video's clip first among the videos, recordings,
//     photos and paragraphs; each score llama.cpp's on the ffmpeg frames to
//     two places, the decoders' difference the third.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.27   Use a shared_ptr<T> to share ownership — the vision program,
//            the image and video stages'.
//     R.20   Use unique_ptr or shared_ptr to represent ownership.
//     C.121  If a base class is used as an interface, make it a pure
//            abstract class — MediaStage.
//     E.27   Use error codes systematically — each refusal named.
//     F.24   Use a span to designate a half-open sequence — a band's pixels.
//   C++ performance guidelines
//     GPU.7  Pipeline CPU and GPU work — the next clip prepared during a
//            pass; two frames in flight.
//     GPU.1  Keep data on the device — each frame's rows copied GPU to GPU.
//     WASM.2 Batch work across the JS boundary — a band a crossing.
//     MEM.9  Allocate at init — the slots at load.

// What a video becomes: its frames, the grid each takes, and how many a clip
// holds.
struct VideoPlan {
    std::uint32_t frames;
    media::ImageGrid grid;
    std::uint32_t clip_frames;
};

enum class VideoError {
    Ok,
    Image,           // the frame size's, in `image`
    TooManyFrames,   // a clip of more frames than plan() allows, or none
    Busy,            // the slot set begin() would fill is still in use
    BadFrame,        // a band for a frame past the clip's
};

struct VideoResult {
    VideoError error = VideoError::Ok;
    media::ImageError image = media::ImageError::Ok;
};

// One frame's staging: its resampler and its patches.
struct FrameSlot {
    media::ImageResampler resampler;
    std::vector<float> patches;   // 1,260 × 3 P² floats, at load
    bool ready = false;
};

// The video encoder: the vision encoder's program, run once a frame.
class Video final : public MediaStage {
public:
    // Preconditions: `program` was built from the vision encoder's graph,
    // with no readback, and is the one the image stage holds; `link` copies
    // its `soft_tokens` into the text program's `hidden`; `pixels` is its
    // plan's `pixels`; `patch` and `pooling` the encoder's; `room` the
    // embedder's context less the file's ends.
    Video(std::shared_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
          kernels::Binding pixels, std::uint32_t patch, std::uint32_t pooling, std::uint32_t room);

    // What a video of `width` × `height` and `duration` seconds becomes. Pure.
    [[nodiscard]] VideoResult plan(std::uint32_t width, std::uint32_t height, double duration,
                                   VideoPlan& out) const noexcept;

    // Starts a clip of `frames` frames of `width` × `height` in the free slot
    // set, discarding what that set held.
    [[nodiscard]] VideoResult begin(std::uint32_t width, std::uint32_t height, std::uint32_t frames);

    // The next band of frame `frame`'s pixels; the last band writes its
    // patches.
    [[nodiscard]] VideoResult rows(std::uint32_t frame, std::uint32_t first_row,
                                   std::span<const std::uint8_t> rgba);

    // One segment a frame, each the grid's tokens, once every frame is
    // ready; else empty.
    [[nodiscard]] std::span<const std::uint32_t> ready_segments() const noexcept override;

    // Submits frame `segment`'s vision step, its rows copied out after the
    // pass by `out`; the clip's set released once its last frame is
    // submitted.
    void run(std::uint32_t segment, const SegmentCopy& out, kernels::StepCallback done, void* userdata) override;

    ~Video() override;
    Video(const Video&) = delete;
    Video& operator=(const Video&) = delete;

private:
    std::shared_ptr<kernels::Program> program_;
    std::unique_ptr<kernels::ProgramLink> link_;
    kernels::Binding pixels_;
    std::array<std::array<FrameSlot, media::kClipFrames>, 2> sets_;
    std::array<std::uint32_t, media::kClipFrames> segments_{};
    media::ImageGrid grid_{};
    std::uint32_t frames_ = 0;
    std::uint32_t filling_ = 0;     // the set begin() filled last
    std::array<bool, 2> in_use_{};
    std::uint32_t patch_;
    std::uint32_t pooling_;
    std::uint32_t room_;
};

}  // namespace bllm::runtime
