#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace bllm::media {

// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// An image made into the rows a vision encoder reads: its size chosen for
// the token budget, its pixels resampled to that size, and the result cut
// into patches. What tokenization is to a text, this is to an image: the
// encoder's input, prepared on the CPU, before anything reaches the GPU.
// Pure: no GPU, no browser, no file. The page decodes the image; the module
// is handed its pixels, a band of rows at a time (runtime/vision.h).
//
// It matches llama.cpp's preparation of an image for `gemma4v`, at the
// embeddings' pin (runtime/embedder.h), byte for byte where llama.cpp works
// in bytes, so the vision encoder's oracle sees the same input Charlotte's
// does (docs/research/2026-10-11-embeddinggemma2-vision.md):
//
//   1. The size: image_grid, llama.cpp's calc_size_preserved_ratio
//      (tools/mtmd/mtmd-image.cpp) with its arithmetic, f32 where it uses f32
//      — a side's fraction x / a rounded with std::round, the scale's square
//      root in f32 — for an alignment a of patch × pooling, 48 for
//      EmbeddingGemma 2, and pixel limits of tokens × a²: each side rounded
//      to a multiple of a, at least a; if the area is over the budget's most,
//      both sides scaled by 1 / √(area / most) and floored to a multiple of a,
//      at least a; if under its least, scaled up and ceiled. The budget is
//      the model's policy, the card's default of 280 tokens at most — 2,520
//      patches — and llama.cpp's 70 at least (policy/policy.h). The rule
//      does not keep every image within the budget: with one side floored to
//      a, the least it may be, the other is as long as the area allows, and
//      past an aspect of the budget's most to 1 — 280 to 1 — the grid holds
//      more tokens than the budget. llama.cpp would run such a grid; here it
//      is refused, OverBudget, before a row is read, since the step's
//      buffers hold the budget's most patches and no more (SL.con.3). No
//      photograph comes near it: a 10 to 1 panorama of 4,000 × 400 is 260
//      tokens.
//   2. The fit: llama.cpp resizes to the grid with its padding style for
//      gemma4v, PAD_CEIL, the default — the image is not stretched to the
//      grid but fitted inside it, its aspect kept: scale s = min(width /
//      W, height / H) in f32, the fitted size min(ceil(W s), width) ×
//      min(ceil(H s), height), placed at offset ((width − w) / 2, (height −
//      h) / 2), integer division, the rest of the grid black. Rounding each
//      side to 48 changes the aspect a little, so a photo gains a bar of a
//      few black pixels on one side or two.
//   3. The pixels: Pillow's separable resampler as llama.cpp adapts it
//      (resize_pillow), bicubic with a = −0.5. For each axis, once an image,
//      each output pixel's taps: its centre (o + 0.5) × s in source pixels, s
//      = source / target, the filter widened by s when shrinking, support 2s,
//      so a shrink filters rather than aliases; the taps' weights computed in
//      f64, normalized to sum to 1, then held as 22-bit fixed point. A
//      horizontal pass over every source row into an intermediate of target
//      width, each sum rounded (+2²¹) and shifted, then clipped to a byte;
//      then a vertical pass over the intermediate, the same way. Integer
//      arithmetic once the weights are fixed, and the weights' f64 is IEEE
//      in the module as natively (no fast-math), so equal pixels give equal
//      bytes on every machine.
//   4. The patches: the grid, with the resampled image placed in it, cut
//      into patch × patch squares,
//      row-major — patch (px, py) is row py × columns + px —
//      each a row of 3 × patch² floats, channel, then row, then column
//      within the patch: element c × patch² + ky × patch + kx. That is the
//      layout of the patch embedding's weight, F32 of shape (16, 16, 3, 768)
//      with kx varying fastest, so the convolution is a matrix product over
//      these rows (graph/vision.h). Each byte u becomes (u / 255) × 2 − 1,
//      in f32: llama.cpp's x / 255 at load and its graph's
//      ggml_scale_bias(x, 2, −1), computed here once, on the CPU, as the
//      rows are written.
//
//   - Pixels arrive as RGBA, 4 bytes a pixel, rows top to bottom, as a
//     canvas gives them; alpha is dropped, not composited, as stb_image's
//     three-channel load drops it. The page decodes without colour-space
//     conversion and with the file's orientation applied
//     (web/image_input.js): a photo is embedded the way it is seen. stb_image,
//     llama.cpp's decoder, ignores orientation; the reference images are PNGs
//     with none, so the two agree there.
//   - The source arrives in bands of whole rows, in order, from row 0
//     (accept): each band is filtered horizontally as it arrives and
//     released, so a 48-megapixel photo never lies whole in the module's
//     memory (WASM.9, WASM.1). The horizontal pass needs nothing but its own
//     row, and the vertical pass runs once every row is in.
//   - Refusals, named, before anything is read: an image with no pixels; a
//     side over kMaxSide, 16,384, the largest a canvas promises; a grid past
//     the budget, OverBudget, above; a band out
//     of order, of the wrong width, or past the last row; patches asked for
//     before the last row is in.
//
// What it costs, counted, for a source of W × H to a target of w × h, taps
// k_x = 2 ceil(2 W / w) + 1 a side when shrinking, k_y the same for H / h:
//   - Weights: w k_x + h k_y, in f64 then fixed point, once an image.
//   - Horizontal pass: w × H × k_x × 3 integer multiply-adds; vertical:
//     w × h × k_y × 3. A 4,032 × 3,024 photo to 912 × 672, k = 19: 1.57 ×
//     10⁸ and 3.5 × 10⁷, about 0.1 to 0.2 s in the module by estimate, to be
//     measured. The horizontal pass over every source row is most of it,
//     and it is Pillow's and llama.cpp's own order; filtering vertically
//     first costs the same, W × h × k_y × 3, here 1.55 × 10⁸.
//   - Memory, allocated for each image and released after it (MEM.9's
//     steady state does not apply: an image is a new input, not a step):
//     the weights, the intermediate w × H × 3 bytes — 8.3 MB for the photo
//     above — the output w × h × 3, and the band's own pixels, which the
//     page bounds at 4 MiB (runtime/vision.h). The patches are written
//     straight into the step's input, 4 × 3 × patch² bytes a patch, 7.7 MB
//     at 2,520.
//   - Calls: one a band; nothing crosses to the browser.
// Optimization (practice): the taps' weights computed once an image and
// read for every row, Pillow's own scheme (CDSA.32's once-then-stream,
// applied to an image).
// Optimization (browser): the source taken in bands and filtered as it
// arrives, so the module's heap holds an 8 MB intermediate, not a 49 MB
// photo, and never grows for one (WASM.1, WASM.9).
//
// Verification the implementation is held to (tests/media_image_test.cpp):
//   - image_grid against llama.cpp's calc_size_preserved_ratio over a table
//     of sizes — square, wide, tall, tiny, panoramic, at both limits — kept
//     as a fixture by the vision reference tool (runtime/vision.h); 280 to 1
//     taking the budget's most, and 16,384 × 1 refused, OverBudget.
//   - The resampled bytes of each reference image equal llama.cpp's, which
//     the tool dumps after its resize; and the patches equal its
//     normalized floats scaled as the graph scales them, within one unit in
//     the last place, where its GPU's fma may differ from the CPU's
//     multiply then add.
//   - The same image in bands of 1, 7 and every row gives the same bytes.
//   - Each refusal, named.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — the grid and the resample depend on
//            their inputs alone.
//     SL.io.2 When reading, always consider ill-formed input — sizes and
//            bands checked before a pixel is read.
//     SL.con.3 Avoid bounds errors — a grid past the budget refused before
//            it could outgrow the step's buffers.
//     I.5    State preconditions — each function's, below.
//     F.24   Use a span to designate a half-open sequence — pixels and
//            patches as spans.
//     E.27   Use error codes systematically — ImageError.
//   C++ performance guidelines
//     WASM.9 Stream assets in bounded chunks — the bands.
//     WASM.1 Size linear memory to the real high-water mark — the
//            intermediate, not the photo.
//     CDSA.32 Transform static data once into the layout its consumer reads
//            — the taps, once an image; the patches in the weight's layout.

// The largest side a source may have.
inline constexpr std::uint32_t kMaxSide = 16384;

// How many soft tokens an image may become: the model's policy.
struct ImageBudget {
    std::uint32_t least;
    std::uint32_t most;
};

// The size an image is resampled to, and what it becomes.
struct ImageGrid {
    std::uint32_t width;     // pixels, a multiple of patch × pooling
    std::uint32_t height;
    std::uint32_t columns;   // patches across: width / patch
    std::uint32_t rows;      // patches down: height / patch
    std::uint32_t tokens;    // soft tokens: (columns / pooling) × (rows / pooling)
};

enum class ImageError {
    Ok,
    Empty,        // a side of 0
    TooLarge,     // a side over kMaxSide
    OverBudget,   // the grid holds more tokens than the budget's most: an aspect past most to 1
    BadBand,      // a band out of order, of the wrong width, or past the last row
    Incomplete,   // patches asked for before every row arrived
};

// The grid an image of `width` × `height` is resampled to. Preconditions:
// both sides from 1 to kMaxSide; patch and pooling at least 1; budget.least
// at least 1 and at most budget.most.
[[nodiscard]] ImageGrid image_grid(std::uint32_t width, std::uint32_t height, std::uint32_t patch,
                                   std::uint32_t pooling, ImageBudget budget) noexcept;

// Where a source of `width` × `height` lies in `grid`: the size it is
// resampled to and its offset, its aspect kept (step 2 above).
struct ImageFit {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t x;
    std::uint32_t y;
};

// Preconditions: both sides at least 1; the grid image_grid gave for them.
[[nodiscard]] ImageFit image_fit(std::uint32_t width, std::uint32_t height, const ImageGrid& grid) noexcept;

// One image, resampled band by band and cut into patches.
class ImageResampler {
public:
    // Starts an image of `source_width` × `source_height` toward `grid`,
    // placed in it at `fit`: image_grid's and image_fit's for an image, a
    // video frame's frame_grid and frame_fit (media/video.h), which fills
    // the grid. Computes both axes' taps and sizes the intermediate.
    // Refuses Empty, TooLarge, and OverBudget for a grid of more tokens than
    // `most`, before anything is allocated. Precondition: `fit` lies within
    // `grid`.
    [[nodiscard]] ImageError begin(std::uint32_t source_width, std::uint32_t source_height, const ImageGrid& grid,
                                   std::uint32_t patch, std::uint32_t most);

    // Takes the next band: whole RGBA rows from `first_row`, which is the row
    // after the last band's. Filters it horizontally and keeps nothing of
    // its bytes. Refuses BadBand.
    [[nodiscard]] ImageError accept(std::span<const std::uint8_t> rgba, std::uint32_t first_row);

    // Once every row is in: the vertical pass, once, then the patches written
    // to `out`, grid.columns × grid.rows rows of 3 × patch² floats. Refuses
    // Incomplete. Precondition: `out` holds exactly that many floats.
    [[nodiscard]] ImageError patches(std::span<float> out);

    // Once every row is in: the grid's pixels, RGB, width × height × 3 bytes,
    // the image placed in black, as llama.cpp's resize leaves them — what the
    // patches are cut from, for the reference test. Refuses Incomplete.
    // Precondition: `out` holds exactly that many bytes.
    [[nodiscard]] ImageError canvas(std::span<std::uint8_t> out);

private:
    // One axis's taps: for each output pixel, its first source pixel, how
    // many it reads, and `size` fixed-point weights, zero past `count`.
    struct Taps {
        std::vector<std::uint32_t> first;
        std::vector<std::uint32_t> count;
        std::vector<std::int32_t> weights;
        std::uint32_t size = 0;
    };

    void finish();

    Taps horizontal_;
    Taps vertical_;
    bool horizontal_pass_ = false;
    bool vertical_pass_ = false;
    std::vector<std::uint8_t> intermediate_;   // fitted width × source height × 3
    std::vector<std::uint8_t> fitted_;         // fitted width × fitted height × 3, after finish
    ImageGrid grid_{};
    ImageFit fit_{};
    std::uint32_t source_width_ = 0;
    std::uint32_t source_height_ = 0;
    std::uint32_t patch_ = 0;
    std::uint32_t next_row_ = 0;
    bool finished_ = false;
};

}  // namespace bllm::media
