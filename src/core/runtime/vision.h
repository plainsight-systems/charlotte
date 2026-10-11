#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/kernels/program.h"
#include "core/media/image.h"

namespace bllm::kernels {
class ProgramLink;   // kernels/program.h, as this design asks of it
}

namespace bllm::runtime {

// Axis M (encode): changes with a new stage in the encode path.
//
// The image stage of an encode: an image becomes rows of the text model's
// width, and those rows stand in the text step where the image's
// placeholder stood, so one embedding represents the text and the image
// together, comparable with any other (docs/research/
// 2026-10-11-embeddinggemma2-vision.md). The embedder runs it; it owns the
// vision encoder's program and the staging slot an image is prepared in.
//
// An encode with an image, end to end:
//   1. Prepare (before the encode, and while another runs): the page
//      decodes the image and hands the module its pixels, a band of rows at
//      a time, into the staging slot (web/image_input.js); the slot sizes
//      the grid for the budget, resamples and cuts the patches
//      (media/image.h). The slot holds one image's patches, 3 P² × the
//      budget's most patches floats, 7.7 MB, allocated at load.
//   2. The text, UTF-8, with `<|image|>` exactly once where the image goes —
//      the card's placeholder; the page writes it alone for an image alone,
//      which the card says takes no task prompt. Encoded by the model's
//      tokenizer, which gives the placeholder as its one special token;
//      wrapped in the file's ends (ends.h); then the placeholder replaced by
//      `<|image>`, M rows for the image, `<image|>` — llama.cpp's
//      img_beg and img_end for gemma4v, three tokens of the text model's
//      vocabulary resolved by text at load. M = the slot's grid's tokens, at
//      most 280. The text's tokens and the M rows together are at most the
//      context, 512: an image of 280 leaves 230 for the text and the ends.
//   3. The vision step: position 0, N = M × k² patches, `columns` the grid's;
//      the slot's patches written into the vision program's `pixels` in one
//      write; one pass (graph/vision.h). wgpuQueueWriteBuffer copies the
//      bytes before it returns, so the slot is free once start() has
//      returned, and the page prepares the next image into it while this
//      one's passes run.
//   4. The text step, with its media rows: `media_first` the row after
//      `<|image>`, `media_rows` M. Before its pass, one copy of rows 0 ..
//      M − 1 of the vision program's `soft_tokens` into those rows of the
//      text program's `hidden`, M × W × 4 bytes, 573 KB at 280 and W =
//      512; gather writes every row of the step but those
//      (kernels/gather/gather.h), whose identifiers are written 0 and never
//      read. Then the text encoder's step exactly as for a text
//      (graph/encoder.h): the per-layer inputs projected from every row,
//      token or image alike, as llama.cpp's gemma-embedding2 graph does; no
//      scale on the image's rows, which llama.cpp gives only token rows; the
//      mean over every row.
//   5. The vector read back, as embedder.h reads it: one callback, Ok and
//      the vector, or the failure.
//   - An image's rows are the vision step's alone and an image is one step:
//     nothing of one image is kept for the next, as nothing of one text is.
//   - Refusals before the GPU runs, each named, start() returning at once:
//     ImageNotReady, the slot empty or still filling; Placeholder, the text
//     holding no `<|image|>` with an image, one without, or more than one;
//     TooManyTokens, the text's tokens and M rows past the context, with
//     the count; and the image's own refusals as it is prepared
//     (media/image.h's ImageError), named to the page when the band that
//     failed returns. After submit: Step, with WebGPU's message, for either
//     pass; DeviceLost; NonFinite; Cancelled — embedder.h's, unchanged.
//
// The companion's load, after the text model's, once the page first needs
// it (web/companion.js):
//   - Its catalog entry names the file, its ranges and their SHA-256, and
//     the image budget (web/models.json): the companion is the curated
//     list's to name; a model from a pasted URL has none.
//   - Preflight reads its header from the first range and judges it stage by
//     stage, as a model's (preflight/preflight.h): Read; Download, its
//     ranges; Describe, its projector type found (capability.h) and
//     described (arch/gemma4v/gemma4v.h); Fit, the plan against the ranges
//     (residency/file_ranges.h) within what the budget leaves beside the
//     text model's plan — 2 GiB less 384 MB by default, so its 325 MB fits;
//     Upload; Run, its graph building, the text model's vocabulary holding
//     the three image tokens, and pairs (arch/vision_encoder.h) accepting the
//     loaded text model. A verdict names everything still missing, as a
//     model's does.
//   - Loaded: its own upload on the same device and its own program, built
//     from its graph with no readback, and the link to the text program
//     below. Released before the text model whenever the text model is,
//     and with it: a link never outlives either program.
//   - Text encodes are untouched by it: an encode without an image runs the
//     text step alone, its media rows 0, as now.
//
// What it asks of the other contracts:
//   - kernels/program.h: (1) a program built with no readback; (2) an input
//     write: one wgpuQueueWriteBuffer of a run's bytes into a named working
//     buffer before its pass — the patches into `pixels`; (3) a link: made
//     at load from a source program's working buffer and a target's —
//     `soft_tokens` into `hidden` — holding both buffers, so a run of the
//     target can encode one copy of a byte range between them before its
//     pass. The program module keeps every GPU object; the embedder names
//     purposes and rows.
//   - kernels/interface.h: the step's `columns`, `media_first` and
//     `media_rows` (graph/vision.h).
//   - kernels/gather/gather.h: an invocation whose row lies in the step's
//     media rows returns before reading its identifier: those rows are the
//     copy's.
//   - runtime/embedder.h: start(text, image), the image the slot's, and
//     optional; the text
//     without an image refused, Placeholder, where it holds `<|image|>`;
//     the context check counting M rows; the vision stage optional,
//     attached once the companion loads.
//   - policy/policy.h: the image budget, least and most tokens, from the
//     catalog — 70 and 280 for EmbeddingGemma 2, the card's default most;
//     and the budget the companion's plan is given, the model's less the
//     text model's plan.
//   - preflight/preflight.h: a companion's verdict, its stages as above,
//     judged with the loaded text model's description and vocabulary.
//   - The boundary, contract 11 (src/wasm/bindings.cpp, web/worker.js):
//       bllm_companion_preflight(request, prefix, length, file_size, ranges…)
//       bllm_companion_load_begin(request, prefix, length, file_size,
//                                 ranges…, policy…)
//       bllm_companion_load_chunk(request, file_offset, length)
//       bllm_companion_load_finish(request)
//       bllm_image_begin(request, width, height) → the grid, or the refusal
//       bllm_image_rows(request, first_row, pointer, rows)
//       bllm_embed(request, text, text_length, image), image 1 to encode
//                                                       the slot's image
//     The pixels cross once a band, as a view of the worker's band copied
//     into the module's memory; the grid crosses back once an image; the
//     vector once an encode, as now (WASM.2).
//
// What it costs, counted, for EmbeddingGemma 2 at Q8_0 and an image of M =
// 280 tokens, N = 2,520 patches:
//   - Preparing, on the CPU: media/image.h's resample, about 0.1 to 0.2 s
//     for a 12-megapixel photo by estimate, and one crossing a band — a
//     4,032-pixel-wide photo in bands of 4 MiB is 260 rows a band, 12
//     crossings. It overlaps the GPU's work on the image before: the worker
//     is free while a pass runs, and the slot is free once start() returns,
//     so the page prepares image k + 1 while image k encodes (GPU.7).
//   - The vision step: 133 launches, 5.4 × 10¹¹ multiply-adds, about 0.4 to
//     0.5 s on the target by estimate (graph/vision.h); the patches' write,
//     7.7 MB, one call.
//   - The link's copy: 573 KB, under 2 µs at 400 GB/s; one command.
//   - The text step: about 284 rows, 29 ms by the measured 26.0 at 254 and
//     33.5 at 336 (docs/research/2026-10-11-chrome-encode-benchmark.md).
//   - The readback: 3 KiB, about 0.5 ms, as for a text.
//   - Memory: the slot, 7.7 MB of the module's heap at load; on the
//     device, the encoder's 226 MB and its working set's 98 MB.
// Optimization (browser): the companion's audio encoder never fetched,
// cached or uploaded (residency/file_ranges.h).
// Optimization (browser): the next photo prepared while the last encodes,
// the single-threaded worker otherwise idle during a pass; one slot is
// enough, since the patches' write ends the slot's part in an encode
// before start() returns (GPU.7).
// Optimization not taken: writing the vision encoder's projection straight
// into the text program's `hidden`, which would bind one upload's buffer in
// another's program; the copy costs under 2 µs and keeps each program to its
// own upload.
//
// Verification the implementation is held to:
//   - Against the reference: tools/make_reference_vision.sh builds
//     llama.cpp at the embeddings' pin, copies the mmproj's header with
//     clip.use_gelu set (gemma4v.h), and embeds, on its CPU and on Metal with
//     --image-max-tokens 280, four public-domain photographs — each committed
//     as a PNG under tests/fixtures/vision/ with its source and license — alone,
//     and one with a caption around its placeholder. It keeps, as fixtures:
//     the grid sizes over a table of sizes and each image's resampled bytes
//     (media/image.h); each image's soft tokens; and each embedding. Each of
//     Charlotte's soft tokens and embeddings within the largest difference
//     llama.cpp shows between its own two backends of Metal's, and its
//     cosine with Metal's at least the CPU's with Metal's, as embedder.h's
//     check is. The check discriminates: the axial rotation's axes swapped,
//     and the feed-forward run with GELU-quick, each fall outside.
//   - The tokens: each fixture text's identifiers, the image tokens placed,
//     as llama.cpp's server tokenized them.
//   - On the GPU: the same image embedded twice gives the same bits, and
//     the same when prepared while another encodes; a text encode after an
//     image's gives the bits a fresh embedder gives; each refusal named.
//   - In the app's Chromium, the page: each of the fixtures' four queries
//     ranks its own photograph first among the four and the search
//     fixture's paragraphs (web/search.js).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — the program and the
//            slot owned, by unique_ptr and value.
//     E.27   Use error codes systematically — each refusal named.
//     SL.io.2 When reading, always consider ill-formed input — the text's
//            placeholders counted, the image's bands checked.
//     F.24   Use a span to designate a half-open sequence — bands and rows.
//   C++ performance guidelines
//     GPU.1  Keep data on the device — the image's rows go from one program
//            to the other on the GPU; only the vector comes back.
//     GPU.7  Pipeline CPU and GPU work — the next image prepared during a
//            pass.
//     WASM.2 Batch work across the JS boundary — a band a crossing, the
//            grid once, the vector once.
//     MEM.9  Allocate at init — the slot at load.

enum class SlotState {
    Empty,
    Filling,   // begun, rows still arriving
    Ready,     // patches written, awaiting an encode
};

// One image's patches, prepared on the CPU.
struct ImageSlot {
    media::ImageResampler resampler;
    media::ImageGrid grid{};
    std::vector<float> patches;   // 3 P² × the budget's most patches, at load
    SlotState state = SlotState::Empty;
};

// The vision encoder, loaded beside the text model, and its slot.
class Vision {
public:
    // Preconditions: `program` was built from the vision encoder's graph,
    // with no readback, and `link` copies its `soft_tokens` into the text
    // program's `hidden`; `budget` is the policy's; `patch` and `pooling`
    // the encoder's (model_description.h).
    Vision(std::unique_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
           media::ImageBudget budget, std::uint32_t patch, std::uint32_t pooling);

    // Starts an image of `width` × `height` in the slot, discarding whatever
    // it held: its grid, or the image's refusal. May be called at any time:
    // an encode's start() writes the slot's patches to the GPU before it
    // returns, so no encode holds the slot after.
    [[nodiscard]] media::ImageError begin(std::uint32_t width, std::uint32_t height);

    // The next band of the slot's image; the last band writes its patches.
    [[nodiscard]] media::ImageError rows(std::uint32_t first_row, std::span<const std::uint8_t> rgba);

    [[nodiscard]] const ImageSlot& slot() const noexcept;

    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

private:
    std::unique_ptr<kernels::Program> program_;
    std::unique_ptr<kernels::ProgramLink> link_;
    ImageSlot slot_;
    media::ImageBudget budget_;
    std::uint32_t patch_;
    std::uint32_t pooling_;
};

}  // namespace bllm::runtime
