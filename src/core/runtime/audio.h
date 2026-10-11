#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/kernels/program.h"
#include "core/media/audio.h"
#include "core/runtime/media_stage.h"

namespace bllm::runtime {

// Axis M (encode): changes with a new stage in the encode path.
//
// The audio stage of an encode: a clip of up to 20 s becomes rows of the
// text model's width, and those rows stand in the text step where the
// clip's placeholder stood, so one embedding represents the text and the
// clip together, comparable with any other (docs/research/
// 2026-10-11-embeddinggemma2-audio.md). The image stage's design
// (runtime/vision.h) with a spectrogram where it has patches; the embedder
// runs it through media_stage.h's interface.
//
// An encode with a clip, end to end:
//   1. Prepare (before the encode, and while another runs): the page decodes
//      a recording, mixes it to mono at 16 kHz and cuts it into clips of at
//      most most_samples() (web/audio_input.js); each clip's samples cross
//      into the module in one call, and prepare() computes its spectrogram
//      into the slot (media/audio.h): T frames of 128 bins, at most 4 × the
//      most rows — 2,032 × 128 floats, 1.0 MB, allocated at load.
//   2. The text, UTF-8, with `<|audio|>` exactly once where the clip goes;
//      the page writes it alone, since the card says audio takes no task
//      prompt. Encoded and wrapped as any text; the placeholder replaced by
//      `<|audio>`, R rows for the clip, `<audio|>` — llama.cpp's aud_beg
//      and aud_end for gemma4a, three tokens of the text model's vocabulary
//      resolved by text at load. R = ⌈⌈T / 2⌉ / 2⌉, at most the context
//      less the ends and the two tokens: 508.
//   3. The audio step: position 0, R tokens, `columns` T; the slot's
//      spectrogram written into the audio program's `mel` in one write; one
//      pass (graph/audio.h). The write copies the bytes before it returns,
//      so the slot is free once run() has returned, and the next clip is
//      prepared into it while this one's passes run.
//   4. The text step, with its media rows: `media_first` the row after
//      `<|audio>`, `media_rows` R; one copy of `soft_tokens` rows 0 .. R − 1
//      into those rows of the text program's `hidden`, R × 512 × 4 bytes,
//      1.0 MB at 508; then the text encoder's step as for an image's rows,
//      unscaled.
//   5. The vector read back, as embedder.h reads it.
//   - Refusals before the GPU runs, each named, start() returning at once:
//     MediaNotReady, no clip prepared; Placeholder, as media_stage.h states;
//     TooManyTokens, the text's tokens and R rows past the context; and the
//     clip's own as it is prepared — TooShort under 161 samples, TooLong
//     past most_samples() (media/audio.h). After submit: embedder.h's,
//     unchanged.
//
// The setup, once, between the program's build and the stage's first step:
// the encoder's setup graph (graph/audio.h), a step of 13 rows, its
// `positions` written with the 13 sinusoidal rows — row p the position 12 −
// p, its first W / 2 floats sin(position × e^(−i × ln(10,000) / (W / 2 −
// 1))) for i below W / 2 and its last W / 2 the cosines, f32, as llama.cpp's
// gemma4a input computes them — and run once, its relative tables then
// read by every step. The stage is attached once the setup's step has
// completed; a failure there is the load's, named.
//
// The companion's load, as the image's (runtime/vision.h), for its own kind:
//   - Its catalog entry, `audio`, names the file, its ranges and their
//     SHA-256 (web/models.json): for EmbeddingGemma 2 the same mmproj as
//     `vision`'s, one range, [0, 328,492,448) — the header and the audio
//     encoder are contiguous — 328.5 MB, cached as its own file. No budget:
//     a clip's rows are the context's.
//   - Preflight from its header, stage by stage, as the image's, by
//     clip.audio.projector_type (arch/media_encoder.h), its plan for the
//     most rows within what the budget leaves beside the text model and
//     whatever other companion is loaded — 2 GiB less 384 MB, less the
//     vision encoder's 320 MB when it is loaded, so its ~400 MB fits; Run,
//     its graph and setup building, the text model's vocabulary holding the
//     three audio tokens, and pairs.
//   - Loaded: its own upload, its own two programs, setup and step, with no
//     readback, and the link. Released with the text model, as the image's;
//     each kind's companion is released apart from the other's.
//
// What it asks of the other contracts:
//   - runtime/embedder.h: its Audio kind's stage attached and started as
//     the image's (media_stage.h).
//   - preflight/preflight.h: CompanionHost gains the text model's context
//     offered, which bounds a clip's rows; the audio kind's working set.
//   - capability/capability.h: find_media_encoder(Audio, "gemma4a").
//   - kernels/program.h: nothing new — a second program on the same upload,
//     its own step, inputs and no readback.
//   - The boundary, contract 11 (src/wasm/bindings.cpp, web/worker.js):
//       bllm_companion_preflight(request, kind, …) and
//       bllm_companion_load_begin(request, kind, …), the kind 0 image or 1
//         audio, the rest as now; chunk and finish as now, one companion
//         loading at a time; finish answering, for audio, the most samples
//         a clip may hold, once its setup has run;
//       bllm_audio_prepare(request, pointer, count) → the clip's frames and
//         rows, or the refusal;
//       bllm_embed(request, text, text_length, media), media 0 none, 1 the
//         image stage's, 2 the audio stage's.
//     A clip's samples cross once, f32, copied into the module's memory;
//     its rows cross back once; the vector once an encode (WASM.2).
//
// What it costs, counted, for EmbeddingGemma 2 at Q8_0 and a 20 s clip, T =
// 1,999, R = 500:
//   - Preparing, on the CPU: one crossing of 1.28 MB; the spectrogram, a
//     few milliseconds (media/audio.h). It overlaps the GPU's work on the
//     clip before, as an image's preparing does (GPU.7).
//   - The audio step: 199 launches, 1.48 × 10¹¹ multiply-adds, about 0.1 s
//     on the target by estimate (graph/audio.h); the spectrogram's write,
//     1.0 MB, one call.
//   - The link's copy: 1.0 MB, a few microseconds; one command.
//   - The text step: about 504 rows, about 45 ms, between the measured
//     39.2 ms at 418 tokens and the estimate at 512.
//   - The readback: 3 KiB, about 0.5 ms.
//   - About 0.15 s a 20 s clip, so a 5 minute recording, 15 clips, indexes
//     in about 2.3 s.
//   - The setup, once: 12 products of 13 rows, 1.6 × 10⁸ multiply-adds.
//   - Memory: the slot, 1.0 MB of the module's heap at load; on the device,
//     the encoder's 328 MB and its working set's ~67 MB.
// Optimization (browser): the companion's vision encoder never fetched,
// cached or uploaded for audio, nor its audio encoder for images; each
// kind's range alone (residency/file_ranges.h).
// Optimization (browser): the next clip prepared while the last encodes,
// one slot, as the image's (GPU.7).
// Optimization (practice): the relative tables computed once, at load
// (graph/audio.h, GDSA.17).
//
// Verification the implementation is held to:
//   - Against the reference: tools/make_reference_audio.sh builds llama.cpp
//     at the embeddings' pin with one line changed — the gemma4a
//     preprocessor's mel_floor_add set, Google's floor (media/audio.h) — and
//     embeds, on its CPU and on Metal: five computed sounds
//     (tests/fixtures/audio/sounds.h: a tone, a sweep, noise, clicks,
//     silence) and four public-domain recordings from NASA's historical
//     sounds page — President Kennedy's "we choose to go to the moon", its
//     first 20 s; Apollo 8's Christmas message; STS-135's countdown, its
//     first 20 s; Sputnik's beep — each decoded from its pinned MP3 by
//     llama.cpp's own decoder to 16 kHz mono and committed as a 16-bit WAV
//     under tests/fixtures/audio/ with its source; alone, and one with a
//     caption around its placeholder. It keeps, as fixtures: the spectrogram
//     frames media/audio.h's check reads; each clip's rows and tokens; each
//     embedding; and beside each, its cosine with unchanged llama.cpp. Each
//     of Charlotte's embeddings within llama.cpp's own backends' spread, as
//     embedder.h's check is (tests/gpu/audio_embed_test.cpp). The check
//     discriminates: the encoder run with llama.cpp's floor falls outside on
//     the tone and the sweep; the relative term read at row j in place of
//     12 − j falls outside on every recording.
//   - Each of the four recordings' queries — "a president explaining why we
//     go to the moon", "a christmas greeting from astronauts orbiting the
//     moon", "a countdown to a rocket launch", "the beeping signal of the
//     first satellite" — ranks its own recording first, natively.
//   - On the GPU: the same clip embedded twice gives the same bits, and the
//     same when prepared while another encodes; a text encode after a
//     clip's gives the bits a fresh embedder gives; an image's encode after a
//     clip's, the image's bits; each refusal named.
//   - In the app's Chromium, the page: the four WAVs, 16 kHz mono 16-bit,
//     decoded by the browser to the same samples, their queries ranking
//     their own recording first among them, the search fixture's paragraphs
//     and the vision fixture's photos, each score llama.cpp's to three
//     places (web/search.js).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — programs and the
//            slot owned, by unique_ptr and value.
//     E.27   Use error codes systematically — each refusal named.
//     SL.io.2 When reading, always consider ill-formed input — a clip's
//            length checked before a sample is read.
//     F.24   Use a span to designate a half-open sequence — a clip's samples.
//   C++ performance guidelines
//     GPU.1  Keep data on the device — rows from one program to the other on
//            the GPU; only the vector comes back.
//     GPU.7  Pipeline CPU and GPU work — the next clip prepared during a
//            pass.
//     WASM.2 Batch work across the JS boundary — a clip a crossing.
//     MEM.9  Allocate at init — the slot at load.

// One clip's spectrogram, prepared on the CPU.
struct AudioSlot {
    media::AudioClip clip{};
    std::vector<float> mel;   // 4 × the most rows × kMelBins floats, at load
    bool ready = false;
};

// The audio encoder, loaded beside the text model, and its slot.
class Audio final : public MediaStage {
public:
    // Preconditions: `program` was built from the audio encoder's graph,
    // with no readback, and its setup has run (run_audio_setup); `link`
    // copies its `soft_tokens` into the text program's `hidden`; `mel` is
    // its plan's `mel`; `most_rows` the plan's rows.
    Audio(std::unique_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
          kernels::Binding mel, std::uint32_t most_rows);

    // Prepares a clip in the slot, discarding whatever it held: its frames
    // and rows, or the clip's refusal. May be called at any time, as an
    // image's begin().
    [[nodiscard]] media::AudioError prepare(std::span<const float> samples);

    [[nodiscard]] const AudioSlot& slot() const noexcept { return slot_; }

    // The most samples a clip may hold.
    [[nodiscard]] std::uint32_t most_samples() const noexcept;

    [[nodiscard]] std::uint32_t ready_rows() const noexcept override;
    void run(kernels::StepCallback done, void* userdata) override;
    [[nodiscard]] const kernels::ProgramLink& link() const noexcept override { return *link_; }

    ~Audio() override;
    Audio(const Audio&) = delete;
    Audio& operator=(const Audio&) = delete;

private:
    std::unique_ptr<kernels::Program> program_;
    std::unique_ptr<kernels::ProgramLink> link_;
    kernels::Binding mel_;
    media::Spectrogram spectrogram_;
    AudioSlot slot_;
    std::uint32_t most_rows_;
};

// Runs the audio encoder's setup step once: the 13 sinusoidal rows of
// `width` written into `positions`, then the setup program's one pass, its
// completion reported to `done`. Preconditions: `setup` was built from the
// encoder's setup graph, with no readback, on the upload the step's program
// is built on; `positions` its plan's `positions`.
void run_audio_setup(kernels::Program& setup, kernels::Binding positions, std::uint32_t width,
                     kernels::StepCallback done, void* userdata);

}  // namespace bllm::runtime
