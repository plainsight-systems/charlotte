#pragma once

#include <cstdint>

#include "core/kernels/program.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::runtime {

// Axis M (encode): changes with what the embedder asks of a media stage.
//
// What the embedder asks of the stage that turns a medium into rows of the
// text model's width (runtime/vision.h for an image, runtime/audio.h for an
// audio clip, runtime/video.h for a video's clip): whether a medium is
// prepared and the rows it makes, as segments; to submit the encoder's step
// over each segment, its rows copied into the text program's `hidden`.
// Everything else — how a medium is prepared, how the page hands it over,
// the encoder's program and its link — is the stage's, and the embedder
// never sees it.
//
// An interface, not a variant over the stages: the embedder holds a stage a
// kind and calls it once a segment, so one indirect call a segment is the
// cost (WASM.4: the hot path is the GPU's), and a kind adds a stage without
// the embedder learning its type (C.121, I.25).
//
// A segment is one run of the encoder's step and one run of rows in the
// text step, between the kind's begin and end tokens: an image is one, an
// audio clip one, a video's clip one a frame (runtime/video.h), since
// EmbeddingGemma 2's processor writes `<|image>`, the frame's rows,
// `<image|>` for each frame and the vision encoder takes one frame a step.
//
// The embedder's encode with a medium, for every kind:
//   - The text holds its kind's placeholder once — `<|image|>`, `<|audio|>`
//     or `<|video|>` — replaced by, for each segment, the kind's begin
//     token, the segment's rows, the kind's end token, segments back to
//     back. A text holding a placeholder of another attached kind, or its
//     own kind's without a medium, is refused, Placeholder: one medium an
//     encode. The card allows several, mixed; the context of one step, 512,
//     holds an image and a short clip at most, and the page never asks for
//     two (web/search.js).
//   - A segment's rows carry the placeholder's identifier, so gather writes
//     every row but them (kernels/gather/gather.h): the step's
//     `media_token` is the placeholder.
//   - run(i, out) submits segment i's step, its rows copied after its pass
//     into the text program's `hidden` at out's offset, before the next
//     segment's step may overwrite them (kernels/program.h). Segments are
//     submitted in order while fewer than two of the stage's are
//     outstanding — a program's limit — the next as each completes; the
//     text step is submitted once the last segment is, behind it on the
//     queue, with no wait. One segment, as an image's or a clip's, is the
//     two submits back to back of before.
//   - The encode's one callback waits for every step it submitted to
//     report: the text step's and each segment's. Two programs' reports
//     are not ordered with each other — a step reports once its work is
//     done and its error scopes have settled, each program in its own order
//     — so a text step's vector may be read back before the last segment's
//     failure is known; delivered then, it would be a vector over rows that
//     were never written. So the embedder counts the steps outstanding and
//     delivers when the count reaches zero: the vector if every report was
//     Ok, else the first failure reported, and the vector dropped. Once a
//     segment's failure is reported, no further segment is submitted; one
//     already submitted runs, and its report is awaited. This holds for one
//     segment as for several: an image's or a clip's step and its text step
//     are two reports, where the embedder before took the media step's to
//     come first.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.121  If a base class is used as an interface, make it a pure abstract
//            class.
//     I.25   Prefer empty abstract classes as interfaces to class hierarchies.
//     C.35   A base class destructor should be either public and virtual, or
//            protected and non-virtual — public and virtual: the embedder
//            owns a stage by its interface.
//     F.24   Use a span to designate a half-open sequence — the segments.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — one call a segment.
//     GPU.7  Pipeline CPU and GPU work — two segments in flight, and the
//            text step behind the last with no wait.
//     E.4    Design your error-handling strategy around invariants — no
//            vector delivered while a step that made its rows may yet
//            fail.

// The text model's tokens a kind's rows stand among, resolved at load from
// its vocabulary by text: the placeholder a text writes, and the tokens
// llama.cpp puts around the rows — img_beg and img_end for gemma4v, aud_beg
// and aud_end for gemma4a.
struct MediaTokens {
    tokenizer::TokenId placeholder;   // "<|image|>", "<|audio|>", "<|video|>"
    tokenizer::TokenId begin;         // "<|image>", "<|audio>", "<|image>"
    tokenizer::TokenId end;           // "<image|>", "<audio|>", "<image|>"
};

class MediaStage {
public:
    virtual ~MediaStage() = default;

    // The rows the prepared medium makes, or 0 when none is prepared.
    [[nodiscard]] virtual std::uint32_t ready_rows() const noexcept = 0;

    // Submits the encoder's step over the prepared medium and empties the
    // stage. Precondition: ready_rows() > 0.
    virtual void run(kernels::StepCallback done, void* userdata) = 0;

    // Copies the step's `soft_tokens` into the text program's `hidden`.
    [[nodiscard]] virtual const kernels::ProgramLink& link() const noexcept = 0;

protected:
    MediaStage() = default;
    MediaStage(const MediaStage&) = default;
    MediaStage& operator=(const MediaStage&) = default;
};

}  // namespace bllm::runtime
