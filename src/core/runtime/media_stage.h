#pragma once

#include <cstdint>

#include "core/kernels/program.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::runtime {

// Axis M (encode): changes with what the embedder asks of a media stage.
//
// What the embedder asks of the stage that turns a medium into rows of the
// text model's width (runtime/vision.h for an image, runtime/audio.h for an
// audio clip): whether a medium is prepared and how many rows it makes; to
// submit the encoder's step over it; and the link that copies those rows
// into the text program's `hidden`. Everything else — how a medium is
// prepared, how the page hands it over, the encoder's program — is the
// stage's, and the embedder never sees it.
//
// An interface, not a variant over the two stages: the embedder holds a
// stage a kind and calls it at most once an encode, so one indirect call an
// encode is the cost (WASM.4: the hot path is the GPU's), and a third kind —
// video's frames, which are images — adds a stage without the embedder
// learning its type (C.121, I.25).
//
// The embedder's encode with a medium, as runtime/vision.h set it out for an
// image, now for either kind:
//   - The text holds its kind's placeholder once — `<|image|>` or
//     `<|audio|>` — replaced by the kind's begin token, the stage's
//     ready_rows() rows, the kind's end token. A text holding a placeholder
//     of the other kind, or its own kind's without a medium, is refused,
//     Placeholder: one medium an encode. The card allows several, mixed;
//     the context of one step, 512, holds an image and a short clip at
//     most, and the page never asks for two (web/search.js).
//   - run() submits the stage's step; the embedder's text step follows it on
//     the queue, its RunCopy through link() first. A failure of the stage's
//     step is the encode's, reported in the text step's place.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.121  If a base class is used as an interface, make it a pure abstract
//            class.
//     I.25   Prefer empty abstract classes as interfaces to class hierarchies.
//     C.35   A base class destructor should be either public and virtual, or
//            protected and non-virtual — public and virtual: the embedder
//            owns a stage by its interface.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — one call an encode.

// The text model's tokens a kind's rows stand among, resolved at load from
// its vocabulary by text: the placeholder a text writes, and the tokens
// llama.cpp puts around the rows — img_beg and img_end for gemma4v, aud_beg
// and aud_end for gemma4a.
struct MediaTokens {
    tokenizer::TokenId placeholder;   // "<|image|>", "<|audio|>"
    tokenizer::TokenId begin;         // "<|image>", "<|audio>"
    tokenizer::TokenId end;           // "<image|>", "<audio|>"
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
