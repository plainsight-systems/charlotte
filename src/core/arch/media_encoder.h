#pragma once

#include <array>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/gguf/index.h"
#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace bllm::arch {

// Contract: what every media encoder supplies.
//
// A media encoder turns an image's patches or an audio clip's spectrogram
// into rows a text model reads in place of tokens: soft tokens. It lives in
// a companion file, an mmproj, beside the text model it serves
// (runtime/vision.h, runtime/audio.h), and that file's general.architecture
// is "clip" whatever the encoder. One file may hold an encoder of each kind,
// as EmbeddingGemma 2's does, each named by its own key: an image encoder by
// clip.vision.projector_type, an audio encoder by clip.audio.projector_type
// (logical-overview.md, principle 2). So an encoder is found by its kind and
// that key's value, through its own lookup in the capability table, and is
// not an Architecture: its graph is not a generator's or an embedder's, it is
// never loaded alone, and an Architecture is found by an identifier this file
// does not use for it.
//
// It supplies what an architecture does, for its own kind:
//   - describe: reads the encoder's numbers from the file's keys for its
//     kind — clip.vision.* or clip.audio.* — into a model description with
//     that kind's description (model_description.h), names each tensor it
//     reads by role, and refuses, naming it, any tensor of its kind's
//     prefixes it doesn't describe. The other kind's tensors it never reads:
//     a companion loaded for one kind leaves the other's outside its ranges
//     (residency/file_ranges.h).
//   - graph: every launch of the encoder's one step over a medium's rows,
//     composed from that kind's blocks (graph/vision.h, graph/audio.h).
//   - setup, or null: every launch of a step run once, after the encoder's
//     upload and before its first step, that writes what depends on the
//     weights alone into a working buffer the step reads and nothing else
//     writes — gemma4a's relative position tables (graph/audio.h). Null for
//     gemma4v.
// And what an architecture has no need of: pairs, whether the text model
// it is loaded beside reads its rows — its projection's width is that
// model's hidden width, and that model is one it is known to serve —
// refused by name otherwise; and the three tokens of that model's
// vocabulary its rows stand among, by text — the placeholder a text writes,
// and the tokens llama.cpp's multimodal library puts around the rows for
// this projector (runtime/media_stage.h). An image encoder that also reads
// a video's frames names a second three, the video's: its placeholder and
// the tokens around each frame's rows — `<|video|>`, `<|image>`, `<image|>`
// for gemma4v, as EmbeddingGemma 2's processor writes a video
// (runtime/video.h); empty for one that does not.
//
// Video is a kind of medium with no encoder and no companion of its own: a
// video's frames are images, read by the image encoder its companion
// already loads, so no projector key names it and find_media_encoder never
// takes it; its stage is attached by the image companion's load.
//
// Function pointers in a static table, as Architecture's, and for the same
// reasons: no state, chosen once at load, never called per step (WASM.4).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed — the kind is an
//            enum, and an encoder is its own type, not an Architecture.
//     Enum.3 Prefer class enums over "plain" enums — MediaKind.
//     E.27   Use error codes systematically — describe's DescribeResult.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — chosen once, at load.

// What a media encoder takes: which key names it, which placeholder a text
// writes for it, which stage runs it.
enum class MediaKind {
    Image,
    Audio,
    Video,   // an image encoder's, by its companion (runtime/video.h)
};

// Fills `out` with every launch of a step. Preconditions: `model` is this
// encoder's describe's, with its kind's description; `plan` was made from it
// with its kind's working set, as upload carried it out.
using MediaGraphFn = graph::GraphResult (*)(const model::ModelDescription& model,
                                            const residency::ResidencyPlan& plan,
                                            std::vector<kernels::Launch>& out);

// Whether a text model reads this encoder's rows: the text model's
// description, and its general.architecture.
using PairsFn = DescribeResult (*)(const model::ModelDescription& encoder, const model::ModelDescription& text,
                                   std::string_view text_architecture);

struct MediaEncoder {
    MediaKind kind;
    std::string_view name;   // the projector type
    DescribeFn describe;
    MediaGraphFn graph;
    MediaGraphFn setup;      // or null
    PairsFn pairs;
    // The placeholder, the token before the rows, the token after.
    std::array<std::string_view, 3> tokens;
    // An image encoder's that reads a video's frames: the video's
    // placeholder, and the tokens before and after each frame's rows; all
    // empty otherwise.
    std::array<std::string_view, 3> video_tokens{};
};

// The key whose value names a companion's encoder of `kind`, an image's or
// an audio clip's. Precondition: kind is not Video, which has no companion.
[[nodiscard]] constexpr std::string_view projector_key(MediaKind kind) noexcept {
    return kind == MediaKind::Image ? "clip.vision.projector_type" : "clip.audio.projector_type";
}

}  // namespace bllm::arch
