#pragma once

#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/gguf/index.h"
#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace bllm::arch {

// Contract: what every vision encoder supplies.
//
// A vision encoder turns an image's patches into rows a text model reads in
// place of tokens: soft tokens. It lives in a companion file, an mmproj,
// beside the text model it serves (runtime/vision.h), and that file's
// general.architecture is "clip" whatever the encoder: the identifier that
// says which encoder it is, and so which implementation runs it, is
// clip.vision.projector_type (logical-overview.md, principle 2). So a
// vision encoder is found through its own lookup in the capability table,
// by projector type, and is not an Architecture: its graph is not a
// generator's or an embedder's, it is never loaded alone, and an
// Architecture is found by an identifier this file does not use for it.
//
// It supplies what an architecture does, for its own kind:
//   - describe: reads the encoder's numbers from the file's clip.vision.*
//     keys into a model description with a vision description
//     (model_description.h), names each layer's tensors by role, and the
//     encoder's own tensors — the patch embedding, the position table, the
//     projection. Fails naming the key or tensor, as an architecture's does.
//   - graph: every launch of the encoder's one step over an image's
//     patches, composed from the vision blocks (graph/vision.h).
// And one check an architecture has no need of: pairs, whether the text
// model it is loaded beside reads its rows — its projection's width is that
// model's hidden width, and that model is one it is known to serve — refused
// by name otherwise (runtime/vision.h).
//
// Function pointers in a static table, as Architecture's, and for the same
// reasons: no state, chosen once at load, never called per step (WASM.4).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed — a vision encoder
//            is its own type, not an Architecture holding a third kind.
//     E.27   Use error codes systematically — describe's DescribeResult.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths — chosen once, at load.

// Fills `out` with every launch of the encoder's step. Preconditions:
// `model` is this encoder's describe's, with a vision description; `plan`
// was made from it with the vision working set, as upload carried it out.
using VisionGraphFn = graph::GraphResult (*)(const model::ModelDescription& model,
                                             const residency::ResidencyPlan& plan,
                                             std::vector<kernels::Launch>& out);

// Whether a text model reads this encoder's rows: the text model's
// description, and its general.architecture.
using PairsFn = DescribeResult (*)(const model::ModelDescription& vision, const model::ModelDescription& text,
                                   std::string_view text_architecture);

struct VisionEncoder {
    std::string_view name;
    DescribeFn describe;
    VisionGraphFn graph;
    PairsFn pairs;
};

}  // namespace bllm::arch
