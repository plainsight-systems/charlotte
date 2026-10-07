#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/formats/format.h"
#include "core/gguf/index.h"
#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace bllm::arch {

// Contract: what every architecture supplies.
//
// Each arch/<arch>/ provides one Architecture, and the capability table lists
// it under every general.architecture value it answers to. An architecture is
// the only code that knows which architecture is loaded. It supplies two
// things:
//
//   - describe: reads its numbers from the tensor index into a model
//     description, and names each layer's tensors by role. Fails naming the
//     key or tensor when the file lacks what the architecture needs.
//   - graph: every kernel launch of a step, in order, composed from the
//     blocks every architecture shares (graph/graph.h) — the order its family
//     runs them in, and what its family does that the description does not
//     carry, such as Gemma 3's embedding scale. A family whose structure no
//     block covers adds blocks; it never changes another family's graph.
//
// It rewrites no weights. A file's converter has already put them in the
// convention the shared kernels expect — llama.cpp's stores Gemma 3's norm
// weights as 1 + w and permutes Llama's Q and K — so upload writes what the
// file holds.
//
// An entry is chosen once, at load. Its graph is built when preflight judges
// Run and again at load, for the program, and never per token, so the
// per-token path makes no call through this table (WASM.4). Preflight runs
// the same graph the load runs, not a check of its own beside it. Function
// pointers, not a pure abstract class: an architecture holds no state, and
// the capability table lists its entries as static data — the interface
// C.121 asks for, without objects to construct.

enum class DescribeError {
    Ok,
    MissingKey,
    WrongKeyType,
    MissingTensor,
    ShapeMismatch,
    // A value no architecture could run with: zero heads, a layer count larger
    // than the file's tensors could fill.
    InvalidValue,
    // A valid value in a form this implementation does not read.
    UnsupportedValue,
};

// The outcome of describing a file. `subject` names the key or tensor at
// fault, so a failure says where it is, not only what kind it is.
struct DescribeResult {
    DescribeError error = DescribeError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == DescribeError::Ok; }
};

using DescribeFn = DescribeResult (*)(const gguf::TensorIndex& index,
                                      model::ModelDescription& out);

// Fills `out` with every launch of a step. Preconditions: `model` is this
// architecture's describe's; `plan` was made from it, as upload carried it
// out (Upload::plan()); `cache_format` has a pack and is the format the plan
// sized the cache in.
using GraphFn = graph::GraphResult (*)(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                                       const formats::Format& cache_format, std::vector<kernels::Launch>& out);

struct Architecture {
    std::string_view name;
    DescribeFn describe;
    GraphFn graph;
};

}  // namespace bllm::arch
