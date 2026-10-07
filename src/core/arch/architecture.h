#pragma once

#include <string>
#include <string_view>

#include "core/gguf/index.h"
#include "core/model/model_description.h"

namespace bllm::arch {

// Contract: what every architecture supplies.
//
// Each arch/<arch>/ provides one Architecture, and the capability table lists
// it under every general.architecture value it answers to. An architecture is
// the only code that knows which architecture is loaded. It supplies one
// thing:
//
//   - describe: reads its numbers from the tensor index into a model
//     description, names each layer's tensors by role, and states every way
//     its family differs — conventions, scales, per-layer windows and bases.
//     Fails naming the key or tensor when the file lacks what the
//     architecture needs.
//
// One graph serves every architecture, reading only the description
// (graph/graph.h).
//
// It rewrites no weights. A file's converter has already put them in the
// convention the shared kernels expect — llama.cpp's stores Gemma 3's norm
// weights as 1 + w and permutes Llama's Q and K — so upload writes what the
// file holds.
//
// An entry is chosen once, at load, and the per-token path makes no call
// through this table (WASM.4).

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

struct Architecture {
    std::string_view name;
    DescribeFn describe;
};

}  // namespace bllm::arch
