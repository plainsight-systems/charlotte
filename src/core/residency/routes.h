#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/formats/format.h"
#include "core/gguf/index.h"
#include "core/residency/plan.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Where every byte of the file goes. A deterministic function, with no I/O, of
// the tensor index, the residency plan and the duplicates the page has
// confirmed: one route for each piece of each weight, in file order, saying which bytes of the file
// it takes and where in which buffer they land. No GPU, no browser; upload
// carries the routes out (upload.h), and they are tested without a device.
//
//   - A route takes a run of whole rows, so whole blocks, as the plan's piece
//     does, and lays them out as its format's device layout says
//     (formats/device_layout.h). Its destination length is the piece's bound
//     length, which that layout fills exactly.
//   - A tensor's format is found through the lookup the caller passes: the
//     capability table's find_format in the harness, so the formats upload
//     routes are exactly the ones preflight passed; a table of the test's own
//     in a test. A format it does not list is a named failure.
//   - Every offset and length is checked against the file's size before any
//     byte arrives, by subtraction so no sum can wrap (WASM.9, ES.103). A
//     route that would read past the file, or write past its buffer, is a
//     named failure, not a write.
//   - A confirmed duplicate takes no route: its bytes go nowhere, its view
//     reads the tensor it copies, and its buffers are never created. Which
//     duplicates are confirmed is decided before upload, by comparing the two
//     tensors' bytes in the cached file (web/duplicates.js); a candidate that
//     is not confirmed is routed like any tensor, into the buffers the plan
//     gave it.
//     Optimization (practice): a head stored as a copy of the embedding is
//     the same tied weight written twice. Sharing it keeps one copy on the
//     device, as a model saved with tied weights does; the plan never counts
//     on it (plan.h).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     E.27   If you can't throw exceptions, use error codes systematically —
//            RouteError, with the tensor at fault as its subject.
//     ES.103 Don't overflow — ranges are checked by subtraction.
//     F.20, F.21 prefer returning values to out parameters. Deliberately not
//            followed: out parameters match plan_residency and the loaders,
//            and leave the caller's objects untouched unless the call
//            succeeds.
//   C++ performance guidelines
//     WASM.9 Stream assets and keep them compressed to their destination —
//            every offset is validated against the file's authoritative size,
//            with checked arithmetic, before any byte arrives.

// Finds the format a tensor type runs as, or null if it runs as none.
using FindFormat = const formats::Format* (*)(gguf::TensorType type) noexcept;

struct Route {
    gguf::TensorId tensor;       // whose piece it is
    std::uint64_t file_offset;   // where its blocks start in the file
    std::uint64_t blocks;        // how many whole blocks it takes
    const formats::DeviceLayout* layout;
    BufferIndex buffer;
    std::uint64_t buffer_offset;   // where the piece starts in its buffer
    std::uint64_t length;          // the piece's bound length: its bytes, rounded up to 4
};

enum class RouteError {
    Ok,
    // A tensor whose format the lookup does not list. The subject names it.
    UnsupportedFormat,
    // A piece that would read past the end of the file, or write past the
    // end of its buffer. The subject names the tensor.
    OutOfRange,
    // A confirmed duplicate the plan does not mark a candidate, or an id no
    // tensor has. The subject names it, or gives the id.
    NotACandidate,
    // A tensor whose rows are not a whole number of unpack's 32-weight
    // groups, which no kernel can step through, and that is not a scalar
    // (format.h). The subject names it.
    RowNotSteppable,
};

struct RouteResult {
    RouteError error = RouteError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == RouteError::Ok; }
};

// The routes for every weight in `plan`, ordered by file offset, and the plan
// as upload carries it out: each confirmed duplicate's view reads the tensor
// it copies, the buffers only it used are left uncreated (size 0), and the
// weight and total bytes are recounted without them. `out` and `routes` are
// left untouched unless it succeeds. A tensor the plan marks not present has
// no route; every route lies within `ranges`, the bytes the load reads
// (file_ranges.h), or the tensor is refused, OutOfRange. Empty: the whole
// file.
[[nodiscard]] RouteResult plan_routes(const gguf::TensorIndex& index, const ResidencyPlan& plan,
                                      std::uint64_t file_size, FindFormat find_format,
                                      std::span<const gguf::TensorId> confirmed,
                                      std::vector<Route>& routes, ResidencyPlan& out,
                                      std::span<const FileRange> ranges = {});

}  // namespace bllm::residency
