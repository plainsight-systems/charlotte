#include "core/residency/routes.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "core/formats/format.h"
#include "core/gguf/checked.h"

namespace bllm::residency {
namespace {

using gguf::checked_mul;
using gguf::range_within;

RouteResult failure(RouteError error, std::string subject) { return {error, std::move(subject)}; }

std::uint64_t padded(std::uint64_t bytes) { return (bytes + 3) / 4 * 4; }

bool contains(std::span<const gguf::TensorId> ids, gguf::TensorId id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// The planned tensor for `id`; the plan holds one for every tensor the index
// lists, in its order.
const PlannedTensor* planned(const ResidencyPlan& plan, gguf::TensorId id) {
    for (const PlannedTensor& t : plan.tensors) {
        if (t.tensor == id) return &t;
    }
    return nullptr;
}

// Routes one tensor's pieces, each a run of whole rows, so whole blocks.
RouteResult route_tensor(const gguf::TensorEntry& entry, const PlannedTensor& planned_tensor,
                         const ResidencyPlan& plan, std::uint64_t file_size, FindFormat find_format,
                         std::vector<Route>& routes) {
    // A format not listed is refused whatever the tensor's size, so the
    // refusal does not depend on whether it has rows.
    const formats::Format* format = find_format(entry.type);
    if (format == nullptr) return failure(RouteError::UnsupportedFormat, entry.name);
    if (!formats::steps_by_groups(entry) && !formats::is_scalar(entry)) {
        return failure(RouteError::RowNotSteppable, entry.name);
    }
    const formats::DeviceLayout& layout = format->layout();

    const auto& pieces = planned_tensor.view.pieces();
    if (pieces.empty()) return {};   // no rows: nothing to write

    std::uint64_t rows = 0;
    for (const WeightPiece& piece : pieces) rows += piece.row_count;
    const std::uint64_t row_bytes = entry.data_length / rows;   // the reader checked rows are whole blocks
    for (const WeightPiece& piece : pieces) {
        std::uint64_t offset_in_tensor = 0;
        std::uint64_t bytes = 0;
        if (!checked_mul(piece.first_row, row_bytes, offset_in_tensor) ||
            !checked_mul(piece.row_count, row_bytes, bytes) ||
            !range_within(offset_in_tensor, bytes, entry.data_length) ||
            !range_within(entry.data_offset, offset_in_tensor, file_size) ||
            !range_within(entry.data_offset + offset_in_tensor, bytes, file_size)) {
            return failure(RouteError::OutOfRange, entry.name);
        }
        const auto buffer = static_cast<std::size_t>(piece.buffer);
        if (buffer >= plan.buffers.size() ||
            !range_within(piece.offset, piece.length, plan.buffers[buffer].size) ||
            piece.length != padded(bytes) || bytes % layout.block_bytes != 0) {
            return failure(RouteError::OutOfRange, entry.name);
        }
        routes.push_back({planned_tensor.tensor, entry.data_offset + offset_in_tensor,
                          bytes / layout.block_bytes, &layout, piece.buffer, piece.offset,
                          piece.length});
    }
    return {};
}

}  // namespace

RouteResult plan_routes(const gguf::TensorIndex& index, const ResidencyPlan& plan, std::uint64_t file_size,
                        FindFormat find_format, std::span<const gguf::TensorId> confirmed,
                        std::vector<Route>& routes, ResidencyPlan& out) {
    ResidencyPlan carried = plan;
    for (const gguf::TensorId id : confirmed) {
        // The ids cross from the page, so one no tensor has is named, never
        // looked up.
        if (static_cast<std::size_t>(id) >= index.tensors().size()) {
            return failure(RouteError::NotACandidate,
                           "tensor " + std::to_string(static_cast<std::uint32_t>(id)));
        }
        const PlannedTensor* t = planned(plan, id);
        if (t == nullptr || !t->candidate_duplicate_of) {
            return failure(RouteError::NotACandidate, index.tensor(id).name);
        }
    }

    std::vector<Route> found;
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        const PlannedTensor& t = plan.tensors[i];
        if (contains(confirmed, t.tensor)) {
            // A confirmed copy reads the tensor it copies, and its own buffers
            // — the plan gave it buffers no other tensor shares — go uncreated.
            const PlannedTensor* original = planned(plan, *t.candidate_duplicate_of);
            if (original == nullptr) return failure(RouteError::NotACandidate, index.tensor(t.tensor).name);
            for (const WeightPiece& piece : t.view.pieces()) {
                PlannedBuffer& buffer = carried.buffers[static_cast<std::size_t>(piece.buffer)];
                // Several pieces may share a buffer; it is subtracted once.
                carried.weight_bytes -= buffer.size;
                carried.total_bytes -= buffer.size;
                buffer.size = 0;
            }
            carried.tensors[i].view = original->view;
            continue;
        }
        if (auto r = route_tensor(index.tensor(t.tensor), t, plan, file_size, find_format, found); !r.ok()) {
            return r;
        }
    }
    std::sort(found.begin(), found.end(),
              [](const Route& a, const Route& b) { return a.file_offset < b.file_offset; });

    routes = std::move(found);
    out = std::move(carried);
    return {};
}

}  // namespace bllm::residency
