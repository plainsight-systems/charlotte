#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/formats/format.h"
#include "core/gguf/reader.h"
#include "core/graph/graph.h"
#include "core/residency/plan.h"
#include "core/residency/routes.h"
#include "support/gguf_fixture.h"

using namespace bllm;
using residency::DeviceLimits;
using residency::ResidencyPlan;
using residency::Route;
using residency::RouteError;

namespace {

// Formats of the test's own, for the fixtures' F32 and Q4_0 tensors. The
// capability table lists none until a format's unpack exists.
inline constexpr formats::Format kF32{formats::kF32Layout, "fn unpack_f32() {}"};
inline constexpr formats::Format kQ4_0{formats::kQ4_0Layout, "fn unpack_q4_0() {}"};

const formats::Format* both(gguf::TensorType type) noexcept {
    if (type == gguf::TensorType::F32) return &kF32;
    if (type == gguf::TensorType::Q4_0) return &kQ4_0;
    return nullptr;
}

const formats::Format* f32_only(gguf::TensorType type) noexcept {
    return type == gguf::TensorType::F32 ? &kF32 : nullptr;
}

struct Planned {
    std::vector<std::byte> bytes;
    gguf::TensorIndex index;
    model::ModelDescription model;
    ResidencyPlan plan;
};

constexpr DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};

Planned planned(const std::string& fixture, const DeviceLimits& limits = kDefaults) {
    Planned p;
    p.bytes = testing::load_gguf_fixture(fixture);
    gguf::MemoryByteSource source{p.bytes};
    REQUIRE(gguf::read_index(source, p.index).error == gguf::ReadError::Ok);
    std::string_view name;
    REQUIRE(p.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    REQUIRE(capability::find_architecture(name)->describe(p.index, p.model).ok());
    (void)residency::plan_residency(p.index, p.model, limits, policy::LoadPolicy{},
                                    graph::decoder_working_set(p.model), p.plan);
    return p;
}

std::uint64_t file_bytes(const Route& r) { return r.blocks * r.layout->block_bytes; }

std::uint64_t pool_bytes(const ResidencyPlan& plan, residency::Pool pool) {
    std::uint64_t sum = 0;
    for (const auto& b : plan.buffers) {
        if (b.pool == pool) sum += b.size;
    }
    return sum;
}

}  // namespace

TEST_CASE("every byte of every tensor is routed once, in file order, inside its buffer") {
    for (const char* fixture : {"tiny_qwen3", "tiny_qwen3_odd_blocks"}) {
        CAPTURE(fixture);
        const auto p = planned(fixture);
        std::vector<Route> routes;
        ResidencyPlan carried;
        const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size(), both, {}, routes, carried);
        REQUIRE_MESSAGE(r.ok(), r.subject);

        std::uint64_t routed = 0;
        std::uint64_t tensors = 0;
        for (const auto& t : p.index.tensors()) tensors += t.data_length;
        for (std::size_t i = 0; i < routes.size(); ++i) {
            const Route& route = routes[i];
            routed += file_bytes(route);
            if (i > 0) CHECK(routes[i - 1].file_offset + file_bytes(routes[i - 1]) <= route.file_offset);
            CHECK(route.length == (file_bytes(route) + 3) / 4 * 4);
            const auto& buffer = p.plan.buffers[static_cast<std::size_t>(route.buffer)];
            CHECK(route.buffer_offset + route.length <= buffer.size);
            CHECK(route.file_offset + file_bytes(route) <= p.bytes.size());
        }
        CHECK(routed == tensors);
    }
}

TEST_CASE("a weight split by rows takes one route per piece, each its rows' blocks") {
    // extra.weight: three rows of one Q4_0 block, 18 bytes a row; a 48-byte
    // binding takes two rows, then one.
    const auto p = planned("tiny_qwen3_odd_blocks", DeviceLimits{4096, 48, 32});
    const auto extra = *p.index.find("extra.weight");
    // The narrow plan stops at a wider row later in the file; route only
    // what it placed, the extra tensor among it.
    ResidencyPlan only_extra = p.plan;
    only_extra.tensors.erase(
        std::remove_if(only_extra.tensors.begin(), only_extra.tensors.end(),
                       [&](const residency::PlannedTensor& t) { return t.tensor != extra; }),
        only_extra.tensors.end());
    std::vector<Route> routes;
    ResidencyPlan carried;
    REQUIRE(residency::plan_routes(p.index, only_extra, p.bytes.size(), both, {}, routes, carried).ok());
    REQUIRE(routes.size() == 2);
    const std::uint64_t start = p.index.tensor(extra).data_offset;
    CHECK(routes[0].file_offset == start);
    CHECK(routes[0].blocks == 2);
    CHECK(routes[0].length == 36);
    CHECK(routes[1].file_offset == start + 36);
    CHECK(routes[1].blocks == 1);
    CHECK(routes[1].length == 20);
    CHECK(routes[0].layout == &formats::kQ4_0Layout);
}

TEST_CASE("a tensor whose format is not listed is refused by name, and nothing is changed") {
    const auto p = planned("tiny_qwen3_odd_blocks");
    std::vector<Route> routes{Route{}};
    ResidencyPlan carried;
    const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size(), f32_only, {}, routes, carried);
    CHECK(r.error == RouteError::UnsupportedFormat);
    CHECK(r.subject == "extra.weight");
    CHECK(routes.size() == 1);   // untouched
    CHECK(carried.tensors.empty());
}

TEST_CASE("a file too short for its tensors is refused before any byte arrives") {
    const auto p = planned("tiny_qwen3");
    std::vector<Route> routes;
    ResidencyPlan carried;
    const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size() / 2, both, {}, routes, carried);
    CHECK(r.error == RouteError::OutOfRange);
    CHECK_FALSE(r.subject.empty());
}

TEST_CASE("a confirmed copy is not routed: it reads what it copies, and its buffers go uncreated") {
    const auto p = planned("tiny_qwen3_output_copy");
    const auto head = *p.model.output_head;
    const auto embedding = *p.model.token_embedding;

    // Unconfirmed, the candidate is routed like any tensor.
    std::vector<Route> routes;
    ResidencyPlan carried;
    REQUIRE(residency::plan_routes(p.index, p.plan, p.bytes.size(), both, {}, routes, carried).ok());
    const auto routed = [&](gguf::TensorId id) {
        return std::count_if(routes.begin(), routes.end(), [&](const Route& r) { return r.tensor == id; });
    };
    CHECK(routed(head) > 0);

    // Confirmed, it is not, and its view is the embedding's.
    const gguf::TensorId confirmed[] = {head};
    REQUIRE(residency::plan_routes(p.index, p.plan, p.bytes.size(), both, confirmed, routes, carried).ok());
    CHECK(routed(head) == 0);
    CHECK(routed(embedding) > 0);
    const auto& head_view = carried.tensors[static_cast<std::size_t>(head)].view;
    const auto& embedding_view = carried.tensors[static_cast<std::size_t>(embedding)].view;
    REQUIRE(head_view.pieces().size() == embedding_view.pieces().size());
    for (std::size_t i = 0; i < head_view.pieces().size(); ++i) {
        CHECK(head_view.pieces()[i].buffer == embedding_view.pieces()[i].buffer);
        CHECK(head_view.pieces()[i].offset == embedding_view.pieces()[i].offset);
    }
    for (const auto& piece : p.plan.tensors[static_cast<std::size_t>(head)].view.pieces()) {
        CHECK(carried.buffers[static_cast<std::size_t>(piece.buffer)].size == 0);
    }

    // The carried plan's totals describe its buffers: smaller by the copy's.
    CHECK(carried.weight_bytes < p.plan.weight_bytes);
    CHECK(carried.weight_bytes == pool_bytes(carried, residency::Pool::Weights));
    CHECK(carried.total_bytes == carried.weight_bytes + carried.cache_bytes + carried.scratch_bytes);
}

TEST_CASE("confirming a tensor the plan never marked a candidate is refused by name") {
    const auto p = planned("tiny_qwen3_output_copy");
    const gguf::TensorId confirmed[] = {*p.model.token_embedding};
    std::vector<Route> routes;
    ResidencyPlan carried;
    const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size(), both, confirmed, routes, carried);
    CHECK(r.error == RouteError::NotACandidate);
    CHECK(r.subject == "token_embd.weight");
}

TEST_CASE("a confirmed id no tensor has is refused by number, and nothing is changed") {
    const auto p = planned("tiny_qwen3_output_copy");
    const gguf::TensorId confirmed[] = {gguf::TensorId{9999}};
    std::vector<Route> routes{Route{}};
    ResidencyPlan carried;
    carried.context_offered = 7;   // a mark the call must leave
    const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size(), both, confirmed, routes, carried);
    CHECK(r.error == RouteError::NotACandidate);
    CHECK(r.subject == "tensor 9999");
    CHECK(routes.size() == 1);
    CHECK(carried.context_offered == 7);
    CHECK(carried.tensors.empty());
}

TEST_CASE("a tensor with no rows is still refused when its format is not listed") {
    const auto p = planned("tiny_qwen3_odd_blocks");
    const auto extra = *p.index.find("extra.weight");
    ResidencyPlan rowless = p.plan;
    auto& t = rowless.tensors[static_cast<std::size_t>(extra)];
    t.view = residency::WeightView(t.view.format(), t.view.shape(), {});
    std::vector<Route> routes;
    ResidencyPlan carried;
    const auto r = residency::plan_routes(p.index, rowless, p.bytes.size(), f32_only, {}, routes, carried);
    CHECK(r.error == RouteError::UnsupportedFormat);
    CHECK(r.subject == "extra.weight");
}

TEST_CASE("a tensor whose rows are not a whole number of 32-weight groups is refused by name") {
    const auto p = planned("tiny_qwen3_odd_row");
    std::vector<Route> routes;
    ResidencyPlan carried;
    const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size(), both, {}, routes, carried);
    CHECK(r.error == RouteError::RowNotSteppable);
    CHECK(r.subject == "extra.norm");
}

TEST_CASE("a scalar, one F32 value, is routed whole though its row is no group") {
    const auto p = planned("tiny_qwen3_scalar");
    std::vector<Route> routes;
    ResidencyPlan carried;
    const auto r = residency::plan_routes(p.index, p.plan, p.bytes.size(), both, {}, routes, carried);
    REQUIRE(r.ok());
    const auto scale = *p.index.find("extra.scale");
    const auto& piece = carried.tensors[static_cast<std::size_t>(scale)].view.pieces();
    REQUIRE(piece.size() == 1);
    CHECK(piece.front().row_count == 1);
    CHECK(piece.front().length == 4);
}
