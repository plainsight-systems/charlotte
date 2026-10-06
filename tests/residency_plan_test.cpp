#include <doctest/doctest.h>

#include <algorithm>
#include <set>
#include <string>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/residency/plan.h"
#include "support/gguf_fixture.h"

using namespace bllm;
using residency::DeviceLimits;
using residency::PlanError;
using residency::Pool;
using residency::ResidencyPlan;

namespace {

// A tiny model, read and described, ready to plan. The bytes stay alive with
// the index that refers to them.
struct Planned {
    std::vector<std::byte> bytes;
    gguf::TensorIndex index;
    model::ModelDescription model;
};

Planned describe(const std::string& fixture) {
    Planned p;
    p.bytes = testing::load_gguf_fixture(fixture);
    gguf::MemoryByteSource source{p.bytes};
    REQUIRE(gguf::read_index(source, p.index).error == gguf::ReadError::Ok);
    std::string_view name;
    REQUIRE(p.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    REQUIRE(capability::find_architecture(name)->describe(p.index, p.model).ok());
    return p;
}

// WebGPU's defaults.
constexpr DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};

// The invariants every plan keeps, whatever the limits.
void check_invariants(const ResidencyPlan& plan, const DeviceLimits& limits) {
    for (const auto& b : plan.buffers) CHECK(b.size <= limits.max_buffer_size);
    auto check_range = [&](const residency::BufferRange& r, Pool pool) {
        REQUIRE(static_cast<std::size_t>(r.buffer) < plan.buffers.size());
        const auto& buffer = plan.buffers[static_cast<std::size_t>(r.buffer)];
        CHECK(buffer.pool == pool);
        CHECK(r.offset % limits.storage_offset_alignment == 0);
        CHECK(r.length <= limits.max_storage_binding_size);
        CHECK(r.offset + r.length <= buffer.size);
    };
    for (const auto& t : plan.tensors) {
        std::uint64_t next_row = 0;
        for (const auto& piece : t.view.pieces()) {
            check_range({piece.buffer, piece.offset, piece.length}, Pool::Weights);
            CHECK(piece.first_row == next_row);   // pieces cover the rows, in order
            next_row += piece.row_count;
        }
    }
    for (const auto& layer : plan.cache) {
        check_range(layer.keys, Pool::Cache);
        check_range(layer.values, Pool::Cache);
    }
    for (const auto& s : plan.scratch) {
        check_range(s.range, Pool::Scratch);
        // Each working buffer alone in its buffer: WebGPU refuses one buffer
        // bound writable and read-only in a dispatch, which a kernel reading
        // one working buffer and writing another would otherwise do.
        CHECK(s.range.offset == 0);
        CHECK(std::count_if(plan.scratch.begin(), plan.scratch.end(),
                            [&](const auto& other) { return other.range.buffer == s.range.buffer; }) == 1);
    }
    CHECK(plan.total_bytes == plan.weight_bytes + plan.cache_bytes + plan.scratch_bytes);
}

// The bytes of the file a tensor's pieces hold: their rows times a row's bytes.
// Each piece binds those bytes rounded up to 4.
std::uint64_t file_bytes(const residency::PlannedTensor& tensor, const gguf::TensorEntry& entry) {
    std::uint64_t rows = 0;
    for (const auto& piece : tensor.view.pieces()) rows += piece.row_count;
    if (rows == 0) return 0;
    const std::uint64_t row_bytes = entry.data_length / rows;
    std::uint64_t bytes = 0;
    for (const auto& piece : tensor.view.pieces()) {
        const std::uint64_t held = piece.row_count * row_bytes;
        CHECK(piece.length == (held + 3) / 4 * 4);
        bytes += held;
    }
    return bytes;
}

}  // namespace

TEST_CASE("every tensor is placed, whole, and within the limits") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    check_invariants(plan, kDefaults);

    REQUIRE(plan.tensors.size() == p.index.tensors().size());
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        const auto& entry = p.index.tensors()[i];
        CHECK(file_bytes(plan.tensors[i], entry) == entry.data_length);
        CHECK(plan.tensors[i].view.format() == entry.type);
    }
    // A model this small packs into one weight buffer.
    CHECK(std::count_if(plan.buffers.begin(), plan.buffers.end(),
                        [](const auto& b) { return b.pool == Pool::Weights; }) == 1);
}

TEST_CASE("a weight wider than a binding is split by whole rows") {
    const auto p = describe("tiny_qwen3");
    // token_embd is 6 rows of 128 bytes: a 256-byte binding takes two rows.
    const DeviceLimits narrow{65536, 256, 32};
    // Room for the working buffers is beside the point here: give them none.
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, narrow, policy::LoadPolicy{}, plan);
    CHECK(r.error == PlanError::ScratchExceedsBinding);   // weights placed first
    const auto embd = static_cast<std::size_t>(*p.index.find("token_embd.weight"));
    REQUIRE(plan.tensors.size() > embd);
    const auto& pieces = plan.tensors[embd].view.pieces();
    REQUIRE(pieces.size() == 3);
    for (const auto& piece : pieces) {
        CHECK(piece.row_count == 2);
        CHECK(piece.length == 256);
        CHECK(piece.offset % narrow.storage_offset_alignment == 0);
    }
}

TEST_CASE("a piece whose bytes are no multiple of 4 binds them rounded up to 4") {
    const auto p = describe("tiny_qwen3_odd_blocks");
    const auto extra = static_cast<std::size_t>(*p.index.find("extra.weight"));
    // Whole: 3 rows of 18 bytes, 54, bound as 56.
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    check_invariants(plan, kDefaults);
    REQUIRE(plan.tensors[extra].view.pieces().size() == 1);
    CHECK(plan.tensors[extra].view.pieces()[0].length == 56);
    // Split two rows to a 48-byte binding: 36 bound as 36, then 18 bound as 20.
    const DeviceLimits narrow{4096, 48, 32};
    ResidencyPlan split;
    (void)residency::plan_residency(p.index, p.model, narrow, policy::LoadPolicy{}, split);
    REQUIRE(split.tensors.size() > extra);
    const auto& pieces = split.tensors[extra].view.pieces();
    REQUIRE(pieces.size() == 2);
    CHECK(pieces[0].length == 36);
    CHECK(pieces[1].length == 20);
    CHECK(file_bytes(split.tensors[extra], p.index.tensor(static_cast<gguf::TensorId>(extra))) == 54);
}

TEST_CASE("a row wider than a binding cannot be placed, and is named") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, DeviceLimits{4096, 16, 32},
                                             policy::LoadPolicy{}, plan);
    CHECK(r.error == PlanError::RowExceedsBinding);
    CHECK(r.subject == "token_embd.weight");
}

TEST_CASE("a working buffer wider than a binding is named") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    // Weights fit 1 KiB bindings; the 512-token hidden buffer (16 KiB) does not.
    const auto r = residency::plan_residency(p.index, p.model, DeviceLimits{1 << 20, 1024, 32},
                                             policy::LoadPolicy{}, plan);
    CHECK(r.error == PlanError::ScratchExceedsBinding);
    CHECK(r.subject == "hidden");
}

TEST_CASE("the context offered is capped by what the model was trained for") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    CHECK(plan.context_offered == p.model.trained_context);
    REQUIRE(plan.cache.size() == p.model.layers.size());
    // 1 key/value head of width 32 at f16: 64 bytes a token, for keys and for values.
    CHECK(plan.cache[0].keys.length == std::uint64_t{plan.context_offered} * 64);
    for (const auto& layer : plan.cache) CHECK(layer.slots == plan.context_offered);   // full attention
}

namespace {

// Gemma's pattern over seven layers: a window layer, except the sixth.
bool global_layer(std::size_t layer) { return layer == 5; }

// The slots a window layer's ring holds: the fixture's window of 16, a
// prefill block, and the default rollback reserve.
constexpr std::uint32_t kRing = 16 + residency::kPrefillBlock + 4096;

}  // namespace

TEST_CASE("a sliding-window layer's cache is a ring of its window, a prefill block and the reserve") {
    const auto p = describe("tiny_gemma3_long");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    check_invariants(plan, kDefaults);
    CHECK(plan.context_offered == 8192);
    REQUIRE(plan.cache.size() == 7);
    for (std::size_t i = 0; i < plan.cache.size(); ++i) {
        const std::uint32_t slots = global_layer(i) ? 8192 : kRing;
        CHECK(plan.cache[i].slots == slots);
        CHECK(plan.cache[i].keys.length == std::uint64_t{slots} * 64);
        CHECK(plan.cache[i].values.length == std::uint64_t{slots} * 64);
    }
}

TEST_CASE("a ring that would reach the context offered is the context offered") {
    const auto p = describe("tiny_gemma3_long");
    policy::LoadPolicy generous{};
    generous.rollback_reserve = 8192;
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, generous, plan).ok());
    for (const auto& layer : plan.cache) CHECK(layer.slots == plan.context_offered);
}

TEST_CASE("only full-attention layers cost the budget a token at a time") {
    const auto p = describe("tiny_gemma3_long");
    ResidencyPlan roomy;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, roomy).ok());

    // Past the ring, one token costs the one global layer 128 bytes (64 of
    // keys, 64 of values). Take away 1,000 tokens' worth at that rate; had
    // every layer been full length, the same bytes would be 143 tokens of seven.
    policy::LoadPolicy tight{};
    tight.memory_budget = roomy.total_bytes - 128 * 1000;
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, tight, plan).ok());
    check_invariants(plan, kDefaults);
    CHECK(plan.total_bytes <= tight.memory_budget);
    CHECK(plan.context_offered <= 8192 - 1000);
    // The planner reserves up to 260 bytes of alignment padding per range,
    // 14 ranges, 3,640 bytes: at most 29 tokens of the global layer.
    CHECK(plan.context_offered >= 8192 - 1000 - 29);
    for (std::size_t i = 0; i < plan.cache.size(); ++i) {
        if (!global_layer(i)) CHECK(plan.cache[i].slots == kRing);
    }
}

TEST_CASE("a model whose every layer uses a window is offered its trained context") {
    const auto p = describe("tiny_gemma3_long_all_window");
    ResidencyPlan roomy;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, roomy).ok());
    for (const auto& layer : roomy.cache) CHECK(layer.slots == kRing);

    // Full-length caches would need 7 layers x 16 bytes x (8192 - kRing)
    // more than the rings: 399,616 bytes. A budget 100,000 bytes over the
    // rings still offers the whole trained context.
    policy::LoadPolicy tight{};
    tight.memory_budget = roomy.total_bytes + 100'000;
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, tight, plan).ok());
    check_invariants(plan, kDefaults);
    CHECK(plan.context_offered == 8192);
}

TEST_CASE("the context offered is the most the budget allows, and fits within it") {
    auto p = describe("tiny_qwen3");
    // The fixture trains for 64 tokens, below a prefill block; a longer
    // trained context lets the budget, not the model, set the limit.
    p.model.trained_context = 4096;
    ResidencyPlan roomy;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, roomy).ok());
    REQUIRE(roomy.context_offered == 4096);

    // 2 layers, keys and values, 64 bytes a token each: 256 bytes a token.
    // Take away 1,000 tokens' worth.
    policy::LoadPolicy tight{};
    tight.memory_budget = roomy.total_bytes - 256 * 1000;
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, tight, plan).ok());
    check_invariants(plan, kDefaults);
    CHECK(plan.total_bytes <= tight.memory_budget);
    CHECK(plan.context_offered <= 4096 - 1000);
    CHECK(plan.context_offered >= residency::kPrefillBlock);
}

TEST_CASE("a budget too small for the shortest context says what it needs") {
    const auto p = describe("tiny_qwen3");
    policy::LoadPolicy tiny{};
    tiny.memory_budget = 4096;
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, kDefaults, tiny, plan);
    CHECK(r.error == PlanError::ExceedsBudget);
    CHECK(plan.total_bytes > tiny.memory_budget);
}

TEST_CASE("a cache precision that cannot store the head dimension is named") {
    auto p = describe("tiny_qwen3");
    p.model.layers[0].head_dimension = 48;   // not a whole number of Q8_0's 32-wide blocks
    policy::LoadPolicy q8{};
    q8.cache_precision = policy::CachePrecision::Q8_0;
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, kDefaults, q8, plan);
    CHECK(r.error == PlanError::UnsupportedCachePrecision);
    CHECK(r.subject == "layer 0");
}

TEST_CASE("an output head stored as a copy of the embedding gets buffers of its own") {
    const auto p = describe("tiny_qwen3_output_copy");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    check_invariants(plan, kDefaults);

    const auto head = static_cast<std::size_t>(*p.model.output_head);
    REQUIRE(plan.tensors[head].candidate_duplicate_of == p.model.token_embedding);
    std::set<residency::BufferIndex> head_buffers;
    for (const auto& piece : plan.tensors[head].view.pieces()) head_buffers.insert(piece.buffer);
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        if (i == head) continue;
        for (const auto& piece : plan.tensors[i].view.pieces()) {
            CHECK(head_buffers.count(piece.buffer) == 0);   // nothing else shares them
        }
        CHECK_FALSE(plan.tensors[i].candidate_duplicate_of.has_value());
    }
}
