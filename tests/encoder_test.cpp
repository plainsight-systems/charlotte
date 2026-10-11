// EmbeddingGemma 2 on the CPU, from its pinned header (tests/fixtures/
// tokenizer/manifest.json): what describe reads (arch/embedding_gemma2), the
// encoder's working set and its plan, its graph's launches and the 296 a
// step dispatches (graph/encoder.h), and preflight's verdict.

#include <doctest/doctest.h>

#include <array>
#include <utility>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/graph/encoder.h"
#include "core/kernels/interface.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "support/model_headers.h"

using namespace bllm;
using model::Role;

namespace {

constexpr residency::DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};
constexpr std::string_view kModel = "embeddinggemma-2-q8_0";

struct Loaded {
    testing::ReadHeader header;
    const arch::Architecture* architecture = nullptr;
    model::ModelDescription model{};
    residency::ResidencyPlan plan;
};

Loaded load() {
    Loaded l{testing::read_model_header(kModel)};
    std::string_view name;
    REQUIRE(l.header.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    l.architecture = capability::find_architecture(name);
    REQUIRE(l.architecture != nullptr);
    const auto described = l.architecture->describe(l.header.index, l.model);
    REQUIRE_MESSAGE(described.ok(), described.subject);
    const auto planned = residency::plan_residency(l.header.index, l.model, kDefaults, policy::LoadPolicy{},
                                                   graph::encoder_working_set(l.model), l.plan);
    REQUIRE_MESSAGE(planned.ok(), planned.subject);
    return l;
}

std::uint64_t workgroups(const kernels::Launch& l, std::uint32_t tokens) {
    return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens,
                                    l.partial_rows},
                                   l.workgroup_size, 0, tokens, true);
}

std::size_t dispatched(const std::vector<kernels::Launch>& launches, std::uint32_t tokens) {
    return static_cast<std::size_t>(
        std::count_if(launches.begin(), launches.end(), [&](const auto& l) { return workgroups(l, tokens) > 0; }));
}

}  // namespace

TEST_CASE("EmbeddingGemma 2 is listed as an architecture that encodes") {
    const arch::Architecture* a = capability::find_architecture("gemma-embedding2");
    REQUIRE(a != nullptr);
    CHECK(arch::encoder_graph(*a) != nullptr);
    CHECK(arch::decoder_graph(*a) == nullptr);
}

TEST_CASE("describe reads EmbeddingGemma 2's layers, two kinds of head, and its encoder") {
    const Loaded l = load();
    const model::ModelDescription& m = l.model;
    CHECK(m.embedding_width == 512);
    CHECK(m.vocabulary_size == 262144);
    REQUIRE(m.layers.size() == 24);
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
        CAPTURE(i);
        const model::LayerDescription& layer = m.layers[i];
        const bool global = i % 6 == 5;
        CHECK(layer.query_heads == 4);
        CHECK(layer.key_value_heads == (global ? 1u : 2u));
        CHECK(layer.head_dimension == (global ? 512u : 256u));
        CHECK(layer.attention_window == (global ? 262144u : 1024u));
        CHECK(layer.rope_base == (global ? 1e6f : 1e4f));
        CHECK(layer.feed_forward_width == 2048);
        for (const Role r : {Role::QueryNorm, Role::KeyNorm, Role::PostAttentionNorm, Role::PostFeedForwardNorm,
                             Role::PerLayerGate, Role::PerLayerProjection, Role::PerLayerPostNorm,
                             Role::OutputScale}) {
            CHECK(layer.tensors[static_cast<std::size_t>(r)].has_value());
        }
    }
    CHECK(m.mask == model::AttentionMask::Bidirectional);
    CHECK(m.value_norm);
    CHECK(m.attention_scale == 1.0f);
    CHECK(m.rotary_pairing == model::RotaryPairing::Halves);
    CHECK(m.activation == model::FeedForwardActivation::GeluTanh);
    CHECK(!m.output_head.has_value());
    REQUIRE(m.encoder.has_value());
    CHECK(m.encoder->per_layer_width == 512);
    CHECK(m.encoder->embedding_length == 768);
    CHECK(m.encoder->pooling == model::Pooling::Mean);
    CHECK(l.header.index.tensor(m.encoder->output_projection).name == "output.weight");
    CHECK(l.header.index.tensor(m.encoder->per_layer_projection).type == gguf::TensorType::BF16);
}

TEST_CASE("an encoder's plan keeps no cache, offers one step's 512 tokens, and fits the default budget") {
    const Loaded l = load();
    CHECK(l.plan.cache.empty());
    CHECK(l.plan.cache_bytes == 0);
    CHECK(l.plan.context_offered == residency::kPrefillBlock);
    CHECK(l.plan.total_bytes <= policy::LoadPolicy{}.memory_budget);
    std::vector<std::string_view> purposes;
    for (const auto& s : l.plan.scratch) purposes.push_back(s.purpose);
    CHECK(purposes == std::vector<std::string_view>{"hidden", "normed", "query", "key", "value", "attention",
                                                    "output", "activation", "per_layer_raw", "per_layer", "scores",
                                                    "score_maxima", "pooled", "projected", "embedding",
                                                    "range_sums"});
    // The most any product's range sums need: the down and global output
    // projections at 512 tokens, 512 rows, 32 ranges, 32 MiB.
    CHECK(l.plan.scratch.back().range.length == 512ull * 512 * 32 * 4);
    // A phone's budget holds it too (web/platform_policy.js): 512 MiB.
    policy::LoadPolicy phone{};
    phone.memory_budget = 512ull << 20;
    residency::ResidencyPlan plan;
    CHECK(residency::plan_residency(l.header.index, l.model, kDefaults, phone, graph::encoder_working_set(l.model),
                                    plan)
              .ok());
}

TEST_CASE("an encoder's step dispatches 296 launches, 12 a layer, and a combine for each product split") {
    const Loaded l = load();
    std::vector<kernels::Launch> launches;
    const auto built = arch::encoder_graph(*l.architecture)(l.model, l.plan, launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);
    // The combines (kernels/matmul/matmul.h): to 32 tokens a sliding layer's
    // Q, K and V, output, down, per-layer gate and projection, 5, and a global
    // layer's 4; to 128 the per-layer pair and the outputs and downs; to 256
    // the sliding outputs and every down; to 512 the global outputs and every
    // down. 20 sliding layers, 4 global.
    const std::array<std::pair<std::uint32_t, std::size_t>, 4> steps{
        {{2, 296 + 20 * 5 + 4 * 4}, {16, 296 + 20 * 5 + 4 * 4}, {33, 296 + 24 * 4}, {512, 296 + 24 + 4}}};
    for (const auto& [tokens, want] : steps) {
        CAPTURE(tokens);
        CHECK(dispatched(launches, tokens) == want);
        // No dispatch past WebGPU's 65,535 workgroups.
        for (const kernels::Launch& launch : launches) CHECK(workgroups(launch, tokens) <= 65535);
    }
    // The roles a layer's dispatches carry, in order, at 512 tokens: its 12,
    // and the down projection's combine beside it, the sliding output's
    // range past it.
    std::vector<std::string_view> layer0;
    for (const kernels::Launch& launch : launches) {
        if (launch.layer == 0 && workgroups(launch, 512) > 0) layer0.push_back(launch.role);
    }
    CHECK(layer0 == std::vector<std::string_view>{"attention.norm", "attention.qkv", "attention.rope",
                                                  "attention.scores", "attention.weigh", "attention.output",
                                                  "ffn.norm", "ffn.gate_up", "ffn.down", "ffn.down",
                                                  "per_layer.add", "per_layer.gate", "per_layer.project"});
    std::vector<std::string_view> outside;
    for (const kernels::Launch& launch : launches) {
        if (launch.layer == kernels::kNoLayer && workgroups(launch, 512) > 0) outside.push_back(launch.role);
    }
    CHECK(outside == std::vector<std::string_view>{"embed", "embed", "per_layer.inputs", "per_layer.inputs_norm",
                                                   "output.norm", "output.pool", "output.project", "output.unit"});
    // The per-layer inputs' norm: ε folded with the projection's 1 / √512.
    for (const kernels::Launch& launch : launches) {
        if (launch.role != "per_layer.inputs_norm") continue;
        float epsilon = 0;
        std::memcpy(&epsilon, launch.constants.data(), sizeof epsilon);
        CHECK(epsilon == l.model.norm_epsilon * 512.0f);
    }
}

TEST_CASE("preflight takes EmbeddingGemma 2 to Run, its tokenizer and its ends resolved") {
    const auto header = testing::read_model_header(kModel);
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    const preflight::Verdict verdict = preflight::preflight(source, header.index, kDefaults, policy::LoadPolicy{});
    for (const auto& b : verdict.blockers) MESSAGE(preflight::to_string(b.stage) << ": " << b.detail);
    CHECK(verdict.blockers.empty());
    CHECK(verdict.reached() == preflight::Stage::Run);
    REQUIRE(verdict.fit.has_value());
    CHECK(verdict.fit->cache_bytes == 0);
    CHECK(verdict.fit->context_offered == 512);
}
