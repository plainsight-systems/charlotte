// EmbeddingGemma 2's vision encoder on the CPU, from its pinned mmproj's
// header (tests/fixtures/external.json): what describe reads
// (arch/gemma4v), whether a text model reads its rows, and its plan against
// the vision span the load reads (residency/file_ranges.h) — the vision
// tensors placed and routed, the audio encoder's never.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <utility>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/arch/gemma4v/gemma4v.h"
#include "core/capability/capability.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/reader.h"
#include "core/graph/vision.h"
#include "core/media/image.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/routes.h"
#include "support/model_headers.h"

using namespace bllm;
using model::Role;

namespace {

constexpr residency::DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};
constexpr std::uint64_t kFileSize = 554'821'024;
// The header and the vision span, as the catalog pins them
// (docs/research/2026-10-11-embeddinggemma2-vision.md).
constexpr residency::FileRange kVisionRanges[] = {{0, 58'656}, {328'492'448, kFileSize}};
constexpr std::uint32_t kRows = 280 * 9;

struct Header {
    std::string bytes;
    gguf::TensorIndex index;
};

Header read_mmproj() {
    const std::string path = std::string(BLLM_TEST_DATA_DIR) + "/models/mmproj-embeddinggemma-2-q8_0.gguf";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing test data: " << path << " (run make test-data)");
    Header h{std::string(64 * 1024, '\0'), {}};
    in.read(h.bytes.data(), static_cast<std::streamsize>(h.bytes.size()));
    REQUIRE(in.gcount() == static_cast<std::streamsize>(h.bytes.size()));
    gguf::MemoryByteSource source{std::as_bytes(std::span{h.bytes}), kFileSize};
    REQUIRE(gguf::read_index(source, h.index).error == gguf::ReadError::Ok);
    return h;
}

model::ModelDescription described(const gguf::TensorIndex& index) {
    model::ModelDescription m{};
    const auto r = arch::kGemma4Vision.describe(index, m);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return m;
}

bool is_vision(std::string_view name) {
    return name.starts_with("v.") || (name.starts_with("mm.") && !name.starts_with("mm.a."));
}

}  // namespace

TEST_CASE("gemma4v describes EmbeddingGemma 2's vision encoder") {
    const Header h = read_mmproj();
    const model::ModelDescription m = described(h.index);
    CHECK(m.layers.size() == 16);
    CHECK(m.embedding_width == 768);
    CHECK(m.vocabulary_size == 0);
    CHECK_FALSE(m.token_embedding);
    CHECK_FALSE(m.output_norm);
    CHECK_FALSE(m.encoder);
    CHECK(m.rotary_pairing == model::RotaryPairing::Axial);
    CHECK(m.activation == model::FeedForwardActivation::GeluTanh);
    CHECK(m.attention_scale == 1.0f);
    CHECK(m.mask == model::AttentionMask::Bidirectional);
    CHECK(m.value_norm);
    CHECK(m.norm_epsilon == doctest::Approx(1e-6));
    REQUIRE(m.vision);
    CHECK(m.vision->patch_size == 16);
    CHECK(m.vision->positions == 10240);
    CHECK(m.vision->pooling == 3);
    CHECK(m.vision->projection_width == 512);
    CHECK(h.index.tensor(m.vision->patch_embedding).name == "v.patch_embd.weight");
    CHECK(h.index.tensor(m.vision->position_table).name == "v.position_embd.weight");
    CHECK(h.index.tensor(m.vision->projection).name == "mm.input_projection.weight");
    for (std::size_t l = 0; l < m.layers.size(); ++l) {
        const model::LayerDescription& layer = m.layers[l];
        CAPTURE(l);
        CHECK(layer.query_heads == 12);
        CHECK(layer.key_value_heads == 12);
        CHECK(layer.head_dimension == 64);
        CHECK(layer.feed_forward_width == 3072);
        CHECK(layer.rope_base == 100.0f);
        CHECK(layer.attention_window == UINT32_MAX);
        for (const Role r : {Role::AttentionNorm, Role::Query, Role::Key, Role::Value, Role::QueryNorm, Role::KeyNorm,
                             Role::AttentionOutput, Role::PostAttentionNorm, Role::FeedForwardNorm, Role::Gate,
                             Role::Up, Role::Down, Role::PostFeedForwardNorm}) {
            CHECK(layer.tensors[static_cast<std::size_t>(r)].has_value());
        }
        CHECK(h.index.tensor(*layer.tensors[static_cast<std::size_t>(Role::AttentionNorm)]).name ==
              "v.blk." + std::to_string(l) + ".ln1.weight");
    }
}

TEST_CASE("gemma4v's rows are read by EmbeddingGemma 2's text model and by no other") {
    const Header h = read_mmproj();
    const model::ModelDescription vision = described(h.index);

    const testing::ReadHeader text = testing::read_model_header("embeddinggemma-2-q8_0");
    model::ModelDescription embedder{};
    REQUIRE(capability::find_architecture("gemma-embedding2")->describe(text.index, embedder).ok());
    CHECK(arch::kGemma4Vision.pairs(vision, embedder, "gemma-embedding2").ok());

    const testing::ReadHeader qwen = testing::read_model_header("qwen3-0.6b-q4_0");
    model::ModelDescription generator{};
    REQUIRE(capability::find_architecture("qwen3")->describe(qwen.index, generator).ok());
    CHECK(arch::kGemma4Vision.pairs(vision, generator, "qwen3").error == arch::DescribeError::UnsupportedValue);
}

TEST_CASE("the vision encoder plans against its span: the vision tensors placed, the audio encoder's never") {
    const Header h = read_mmproj();
    REQUIRE(residency::check_ranges(kVisionRanges, kFileSize, 58'656).ok());
    const model::ModelDescription m = described(h.index);
    residency::ResidencyPlan plan;
    const auto planned = residency::plan_residency(h.index, m, kDefaults, policy::LoadPolicy{},
                                                   graph::vision_working_set(m, kRows), plan, kVisionRanges);
    REQUIRE_MESSAGE(planned.ok(), planned.subject);

    std::size_t present = 0;
    for (const residency::PlannedTensor& t : plan.tensors) {
        const std::string_view name = h.index.tensor(t.tensor).name;
        CAPTURE(name);
        CHECK(t.present == is_vision(name));
        if (t.present) ++present;
        if (!t.present) CHECK(t.view.pieces().empty());
    }
    CHECK(present == 211);
    CHECK(plan.tensors.size() == h.index.tensors().size());
    CHECK(plan.groups.size() == 32);   // each layer's Q, K and V, and its gate and up
    CHECK(plan.context_offered == kRows);
    CHECK(plan.cache.empty());
    // The vision span's bytes, padded and aligned, and nothing of the audio's.
    CHECK(plan.weight_bytes >= 226'328'576);
    CHECK(plan.weight_bytes < 228'000'000);

    std::vector<residency::Route> routes;
    residency::ResidencyPlan carried;
    const auto routed = residency::plan_routes(h.index, plan, kFileSize, capability::find_format, {}, routes, carried,
                                               kVisionRanges);
    REQUIRE_MESSAGE(routed.ok(), routed.subject);
    CHECK_FALSE(routes.empty());
    for (const residency::Route& r : routes) CHECK(r.file_offset >= kVisionRanges[1].begin);
}

TEST_CASE("a description naming a tensor outside the ranges is refused by name") {
    const Header h = read_mmproj();
    model::ModelDescription m = described(h.index);
    const auto audio = h.index.find("a.blk.0.attn_q.weight");
    REQUIRE(audio);
    m.layers[0].tensors[static_cast<std::size_t>(Role::Query)] = *audio;
    residency::ResidencyPlan plan;
    const auto planned = residency::plan_residency(h.index, m, kDefaults, policy::LoadPolicy{},
                                                   graph::vision_working_set(m, kRows), plan, kVisionRanges);
    CHECK(planned.error == residency::PlanError::MissingTensorBytes);
    CHECK(planned.subject == "a.blk.0.attn_q.weight");
}

namespace {

std::uint64_t workgroups(const kernels::Launch& l, std::uint32_t tokens) {
    return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens,
                                    l.partial_rows, l.row_divisor},
                                   l.workgroup_size, 0, tokens, true);
}

}  // namespace

TEST_CASE("the vision graph: 133 launches a step at every grid, a layer's 8 in order, ε / W to the pooling") {
    const Header h = read_mmproj();
    const arch::VisionEncoder* encoder = capability::find_vision_encoder("gemma4v");
    REQUIRE(encoder == &arch::kGemma4Vision);
    CHECK(capability::find_vision_encoder("gemma3") == nullptr);
    const model::ModelDescription m = described(h.index);
    residency::ResidencyPlan plan;
    REQUIRE(residency::plan_residency(h.index, m, kDefaults, policy::LoadPolicy{}, graph::vision_working_set(m, kRows),
                                      plan, kVisionRanges)
                .ok());
    std::vector<kernels::Launch> launches;
    const auto built = encoder->graph(m, plan, launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);

    for (const auto [columns, rows] : {std::pair{3u, 3u}, std::pair{6u, 9u}, std::pair{21u, 30u}, std::pair{42u, 60u}}) {
        CAPTURE(columns);
        const std::uint32_t patches = columns * rows;
        std::vector<std::string_view> dispatched;
        for (const kernels::Launch& l : launches) {
            const std::uint64_t w = workgroups(l, patches);
            CHECK(w <= 65535);
            if (w > 0) dispatched.push_back(l.role);
        }
        CHECK(dispatched.size() == 133);
        CHECK(dispatched.front() == "patch.embed");
        CHECK(dispatched[1] == "patch.positions");
        const std::vector<std::string_view> layer{"attention.norm", "attention.qkv", "attention.rope", "attention.stream",
                                                  "attention.output", "ffn.norm", "ffn.gate_up", "ffn.down"};
        CHECK(std::equal(layer.begin(), layer.end(), dispatched.begin() + 2));
        CHECK(std::vector<std::string_view>(dispatched.end() - 3, dispatched.end()) ==
              std::vector<std::string_view>{"pool.add", "pool.grid", "pool.project"});
    }
    // The pooling's ε is the file's over W, the √W scale folded in.
    const auto grid = std::find_if(launches.begin(), launches.end(),
                                   [](const kernels::Launch& l) { return l.role == "pool.grid"; });
    REQUIRE(grid != launches.end());
    float epsilon = 0;
    std::memcpy(&epsilon, grid->constants.data() + 8, 4);
    CHECK(epsilon == m.norm_epsilon / 768.0f);
    CHECK(grid->row_divisor == 9);
    // The projection's launches run over the pooled rows.
    for (const kernels::Launch& l : launches) {
        if (l.role == "pool.project") CHECK(l.row_divisor == 9);
    }
}

TEST_CASE("a companion's verdict: Run beside EmbeddingGemma 2, each stage it cannot reach named") {
    const Header h = read_mmproj();
    const testing::ReadHeader text = testing::read_model_header("embeddinggemma-2-q8_0");
    model::ModelDescription embedder{};
    REQUIRE(capability::find_architecture("gemma-embedding2")->describe(text.index, embedder).ok());
    gguf::MemoryByteSource source{std::as_bytes(std::span{text.bytes}), text.file_size};
    std::unique_ptr<tokenizer::Tokenizer> t;
    REQUIRE(preflight::load_tokenizer(source, text.index, t).empty());
    const media::ImageBudget budget{70, 280};
    const preflight::CompanionHost host{embedder, "gemma-embedding2", t->vocabulary(), 384ull << 20};

    const auto verdict = preflight::preflight_companion(h.index, kVisionRanges, kFileSize, 58'656, kDefaults,
                                                        policy::LoadPolicy{}, budget, host);
    for (const auto& b : verdict.blockers) MESSAGE(preflight::to_string(b.stage) << ": " << b.detail);
    CHECK(verdict.reached() == preflight::Stage::Run);
    REQUIRE(verdict.fit);
    CHECK(verdict.fit->context_offered == 2520);
    CHECK(verdict.fit->total_bytes < 330'000'000);

    // The whole file: the audio encoder's tensors would be uploaded, its
    // conv weights' rows of 3 refused.
    const residency::FileRange whole[] = {{0, kFileSize}};
    const auto all = preflight::preflight_companion(h.index, whole, kFileSize, 58'656, kDefaults, policy::LoadPolicy{},
                                                    budget, host);
    CHECK(all.reached() == preflight::Stage::Fit);

    // Ranges that miss the header; a budget the model leaves too little of;
    // a model it does not serve.
    const residency::FileRange late[] = {{1, 58'656}};
    CHECK(preflight::preflight_companion(h.index, late, kFileSize, 58'656, kDefaults, policy::LoadPolicy{}, budget, host)
              .reached() == preflight::Stage::Read);
    policy::LoadPolicy tight{};
    tight.memory_budget = 500ull << 20;
    CHECK(preflight::preflight_companion(h.index, kVisionRanges, kFileSize, 58'656, kDefaults, tight, budget, host)
              .reached() == preflight::Stage::Describe);
    const preflight::CompanionHost other{embedder, "qwen3", t->vocabulary(), 384ull << 20};
    CHECK(preflight::preflight_companion(h.index, kVisionRanges, kFileSize, 58'656, kDefaults, policy::LoadPolicy{},
                                         budget, other)
              .reached() == preflight::Stage::Upload);
}
