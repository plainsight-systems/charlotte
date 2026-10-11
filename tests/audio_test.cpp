// EmbeddingGemma 2's audio encoder on the CPU, from its pinned mmproj's
// header (tests/fixtures/external.json): what describe reads (arch/gemma4a),
// whether a text model reads its rows, and its plan against the audio range
// the load reads (residency/file_ranges.h) — the audio tensors placed, every
// clamped product in one binding with its bounds, the vision encoder's
// tensors never.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/arch/gemma4a/gemma4a.h"
#include "core/capability/capability.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/reader.h"
#include "core/graph/audio.h"
#include "core/residency/file_ranges.h"
#include "core/residency/plan.h"
#include "core/residency/routes.h"
#include "support/model_headers.h"

using namespace bllm;

namespace {

constexpr residency::DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};
constexpr std::uint64_t kFileSize = 554'821'024;
// The header and the audio encoder, one range, as the catalog pins it
// (docs/research/2026-10-11-embeddinggemma2-audio.md).
constexpr residency::FileRange kAudioRanges[] = {{0, 328'492'448}};
constexpr residency::FileRange kVisionRanges[] = {{0, 58'656}, {328'492'448, kFileSize}};
constexpr std::uint32_t kRows = 508;

struct Header {
    std::string bytes;
    gguf::TensorIndex index;
};

// The mmproj's header, with each of `renames` — a name and another of the
// same length — applied to its bytes first.
Header read_mmproj(std::initializer_list<std::pair<std::string_view, std::string_view>> renames = {}) {
    const std::string path = std::string(BLLM_TEST_DATA_DIR) + "/models/mmproj-embeddinggemma-2-q8_0.gguf";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing test data: " << path << " (run make test-data)");
    Header h{std::string(64 * 1024, '\0'), {}};
    in.read(h.bytes.data(), static_cast<std::streamsize>(h.bytes.size()));
    REQUIRE(in.gcount() == static_cast<std::streamsize>(h.bytes.size()));
    for (const auto& [from, to] : renames) {
        REQUIRE(from.size() == to.size());
        const std::size_t at = h.bytes.find(from);
        REQUIRE(at != std::string::npos);
        h.bytes.replace(at, from.size(), to);
    }
    gguf::MemoryByteSource source{std::as_bytes(std::span{h.bytes}), kFileSize};
    REQUIRE(gguf::read_index(source, h.index).error == gguf::ReadError::Ok);
    return h;
}

model::ModelDescription described(const gguf::TensorIndex& index) {
    model::ModelDescription m{};
    const auto r = arch::describe_gemma4a(index, m);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return m;
}

std::string_view name_of(const Header& h, gguf::TensorId id) { return h.index.tensor(id).name; }

std::uint64_t workgroups(const kernels::Launch& l, std::uint32_t tokens) {
    return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens,
                                    l.partial_rows, l.row_divisor},
                                   l.workgroup_size, 0, tokens, true);
}

residency::ResidencyPlan planned(const Header& h, const model::ModelDescription& m) {
    residency::ResidencyPlan plan;
    const auto r = residency::plan_residency(h.index, m, kDefaults, policy::LoadPolicy{},
                                             graph::audio_working_set(m, kRows), plan, kAudioRanges);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return plan;
}

const residency::BufferRange& scratch(const residency::ResidencyPlan& plan, std::string_view purpose) {
    const auto s = std::find_if(plan.scratch.begin(), plan.scratch.end(),
                                [&](const residency::PlannedScratch& p) { return p.purpose == purpose; });
    REQUIRE(s != plan.scratch.end());
    return s->range;
}

bool binds(const kernels::Binding& b, const residency::BufferRange& r) {
    return b.buffer == r.buffer && b.offset == r.offset && b.size == r.length;
}

}  // namespace

TEST_CASE("gemma4a describes EmbeddingGemma 2's audio encoder") {
    const Header h = read_mmproj();
    const model::ModelDescription m = described(h.index);
    REQUIRE(m.audio);
    const model::AudioDescription& a = *m.audio;
    CHECK(m.layers.empty());
    CHECK(m.vocabulary_size == 0);
    CHECK_FALSE(m.token_embedding);
    CHECK(m.embedding_width == 1024);
    CHECK(m.norm_epsilon == doctest::Approx(1e-6));
    CHECK(a.layers.size() == 12);
    CHECK(a.width == 1024);
    CHECK(a.heads == 8);
    CHECK(a.feed_forward == 4096);
    CHECK(a.mel_bins == 128);
    CHECK(a.channels == std::array<std::uint32_t, 2>{128, 32});
    CHECK(a.kernel == 5);
    CHECK(a.chunk == 12);
    CHECK(a.window == 12);
    CHECK(a.softcap == 50.0f);
    CHECK(a.residual_weight == 0.5f);
    CHECK(a.q_scale == doctest::Approx((1.0 / std::sqrt(128.0)) / std::log(2.0)));
    CHECK(a.k_scale == doctest::Approx(std::log(1.0 + std::exp(1.0)) / std::log(2.0)));
    CHECK(a.output_width == 1536);
    CHECK(a.projection_width == 512);
    CHECK(name_of(h, a.input_projection) == "a.input_projection.weight");
    CHECK(name_of(h, a.output_projection) == "a.pre_encode.out.weight");
    CHECK(name_of(h, a.output_bias) == "a.pre_encode.out.bias");
    CHECK(name_of(h, a.projection) == "mm.a.input_projection.weight");
    for (std::size_t l = 0; l < a.layers.size(); ++l) {
        CAPTURE(l);
        const model::ConformerLayer& layer = a.layers[l];
        const std::string prefix = "a.blk." + std::to_string(l) + ".";
        // conv_norm is the module's pre-norm, norm_conv the norm after the
        // depthwise convolution, as llama.cpp loads them.
        CHECK(name_of(h, layer.convolution_norm) == prefix + "conv_norm.weight");
        CHECK(name_of(h, layer.depthwise_norm) == prefix + "norm_conv.weight");
        CHECK(name_of(h, layer.output_norm) == prefix + "ln2.weight");
        for (const model::AudioProduct* p : {&layer.ffn_up, &layer.ffn_down, &layer.query, &layer.key, &layer.value,
                                             &layer.attention_output, &layer.pointwise_in, &layer.pointwise_out,
                                             &layer.ffn_up_1, &layer.ffn_down_1}) {
            REQUIRE(p->bounds);
            const std::string_view weight = name_of(h, p->weight);
            const std::string stem(weight.substr(0, weight.size() - std::string_view(".weight").size()));
            CHECK(name_of(h, p->bounds->input_min) == stem + ".input_min");
            CHECK(name_of(h, p->bounds->input_max) == stem + ".input_max");
            CHECK(name_of(h, p->bounds->output_min) == stem + ".output_min");
            CHECK(name_of(h, p->bounds->output_max) == stem + ".output_max");
        }
    }
    // Ten clamped products a layer, and the output projection with its bias.
    CHECK(m.product_spans.size() == 12 * 10 + 1);
    CHECK(m.product_spans.back().members.size() == 2);
}

TEST_CASE("gemma4a refuses what its graph has no block for, naming it") {
    model::ModelDescription m{};
    // A product with three of its four bounds.
    const Header no_bound = read_mmproj({{"a.blk.3.attn_q.input_min", "a.blk.3.attn_q.inpXt_min"}});
    const auto bound = arch::describe_gemma4a(no_bound.index, m);
    CHECK(bound.error == arch::DescribeError::MissingTensor);
    CHECK(bound.subject == "a.blk.3.attn_q.input_min");
    // An audio tensor the description does not name.
    const Header extra = read_mmproj({{"v.blk.0.ln1.weight", "a.blk.0.xyz.weight"}});
    const auto unnamed = arch::describe_gemma4a(extra.index, m);
    CHECK(unnamed.error == arch::DescribeError::UnsupportedValue);
    CHECK(unnamed.subject == "a.blk.0.xyz.weight");
    // Another spectrogram's bins.
    const Header projector = read_mmproj({{"gemma4a", "gemma4b"}});
    CHECK(arch::describe_gemma4a(projector.index, m).subject == "clip.audio.projector_type");
}

TEST_CASE("gemma4a's rows are read by EmbeddingGemma 2's text model and by no other") {
    const Header h = read_mmproj();
    const model::ModelDescription audio = described(h.index);
    const testing::ReadHeader text = testing::read_model_header("embeddinggemma-2-q8_0");
    model::ModelDescription embedder{};
    REQUIRE(capability::find_architecture("gemma-embedding2")->describe(text.index, embedder).ok());
    CHECK(arch::pairs_gemma4a(audio, embedder, "gemma-embedding2").ok());
    const testing::ReadHeader qwen = testing::read_model_header("qwen3-0.6b-q4_0");
    model::ModelDescription generator{};
    REQUIRE(capability::find_architecture("qwen3")->describe(qwen.index, generator).ok());
    CHECK(arch::pairs_gemma4a(audio, generator, "qwen3").error == arch::DescribeError::UnsupportedValue);
}

TEST_CASE("the audio encoder plans against its range: every clamped product in one binding with its bounds") {
    const Header h = read_mmproj();
    REQUIRE(residency::check_ranges(kAudioRanges, kFileSize, 58'656).ok());
    const model::ModelDescription m = described(h.index);
    residency::ResidencyPlan plan;
    const auto planned = residency::plan_residency(h.index, m, kDefaults, policy::LoadPolicy{},
                                                   graph::audio_working_set(m, kRows), plan, kAudioRanges);
    REQUIRE_MESSAGE(planned.ok(), planned.subject);
    CHECK(plan.cache.empty());
    REQUIRE(plan.spans.size() == m.product_spans.size());
    for (std::size_t i = 0; i < plan.spans.size(); ++i) {
        const residency::PlannedSpan& span = plan.spans[i];
        CAPTURE(name_of(h, span.weight));
        CHECK(span.span.length <= kDefaults.max_storage_binding_size);
        for (const gguf::TensorId member : m.product_spans[i].members) {
            const auto& pieces = plan.tensors[static_cast<std::size_t>(member)].view.pieces();
            REQUIRE(pieces.size() == 1);
            CHECK(pieces.front().buffer == span.span.buffer);
            CHECK(pieces.front().offset >= span.span.offset);
            CHECK(pieces.front().offset + pieces.front().length <= span.span.offset + span.span.length);
        }
    }
    // The vision encoder's tensors are neither placed nor described.
    for (const residency::PlannedTensor& t : plan.tensors) {
        const std::string_view name = h.index.tensor(t.tensor).name;
        if (name.starts_with("v.") || name == "mm.input_projection.weight") CHECK_FALSE(t.present);
    }
    for (const std::string_view purpose : {"mel", "subsampled", "relative", "positions", "soft_tokens"}) {
        CHECK(std::any_of(plan.scratch.begin(), plan.scratch.end(),
                          [&](const residency::PlannedScratch& s) { return s.purpose == purpose; }));
    }
    MESSAGE("weights " << plan.weight_bytes / 1e6 << " MB, working set " << plan.scratch_bytes / 1e6 << " MB");

    // Every audio tensor routes, the convolutions' weights — rows of 9 and of
    // 5, read whole by their kernels — among them.
    std::vector<residency::Route> routes;
    residency::ResidencyPlan carried;
    const auto routed = residency::plan_routes(h.index, plan, kFileSize, capability::find_format, {}, routes, carried,
                                               kAudioRanges);
    REQUIRE_MESSAGE(routed.ok(), routed.subject);
    for (const gguf::TensorId id : m.read_whole) {
        CHECK(plan.tensors[static_cast<std::size_t>(id)].whole);
    }
    CHECK(m.read_whole.size() == 2 + 12);
}

TEST_CASE("the audio encoder cannot plan against the vision span: its tensors lie outside it") {
    const Header h = read_mmproj();
    const model::ModelDescription m = described(h.index);
    residency::ResidencyPlan plan;
    const auto planned = residency::plan_residency(h.index, m, kDefaults, policy::LoadPolicy{},
                                                   graph::audio_working_set(m, kRows), plan, kVisionRanges);
    CHECK(planned.error == residency::PlanError::MissingTensorBytes);
}

TEST_CASE("the audio graph: 199 blocks' launches at R of 1, 75 and 508, a layer's 16 in order, clamps on their spans") {
    const Header h = read_mmproj();
    const arch::MediaEncoder* encoder = capability::find_media_encoder(arch::MediaKind::Audio, "gemma4a");
    REQUIRE(encoder == &arch::kGemma4Audio);
    CHECK(capability::find_media_encoder(arch::MediaKind::Image, "gemma4a") == nullptr);
    const model::ModelDescription m = described(h.index);
    const residency::ResidencyPlan plan = planned(h, m);
    std::vector<kernels::Launch> launches;
    const auto built = encoder->graph(m, plan, launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);

    const std::vector<std::string_view> layer{"ffn.norm",       "ffn.up",           "ffn.down",  "attention.norm",
                                              "attention.q",    "attention.k",      "attention.v", "attention.local",
                                              "attention.output", "conv.norm",      "conv.in",   "conv.light",
                                              "conv.out",       "ffn2.norm",        "ffn2.up",   "ffn2.down"};
    for (const std::uint32_t rows : {1u, 75u, kRows}) {
        CAPTURE(rows);
        // A product's launches for one step: its form for the step, and a
        // combine where the gate splits it; one block each.
        std::vector<std::pair<std::string_view, std::uint32_t>> blocks;
        for (const kernels::Launch& l : launches) {
            const std::uint64_t w = workgroups(l, rows);
            CHECK(w <= 65535);
            if (w == 0) continue;
            if (blocks.empty() || blocks.back() != std::pair{l.role, l.layer}) blocks.emplace_back(l.role, l.layer);
        }
        REQUIRE(blocks.size() == 199);
        CHECK(blocks[0].first == "subsample.first");
        CHECK(blocks[1].first == "subsample.second");
        CHECK(blocks[2].first == "subsample.project");
        for (std::uint32_t l = 0; l < 12; ++l) {
            for (std::size_t i = 0; i < layer.size(); ++i) {
                CHECK(blocks[3 + 16 * l + i] == std::pair{layer[i], l});
            }
        }
        CHECK(blocks[195].first == "out.renorm");
        CHECK(blocks[196].first == "out.project");
        CHECK(blocks[197].first == "out.norm");
        CHECK(blocks[198].first == "out.embed");
    }

    // Every launch of a clamped product binds its weight's span, as does the
    // biased output projection; none of them splits along K.
    const auto span_of = [&](gguf::TensorId weight) -> const residency::BufferRange& {
        const auto s = std::find_if(plan.spans.begin(), plan.spans.end(),
                                    [&](const residency::PlannedSpan& p) { return p.weight == weight; });
        REQUIRE(s != plan.spans.end());
        return s->span;
    };
    const auto weight_of = [&](const kernels::Launch& l) -> std::optional<gguf::TensorId> {
        if (l.layer == kernels::kNoLayer) {
            if (l.role == "out.project") return m.audio->output_projection;
            return std::nullopt;
        }
        const model::ConformerLayer& c = m.audio->layers[l.layer];
        const std::pair<std::string_view, const model::AudioProduct*> products[] = {
            {"ffn.up", &c.ffn_up},       {"ffn.down", &c.ffn_down},   {"attention.q", &c.query},
            {"attention.k", &c.key},     {"attention.v", &c.value},   {"attention.output", &c.attention_output},
            {"conv.in", &c.pointwise_in}, {"conv.out", &c.pointwise_out}, {"ffn2.up", &c.ffn_up_1},
            {"ffn2.down", &c.ffn_down_1}};
        for (const auto& [role, p] : products) {
            if (l.role == role) return p->weight;
        }
        return std::nullopt;
    };
    std::size_t spanned = 0;
    for (const kernels::Launch& l : launches) {
        const auto weight = weight_of(l);
        if (!weight) continue;
        CAPTURE(l.role);
        CAPTURE(l.layer);
        CHECK(binds(l.bindings[0], span_of(*weight)));
        CHECK(l.entry_point.find("split") == std::string_view::npos);
        ++spanned;
    }
    CHECK(spanned >= 12 * 10 + 1);

    // The reused buffers: the second convolution's rows in `normed`, the
    // GLU's in `attention`, the light convolution's in `query`, the output
    // norm's in `activation`; each layer's attention its own relative rows.
    const auto first = [&](std::string_view role, std::uint32_t at_layer = kernels::kNoLayer) -> const kernels::Launch& {
        const auto l = std::find_if(launches.begin(), launches.end(),
                                    [&](const kernels::Launch& x) { return x.role == role && x.layer == at_layer; });
        REQUIRE(l != launches.end());
        return *l;
    };
    CHECK(binds(first("subsample.first").bindings[3], scratch(plan, "subsampled")));
    CHECK(binds(first("subsample.second").bindings[3], scratch(plan, "normed")));
    CHECK(binds(first("conv.light", 3).bindings[2], scratch(plan, "attention")));
    CHECK(binds(first("conv.light", 3).bindings[3], scratch(plan, "query")));
    CHECK(binds(first("out.norm").bindings[4], scratch(plan, "activation")));
    const residency::BufferRange& relative = scratch(plan, "relative");
    const std::uint64_t table = 13ull * 1024 * 4;
    for (std::uint32_t l = 0; l < 12; ++l) {
        const kernels::Binding& b = first("attention.local", l).bindings[1];
        CHECK(b.buffer == relative.buffer);
        CHECK(b.offset == relative.offset + l * table);
        CHECK(b.size == table);
    }
}

TEST_CASE("the audio setup: each layer's relative table, attn_k_rel times the 13 positions, into its own rows") {
    const Header h = read_mmproj();
    const model::ModelDescription m = described(h.index);
    const residency::ResidencyPlan plan = planned(h, m);
    std::vector<kernels::Launch> launches;
    const auto built = arch::kGemma4Audio.setup(m, plan, launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);
    std::vector<std::uint32_t> layers;
    const residency::BufferRange& relative = scratch(plan, "relative");
    const std::uint64_t table = 13ull * 1024 * 4;
    for (const kernels::Launch& l : launches) {
        CHECK(l.role == "relative.project");
        if (workgroups(l, 13) == 0) continue;
        layers.push_back(l.layer);
        CAPTURE(l.entry_point);
        const auto bound = [&](const residency::BufferRange& r) {
            return std::any_of(l.bindings.begin(), l.bindings.end(),
                               [&](const kernels::Binding& b) { return binds(b, r); });
        };
        // A split's ranges read the positions and write range sums; its
        // combine reads those and writes the table; any other form both.
        const bool combine = l.entry_point.starts_with("combine");
        const bool split = l.entry_point.starts_with("split");
        if (!combine) CHECK(bound(scratch(plan, "positions")));
        if (!split) CHECK(bound({relative.buffer, relative.offset + l.layer * table, table}));
    }
    CHECK(std::unique(layers.begin(), layers.end()) - layers.begin() == 12);
}
