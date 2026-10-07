// The forward pass, on the CPU (graph/graph.h): each listed architecture's
// graph over its real file's description and plan, launch for launch against
// the order the blocks state; the launches a step dispatches; and each
// refusal, naming its layer.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/graph/graph.h"
#include "core/kernels/attention/attention.h"
#include "core/kernels/gather/gather.h"
#include "core/kernels/matmul/matmul.h"
#include "core/kernels/norm/norm.h"
#include "core/kernels/rope/rope.h"
#include "core/kernels/schedule.h"
#include "core/kernels/topk/topk.h"
#include "core/sampler/sampler.h"
#include "core/residency/plan.h"
#include "support/model_headers.h"

using namespace bllm;
using model::Role;

namespace {

// WebGPU's default limits, which the harness asks for.
constexpr residency::DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};

struct Loaded {
    testing::ReadHeader header;
    const arch::Architecture* architecture = nullptr;
    model::ModelDescription model{};
    residency::ResidencyPlan plan;
};

Loaded load(std::string_view id) {
    Loaded l{testing::read_model_header(id)};
    std::string_view name;
    REQUIRE(l.header.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    l.architecture = capability::find_architecture(name);
    REQUIRE(l.architecture != nullptr);
    const auto described = l.architecture->describe(l.header.index, l.model);
    REQUIRE_MESSAGE(described.ok(), described.subject);
    REQUIRE(residency::plan_residency(l.header.index, l.model, kDefaults, policy::LoadPolicy{}, l.plan).ok());
    return l;
}

const formats::Format& f16() { return *capability::find_format(gguf::TensorType::F16); }

std::vector<kernels::Launch> graph_of(const Loaded& l, graph::GraphResult* result = nullptr) {
    std::vector<kernels::Launch> out;
    const graph::GraphResult r = l.architecture->graph(l.model, l.plan, f16(), out);
    if (result != nullptr) *result = r;
    else REQUIRE_MESSAGE(r.ok(), r.subject);
    return out;
}

bool same(const kernels::Binding& a, const kernels::Binding& b) {
    return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size;
}

bool same(const kernels::Launch& a, const kernels::Launch& b) {
    return a.kernel == b.kernel && a.format == b.format && a.constants == b.constants &&
           std::equal(a.bindings.begin(), a.bindings.end(), b.bindings.begin(), b.bindings.end(),
                      [](const auto& x, const auto& y) { return same(x, y); }) &&
           a.invocations_per_row == b.invocations_per_row && a.workgroup_size == b.workgroup_size &&
           a.rows == b.rows && a.overrides == b.overrides && a.pack_format == b.pack_format &&
           a.rows_per_tile == b.rows_per_tile && a.key_split == b.key_split && a.window == b.window &&
           a.entry_point == b.entry_point && a.tokens == b.tokens;
}

// graph.h's order, spelled out block by block from the description and the
// plan: what every architecture's graph must equal, given its embedding
// scale.
std::vector<kernels::Launch> stated_order(const model::ModelDescription& m, const residency::ResidencyPlan& p,
                                          float scale) {
    const auto view = [&](gguf::TensorId id) -> const residency::WeightView& {
        return std::find_if(p.tensors.begin(), p.tensors.end(), [&](const auto& t) { return t.tensor == id; })->view;
    };
    const auto at = [&](std::uint32_t layer, Role r) -> const residency::WeightView* {
        const auto& t = m.layers[layer].tensors[static_cast<std::size_t>(r)];
        return t ? &view(*t) : nullptr;
    };
    const auto buffer = [&](std::string_view purpose) {
        return std::find_if(p.scratch.begin(), p.scratch.end(), [&](const auto& s) { return s.purpose == purpose; })
            ->range;
    };
    const auto hidden = buffer("hidden"), normed = buffer("normed"), query = buffer("query"), key = buffer("key"),
               value = buffer("value"), attention = buffer("attention"), partials = buffer("partials"),
               stats = buffer("partial_stats"), output = buffer("output"), activation = buffer("activation"),
               logits = buffer("logits");
    const auto grouped = [&](std::initializer_list<const residency::WeightView*> members) {
        return std::any_of(p.groups.begin(), p.groups.end(), [&](const residency::PlannedGroup& g) {
            if (g.count != members.size()) return false;
            std::size_t i = 0;
            for (const auto* v : members) {
                if (&view(g.members[i++]) != v) return false;
            }
            return true;
        });
    };
    std::vector<kernels::Launch> out;
    const auto add = [&](std::vector<kernels::Launch> ls) {
        for (auto& l : ls) out.push_back(std::move(l));
    };
    const auto write = [&](const residency::WeightView* w, residency::BufferRange in, residency::BufferRange to,
                           kernels::Rows rows) {
        add(kernels::matmul_launches({{w, nullptr, nullptr}, 1, in, {to, {}, {}}, kernels::Epilogue::Write,
                                      m.activation, rows}));
    };

    add(kernels::gather_launches(view(m.token_embedding), hidden, buffer("sampled"), scale));
    const residency::WeightView* carried = nullptr;   // the post-norm of what `output` holds
    for (std::uint32_t i = 0; i < m.layers.size(); ++i) {
        const model::LayerDescription& l = m.layers[i];
        out.push_back(kernels::norm_launch({at(i, Role::AttentionNorm), carried, i > 0, output, hidden, normed,
                                            m.norm_epsilon, kernels::Rows::EveryToken}));
        const auto *q = at(i, Role::Query), *k = at(i, Role::Key), *v = at(i, Role::Value);
        if (grouped({q, k, v})) {
            add(kernels::matmul_launches({{q, k, v}, 3, normed, {query, key, value}, kernels::Epilogue::QKV,
                                          m.activation, kernels::Rows::EveryToken}));
        } else {
            write(q, normed, query, kernels::Rows::EveryToken);
            write(k, normed, key, kernels::Rows::EveryToken);
            write(v, normed, value, kernels::Rows::EveryToken);
        }
        out.push_back(kernels::rope_launch({l, m.rotary_pairing, at(i, Role::QueryNorm), at(i, Role::KeyNorm),
                                            m.rotary_factors ? &view(*m.rotary_factors) : nullptr, m.norm_epsilon,
                                            query, key, value, p.cache[i], &f16()}));
        for (auto& a : kernels::attention_launches(
                 {l, m.attention_scale, query, p.cache[i], &f16(), attention, partials, stats})) {
            out.push_back(std::move(a));
        }
        write(at(i, Role::AttentionOutput), attention, output, kernels::Rows::EveryToken);
        out.push_back(kernels::norm_launch({at(i, Role::FeedForwardNorm), at(i, Role::PostAttentionNorm), true,
                                            output, hidden, normed, m.norm_epsilon, kernels::Rows::EveryToken}));
        add(kernels::matmul_launches({{at(i, Role::Gate), at(i, Role::Up), nullptr}, 2, normed,
                                      {activation, {}, {}}, kernels::Epilogue::GatedActivation, m.activation,
                                      kernels::Rows::EveryToken}));
        write(at(i, Role::Down), activation, output, kernels::Rows::EveryToken);
        carried = at(i, Role::PostFeedForwardNorm);
    }
    out.push_back(kernels::norm_launch({&view(m.output_norm), carried, true, output, hidden, normed, m.norm_epsilon,
                                        kernels::Rows::LastToken}));
    write(&view(m.output_head.value_or(m.token_embedding)), normed, logits, kernels::Rows::LastToken);
    add(kernels::topk_launches(logits, m.vocabulary_size, buffer("partials_a"), buffer("partials_b"),
                               buffer("candidates")));
    out.push_back(sampler::draw_launch(buffer("candidates"), buffer("sampled")));
    return out;
}

void check_stated_order(const std::vector<kernels::Launch>& got, const std::vector<kernels::Launch>& want) {
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        CAPTURE(i);
        CHECK(same(got[i], want[i]));
    }
}

// The launches a step dispatches: those given workgroups.
std::size_t dispatched(const std::vector<kernels::Launch>& launches, std::uint32_t position, std::uint32_t tokens,
                       bool logits) {
    return std::count_if(launches.begin(), launches.end(), [&](const kernels::Launch& l) {
        return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                       l.workgroup_size, position, tokens, logits) > 0;
    });
}

std::size_t count_override(const std::vector<kernels::Launch>& launches, std::string_view name, double value) {
    return std::count_if(launches.begin(), launches.end(), [&](const kernels::Launch& l) {
        return std::find(l.overrides.begin(), l.overrides.end(), kernels::Override{name, value}) != l.overrides.end();
    });
}

}  // namespace

TEST_CASE("every listed architecture supplies its describe and its graph") {
    for (const std::string_view name : {"qwen3", "llama", "gemma3"}) {
        CAPTURE(name);
        const arch::Architecture* a = capability::find_architecture(name);
        REQUIRE(a != nullptr);
        CHECK(a->describe != nullptr);
        CHECK(a->graph != nullptr);
    }
}

TEST_CASE("Qwen3's graph is the stated order, 595 launches, 231 dispatched a step") {
    const Loaded l = load("qwen3-0.6b-q4_0");
    const auto launches = graph_of(l);
    check_stated_order(launches, stated_order(l.model, l.plan, 1.0f));
    CHECK(launches.size() == 595);
    CHECK(dispatched(launches, 0, 1, true) == 231);
    CHECK(dispatched(launches, 0, 512, true) == 231);
    // A prefill step that does not end the prompt: no final norm, head,
    // selection or draw.
    CHECK(dispatched(launches, 0, 512, false) == 225);
    // Decode once its keys span two chunks: every layer's combine.
    CHECK(dispatched(launches, 300, 1, true) == 259);
    // No post-norms: no norm normalizes `output` before adding it.
    CHECK(count_override(launches, "post_norm", 1.0) == 0);
}

TEST_CASE("each step's schedule holds every launch the step dispatches, and walks 253 to 259") {
    const auto launches = graph_of(load("qwen3-0.6b-q4_0"));
    std::vector<kernels::Geometry> geometries;
    for (const kernels::Launch& l : launches) {
        geometries.push_back({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens});
    }
    const kernels::Schedules schedules{geometries};
    for (std::uint32_t tokens = 1; tokens <= residency::kPrefillBlock; ++tokens) {
        for (const bool logits : {false, true}) {
            const auto list = schedules.of(tokens, logits);
            CHECK(list.size() == (logits ? 259u : 253u));
            for (const std::uint32_t position : {0u, 255u, 300u, 4095u, 40000u}) {
                std::vector<std::uint32_t> runs, listed;
                for (std::uint32_t i = 0; i < geometries.size(); ++i) {
                    if (kernels::workgroups_for(geometries[i], launches[i].workgroup_size, position, tokens, logits) > 0) {
                        runs.push_back(i);
                    }
                }
                for (const std::uint32_t i : list) {
                    if (kernels::workgroups_for(geometries[i], launches[i].workgroup_size, position, tokens, logits) > 0) {
                        listed.push_back(i);
                    }
                }
                if (runs != listed) FAIL_CHECK("tokens " << tokens << " logits " << logits << " position " << position);
            }
        }
    }
}

TEST_CASE("Llama 3.2's graph is the stated order, its rotary factors bound in every layer") {
    const Loaded l = load("llama-3.2-1b-instruct-q4_0");
    REQUIRE(l.model.rotary_factors.has_value());
    check_stated_order(graph_of(l), stated_order(l.model, l.plan, 1.0f));
}

TEST_CASE("Gemma 3's graph is the stated order: embedding scaled by √width, every block's output post-normed") {
    const Loaded l = load("gemma-3-1b-it-q4_0");
    const auto launches = graph_of(l);
    const float scale = std::sqrt(static_cast<float>(l.model.embedding_width));
    check_stated_order(launches, stated_order(l.model, l.plan, scale));
    float got = 0;
    std::memcpy(&got, launches.front().constants.data() + 16, sizeof got);   // Gather's scale
    CHECK(got == scale);
    // Every norm that adds post-normalizes: each layer's two but the first
    // layer's attention norm, and the final norm.
    const std::size_t layers = l.model.layers.size();
    CHECK(count_override(launches, "post_norm", 1.0) == 2 * layers - 1 + 1);
}

TEST_CASE("the head reads the token embedding where the file ties them") {
    const Loaded l = load("qwen3-0.6b-q4_0");
    REQUIRE_FALSE(l.model.output_head.has_value());
    const auto launches = graph_of(l);
    const auto& embedding =
        std::find_if(l.plan.tensors.begin(), l.plan.tensors.end(), [&](const auto& t) {
            return t.tensor == l.model.token_embedding;
        })->view.pieces().front();
    // The head: the last matrix product, before selection and the draw.
    const auto product = std::find_if(launches.rbegin(), launches.rend(), [](const kernels::Launch& l) {
        return l.entry_point.starts_with("decode_");
    });
    REQUIRE(product != launches.rend());
    const kernels::Binding& head = product->bindings.front();
    CHECK(head.buffer == embedding.buffer);
    CHECK(head.offset == embedding.offset);
    CHECK(product->rows == kernels::Rows::LastToken);
}

TEST_CASE("Q, K and V the plan did not group are three products, each into its own buffer") {
    Loaded l = load("qwen3-0.6b-q4_0");
    std::erase_if(l.plan.groups, [](const residency::PlannedGroup& g) { return g.count == 3; });
    const auto launches = graph_of(l);
    check_stated_order(launches, stated_order(l.model, l.plan, 1.0f));
    // 3 products of 4 launches where one was: 8 more a layer.
    CHECK(launches.size() == 595 + 8 * l.model.layers.size());
    CHECK(std::none_of(launches.begin(), launches.end(),
                       [](const kernels::Launch& x) { return x.entry_point == "decode_qkv"; }));
}

TEST_CASE("a layer whose gate and up the plan did not group is refused, by layer") {
    Loaded l = load("qwen3-0.6b-q4_0");
    std::erase_if(l.plan.groups, [](const residency::PlannedGroup& g) { return g.count == 2; });
    graph::GraphResult r;
    (void)graph_of(l, &r);
    CHECK(r.error == graph::GraphError::UngroupedFeedForward);
    CHECK(r.subject == "layer 0");
}

TEST_CASE("a layer outside a kernel's shapes is refused, naming the layer and the value") {
    struct Case {
        void (*change)(model::LayerDescription&);
        std::string_view subject;
    };
    for (const Case& c : {
             Case{[](model::LayerDescription& x) { x.head_dimension = 96; },
                  "layer 2: head dimension 96 is not 64, 128 or 256"},
             Case{[](model::LayerDescription& x) { x.key_value_heads = 3; },
                  "layer 2: 16 query heads do not share 3 key-value heads evenly"},
             Case{[](model::LayerDescription& x) { x.key_value_heads = 1; },
                  "layer 2: 16 query heads a key-value head do not divide 8"},
             Case{[](model::LayerDescription& x) { x.tensors[static_cast<std::size_t>(Role::KeyNorm)].reset(); },
                  "layer 2: QK-norm on queries or keys alone"},
         }) {
        CAPTURE(c.subject);
        Loaded l = load("qwen3-0.6b-q4_0");
        c.change(l.model.layers[2]);
        graph::GraphResult r;
        (void)graph_of(l, &r);
        CHECK(r.error == graph::GraphError::UnsupportedShape);
        CHECK(r.subject == c.subject);
    }
}

TEST_CASE("every launch is named by its role and layer, as graph.h lists them") {
    // The roles of a layer, in the order its launches run; consecutive
    // launches of one role, a product's decode and prefill forms, count once.
    const auto roles_of = [](const std::vector<kernels::Launch>& launches, std::uint32_t layer) {
        std::vector<std::string_view> roles;
        for (const kernels::Launch& l : launches) {
            if (l.layer != layer) continue;
            if (roles.empty() || roles.back() != l.role) roles.push_back(l.role);
        }
        return roles;
    };
    const std::vector<std::string_view> grouped{"attention.norm", "attention.qkv",    "attention.rope",
                                                "attention.scores", "attention.combine", "attention.output",
                                                "ffn.norm",       "ffn.gate_up",      "ffn.down"};
    for (const char* model : {"qwen3-0.6b-q4_0", "llama-3.2-1b-instruct-q4_0", "gemma-3-1b-it-q4_0"}) {
        CAPTURE(model);
        const Loaded l = load(model);
        const auto launches = graph_of(l);
        std::uint32_t last_layer = 0;
        for (const kernels::Launch& x : launches) {
            CHECK(!x.role.empty());
            const bool outside = x.role == "embed" || x.role.starts_with("output.");
            CHECK(outside == (x.layer == kernels::kNoLayer));
            if (x.layer != kernels::kNoLayer) {
                CHECK(x.layer >= last_layer);   // layers in order
                last_layer = x.layer;
            }
        }
        CHECK(launches.front().role == "embed");
        CHECK(launches.back().role == "output.draw");
        for (std::uint32_t layer = 0; layer < l.model.layers.size(); ++layer) {
            CAPTURE(layer);
            CHECK(roles_of(launches, layer) == grouped);
        }
        std::vector<std::string_view> output;
        for (const kernels::Launch& x : launches) {
            if (x.role.starts_with("output.") && (output.empty() || output.back() != x.role)) output.push_back(x.role);
        }
        CHECK(output == std::vector<std::string_view>{"output.norm", "output.head", "output.select", "output.draw"});
    }
    // Q, K and V ungrouped: three products, each named.
    Loaded l = load("qwen3-0.6b-q4_0");
    std::erase_if(l.plan.groups, [](const residency::PlannedGroup& g) { return g.count == 3; });
    const auto launches = graph_of(l);
    const auto layer0 = roles_of(launches, 0);
    CHECK(std::vector<std::string_view>(layer0.begin() + 1, layer0.begin() + 4) ==
          std::vector<std::string_view>{"attention.q", "attention.k", "attention.v"});
}
