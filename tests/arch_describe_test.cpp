#include <doctest/doctest.h>

#include <algorithm>
#include <string>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/preflight/preflight.h"
#include "support/gguf_fixture.h"
#include "support/model_headers.h"

// The tiny fixtures are complete transformers in the shape a real converter
// writes (tools/make_fixture_gguf.py: embedding 8, 2 query heads and 1
// key/value head of width 4, feed-forward 16, a 6-token vocabulary).
using namespace bllm;
using arch::DescribeError;
using model::Role;

namespace {

struct Described {
    arch::DescribeResult result;
    model::ModelDescription model;
};

// Reads a fixture and describes it with the architecture it names, found
// through the capability table as preflight finds it.
Described describe_fixture(const std::string& name) {
    const auto bytes = testing::load_gguf_fixture(name);
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    std::string_view architecture;
    REQUIRE(index.read_string("general.architecture", architecture) == gguf::MetadataError::Ok);
    const arch::Architecture* a = capability::find_architecture(architecture);
    REQUIRE(a != nullptr);
    Described d{};
    d.result = a->describe(index, d.model);
    return d;
}

bool has(const model::LayerDescription& layer, Role role) {
    return layer.tensors[static_cast<std::size_t>(role)].has_value();
}

}  // namespace

TEST_CASE("each architecture is in the capability table under its own name") {
    for (const char* name : {"qwen3", "llama", "gemma3"}) {
        CAPTURE(name);
        const arch::Architecture* a = capability::find_architecture(name);
        REQUIRE(a != nullptr);
        CHECK(a->name == name);
    }
    CHECK(capability::find_architecture("mistral") == nullptr);
}

TEST_CASE("the shared numbers are read the same way for every architecture") {
    for (const char* name : {"tiny_qwen3", "tiny_llama", "tiny_gemma3_no_window"}) {
        CAPTURE(name);
        const auto d = describe_fixture(name);
        REQUIRE(d.result.ok());
        CHECK(d.model.vocabulary_size == 6);
        CHECK(d.model.embedding_width == 32);
        CHECK(d.model.trained_context == 64);
        CHECK_FALSE(d.model.output_head.has_value());   // the head reads the embedding
        for (const auto& layer : d.model.layers) {
            CHECK(layer.query_heads == 2);
            CHECK(layer.key_value_heads == 1);
            CHECK(layer.head_dimension == 32);
            CHECK(layer.feed_forward_width == 64);
            CHECK(layer.attention_window == 64);
            CHECK(layer.rope_base == doctest::Approx(1e6));
        }
    }
}

TEST_CASE("each layer's weights are found by the roles its architecture has") {
    const auto qwen = describe_fixture("tiny_qwen3");
    REQUIRE(qwen.result.ok());
    CHECK(has(qwen.model.layers[1], Role::QueryNorm));
    CHECK_FALSE(has(qwen.model.layers[1], Role::PostAttentionNorm));

    const auto llama = describe_fixture("tiny_llama");
    REQUIRE(llama.result.ok());
    CHECK(has(llama.model.layers[1], Role::Down));
    CHECK_FALSE(has(llama.model.layers[1], Role::QueryNorm));

    const auto gemma = describe_fixture("tiny_gemma3");
    REQUIRE(gemma.result.ok());
    CHECK(has(gemma.model.layers[6], Role::PostAttentionNorm));
    CHECK(has(gemma.model.layers[6], Role::PostFeedForwardNorm));
}

TEST_CASE("Gemma 3 attends over a window on five layers in six, with their own rotary base") {
    const auto d = describe_fixture("tiny_gemma3");
    REQUIRE(d.result.ok());
    REQUIRE(d.model.layers.size() == 7);
    for (std::size_t layer = 0; layer < 7; ++layer) {
        CAPTURE(layer);
        const bool global = layer == 5;
        CHECK(d.model.layers[layer].attention_window == (global ? 64u : 16u));
        CHECK(d.model.layers[layer].rope_base == doctest::Approx(global ? 1e6 : 1e4));
    }
}

TEST_CASE("each architecture rotates the pairs llama.cpp gives it") {
    struct Case {
        const char* fixture;
        model::RotaryPairing pairing;
    };
    for (const Case& c : {Case{"tiny_qwen3", model::RotaryPairing::Halves},
                          Case{"tiny_llama", model::RotaryPairing::Adjacent},
                          Case{"tiny_gemma3", model::RotaryPairing::Halves}}) {
        CAPTURE(c.fixture);
        const auto d = describe_fixture(c.fixture);
        REQUIRE(d.result.ok());
        CHECK(d.model.rotary_pairing == c.pairing);
        CHECK_FALSE(d.model.rotary_factors.has_value());
    }
}

TEST_CASE("Llama 3's frequency factors are found, one a pair") {
    const auto d = describe_fixture("tiny_llama_rope_freqs");
    REQUIRE(d.result.ok());
    REQUIRE(d.model.rotary_factors.has_value());
}

TEST_CASE("the listed models describe, with their pairing and Llama's factors") {
    struct Case {
        const char* model;
        model::RotaryPairing pairing;
        bool factors;
    };
    for (const Case& c : {Case{"qwen3-0.6b-q4_0", model::RotaryPairing::Halves, false},
                          Case{"llama-3.2-1b-instruct-q4_0", model::RotaryPairing::Adjacent, true},
                          Case{"gemma-3-1b-it-q4_0", model::RotaryPairing::Halves, false}}) {
        CAPTURE(c.model);
        const testing::ReadHeader header = testing::read_model_header(c.model);
        std::string_view architecture;
        REQUIRE(header.index.read_string("general.architecture", architecture) == gguf::MetadataError::Ok);
        model::ModelDescription description;
        const auto result = capability::find_architecture(architecture)->describe(header.index, description);
        REQUIRE_MESSAGE(result.ok(), result.subject);
        CHECK(description.rotary_pairing == c.pairing);
        CHECK(description.rotary_factors.has_value() == c.factors);
    }
}

TEST_CASE("rotary keys declaring whole heads and no scaling describe as if absent") {
    const auto d = describe_fixture("tiny_qwen3_rope_declared");
    REQUIRE(d.result.ok());
    CHECK(d.model.layers[0].head_dimension == 32);
}

TEST_CASE("a file that breaks what describe needs says what and where") {
    struct Case {
        const char* fixture;
        DescribeError error;
        const char* subject;
    };
    for (const Case& c : {
             Case{"tiny_qwen3_missing_key", DescribeError::MissingKey, "qwen3.attention.head_count_kv"},
             Case{"tiny_qwen3_missing_tensor", DescribeError::MissingTensor, "blk.1.ffn_up.weight"},
             Case{"tiny_qwen3_wrong_shape", DescribeError::ShapeMismatch, "blk.1.attn_k.weight"},
             Case{"tiny_qwen3_heads_not_grouped", DescribeError::InvalidValue, "qwen3.attention.head_count_kv"},
             Case{"tiny_qwen3_too_many_layers", DescribeError::InvalidValue, "qwen3.block_count"},
             Case{"tiny_gemma3_pattern_per_layer", DescribeError::UnsupportedValue,
                  "gemma3.attention.sliding_window_pattern"},
             // Rotary forms the rope kernel does not implement.
             Case{"tiny_qwen3_partial_rotation", DescribeError::UnsupportedValue, "qwen3.rope.dimension_count"},
             Case{"tiny_qwen3_rope_scaling", DescribeError::UnsupportedValue, "qwen3.rope.scaling.type"},
             Case{"tiny_llama_rope_freqs_wrong_shape", DescribeError::ShapeMismatch, "rope_freqs.weight"},
             Case{"tiny_llama_rope_freqs_f16", DescribeError::UnsupportedValue, "rope_freqs.weight"},
         }) {
        CAPTURE(c.fixture);
        const auto d = describe_fixture(c.fixture);
        CHECK(d.result.error == c.error);
        CHECK(d.result.subject == c.subject);
    }
}

TEST_CASE("a model that describes reaches the describe stage, and preflight names the next") {
    const auto bytes = testing::load_gguf_fixture("tiny_qwen3");
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    const auto verdict = preflight::preflight(index, residency::DeviceLimits{}, policy::LoadPolicy{});
    CHECK(verdict.reached() == preflight::Stage::Describe);
}

TEST_CASE("a describe failure reaches preflight with its subject named") {
    const auto bytes = testing::load_gguf_fixture("tiny_qwen3_missing_tensor");
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    const auto verdict = preflight::preflight(index, residency::DeviceLimits{}, policy::LoadPolicy{});
    CHECK(verdict.reached() == preflight::Stage::Download);
    const bool named = std::any_of(verdict.blockers.begin(), verdict.blockers.end(), [](const auto& b) {
        return b.detail == "architecture \"qwen3\" cannot read this file: a required tensor is "
                           "missing (blk.1.ffn_up.weight)";
    });
    CHECK(named);
}
