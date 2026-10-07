#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/gguf/reader.h"
#include "core/capability/capability.h"
#include "core/preflight/preflight.h"
#include "core/residency/routes.h"
#include "support/gguf_fixture.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using preflight::Blocker;
using preflight::Stage;
using preflight::Verdict;

namespace {

Verdict preflight_fixture(const std::string& name) {
    const auto bytes = testing::load_gguf_fixture(name);
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    return preflight::preflight(source, index, residency::DeviceLimits{}, policy::LoadPolicy{});
}

bool blocked(const Verdict& verdict, Stage stage, const std::string& detail) {
    return std::any_of(verdict.blockers.begin(), verdict.blockers.end(),
                       [&](const Blocker& b) { return b.stage == stage && b.detail == detail; });
}

}  // namespace

TEST_CASE("the stage reached is the one before the earliest stage blocked") {
    CHECK(Verdict{}.reached() == Stage::Run);
    CHECK(Verdict{{{Stage::Run, "r"}}}.reached() == Stage::Upload);
    CHECK(Verdict{{{Stage::Run, "r"}, {Stage::Describe, "d"}, {Stage::Fit, "f"}}}.reached() ==
          Stage::Download);
}

TEST_CASE("a readable file can always be downloaded, whatever this build can run") {
    for (const char* name : {"valid", "no_architecture", "architecture_not_string",
                             "tokenizer_named", "shared_format", "q6_k_tensor"}) {
        CAPTURE(name);
        CHECK(preflight_fixture(name).reached() >= Stage::Download);
    }
}

TEST_CASE("every check reports, each naming the stage it stops") {
    // The fixture names a real architecture but declares too few of its
    // numbers to describe, uses formats this build runs, and declares no
    // tokenizer.
    const auto verdict = preflight_fixture("valid");

    CHECK(verdict.reached() == Stage::Download);
    CHECK(blocked(verdict, Stage::Describe,
                  "architecture \"qwen3\" cannot read this file: a required key is missing "
                  "(qwen3.context_length)"));
    CHECK_FALSE(blocked(verdict, Stage::Upload,
                        "format Q4_0 is not supported (1 tensor, first token_embd.weight)"));
    CHECK_FALSE(blocked(verdict, Stage::Upload,
                        "format F32 is not supported (1 tensor, first output_norm.weight)"));
    CHECK(blocked(verdict, Stage::Run, "the file does not declare tokenizer.ggml.model"));
}

TEST_CASE("every stage is implemented, so none is blocked for being missing") {
    CHECK(preflight::kImplementedThrough == Stage::Run);
    const auto verdict = preflight_fixture("valid");
    CHECK(std::none_of(verdict.blockers.begin(), verdict.blockers.end(), [](const Blocker& b) {
        return b.detail.ends_with("stage is not implemented in this build");
    }));
}

TEST_CASE("an unsupported format is reported once, counting every tensor that uses it") {
    const auto verdict = preflight_fixture("shared_format");
    const auto formats = std::count_if(verdict.blockers.begin(), verdict.blockers.end(),
                                       [](const Blocker& b) {
                                           return b.detail.rfind("format ", 0) == 0;
                                       });
    CHECK(formats == 1);   // Q5_0, once; its F32 norm runs
    CHECK(blocked(verdict, Stage::Upload, "format Q5_0 is not supported (2 tensors, first first.weight)"));
}

TEST_CASE("rows that are not a whole number of 32-weight groups block Upload, counted and named") {
    CHECK(blocked(preflight_fixture("tiny_qwen3_odd_row"), Stage::Upload,
                  "rows must be a multiple of 32 weights (1 tensor, first extra.norm, rows of 33)"));
    const auto clean = preflight_fixture("tiny_qwen3");
    CHECK(std::none_of(clean.blockers.begin(), clean.blockers.end(),
                       [](const Blocker& b) { return b.detail.rfind("rows must", 0) == 0; }));
}

TEST_CASE("a file that names no architecture, or names it wrongly, says which") {
    CHECK(blocked(preflight_fixture("no_architecture"), Stage::Describe,
                  "the file does not declare general.architecture"));
    CHECK(blocked(preflight_fixture("architecture_not_string"), Stage::Describe,
                  "general.architecture is not a string"));
}

TEST_CASE("the tokenizer and the pre-tokenizer are judged separately, each by name") {
    // gpt2 and qwen2 are both implemented: neither is named unsupported. The
    // fixture carries no vocabulary, so the tokenizer, listed, does not load.
    const auto verdict = preflight_fixture("tokenizer_named");
    CHECK(std::none_of(verdict.blockers.begin(), verdict.blockers.end(),
                       [](const Blocker& b) { return b.detail.find("not supported") != std::string::npos; }));
    CHECK(std::any_of(verdict.blockers.begin(), verdict.blockers.end(), [](const Blocker& b) {
        return b.stage == Stage::Run && b.detail.starts_with("tokenizer \"gpt2\" does not load from this file: ");
    }));
    CHECK(blocked(preflight_fixture("unknown_pretokenizer"), Stage::Run,
                  "pre-tokenizer \"no-such-split\" is not supported"));
}

TEST_CASE("a tokenizer that splits no text first ignores the pre-tokenizer named") {
    // llama names "default", which no row lists: it is not judged. The
    // fixture carries no vocabulary, so the tokenizer does not load.
    const auto verdict = preflight_fixture("sentencepiece_default_pre");
    CHECK(std::none_of(verdict.blockers.begin(), verdict.blockers.end(),
                       [](const Blocker& b) { return b.detail.find("pre-tokenizer") != std::string::npos; }));
    CHECK(std::any_of(verdict.blockers.begin(), verdict.blockers.end(), [](const Blocker& b) {
        return b.stage == Stage::Run && b.detail.starts_with("tokenizer \"llama\" does not load from this file: ");
    }));
}

TEST_CASE("a tokenizer that splits text first is blocked without its pre-tokenizer") {
    CHECK(blocked(preflight_fixture("tokenizer_without_pre"), Stage::Run,
                  "tokenizer \"gpt2\" needs a pre-tokenizer, and the file does not declare tokenizer.ggml.pre"));
}

TEST_CASE("no blocker names a stage the reader alone decides, and every one says something") {
    for (const char* name : {"valid", "no_architecture", "architecture_not_string",
                             "tokenizer_named", "tokenizer_without_pre", "sentencepiece_default_pre",
                             "shared_format"}) {
        CAPTURE(name);
        for (const Blocker& b : preflight_fixture(name).blockers) {
            CHECK(b.stage > Stage::Download);
            CHECK_FALSE(b.detail.empty());
        }
    }
}

TEST_CASE("with a device's limits, a model that describes reaches fit and says how it fits") {
    const auto bytes = testing::load_gguf_fixture("tiny_qwen3");
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);

    const residency::DeviceLimits defaults{256ull << 20, 128ull << 20, 256};
    const auto verdict = preflight::preflight(source, index, defaults, policy::LoadPolicy{});
    CHECK(verdict.reached() == Stage::Upload);   // Run needs a tokenizer the fixture lacks
    REQUIRE(verdict.fit.has_value());
    CHECK(verdict.fit->context_offered == 64);
    CHECK(verdict.fit->total_bytes <= verdict.fit->memory_budget);

    // Without a device there are no limits, and fit cannot be judged.
    const auto blind = preflight::preflight(source, index, residency::DeviceLimits{}, policy::LoadPolicy{});
    CHECK(blind.reached() == Stage::Describe);
    CHECK(blocked(blind, Stage::Fit, "no GPU device was acquired, so fit cannot be judged"));
}

TEST_CASE("a model that fits is judged for Run by its architecture's graph, which names the layer it refuses") {
    const residency::DeviceLimits defaults{256ull << 20, 128ull << 20, 256};
    // The tiny fixture's heads are 32 wide, which no attention kernel takes.
    const auto bytes = testing::load_gguf_fixture("tiny_qwen3");
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    CHECK(blocked(preflight::preflight(source, index, defaults, policy::LoadPolicy{}), Stage::Run,
                  "architecture \"qwen3\" cannot run this file: a layer's shape is outside the kernels "
                  "(layer 0: head dimension 32 is not 64, 128 or 256)"));
    // A cache precision no kernel writes.
    policy::LoadPolicy bf16;
    bf16.cache_precision = policy::CachePrecision::BF16;
    const testing::ReadHeader qwen3 = testing::read_model_header("qwen3-0.6b-q4_0");
    gguf::MemoryByteSource qwen3_source{std::as_bytes(std::span{qwen3.bytes}), qwen3.file_size};
    CHECK(blocked(preflight::preflight(qwen3_source, qwen3.index, defaults, bf16), Stage::Run,
                  "the cache format BF16 cannot be written by this build"));
    // Each listed model runs: nothing blocks it.
    for (const char* id : {"qwen3-0.6b-q4_0", "llama-3.2-1b-instruct-q4_0", "gemma-3-1b-it-q4_0"}) {
        CAPTURE(id);
        const testing::ReadHeader header = testing::read_model_header(id);
        gguf::MemoryByteSource header_source{std::as_bytes(std::span{header.bytes}), header.file_size};
        const Verdict verdict = preflight::preflight(header_source, header.index, defaults, policy::LoadPolicy{});
        CHECK(verdict.reached() == Stage::Run);
        CHECK(verdict.blockers.empty());
    }
}

TEST_CASE("a model that fits reports its duplicate candidates with both byte ranges") {
    const residency::DeviceLimits limits{256ull << 20, 128ull << 20, 256};
    const auto judged = [&](const std::string& name, gguf::TensorIndex& index) {
        const auto bytes = testing::load_gguf_fixture(name);
        gguf::MemoryByteSource source{bytes};
        REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
        return preflight::preflight(source, index, limits, policy::LoadPolicy{});
    };

    gguf::TensorIndex index;
    const auto copy = judged("tiny_qwen3_output_copy", index);
    REQUIRE(copy.fit.has_value());
    REQUIRE(copy.fit->duplicates.size() == 1);
    const auto& d = copy.fit->duplicates[0];
    const auto& head = index.tensor(d.tensor);
    const auto& embedding = index.tensor(d.copies);
    CHECK(head.name == "output.weight");
    CHECK(embedding.name == "token_embd.weight");
    CHECK(d.offset == head.data_offset);
    CHECK(d.copies_offset == embedding.data_offset);
    CHECK(d.length == head.data_length);

    gguf::TensorIndex tied;
    const auto plain = judged("tiny_qwen3", tied);
    REQUIRE(plain.fit.has_value());
    CHECK(plain.fit->duplicates.empty());   // the head reads the embedding
}

TEST_CASE("a load plans what preflight judged Fit, or says what stops it as preflight does") {
    const auto bytes = testing::load_gguf_fixture("tiny_qwen3");
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    const residency::DeviceLimits limits{256ull << 20, 128ull << 20, 256};
    model::ModelDescription description;
    residency::ResidencyPlan plan;
    REQUIRE(preflight::plan_load(index, limits, policy::LoadPolicy{}, description, plan).empty());
    const auto verdict = preflight::preflight(source, index, limits, policy::LoadPolicy{});
    REQUIRE(verdict.fit.has_value());
    CHECK(plan.total_bytes == verdict.fit->total_bytes);
    CHECK(plan.buffers.size() == verdict.fit->buffer_count);
    CHECK(description.layers.size() == 2);

    const auto refused = preflight::plan_load(index, residency::DeviceLimits{}, policy::LoadPolicy{}, description, plan);
    CHECK(refused == "no GPU device was acquired, so fit cannot be judged");
}

TEST_CASE("preflight reaches Upload exactly when routes would let the upload begin") {
    const residency::DeviceLimits limits{256ull << 20, 128ull << 20, 256};
    for (const char* name : {"tiny_qwen3", "tiny_qwen3_odd_row"}) {
        CAPTURE(name);
        const auto bytes = testing::load_gguf_fixture(name);
        gguf::MemoryByteSource source{bytes};
        gguf::TensorIndex index;
        REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
        model::ModelDescription description;
        residency::ResidencyPlan plan;
        REQUIRE(preflight::plan_load(index, limits, policy::LoadPolicy{}, description, plan).empty());
        std::vector<residency::Route> routes;
        residency::ResidencyPlan carried;
        const auto routed = residency::plan_routes(index, plan, bytes.size(), capability::find_format, {}, routes,
                                                   carried);
        const auto verdict = preflight::preflight(source, index, limits, policy::LoadPolicy{});
        CHECK((verdict.reached() >= Stage::Upload) == routed.ok());
    }
}

TEST_CASE("Run is blocked, by name, when the tokenizer does not load or a stop token does not resolve") {
    const residency::DeviceLimits defaults{256ull << 20, 128ull << 20, 256};
    // Each listed model's tokenizer loads and its file's stop resolves: no
    // Run blocker names either (the test above).
    const testing::ReadHeader qwen3 = testing::read_model_header("qwen3-0.6b-q4_0");
    gguf::MemoryByteSource source{std::as_bytes(std::span{qwen3.bytes}), qwen3.file_size};
    policy::LoadPolicy stops;
    stops.stop = {"<|endoftext|>"};
    const Verdict measured = preflight::preflight(source, qwen3.index, defaults, stops);
    CHECK(measured.reached() == Stage::Run);
    CHECK(measured.blockers.empty());
    // A stop text the vocabulary does not hold.
    stops.stop = {"<nope>"};
    CHECK(blocked(preflight::preflight(source, qwen3.index, defaults, stops), Stage::Run,
                  "the stop tokens cannot be resolved: stop \"<nope>\" is not a token of this vocabulary"));
    // A byte-level vocabulary missing the byte tokens: listed, and
    // supported, but it does not load.
    const auto bytes = testing::MetadataFile{}
                           .text("tokenizer.ggml.model", "gpt2")
                           .text("tokenizer.ggml.pre", "qwen2")
                           .strings("tokenizer.ggml.tokens", {"a"})
                           .int32s("tokenizer.ggml.token_type", {1})
                           .strings("tokenizer.ggml.merges", {})
                           .bytes();
    gguf::MemoryByteSource broken{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(broken, index).error == gguf::ReadError::Ok);
    const Verdict verdict = preflight::preflight(broken, index, defaults, policy::LoadPolicy{});
    const bool named = std::any_of(verdict.blockers.begin(), verdict.blockers.end(), [](const Blocker& b) {
        return b.stage == Stage::Run && b.detail.starts_with("tokenizer \"gpt2\" does not load from this file: ");
    });
    CHECK(named);
}
