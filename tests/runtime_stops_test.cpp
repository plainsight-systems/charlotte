// The stop set (runtime/stops.h): each listed model's, from its file and the
// stop tokens its generation config adds, and each refusal by name.

#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/reader.h"
#include "core/runtime/stops.h"
#include "core/tokenizer/vocabulary.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using bllm::testing::MetadataFile;
using runtime::StopsError;
using runtime::StopSet;
using tokenizer::TokenId;

namespace {

std::vector<std::uint32_t> ids(const StopSet& set) {
    std::vector<std::uint32_t> out;
    for (const TokenId t : set.tokens()) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

struct Resolved {
    runtime::StopsResult result;
    StopSet set;
};

Resolved resolve_real(std::string_view model, const std::vector<std::string>& policy) {
    const auto header = bllm::testing::read_model_header(model);
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    tokenizer::Vocabulary vocabulary;
    const auto loaded = tokenizer::load_vocabulary(source, header.index, vocabulary);
    REQUIRE_MESSAGE(loaded.ok(), loaded.subject);
    Resolved r;
    r.result = runtime::resolve_stops(header.index, vocabulary, policy, r.set);
    return r;
}

constexpr std::int32_t kNormal = 1, kControl = 3;

// A five-token vocabulary, two of them control tokens, and the file's keys
// `with` adds.
Resolved resolve_small(MetadataFile file, const std::vector<std::string>& policy, std::size_t controls = 2) {
    std::vector<std::string> texts{"a", "b", "c"};
    std::vector<std::int32_t> types{kNormal, kNormal, kNormal};
    for (std::size_t i = 0; i < controls; ++i) {
        texts.push_back("<c" + std::to_string(i) + ">");
        types.push_back(kControl);
    }
    file.strings("tokenizer.ggml.tokens", texts).int32s("tokenizer.ggml.token_type", types);
    const auto bytes = file.bytes();
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    tokenizer::Vocabulary vocabulary;
    REQUIRE(tokenizer::load_vocabulary(source, index, vocabulary).ok());
    Resolved r;
    r.result = runtime::resolve_stops(index, vocabulary, policy, r.set);
    return r;
}

}  // namespace

TEST_CASE("each listed model stops on its file's end of turn, and on what its generation config adds") {
    struct Case {
        std::string_view model;
        std::vector<std::string> policy;
        std::vector<std::uint32_t> file_only, with_policy;
    };
    const Case cases[] = {
        {"qwen3-0.6b-q4_0", {"<|endoftext|>"}, {151645}, {151645, 151643}},
        {"llama-3.2-1b-instruct-q4_0", {"<|eom_id|>", "<|end_of_text|>"}, {128009}, {128009, 128008, 128001}},
        {"gemma-3-1b-it-q4_0", {"<eos>"}, {106}, {106, 1}},
    };
    for (const Case& c : cases) {
        CAPTURE(c.model);
        const Resolved unmeasured = resolve_real(c.model, {});
        REQUIRE_MESSAGE(unmeasured.result.error == StopsError::Ok, unmeasured.result.subject);
        CHECK(ids(unmeasured.set) == c.file_only);
        const Resolved measured = resolve_real(c.model, c.policy);
        REQUIRE_MESSAGE(measured.result.error == StopsError::Ok, measured.result.subject);
        CHECK(ids(measured.set) == c.with_policy);
        for (const std::uint32_t id : c.with_policy) CHECK(measured.set.contains(static_cast<TokenId>(id)));
        CHECK(!measured.set.contains(static_cast<TokenId>(0x10)));
    }
}

TEST_CASE("a stop named by the file and the policy both is held once; eot and eom count") {
    const Resolved r = resolve_small(MetadataFile{}
                                         .uint32("tokenizer.ggml.eos_token_id", 3)
                                         .uint32("tokenizer.ggml.eot_token_id", 4)
                                         .uint32("tokenizer.ggml.eom_token_id", 3),
                                     {"<c0>", "<c1>"});
    REQUIRE(r.result.error == StopsError::Ok);
    CHECK(ids(r.set) == std::vector<std::uint32_t>{3, 4});
}

TEST_CASE("the stop set refuses by name what it cannot hold") {
    const auto refused = [](const Resolved& r, StopsError error, std::string_view names) {
        CHECK(r.result.error == error);
        CHECK_MESSAGE(r.result.subject.find(names) != std::string::npos, r.result.subject);
        CHECK(r.set.tokens().empty());   // left untouched
    };
    refused(resolve_small(MetadataFile{}.boolean("tokenizer.ggml.eos_token_id", true), {}), StopsError::WrongType,
            "tokenizer.ggml.eos_token_id");
    refused(resolve_small(MetadataFile{}.uint32("tokenizer.ggml.eot_token_id", 5), {}), StopsError::OutOfVocabulary,
            "tokenizer.ggml.eot_token_id is 5");
    refused(resolve_small(MetadataFile{}.uint32("tokenizer.ggml.eos_token_id", 3), {"<nope>"}), StopsError::NotAToken,
            "\"<nope>\"");
    // Text spelling two tokens is not one token.
    refused(resolve_small(MetadataFile{}.uint32("tokenizer.ggml.eos_token_id", 3), {"ab"}), StopsError::NotAToken,
            "\"ab\"");
    refused(resolve_small(MetadataFile{}.uint32("tokenizer.ggml.eos_token_id", 3), {"b"}), StopsError::NotControl,
            "\"b\"");
    refused(resolve_small(MetadataFile{}.uint32("tokenizer.ggml.padding_token_id", 3), {}), StopsError::None,
            "no end-of-generation token");
    std::vector<std::string> nine;
    for (int i = 0; i < 9; ++i) nine.push_back("<c" + std::to_string(i) + ">");
    refused(resolve_small(MetadataFile{}, nine, 9), StopsError::TooMany, "9 stop tokens");
}
