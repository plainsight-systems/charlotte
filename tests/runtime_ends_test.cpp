// An embedder's ends (runtime/ends.h): EmbeddingGemma 2's, from its file, and
// each refusal by name.

#include <doctest/doctest.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/reader.h"
#include "core/runtime/ends.h"
#include "core/tokenizer/vocabulary.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using bllm::testing::MetadataFile;
using runtime::EndsError;

namespace {

struct Resolved {
    runtime::EndsResult result;
    runtime::Ends ends;
};

// A three-token vocabulary, and the file's keys `file` adds.
Resolved resolve_small(MetadataFile file) {
    const std::vector<std::string> texts{"<bos>", "<eos>", "a"};
    const std::vector<std::int32_t> types{3, 3, 1};
    file.strings("tokenizer.ggml.tokens", texts).int32s("tokenizer.ggml.token_type", types);
    const auto bytes = file.bytes();
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    tokenizer::Vocabulary vocabulary;
    REQUIRE(tokenizer::load_vocabulary(source, index, vocabulary).ok());
    Resolved r;
    r.result = runtime::resolve_ends(index, vocabulary, r.ends);
    return r;
}

}  // namespace

TEST_CASE("EmbeddingGemma 2's file adds BOS and EOS, as both references tokenize it") {
    const auto header = testing::read_model_header("embeddinggemma-2-q8_0");
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    tokenizer::Vocabulary vocabulary;
    REQUIRE(tokenizer::load_vocabulary(source, header.index, vocabulary).ok());
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(header.index, vocabulary, ends).error == EndsError::Ok);
    REQUIRE(ends.bos.has_value());
    REQUIRE(ends.eos.has_value());
    CHECK(static_cast<std::uint32_t>(*ends.bos) == 2);
    CHECK(static_cast<std::uint32_t>(*ends.eos) == 1);
}

TEST_CASE("a flag false adds nothing, and each end is resolved on its own") {
    MetadataFile file;
    file.boolean("tokenizer.ggml.add_bos_token", true)
        .uint32("tokenizer.ggml.bos_token_id", 0)
        .boolean("tokenizer.ggml.add_eos_token", false);
    const Resolved r = resolve_small(file);
    REQUIRE(r.result.error == EndsError::Ok);
    CHECK(r.ends.bos == tokenizer::TokenId{0});
    CHECK(!r.ends.eos.has_value());
}

TEST_CASE("an undeclared flag, a flag of another type, a missing or out-of-range token: each refused by name") {
    SUBCASE("a flag undeclared") {
        MetadataFile file;
        file.boolean("tokenizer.ggml.add_bos_token", false);
        const Resolved r = resolve_small(file);
        CHECK(r.result.error == EndsError::Undeclared);
        CHECK(r.result.subject.starts_with("tokenizer.ggml.add_eos_token"));
    }
    SUBCASE("a flag of another type") {
        MetadataFile file;
        file.uint32("tokenizer.ggml.add_bos_token", 1).boolean("tokenizer.ggml.add_eos_token", false);
        CHECK(resolve_small(file).result.error == EndsError::WrongType);
    }
    SUBCASE("a flag true with no token") {
        MetadataFile file;
        file.boolean("tokenizer.ggml.add_bos_token", true).boolean("tokenizer.ggml.add_eos_token", false);
        const Resolved r = resolve_small(file);
        CHECK(r.result.error == EndsError::MissingToken);
        CHECK(r.result.subject.starts_with("tokenizer.ggml.bos_token_id"));
    }
    SUBCASE("a token past the vocabulary") {
        MetadataFile file;
        file.boolean("tokenizer.ggml.add_bos_token", false)
            .boolean("tokenizer.ggml.add_eos_token", true)
            .uint32("tokenizer.ggml.eos_token_id", 3);
        CHECK(resolve_small(file).result.error == EndsError::OutOfVocabulary);
    }
}
