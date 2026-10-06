#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/reader.h"
#include "support/gguf_fixture.h"
#include "support/model_headers.h"

using namespace bllm::gguf;

namespace {

using bllm::testing::load_gguf_fixture;

// Reads a whole fixture. The bytes outlive the source because the caller
// keeps them.
TensorIndex read_fixture(const std::vector<std::byte>& bytes) {
    MemoryByteSource source{bytes};
    TensorIndex index;
    REQUIRE(read_index(source, index).error == ReadError::Ok);
    return index;
}

}  // namespace

TEST_CASE("a valid file yields the expected tensors") {
    const auto bytes = load_gguf_fixture("valid");
    const TensorIndex index = read_fixture(bytes);

    REQUIRE(index.tensors().size() == 2);
    const auto& embd = index.tensors()[0];
    CHECK(embd.name == "token_embd.weight");
    CHECK(embd.type == TensorType::Q4_0);
    CHECK(embd.dimension_count == 2);
    CHECK(embd.dimensions[0] == 64);
    CHECK(embd.dimensions[1] == 2);
    CHECK(embd.element_count == 128);
    // 128 elements = 4 blocks = 4 * 18 bytes.
    CHECK(embd.data_length == 72);
    CHECK(embd.is_quantized());

    const auto& norm = index.tensors()[1];
    CHECK(norm.name == "output_norm.weight");
    CHECK(norm.type == TensorType::F32);
    CHECK(norm.element_count == 4);
    CHECK(norm.data_length == 16);
    CHECK_FALSE(norm.is_quantized());
}

TEST_CASE("tensors are found by name") {
    const auto bytes = load_gguf_fixture("valid");
    const TensorIndex index = read_fixture(bytes);

    const auto id = index.find("output_norm.weight");
    REQUIRE(id.has_value());
    CHECK(index.tensor(*id).name == "output_norm.weight");
    CHECK_FALSE(index.find("does.not.exist").has_value());
}

TEST_CASE("every tensor of a real model is found at its own id, and no other name is") {
    const bllm::testing::ReadHeader header = bllm::testing::read_model_header("gemma-3-1b-it-q4_0");
    const TensorIndex& index = header.index;
    REQUIRE(index.tensors().size() == 340);
    for (std::size_t i = 0; i < index.tensors().size(); ++i) {
        const auto id = index.find(index.tensors()[i].name);
        REQUIRE(id.has_value());
        CHECK(static_cast<std::size_t>(*id) == i);
    }
    // Before every name, after every name, and between two.
    for (const char* absent : {"", "a", "zzz", "blk.0.attn_k.weightx", "blk.0.attn_k", "blk.10"}) {
        CAPTURE(absent);
        CHECK_FALSE(index.find(absent).has_value());
    }
}

TEST_CASE("scalar and string metadata is decoded and typed") {
    const auto bytes = load_gguf_fixture("valid");
    const TensorIndex index = read_fixture(bytes);

    std::string_view architecture;
    CHECK(index.read_string("general.architecture", architecture) == MetadataError::Ok);
    CHECK(architecture == "qwen3");

    std::uint32_t blocks = 0;
    CHECK(index.read_u32("qwen3.block_count", blocks) == MetadataError::Ok);
    CHECK(blocks == 28);

    // A missing key and a key of the wrong type are different defects.
    CHECK(index.read_u32("general.architecture", blocks) == MetadataError::WrongType);
    CHECK(index.read_u32("qwen3.missing", blocks) == MetadataError::MissingKey);
    std::uint64_t wide = 0;
    CHECK(index.read_u64("qwen3.block_count", wide) == MetadataError::WrongType);
}

TEST_CASE("arrays are located, not decoded") {
    const auto bytes = load_gguf_fixture("valid");
    const TensorIndex index = read_fixture(bytes);

    ArrayLocation tokens{};
    REQUIRE(index.read_array("tokenizer.ggml.tokens", tokens) == MetadataError::Ok);
    CHECK(tokens.element_type == ValueType::String);
    CHECK(tokens.element_count == 3);
    // Three strings, each a u64 length and its bytes: "a", "bb", "ccc".
    CHECK(tokens.bytes.length == 3 * 8 + 1 + 2 + 3);
    CHECK(tokens.bytes.offset + tokens.bytes.length < bytes.size());
}

TEST_CASE("tensor offsets are absolute, aligned, and inside the file") {
    const auto bytes = load_gguf_fixture("valid");
    const TensorIndex index = read_fixture(bytes);

    for (const auto& t : index.tensors()) {
        CAPTURE(t.name);
        CHECK(t.data_offset % kDefaultAlignment == 0);
        CHECK(t.data_offset + t.data_length <= bytes.size());
    }
}

TEST_CASE("a string array's elements are read by position") {
    const auto bytes = load_gguf_fixture("valid");
    MemoryByteSource source{bytes};
    const TensorIndex index = read_fixture(bytes);
    ArrayLocation tokens{};
    REQUIRE(index.read_array("tokenizer.ggml.tokens", tokens) == MetadataError::Ok);

    std::string text;
    CHECK(read_string_element(source, tokens, 0, text).error == ReadError::Ok);
    CHECK(text == "a");
    CHECK(read_string_element(source, tokens, 2, text).error == ReadError::Ok);
    CHECK(text == "ccc");
    CHECK(read_string_element(source, tokens, 3, text).error == ReadError::ShortRead);
}

TEST_CASE("a declared alignment other than the default is honoured") {
    const auto bytes = load_gguf_fixture("alignment_64");
    const TensorIndex index = read_fixture(bytes);
    REQUIRE(index.tensors().size() == 2);
    for (const auto& t : index.tensors()) {
        CAPTURE(t.name);
        CHECK(t.data_offset % 64 == 0);
    }
    // Two 16-byte tensors, each padded to the declared 64.
    CHECK(index.tensors()[1].data_offset - index.tensors()[0].data_offset == 64);
}

TEST_CASE("a tensor name of exactly the format's 64-byte limit is accepted") {
    const auto bytes = load_gguf_fixture("tensor_name_64_bytes");
    const TensorIndex index = read_fixture(bytes);
    REQUIRE(index.tensors().size() == 1);
    CHECK(index.tensors()[0].name.size() == 64);
}
