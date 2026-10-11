#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/arrays.h"
#include "core/gguf/reader.h"
#include "support/gguf_fixture.h"
#include "support/metadata_file.h"

using namespace bllm::gguf;

namespace {

using bllm::testing::load_gguf_fixture;

TensorIndex read_fixture(const std::vector<std::byte>& bytes) {
    MemoryByteSource source{bytes};
    TensorIndex index;
    REQUIRE(read_index(source, index).error == ReadError::Ok);
    return index;
}

ArrayLocation array(const TensorIndex& index, std::string_view key) {
    ArrayLocation location{};
    REQUIRE(index.read_array(key, location) == MetadataError::Ok);
    return location;
}

// A file holding one string array of `count` elements, "m0", "m1", ... Built
// here rather than committed: at a vocabulary's size it is megabytes.
std::vector<std::byte> one_string_array(std::uint64_t count) {
    std::vector<std::byte> out;
    const auto put = [&](const void* p, std::size_t n) {
        const auto* b = static_cast<const std::byte*>(p);
        out.insert(out.end(), b, b + n);
    };
    const auto u32 = [&](std::uint32_t v) { put(&v, sizeof v); };
    const auto u64 = [&](std::uint64_t v) { put(&v, sizeof v); };
    const auto str = [&](std::string_view s) {
        u64(s.size());
        put(s.data(), s.size());
    };
    put("GGUF", 4);
    u32(3);   // version
    u64(0);   // tensors
    u64(1);   // metadata entries
    str("tokenizer.ggml.merges");
    u32(static_cast<std::uint32_t>(ValueType::Array));
    u32(static_cast<std::uint32_t>(ValueType::String));
    u64(count);
    for (std::uint64_t i = 0; i < count; ++i) str("m" + std::to_string(i));
    // The (empty) tensor data starts at the next multiple of the default
    // alignment, which must lie inside the file.
    out.resize((out.size() + kDefaultAlignment - 1) / kDefaultAlignment * kDefaultAlignment);
    return out;
}

}  // namespace

TEST_CASE("a string array is read whole, empty and multi-byte elements included") {
    const auto bytes = load_gguf_fixture("token_arrays");
    MemoryByteSource source{bytes};
    const TensorIndex index = read_fixture(bytes);

    StringTable tokens;
    REQUIRE(read_strings(source, array(index, "tokenizer.ggml.tokens"), tokens).error == ReadError::Ok);
    REQUIRE(tokens.size() == 4);
    CHECK(tokens[0].empty());
    CHECK(tokens[1] == "a");
    CHECK(tokens[2] == "h\xc3\xa9llo");
    CHECK(tokens[3] == "<|x|>");
}

TEST_CASE("int32 and float32 arrays are read element for element") {
    const auto bytes = load_gguf_fixture("token_arrays");
    MemoryByteSource source{bytes};
    const TensorIndex index = read_fixture(bytes);

    std::vector<std::int32_t> types;
    REQUIRE(read_int32s(source, array(index, "tokenizer.ggml.token_type"), types).error == ReadError::Ok);
    CHECK(types == std::vector<std::int32_t>{3, 1, 1, -4});

    std::vector<float> scores;
    REQUIRE(read_float32s(source, array(index, "tokenizer.ggml.scores"), scores).error == ReadError::Ok);
    CHECK(scores == std::vector<float>{-1000.0f, -1.5f, 0.0f, 2.25f});
}

TEST_CASE("an array read as another type is refused, and the output is untouched") {
    const auto bytes = load_gguf_fixture("token_arrays");
    MemoryByteSource source{bytes};
    const TensorIndex index = read_fixture(bytes);
    const auto tokens = array(index, "tokenizer.ggml.tokens");
    const auto types = array(index, "tokenizer.ggml.token_type");
    const auto scores = array(index, "tokenizer.ggml.scores");

    StringTable table;
    REQUIRE(read_strings(source, tokens, table).error == ReadError::Ok);
    CHECK(read_strings(source, scores, table).error == ReadError::WrongElementType);
    CHECK(table.size() == 4);

    std::vector<std::int32_t> ints{7};
    CHECK(read_int32s(source, tokens, ints).error == ReadError::WrongElementType);
    CHECK(read_int32s(source, scores, ints).error == ReadError::WrongElementType);
    CHECK(ints == std::vector<std::int32_t>{7});

    std::vector<float> floats{7.0f};
    CHECK(read_float32s(source, types, floats).error == ReadError::WrongElementType);
    CHECK(floats == std::vector<float>{7.0f});
}

TEST_CASE("an array not wholly resident asks for the bytes that end it") {
    const auto bytes = load_gguf_fixture("token_arrays");
    const TensorIndex index = read_fixture(bytes);
    const auto tokens = array(index, "tokenizer.ggml.tokens");
    const auto types = array(index, "tokenizer.ggml.token_type");

    for (const ArrayLocation& location : {tokens, types}) {
        // Resident up to one byte short of the array's end.
        const std::uint64_t end = location.bytes.offset + location.bytes.length;
        MemoryByteSource partial{std::span{bytes}.first(static_cast<std::size_t>(end - 1)), bytes.size()};
        StringTable table;
        std::vector<std::int32_t> ints;
        const ReadResult r = location.element_type == ValueType::String ? read_strings(partial, location, table)
                                                                       : read_int32s(partial, location, ints);
        CHECK(r.error == ReadError::NeedMoreBytes);
        CHECK(r.bytes_needed == end);
    }
}

TEST_CASE("a vocabulary-sized string array is read in one pass") {
    // Llama 3.2 has 280,147 merges. Reading them one element at a time from
    // the array's start would be tens of billions of skips; one pass is
    // milliseconds.
    constexpr std::uint64_t kCount = 300'000;
    const auto bytes = one_string_array(kCount);
    MemoryByteSource source{bytes};
    const TensorIndex index = read_fixture(bytes);

    StringTable merges;
    REQUIRE(read_strings(source, array(index, "tokenizer.ggml.merges"), merges).error == ReadError::Ok);
    REQUIRE(merges.size() == kCount);
    CHECK(merges[0] == "m0");
    CHECK(merges[150'000] == "m150000");
    CHECK(merges[kCount - 1] == "m299999");
}

namespace {

TensorIndex read_metadata(const bllm::testing::MetadataFile& file) {
    const auto bytes = file.bytes();
    MemoryByteSource source{bytes};
    TensorIndex index;
    REQUIRE(read_index(source, index).error == ReadError::Ok);
    return index;
}

}  // namespace

TEST_CASE("a short array of counts or flags is decoded at read, as a value a layer is") {
    bllm::testing::MetadataFile file;
    file.int32s("heads", {2, 2, 1, 2})
        .uint32s("widths", {256, 512})
        .booleans("sliding", {true, true, false})
        .int32s("negative", {2, -1})
        .uint32("scalar", 7);
    const TensorIndex index = read_metadata(file);
    std::vector<std::uint32_t> counts;
    REQUIRE(index.read_u32_array("heads", counts) == MetadataError::Ok);
    CHECK(counts == std::vector<std::uint32_t>{2, 2, 1, 2});
    REQUIRE(index.read_u32_array("widths", counts) == MetadataError::Ok);
    CHECK(counts == std::vector<std::uint32_t>{256, 512});
    std::vector<bool> flags;
    REQUIRE(index.read_bool_array("sliding", flags) == MetadataError::Ok);
    CHECK(flags == std::vector<bool>{true, true, false});
    // A negative count, a scalar, flags read as counts, a missing key: each
    // refused, the output untouched.
    std::vector<std::uint32_t> untouched{9};
    CHECK(index.read_u32_array("negative", untouched) == MetadataError::WrongType);
    CHECK(index.read_u32_array("scalar", untouched) == MetadataError::WrongType);
    CHECK(index.read_u32_array("sliding", untouched) == MetadataError::WrongType);
    CHECK(index.read_u32_array("absent", untouched) == MetadataError::MissingKey);
    CHECK(untouched == std::vector<std::uint32_t>{9});
    CHECK(index.read_bool_array("heads", flags) == MetadataError::WrongType);
}

TEST_CASE("an array of numbers longer than kMaxDecodedArray is located, not decoded") {
    std::vector<std::int32_t> many(kMaxDecodedArray + 1, 3);
    bllm::testing::MetadataFile file;
    file.int32s("many", std::span<const std::int32_t>{many});
    const TensorIndex index = read_metadata(file);
    std::vector<std::uint32_t> counts;
    CHECK(index.read_u32_array("many", counts) == MetadataError::WrongType);
    ArrayLocation location{};
    REQUIRE(index.read_array("many", location) == MetadataError::Ok);
    CHECK(location.element_count == kMaxDecodedArray + 1);
    CHECK(location.decoded.empty());
}
