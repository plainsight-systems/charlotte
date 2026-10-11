#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/gemma4_bpe.h"
#include "core/tokenizer/unicode.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;
using bllm::testing::MetadataFile;

namespace {

struct EncodeCase {
    std::string_view name;
    std::string_view text;
    std::vector<std::uint32_t> ids;
};

// Each text and the token IDs Hugging Face tokenizers and llama.cpp both
// encode it to (tools/make_tokenizer_fixtures.py).
const EncodeCase kCases[] = {
#include "fixtures/tokenizer/embeddinggemma-2-q8_0.inc"
};

constexpr std::string_view kSpace = "\xE2\x96\x81";   // "▁"

std::vector<std::uint32_t> ids(const std::vector<TokenId>& tokens) {
    std::vector<std::uint32_t> out;
    for (const TokenId t : tokens) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

// A small gemma4 vocabulary: the 256 byte tokens (identifiers 0 to 255),
// then a, b, ▁, \n, ab, ▁a, \n\n (256 to 262), then the control token
// <s ep> (263), whose text holds a space, with merges making ab, ▁a and
// \n\n, ab first.
struct Small {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    std::vector<std::string> merges{"a b", std::string(kSpace) + " a", "\n \n"};
    std::optional<bool> space_prefix = false;
    bool with_merges = true;

    Small() {
        for (int byte = 0; byte < 256; ++byte) {
            char name[8];
            std::snprintf(name, sizeof name, "<0x%02X>", byte);
            add(name, 6);
        }
        for (const std::string& normal : {std::string("a"), std::string("b"), std::string(kSpace), std::string("\n"),
                                          std::string("ab"), std::string(kSpace) + "a", std::string("\n\n")}) {
            add(normal, 1);
        }
        add("<s ep>", 3);
    }

    void add(std::string text, std::int32_t type) {
        tokens.push_back(std::move(text));
        types.push_back(type);
    }

    LoadResult load(bpe::Gemma4Bpe& out) const {
        MetadataFile file;
        file.strings("tokenizer.ggml.tokens", std::span<const std::string>{tokens})
            .int32s("tokenizer.ggml.token_type", std::span<const std::int32_t>{types});
        if (with_merges) file.strings("tokenizer.ggml.merges", std::span<const std::string>{merges});
        if (space_prefix) file.boolean("tokenizer.ggml.add_space_prefix", *space_prefix);
        const auto bytes = file.bytes();
        gguf::MemoryByteSource source{bytes};
        gguf::TensorIndex index;
        REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
        return bpe::load_gemma4_bpe(source, index, out);
    }
};

enum : std::uint32_t { kA = 256, kB, kSp, kNl, kAb, kSpA, kNlNl, kControl };

bpe::Gemma4Bpe small() {
    bpe::Gemma4Bpe g;
    const auto r = Small{}.load(g);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return g;
}

std::vector<std::uint32_t> encode(const bpe::Gemma4Bpe& g, std::string_view text) {
    std::vector<TokenId> tokens;
    REQUIRE(g.encode(text, kNoTokenLimit, tokens).error == EncodeError::Ok);
    return ids(tokens);
}

bpe::Gemma4Bpe embedding_gemma() {
    const auto header = bllm::testing::read_model_header("embeddinggemma-2-q8_0");
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    bpe::Gemma4Bpe g;
    const auto r = bpe::load_gemma4_bpe(source, header.index, g);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return g;
}

std::string decoded(const bpe::Gemma4Bpe& g, std::uint32_t id) {
    std::string out;
    g.decode(static_cast<TokenId>(id), out);
    return out;
}

bool well_formed(std::string_view s) {
    for (std::size_t at = 0; at < s.size();) {
        Utf8Char c{};
        if (!decode_utf8(s, at, c)) return false;
        at += c.length;
    }
    return true;
}

}  // namespace

TEST_CASE("EmbeddingGemma 2 encodes every fixture text to the references' token IDs") {
    const auto g = embedding_gemma();
    for (const EncodeCase& c : kCases) {
        CAPTURE(c.name);
        CHECK(encode(g, c.text) == c.ids);
    }
}

TEST_CASE("EmbeddingGemma 2 decodes every fixture's token IDs back to its text, a token at a time") {
    // A typed ▁ comes back as a space: the vocabulary spells both alike.
    const auto g = embedding_gemma();
    for (const EncodeCase& c : kCases) {
        CAPTURE(c.name);
        std::string expected;
        for (std::size_t at = 0; at < c.text.size();) {
            if (c.text.substr(at, kSpace.size()) == kSpace) {
                expected.push_back(' ');
                at += kSpace.size();
            } else {
                expected.push_back(c.text[at++]);
            }
        }
        std::string whole;
        std::string streamed;
        Utf8Stream stream;
        for (const std::uint32_t id : c.ids) {
            std::string bytes;
            g.decode(static_cast<TokenId>(id), bytes);
            whole += bytes;
            std::string emitted;
            stream.push(bytes, emitted);
            CHECK(well_formed(emitted));
            streamed += emitted;
        }
        CHECK(stream.finish(streamed));
        CHECK(whole == expected);
        CHECK(streamed == expected);
    }
}

TEST_CASE("EmbeddingGemma 2's longest cover is its longest token's spelling") {
    const auto g = embedding_gemma();
    std::size_t longest = 1;
    for (std::size_t id = 0; id < g.vocabulary().size(); ++id) {
        longest = std::max(longest, g.vocabulary().text(static_cast<TokenId>(id)).size());
    }
    CHECK(g.longest_cover() == longest);
}

TEST_CASE("spaces become ▁ and merge by the listed ranks") {
    const auto g = small();
    CHECK(encode(g, "ab a") == std::vector<std::uint32_t>{kAb, kSpA});
    // ab ranks before ▁a, so it takes the a first, and nothing joins ▁ to ab.
    CHECK(encode(g, " ab") == std::vector<std::uint32_t>{kSp, kAb});
}

TEST_CASE("a special token is matched in the raw text, before its space is respelled") {
    const auto g = small();
    CHECK(encode(g, "a<s ep>b") == std::vector<std::uint32_t>{kA, kControl, kB});
}

TEST_CASE("a run of newlines that is a token is that token; text is cut at line breaks") {
    const auto g = small();
    CHECK(encode(g, "a\n\nb") == std::vector<std::uint32_t>{kA, kNlNl, kB});
    CHECK(encode(g, "a\n\n\nb") == std::vector<std::uint32_t>{kA, kNlNl, kNl, kB});
    CHECK(encode(g, "a\nb") == std::vector<std::uint32_t>{kA, kNl, kB});
}

TEST_CASE("a character the vocabulary lacks becomes a byte token for each of its bytes") {
    const auto g = small();
    CHECK(encode(g, "a\xC3\xA9") == std::vector<std::uint32_t>{kA, 0xC3, 0xA9});
}

TEST_CASE("a token decodes to its text with ▁ a space, a byte token to its byte, a special one as written") {
    const auto g = small();
    CHECK(decoded(g, kSpA) == " a");
    CHECK(decoded(g, 0x41) == "A");
    CHECK(decoded(g, kControl) == "<s ep>");
}

TEST_CASE("text that is not UTF-8 leaves the output untouched") {
    const auto g = small();
    std::vector<TokenId> tokens{static_cast<TokenId>(7)};
    CHECK(g.encode("a\xFF", kNoTokenLimit, tokens).error == EncodeError::InvalidUtf8);
    CHECK(tokens.size() == 1);
}

TEST_CASE("a file this algorithm cannot read as the references do is refused, named") {
    bpe::Gemma4Bpe g;
    SUBCASE("no merges") {
        Small s;
        s.with_merges = false;
        const auto r = s.load(g);
        CHECK(r.error == LoadError::MissingKey);
        CHECK(r.subject == "tokenizer.ggml.merges");
    }
    SUBCASE("a space before the text") {
        Small s;
        s.space_prefix = true;
        CHECK(s.load(g).error == LoadError::Unsupported);
    }
    SUBCASE("a space prefix not declared") {
        Small s;
        s.space_prefix.reset();
        CHECK(s.load(g).error == LoadError::Unsupported);
    }
    SUBCASE("a merge naming a token the vocabulary lacks") {
        Small s;
        s.merges.push_back("a c");
        CHECK(s.load(g).error == LoadError::UnknownMergeToken);
    }
    SUBCASE("a byte token missing") {
        Small s;
        s.types[0x41] = 1;   // <0x41> a normal token
        CHECK(s.load(g).error == LoadError::MissingByte);
    }
}
