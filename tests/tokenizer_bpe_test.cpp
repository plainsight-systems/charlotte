#include <doctest/doctest.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/byte_level_bpe.h"
#include "core/tokenizer/bpe/byte_map.h"
#include "core/tokenizer/bpe/piece_cache.h"
#include "core/tokenizer/nfc.h"
#include "core/tokenizer/pretokenize.h"
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
// encode it to, or the one DECISIONS follows (tools/make_tokenizer_fixtures.py).
const EncodeCase kQwen3Cases[] = {
#include "fixtures/tokenizer/qwen3-0.6b-q4_0.inc"
};
const EncodeCase kLlamaCases[] = {
#include "fixtures/tokenizer/llama-3.2-1b-instruct-q4_0.inc"
};

bpe::ByteLevelBpe load(std::string_view model, const PreTokenizer& pretokenizer) {
    const auto header = bllm::testing::read_model_header(model);
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    bpe::ByteLevelBpe bpe;
    const auto r = bpe::load_byte_level_bpe(source, header.index, pretokenizer, bpe);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return bpe;
}

std::vector<std::uint32_t> encode(const bpe::ByteLevelBpe& bpe, std::string_view text) {
    std::vector<TokenId> tokens;
    REQUIRE(bpe.encode(text, kNoTokenLimit, tokens).error == EncodeError::Ok);
    std::vector<std::uint32_t> ids;
    for (const TokenId t : tokens) ids.push_back(static_cast<std::uint32_t>(t));
    return ids;
}

std::vector<std::uint32_t> encode(const bpe::ByteLevelBpe& bpe, std::string_view text, bpe::PieceCache& cache) {
    std::vector<TokenId> tokens;
    REQUIRE(bpe.encode(text, kNoTokenLimit, tokens, cache).error == EncodeError::Ok);
    std::vector<std::uint32_t> ids;
    for (const TokenId t : tokens) ids.push_back(static_cast<std::uint32_t>(t));
    return ids;
}

// Every case without a cache, then three times through one cache (cold, then
// warm), then through a one-slot cache whose every piece evicts the last.
void check_cases(const bpe::ByteLevelBpe& bpe, std::span<const EncodeCase> cases) {
    bpe::PieceCache cache;
    bpe::PieceCache tiny{1};
    for (const EncodeCase& c : cases) {
        CAPTURE(c.name);
        CHECK(encode(bpe, c.text) == c.ids);
        for (int pass = 0; pass < 3; ++pass) CHECK(encode(bpe, c.text, cache) == c.ids);
        CHECK(encode(bpe, c.text, tiny) == c.ids);
    }
}

bool well_formed(std::string_view s) {
    for (std::size_t at = 0; at < s.size();) {
        Utf8Char c{};
        if (!decode_utf8(s, at, c)) return false;
        at += c.length;
    }
    return true;
}

// Every case's reference IDs decode back to its text, NFC-normalized where the
// tokenizer normalizes: whole, and a token at a time through a stream that
// emits only whole characters. Returns how many tokens ended partway through a
// character, so a test can show the stream had characters to hold.
std::size_t check_round_trips(const bpe::ByteLevelBpe& bpe, std::span<const EncodeCase> cases, bool nfc) {
    std::size_t split_characters = 0;
    for (const EncodeCase& c : cases) {
        CAPTURE(c.name);
        std::string expected(c.text);
        if (nfc) REQUIRE(to_nfc(c.text, expected));
        std::string whole;
        std::string streamed;
        Utf8Stream stream;
        for (const std::uint32_t id : c.ids) {
            std::string bytes;
            bpe.decode(static_cast<TokenId>(id), bytes);
            whole += bytes;
            if (!well_formed(whole)) ++split_characters;
            std::string emitted;
            stream.push(bytes, emitted);
            CHECK(well_formed(emitted));
            streamed += emitted;
        }
        CHECK(stream.finish(streamed));
        CHECK(whole == expected);
        CHECK(streamed == expected);
    }
    return split_characters;
}

// A vocabulary of the 256 byte tokens, as normal tokens, and then `token` of
// `type`, as identifier 256. No merges.
LoadResult load_small(std::string_view token, TokenType type, bpe::ByteLevelBpe& out) {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    for (const char32_t c : bpe::kByteChars) {
        tokens.emplace_back();
        append_utf8(c, tokens.back());
        types.push_back(static_cast<std::int32_t>(TokenType::Normal));
    }
    tokens.emplace_back(token);
    types.push_back(static_cast<std::int32_t>(type));
    const auto bytes = MetadataFile{}
                           .strings("tokenizer.ggml.tokens", std::span<const std::string>{tokens})
                           .int32s("tokenizer.ggml.token_type", std::span<const std::int32_t>{types})
                           .strings("tokenizer.ggml.merges", {})
                           .bytes();
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    return bpe::load_byte_level_bpe(source, index, kQwen2, out);
}

std::string decoded(const bpe::ByteLevelBpe& bpe, std::uint32_t id) {
    std::string out;
    bpe.decode(static_cast<TokenId>(id), out);
    return out;
}

}  // namespace

TEST_CASE("Qwen3 encodes every fixture text to the references' token IDs") {
    check_cases(load("qwen3-0.6b-q4_0", kQwen2), kQwen3Cases);
}

TEST_CASE("Llama 3.2 encodes every fixture text to the references' token IDs") {
    check_cases(load("llama-3.2-1b-instruct-q4_0", kLlamaBpe), kLlamaCases);
}

TEST_CASE("Qwen3 decodes every fixture's token IDs back to its text, normalized") {
    // Some tokens end inside a character, so the stream has bytes to hold.
    CHECK(check_round_trips(load("qwen3-0.6b-q4_0", kQwen2), kQwen3Cases, true) > 0);
}

TEST_CASE("Llama 3.2 decodes every fixture's token IDs back to its text") {
    CHECK(check_round_trips(load("llama-3.2-1b-instruct-q4_0", kLlamaBpe), kLlamaCases, false) > 0);
}

TEST_CASE("a normal token decodes through the byte map, any other as its text") {
    // A space is not in the byte map, which spells it "Ġ"; "Ġ" stands for it.
    bpe::ByteLevelBpe bpe;
    REQUIRE(load_small("<a b>", TokenType::Control, bpe).ok());
    CHECK(decoded(bpe, ' ') == " ");
    CHECK(decoded(bpe, 0xFF) == "\xFF");   // one byte of a character
    CHECK(decoded(bpe, 256) == "<a b>");
    REQUIRE(load_small("\xC4\xA0" "a", TokenType::Normal, bpe).ok());   // "Ġa"
    CHECK(decoded(bpe, 256) == " a");
}

TEST_CASE("Qwen3's tokens that are not normal decode to their text as written") {
    const auto bpe = load("qwen3-0.6b-q4_0", kQwen2);
    const auto& vocabulary = bpe.vocabulary();
    for (const std::string_view text : {"<|im_end|>", "<think>", "[PAD151669]"}) {
        CAPTURE(text);
        const auto token = vocabulary.find(text);
        REQUIRE(token.has_value());
        CHECK(vocabulary.type(*token) != TokenType::Normal);
        std::string out = "x";
        bpe.decode(*token, out);
        CHECK(out == "x" + std::string(text));
    }
}

TEST_CASE("a normal token holding a character no byte stands for is refused at load") {
    bpe::ByteLevelBpe bpe;
    const auto r = load_small("a b", TokenType::Normal, bpe);
    CHECK(r.error == LoadError::UnmappedCharacter);
    CHECK(r.subject == "a b");
    CHECK(bpe.vocabulary().size() == 0);   // untouched
    // The same text in a user-defined token is its text as written.
    CHECK(load_small("a b", TokenType::UserDefined, bpe).ok());
}

TEST_CASE("encoding appends, and text that is not UTF-8 leaves the output untouched") {
    const auto bpe = load("qwen3-0.6b-q4_0", kQwen2);
    std::vector<TokenId> out{static_cast<TokenId>(7)};
    REQUIRE(bpe.encode("Hello", kNoTokenLimit, out).error == EncodeError::Ok);
    CHECK(out.size() == 2);
    CHECK(out[0] == static_cast<TokenId>(7));
    CHECK(bpe.encode("ok \xC0\x80", kNoTokenLimit, out).error == EncodeError::InvalidUtf8);
    CHECK(out.size() == 2);
}

TEST_CASE("a cache another tokenizer filled is emptied, never trusted") {
    const auto qwen = load("qwen3-0.6b-q4_0", kQwen2);
    const auto llama = load("llama-3.2-1b-instruct-q4_0", kLlamaBpe);
    bpe::PieceCache cache;
    for (const EncodeCase& c : kQwen3Cases) (void)encode(qwen, c.text, cache);
    // The same words, now Llama's: every one must be Llama's own tokens.
    for (const EncodeCase& c : kLlamaCases) {
        CAPTURE(c.name);
        CHECK(encode(llama, c.text, cache) == c.ids);
    }
}
