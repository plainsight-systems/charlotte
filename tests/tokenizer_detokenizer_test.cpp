// The detokenizer (tokenizer/detokenizer.h): chosen tokens of a real
// vocabulary as the whole characters they complete, and U+FFFD for bytes a
// reply ends inside.

#include <doctest/doctest.h>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/preflight/preflight.h"
#include "core/tokenizer/detokenizer.h"
#include "core/tokenizer/vocabulary.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;

namespace {

std::unique_ptr<Tokenizer> load(std::string_view model) {
    const auto header = bllm::testing::read_model_header(model);
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    std::unique_ptr<Tokenizer> t;
    const std::string stop = preflight::load_tokenizer(source, header.index, t);
    REQUIRE_MESSAGE(stop.empty(), stop);
    return t;
}

TokenId find(const Tokenizer& t, std::string_view text) {
    const auto id = t.vocabulary().find(text);
    REQUIRE_MESSAGE(id.has_value(), text);
    return *id;
}

}  // namespace

TEST_CASE("a token completing no character gives nothing, and the one that finishes it gives it whole") {
    // Gemma 3's byte tokens spell 🦀, F0 9F A6 80, a byte at a time.
    const auto gemma = load("gemma-3-1b-it-q4_0");
    Detokenizer d{*gemma};
    CHECK(d.push(find(*gemma, "<0xF0>")).empty());
    CHECK(d.push(find(*gemma, "<0x9F>")).empty());
    CHECK(d.push(find(*gemma, "<0xA6>")).empty());
    CHECK(d.push(find(*gemma, "<0x80>")) == "🦀");
    // A whole token, a space before it as SentencePiece spells one.
    CHECK(d.push(find(*gemma, "▁the")) == " the");
    CHECK(d.finish().empty());   // nothing pending
}

TEST_CASE("a reply that ends inside a character ends with U+FFFD, and the stream is then empty") {
    const auto gemma = load("gemma-3-1b-it-q4_0");
    Detokenizer d{*gemma};
    CHECK(d.push(find(*gemma, "Hi")) == "Hi");
    CHECK(d.push(find(*gemma, "<0xF0>")).empty());
    CHECK(d.push(find(*gemma, "<0x9F>")).empty());
    CHECK(d.finish() == "\xEF\xBF\xBD");
    // The next reply starts clean.
    CHECK(d.push(find(*gemma, "Hi")) == "Hi");
    CHECK(d.finish().empty());
}

TEST_CASE("byte-level tokens decode through the same stream") {
    // Qwen3 spells a space then the first three bytes of an emoji as one
    // token, the last byte as another: the space comes at once, the emoji
    // with the second.
    const auto qwen = load("qwen3-0.6b-q4_0");
    Detokenizer d{*qwen};
    std::vector<TokenId> tokens;
    REQUIRE(qwen->encode(" 😊", kNoTokenLimit, tokens).error == EncodeError::Ok);
    std::string text;
    for (const TokenId t : tokens) text += d.push(t);
    CHECK(text == " 😊");
    CHECK(d.longest() >= 1);
}
