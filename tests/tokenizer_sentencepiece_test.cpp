#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/sentencepiece_bpe.h"
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
const EncodeCase kGemmaCases[] = {
#include "fixtures/tokenizer/gemma-3-1b-it-q4_0.inc"
};

constexpr std::string_view kSpace = "\xE2\x96\x81";   // "▁"

std::vector<std::uint32_t> ids(const std::vector<TokenId>& tokens) {
    std::vector<std::uint32_t> out;
    for (const TokenId t : tokens) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

// A small SentencePiece vocabulary: the 256 byte tokens (identifiers 0 to
// 255), then a, b, ▁, ab, ▁a, ▁ab (256 to 261), each scoring lower than the
// last, then the control token <s> (262).
struct Small {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    std::vector<float> scores;
    std::optional<bool> space_prefix = false;

    Small() {
        for (int byte = 0; byte < 256; ++byte) {
            char name[8];
            std::snprintf(name, sizeof name, "<0x%02X>", byte);
            add(name, 6, 0);
        }
        float score = -1;
        for (const std::string& normal : {std::string("a"), std::string("b"), std::string(kSpace), std::string("ab"),
                                          std::string(kSpace) + "a", std::string(kSpace) + "ab"}) {
            add(normal, 1, score--);
        }
        add("<s>", 3, 0);
    }

    void add(std::string text, std::int32_t type, float score) {
        tokens.push_back(std::move(text));
        types.push_back(type);
        scores.push_back(score);
    }

    LoadResult load(bpe::SentencePieceBpe& out) const {
        MetadataFile file;
        file.strings("tokenizer.ggml.tokens", std::span<const std::string>{tokens})
            .int32s("tokenizer.ggml.token_type", std::span<const std::int32_t>{types})
            .float32s("tokenizer.ggml.scores", std::span<const float>{scores});
        if (space_prefix) file.boolean("tokenizer.ggml.add_space_prefix", *space_prefix);
        const auto bytes = file.bytes();
        gguf::MemoryByteSource source{bytes};
        gguf::TensorIndex index;
        REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
        return bpe::load_sentencepiece_bpe(source, index, out);
    }
};

enum : std::uint32_t { kA = 256, kB, kSp, kAb, kSpA, kSpAb, kControl };

bpe::SentencePieceBpe small() {
    bpe::SentencePieceBpe spm;
    const auto r = Small{}.load(spm);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return spm;
}

std::vector<std::uint32_t> encode(const bpe::SentencePieceBpe& spm, std::string_view text) {
    std::vector<TokenId> tokens;
    REQUIRE(spm.encode(text, kNoTokenLimit, tokens).error == EncodeError::Ok);
    return ids(tokens);
}

bpe::SentencePieceBpe gemma() {
    const auto header = bllm::testing::read_model_header("gemma-3-1b-it-q4_0");
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    bpe::SentencePieceBpe spm;
    const auto r = bpe::load_sentencepiece_bpe(source, header.index, spm);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return spm;
}

std::string decoded(const bpe::SentencePieceBpe& spm, std::uint32_t id) {
    std::string out;
    spm.decode(static_cast<TokenId>(id), out);
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

TEST_CASE("Gemma 3 encodes every fixture text to the references' token IDs") {
    const auto spm = gemma();
    for (const EncodeCase& c : kGemmaCases) {
        CAPTURE(c.name);
        CHECK(encode(spm, c.text) == c.ids);
    }
}

TEST_CASE("Gemma 3 decodes every fixture's token IDs back to its text") {
    // Whole, and a token at a time through a stream that emits only whole
    // characters. Some tokens are single bytes of a character, so the stream
    // has bytes to hold. A typed ▁ comes back as a space: the vocabulary
    // spells both alike.
    const auto spm = gemma();
    std::size_t split_characters = 0;
    for (const EncodeCase& c : kGemmaCases) {
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
            spm.decode(static_cast<TokenId>(id), bytes);
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
    CHECK(split_characters > 0);
}

TEST_CASE("a token decodes to its text with ▁ a space, a byte token to its byte, a special one as written") {
    const auto spm = small();
    CHECK(decoded(spm, kSpAb) == " ab");
    CHECK(decoded(spm, kSp) == " ");
    CHECK(decoded(spm, kA) == "a");
    CHECK(decoded(spm, 0xC3) == "\xC3");
    CHECK(decoded(spm, 0x00) == std::string(1, '\0'));
    CHECK(decoded(spm, kControl) == "<s>");
}

TEST_CASE("spaces become the vocabulary's ▁ and merge with what follows") {
    const auto spm = small();
    CHECK(encode(spm, "ab") == std::vector<std::uint32_t>{kAb});
    CHECK(encode(spm, " ab") == std::vector<std::uint32_t>{kSpAb});
    CHECK(encode(spm, "ab ab") == std::vector<std::uint32_t>{kAb, kSpAb});
    // No "a▁" or "▁b" token: the space stands alone. No space is added first.
    CHECK(encode(spm, "a b") == std::vector<std::uint32_t>{kA, kSp, kB});
}

TEST_CASE("special tokens match after spaces become ▁, so a typed ▁ is one with a space") {
    // Two user-defined runs of spaces and a control token holding one.
    Small with_runs;
    with_runs.add("  ", 4, 0);      // 263
    with_runs.add("   ", 4, 0);     // 264
    with_runs.add("<a b>", 3, 0);   // 265
    bpe::SentencePieceBpe spm;
    const auto r = with_runs.load(spm);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    const std::string bar(kSpace);   // the character ▁, typed
    CHECK(encode(spm, "a  b") == std::vector<std::uint32_t>{kA, 263, kB});
    CHECK(encode(spm, "a " + bar + "b") == std::vector<std::uint32_t>{kA, 263, kB});
    CHECK(encode(spm, "a" + bar + bar + bar + "b") == std::vector<std::uint32_t>{kA, 264, kB});
    // Four in a row: the longest run, then a lone ▁ that merges with what follows.
    CHECK(encode(spm, "a " + bar + bar + " ab") == std::vector<std::uint32_t>{kA, 264, kSpAb});
    CHECK(encode(spm, "a" + bar) == std::vector<std::uint32_t>{kA, kSp});
    CHECK(encode(spm, "<a b>") == std::vector<std::uint32_t>{265});
    CHECK(encode(spm, "<a" + bar + "b>") == std::vector<std::uint32_t>{265});
}

TEST_CASE("a stretch is cut before ▁ only where no merge could cross the cut") {
    // x, y, ▁y (263 to 265) and, where given, x▁y (266): a token holding ▁
    // after its first character, which a merge across the cut would make.
    const auto with = [](bool straddle, int padding) {
        Small v;
        v.add("x", 1, -10);
        v.add("y", 1, -11);
        v.add(std::string(kSpace) + "y", 1, -12);
        if (straddle) v.add("x" + std::string(kSpace) + "y", 1, -13);
        // More tokens holding ▁ after their first character, which no text
        // here spells: past 64 of them, nothing is cut.
        for (int i = 0; i < padding; ++i) v.add("q" + std::string(kSpace) + std::to_string(i), 1, -20.0f - i);
        bpe::SentencePieceBpe spm;
        const auto r = v.load(spm);
        REQUIRE_MESSAGE(r.ok(), r.subject);
        return spm;
    };
    for (const int padding : {0, 64}) {
        CAPTURE(padding);
        // x▁y stands across the cut before ▁, so x and ▁y still merge into it.
        CHECK(encode(with(true, padding), "x y") == std::vector<std::uint32_t>{266});
        CHECK(encode(with(true, padding), "y x y x") == std::vector<std::uint32_t>{264, kSp, 266, kSp, 263});
        // Without it, the cut changes nothing: x and ▁y never merge.
        CHECK(encode(with(false, padding), "x y") == std::vector<std::uint32_t>{263, 265});
    }
}

TEST_CASE("a character the vocabulary lacks becomes a byte token for each of its bytes") {
    const auto spm = small();
    CHECK(encode(spm, "a\xC3\xA9" "b") == std::vector<std::uint32_t>{kA, 0xC3, 0xA9, kB});
    CHECK(encode(spm, "c") == std::vector<std::uint32_t>{'c'});
}

TEST_CASE("a special token's text is its token, and nothing merges across it") {
    const auto spm = small();
    CHECK(encode(spm, "a<s>b") == std::vector<std::uint32_t>{kA, kControl, kB});
}

TEST_CASE("text that is not UTF-8 leaves the output untouched") {
    const auto spm = small();
    std::vector<TokenId> out{static_cast<TokenId>(7)};
    CHECK(spm.encode("ab \xC0\x80", kNoTokenLimit, out).error == EncodeError::InvalidUtf8);
    CHECK(ids(out) == std::vector<std::uint32_t>{7});
}

TEST_CASE("a file the references would read differently from this one is refused") {
    bpe::SentencePieceBpe spm;
    Small asks;
    asks.space_prefix = true;
    auto r = asks.load(spm);
    CHECK(r.error == LoadError::Unsupported);
    CHECK(r.subject == "tokenizer.ggml.add_space_prefix is true");
    Small silent;
    silent.space_prefix.reset();   // llama.cpp reads a missing key as true
    r = silent.load(spm);
    CHECK(r.error == LoadError::Unsupported);
    CHECK(r.subject == "tokenizer.ggml.add_space_prefix is not declared");
    Small no_byte;
    no_byte.types[0x41] = 1;   // <0x41> no longer a byte token
    r = no_byte.load(spm);
    CHECK(r.error == LoadError::MissingByte);
    CHECK(r.subject == "byte <0x41>");
    Small unused;
    unused.add("c", 5, 0);   // an unused token that is one character
    r = unused.load(spm);
    CHECK(r.error == LoadError::Unsupported);
    CHECK(r.subject == "token 263, one character that is not a normal token");
    Small extra_byte;
    extra_byte.add("<0x100>", 6, 0);   // a byte token past the 256
    r = extra_byte.load(spm);
    CHECK(r.error == LoadError::Unsupported);
    CHECK(r.subject == "byte tokens other than <0x00> to <0xFF>");
    Small alike;
    alike.add("  ", 4, 0);
    alike.add(std::string(kSpace) + std::string(kSpace), 4, 0);   // the same, once spaces are ▁
    r = alike.load(spm);
    CHECK(r.error == LoadError::Unsupported);
    CHECK(r.subject == "two special tokens spelled \xE2\x96\x81\xE2\x96\x81 once spaces are \xE2\x96\x81");
    Small short_scores;
    short_scores.scores.pop_back();
    CHECK(short_scores.load(spm).error == LoadError::CountMismatch);
    CHECK(spm.vocabulary().size() == 0);   // untouched
}
