// The tokenizer interface (tokenizer/tokenizer.h): each listed model's
// tokenizer loaded through the capability table's algorithm, as a load does,
// encoding and decoding as its algorithm's encoder does directly.

#include <doctest/doctest.h>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/byte_source.h"
#include "core/tokenizer/bpe/byte_level_bpe.h"
#include "core/tokenizer/bpe/sentencepiece_bpe.h"
#include "core/tokenizer/pretokenize.h"
#include "core/tokenizer/tokenizer.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;

namespace {

struct Loaded {
    bllm::testing::ReadHeader header;
    std::unique_ptr<Tokenizer> tokenizer;
};

// Through the table, by the file's own keys.
Loaded load_through_table(std::string_view model) {
    Loaded l{bllm::testing::read_model_header(model), nullptr};
    std::string_view name, pre;
    REQUIRE(l.header.index.read_string("tokenizer.ggml.model", name) == gguf::MetadataError::Ok);
    const Algorithm* algorithm = capability::find_tokenizer(name);
    REQUIRE(algorithm != nullptr);
    REQUIRE(algorithm->load != nullptr);
    const PreTokenizer* pretokenizer = nullptr;
    if (l.header.index.read_string("tokenizer.ggml.pre", pre) == gguf::MetadataError::Ok) {
        pretokenizer = capability::find_pretokenizer(pre);
    }
    REQUIRE((pretokenizer != nullptr || !algorithm->requires_pretokenizer));
    gguf::MemoryByteSource source{std::as_bytes(std::span{l.header.bytes}), l.header.file_size};
    const LoadResult r = algorithm->load(source, l.header.index, pretokenizer, l.tokenizer);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    REQUIRE(l.tokenizer != nullptr);
    return l;
}

const std::vector<std::string> kTexts{
    "Hello, world! It's 2026 — and naïve café-goers say 你好.",
    "<|im_start|>user\nWhat is 12345 + 678?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<start_of_turn>user\nWrite a haiku.<end_of_turn>\n<start_of_turn>model\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "    indented\tcode();\n\n\n  ",
};

}  // namespace

TEST_CASE("each listed model's tokenizer, loaded through the table, encodes and decodes as its encoder does") {
    for (const std::string_view model : {"qwen3-0.6b-q4_0", "llama-3.2-1b-instruct-q4_0", "gemma-3-1b-it-q4_0"}) {
        CAPTURE(model);
        Loaded l = load_through_table(model);
        // The encoder the algorithm wraps, loaded directly.
        gguf::MemoryByteSource source{std::as_bytes(std::span{l.header.bytes}), l.header.file_size};
        std::string_view name, pre;
        REQUIRE(l.header.index.read_string("tokenizer.ggml.model", name) == gguf::MetadataError::Ok);
        bpe::ByteLevelBpe byte_level;
        bpe::SentencePieceBpe sentencepiece;
        const bool is_byte_level = name == "gpt2";
        if (is_byte_level) REQUIRE(l.header.index.read_string("tokenizer.ggml.pre", pre) == gguf::MetadataError::Ok);
        if (is_byte_level) {
            const LoadResult r =
                bpe::load_byte_level_bpe(source, l.header.index, *capability::find_pretokenizer(pre), byte_level);
            REQUIRE_MESSAGE(r.ok(), r.subject);
        } else {
            const LoadResult r = bpe::load_sentencepiece_bpe(source, l.header.index, sentencepiece);
            REQUIRE_MESSAGE(r.ok(), r.subject);
        }
        CHECK(l.tokenizer->vocabulary().size() ==
              (is_byte_level ? byte_level.vocabulary().size() : sentencepiece.vocabulary().size()));
        for (const std::string& text : kTexts) {
            CAPTURE(text);
            std::vector<TokenId> want;
            REQUIRE((is_byte_level ? byte_level.encode(text, kNoTokenLimit, want) : sentencepiece.encode(text, kNoTokenLimit, want))
                        .error == EncodeError::Ok);
            // Twice: whatever the first left in a cache, the second's tokens are
            // the same. A cache changes how long encoding takes, never what it
            // returns, so this cannot see whether one was used.
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<TokenId> got;
                REQUIRE(l.tokenizer->encode(text, kNoTokenLimit, got).error == EncodeError::Ok);
                CHECK(got == want);
            }
            std::string decoded, direct;
            for (const TokenId t : want) {
                l.tokenizer->decode(t, decoded);
                if (is_byte_level) byte_level.decode(t, direct);
                else sentencepiece.decode(t, direct);
            }
            CHECK(decoded == direct);
        }
        // Text that is not UTF-8 is refused, and `out` left untouched.
        std::vector<TokenId> out{TokenId{7}};
        CHECK(l.tokenizer->encode(std::string_view{"\xFF\xFE", 2}, kNoTokenLimit, out).error == EncodeError::InvalidUtf8);
        CHECK(out == std::vector<TokenId>{TokenId{7}});
    }
}

TEST_CASE("every algorithm the table lists carries its loader") {
    for (const std::string_view name : {"gpt2", "llama"}) {
        const Algorithm* algorithm = capability::find_tokenizer(name);
        REQUIRE(algorithm != nullptr);
        CHECK(algorithm->load != nullptr);
    }
}

TEST_CASE("each listed model's encode is bounded by its token limit and kMaxEncodeBytes before it merges") {
    struct Model {
        std::string_view name;
        std::size_t cover;
    };
    for (const Model& m : {Model{"qwen3-0.6b-q4_0", 128}, Model{"llama-3.2-1b-instruct-q4_0", 128},
                           Model{"gemma-3-1b-it-q4_0", 48}}) {
        CAPTURE(m.name);
        const Loaded l = load_through_table(m.name);
        CHECK(l.tokenizer->longest_cover() == m.cover);

        // A run of one letter, a byte each normalized and matching no
        // special token, whose least count meets a limit of 10: admitted, its
        // tokens those an encode without a limit gives. (Spaces will not do:
        // Gemma 3's user-defined tokens for runs of them are split out as
        // specials.)
        const std::string letters(10 * m.cover, 'a');
        std::vector<TokenId> bounded, free;
        const EncodeResult met = l.tokenizer->encode(letters, 10, bounded);
        REQUIRE(met.error == EncodeError::Ok);
        CHECK(met.least_tokens == 10);
        REQUIRE(l.tokenizer->encode(letters, kNoTokenLimit, free).error == EncodeError::Ok);
        CHECK(bounded == free);
        // One more letter: refused naming its count, out untouched.
        std::vector<TokenId> out{TokenId{7}};
        const EncodeResult over = l.tokenizer->encode(letters + "a", 10, out);
        CHECK(over.error == EncodeError::TooManyTokens);
        CHECK(over.least_tokens == 11);
        CHECK(out == std::vector<TokenId>{TokenId{7}});

        // A byte past kMaxEncodeBytes normalized: TooLong, naming its bytes.
        const std::string past(kMaxEncodeBytes + 1, 'a');
        const EncodeResult large = l.tokenizer->encode(past, kNoTokenLimit, out);
        CHECK(large.error == EncodeError::TooLong);
        CHECK(large.normalized_bytes == kMaxEncodeBytes + 1);
        CHECK(out == std::vector<TokenId>{TokenId{7}});
        // Raw text past 7/2 of it and a limit of 10's worth of special
        // tokens, every listed one under 1,000 bytes: refused before it is
        // segmented or normalized, naming its raw bytes.
        const std::string raw(kMaxEncodeBytes * 7 / 2 + 10 * 1000 + 1, 'a');
        const EncodeResult unread = l.tokenizer->encode(raw, 10, out);
        CHECK(unread.error == EncodeError::TooLong);
        CHECK(unread.normalized_bytes == 0);
        CHECK(unread.raw_bytes == raw.size());
        CHECK(out == std::vector<TokenId>{TokenId{7}});
    }
}

TEST_CASE("special tokens count as one token each, and their bytes are not the encode's to bound") {
    // 61,167 of Llama 3.2's longest special token, 1,835,010 bytes, past 7/2
    // of kMaxEncodeBytes: 61,167 tokens, within its 131,072.
    const Loaded llama = load_through_table("llama-3.2-1b-instruct-q4_0");
    std::string many;
    for (int i = 0; i < 61'167; ++i) many += "<|reserved_special_token_247|>";
    std::vector<TokenId> tokens;
    const EncodeResult r = llama.tokenizer->encode(many, 131'072, tokens);
    CHECK(r.error == EncodeError::Ok);
    CHECK(tokens.size() == 61'167);
    // Past the limit's worth of specials, segmenting stops: refused naming
    // one more than the limit, out untouched.
    const Loaded qwen = load_through_table("qwen3-0.6b-q4_0");
    std::string eleven;
    for (int i = 0; i < 11; ++i) eleven += "<|im_end|>";
    std::vector<TokenId> out{TokenId{7}};
    const EncodeResult over = qwen.tokenizer->encode(eleven, 10, out);
    CHECK(over.error == EncodeError::TooManyTokens);
    CHECK(over.least_tokens == 11);
    CHECK(out == std::vector<TokenId>{TokenId{7}});
}
