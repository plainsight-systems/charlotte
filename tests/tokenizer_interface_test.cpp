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
            REQUIRE((is_byte_level ? byte_level.encode(text, want) : sentencepiece.encode(text, want)) ==
                    EncodeError::Ok);
            // Twice: the second encode finds what the first left in any cache.
            for (int pass = 0; pass < 2; ++pass) {
                std::vector<TokenId> got;
                REQUIRE(l.tokenizer->encode(text, got) == EncodeError::Ok);
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
        CHECK(l.tokenizer->encode(std::string_view{"\xFF\xFE", 2}, out) == EncodeError::InvalidUtf8);
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
