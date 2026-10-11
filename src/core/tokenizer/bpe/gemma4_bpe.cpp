#include "core/tokenizer/bpe/gemma4_bpe.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

#include "core/tokenizer/admission.h"
#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/load.h"
#include "core/tokenizer/unicode.h"

namespace bllm::tokenizer::bpe {
namespace {

constexpr std::string_view kSpace = "\xE2\x96\x81";   // U+2581, how the vocabulary spells a space

// `text` with each space spelled "▁", into `out`. A space is one byte, never
// part of another character, so UTF-8 stays UTF-8.
void escape_into(std::string_view text, std::string& out) {
    out.clear();
    for (const char c : text) {
        if (c == ' ') {
            out.append(kSpace);
        } else {
            out.push_back(c);
        }
    }
}

// The bytes `text` takes once each space is "▁": two more a space.
std::uint64_t escaped_length(std::string_view text) {
    return text.size() + 2 * static_cast<std::uint64_t>(std::count(text.begin(), text.end(), ' '));
}

// Whether `text` is exactly one character.
bool one_character(std::string_view text) noexcept {
    Utf8Char c{};
    return !text.empty() && decode_utf8(text, 0, c) && c.length == text.size();
}

}  // namespace

const Algorithm kGemma4{"gemma4", false, load_gemma4_tokenizer};

EncodeResult Gemma4Bpe::encode(std::string_view raw, std::uint32_t max_tokens, std::vector<TokenId>& out) const {
    if (const EncodeResult r = admit_raw(raw.size(), max_tokens, special_.longest()); r.error != EncodeError::Ok) {
        return r;
    }
    // Special tokens are found in the raw text first, before any space is
    // respelled, as llama.cpp partitions them out of the text it escapes;
    // segmenting stops past the limit's worth of them.
    std::vector<Segment> segments;
    if (!special_.segment(raw, segments, max_tokens)) {
        return {EncodeError::TooManyTokens, std::uint64_t{max_tokens} + 1, 0, raw.size()};
    }

    // Every segment counted at its escaped length before any run is merged
    // (tokenizer.h, encoding is bounded). A token never spans a line break,
    // so the least count over a whole segment stays a lower bound.
    std::vector<std::uint64_t> lengths;
    std::uint64_t specials = 0;
    for (const Segment& segment : segments) {
        if (segment.special) {
            ++specials;
        } else {
            lengths.push_back(escaped_length(raw.substr(segment.offset, segment.length)));
        }
    }
    EncodeResult admitted = admit(specials, lengths, cover_, max_tokens);
    admitted.raw_bytes = raw.size();
    if (admitted.error != EncodeError::Ok) return admitted;

    std::vector<TokenId> tokens;
    std::vector<TokenId> symbols;
    std::string text;
    for (const Segment& segment : segments) {
        if (segment.special) {
            tokens.push_back(*segment.special);
            continue;
        }
        escape_into(raw.substr(segment.offset, segment.length), text);
        // Lines: each run of characters other than "\n", and each run of
        // "\n", llama.cpp's split for gemma4.
        for (std::size_t begin = 0; begin < text.size();) {
            const bool newlines = text[begin] == '\n';
            std::size_t end = begin;
            while (end < text.size() && (text[end] == '\n') == newlines) ++end;
            const std::string_view piece = std::string_view(text).substr(begin, end - begin);
            begin = end;
            if (newlines) {
                // A run of newlines that is itself a token is that token.
                if (const auto whole = vocabulary_.find(piece); whole && vocabulary_.type(*whole) == TokenType::Normal) {
                    tokens.push_back(*whole);
                    continue;
                }
            }
            symbols.clear();
            for (std::size_t at = 0; at < piece.size();) {
                Utf8Char c{};
                if (!decode_utf8(piece, at, c)) return {EncodeError::InvalidUtf8, 0, 0, raw.size()};
                const std::string_view character = piece.substr(at, c.length);
                const auto token = vocabulary_.find(character);
                if (token && vocabulary_.type(*token) == TokenType::Normal) {
                    symbols.push_back(*token);
                } else {
                    for (const char byte : character) symbols.push_back(byte_tokens_[static_cast<unsigned char>(byte)]);
                }
                at += c.length;
            }
            merge(symbols, merges_, tokens);
        }
    }
    out.insert(out.end(), tokens.begin(), tokens.end());
    return admitted;
}

void Gemma4Bpe::decode(TokenId token, std::string& out) const {
    const std::string_view text = vocabulary_.text(token);
    switch (vocabulary_.type(token)) {
        case TokenType::Normal:
            for (std::size_t at = 0; at < text.size();) {
                if (text.substr(at, kSpace.size()) == kSpace) {
                    out.push_back(' ');
                    at += kSpace.size();
                } else {
                    out.push_back(text[at++]);
                }
            }
            return;
        case TokenType::Byte: {
            // "<0xXX>": load_gemma4_bpe found each of the 256, and no other.
            unsigned value = 0;
            (void)std::from_chars(text.data() + 3, text.data() + 5, value, 16);
            out.push_back(static_cast<char>(value));
            return;
        }
        default:
            out.append(text);
            return;
    }
}

LoadResult load_gemma4_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index, Gemma4Bpe& out) {
    constexpr std::string_view kSpacePrefix = "tokenizer.ggml.add_space_prefix";
    bool space_prefix = true;
    switch (index.read_bool(kSpacePrefix, space_prefix)) {
        case gguf::MetadataError::Ok: break;
        case gguf::MetadataError::MissingKey: return {LoadError::Unsupported, std::string(kSpacePrefix) + " is not declared"};
        case gguf::MetadataError::WrongType: return {LoadError::WrongKeyType, std::string(kSpacePrefix)};
    }
    if (space_prefix) return {LoadError::Unsupported, std::string(kSpacePrefix) + " is true"};

    Gemma4Bpe g;
    if (auto r = load_vocabulary(source, index, g.vocabulary_); !r.ok()) return r;
    if (auto r = load_merges(source, index, g.vocabulary_, g.merges_); !r.ok()) return r;

    // A character becomes its normal token, and a special token's text is
    // matched before that. One with a token of another type (unused, unknown)
    // would be that token to both references and its bytes here: refused.
    for (std::size_t id = 0; id < g.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        const TokenType type = g.vocabulary_.type(token);
        if (type == TokenType::Normal || type == TokenType::Control || type == TokenType::UserDefined ||
            type == TokenType::Byte || !one_character(g.vocabulary_.text(token))) {
            continue;
        }
        return {LoadError::Unsupported, "token " + std::to_string(id) + ", one character that is not a normal token"};
    }

    for (std::size_t byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02zX>", byte);
        const auto token = g.vocabulary_.find(name);
        if (!token || g.vocabulary_.type(*token) != TokenType::Byte) {
            return {LoadError::MissingByte, "byte " + std::string(name)};
        }
        g.byte_tokens_[byte] = *token;
    }
    std::size_t byte_tokens = 0;
    for (std::size_t id = 0; id < g.vocabulary_.size(); ++id) {
        byte_tokens += g.vocabulary_.type(static_cast<TokenId>(id)) == TokenType::Byte;
    }
    if (byte_tokens != 256) return {LoadError::Unsupported, "byte tokens other than <0x00> to <0xFF>"};

    // Special tokens are matched in the raw text, so they are spelled as the
    // file spells them.
    std::vector<SpecialTokens::Entry> special;
    for (std::size_t id = 0; id < g.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        const TokenType type = g.vocabulary_.type(token);
        const std::string_view text = g.vocabulary_.text(token);
        if ((type == TokenType::Control || type == TokenType::UserDefined) && !text.empty()) {
            special.push_back({std::string(text), token});
        }
    }
    std::vector<std::string_view> texts;
    for (const auto& entry : special) texts.push_back(entry.text);
    std::sort(texts.begin(), texts.end());
    if (const auto repeat = std::adjacent_find(texts.begin(), texts.end()); repeat != texts.end()) {
        return {LoadError::Unsupported, "two special tokens spelled " + std::string(*repeat)};
    }
    g.special_ = SpecialTokens{std::move(special)};
    // The most normalized bytes one token covers: its spelling, over the
    // whole vocabulary (tokenizer.h).
    for (std::size_t id = 0; id < g.vocabulary_.size(); ++id) {
        g.cover_ = std::max(g.cover_, g.vocabulary_.text(static_cast<TokenId>(id)).size());
    }
    out = std::move(g);
    return {};
}

const Vocabulary& Gemma4Tokenizer::vocabulary() const noexcept { return bpe_.vocabulary(); }

EncodeResult Gemma4Tokenizer::encode(std::string_view text, std::uint32_t max_tokens, std::vector<TokenId>& out) {
    return bpe_.encode(text, max_tokens, out);
}

std::size_t Gemma4Tokenizer::longest_cover() const noexcept { return bpe_.longest_cover(); }

void Gemma4Tokenizer::decode(TokenId token, std::string& out) const { bpe_.decode(token, out); }

LoadResult load_gemma4_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                 const PreTokenizer* /*pretokenizer*/, std::unique_ptr<Tokenizer>& out) {
    auto loaded = std::make_unique<Gemma4Tokenizer>();
    const LoadResult r = load_gemma4_bpe(source, index, loaded->bpe_);
    if (r.ok()) out = std::move(loaded);
    return r;
}

}  // namespace bllm::tokenizer::bpe
