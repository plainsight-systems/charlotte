#include "core/tokenizer/bpe/sentencepiece_bpe.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/bpe/sentencepiece_merges.h"
#include "core/tokenizer/load.h"
#include "core/tokenizer/unicode.h"

namespace bllm::tokenizer::bpe {
namespace {

constexpr std::string_view kSpace = "\xE2\x96\x81";   // U+2581, how the vocabulary spells a space

// `text` with each space spelled "▁". A space is one byte, never part of
// another character, so UTF-8 stays UTF-8.
std::string escaped(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c == ' ') {
            out.append(kSpace);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

// Whether `text` is exactly one character.
bool one_character(std::string_view text) noexcept {
    Utf8Char c{};
    return !text.empty() && decode_utf8(text, 0, c) && c.length == text.size();
}

}  // namespace

const Algorithm kSentencePiece{"llama", false, load_sentencepiece_tokenizer};

EncodeError SentencePieceBpe::encode(std::string_view raw, std::vector<TokenId>& out) const {
    if (raw.size() > std::numeric_limits<std::uint32_t>::max()) return EncodeError::TooLong;
    const std::string spelled = escaped(raw);
    if (spelled.size() > std::numeric_limits<std::uint32_t>::max()) return EncodeError::TooLong;
    const std::string_view text = spelled;

    std::vector<Segment> segments;
    special_.segment(text, segments);

    std::vector<TokenId> tokens;
    std::vector<TokenId> symbols;
    for (const Segment& segment : segments) {
        if (segment.special) {
            tokens.push_back(*segment.special);
            continue;
        }
        const std::string_view ordinary = text.substr(segment.offset, segment.length);
        symbols.clear();
        for (std::size_t at = 0; at < ordinary.size();) {
            Utf8Char c{};
            if (!decode_utf8(ordinary, at, c)) return EncodeError::InvalidUtf8;
            if (!symbols.empty() && cut_before(ordinary, at)) {
                merge(symbols, merges_, tokens);
                symbols.clear();
            }
            const std::string_view character = ordinary.substr(at, c.length);
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
    out.insert(out.end(), tokens.begin(), tokens.end());
    return EncodeError::Ok;
}

void SentencePieceBpe::decode(TokenId token, std::string& out) const {
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
            // "<0xXX>": load_sentencepiece_bpe found each of the 256, and no other.
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

bool SentencePieceBpe::cut_before(std::string_view text, std::size_t at) const noexcept {
    if (!cuts_ || text.substr(at, kSpace.size()) != kSpace) return false;
    for (const Straddle& s : straddles_) {
        for (const std::uint32_t mark : s.marks) {
            if (at >= mark && text.substr(at - mark, s.text.size()) == s.text) return false;
        }
    }
    return true;
}

LoadResult load_sentencepiece_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                  SentencePieceBpe& out) {
    constexpr std::string_view kSpacePrefix = "tokenizer.ggml.add_space_prefix";
    bool space_prefix = true;
    switch (index.read_bool(kSpacePrefix, space_prefix)) {
        case gguf::MetadataError::Ok: break;
        case gguf::MetadataError::MissingKey: return {LoadError::Unsupported, std::string(kSpacePrefix) + " is not declared"};
        case gguf::MetadataError::WrongType: return {LoadError::WrongKeyType, std::string(kSpacePrefix)};
    }
    if (space_prefix) return {LoadError::Unsupported, std::string(kSpacePrefix) + " is true"};

    SentencePieceBpe spm;
    if (auto r = load_vocabulary(source, index, spm.vocabulary_); !r.ok()) return r;
    std::vector<float> scores;
    if (auto r = read_float32_array(source, index, "tokenizer.ggml.scores", scores); !r.ok()) return r;
    if (scores.size() != spm.vocabulary_.size()) return {LoadError::CountMismatch, "tokenizer.ggml.scores"};
    if (auto r = implied_merges(spm.vocabulary_, scores, spm.merges_); !r.ok()) return r;

    // A character becomes its normal token, and a special token's text is
    // matched before that. One with a token of another type (unused, unknown)
    // would be that token to both references and its bytes here: refused.
    for (std::size_t id = 0; id < spm.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        const TokenType type = spm.vocabulary_.type(token);
        if (type == TokenType::Normal || type == TokenType::Control || type == TokenType::UserDefined ||
            type == TokenType::Byte || !one_character(spm.vocabulary_.text(token))) {
            continue;
        }
        return {LoadError::Unsupported, "token " + std::to_string(id) + ", one character that is not a normal token"};
    }

    for (std::size_t byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02zX>", byte);
        const auto token = spm.vocabulary_.find(name);
        if (!token || spm.vocabulary_.type(*token) != TokenType::Byte) {
            return {LoadError::MissingByte, "byte " + std::string(name)};
        }
        spm.byte_tokens_[byte] = *token;
    }
    std::size_t byte_tokens = 0;
    for (std::size_t id = 0; id < spm.vocabulary_.size(); ++id) {
        byte_tokens += spm.vocabulary_.type(static_cast<TokenId>(id)) == TokenType::Byte;
    }
    if (byte_tokens != 256) return {LoadError::Unsupported, "byte tokens other than <0x00> to <0xFF>"};
    // Special tokens are matched in the text after spaces become ▁, so they are
    // spelled that way too.
    std::vector<SpecialTokens::Entry> special;
    for (std::size_t id = 0; id < spm.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        const TokenType type = spm.vocabulary_.type(token);
        const std::string_view text = spm.vocabulary_.text(token);
        if ((type == TokenType::Control || type == TokenType::UserDefined) && !text.empty()) {
            special.push_back({escaped(text), token});
        }
    }
    std::vector<std::string_view> texts;
    for (const auto& entry : special) texts.push_back(entry.text);
    std::sort(texts.begin(), texts.end());
    if (const auto repeat = std::adjacent_find(texts.begin(), texts.end()); repeat != texts.end()) {
        return {LoadError::Unsupported, "two special tokens spelled " + std::string(*repeat) + " once spaces are ▁"};
    }
    spm.special_ = SpecialTokens{std::move(special)};

    for (std::size_t id = 0; id < spm.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        if (spm.vocabulary_.type(token) != TokenType::Normal) continue;
        const std::string_view text = spm.vocabulary_.text(token);
        SentencePieceBpe::Straddle straddle{std::string(text), {}};
        for (std::size_t at = text.find(kSpace, 1); at != std::string_view::npos; at = text.find(kSpace, at + 1)) {
            straddle.marks.push_back(static_cast<std::uint32_t>(at));
        }
        if (!straddle.marks.empty()) spm.straddles_.push_back(std::move(straddle));
    }
    spm.cuts_ = spm.straddles_.size() <= SentencePieceBpe::kMaxStraddles;
    out = std::move(spm);
    return {};
}

const Vocabulary& SentencePieceTokenizer::vocabulary() const noexcept { return bpe_.vocabulary(); }

EncodeError SentencePieceTokenizer::encode(std::string_view text, std::vector<TokenId>& out) {
    return bpe_.encode(text, out);
}

void SentencePieceTokenizer::decode(TokenId token, std::string& out) const { bpe_.decode(token, out); }

LoadResult load_sentencepiece_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                        const PreTokenizer* /*pretokenizer*/, std::unique_ptr<Tokenizer>& out) {
    auto loaded = std::make_unique<SentencePieceTokenizer>();
    const LoadResult r = load_sentencepiece_bpe(source, index, loaded->bpe_);
    if (r.ok()) out = std::move(loaded);
    return r;
}

}  // namespace bllm::tokenizer::bpe
