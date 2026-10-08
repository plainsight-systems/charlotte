#include "core/tokenizer/bpe/byte_level_bpe.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

#include "core/tokenizer/admission.h"
#include "core/tokenizer/bpe/byte_map.h"
#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/nfc.h"
#include "core/tokenizer/unicode.h"

namespace bllm::tokenizer::bpe {
namespace {

// Whether every character of `text` stands for a byte.
bool spells_bytes(std::string_view text) noexcept {
    for (std::size_t at = 0; at < text.size();) {
        Utf8Char c{};
        if (!decode_utf8(text, at, c) || !byte_of(c.code_point)) return false;
        at += c.length;
    }
    return true;
}

}  // namespace

const Algorithm kByteLevel{"gpt2", true, load_byte_level_tokenizer};

EncodeResult ByteLevelBpe::encode(std::string_view text, std::uint32_t max_tokens, std::vector<TokenId>& out) const {
    return encode_into(text, max_tokens, out, nullptr);
}

EncodeResult ByteLevelBpe::encode(std::string_view text, std::uint32_t max_tokens, std::vector<TokenId>& out,
                                  PieceCache& cache) const {
    if (cache.owner_ != this) {
        cache.clear();
        cache.owner_ = this;
    }
    return encode_into(text, max_tokens, out, &cache);
}

EncodeResult ByteLevelBpe::encode_into(std::string_view text, std::uint32_t max_tokens, std::vector<TokenId>& out,
                                       PieceCache* cache) const {
    if (const EncodeResult raw = admit_raw(text.size(), max_tokens, special_.longest()); raw.error != EncodeError::Ok) {
        return raw;
    }
    // The special tokens first, the raw text's, then NFC on each ordinary
    // segment; segmenting stops past the limit's worth of specials.
    std::vector<Segment> segments;
    if (!special_.segment(text, segments, max_tokens)) {
        return {EncodeError::TooManyTokens, std::uint64_t{max_tokens} + 1, 0, text.size()};
    }

    // Every ordinary segment normalized first, one after another in
    // `normalized`, so admission counts the whole text before any piece is
    // merged (tokenizer.h, encoding is bounded).
    std::string normalized;
    std::string scratch;
    std::vector<std::uint64_t> lengths;
    std::uint64_t specials = 0;
    for (const Segment& segment : segments) {
        if (segment.special) {
            ++specials;
            continue;
        }
        std::string_view ordinary = text.substr(segment.offset, segment.length);
        if (pretokenizer_->normalization == Normalization::Nfc) {
            if (!to_nfc(ordinary, scratch)) return {EncodeError::InvalidUtf8, 0, 0, text.size()};
            ordinary = scratch;
        }
        normalized.append(ordinary);
        lengths.push_back(ordinary.size());
    }
    EncodeResult admitted = admit(specials, lengths, cover_, max_tokens);
    admitted.raw_bytes = text.size();
    if (admitted.error != EncodeError::Ok) return admitted;

    std::vector<TokenId> tokens;
    std::vector<Piece> pieces;
    std::string spelled;              // scratch for each piece, kept across pieces
    std::vector<TokenId> symbols;
    std::size_t at = 0;
    std::size_t next = 0;             // the next ordinary segment's length
    for (const Segment& segment : segments) {
        if (segment.special) {
            tokens.push_back(*segment.special);
            continue;
        }
        const std::string_view ordinary = std::string_view{normalized}.substr(at, lengths[next]);
        at += lengths[next++];
        switch (split(*pretokenizer_, ordinary, pieces)) {
            case SplitError::Ok: break;
            case SplitError::InvalidUtf8: return {EncodeError::InvalidUtf8, 0, 0, text.size()};
            case SplitError::TooLong: return {EncodeError::TooLong, admitted.least_tokens, admitted.normalized_bytes,
                                              text.size()};
        }
        for (const Piece& piece : pieces) {
            encode_piece(ordinary.substr(piece.offset, piece.length), tokens, cache, spelled, symbols);
        }
    }
    out.insert(out.end(), tokens.begin(), tokens.end());
    return admitted;
}

void ByteLevelBpe::encode_piece(std::string_view piece, std::vector<TokenId>& out, PieceCache* cache,
                                std::string& spelled, std::vector<TokenId>& symbols) const {
    if (cache != nullptr) {
        if (const auto hit = cache->find(piece)) {
            out.insert(out.end(), hit->begin(), hit->end());
            return;
        }
    }
    const std::size_t first = out.size();
    if (pretokenizer_->ignore_merges) {
        spelled.clear();
        for (const char byte : piece) append_utf8(kByteChars[static_cast<unsigned char>(byte)], spelled);
        if (const auto whole = vocabulary_.find(spelled)) out.push_back(*whole);
    }
    if (out.size() == first) {
        symbols.clear();
        for (const char byte : piece) symbols.push_back(byte_tokens_[static_cast<unsigned char>(byte)]);
        merge(symbols, merges_, out);
    }
    if (cache != nullptr) cache->insert(piece, std::span<const TokenId>{out}.subspan(first));
}

void ByteLevelBpe::decode(TokenId token, std::string& out) const {
    const std::string_view text = vocabulary_.text(token);
    if (vocabulary_.type(token) != TokenType::Normal) {
        out.append(text);
        return;
    }
    // Every character stands for a byte: load_byte_level_bpe checked it.
    for (std::size_t at = 0; at < text.size();) {
        Utf8Char c{};
        (void)decode_utf8(text, at, c);
        out.push_back(static_cast<char>(*byte_of(c.code_point)));
        at += c.length;
    }
}

LoadResult load_byte_level_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                               const PreTokenizer& pretokenizer, ByteLevelBpe& out) {
    ByteLevelBpe bpe;
    if (auto r = load_vocabulary(source, index, bpe.vocabulary_); !r.ok()) return r;
    if (auto r = load_merges(source, index, bpe.vocabulary_, bpe.merges_); !r.ok()) return r;
    for (std::size_t id = 0; id < bpe.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        if (bpe.vocabulary_.type(token) == TokenType::Normal && !spells_bytes(bpe.vocabulary_.text(token))) {
            return {LoadError::UnmappedCharacter, std::string(bpe.vocabulary_.text(token))};
        }
    }

    std::string spelled;
    for (std::size_t byte = 0; byte < 256; ++byte) {
        spelled.clear();
        append_utf8(kByteChars[byte], spelled);
        const auto token = bpe.vocabulary_.find(spelled);
        if (!token) {
            char name[8];
            std::snprintf(name, sizeof name, "0x%02zX", byte);
            return {LoadError::MissingByte, "byte " + std::string(name)};
        }
        bpe.byte_tokens_[byte] = *token;
    }
    bpe.special_ = SpecialTokens{bpe.vocabulary_};
    bpe.pretokenizer_ = &pretokenizer;
    // The most normalized bytes one token covers: its decoded bytes, over the
    // whole vocabulary (tokenizer.h).
    std::string decoded;
    for (std::size_t id = 0; id < bpe.vocabulary_.size(); ++id) {
        decoded.clear();
        bpe.decode(static_cast<TokenId>(id), decoded);
        bpe.cover_ = std::max(bpe.cover_, decoded.size());
    }
    out = std::move(bpe);
    return {};
}

const Vocabulary& ByteLevelTokenizer::vocabulary() const noexcept { return bpe_.vocabulary(); }

EncodeResult ByteLevelTokenizer::encode(std::string_view text, std::uint32_t max_tokens, std::vector<TokenId>& out) {
    return bpe_.encode(text, max_tokens, out, cache_);
}

std::size_t ByteLevelTokenizer::longest_cover() const noexcept { return bpe_.longest_cover(); }

void ByteLevelTokenizer::decode(TokenId token, std::string& out) const { bpe_.decode(token, out); }

LoadResult load_byte_level_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                     const PreTokenizer* pretokenizer, std::unique_ptr<Tokenizer>& out) {
    auto loaded = std::make_unique<ByteLevelTokenizer>();
    const LoadResult r = load_byte_level_bpe(source, index, *pretokenizer, loaded->bpe_);
    if (r.ok()) out = std::move(loaded);
    return r;
}

}  // namespace bllm::tokenizer::bpe
