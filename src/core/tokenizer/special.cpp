#include "core/tokenizer/special.h"

#include <algorithm>
#include <utility>

namespace bllm::tokenizer {

namespace {

std::vector<SpecialTokens::Entry> special_entries(const Vocabulary& vocabulary) {
    std::vector<SpecialTokens::Entry> entries;
    for (std::size_t i = 0; i < vocabulary.size(); ++i) {
        const auto id = static_cast<TokenId>(i);
        const TokenType type = vocabulary.type(id);
        if ((type == TokenType::Control || type == TokenType::UserDefined) && !vocabulary.text(id).empty()) {
            entries.push_back({std::string(vocabulary.text(id)), id});
        }
    }
    return entries;
}

}  // namespace

SpecialTokens::SpecialTokens(const Vocabulary& vocabulary) : SpecialTokens(special_entries(vocabulary)) {}

SpecialTokens::SpecialTokens(std::vector<Entry> entries) : size_(entries.size()) {
    first_.fill(kNone);
    // Built in order of text, so each node's children are made in order of
    // byte, and a byte already followed from a node is its last child.
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.text < b.text; });
    std::vector<std::vector<std::pair<unsigned char, std::uint32_t>>> children(1);   // node 0: the root
    std::vector<std::uint32_t> token(1, kNone);
    for (const Entry& entry : entries) {
        longest_ = std::max(longest_, entry.text.size());
        std::uint32_t node = 0;
        for (const char c : entry.text) {
            const auto byte = static_cast<unsigned char>(c);
            auto& edges = children[node];
            if (edges.empty() || edges.back().first != byte) {
                edges.push_back({byte, static_cast<std::uint32_t>(children.size())});
                children.emplace_back();
                token.push_back(kNone);
            }
            node = children[node].back().second;
        }
        token[node] = static_cast<std::uint32_t>(entry.id);
    }
    for (const auto& [byte, child] : children[0]) first_[byte] = child;
    edge_start_.reserve(children.size() + 1);
    for (const auto& edges : children) {
        edge_start_.push_back(static_cast<std::uint32_t>(edge_byte_.size()));
        for (const auto& [byte, child] : edges) {
            edge_byte_.push_back(byte);
            edge_child_.push_back(child);
        }
    }
    edge_start_.push_back(static_cast<std::uint32_t>(edge_byte_.size()));
    token_ = std::move(token);
}

bool SpecialTokens::segment(std::string_view text, std::vector<Segment>& out, std::uint64_t max_specials) const {
    std::size_t plain = 0;   // where the current run of ordinary text began
    std::uint64_t found = 0;
    for (std::size_t i = 0; i < text.size();) {
        // Walk the trie from i; the last token passed is the longest match.
        std::uint32_t node = first_[static_cast<unsigned char>(text[i])];
        std::uint32_t match = kNone;
        std::size_t length = 0;
        for (std::size_t at = i + 1; node != kNone; ++at) {
            if (token_[node] != kNone) {
                match = token_[node];
                length = at - i;
            }
            if (at == text.size()) break;
            const auto byte = static_cast<unsigned char>(text[at]);
            std::uint32_t next = kNone;
            for (std::uint32_t e = edge_start_[node]; e < edge_start_[node + 1]; ++e) {
                if (edge_byte_[e] == byte) {
                    next = edge_child_[e];
                    break;
                }
            }
            node = next;
        }
        if (match == kNone) {
            ++i;
            continue;
        }
        if (i > plain) {
            out.push_back({static_cast<std::uint32_t>(plain), static_cast<std::uint32_t>(i - plain), std::nullopt});
        }
        if (++found > max_specials) return false;
        out.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(length), static_cast<TokenId>(match)});
        i += length;
        plain = i;
    }
    if (text.size() > plain) {
        out.push_back({static_cast<std::uint32_t>(plain), static_cast<std::uint32_t>(text.size() - plain),
                       std::nullopt});
    }
    return true;
}

}  // namespace bllm::tokenizer
