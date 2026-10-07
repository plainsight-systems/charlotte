#include "core/tokenizer/detokenizer.h"

#include <algorithm>

#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer {

Detokenizer::Detokenizer(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {
    const Vocabulary& vocabulary = tokenizer.vocabulary();
    for (std::size_t id = 0; id < vocabulary.size(); ++id) {
        longest_ = std::max(longest_, vocabulary.text(static_cast<TokenId>(id)).size());
    }
    bytes_.reserve(longest_);
    // At the worst every byte — the up to three held before a token's, then
    // its own — is one the stream cannot place, each written as U+FFFD's
    // three bytes.
    text_.reserve(3 * (longest_ + 3));
}

std::string_view Detokenizer::push(TokenId token) {
    bytes_.clear();
    tokenizer_->decode(token, bytes_);
    text_.clear();
    stream_.push(bytes_, text_);
    return text_;
}

std::string_view Detokenizer::finish() {
    text_.clear();
    (void)stream_.finish(text_);
    return text_;
}

}  // namespace bllm::tokenizer
