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
    text_.reserve(longest_ + 3);   // at most three bytes held before a token's
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
