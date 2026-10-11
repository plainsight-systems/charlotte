#include "core/runtime/ends.h"

#include <string_view>

namespace bllm::runtime {
namespace {

EndsResult refused(EndsError error, std::string_view key, std::string_view why) {
    return {error, std::string(key) + " " + std::string(why)};
}

// One end: its flag, and where the flag is true, the token it adds.
EndsResult resolve(const gguf::TensorIndex& index, const tokenizer::Vocabulary& vocabulary, std::string_view flag_key,
                   std::string_view id_key, std::optional<tokenizer::TokenId>& out) {
    bool add = false;
    switch (index.read_bool(flag_key, add)) {
        case gguf::MetadataError::Ok: break;
        case gguf::MetadataError::MissingKey: return refused(EndsError::Undeclared, flag_key, "is not declared");
        case gguf::MetadataError::WrongType: return refused(EndsError::WrongType, flag_key, "is not a bool");
    }
    if (!add) {
        out.reset();
        return {};
    }
    std::uint32_t id = 0;
    switch (index.read_u32(id_key, id)) {
        case gguf::MetadataError::Ok: break;
        case gguf::MetadataError::MissingKey: return refused(EndsError::MissingToken, id_key, "is not declared");
        case gguf::MetadataError::WrongType:
            return refused(EndsError::WrongType, id_key, "is not an unsigned 32-bit integer");
    }
    if (id >= vocabulary.size()) return refused(EndsError::OutOfVocabulary, id_key, "is past the vocabulary");
    out = static_cast<tokenizer::TokenId>(id);
    return {};
}

}  // namespace

EndsResult resolve_ends(const gguf::TensorIndex& index, const tokenizer::Vocabulary& vocabulary, Ends& out) {
    Ends ends;
    if (auto r = resolve(index, vocabulary, "tokenizer.ggml.add_bos_token", "tokenizer.ggml.bos_token_id", ends.bos);
        r.error != EndsError::Ok) {
        return r;
    }
    if (auto r = resolve(index, vocabulary, "tokenizer.ggml.add_eos_token", "tokenizer.ggml.eos_token_id", ends.eos);
        r.error != EndsError::Ok) {
        return r;
    }
    out = ends;
    return {};
}

}  // namespace bllm::runtime
