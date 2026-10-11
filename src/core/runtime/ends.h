#pragma once

#include <optional>
#include <string>

#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::runtime {

// Axis M (encode): changes with a new stage in the encode path.
//
// The tokens an embedder puts around a text (runtime/embedder.h): BOS
// before and EOS after, each where the file says. Made once at load, as the
// stop set is for a generator (stops.h), and resolved by preflight's Run
// stage over the header, so a file whose ends cannot be made is refused
// before its weights are fetched.
//
//   - From the file: tokenizer.ggml.add_bos_token and add_eos_token, each a
//     bool, and where one is true, tokenizer.ggml.bos_token_id or
//     eos_token_id, the token it adds.
//   - Refused by name, never guessed: a flag undeclared — the references
//     disagree on its default, llama.cpp forcing BOS for its gemma4
//     tokenizer and Hugging Face reading the tokenizer's own config — or
//     declared with another type; a flag true whose identifier is missing,
//     not an unsigned 32-bit integer, or at or past the vocabulary.
//   - Both flags false is valid: a text then goes in bare.
//   - A generator reads neither flag: its chat template writes BOS as text
//     (tokenizer.h), and a file's add_bos_token says nothing about a
//     rendered conversation.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     E.27   Use error codes systematically — EndsError, the key named.
//     SL.io.2 When reading, always consider ill-formed input — every key
//            checked for presence, type and range.
//   C++ performance guidelines
//     MEM.9  Allocate at init — resolved at load; an encode reads two
//            optionals.

struct Ends {
    std::optional<tokenizer::TokenId> bos;
    std::optional<tokenizer::TokenId> eos;
};

enum class EndsError {
    Ok,
    Undeclared,        // a flag the file does not declare
    WrongType,         // a flag or an identifier of another type
    MissingToken,      // a flag true, its identifier absent
    OutOfVocabulary,   // an identifier at or past the vocabulary
};

struct EndsResult {
    EndsError error = EndsError::Ok;
    std::string subject;   // the key refused, and why
};

// The ends of `index`'s file, into `out`; `out` is left untouched on failure.
// Precondition: `vocabulary` was loaded from `index`'s file.
[[nodiscard]] EndsResult resolve_ends(const gguf::TensorIndex& index, const tokenizer::Vocabulary& vocabulary,
                                      Ends& out);

}  // namespace bllm::runtime
