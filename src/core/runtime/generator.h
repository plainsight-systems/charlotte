#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "core/policy/policy.h"
#include "core/runtime/runtime.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// A loaded model as the page uses it: the rendered conversation in, the
// reply's text out. The runtime (runtime.h) works on identifiers alone; this
// puts the model's tokenizer (tokenizer.h) on either side of it, so the
// boundary (src/wasm/bindings.cpp) only translates.
//
//   - A turn: the text, UTF-8, is encoded whole — the chat template wrote
//     the special tokens as text, and they encode as their identifiers —
//     into tokens held for the generator's life, reserved at the context
//     offered plus one; a prompt that encodes longer grows them once, and
//     the runtime refuses it, naming both counts. Then the runtime's turn
//     starts over them.
//   - Refused at once, without a callback: text the tokenizer cannot encode,
//     named; and whatever the runtime refuses (runtime.h).
//   - Each token the runtime emits is decoded into bytes and pushed through
//     a UTF-8 stream (tokenizer.h's Utf8Stream), and the characters it
//     completes are passed to the text callback; a token that completes
//     none — the first bytes of a character another token finishes — passes
//     nothing. Stop tokens are never emitted, so never decoded; any other
//     control token's text, as written, is, since the page reads the reply's
//     structure from it, as Qwen3's <think> (web/conversation.js).
//   - When the turn finishes, the stream is finished: bytes still pending,
//     the turn having stopped inside a character, are passed as U+FFFD
//     rather than dropped unseen. Then the end callback, with the runtime's
//     result.
//   - It owns the tokenizer and the runtime, and its callbacks carry its
//     state, held while a turn runs, as the runtime's are: destroying a
//     generator with a turn running destroys the runtime, which finishes the
//     turn Cancelled (runtime.h), and its end callback is still called.
//
// What it costs, counted:
//   - A turn, once: the text's bytes encoded, about 26 ns a byte for
//     byte-level BPE with its piece cache warm, 39 for SentencePiece (their
//     headers) — 4 to 5 ms for a conversation of a listed model's whole
//     context — then the runtime's turn.
//   - A token: one decode, an indirect call that appends the token's bytes
//     to a string reserved at load for the vocabulary's longest token; the
//     stream's push over those bytes; and, when they complete a character,
//     one text callback — the boundary's one crossing a token (WASM.2).
//     Nothing is allocated.
//
// Verification the implementation is held to, on the GPU with Qwen3's file:
//   - A turn over a rendered chat prompt streams text whose concatenation is
//     the tokenizer's decoding of the tokens the runtime emitted, through one
//     stream; a token ending mid-character passes nothing until the next
//     completes it; the end callback follows the last text.
//   - A turn stopped inside a character ends with U+FFFD.
//   - Text that is not UTF-8 is refused, named, and no callback is called.
//   - Destroying a generator mid-turn calls the end callback, Cancelled.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — the tokenizer and the
//            runtime owned, by unique_ptr.
//     E.27   Use error codes systematically — GenerateResult.
//   C++ performance guidelines
//     WASM.2 Batch work across the JS boundary — the conversation crosses
//            once a turn, as bytes; text, once a token that completes some.
//     WASM.4 Reduce indirect dispatch in hot paths — one decode a token.
//     MEM.9  Allocate at init — the tokens, the decode buffer and the stream,
//            at load.

// Each piece of the reply's text, whole characters, in order. Valid only
// during the call.
using TextCallback = void (*)(std::string_view text, void* userdata);
// Once a turn, after its last text.
using EndCallback = void (*)(const TurnResult& result, void* userdata);

// Why a turn did not start: the text, or the runtime.
struct GenerateResult {
    tokenizer::EncodeError encode = tokenizer::EncodeError::Ok;
    StartResult start;
    [[nodiscard]] bool ok() const noexcept {
        return encode == tokenizer::EncodeError::Ok && start.error == StartError::Ok;
    }
};

class Generator {
public:
    // Preconditions: `tokenizer` is the model's, and `runtime` was made over
    // its program and cache (runtime.h).
    Generator(std::unique_ptr<tokenizer::Tokenizer> tokenizer, std::unique_ptr<Runtime> runtime);

    // Starts a turn over `text`, the whole rendered conversation. On ok(),
    // `on_text` is called for each piece of the reply and `on_end` once,
    // both with `userdata`; otherwise neither is. `text` is not read after
    // the call.
    [[nodiscard]] GenerateResult start(std::string_view text, const policy::TurnPolicy& policy, TextCallback on_text,
                                       EndCallback on_end, void* userdata);

    // Ends the running turn at its next report (runtime.h); with none,
    // nothing.
    void cancel() noexcept;

    ~Generator();
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // What the generator and its in-flight callbacks share; defined only
    // where the generator is.
    struct State;

private:
    std::shared_ptr<State> state_;   // shared with in-flight callbacks
};

}  // namespace bllm::runtime
