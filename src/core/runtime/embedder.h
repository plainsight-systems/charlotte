#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "core/kernels/program.h"
#include "core/runtime/ends.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::runtime {

// Axis M (encode): changes with a new stage in the encode path.
//
// The encode path: text in, one embedding out — a unit vector the page
// compares by dot product. Where a generator (generator.h) runs turns over
// a conversation, an embedder runs one forward pass over a text, the whole
// text at once, every token seeing every other (graph/encoder.h), and
// reads back the pooled vector. No cache, no sampler, no stop set, no
// decode: the pass is one step, and nothing of it is kept for the next.
//
//   - Text, UTF-8, is the page's whole input: the task's prompt written
//     around it — "task: search result | query: …", "title: none | text: …"
//     — by the page, as the page renders a chat template for a generator.
//     Then:
//       1. Encoded by the model's tokenizer (tokenizer.h) with the context
//          offered, less the ends below, as its limit. Admission refuses
//          text whose least count is over it before any piece is merged,
//          TooManyTokens with that count; but the least count is a lower
//          bound (tokenizer.h), so admitted text can still make more. The
//          tokens are counted again after the encode, with the ends, and
//          more than the context is refused then, TooManyTokens with the
//          count it made, before the step is written: the step holds
//          kPrefillBlock identifiers and no more (SL.con.3). Either way the
//          page can cut the text into chunks that fit.
//       2. Wrapped in the file's ends (ends.h): BOS before and EOS after,
//          each where tokenizer.ggml.add_bos_token and add_eos_token say
//          so — both, for EmbeddingGemma 2, as llama.cpp's and
//          sentence-transformers' tokenization of it add them. A generator
//          adds neither: its chat template writes them.
//       3. One step (kernels/interface.h): position 0, its tokens, logits 1,
//          fed 0. A text admitted is at most kPrefillBlock tokens with its
//          ends, so it is one step, which is what lets every token attend
//          to every other within it (graph/encoder.h).
//       4. The program reads back the step's embedding: embedding_length_out
//          floats, f32, unit length, from the plan's `embedding` working
//          buffer — 3 KiB for EmbeddingGemma 2's 768.
//       5. The callback, once, with the vector and how many tokens the
//          text made; valid only during the call.
//   - Two ways an encode can end, kept apart. start() returns at once with
//     a refusal for what it can know before the GPU runs — Busy,
//     DeviceLost, a text the tokenizer refuses, TooManyTokens, an empty
//     text — and then never calls back. An encode it accepts calls back
//     exactly once, with Ok and the vector, or with what happened after it
//     was submitted — Step, NonFinite, DeviceLost, Cancelled — and an empty
//     vector.
//   - The context offered is the smaller of kPrefillBlock, 512 tokens, and
//     the model's trained context (residency/plan.h): the most one step
//     holds. EmbeddingGemma 2 is trained to 8,192 by its card and declares
//     262,144 in its file; a text longer than 510 tokens of its own is
//     refused, named, and a page splits it — document retrieval embeds
//     chunks at this size or smaller. Longer inputs take a pass whose
//     layers each run over the whole sequence before the next begins,
//     with working buffers sized to it: another design, not this one.
//   - One encode at a time: a second while one runs is refused, Busy, at
//     once. A lost device refuses every later one. A step's failure is
//     reported, through the callback, with WebGPU's message, and nothing is
//     kept, so the next encode is unaffected.
//   - Failure is visible: an embedding is delivered only from a completed
//     mapping (kernels/program.h), so a lost device cannot report a vector
//     it did not compute. A non-finite component — the pass overflowing —
//     is reported as NonFinite, the vector not delivered, rather than
//     handed to a page that would rank by it.
//   - It owns the tokenizer and the program; its callbacks carry its state,
//     held while an encode runs, as the generator's are, so destroying an
//     embedder mid-encode reports Cancelled. The upload the program borrows
//     outlives it (I.11).
//
// What it costs, counted, for EmbeddingGemma 2 at Q8_0 on the target; N the
// text's tokens with its ends:
//   - On the CPU, once an encode: the text's B normalized bytes encoded,
//     about 43 ns a byte as SentencePiece-style BPE runs
//     (tokenizer/bpe/gemma4_bpe.h). Admission bounds what is merged: text
//     past the limit × longest_cover() normalized bytes makes more tokens
//     than the limit at least, and is refused unmerged (tokenizer.h) — so B
//     is at most 510 × the vocabulary's longest cover, found at load, and
//     the encode at most 43 ns × that. Then N identifiers written into the
//     step — one write of 48 + 16 × ceil(N / 4) bytes; one program run: 17
//     calls into WebGPU reading back, and 2 or 3 a launch
//     (kernels/program.h); one finite check over 768 floats; one callback.
//   - Allocations an encode: the tokenizer's working lists, which every
//     encoder allocates afresh each encode — its normalized text, segments
//     and pieces, and for a piece too long for the rescan its symbols and
//     their heap — as generator.h counts for a turn. The embedder's own
//     path allocates nothing: the tokens are reserved at the context, the
//     step and the readback slots at load (MEM.9).
//   - On the GPU: 296 dispatches a step (graph/encoder.h), about 0.44 ms at
//     1.5 µs each (kernels/interface.h); the layers' weights, 151 MB with
//     the per-layer inputs' projection, read once a 32-token tile of the
//     step — once for N up to 32, 0.38 ms at 400 GB/s; and 1.36 × 10⁸
//     multiply-adds a token in the products, with attention's 57,344 N²
//     beside them. At N = 512 that is 8.5 × 10¹⁰: about 59 ms at the 2.9
//     TFLOPS Qwen3's 951-token prefill reached on the target
//     (docs/research/2026-10-09-webkit-shader-compiler.md, 285 ms for 8.4 ×
//     10¹¹ operations) — an estimate from a measured rate, not a count of
//     this pass. A query of 16 tokens is 2.2 × 10⁹ multiply-adds, about 1.5
//     ms at that rate, beside which the dispatches' 0.44 ms and the
//     weights' one read overlap; a step of 16 tokens fills a 16-token tile,
//     whose rate the 951-token measurement does not give, so this too is
//     an estimate.
//   - Its result waits on the one step and its 3 KiB mapping, about 0.5 ms
//     median on the target (sampler/sampler.h's map): about 2 ms for a
//     16-token query and 60 ms for 512 tokens, by the estimates above.
//     Measured on the page in Chrome: 13.9 ms for 21 tokens, 15.6 ms for
//     70 and 38.5 ms for 418 — the long text on its estimate, the short
//     ones carrying 11 to 13 ms that does not scale with tokens, which this
//     count does not account for and the worker's hops do not explain
//     (docs/research/2026-10-11-chrome-encode-benchmark.md).
//
// What it asks of the other contracts:
//   - kernels/program.h: a readback of up to kMaxReadback, 4 KiB, so a
//     program reads back an embedding as it reads back the draw's record —
//     two slots of the range's size, copied after the step's pass, mapped
//     once submitted. A generator's range is its 16 bytes.
//   - tokenizer/tokenizer.h: nothing; the embedder calls encode with its
//     limit, as the generator does, and adds the ends itself.
//   - preflight/preflight.h: Run, for a model whose architecture encodes
//     (arch/architecture.h), is its encoder's graph building, its tokenizer
//     loading, and its ends resolving (ends.h) — no stop set, which a model
//     that never samples has no use for.
//   - src/wasm/bindings.cpp: bllm_load_finish builds an embedder, not a
//     generator, for an architecture that encodes, answering with the
//     context offered and the embedding's length; and one crossing more,
//       bllm_embed(request, text, text_length)
//     answered through bllm_vector(request, pointer, length) — a
//     Float32Array view of the module's memory, valid for that call, the
//     vector's one crossing (WASM.2) — then once with the tokens the text
//     made, or with the refusal or failure, named. bllm_generate refuses a
//     model that encodes, and bllm_embed one that generates, each by name.
//
// Verification the implementation is held to:
//   - Against the reference, EmbeddingGemma 2 at Q8_0, the file pinned by
//     SHA-256 and fetched as test data: for each of a set of pinned texts —
//     a short query, a passage of a few hundred tokens, one of exactly 510
//     tokens of its own, and text in three scripts — the embedding llama.cpp's
//     llama-embedding gives, mean pooling and its own normalization, on its
//     CPU backend and on Metal, kept as a fixture by
//     tools/make_reference_embeddings.sh. Each of ours is within the
//     largest difference llama.cpp shows between its own two backends on
//     that text of Metal's, component by component, and its cosine with
//     Metal's at least the CPU's with Metal's: Metal the reference, as the
//     logits' reference test takes it. The check discriminates: the model
//     with its RoPE pairing swapped falls outside on every text. llama.cpp
//     is pinned for
//     this tool at a commit of its own, a9d27ac693ff0a1d3cba449c912c2b891904a4c7,
//     the first holding gemma-embedding2 (src/models/gemma-embedding2.cpp);
//     the logits fixtures keep their pin, 988190680d5a89fce97de3c20df2c2813731fd61,
//     and are not regenerated. Two pins of one dependency, each for one
//     tool's fixtures, so advancing either regenerates only its own.
//   - The tokens, before the GPU: each pinned text's identifiers, ends
//     included, those llama-tokenize gives at the same pin
//     (tokenizer/bpe/gemma4_bpe.h).
//   - On the GPU, the same text embedded twice gives the same bits; an
//     encode after a refused one gives the bits a fresh embedder gives; a
//     step that reads back no vector is Step; a lost device DeviceLost, and
//     every later encode refused so; destroying an embedder mid-encode
//     reports Cancelled; and each refusal — Busy, Text, TooManyTokens with
//     its count — named, a text admitted that merges past the context among
//     them, refused after its encode with the count it made. ends.h's
//     refusals, on the CPU, over headers with each flag removed or retyped.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — the tokenizer and the
//            program owned, by unique_ptr; the upload borrowed, and stated so.
//     E.27   Use error codes systematically — EmbedError, each refusal
//            named.
//     SL.io.2 When reading, always consider ill-formed input — the text is
//            bounded by the context before it is merged; a file without its
//            ends declared is refused, not guessed at.
//     SL.con.3 Avoid bounds errors — the tokens counted after the encode
//            against the step's kPrefillBlock identifiers.
//     I.4    Make interfaces precisely and strongly typed — the vector is a
//            span of floats, its length the model's.
//   C++ performance guidelines
//     GPU.1  Keep data on the device; every host-device round trip needs a
//            budget — one 3 KiB readback an encode; nothing else leaves the
//            GPU.
//     WASM.2 Batch work across the JS boundary — the text crosses once, the
//            vector once, as a view.
//     MEM.9  Allocate at init — the tokens, the step and the readback slots at
//            load; the embedder's own path allocates nothing an encode, and
//            the tokenizer's working lists are counted above.
//     GPU.6  Batch tiny GPU work — the whole text one step, one pass, one
//            submit.

enum class EmbedError {
    Ok,
    Busy,             // an encode is running
    DeviceLost,       // a step reported the device lost; no encode can run
    TooManyTokens,    // the text makes more than the context offered, ends included
    Text,             // the tokenizer refused the text: not UTF-8, or too long to encode
    EmptyText,        // no tokens at all: an empty text, and a file that adds no ends
    NonFinite,        // a component of the vector is not finite
    Step,             // the step failed; the message is WebGPU's
    Cancelled,        // the embedder was destroyed with the encode in flight
};

// An encode's outcome. `tokens` is what the text made with its ends, or, for
// TooManyTokens, the count that refused it — the least it makes, where
// admission refused it, or what it made, where the count after the encode
// did — with `context` the most one encode holds, so the page can cut the
// text by the excess.
struct EmbedResult {
    EmbedError error = EmbedError::Ok;
    tokenizer::EncodeResult encode;   // the tokenizer's refusal, where it refused
    std::string_view message;         // WebGPU's, for Step; valid only during the call
    std::uint64_t tokens = 0;
    std::uint32_t context = 0;
};

// Exactly once for each encode start() accepts: the unit vector with Ok, or
// an empty vector with Step, NonFinite, DeviceLost or Cancelled. Both are
// valid only during the call.
using EmbedCallback = void (*)(const EmbedResult& result, std::span<const float> embedding, void* userdata);

class Embedder {
public:
    // Preconditions: `tokenizer` is the model's; `program` was built for its
    // encoder's graph, reading back its `embedding` working buffer
    // (graph/encoder.h); `ends` are the file's (ends.h); `context` is the
    // context the plan offers; `dimensions` the encoder's embedding length
    // (model_description.h).
    Embedder(std::unique_ptr<tokenizer::Tokenizer> tokenizer, std::unique_ptr<kernels::Program> program, Ends ends,
             std::uint32_t context, std::uint32_t dimensions);

    // Starts an encode of `text`. Returns Ok once the step is submitted, and
    // then `done` is called exactly once, however the encode ends; returns a
    // refusal — Busy, DeviceLost, Text, TooManyTokens, EmptyText — without
    // running anything, and then `done` is never called. `text` is not read
    // after the call.
    [[nodiscard]] EmbedResult start(std::string_view text, EmbedCallback done, void* userdata);

    // The most tokens one encode holds, its ends included.
    [[nodiscard]] std::uint32_t context() const noexcept;

    // The embedding's length: embedding_length_out, 768 for EmbeddingGemma 2.
    [[nodiscard]] std::uint32_t dimensions() const noexcept;

    ~Embedder();
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    // What the embedder and its in-flight callbacks share; defined only where
    // the embedder is.
    struct State;

private:
    std::shared_ptr<State> state_;   // shared with in-flight callbacks
};

}  // namespace bllm::runtime
