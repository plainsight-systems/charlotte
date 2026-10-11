#include "core/runtime/embedder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace bllm::runtime {

struct Embedder::State {
    std::unique_ptr<tokenizer::Tokenizer> tokenizer;
    std::unique_ptr<kernels::Program> program;   // null once the embedder is destroyed
    Ends ends;
    std::uint32_t context = 0;
    std::uint32_t dimensions = 0;
    std::vector<tokenizer::TokenId> tokens;       // reserved at the context
    std::vector<float> vector;                    // reserved at the dimensions
    kernels::Step step{};
    bool lost = false;   // a step reported the device lost

    // The running encode.
    EmbedCallback done = nullptr;
    void* userdata = nullptr;
    std::uint64_t encoded = 0;   // its tokens, ends included
    // This state, held while an encode runs, so its callback outlives the embedder.
    std::shared_ptr<State> in_flight;
};

namespace {

EmbedError error_of(kernels::ProgramError error) {
    switch (error) {
        case kernels::ProgramError::DeviceLost: return EmbedError::DeviceLost;
        case kernels::ProgramError::Cancelled: return EmbedError::Cancelled;
        default: return EmbedError::Step;
    }
}

// The step's report: the vector from its completed mapping, or what failed.
void on_step(kernels::ProgramError error, std::string_view message, std::span<const std::byte> bytes,
             void* userdata) {
    Embedder::State& s = *static_cast<Embedder::State*>(userdata);
    const std::shared_ptr<Embedder::State> keep = std::move(s.in_flight);   // the callback may destroy the embedder
    EmbedResult result;
    result.tokens = s.encoded;
    result.context = s.context;
    std::span<const float> vector;
    if (error != kernels::ProgramError::Ok) {
        if (error == kernels::ProgramError::DeviceLost) s.lost = true;
        result.error = error_of(error);
        result.message = message;
    } else if (bytes.size() != std::size_t{s.dimensions} * sizeof(float)) {
        // A step that read nothing back cannot have its vector delivered.
        result.error = EmbedError::Step;
        result.message = "the step read back no embedding";
    } else {
        std::memcpy(s.vector.data(), bytes.data(), bytes.size());
        const bool finite = std::all_of(s.vector.begin(), s.vector.end(), [](float f) { return std::isfinite(f); });
        if (finite) {
            vector = s.vector;
        } else {
            result.error = EmbedError::NonFinite;
        }
    }
    s.done(result, vector, s.userdata);
}

}  // namespace

Embedder::Embedder(std::unique_ptr<tokenizer::Tokenizer> tokenizer, std::unique_ptr<kernels::Program> program,
                   Ends ends, std::uint32_t context, std::uint32_t dimensions)
    : state_(std::make_shared<State>()) {
    State& s = *state_;
    s.tokenizer = std::move(tokenizer);
    s.program = std::move(program);
    s.ends = ends;
    s.context = context;
    s.dimensions = dimensions;
    s.tokens.reserve(std::size_t{context} + 2);
    s.vector.resize(dimensions);
}

Embedder::~Embedder() {
    // An encode running finishes Cancelled: the program, destroyed, reports
    // its step so, and this state lives until its callback has run.
    state_->program.reset();
}

EmbedResult Embedder::start(std::string_view text, EmbedCallback done, void* userdata) {
    State& s = *state_;
    EmbedResult result;
    result.context = s.context;
    if (s.program == nullptr || s.in_flight != nullptr) {
        result.error = EmbedError::Busy;
        return result;
    }
    if (s.lost) {
        result.error = EmbedError::DeviceLost;
        return result;
    }
    const std::uint32_t ends = (s.ends.bos ? 1 : 0) + (s.ends.eos ? 1 : 0);
    const std::uint32_t limit = s.context > ends ? s.context - ends : 0;
    s.tokens.clear();
    if (s.ends.bos) s.tokens.push_back(*s.ends.bos);
    // Encoded with the context, less the ends, as its limit: text that makes
    // more at least is refused before it is merged.
    const tokenizer::EncodeResult encoded = s.tokenizer->encode(text, limit, s.tokens);
    if (encoded.error == tokenizer::EncodeError::TooManyTokens) {
        result.error = EmbedError::TooManyTokens;
        result.tokens = encoded.least_tokens + ends;
        return result;
    }
    if (encoded.error != tokenizer::EncodeError::Ok) {
        result.error = EmbedError::Text;
        result.encode = encoded;
        return result;
    }
    if (s.ends.eos) s.tokens.push_back(*s.ends.eos);
    // The least count admission takes is a lower bound: counted again, ends
    // included, before the step is written, which holds kPrefillBlock
    // identifiers and no more (SL.con.3).
    if (s.tokens.size() > s.context) {
        result.error = EmbedError::TooManyTokens;
        result.tokens = s.tokens.size();
        return result;
    }
    if (s.tokens.empty()) {
        result.error = EmbedError::EmptyText;
        return result;
    }

    s.step = {};
    s.step.position = 0;
    s.step.tokens = static_cast<std::uint32_t>(s.tokens.size());
    s.step.logits = 1;
    std::transform(s.tokens.begin(), s.tokens.end(), s.step.ids.begin(),
                   [](tokenizer::TokenId t) { return static_cast<std::uint32_t>(t); });
    s.done = done;
    s.userdata = userdata;
    s.encoded = s.tokens.size();
    s.in_flight = state_;
    s.program->run(s.step, on_step, &s);
    result.tokens = s.encoded;
    return result;
}

std::uint32_t Embedder::context() const noexcept { return state_->context; }

std::uint32_t Embedder::dimensions() const noexcept { return state_->dimensions; }

}  // namespace bllm::runtime
