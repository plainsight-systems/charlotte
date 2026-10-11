#include "core/runtime/embedder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
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

    // The vision encoder, once attached (runtime/vision.h).
    std::unique_ptr<Vision> vision;
    ImageTokens image_tokens{};
    std::uint32_t width = 0;
    // The running encode's vision step, where it has one: its failure,
    // reported in the text step's place.
    kernels::ProgramError vision_error = kernels::ProgramError::Ok;
    std::string vision_message;

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
    // The vision step ran before this one and failed: its failure is the
    // encode's.
    if (s.vision_error != kernels::ProgramError::Ok) {
        error = s.vision_error;
        message = s.vision_message;
    }
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

// The vision step's report: nothing to deliver, a failure kept for the text
// step's report, which follows it in queue order.
void on_vision(kernels::ProgramError error, std::string_view message, std::span<const std::byte>, void* userdata) {
    Embedder::State& s = *static_cast<Embedder::State*>(userdata);
    if (error != kernels::ProgramError::Ok && s.vision_error == kernels::ProgramError::Ok) {
        s.vision_error = error;
        s.vision_message.assign(message);
    }
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
    // An encode running finishes Cancelled: the programs, destroyed, report
    // their steps so, and this state lives until its callback has run.
    state_->vision.reset();
    state_->program.reset();
}

void Embedder::attach(std::unique_ptr<Vision> vision, ImageTokens tokens, std::uint32_t width) {
    state_->vision = std::move(vision);
    state_->image_tokens = tokens;
    state_->width = width;
}

Vision* Embedder::vision() noexcept { return state_->vision.get(); }

EmbedResult Embedder::start(std::string_view text, EmbedCallback done, void* userdata) {
    return start(text, false, done, userdata);
}

EmbedResult Embedder::start(std::string_view text, bool image, EmbedCallback done, void* userdata) {
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
    // The image's rows: its soft tokens and the two tokens around them, in
    // the placeholder's place, so the text may make M + 1 fewer.
    const std::uint32_t rows = image && s.vision != nullptr ? s.vision->ready_tokens() : 0;
    if (image && rows == 0) {
        result.error = EmbedError::ImageNotReady;
        return result;
    }
    const std::uint32_t ends = (s.ends.bos ? 1 : 0) + (s.ends.eos ? 1 : 0) + (image ? rows + 1 : 0);
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
    // The placeholder: once with an image, never without.
    std::size_t placeholder = s.tokens.size();
    std::size_t placeholders = 0;
    if (s.vision != nullptr || image) {
        for (std::size_t i = 0; i < s.tokens.size(); ++i) {
            if (s.tokens[i] == s.image_tokens.placeholder) {
                placeholder = i;
                ++placeholders;
            }
        }
    }
    if (placeholders != (image ? 1u : 0u)) {
        result.error = EmbedError::Placeholder;
        return result;
    }
    // Replaced by the image's opening token, its rows and its closing token;
    // a row's identifier is never read, the copy fills it (gather.h).
    if (image) {
        s.tokens[placeholder] = s.image_tokens.begin;
        s.tokens.insert(s.tokens.begin() + static_cast<std::ptrdiff_t>(placeholder) + 1, rows + 1, tokenizer::TokenId{0});
        s.tokens[placeholder + 1 + rows] = s.image_tokens.end;
    }
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
    s.vision_error = kernels::ProgramError::Ok;
    s.vision_message.clear();
    if (!image) {
        s.program->run(s.step, on_step, &s);
        result.tokens = s.encoded;
        return result;
    }
    // The vision step, then the text step with the image's rows copied into
    // its hidden rows before its pass: two submits, in queue order, with no
    // wait between them (GPU.6).
    s.step.media_first = static_cast<std::uint32_t>(placeholder + 1);
    s.step.media_rows = rows;
    s.vision->run(on_vision, &s);
    const std::uint64_t row_bytes = std::uint64_t{s.width} * sizeof(float);
    const kernels::RunCopy copy{&s.vision->link(), 0, s.step.media_first * row_bytes, rows * row_bytes};
    s.program->run(s.step, {}, std::span(&copy, 1), on_step, &s);
    result.tokens = s.encoded;
    return result;
}

std::uint32_t Embedder::context() const noexcept { return state_->context; }

std::uint32_t Embedder::dimensions() const noexcept { return state_->dimensions; }

}  // namespace bllm::runtime
