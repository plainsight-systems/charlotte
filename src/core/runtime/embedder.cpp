#include "core/runtime/embedder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace bllm::runtime {

namespace {

constexpr std::size_t kMediaKinds = 2;

constexpr std::size_t index_of(arch::MediaKind kind) noexcept { return static_cast<std::size_t>(kind); }
static_assert(index_of(arch::MediaKind::Image) < kMediaKinds && index_of(arch::MediaKind::Audio) < kMediaKinds);

}  // namespace

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

    // A stage a kind, once attached (runtime/media_stage.h), indexed by
    // MediaKind, and each kind's tokens.
    std::array<std::unique_ptr<MediaStage>, kMediaKinds> stages;
    std::array<MediaTokens, kMediaKinds> media_tokens{};
    std::uint32_t width = 0;
    // The running encode's steps not yet reported, and the first failure
    // among those that have: delivered once the count reaches zero
    // (runtime/media_stage.h), since two programs' reports are not ordered.
    std::uint32_t outstanding = 0;
    kernels::ProgramError failure = kernels::ProgramError::Ok;
    std::string failure_message;
    bool read_back = false;   // the text step's vector copied into `vector`

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

// A report's failure, kept where it is the first.
void keep_failure(Embedder::State& s, kernels::ProgramError error, std::string_view message) {
    if (error == kernels::ProgramError::Ok || s.failure != kernels::ProgramError::Ok) return;
    s.failure = error;
    s.failure_message.assign(message);
}

// One step reported: once every step has, the encode's one callback — the
// vector only if every report was Ok.
void reported(Embedder::State& s) {
    if (--s.outstanding > 0) return;
    const std::shared_ptr<Embedder::State> keep = std::move(s.in_flight);   // the callback may destroy the embedder
    EmbedResult result;
    result.tokens = s.encoded;
    result.context = s.context;
    std::span<const float> vector;
    if (s.failure != kernels::ProgramError::Ok) {
        if (s.failure == kernels::ProgramError::DeviceLost) s.lost = true;
        result.error = error_of(s.failure);
        result.message = s.failure_message;
    } else if (!s.read_back) {
        // A step that read nothing back cannot have its vector delivered.
        result.error = EmbedError::Step;
        result.message = "the step read back no embedding";
    } else if (std::all_of(s.vector.begin(), s.vector.end(), [](float f) { return std::isfinite(f); })) {
        vector = s.vector;
    } else {
        result.error = EmbedError::NonFinite;
    }
    s.done(result, vector, s.userdata);
}

// The text step's report: its vector, from its completed mapping, copied
// while the mapping is valid; delivered with the others' reports.
void on_step(kernels::ProgramError error, std::string_view message, std::span<const std::byte> bytes,
             void* userdata) {
    Embedder::State& s = *static_cast<Embedder::State*>(userdata);
    keep_failure(s, error, message);
    if (error == kernels::ProgramError::Ok && bytes.size() == std::size_t{s.dimensions} * sizeof(float)) {
        std::memcpy(s.vector.data(), bytes.data(), bytes.size());
        s.read_back = true;
    }
    reported(s);
}

// The media step's report: nothing to deliver, a failure kept.
void on_media(kernels::ProgramError error, std::string_view message, std::span<const std::byte>, void* userdata) {
    Embedder::State& s = *static_cast<Embedder::State*>(userdata);
    keep_failure(s, error, message);
    reported(s);
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
    for (auto& stage : state_->stages) stage.reset();
    state_->program.reset();
}

void Embedder::attach(arch::MediaKind kind, std::unique_ptr<MediaStage> stage, MediaTokens tokens,
                      std::uint32_t width) {
    state_->stages[index_of(kind)] = std::move(stage);
    state_->media_tokens[index_of(kind)] = tokens;
    state_->width = width;
}

MediaStage* Embedder::stage(arch::MediaKind kind) noexcept { return state_->stages[index_of(kind)].get(); }

const tokenizer::Tokenizer& Embedder::tokenizer() const noexcept { return *state_->tokenizer; }

EmbedResult Embedder::start(std::string_view text, EmbedCallback done, void* userdata) {
    return start(text, std::nullopt, done, userdata);
}

EmbedResult Embedder::start(std::string_view text, std::optional<arch::MediaKind> media, EmbedCallback done,
                            void* userdata) {
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
    // The medium's rows and the two tokens around them, in the placeholder's
    // place, so the text may make M + 1 fewer.
    MediaStage* const stage = media ? s.stages[index_of(*media)].get() : nullptr;
    const std::uint32_t rows = stage != nullptr ? stage->ready_rows() : 0;
    if (media && rows == 0) {
        result.error = EmbedError::MediaNotReady;
        return result;
    }
    const std::uint32_t ends = (s.ends.bos ? 1 : 0) + (s.ends.eos ? 1 : 0) + (media ? rows + 1 : 0);
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
    // Each placeholder of an attached kind, or the encode's: the encode's
    // kind's once, every other's never.
    std::size_t placeholder = s.tokens.size();
    for (std::size_t k = 0; k < kMediaKinds; ++k) {
        const bool carried = media && index_of(*media) == k;
        if (s.stages[k] == nullptr && !carried) continue;
        std::size_t placeholders = 0;
        for (std::size_t i = 0; i < s.tokens.size(); ++i) {
            if (s.tokens[i] == s.media_tokens[k].placeholder) {
                if (carried) placeholder = i;
                ++placeholders;
            }
        }
        if (placeholders != (carried ? 1u : 0u)) {
            result.error = EmbedError::Placeholder;
            return result;
        }
    }
    // Replaced by the kind's opening token, its rows and its closing token;
    // the rows carry the placeholder, which gather skips, the copy filling
    // them (gather.h).
    const tokenizer::TokenId placeholder_id = media ? s.media_tokens[index_of(*media)].placeholder : tokenizer::TokenId{0};
    if (media) {
        const MediaTokens& tokens = s.media_tokens[index_of(*media)];
        s.tokens[placeholder] = tokens.begin;
        s.tokens.insert(s.tokens.begin() + static_cast<std::ptrdiff_t>(placeholder) + 1, rows + 1, placeholder_id);
        s.tokens[placeholder + 1 + rows] = tokens.end;
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
    s.failure = kernels::ProgramError::Ok;
    s.failure_message.clear();
    s.read_back = false;
    if (!media) {
        s.outstanding = 1;
        s.program->run(s.step, on_step, &s);
        result.tokens = s.encoded;
        return result;
    }
    // The media step, then the text step with the medium's rows copied into
    // its hidden rows before its pass: two submits, in queue order, with no
    // wait between them (GPU.6), and two reports, in no order, awaited both.
    s.outstanding = 2;
    s.step.media = 1;
    s.step.media_token = static_cast<std::uint32_t>(placeholder_id);
    stage->run(on_media, &s);
    const std::uint64_t row_bytes = std::uint64_t{s.width} * sizeof(float);
    const kernels::RunCopy copy{&stage->link(), 0, (placeholder + 1) * row_bytes, rows * row_bytes};
    s.program->run(s.step, {}, std::span(&copy, 1), on_step, &s);
    result.tokens = s.encoded;
    return result;
}

std::uint32_t Embedder::context() const noexcept { return state_->context; }

std::uint32_t Embedder::most_media_rows() const noexcept {
    const State& s = *state_;
    const std::uint32_t reserved = (s.ends.bos ? 1u : 0u) + (s.ends.eos ? 1u : 0u) + 2u;
    return s.context > reserved ? s.context - reserved : 0;
}

std::uint32_t Embedder::dimensions() const noexcept { return state_->dimensions; }

}  // namespace bllm::runtime
