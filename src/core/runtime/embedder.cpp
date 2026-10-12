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
    // The running encode's medium: its stage, its segments' rows and first
    // rows in the text step, kept from start() since the stage offers them
    // no longer once segment 0 runs; and how many are submitted (EMB.2).
    MediaStage* running = nullptr;
    std::array<std::uint32_t, kMostSegments> segment_rows{};
    std::array<std::uint32_t, kMostSegments> segment_first{};
    std::uint32_t segments = 0;
    std::uint32_t submitted = 0;

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

// The text step, behind the last segment on the queue.
void submit_text(Embedder::State& s) {
    ++s.outstanding;
    s.program->run(s.step, on_step, &s);
}

void on_media(kernels::ProgramError error, std::string_view message, std::span<const std::byte>, void* userdata);

// The next segment's step, its rows copied after its pass to their place in
// the text step's `hidden`; once the last is submitted, the text step, with
// no wait (GPU.7). Nothing more once a step has failed.
void submit_segment(Embedder::State& s) {
    const std::uint32_t i = s.submitted++;
    const std::uint64_t row_bytes = std::uint64_t{s.width} * sizeof(float);
    ++s.outstanding;
    s.running->run(i, SegmentCopy{s.segment_first[i] * row_bytes, s.segment_rows[i] * row_bytes}, on_media, &s);
    if (s.submitted == s.segments && s.failure == kernels::ProgramError::Ok) submit_text(s);
}

// A segment's report: nothing to deliver, a failure kept; and the next
// segment submitted in its place, two in flight at most, the program's
// limit (runtime/media_stage.h). Its own count is held until after, so a
// step refused at once cannot deliver the encode early.
void on_media(kernels::ProgramError error, std::string_view message, std::span<const std::byte>, void* userdata) {
    Embedder::State& s = *static_cast<Embedder::State*>(userdata);
    keep_failure(s, error, message);
    if (s.failure == kernels::ProgramError::Ok && s.program != nullptr && s.submitted < s.segments) submit_segment(s);
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
    // The medium's segments, each its rows and the two tokens around them,
    // in the placeholder's place, so the text may make that less one fewer.
    MediaStage* const stage = media ? s.stages[index_of(*media)].get() : nullptr;
    const std::span<const std::uint32_t> segments =
        stage != nullptr ? stage->ready_segments() : std::span<const std::uint32_t>{};
    // A stage offering more segments than an encode holds offers none it can take.
    if (media && (segments.empty() || segments.size() > kMostSegments)) {
        result.error = EmbedError::MediaNotReady;
        return result;
    }
    std::uint32_t media_rows = 0;
    for (const std::uint32_t rows : segments) media_rows += rows + 2;
    const std::uint32_t ends = (s.ends.bos ? 1 : 0) + (s.ends.eos ? 1 : 0) + (media ? media_rows - 1 : 0);
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
    // Replaced by, for each segment, the kind's opening token, its rows and
    // its closing token, back to back; the rows carry the placeholder, which
    // gather skips, the segment's copy filling them (gather.h).
    const tokenizer::TokenId placeholder_id = media ? s.media_tokens[index_of(*media)].placeholder : tokenizer::TokenId{0};
    if (media) {
        const MediaTokens& tokens = s.media_tokens[index_of(*media)];
        // The placeholder's place becomes the media rows' span, filled in order.
        s.tokens.insert(s.tokens.begin() + static_cast<std::ptrdiff_t>(placeholder), media_rows - 1, placeholder_id);
        std::size_t at = placeholder;
        for (std::size_t i = 0; i < segments.size(); ++i) {
            s.tokens[at] = tokens.begin;
            s.segment_rows[i] = segments[i];
            s.segment_first[i] = static_cast<std::uint32_t>(at + 1);
            at += std::size_t{segments[i]} + 1;
            s.tokens[at++] = tokens.end;
        }
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
    s.outstanding = 0;
    if (!media) {
        submit_text(s);
        result.tokens = s.encoded;
        return result;
    }
    // Each segment's step, its rows copied out after its pass, two in flight
    // at most, then the text step behind the last: submits in queue order
    // with no wait between them (GPU.6), and reports, in no order between
    // the two programs, all awaited. One count held while submitting, so a
    // step refused at once cannot deliver the encode before start() returns.
    s.step.media = 1;
    s.step.media_token = static_cast<std::uint32_t>(placeholder_id);
    s.running = stage;
    s.segments = static_cast<std::uint32_t>(segments.size());
    s.submitted = 0;
    ++s.outstanding;
    while (s.submitted < std::min(s.segments, 2u) && s.failure == kernels::ProgramError::Ok) submit_segment(s);
    result.tokens = s.encoded;
    reported(s);
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
