#include "core/runtime/audio.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace bllm::runtime {
namespace {

// The relative positions a layer's table holds: distances 12 down to 0
// (graph/audio.h).
constexpr std::uint32_t kPositions = 13;

}  // namespace

Audio::Audio(std::unique_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
             kernels::Binding mel, std::uint32_t most_rows)
    : program_(std::move(program)), link_(std::move(link)), mel_(mel), most_rows_(most_rows) {
    // The slot holds the most frames a clip makes, at load (MEM.9).
    slot_.mel.resize(std::size_t{4} * most_rows * media::kMelBins);
}

Audio::~Audio() = default;

media::AudioError Audio::prepare(std::span<const float> samples) {
    slot_.ready = false;
    if (const auto e = media::audio_clip(samples.size(), most_rows_, slot_.clip); e != media::AudioError::Ok) {
        return e;
    }
    spectrogram_.compute(samples, slot_.clip,
                         std::span(slot_.mel).first(std::size_t{slot_.clip.frames} * media::kMelBins));
    segment_[0] = slot_.clip.rows;
    slot_.ready = true;
    return media::AudioError::Ok;
}

std::uint32_t Audio::most_samples() const noexcept { return media::clip_samples(most_rows_); }

std::span<const std::uint32_t> Audio::ready_segments() const noexcept {
    if (!slot_.ready) return {};
    return segment_;
}

void Audio::run(std::uint32_t, const SegmentCopy& out, kernels::StepCallback done, void* userdata) {
    kernels::Step step{};
    step.position = 0;
    step.tokens = slot_.clip.rows;
    step.columns = slot_.clip.frames;
    const std::size_t floats = std::size_t{slot_.clip.frames} * media::kMelBins;
    const kernels::RunInput input{mel_, std::as_bytes(std::span(slot_.mel).first(floats))};
    const kernels::RunCopy copy{link_.get(), 0, out.to_offset, out.bytes, kernels::CopyAt::AfterPass};
    // The write copies the spectrogram before run() returns: the slot is free.
    program_->run(step, std::span(&input, 1), std::span(&copy, 1), done, userdata);
    slot_.ready = false;
}

void run_audio_setup(kernels::Program& setup, kernels::Binding positions, std::uint32_t width,
                     kernels::StepCallback done, void* userdata) {
    // llama.cpp's gemma4a input: row p the position 12 − p, sines then
    // cosines over W / 2 timescales.
    const std::uint32_t timescales = width / 2;
    const float increment = static_cast<float>(std::log(10000.0)) / static_cast<float>(std::max(timescales - 1, 1u));
    std::vector<float> rows(std::size_t{kPositions} * width, 0.0f);
    for (std::uint32_t p = 0; p < kPositions; ++p) {
        const auto position = static_cast<float>(kPositions - 1 - p);
        for (std::uint32_t i = 0; i < timescales; ++i) {
            const auto inverse = static_cast<float>(std::exp(static_cast<double>(-static_cast<float>(i) * increment)));
            const float scaled = position * inverse;
            rows[std::size_t{p} * width + i] = static_cast<float>(std::sin(static_cast<double>(scaled)));
            rows[std::size_t{p} * width + i + timescales] = static_cast<float>(std::cos(static_cast<double>(scaled)));
        }
    }
    kernels::Step step{};
    step.position = 0;
    step.tokens = kPositions;
    const kernels::RunInput input{positions, std::as_bytes(std::span(rows))};
    // The write copies the rows before run() returns.
    setup.run(step, std::span(&input, 1), {}, done, userdata);
}

}  // namespace bllm::runtime
