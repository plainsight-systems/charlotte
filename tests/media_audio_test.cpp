// An audio clip's preparation (media/audio.h) against llama.cpp's: the
// frames its preprocessor keeps and the rows gemma4a makes of them, over a
// table of lengths, and every frame of three half-second spectrograms — a
// tone, a siren and the countdown recording — from llama.cpp's gemma4a
// preprocessor with Google's floor, kept by tools/make_reference_audio.sh.

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/media/audio.h"
#include "fixtures/audio/sounds.h"
#include "fixtures/audio/wav.h"

namespace {

#include "fixtures/reference/audio_prepare.inc"

namespace fx = bllm::fixtures::audio;
using bllm::media::AudioClip;
using bllm::media::AudioError;
using bllm::media::kMelBins;

constexpr std::uint32_t kMostRows = 508;   // EmbeddingGemma 2's context of 512, less 4
constexpr std::uint32_t kHalf = fx::kRate / 2;
// Each bin within these of llama.cpp's, absolute (media/audio.h): a bin above
// log 0.01, and any bin, the quiet ones carrying the twiddles' last bits.
constexpr float kLoudTolerance = 1e-4f;
constexpr float kTolerance = 5e-3f;
// The hash of the tone's half-second spectrogram (below).
constexpr std::uint64_t kToneHash = 5616082597242563373ull;

std::vector<float> samples_of(std::string_view name) {
    if (name == "countdown") {
        std::string error;
        std::vector<float> wav = fx::read_wav(std::string(BLLM_AUDIO_FIXTURE_DIR) + "/countdown.wav", error);
        REQUIRE_MESSAGE(!wav.empty(), error);
        wav.resize(kHalf);
        return wav;
    }
    for (const fx::Sound& s : fx::kSounds) {
        if (s.name != name) continue;
        std::vector<float> out(kHalf);
        for (std::uint32_t i = 0; i < kHalf; ++i) out[i] = fx::sample(s, i);
        return out;
    }
    FAIL("no sound named " << name);
    return {};
}

std::vector<float> spectrogram(const std::vector<float>& samples, AudioClip& clip) {
    REQUIRE(bllm::media::audio_clip(samples.size(), kMostRows, clip) == AudioError::Ok);
    std::vector<float> mel(std::size_t{clip.frames} * kMelBins);
    bllm::media::Spectrogram spectrogram;
    spectrogram.compute(samples, clip, mel);
    return mel;
}

}  // namespace

TEST_CASE("a clip's frames and rows are llama.cpp's, the most at 508 rows 640 R + 160 samples") {
    CHECK(bllm::media::clip_samples(kMostRows) == 325'280);
    for (const AudioClipReference& r : kClipReferences) {
        CAPTURE(r.samples);
        AudioClip clip{};
        if (r.samples == 160) {
            // llama.cpp keeps one frame of 2 to 160 samples, its frame count
            // (n + 160 − 321) / 160 + 1 truncating a negative toward zero;
            // Google's, floored, keeps none, and so does this.
            CHECK(bllm::media::audio_clip(r.samples, kMostRows, clip) == AudioError::TooShort);
            continue;
        }
        // A clip one row past the most: refused at 508, as llama.cpp counts
        // it at 509.
        const std::uint32_t most = r.rows > kMostRows ? r.rows : kMostRows;
        REQUIRE(bllm::media::audio_clip(r.samples, most, clip) == AudioError::Ok);
        CHECK(clip.frames == r.frames);
        CHECK(clip.rows == r.rows);
        if (r.rows > kMostRows) CHECK(bllm::media::audio_clip(r.samples, kMostRows, clip) == AudioError::TooLong);
    }
    AudioClip clip{};
    CHECK(bllm::media::audio_clip(0, kMostRows, clip) == AudioError::TooShort);
    CHECK(bllm::media::audio_clip(std::uint64_t{1} << 40, kMostRows, clip) == AudioError::TooLong);
}

TEST_CASE("every bin of three spectrograms is llama.cpp's with Google's floor: 1e-4 where loud, 5e-3 anywhere") {
    for (const AudioMelReference& r : kMelReferences) {
        CAPTURE(r.name);
        AudioClip clip{};
        const std::vector<float> mel = spectrogram(samples_of(r.name), clip);
        REQUIRE(mel.size() == r.mel.size());
        float worst = 0.0f, worst_loud = 0.0f;
        for (std::size_t i = 0; i < mel.size(); ++i) {
            const float d = std::abs(mel[i] - r.mel[i]);
            worst = std::max(worst, d);
            if (r.mel[i] > std::log(0.01f)) worst_loud = std::max(worst_loud, d);
        }
        MESSAGE(r.name << ": largest difference " << worst << ", above 0.01: " << worst_loud);
        CHECK(worst_loud <= kLoudTolerance);
        CHECK(worst <= kTolerance);
    }
}

TEST_CASE("the check tells the floors apart: llama.cpp's own form misses the tone's quiet bins by more") {
    // Recovered from the fixture, which is log(x + f): x, then llama.cpp's
    // unchanged log(max(x, f)). A bin whose two values differ by more than
    // the tolerance would fail the check above had the module taken that
    // form.
    const AudioMelReference& tone = kMelReferences[0];
    REQUIRE(tone.name == "tone");
    const double floor = static_cast<double>(bllm::media::kMelFloor);
    std::size_t apart = 0;
    for (const float v : tone.mel) {
        const double x = std::exp(static_cast<double>(v)) - floor;
        if (std::abs(static_cast<double>(v) - std::log(std::max(x, floor))) > 0.1) ++apart;
    }
    MESSAGE(apart << " of " << tone.mel.size() << " bins more than 0.1 apart");
    CHECK(apart >= 100);
}

// FNV-1a over a spectrogram's bytes.
std::uint64_t hash_of(const std::vector<float>& mel) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const float f : mel) {
        std::uint32_t bits;
        std::memcpy(&bits, &f, sizeof bits);
        for (int i = 0; i < 4; ++i) {
            h ^= (bits >> (8 * i)) & 0xffu;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

TEST_CASE("the same samples give the same bytes, run to run and build to build") {
    const std::vector<float> samples = samples_of("tone");
    AudioClip a{}, b{};
    const std::vector<float> first = spectrogram(samples, a);
    const std::vector<float> second = spectrogram(samples, b);
    CHECK(first == second);
    // The tone's half second, as every build of the module computes it:
    // pinned, so a build whose optimizer or math library moved a bit fails.
    CHECK(hash_of(first) == kToneHash);
}
