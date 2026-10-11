#pragma once

// The sounds the audio reference tool gives llama.cpp
// (tools/reference_audio) and the tests give Charlotte
// (tests/media_audio_test.cpp, tests/gpu/audio_embed_test.cpp), each computed
// from its sample index at 16 kHz, so both see the same samples with no
// file, decoder or licence between them: a 1 kHz tone, a siren sweeping 600
// to 1,200 Hz, white noise, clicks twice a second and silence. A pure tone's
// spectrum lies near the log's floor in every bin but one, which is where
// llama.cpp's floor and Google's differ (media/audio.h). Speech comes from
// the recordings beside this file (SOURCES.md).

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace bllm::fixtures::audio {

enum class Kind {
    Tone,
    Siren,
    Noise,
    Clicks,
    Silence,
};

struct Sound {
    std::string_view name;
    std::uint32_t samples;
    Kind kind;
};

inline constexpr std::uint32_t kRate = 16000;

inline constexpr std::array<Sound, 5> kSounds{{
    {"tone", 3 * kRate, Kind::Tone},
    {"siren", 4 * kRate, Kind::Siren},
    {"noise", 3 * kRate, Kind::Noise},
    {"clicks", 3 * kRate, Kind::Clicks},
    {"silence", 3 * kRate, Kind::Silence},
}};

// A hash of the sample index, uniform in [−1, 1): the noise and the clicks'
// bursts, the same on any machine.
inline float noise(std::uint32_t i) {
    std::uint32_t h = i * 2654435761u;
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    return static_cast<float>(h >> 8) / 8388608.0f - 1.0f;
}

// Sample i of `sound`, in [−1, 1]. The siren's frequency is 900 + 300
// sin(πt) Hz, its phase that frequency's integral in closed form, computed
// in f64 so the sweep is the same however it is stepped.
inline float sample(const Sound& sound, std::uint32_t i) {
    constexpr double kPi = 3.14159265358979323846;
    const double t = static_cast<double>(i) / kRate;
    switch (sound.kind) {
        case Kind::Tone: return static_cast<float>(0.3 * std::sin(2 * kPi * 1000.0 * t));
        case Kind::Siren: {
            const double phase = 2 * kPi * (900.0 * t + 300.0 * (1.0 - std::cos(kPi * t)) / kPi);
            return static_cast<float>(0.3 * std::sin(phase));
        }
        case Kind::Noise: return 0.3f * noise(i);
        case Kind::Clicks: return i % (kRate / 2) < kRate / 100 ? 0.5f * noise(i) : 0.0f;
        case Kind::Silence: return 0.0f;
    }
    return 0.0f;
}

}  // namespace bllm::fixtures::audio
