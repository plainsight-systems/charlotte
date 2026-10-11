#pragma once

// The recordings beside this file as samples: 16-bit PCM WAV, mono, 16 kHz,
// each sample its integer over 32,768 — what llama.cpp's decoder and a
// browser's decodeAudioData give for such a file. For the audio reference
// tool and the tests alike; a file of any other form is refused, by name,
// rather than read some other way.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace bllm::fixtures::audio {

// The samples of the WAV at `path`, or an empty vector and `error` naming
// what is wrong.
inline std::vector<float> read_wav(const std::string& path, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> b((std::istreambuf_iterator<char>(file)), {});
    const auto u32 = [&](std::size_t at) { std::uint32_t v; std::memcpy(&v, &b[at], 4); return v; };
    const auto u16 = [&](std::size_t at) { std::uint16_t v; std::memcpy(&v, &b[at], 2); return v; };
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(&b[8], "WAVE", 4) != 0) {
        error = path + " is not a WAV file";
        return {};
    }
    bool formatted = false;
    for (std::size_t p = 12; p + 8 <= b.size();) {
        const std::uint32_t length = u32(p + 4);
        if (p + 8 + length > b.size()) break;
        if (std::memcmp(&b[p], "fmt ", 4) == 0) {
            if (length < 16 || u16(p + 8) != 1 || u16(p + 10) != 1 || u32(p + 12) != 16000 || u16(p + 22) != 16) {
                error = path + " is not 16-bit PCM, mono, at 16 kHz";
                return {};
            }
            formatted = true;
        } else if (std::memcmp(&b[p], "data", 4) == 0) {
            if (!formatted) break;
            std::vector<float> samples(length / 2);
            for (std::size_t i = 0; i < samples.size(); ++i) {
                std::int16_t v;
                std::memcpy(&v, &b[p + 8 + 2 * i], 2);
                samples[i] = static_cast<float>(v) / 32768.0f;
            }
            return samples;
        }
        p += 8 + length + (length & 1);
    }
    error = path + " has no format before its data";
    return {};
}

}  // namespace bllm::fixtures::audio
