#include "core/media/audio.h"

#include <algorithm>
#include <cmath>
#include <limits>

// No multiply and add fused into one rounding here: WebAssembly has no fused
// multiply-add in its core instructions, so the module never fuses, and a
// native build that did would round differently (media/audio.h).
#pragma clang fp contract(off)

namespace bllm::media {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::uint32_t kLeftPad = kFrameLength / 2;   // 160 zeros before the first sample
constexpr std::uint32_t kFftBits = 9;                  // 512 = 2⁹
static_assert(std::uint32_t{1} << kFftBits == kFftLength);

double hz_to_mel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double mel_to_hz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

}  // namespace

AudioError audio_clip(std::uint64_t samples, std::uint32_t most_rows, AudioClip& out) noexcept {
    if (samples < kFrameLength / 2 + 1) return AudioError::TooShort;
    if (samples > clip_samples(most_rows)) return AudioError::TooLong;
    const auto frames = static_cast<std::uint32_t>((samples - (kFrameLength / 2 + 1)) / kHop + 1);
    // The subsampling's two stride-2 convolutions: ⌈⌈T / 2⌉ / 2⌉.
    const std::uint32_t rows = ((frames + 1) / 2 + 1) / 2;
    out = {static_cast<std::uint32_t>(samples), frames, rows};
    return rows > most_rows ? AudioError::TooLong : AudioError::Ok;
}

std::uint32_t clip_samples(std::uint32_t most_rows) noexcept {
    // The last length whose frames number 4 R: T = ⌊(n − 161) / 160⌋ + 1 ≤ 4 R.
    return kHop * 4 * most_rows + kFrameLength / 2;
}

Spectrogram::Spectrogram() {
    // The periodic Hann window of 320 and the twiddles, a whole turn of 512,
    // each computed in f64 and rounded once to f32: the same values from any
    // math library, where f32 sinf and cosf, which llama.cpp calls, differ
    // between libraries and between one call and a fused sincos.
    for (std::uint32_t j = 0; j < kFrameLength; ++j) {
        window_[j] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * j / kFrameLength));
    }
    for (std::uint32_t k = 0; k < kFftLength; ++k) {
        const double theta = 2.0 * kPi * k / kFftLength;
        cos_[k] = static_cast<float>(std::cos(theta));
        sin_[k] = static_cast<float>(std::sin(theta));
    }
    // 128 triangular filters on the HTK mel scale, 0 to 8 kHz, unnormalized,
    // computed in f64 as llama.cpp's fill_mel_filterbank_matrix computes
    // them; each kept as its nonzero span.
    const double low = hz_to_mel(0.0);
    const double high = hz_to_mel(kSampleRate / 2.0);
    std::array<double, kMelBins + 2> edges{};
    for (std::uint32_t i = 0; i < kMelBins + 2; ++i) {
        edges[i] = mel_to_hz(low + (high - low) * (static_cast<double>(i) / (kMelBins + 1)));
    }
    const double step = static_cast<double>(kSampleRate) / kFftLength;
    for (std::uint32_t m = 0; m < kMelBins; ++m) {
        const double left = edges[m], centre = edges[m + 1], right = edges[m + 2];
        const double rise = std::max(1e-30, centre - left);
        const double fall = std::max(1e-30, right - centre);
        std::array<float, kFftBins> row{};
        for (std::uint32_t k = 0; k < kFftBins; ++k) {
            const double f = k * step;
            double w = 0.0;
            if (f >= left && f <= centre) {
                w = (f - left) / rise;
            } else if (f > centre && f <= right) {
                w = (right - f) / fall;
            }
            row[k] = static_cast<float>(w);
        }
        std::uint32_t first = 0;
        while (first < kFftBins && row[first] == 0.0f) ++first;
        std::uint32_t end = kFftBins;
        while (end > first && row[end - 1] == 0.0f) --end;
        filters_[m] = {first, end - first, static_cast<std::uint32_t>(weights_.size())};
        weights_.insert(weights_.end(), row.begin() + first, row.begin() + end);
    }
}

namespace {

// llama.cpp's fft_impl<false, true> (mtmd-audio.cpp), its operations in its
// order: the real input `in` of n floats, its evens and odds copied past it
// and transformed into `out` past the n complex outputs, then combined.
// `in` holds 2n floats from where it points, `out` 8n; the twiddles are a
// table of a whole turn of kFftLength.
void fft(const float* cos_table, const float* sin_table, float* in, std::uint32_t n, float* out) {
    if (n == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    const std::uint32_t half = n / 2;
    float* even = in + n;
    for (std::uint32_t i = 0; i < half; ++i) even[i] = in[2 * i];
    float* even_fft = out + 2 * n;
    fft(cos_table, sin_table, even, half, even_fft);
    float* odd = even;
    for (std::uint32_t i = 0; i < half; ++i) odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + n;
    fft(cos_table, sin_table, odd, half, odd_fft);
    const std::uint32_t step = kFftLength / n;
    for (std::uint32_t k = 0; k < half; ++k) {
        const float re = cos_table[k * step];
        const float im = -sin_table[k * step];
        const float re_odd = odd_fft[2 * k + 0];
        const float im_odd = odd_fft[2 * k + 1];
        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

}  // namespace

void Spectrogram::compute(std::span<const float> samples, const AudioClip& clip, std::span<float> out) noexcept {
    for (std::uint32_t t = 0; t < clip.frames; ++t) {
        // The frame: 320 windowed samples from padded index 160 t — sample
        // 160 t − 160 + j of the clip — then zeros to 512.
        input_.fill(0.0f);
        for (std::uint32_t j = 0; j < kFrameLength; ++j) {
            const std::int64_t at = std::int64_t{kHop} * t - kLeftPad + j;
            const float x = at >= 0 && at < clip.samples ? samples[static_cast<std::size_t>(at)] : 0.0f;
            input_[j] = window_[j] * x;
        }
        fft(cos_.data(), sin_.data(), input_.data(), kFftLength, output_.data());
        // The magnitudes of the 257 bins, as llama.cpp keeps them, over the
        // output's first floats.
        std::array<float, kFftBins> magnitude{};
        for (std::uint32_t k = 0; k < kFftBins; ++k) {
            const float power = output_[2 * k] * output_[2 * k] + output_[2 * k + 1] * output_[2 * k + 1];
            magnitude[k] = std::sqrt(power);
        }
        float* row = &out[std::size_t{t} * kMelBins];
        for (std::uint32_t m = 0; m < kMelBins; ++m) {
            const Filter& filter = filters_[m];
            const auto weight = [&](std::uint32_t k) {
                return k >= filter.first && k < filter.first + filter.count ? weights_[filter.weight + k - filter.first]
                                                                            : 0.0f;
            };
            double sum = 0.0;
            if (filter.count > 0) {
                // Groups of four from bin 0, the ones the span touches, then
                // bin 256 alone, as llama.cpp's loop adds them.
                const std::uint32_t last = filter.first + filter.count - 1;
                for (std::uint32_t k = filter.first / 4 * 4; k <= last && k + 3 < kFftBins - 1; k += 4) {
                    sum += magnitude[k + 0] * weight(k + 0) + magnitude[k + 1] * weight(k + 1) +
                           magnitude[k + 2] * weight(k + 2) + magnitude[k + 3] * weight(k + 3);
                }
                if (last == kFftBins - 1) sum += magnitude[kFftBins - 1] * weight(kFftBins - 1);
            }
            row[m] = static_cast<float>(std::log(sum + static_cast<double>(kMelFloor)));
        }
    }
}

}  // namespace bllm::media
