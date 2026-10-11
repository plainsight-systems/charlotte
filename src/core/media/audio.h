#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace bllm::media {

// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// A clip of audio made into what an audio encoder reads: its log-mel
// spectrogram, frames of 128 bins. What tokenization is to a text and
// resampling is to an image, this is to a waveform: the encoder's input,
// prepared on the CPU before anything reaches the GPU. Pure: no GPU, no
// browser, no file. The page decodes the recording, makes it mono at
// 16 kHz and cuts it into clips; the module is handed a clip's samples in
// one call (runtime/audio.h).
//
// It computes Google's feature extraction for EmbeddingGemma 2
// (Gemma4AudioFeatureExtractor, processor_config.json), which llama.cpp's
// gemma4a preprocessor at the embeddings' pin follows but in its log's floor
// (docs/research/2026-10-11-embeddinggemma2-audio.md). For n samples, f32,
// at 16 kHz:
//
//   1. Frames: 160 zeros prepended, then frames of 320 samples every 160,
//      frame i at padded sample 160 i. A frame is kept while its last sample
//      — Google takes 321, the 321st for a pre-emphasis this model doesn't
//      use — is the clip's: T = ⌊(n − 161) / 160⌋ + 1 frames, as llama.cpp
//      trims to and Google's mask keeps. A clip of fewer than 161 samples has
//      none, and is refused, TooShort.
//   2. Each frame times a periodic Hann window of 320, w[j] = 0.5 − 0.5
//      cos(2πj / 320) in f32, zero-padded to 512 — the window at the frame's
//      start, as both llama.cpp's and Google's are.
//   3. A real FFT of 512: radix-2, iterative, in f32, its twiddles
//      cos and sin of 2πk / 512 tabulated once; the magnitude of each of the
//      257 bins, √(re² + im²), not the power.
//   4. 128 triangular filters on the HTK mel scale, 2595 log₁₀(1 + f / 700),
//      from 0 to 8,000 Hz, unnormalized, their edges and weights computed in
//      f64 as llama.cpp's fill_mel_filterbank_matrix computes them, held as
//      f32. Each filter is nonzero over a few adjacent bins, so a filter is
//      its first bin and its weights; a bin's sum is taken in f64 over them.
//   5. log(sum + 0.001), natural: Google's form. llama.cpp at the pin takes
//      log(max(sum, 0.001)), which differs by up to ln 2 where a bin's energy
//      is near the floor — a pure tone's spectrum almost everywhere; the
//      embeddings differ by a cosine of 0.9965 for a tone and 0.9998 or more
//      for speech, measured. The reference sets llama.cpp's other form
//      (below).
//   Out: T frames of 128 floats, frame-major, the encoder's subsampling's
//   input layout (kernels/subsample/subsample.h).
//
// The rows a clip makes, which the text step counts before any GPU runs: the
// subsampling's two stride-2 convolutions, R = ⌈⌈T / 2⌉ / 2⌉ — 25 a second.
// A clip's rows and the text's tokens are at most the context: for an audio
// clip alone, R ≤ context − 4, the ends, `<|audio>` and `<audio|>` — 508 at
// a context of 512, so T ≤ 2,032 frames and n ≤ 325,120 samples, 20.32 s.
// A clip over its most rows is refused, TooLong, before a sample is read; the
// page cuts a longer recording into pieces of at most clip_samples(most)
// (web/audio_input.js).
//
// Floating point, unlike the image resampler: the FFT's adds and the
// filters' sums round, and their order is this file's, not llama.cpp's, so a
// spectrogram matches llama.cpp's to a tolerance, not in bytes (below).
//
// What it costs, for a clip of T frames: T FFTs of 512, each 2,304
// butterflies of a complex multiply and two adds; T × 257 magnitudes; the
// filters' nonzero weights, about 2 × 257 a frame, T × 128 logs. At T =
// 2,000, about 5 × 10⁷ operations: a few milliseconds in the module. The
// frames are read from the clip in place, windowed into one 512-float
// buffer reused for every frame; the spectrogram is written once, into the
// slot the caller sized at load. No allocation per clip.
// Optimization (practice): the window, the twiddles, the bit reversal and
// the filters' nonzero spans built once, in the constructor, into the order
// the loops read them (CDSA.32); each frame's FFT then reads tables, never
// calls cos or sin.
// Optimization (practice): a filter's sum runs over its nonzero bins alone,
// about 4 of 257 on average, rather than every bin as llama.cpp's dense
// filter matrix does. The terms skipped are zero; the sum is f64 term by
// term, where llama.cpp adds four f32 products then the f64 total, so the
// two round differently, within the tolerance below.
//
// Verification the implementation is held to:
//   - The frame and row counts of llama.cpp's preprocessor and gemma4a's
//     n_patches, over a table of lengths: 160 samples (none), 161, 320, 321,
//     16,000, 48,000, 325,120 and 325,121 (one row past 508), held exactly
//     (tests/media_audio_test.cpp).
//   - Against the reference: tools/make_reference_audio.sh builds llama.cpp
//     at the embeddings' pin with the gemma4a preprocessor's mel_floor_add
//     set — Google's floor, the one change — and writes, as fixtures, every
//     frame of three half-second clips: a 1 kHz tone, a sweep, and the first
//     half second of the countdown recording (tests/fixtures/audio/). Each
//     bin within 10⁻⁴ of llama.cpp's, absolute. The check discriminates the
//     floor: llama.cpp's own form differs from Google's by more than 0.1 on
//     the tone's quiet bins.
//   - Determinism: the same samples give the same bytes, run to run.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.24   Use a span to designate a half-open sequence — samples in,
//            frames out.
//     E.27   Use error codes systematically — AudioError.
//     SL.io.2 When reading, always consider ill-formed input — a clip's
//            length checked before a sample is read.
//   C++ performance guidelines
//     CDSA.32 Transform static data once into the layout its consumer reads
//            — every table at construction.
//     MEM.9  Allocate at init — one frame buffer, the output the caller's.
//     WASM.2 Batch work across the JS boundary — a clip's samples in one
//            call.

inline constexpr std::uint32_t kSampleRate = 16000;
inline constexpr std::uint32_t kFrameLength = 320;   // 20 ms
inline constexpr std::uint32_t kHop = 160;           // 10 ms
inline constexpr std::uint32_t kFftLength = 512;
inline constexpr std::uint32_t kFftBins = kFftLength / 2 + 1;
inline constexpr std::uint32_t kMelBins = 128;
inline constexpr float kMelFloor = 0.001f;

enum class AudioError {
    Ok,
    TooShort,   // fewer than 161 samples: no frame
    TooLong,    // more rows than the most the caller allows
};

// What a clip of `samples` becomes.
struct AudioClip {
    std::uint32_t samples;
    std::uint32_t frames;   // T
    std::uint32_t rows;     // R, the encoder's rows and the text step's media rows
};

// The frames and rows of a clip of `samples`, or TooShort, or TooLong when
// its rows exceed `most_rows`. Pure.
[[nodiscard]] AudioError audio_clip(std::uint64_t samples, std::uint32_t most_rows, AudioClip& out) noexcept;

// The most samples a clip may hold for at most `most_rows` rows: 4 R frames,
// 160 (4 R − 1) + 161 samples. Pure.
[[nodiscard]] std::uint32_t clip_samples(std::uint32_t most_rows) noexcept;

// The spectrogram's tables, built once, and its one frame buffer: one
// Spectrogram computes one clip at a time.
class Spectrogram {
public:
    Spectrogram();

    // The log-mel spectrogram of `clip`'s samples into `out`, frame-major:
    // clip.frames × kMelBins floats. Preconditions: `samples` holds exactly
    // clip.samples floats and `clip` is what audio_clip gave for them; `out`
    // holds exactly clip.frames × kMelBins floats.
    void compute(std::span<const float> samples, const AudioClip& clip, std::span<float> out) noexcept;

private:
    // A filter's nonzero bins: the first, and its weights in bin order.
    struct Filter {
        std::uint32_t first;
        std::uint32_t count;
        std::uint32_t weight;   // index of its first weight in weights_
    };

    std::array<float, kFrameLength> window_{};
    std::array<float, kFftLength / 2> cos_{};
    std::array<float, kFftLength / 2> sin_{};
    std::array<std::uint16_t, kFftLength> reversed_{};   // bit reversal of 9 bits
    std::array<Filter, kMelBins> filters_{};
    std::vector<float> weights_;
    // One frame, windowed and transformed in place: re, im interleaved.
    std::array<float, 2 * kFftLength> frame_{};
};

}  // namespace bllm::media
