// The subsampling convolutions on Dawn, against f64 from the same inputs
// (kernels/subsample/subsample.h): each form at T of 1, 2, 7 and 2,032 —
// the first and last positions' padding, an odd T, the most frames —
// written where it should be and nowhere past its last row, the input past
// the clip read as padding; a position of equal channels normalized to 0;
// negative outputs cut to 0.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/subsample/subsample.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kMel = 128;
constexpr std::uint32_t kChannels1 = 128;
constexpr std::uint32_t kWidth1 = 64;
constexpr std::uint32_t kChannels2 = 32;
constexpr std::uint32_t kWidth2 = 32;
constexpr std::uint32_t kRowFloats = kWidth2 * kChannels2;   // 1,024
constexpr std::uint32_t kMostFrames = 2032;
constexpr std::uint32_t kMostPositions = (kMostFrames + 1) / 2;
constexpr std::uint32_t kMostRows = (kMostPositions + 1) / 2;
constexpr float kEpsilon = 1e-6f;
// What the outputs hold before a launch, so a write past the last row shows.
constexpr float kSentinel = -7.25f;

std::uint32_t positions_of(std::uint32_t frames) { return (frames + 1) / 2; }
std::uint32_t rows_of(std::uint32_t frames) { return (positions_of(frames) + 1) / 2; }

// Deterministic values in [−amplitude, amplitude).
class Values {
public:
    explicit Values(std::uint64_t seed) : state_(seed) {}
    float next(float amplitude) {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const auto bits = static_cast<std::uint32_t>(state_ >> 40);   // 24 bits
        return (static_cast<float>(bits) / 8388608.0f - 1.0f) * amplitude;
    }
    // A small dyadic value, k / 8 for k in [−8, 8]: sums of their products
    // are exact in f32.
    float dyadic() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<std::int32_t>((state_ >> 40) % 17) - 8) / 8.0f;
    }

private:
    std::uint64_t state_;
};

struct Buffers {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange conv1{}, gain1{}, conv2{}, gain2{}, mel{}, subsampled{}, rows{};

    [[nodiscard]] residency::WeightView view(const residency::BufferRange& r) const {
        gguf::TensorShape shape{};
        shape.dimension_count = 1;
        shape.dimensions[0] = r.length / 4;
        shape.element_count = r.length / 4;
        return residency::WeightView(gguf::TensorType::F32, shape, {{r.buffer, r.offset, r.length, 0, 1}});
    }
};

Buffers upload_buffers(WGPUInstance instance, const gpu::Device& device) {
    Buffers b;
    b.model.bytes = load_gguf_fixture("rope_rows");   // any file: no weight is read
    gguf::MemoryByteSource source{b.model.bytes};
    REQUIRE(gguf::read_index(source, b.model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = b.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    const auto alone = [&](std::uint64_t floats) {
        plan.buffers.push_back({residency::Pool::Scratch, floats * 4});
        return residency::BufferRange{static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, floats * 4};
    };
    b.conv1 = alone(kChannels1 * 9);
    b.gain1 = alone(kChannels1);
    b.conv2 = alone(std::uint64_t{kChannels2} * kChannels1 * 9);
    b.gain2 = alone(kChannels2);
    b.mel = alone(std::uint64_t{kMostFrames} * kMel);
    b.subsampled = alone(std::uint64_t{kMostPositions} * kWidth1 * kChannels1);
    b.rows = alone(std::uint64_t{kMostRows} * kRowFloats);
    Ready ready;
    residency::Upload::begin(device, b.model.index, plan, b.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    b.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *b.upload, b.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *b.upload) == residency::UploadError::Ok);
    return b;
}

void write(const gpu::Device& device, const Buffers& b, const residency::BufferRange& r,
           const std::vector<float>& values) {
    REQUIRE(values.size() * 4 == r.length);
    wgpuQueueWriteBuffer(device.queue(), b.upload->buffer(r.buffer), r.offset, values.data(), r.length);
}

std::vector<float> read(WGPUInstance instance, const gpu::Device& device, const Buffers& b,
                        const residency::BufferRange& r) {
    return read_floats(instance, device, b.upload->buffer(r.buffer), r.offset, r.length / 4);
}

double gamma(std::uint64_t n) {
    const double nu = static_cast<double>(n) * std::ldexp(1.0, -24);
    return nu / (1 - nu);
}

// One clip's step: R rows, T in `columns`.
void run(WGPUInstance instance, const Buffers& b, kernels::Launch launch, std::uint32_t frames) {
    kernels::Step step{};
    step.tokens = rows_of(frames);
    step.columns = frames;
    step.logits = 1;
    const auto program = build_program(instance, *b.upload, {std::move(launch)});
    const StepOutcome ran = try_step(instance, *program, step);
    REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
}

// A position's convolution sums in f64, each with its bound γ(n) Σ|w x|,
// and its layer norm and ReLU in f64; checks the kernel's `got` against
// them. The norm's bound: a sum's error E carries through x − μ and σ, so
// an output may move by (|g| + |y|) × 4E / σ, beside a few units of f32
// rounding of the norm's own steps, inverseSqrt's 2 ULP among them.
struct Position {
    std::vector<double> sums, bounds;
};

void check_position(const Position& p, const std::vector<float>& gain, const float* got, std::size_t& negatives,
                    double& worst) {
    const std::size_t n = p.sums.size();
    double mean = 0;
    for (const double x : p.sums) mean += x;
    mean /= static_cast<double>(n);
    double variance = 0;
    double largest = 0;
    double error = 0;
    for (std::size_t c = 0; c < n; ++c) {
        variance += (p.sums[c] - mean) * (p.sums[c] - mean);
        largest = std::max(largest, std::abs(p.sums[c]));
        error = std::max(error, p.bounds[c]);
    }
    variance /= static_cast<double>(n);
    // The statistics' own f32 sums, n terms each.
    error += gamma(n + 2) * largest;
    const double sigma = std::sqrt(variance + kEpsilon);
    for (std::size_t c = 0; c < n; ++c) {
        const double y = (p.sums[c] - mean) / sigma * gain[c];
        const double want = std::max(y, 0.0);
        const double bound =
            (std::abs(gain[c]) + std::abs(y)) * (4 * error / sigma + std::ldexp(1.0, -19));
        CHECK(std::abs(got[c] - want) <= bound);
        worst = std::max(worst, std::abs(got[c] - want) / bound);
        if (y < -bound) {
            ++negatives;
            CHECK(got[c] == 0.0f);
        }
    }
}

Position first_position(const std::vector<float>& conv, const std::vector<float>& mel, std::uint32_t frames,
                        std::uint32_t t, std::uint32_t f) {
    Position p{std::vector<double>(kChannels1), std::vector<double>(kChannels1)};
    for (std::uint32_t c = 0; c < kChannels1; ++c) {
        double sum = 0, absolute = 0;
        for (std::uint32_t kh = 0; kh < 3; ++kh) {
            for (std::uint32_t kw = 0; kw < 3; ++kw) {
                const std::int64_t time = std::int64_t{2} * t + kh - 1;
                const std::int64_t bin = std::int64_t{2} * f + kw - 1;
                if (time < 0 || time >= frames || bin < 0 || bin >= kMel) continue;
                const double term = double{conv[c * 9 + kh * 3 + kw]} * mel[time * kMel + bin];
                sum += term;
                absolute += std::abs(term);
            }
        }
        p.sums[c] = sum;
        p.bounds[c] = gamma(9) * absolute;
    }
    return p;
}

Position second_position(const std::vector<float>& conv, const std::vector<float>& sub, std::uint32_t frames,
                         std::uint32_t r, std::uint32_t f) {
    const std::uint32_t positions = positions_of(frames);
    Position p{std::vector<double>(kChannels2), std::vector<double>(kChannels2)};
    for (std::uint32_t c = 0; c < kChannels2; ++c) {
        double sum = 0, absolute = 0;
        for (std::uint32_t kh = 0; kh < 3; ++kh) {
            const std::int64_t time = std::int64_t{2} * r + kh - 1;
            if (time < 0 || time >= positions) continue;
            for (std::uint32_t kw = 0; kw < 3; ++kw) {
                const std::int64_t frequency = std::int64_t{2} * f + kw - 1;
                if (frequency < 0 || frequency >= kWidth1) continue;
                const std::size_t from = (static_cast<std::size_t>(time) * kWidth1 + frequency) * kChannels1;
                for (std::uint32_t i = 0; i < kChannels1; ++i) {
                    const double term = double{conv[((std::size_t{c} * kChannels1 + i) * 3 + kh) * 3 + kw]} *
                                        sub[from + i];
                    sum += term;
                    absolute += std::abs(term);
                }
            }
        }
        p.sums[c] = sum;
        p.bounds[c] = gamma(kChannels1 * 9) * absolute;
    }
    return p;
}

std::vector<float> random(std::size_t count, std::uint64_t seed, float amplitude) {
    Values v{seed};
    std::vector<float> out(count);
    for (float& x : out) x = v.next(amplitude);
    return out;
}

}  // namespace

TEST_CASE("the first convolution matches f64 at T of 1, 2, 7 and 2,032, and writes no position past the clip's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    const auto conv = random(kChannels1 * 9, 1, 0.5f);
    const auto gain = random(kChannels1, 2, 1.5f);
    // The whole buffer: frames past the clip hold values, which must read as
    // padding.
    const auto mel = random(std::size_t{kMostFrames} * kMel, 3, 8.0f);
    write(*device, b, b.conv1, conv);
    write(*device, b, b.gain1, gain);
    write(*device, b, b.mel, mel);
    for (const std::uint32_t frames : {1u, 2u, 7u, kMostFrames}) {
        CAPTURE(frames);
        write(*device, b, b.subsampled,
              std::vector<float>(std::size_t{kMostPositions} * kWidth1 * kChannels1, kSentinel));
        run(instance.get(), b, kernels::subsample_first_launch(b.view(b.conv1), b.view(b.gain1), b.mel, b.subsampled,
                                                                kEpsilon),
            frames);
        const auto got = read(instance.get(), *device, b, b.subsampled);
        const std::uint32_t positions = positions_of(frames);
        std::size_t negatives = 0;
        double worst = 0;
        for (std::uint32_t t = 0; t < positions; ++t) {
            for (std::uint32_t f = 0; f < kWidth1; ++f) {
                check_position(first_position(conv, mel, frames, t, f), gain,
                               &got[(std::size_t{t} * kWidth1 + f) * kChannels1], negatives, worst);
            }
        }
        CHECK(negatives > 0);
        MESSAGE("worst error over its bound: " << worst);
        const auto past = std::count_if(got.begin() + std::size_t{positions} * kWidth1 * kChannels1, got.end(),
                                        [](float x) { return x != kSentinel; });
        CHECK(past == 0);
    }
}

TEST_CASE("the second convolution matches f64 at T of 1, 2, 7 and 2,032, and writes no row past the clip's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    const auto conv = random(std::size_t{kChannels2} * kChannels1 * 9, 4, 0.1f);
    const auto gain = random(kChannels2, 5, 1.5f);
    // Rectified, as the first leaves them; positions past the clip's hold
    // values, which must read as padding.
    auto sub = random(std::size_t{kMostPositions} * kWidth1 * kChannels1, 6, 2.0f);
    for (float& x : sub) x = std::max(x, 0.0f);
    write(*device, b, b.conv2, conv);
    write(*device, b, b.gain2, gain);
    write(*device, b, b.subsampled, sub);
    for (const std::uint32_t frames : {1u, 2u, 7u, kMostFrames}) {
        CAPTURE(frames);
        write(*device, b, b.rows, std::vector<float>(std::size_t{kMostRows} * kRowFloats, kSentinel));
        run(instance.get(), b,
            kernels::subsample_second_launch(b.view(b.conv2), b.view(b.gain2), b.subsampled, b.rows, kEpsilon),
            frames);
        const auto got = read(instance.get(), *device, b, b.rows);
        const std::uint32_t rows = rows_of(frames);
        std::size_t negatives = 0;
        double worst = 0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t f = 0; f < kWidth2; ++f) {
                check_position(second_position(conv, sub, frames, r, f), gain,
                               &got[std::size_t{r} * kRowFloats + std::size_t{f} * kChannels2], negatives, worst);
            }
        }
        CHECK(negatives > 0);
        MESSAGE("worst error over its bound: " << worst);
        const auto past = std::count_if(got.begin() + std::size_t{rows} * kRowFloats, got.end(),
                                        [](float x) { return x != kSentinel; });
        CHECK(past == 0);
    }
}

TEST_CASE("a position whose channels are equal is normalized to 0, in both convolutions") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    // Every output channel the same weights, every value a small dyadic, so
    // each channel's sum is the same exact value and so is their mean.
    Values v{7};
    std::vector<float> conv1(kChannels1 * 9);
    for (std::uint32_t k = 0; k < 9; ++k) {
        const float w = v.dyadic();
        for (std::uint32_t c = 0; c < kChannels1; ++c) conv1[c * 9 + k] = w;
    }
    std::vector<float> conv2(std::size_t{kChannels2} * kChannels1 * 9);
    for (std::uint32_t k = 0; k < kChannels1 * 9; ++k) {
        const float w = v.dyadic();
        for (std::uint32_t c = 0; c < kChannels2; ++c) conv2[std::size_t{c} * kChannels1 * 9 + k] = w;
    }
    std::vector<float> mel(std::size_t{kMostFrames} * kMel);
    for (float& x : mel) x = v.dyadic() * 4;
    std::vector<float> sub(std::size_t{kMostPositions} * kWidth1 * kChannels1);
    for (float& x : sub) x = v.dyadic() * 4;
    write(*device, b, b.conv1, conv1);
    write(*device, b, b.gain1, random(kChannels1, 8, 1.5f));
    write(*device, b, b.conv2, conv2);
    write(*device, b, b.gain2, random(kChannels2, 9, 1.5f));
    write(*device, b, b.mel, mel);
    constexpr std::uint32_t kFrames = 7;
    run(instance.get(), b,
        kernels::subsample_first_launch(b.view(b.conv1), b.view(b.gain1), b.mel, b.subsampled, kEpsilon), kFrames);
    const auto first = read(instance.get(), *device, b, b.subsampled);
    CHECK(std::all_of(first.begin(), first.begin() + std::size_t{positions_of(kFrames)} * kWidth1 * kChannels1,
                      [](float x) { return x == 0.0f; }));
    write(*device, b, b.subsampled, sub);
    run(instance.get(), b,
        kernels::subsample_second_launch(b.view(b.conv2), b.view(b.gain2), b.subsampled, b.rows, kEpsilon),
        kFrames);
    const auto second = read(instance.get(), *device, b, b.rows);
    CHECK(std::all_of(second.begin(), second.begin() + std::size_t{rows_of(kFrames)} * kRowFloats,
                      [](float x) { return x == 0.0f; }));
}
