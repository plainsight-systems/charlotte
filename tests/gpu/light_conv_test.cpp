// The light convolution on Dawn, against f64 from the same inputs
// (kernels/light_conv/light_conv.h): at R of 1, 4, 5 and 508 — rows with
// fewer than 4 predecessors, the first with all, the most — rows past R
// never written; a row whose taps all read zeros written as zeros.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/light_conv/light_conv.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kWidth = 1024;
constexpr std::uint32_t kTaps = 5;
constexpr std::uint32_t kMostRows = 508;
constexpr float kEpsilon = 1e-6f;
constexpr float kSentinel = -7.25f;

struct Buffers {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange depthwise{}, gain{}, input{}, output{};

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
    b.depthwise = alone(kTaps * kWidth);
    b.gain = alone(kWidth);
    b.input = alone(std::uint64_t{kMostRows} * kWidth);
    b.output = alone(std::uint64_t{kMostRows} * kWidth);
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

std::vector<float> random(std::size_t count, std::uint64_t seed, float amplitude) {
    std::vector<float> out(count);
    std::uint64_t state = seed;
    for (float& x : out) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        x = (static_cast<float>(static_cast<std::uint32_t>(state >> 40)) / 8388608.0f - 1.0f) * amplitude;
    }
    return out;
}

double gamma(std::uint64_t n) {
    const double nu = static_cast<double>(n) * std::ldexp(1.0, -24);
    return nu / (1 - nu);
}

// Runs the launch over `rows` and checks every row against f64. The bound:
// y's taps within γ(K) Σ|w g|; ρ's relative error from the squares'
// (2 Σ|y| δy + γ(W) Σ y²) over twice the row's Σ y² + W ε, beside
// inverseSqrt's 2 ULP; z from both and its two roundings; SiLU at most 1.1
// times z's error, and exp's 3 + 2|z| ULP of its output.
std::vector<float> check(WGPUInstance instance, const gpu::Device& device, const Buffers& b,
                         const std::vector<float>& depthwise, const std::vector<float>& gain,
                         const std::vector<float>& input, std::uint32_t rows) {
    CAPTURE(rows);
    write(device, b, b.output, std::vector<float>(std::size_t{kMostRows} * kWidth, kSentinel));
    const auto program = build_program(
        instance, *b.upload,
        {kernels::light_conv_launch(b.view(b.depthwise), b.view(b.gain), b.input, b.output, kWidth, kTaps, kEpsilon)});
    run_step(instance, *program, rows);
    const auto got =
        read_floats(instance, device, b.upload->buffer(b.output.buffer), 0, std::size_t{kMostRows} * kWidth);
    const double u = std::ldexp(1.0, -24);
    double worst = 0;
    std::size_t outside = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<double> y(kWidth), dy(kWidth);
        double squares = 0, carried = 0;
        for (std::uint32_t c = 0; c < kWidth; ++c) {
            double sum = 0, absolute = 0;
            for (std::uint32_t i = 0; i < kTaps; ++i) {
                if (r + i + 1 < kTaps) continue;
                const std::size_t row = r + i + 1 - kTaps;
                const double term = double{depthwise[c * kTaps + i]} * input[row * kWidth + c];
                sum += term;
                absolute += std::abs(term);
            }
            y[c] = sum;
            dy[c] = gamma(kTaps) * absolute;
            squares += sum * sum;
            carried += 2 * std::abs(sum) * dy[c];
        }
        const double mean = squares + kWidth * double{kEpsilon};
        const double rho = 1 / std::sqrt(squares / kWidth + kEpsilon);
        const double drho = (carried + gamma(kWidth + 2) * squares) / (2 * mean) + 4 * u;
        for (std::uint32_t c = 0; c < kWidth; ++c) {
            const double z = y[c] * rho * gain[c];
            const double want = z / (1 + std::exp(-z));
            const double dz = std::abs(gain[c]) * rho * (dy[c] + std::abs(y[c]) * drho) + 3 * u * std::abs(z);
            const double bound = 1.1 * dz + std::abs(want) * ((3 + 2 * std::abs(z)) * std::ldexp(1.0, -23) + 4 * u);
            const double error = std::abs(got[std::size_t{r} * kWidth + c] - want);
            if (!(error <= bound)) ++outside;
            if (bound > 0) worst = std::max(worst, error / bound);
        }
    }
    CHECK(outside == 0);
    MESSAGE("worst error over its bound: " << worst);
    CHECK(std::all_of(got.begin() + std::size_t{rows} * kWidth, got.end(), [](float x) { return x == kSentinel; }));
    return got;
}

}  // namespace

TEST_CASE("the light convolution matches f64 at R of 1, 4, 5 and 508, and writes no row past R") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    const auto depthwise = random(kTaps * kWidth, 41, 1.0f);
    const auto gain = random(kWidth, 42, 2.0f);
    const auto input = random(std::size_t{kMostRows} * kWidth, 43, 3.0f);
    write(*device, b, b.depthwise, depthwise);
    write(*device, b, b.gain, gain);
    write(*device, b, b.input, input);
    for (const std::uint32_t rows : {1u, 4u, 5u, kMostRows}) {
        check(instance.get(), *device, b, depthwise, gain, input, rows);
    }
}

TEST_CASE("a light convolution row whose taps all read zeros is written as zeros") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    const auto depthwise = random(kTaps * kWidth, 51, 1.0f);
    const auto gain = random(kWidth, 52, 2.0f);
    auto input = random(std::size_t{kMostRows} * kWidth, 53, 3.0f);
    // Rows 20 to 24 zero: row 24's five taps read only them.
    std::fill(input.begin() + std::size_t{20} * kWidth, input.begin() + std::size_t{25} * kWidth, 0.0f);
    write(*device, b, b.depthwise, depthwise);
    write(*device, b, b.gain, gain);
    write(*device, b, b.input, input);
    const auto got = check(instance.get(), *device, b, depthwise, gain, input, 30);
    CHECK(std::all_of(got.begin() + std::size_t{24} * kWidth, got.begin() + std::size_t{25} * kWidth,
                      [](float x) { return x == 0.0f; }));
}
