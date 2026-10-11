// Local attention on Dawn (kernels/local_attention/local_attention.h),
// against an f64 reference that computes llama.cpp's gemma4a blocked form
// literally: rows in chunks of 12, a context of 24 keys padded on the left,
// the relative scores padded, flattened and shifted (Transformer-XL's
// appendix B), the cap, the blocked mask, the softmax, the values, the
// padding trimmed. At R of 1, 11, 12, 13 and 508; with scores far past the
// cap; with equal scores. Rows past R are never written.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/local_attention/local_attention.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kHeads = 8;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kWidth = kHeads * kHeadDim;
constexpr std::uint32_t kMostRows = 508;
// llama.cpp's: chunk C, past horizon P, context S, relative positions.
constexpr std::int64_t kBlockRows = 12;
constexpr std::int64_t kPast = 12;
constexpr std::int64_t kContext = kBlockRows + kPast;
constexpr std::int64_t kPositions = kPast + 1;
constexpr std::uint32_t kWindow = 12;
const float kQScale = (1.0f / std::sqrt(128.0f)) / std::log(2.0f);
const float kKScale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
constexpr float kSoftcap = 50.0f;
constexpr float kSentinel = -7.25f;

struct Buffers {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange scale{}, relative{}, query{}, key{}, value{}, attention{};

    [[nodiscard]] residency::WeightView scale_view() const {
        gguf::TensorShape shape{};
        shape.dimension_count = 1;
        shape.dimensions[0] = kHeadDim;
        shape.element_count = kHeadDim;
        return residency::WeightView(gguf::TensorType::F32, shape,
                                     {{scale.buffer, scale.offset, scale.length, 0, 1}});
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
    b.scale = alone(kHeadDim);
    b.relative = alone(std::uint64_t{kPositions} * kWidth);
    b.query = alone(std::uint64_t{kMostRows} * kWidth);
    b.key = alone(std::uint64_t{kMostRows} * kWidth);
    b.value = alone(std::uint64_t{kMostRows} * kWidth);
    b.attention = alone(std::uint64_t{kMostRows} * kWidth);
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

struct Inputs {
    std::vector<float> scale, relative, query, key, value;
};

// llama.cpp's blocked form in f64, and for each output the bound its f32
// computation is held to. Scores: each of the two dot products within
// (γ(128) + 3u) of its terms' absolute sum, the scalings' roundings among
// them; through the cap at most its slope over that interval, beside tanh's
// own error, taken as 2⁻¹⁹ c. Weights: twice the largest score error, and
// the exponentials' 3 + 2|x| ULP, |x| ≤ 2c, twice. The output within those
// times Σ p |v|, and γ(14) more for its sum.
struct Reference {
    std::vector<double> out, bound;
};

Reference blocked(const Inputs& in, std::uint32_t n) {
    const std::int64_t blocks = (n + kBlockRows - 1) / kBlockRows;
    const double u = std::ldexp(1.0, -24);
    const auto q_hat = [&](std::int64_t t, std::uint32_t h, std::uint32_t i) -> double {
        if (t >= n) return 0.0;   // ggml_pad: the padded queries are zeros
        return double{in.query[t * kWidth + h * kHeadDim + i]} * kQScale * in.scale[i];
    };
    const auto k_hat = [&](std::int64_t t, std::uint32_t h, std::uint32_t i) -> double {
        if (t < 0 || t >= n) return 0.0;   // ggml_pad_ext: zeros either side
        return double{in.key[t * kWidth + h * kHeadDim + i]} * kKScale;
    };
    const auto v_at = [&](std::int64_t t, std::uint32_t h, std::uint32_t i) -> double {
        if (t < 0 || t >= n) return 0.0;
        return in.value[t * kWidth + h * kHeadDim + i];
    };
    Reference ref{std::vector<double>(std::size_t{n} * kWidth), std::vector<double>(std::size_t{n} * kWidth)};
    for (std::int64_t b = 0; b < blocks; ++b) {
        for (std::uint32_t h = 0; h < kHeads; ++h) {
            // matrix_bd before its shift: [R, C], and its terms' absolute sums.
            std::vector<double> bd(kBlockRows * kPositions), bd_abs(kBlockRows * kPositions);
            for (std::int64_t q = 0; q < kBlockRows; ++q) {
                for (std::int64_t p = 0; p < kPositions; ++p) {
                    double sum = 0, absolute = 0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        const double term = q_hat(b * kBlockRows + q, h, i) * in.relative[p * kWidth + h * kHeadDim + i];
                        sum += term;
                        absolute += std::abs(term);
                    }
                    bd[q * kPositions + p] = sum;
                    bd_abs[q * kPositions + p] = absolute;
                }
            }
            // The shift: pad each query's row to S + 1 with zeros, flatten the
            // C rows, keep the first C × S, read back as C rows of S.
            const auto shifted = [&](const std::vector<double>& m, std::int64_t q, std::int64_t k) {
                const std::int64_t flat = q * kContext + k;
                const std::int64_t row = flat / (kContext + 1);
                const std::int64_t column = flat % (kContext + 1);
                return column < kPositions ? m[row * kPositions + column] : 0.0;
            };
            for (std::int64_t q = 0; q < kBlockRows; ++q) {
                const std::int64_t gq = b * kBlockRows + q;
                if (gq >= n) continue;   // trimmed
                std::vector<double> score(kContext), slope_error(kContext);
                std::vector<bool> kept(kContext);
                for (std::int64_t k = 0; k < kContext; ++k) {
                    const std::int64_t gk = b * kBlockRows - kPast + k;
                    double ac = 0, ac_abs = 0;
                    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                        const double term = q_hat(gq, h, i) * k_hat(gk, h, i);
                        ac += term;
                        ac_abs += std::abs(term);
                    }
                    const double x = (ac + shifted(bd, q, k)) / kSoftcap;
                    const double dx = (gamma(kHeadDim) + 3 * u) * (ac_abs + shifted(bd_abs, q, k)) / kSoftcap;
                    const double t = std::tanh(std::max(0.0, std::abs(x) - dx));
                    score[k] = std::tanh(x) * kSoftcap;
                    slope_error[k] = kSoftcap * (1 - t * t) * dx + std::ldexp(kSoftcap, -19);
                    // The mask, as clip.cpp fills it.
                    kept[k] = gk >= 0 && gk < n && gk <= gq && gq - gk < kPast;
                }
                double largest = -std::numeric_limits<double>::infinity();
                double score_error = 0;
                for (std::int64_t k = 0; k < kContext; ++k) {
                    if (!kept[k]) continue;
                    largest = std::max(largest, score[k]);
                    score_error = std::max(score_error, slope_error[k]);
                }
                std::vector<double> p(kContext);
                double total = 0;
                for (std::int64_t k = 0; k < kContext; ++k) {
                    p[k] = kept[k] ? std::exp(score[k] - largest) : 0.0;
                    total += p[k];
                }
                for (double& e : p) e /= total;
                const double weight_error = 2 * score_error + 2 * (3 + 4 * kSoftcap) * std::ldexp(1.0, -23) + 4 * u;
                for (std::uint32_t i = 0; i < kHeadDim; ++i) {
                    double sum = 0, absolute = 0;
                    for (std::int64_t k = 0; k < kContext; ++k) {
                        const double v = v_at(b * kBlockRows - kPast + k, h, i);
                        sum += p[k] * v;
                        absolute += p[k] * std::abs(v);
                    }
                    const std::size_t at = static_cast<std::size_t>(gq) * kWidth + h * kHeadDim + i;
                    ref.out[at] = sum;
                    ref.bound[at] = absolute * (weight_error + gamma(kWindow + 2));
                }
            }
        }
    }
    return ref;
}

void check(WGPUInstance instance, const gpu::Device& device, const Buffers& b, const Inputs& in,
           std::uint32_t rows) {
    CAPTURE(rows);
    write(device, b, b.scale, in.scale);
    write(device, b, b.relative, in.relative);
    write(device, b, b.query, in.query);
    write(device, b, b.key, in.key);
    write(device, b, b.value, in.value);
    write(device, b, b.attention, std::vector<float>(std::size_t{kMostRows} * kWidth, kSentinel));
    const residency::WeightView scale = b.scale_view();
    const auto program = build_program(
        instance, *b.upload,
        {kernels::local_attention_launch(
            {&scale, b.relative, b.query, b.key, b.value, b.attention, kQScale, kKScale, kSoftcap, kWindow})});
    run_step(instance, *program, rows);
    const auto got =
        read_floats(instance, device, b.upload->buffer(b.attention.buffer), 0, std::size_t{kMostRows} * kWidth);
    const Reference ref = blocked(in, rows);
    double worst = 0;
    std::size_t outside = 0;
    for (std::size_t i = 0; i < std::size_t{rows} * kWidth; ++i) {
        const double error = std::abs(got[i] - ref.out[i]);
        if (!(error <= ref.bound[i])) ++outside;
        worst = std::max(worst, error / ref.bound[i]);
    }
    CHECK(outside == 0);
    MESSAGE("worst error over its bound: " << worst);
    CHECK(std::all_of(got.begin() + std::size_t{rows} * kWidth, got.end(), [](float x) { return x == kSentinel; }));
}

Inputs inputs(float query_amplitude, std::uint64_t seed) {
    return {random(kHeadDim, seed, 1.0f), random(std::size_t{kPositions} * kWidth, seed + 1, 1.0f),
            random(std::size_t{kMostRows} * kWidth, seed + 2, query_amplitude),
            random(std::size_t{kMostRows} * kWidth, seed + 3, 1.0f),
            random(std::size_t{kMostRows} * kWidth, seed + 4, 1.0f)};
}

}  // namespace

TEST_CASE("local attention is llama.cpp's blocked form at R of 1, 11, 12, 13 and 508, and writes no row past R") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    const Inputs in = inputs(1.0f, 11);
    for (const std::uint32_t rows : {1u, 11u, 12u, 13u, kMostRows}) {
        check(instance.get(), *device, b, in, rows);
    }
}

TEST_CASE("local attention's scores far past the cap stay finite and capped") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    // Queries 5,000 times as large: (a + b) / c reaches past 89, where tanh
    // as sinh / cosh is NaN.
    check(instance.get(), *device, b, inputs(5000.0f, 21), 40);
}

TEST_CASE("local attention's equal scores weigh a row's window evenly") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers b = upload_buffers(instance.get(), *device);
    // Zero queries: every score 0, so each weight is 1 over the row's window.
    check(instance.get(), *device, b, inputs(0.0f, 31), 40);
}
