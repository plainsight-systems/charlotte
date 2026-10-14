// Pooling on Dawn, against f64 (kernels/pool/pool.h): the mean of a step's
// rows written at row N − 1 and nowhere else, at N of 1, 2, 33 and 512; and
// a row made unit length, large and tiny rows among them, a row of zeros
// written as zeros.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/pool/pool.h"
#include "support/acquire.h"
#include "support/fill.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kWidth = 512;
constexpr std::uint32_t kDimensions = 768;
constexpr std::uint32_t kMaxTokens = 512;

struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange normed{}, pooled{}, projected{}, embedding{};
};

Uploaded upload_buffers(WGPUInstance instance, const gpu::Device& device) {
    Uploaded u;
    u.model.bytes = load_gguf_fixture("rope_rows");   // any file: no weight is read
    gguf::MemoryByteSource source{u.model.bytes};
    REQUIRE(gguf::read_index(source, u.model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = u.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    const auto alone = [&](std::uint64_t bytes) {
        plan.buffers.push_back({residency::Pool::Scratch, bytes});
        return residency::BufferRange{static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, bytes};
    };
    u.normed = alone(std::uint64_t{kMaxTokens} * kWidth * 4);
    u.pooled = alone(std::uint64_t{kMaxTokens} * kWidth * 4);
    u.projected = alone(kDimensions * 4);
    u.embedding = alone(kDimensions * 4);
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == residency::UploadError::Ok);
    return u;
}

double rounding_gamma(std::uint64_t n) {
    const double nu = static_cast<double>(n) * std::ldexp(1.0, -24);
    return nu / (1 - nu);
}

}  // namespace

TEST_CASE("the mean of a step's rows is written at row N - 1 of pooled, and nowhere else") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    run_step(instance.get(), *build_program(instance.get(), *u.upload, {fill_launch(u.normed, 5, 3.0f, false)}), 1);
    const auto normed = read_floats(instance.get(), *device, u.upload->buffer(u.normed.buffer), 0,
                                    std::size_t{kMaxTokens} * kWidth);
    for (const std::uint32_t tokens : {1u, 2u, 33u, kMaxTokens}) {
        CAPTURE(tokens);
        // pooled cleared first, in a step of its own, so a write anywhere
        // but row N - 1 shows.
        run_step(instance.get(), *build_program(instance.get(), *u.upload, {fill_launch(u.pooled, 0, 0.0f, false)}), 1);
        run_step(instance.get(), *build_program(instance.get(), *u.upload, {kernels::mean_launch(u.normed, u.pooled, kWidth)}),
                 tokens);
        const auto pooled = read_floats(instance.get(), *device, u.upload->buffer(u.pooled.buffer), 0,
                                        std::size_t{kMaxTokens} * kWidth);
        for (std::uint32_t c = 0; c < kWidth; ++c) {
            double sum = 0, absolute = 0;
            for (std::uint32_t row = 0; row < tokens; ++row) {
                sum += normed[std::size_t{row} * kWidth + c];
                absolute += std::abs(normed[std::size_t{row} * kWidth + c]);
            }
            const double want = sum / tokens;
            const double bound = rounding_gamma(tokens + 1) * absolute / tokens + std::ldexp(std::abs(want), -23);
            CHECK(std::abs(pooled[std::size_t{tokens - 1} * kWidth + c] - want) <= bound);
        }
        std::size_t elsewhere = 0;
        for (std::size_t i = 0; i < pooled.size(); ++i) {
            if (i / kWidth != tokens - 1 && pooled[i] != 0.0f) ++elsewhere;
        }
        CHECK(elsewhere == 0);
    }
}

TEST_CASE("a row made unit length is its components over its length, and a row of zeros stays zeros") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    for (const float amplitude : {1.0f, 1e4f, 1e-4f, 0.0f}) {
        CAPTURE(amplitude);
        std::vector<kernels::Launch> launches;
        launches.push_back(fill_launch(u.projected, 17, amplitude, false));
        launches.push_back(kernels::unit_launch(u.projected, u.embedding, kDimensions));
        run_step(instance.get(), *build_program(instance.get(), *u.upload, std::move(launches)), 1);
        const auto x = read_floats(instance.get(), *device, u.upload->buffer(u.projected.buffer), 0, kDimensions);
        const auto got = read_floats(instance.get(), *device, u.upload->buffer(u.embedding.buffer), 0, kDimensions);
        double squares = 0;
        for (const float e : x) squares += double{e} * e;
        if (squares == 0) {
            CHECK(std::all_of(got.begin(), got.end(), [](float f) { return f == 0.0f; }));
            continue;
        }
        const double length = std::sqrt(squares);
        double got_squares = 0;
        for (std::uint32_t i = 0; i < kDimensions; ++i) {
            // The norm's margin against the row's largest, which is at most 1.
            CHECK(std::abs(got[i] - x[i] / length) <= std::ldexp(1.0, -18));
            got_squares += double{got[i]} * got[i];
        }
        CHECK(std::abs(got_squares - 1.0) <= std::ldexp(1.0, -16));
    }
}

TEST_CASE("pooling's launches: one workgroup a step, over the last token, each its width in its constants") {
    const residency::BufferRange r{residency::BufferIndex{1}, 0, 64};
    for (const kernels::Launch& l : {kernels::mean_launch(r, r, 512), kernels::unit_launch(r, r, 768)}) {
        CHECK(l.rows == kernels::Rows::LastToken);
        CHECK(l.invocations_per_row == l.workgroup_size);
        CHECK(l.bindings.size() == 2);
        REQUIRE(l.constants.size() == 12);   // vec4s, then grid's k and ε, unread here
    }
    CHECK(kernels::mean_launch(r, r, 512).entry_point == "mean");
    CHECK(kernels::unit_launch(r, r, 768).entry_point == "unit");
}

TEST_CASE("a vision step's patches pool k x k into soft tokens, each normalized without a gain") {
    // A grid of C × R patches of W = 512 (the buffers' width), pooled 3 × 3:
    // each soft token the mean of its nine patches, then RMS-normalized at
    // ε / W, against f64 from the same patches.
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    constexpr std::uint32_t kPooling = 3;
    const float epsilon = 1e-6f / kWidth;
    for (const auto [columns, rows] : {std::pair{3u, 3u}, std::pair{6u, 9u}, std::pair{21u, 24u}}) {
        CAPTURE(columns);
        CAPTURE(rows);
        const std::uint32_t patches = columns * rows;
        REQUIRE(patches <= kMaxTokens);
        run_step(instance.get(), *build_program(instance.get(), *u.upload, {fill_launch(u.normed, 11, 2.0f, false)}),
                 1);
        const auto in = read_floats(instance.get(), *device, u.upload->buffer(u.normed.buffer), 0,
                                    std::size_t{patches} * kWidth);
        const auto program = build_program(instance.get(), *u.upload,
                                            {kernels::grid_launch(u.normed, u.pooled, kWidth, kPooling, epsilon)});
        kernels::Step step{};
        step.tokens = patches;
        step.columns = columns;
        const StepOutcome ran = try_step(instance.get(), *program, step);
        REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
        const std::uint32_t pooled_columns = columns / kPooling;
        const std::uint32_t tokens = patches / (kPooling * kPooling);
        const auto got = read_floats(instance.get(), *device, u.upload->buffer(u.pooled.buffer), 0,
                                     std::size_t{tokens} * kWidth);
        for (std::uint32_t r = 0; r < tokens; ++r) {
            const std::uint32_t pu = r % pooled_columns, pv = r / pooled_columns;
            std::vector<double> mean(kWidth, 0.0);
            for (std::uint32_t dy = 0; dy < kPooling; ++dy) {
                for (std::uint32_t dx = 0; dx < kPooling; ++dx) {
                    const std::uint32_t patch = (kPooling * pv + dy) * columns + kPooling * pu + dx;
                    for (std::uint32_t i = 0; i < kWidth; ++i) mean[i] += in[std::size_t{patch} * kWidth + i];
                }
            }
            double squares = 0, largest = 0;
            for (double& m : mean) {
                m /= kPooling * kPooling;
                squares += m * m;
            }
            const double scale = 1.0 / std::sqrt(squares / kWidth + epsilon);
            for (const double m : mean) largest = std::max(largest, std::abs(m * scale));
            for (std::uint32_t i = 0; i < kWidth; ++i) {
                // The norm's margin against the row's largest, and the mean's
                // nine adds and division.
                const double error = std::abs(got[std::size_t{r} * kWidth + i] - mean[i] * scale);
                if (!(error <= largest * std::ldexp(1.0, -17) + std::abs(mean[i] * scale) * rounding_gamma(10))) {
                    FAIL_CHECK("soft token " << r << " value " << i << " off by " << error);
                    break;
                }
            }
        }
        // Rows past the soft tokens' are not written.
        const auto past = read_floats(instance.get(), *device, u.upload->buffer(u.pooled.buffer),
                                      std::size_t{tokens} * kWidth * 4, kWidth);
        CHECK(std::all_of(past.begin(), past.end(), [](float f) { return f == 0.0f; }));
    }
}
