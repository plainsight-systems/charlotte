// Block attention on Dawn, against the same computation in f64 from the same
// queries, keys and values (kernels/block_attention/block_attention.h):
// EmbeddingGemma 2's two layer shapes at steps either side of a 32-key and a
// 64-key boundary, a window narrower than the step, every output within the
// stated margin, and the same step run twice giving the same bits.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/block_attention/block_attention.h"
#include "support/acquire.h"
#include "support/fill.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kMaxTokens = 512;

struct Shape {
    const char* name;
    std::uint32_t query_heads, key_value_heads, d;
};

constexpr Shape kSliding{"sliding", 4, 2, 256};
constexpr Shape kGlobal{"global", 4, 1, 512};

struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange query{}, key{}, value{}, scores{}, maxima{}, attended{};
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
    const std::uint64_t rows = std::uint64_t{kMaxTokens} * 2048 * 4;   // the widest query row, 4 × 512
    u.query = alone(rows);
    u.key = alone(std::uint64_t{kMaxTokens} * 512 * 4);
    u.value = alone(std::uint64_t{kMaxTokens} * 512 * 4);
    u.scores = alone(std::uint64_t{4} * kMaxTokens * kMaxTokens * 4);
    u.maxima = alone(std::uint64_t{4} * kMaxTokens * (kMaxTokens / 64) * 4);
    u.attended = alone(rows);
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == residency::UploadError::Ok);
    // Queries, keys and values, filled once on the device.
    const auto program = build_program(instance, *u.upload,
                                       {fill_launch(u.query, 11, 1.0f, false), fill_launch(u.key, 23, 1.0f, false),
                                        fill_launch(u.value, 37, 1.0f, false)});
    run_step(instance, *program, 1);
    return u;
}

std::array<kernels::Launch, 2> launches_for(const Uploaded& u, const Shape& s, std::uint32_t window) {
    return kernels::block_attention_launches({model::LayerDescription{s.query_heads, s.key_value_heads, s.d, 0,
                                                                      window, 1e4f, {}},
                                              1.0f, u.query, u.key, u.value, u.scores, u.maxima, u.attended});
}

std::vector<float> run(WGPUInstance instance, const gpu::Device& device, const Uploaded& u, const Shape& s,
                       std::uint32_t window, std::uint32_t tokens) {
    auto pair = launches_for(u, s, window);
    const auto program = build_program(instance, *u.upload, {std::move(pair[0]), std::move(pair[1])});
    run_step(instance, *program, tokens);
    return read_floats(instance, device, u.upload->buffer(u.attended.buffer), 0,
                       std::size_t{tokens} * s.query_heads * s.d);
}

double gamma(std::uint64_t n) {
    const double nu = static_cast<double>(n) * std::ldexp(1.0, -24);
    return nu / (1 - nu);
}

// Every output of the step against f64, within block_attention.h's margin;
// returns the largest error seen against it.
double check(const Shape& s, std::uint32_t window, std::uint32_t tokens, std::span<const float> q,
             std::span<const float> k, std::span<const float> v, std::span<const float> got) {
    const std::uint32_t half = window / 2;
    const std::uint32_t group = s.query_heads / s.key_value_heads;
    const double log2e = std::numbers::log2e;
    double worst = 0;
    for (std::uint32_t i = 0; i < tokens; ++i) {
        for (std::uint32_t h = 0; h < s.query_heads; ++h) {
            const std::uint32_t kv = h / group;
            const auto qi = q.subspan((std::size_t{i} * s.query_heads + h) * s.d, s.d);
            std::vector<double> score(tokens, 0.0);
            std::vector<bool> live(tokens, false);
            double top = -1e300, magnitude = 0, largest_v = 0;
            for (std::uint32_t j = 0; j < tokens; ++j) {
                const std::uint32_t apart = i > j ? i - j : j - i;
                if (apart > half) continue;
                live[j] = true;
                const auto kj = k.subspan((std::size_t{j} * s.key_value_heads + kv) * s.d, s.d);
                double dot = 0, absolute = 0;
                for (std::uint32_t c = 0; c < s.d; ++c) {
                    dot += double{qi[c]} * kj[c];
                    absolute += std::abs(double{qi[c]} * kj[c]);
                }
                score[j] = dot * log2e;
                top = std::max(top, score[j]);
                magnitude = std::max(magnitude, absolute);
                for (std::uint32_t c = 0; c < s.d; ++c) {
                    largest_v = std::max(largest_v, std::abs(double{v[(std::size_t{j} * s.key_value_heads + kv) * s.d + c]}));
                }
            }
            double sum = 0;
            std::vector<double> out(s.d, 0.0);
            for (std::uint32_t j = 0; j < tokens; ++j) {
                if (!live[j]) continue;
                const double w = std::exp2(score[j] - top);
                sum += w;
                for (std::uint32_t c = 0; c < s.d; ++c) {
                    out[c] += w * v[(std::size_t{j} * s.key_value_heads + kv) * s.d + c];
                }
            }
            // The scores' rounding, through exp2, relatively: in log2 units,
            // a weight moves by ln 2 times its score's error; then the sums'
            // and exp2's own (block_attention.h).
            const double bound = largest_v * (2 * gamma(s.d) * magnitude * log2e * std::numbers::ln2 +
                                              2 * gamma(tokens + 1) + std::ldexp(1.0, -16));
            for (std::uint32_t c = 0; c < s.d; ++c) {
                const double want = out[c] / sum;
                const double error = std::abs(got[(std::size_t{i} * s.query_heads + h) * s.d + c] - want);
                worst = std::max(worst, error / bound);
                if (!(error <= bound)) {   // NaN fails too
                    FAIL_CHECK("row " << i << " head " << h << " value " << c << " off by " << error);
                    return worst;
                }
            }
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("block attention is within its margin of f64, for both of EmbeddingGemma 2's shapes, at any step size") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    double worst = 0;
    for (const Shape& s : {kSliding, kGlobal}) {
        const auto q = read_floats(instance.get(), *device, u.upload->buffer(u.query.buffer), 0,
                                   std::size_t{kMaxTokens} * s.query_heads * s.d);
        const auto k = read_floats(instance.get(), *device, u.upload->buffer(u.key.buffer), 0,
                                   std::size_t{kMaxTokens} * s.key_value_heads * s.d);
        const auto v = read_floats(instance.get(), *device, u.upload->buffer(u.value.buffer), 0,
                                   std::size_t{kMaxTokens} * s.key_value_heads * s.d);
        struct Case {
            std::uint32_t tokens, window;
        };
        // Steps either side of a 32-key weigh step and a 64-key chunk; a full
        // layer's window, and one narrower than the step, masking either side.
        for (const Case c : {Case{2, 262144}, Case{31, 262144}, Case{33, 262144}, Case{64, 262144},
                             Case{65, 262144}, Case{130, 262144}, Case{130, 32}, Case{130, 1024}}) {
            CAPTURE(s.name);
            CAPTURE(c.tokens);
            CAPTURE(c.window);
            const auto got = run(instance.get(), *device, u, s, c.window, c.tokens);
            worst = std::max(worst, check(s, c.window, c.tokens, q, k, v, got));
        }
    }
    MESSAGE("largest error, against its bound: " << worst);
}

TEST_CASE("block attention over a whole block of 512 tokens is within its margin, and the same bits run twice") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    const Shape& s = kSliding;
    const auto q = read_floats(instance.get(), *device, u.upload->buffer(u.query.buffer), 0,
                               std::size_t{kMaxTokens} * s.query_heads * s.d);
    const auto k = read_floats(instance.get(), *device, u.upload->buffer(u.key.buffer), 0,
                               std::size_t{kMaxTokens} * s.key_value_heads * s.d);
    const auto v = read_floats(instance.get(), *device, u.upload->buffer(u.value.buffer), 0,
                               std::size_t{kMaxTokens} * s.key_value_heads * s.d);
    const auto first = run(instance.get(), *device, u, s, 1024, kMaxTokens);
    check(s, 1024, kMaxTokens, q, k, v, first);
    const auto second = run(instance.get(), *device, u, s, 1024, kMaxTokens);
    CHECK(std::equal(first.begin(), first.end(), second.begin()));
}

TEST_CASE("block attention's launches: a scores launch over the step's chunks, then a weigh launch") {
    const residency::BufferRange r{residency::BufferIndex{1}, 0, 64};
    const auto pair = kernels::block_attention_launches(
        {model::LayerDescription{4, 1, 512, 0, 262144, 1e6f, {}}, 1.0f, r, r, r, r, r, r});
    CHECK(pair[0].entry_point == "scores_of");
    CHECK(pair[0].key_split == kernels::KeySplit::PerChunk);
    CHECK(pair[0].rows_per_tile == 8);   // 32 vectors, 4 heads a row
    CHECK(pair[1].entry_point == "weigh");
    CHECK(pair[1].key_split == kernels::KeySplit::None);
    // At 512 tokens: 64 tiles of 8 rows; scores a key-value head and a chunk
    // each, 64 × 8 = 512; weigh a key-value head and 64 of d, 64 × 8 = 512.
    const auto workgroups = [](const kernels::Launch& l, std::uint32_t tokens) {
        return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window,
                                        l.tokens, l.partial_rows},
                                       l.workgroup_size, 0, tokens, true);
    };
    CHECK(workgroups(pair[0], 512) == 512);
    CHECK(workgroups(pair[1], 512) == 512);
    CHECK(workgroups(pair[0], 65) == 9 * 2);   // 9 tiles, 2 chunks
}

namespace {

constexpr Shape kVision{"vision", 12, 12, 64};
constexpr std::uint32_t kVisionRows = 2520;

// Buffers for a vision step of up to 2,520 patches, queries, keys and values
// filled once on the device.
struct Streamed {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange query{}, key{}, value{}, attended{};
};

Streamed upload_vision(WGPUInstance instance, const gpu::Device& device) {
    Streamed u;
    u.model.bytes = load_gguf_fixture("rope_rows");   // any file: no weight is read
    gguf::MemoryByteSource source{u.model.bytes};
    REQUIRE(gguf::read_index(source, u.model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = u.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    const auto alone = [&](std::uint64_t bytes) {
        plan.buffers.push_back({residency::Pool::Scratch, bytes});
        return residency::BufferRange{static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, bytes};
    };
    const std::uint64_t bytes = std::uint64_t{kVisionRows} * 12 * 64 * 4;
    u.query = alone(bytes);
    u.key = alone(bytes);
    u.value = alone(bytes);
    u.attended = alone(bytes);
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == residency::UploadError::Ok);
    const auto program = build_program(instance, *u.upload,
                                       {fill_launch(u.query, 11, 1.0f, false), fill_launch(u.key, 23, 1.0f, false),
                                        fill_launch(u.value, 37, 1.0f, false)});
    run_step(instance, *program, 1);
    return u;
}

kernels::Launch stream_launch(const Streamed& u, std::uint32_t window) {
    return kernels::streamed_attention_launch(
        {model::LayerDescription{kVision.query_heads, kVision.key_value_heads, kVision.d, 0, window, 100.0f, {}}, 1.0f,
         u.query, u.key, u.value, u.attended});
}

// `rows` of a streaming step against f64, every head and dimension of each,
// within block_attention.h's margin and its rescales'.
double check_rows(std::uint32_t window, std::uint32_t tokens, std::span<const std::uint32_t> rows,
                  std::span<const float> q, std::span<const float> k, std::span<const float> v,
                  std::span<const float> got) {
    const Shape& s = kVision;
    const std::uint32_t half = window / 2;
    const double log2e = std::numbers::log2e;
    const double rescales = 2.0 * ((tokens + 15) / 16) * std::ldexp(1.0, -24);
    double worst = 0;
    for (const std::uint32_t i : rows) {
        for (std::uint32_t h = 0; h < s.query_heads; ++h) {
            const auto qi = q.subspan((std::size_t{i} * s.query_heads + h) * s.d, s.d);
            std::vector<double> score(tokens, 0.0);
            std::vector<bool> live(tokens, false);
            double top = -1e300, magnitude = 0, largest_v = 0;
            for (std::uint32_t j = 0; j < tokens; ++j) {
                const std::uint32_t apart = i > j ? i - j : j - i;
                if (apart > half) continue;
                live[j] = true;
                const auto kj = k.subspan((std::size_t{j} * s.key_value_heads + h) * s.d, s.d);
                double dot = 0, absolute = 0;
                for (std::uint32_t c = 0; c < s.d; ++c) {
                    dot += double{qi[c]} * kj[c];
                    absolute += std::abs(double{qi[c]} * kj[c]);
                }
                score[j] = dot * log2e;
                top = std::max(top, score[j]);
                magnitude = std::max(magnitude, absolute);
                for (std::uint32_t c = 0; c < s.d; ++c) {
                    largest_v = std::max(largest_v, std::abs(double{v[(std::size_t{j} * s.key_value_heads + h) * s.d + c]}));
                }
            }
            double sum = 0;
            std::vector<double> out(s.d, 0.0);
            for (std::uint32_t j = 0; j < tokens; ++j) {
                if (!live[j]) continue;
                const double w = std::exp2(score[j] - top);
                sum += w;
                for (std::uint32_t c = 0; c < s.d; ++c) out[c] += w * v[(std::size_t{j} * s.key_value_heads + h) * s.d + c];
            }
            const double bound = largest_v * (2 * gamma(s.d) * magnitude * log2e * std::numbers::ln2 +
                                              2 * gamma(tokens + 1) + std::ldexp(1.0, -16) + rescales);
            for (std::uint32_t c = 0; c < s.d; ++c) {
                const double want = out[c] / sum;
                const double error = std::abs(got[(std::size_t{i} * s.query_heads + h) * s.d + c] - want);
                worst = std::max(worst, error / bound);
                if (!(error <= bound)) {
                    FAIL_CHECK("row " << i << " head " << h << " value " << c << " off by " << error);
                    return worst;
                }
            }
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("the streaming form is within its margin of f64 at a vision encoder's shape, up to 2,520 patches") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Streamed u = upload_vision(instance.get(), *device);
    const auto read = [&](const residency::BufferRange& r, std::uint32_t rows) {
        return read_floats(instance.get(), *device, u.upload->buffer(r.buffer), 0, std::size_t{rows} * 12 * 64);
    };
    const auto q = read(u.query, kVisionRows);
    const auto k = read(u.key, kVisionRows);
    const auto v = read(u.value, kVisionRows);
    double worst = 0;
    struct Case {
        std::uint32_t tokens, window;
    };
    // Steps either side of a 16-key step; a window narrower than the step,
    // so a vector meets steps with no live key; and the most patches.
    for (const Case c : {Case{2, UINT32_MAX}, Case{15, UINT32_MAX}, Case{16, UINT32_MAX}, Case{17, UINT32_MAX},
                         Case{33, UINT32_MAX}, Case{130, 32}, Case{630, UINT32_MAX}, Case{kVisionRows, UINT32_MAX}}) {
        CAPTURE(c.tokens);
        CAPTURE(c.window);
        const auto program = build_program(instance.get(), *u.upload, {stream_launch(u, c.window)});
        run_step(instance.get(), *program, c.tokens);
        const auto got = read(u.attended, c.tokens);
        std::vector<std::uint32_t> rows;
        if (c.tokens <= 130) {
            for (std::uint32_t i = 0; i < c.tokens; ++i) rows.push_back(i);
        } else {
            for (const std::uint32_t i : {0u, 1u, 31u, 32u, c.tokens / 2, c.tokens - 2, c.tokens - 1}) rows.push_back(i);
        }
        worst = std::max(worst, check_rows(c.window, c.tokens, rows, q, k, v, got));
    }
    MESSAGE("largest error, against its bound: " << worst);
}

TEST_CASE("the streaming form gives the same bits run twice, and its workgroups are its tiles by heads") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Streamed u = upload_vision(instance.get(), *device);
    const kernels::Launch launch = stream_launch(u, UINT32_MAX);
    CHECK(launch.entry_point == "stream");
    CHECK(launch.bindings.size() == 4);
    CHECK(kernels::workgroups_for({launch.rows, launch.invocations_per_row, launch.rows_per_tile, launch.key_split,
                                   launch.window, launch.tokens},
                                  launch.workgroup_size, 0, kVisionRows, true) == 79 * 12);
    const auto program = build_program(instance.get(), *u.upload, {launch});
    std::vector<std::vector<float>> runs;
    for (int r = 0; r < 2; ++r) {
        run_step(instance.get(), *program, 630);
        runs.push_back(read_floats(instance.get(), *device, u.upload->buffer(u.attended.buffer), 0,
                                   std::size_t{630} * 12 * 64));
    }
    CHECK(runs[0] == runs[1]);
}
