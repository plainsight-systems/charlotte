// Attention on Dawn, against the same computation in f64 from the same
// queries and stored keys and values (kernels/attention/attention.h): each
// listed shape, decode and prefill steps at and across chunk boundaries,
// split and folded within a workgroup, Gemma 3's window over a ring; and
// the same query decoded and prefilled giving the same bits.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "core/capability/capability.h"
#include "core/formats/f16/f16.h"
#include "core/gguf/reader.h"
#include "core/kernels/attention/attention.h"
#include "core/quant/q4_0.h"
#include "support/acquire.h"
#include "support/fill.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kSlots = 1024;
constexpr std::uint64_t kRowFloats = 2048;   // every shape's query width: 16 × 128 = 32 × 64

struct Shape {
    const char* name;
    std::uint32_t query_heads, key_value_heads, d, window, slots;
};

constexpr Shape kQwen{"qwen", 16, 8, 128, 40960, kSlots};
constexpr Shape kLlama{"llama", 32, 8, 64, 131072, kSlots};
// A window of 300 over a ring of 320 slots, so positions past 320 wrap.
constexpr Shape kGemma{"gemma", 4, 1, 256, 300, 320};


// Test-only: copies rows first .. first + tokens of the source into the
// query buffer.
constexpr std::string_view kCopy = R"(
struct Copy { vec4s: u32, first: u32 }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> copy: Copy;
@group(0) @binding(2) var<storage, read> copy_from: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> copy_to: array<vec4<f32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < step.tokens * copy.vec4s) {
        copy_to[id.x] = copy_from[copy.first * copy.vec4s + id.x];
    }
}
)";

// Working buffers, each its own, as the plan places them; the KV cache in a
// buffer that can be read back, as the plan's cache pool's cannot.
struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange source{}, query{}, attended{}, partials{}, stats{};
    residency::PlannedCacheLayer cache{};
};

std::vector<std::byte> words(std::initializer_list<std::uint32_t> values) {
    std::vector<std::byte> bytes(values.size() * 4);
    std::memcpy(bytes.data(), std::data(values), bytes.size());
    return bytes;
}


Uploaded upload_buffers(WGPUInstance instance, const gpu::Device& device) {
    Uploaded u;
    u.model.bytes = load_gguf_fixture("rope_rows");   // any file: the weights are not read
    gguf::MemoryByteSource source{u.model.bytes};
    REQUIRE(gguf::read_index(source, u.model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = u.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    const auto alone = [&](std::uint64_t bytes) {
        plan.buffers.push_back({residency::Pool::Scratch, bytes});
        return residency::BufferRange{static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, bytes};
    };
    const std::uint64_t rows = 512 * kRowFloats * 4;
    u.source = alone(rows);
    u.query = alone(rows);
    u.attended = alone(rows);
    // Partial buffers of 1,024 rows, so a step's partial rows can be set
    // past the prefill block's.
    u.partials = alone(2 * rows);
    u.stats = alone(1024 * 32 * 2 * 4);
    const std::uint64_t layer = std::uint64_t{kSlots} * 8 * 128 * 2;   // the widest key-value heads
    const residency::BufferRange cache = alone(2 * layer);
    u.cache = {{cache.buffer, 0, layer}, {cache.buffer, layer, layer}, kSlots};
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == residency::UploadError::Ok);

    // Queries in [−3, 3); keys in [−3, 3) and values in [−2, 2), as halves.
    const auto fill = [&](const residency::BufferRange& range, std::uint32_t seed, float amplitude, bool halves) {
        const auto count = static_cast<std::uint32_t>(range.length / 4);
        return kernels::Launch{kFill, nullptr, fill_constants(seed, count, amplitude, halves),
                               {{range.buffer, range.offset, range.length}}, count, 64};
    };
    const auto program = build_program(instance, *u.upload,
                                       {fill(u.source, 1, 3.0f, false), fill(u.cache.keys, 1u << 24, 3.0f, true),
                                        fill(u.cache.values, 1u << 25, 2.0f, true)});
    run_step(instance, *program, 1);
    return u;
}

model::LayerDescription layer_of(const Shape& s) {
    return {s.query_heads, s.key_value_heads, s.d, 0, s.window, 1e6f, {}};
}

double scale_of(const Shape& s) { return 1.0 / std::sqrt(static_cast<double>(s.d)); }

// A step of `tokens` query rows of the source from row `first`, at
// `position`: its attention output, rows of H_q × d.
std::vector<float> run_attention(WGPUInstance instance, const gpu::Device& device, const Uploaded& u,
                                 const Shape& s, std::uint32_t position, std::uint32_t tokens, std::uint32_t first,
                                 std::uint32_t partial_rows = residency::kPrefillBlock) {
    const std::uint32_t width = s.query_heads * s.d;
    residency::PlannedCacheLayer cache = u.cache;
    cache.slots = s.slots;
    const auto launches = kernels::attention_launches({layer_of(s), static_cast<float>(scale_of(s)), u.query, cache,
                                                       &formats::kF16, u.attended, u.partials, u.stats,
                                                       partial_rows});
    const auto program = build_program(
        instance, *u.upload,
        {kernels::Launch{kCopy, nullptr, words({width / 4, first}),
                         {{u.source.buffer, 0, u.source.length}, {u.query.buffer, 0, u.query.length}}, width / 4, 64},
         launches[0], launches[1]});
    run_step(instance, *program, tokens, {}, position);
    return read_floats(instance, device, u.upload->buffer(u.attended.buffer), 0, std::size_t{tokens} * width);
}

// What the KV cache holds, read back once: keys then values, as halves.
struct Cache {
    std::vector<std::uint16_t> keys, values;
};

Cache read_cache(WGPUInstance instance, const gpu::Device& device, const Uploaded& u) {
    const auto halves = [&](const residency::BufferRange& r) {
        const auto w = read_floats(instance, device, u.upload->buffer(r.buffer), r.offset, r.length / 4);
        std::vector<std::uint16_t> h(w.size() * 2);
        std::memcpy(h.data(), w.data(), w.size() * 4);
        return h;
    };
    return {halves(u.cache.keys), halves(u.cache.values)};
}

// Checks a step's output against f64 within attention.h's bound; returns
// the largest error seen against it.
double check_step(const Shape& s, std::span<const float> got, std::span<const float> source, const Cache& cache,
                  std::uint32_t position, std::uint32_t tokens, std::uint32_t first) {
    const std::uint32_t group = s.query_heads / s.key_value_heads;
    const double scale = scale_of(s);
    double worst = 0;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const std::uint32_t p = position + t;
        const std::uint32_t earliest = p + 1 > s.window ? p + 1 - s.window : 0;
        for (std::uint32_t h = 0; h < s.query_heads; ++h) {
            const float* q = source.data() + (std::size_t{first + t} * s.query_heads + h) * s.d;
            const std::uint32_t kv = h / group;
            const auto at = [&](std::uint32_t j) { return (std::size_t{j % s.slots} * s.key_value_heads + kv) * s.d; };
            std::vector<double> score;
            double magnitude = 0, largest_value = 0;
            for (std::uint32_t j = earliest; j <= p; ++j) {
                double dot = 0, absolute = 0;
                for (std::uint32_t e = 0; e < s.d; ++e) {
                    const double k = quant::fp16_to_fp32(cache.keys[at(j) + e]);
                    dot += q[e] * k;
                    absolute += std::abs(q[e] * k);
                    largest_value = std::max<double>(largest_value, std::abs(quant::fp16_to_fp32(cache.values[at(j) + e])));
                }
                score.push_back(dot * scale);
                magnitude = std::max(magnitude, absolute);
            }
            const double top = *std::max_element(score.begin(), score.end());
            double sum = 0;
            for (double& x : score) sum += (x = std::exp(x - top));
            const double allowed =
                largest_value * (s.d * std::ldexp(1.0, -24) * magnitude * scale * std::numbers::log2e + std::ldexp(1.0, -16));
            for (std::uint32_t e = 0; e < s.d; ++e) {
                double want = 0;
                for (std::uint32_t j = earliest; j <= p; ++j) {
                    want += score[j - earliest] / sum * quant::fp16_to_fp32(cache.values[at(j) + e]);
                }
                const double error = std::abs(got[(std::size_t{t} * s.query_heads + h) * s.d + e] - want);
                worst = std::max(worst, error / allowed);
                if (!(error <= allowed)) {   // NaN fails too
                    FAIL_CHECK("row " << t << " head " << h << " value " << e << " off by " << error << ", allowed "
                                      << allowed);
                    return worst;
                }
            }
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("each listed shape attends within the bound, decoded and prefilled, split and folded") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    const auto source = read_floats(instance.get(), *device, u.upload->buffer(u.source.buffer), 0, u.source.length / 4);
    const Cache cache = read_cache(instance.get(), *device, u);
    struct Step {
        std::uint32_t position, tokens, first;
    };
    double worst = 0;
    for (const Shape& s : {kQwen, kLlama, kGemma}) {
        for (const Step& step : {
                 // Decode: one chunk, then split across two, then many.
                 Step{0, 1, 0}, Step{63, 1, 3}, Step{64, 1, 5}, Step{700, 1, 7}, Step{1023, 1, 9},
                 // Prefill: one chunk; split across a boundary; a partial
                 // tile at the end; folded within each workgroup.
                 Step{0, 4, 0}, Step{62, 3, 11}, Step{1019, 5, 20}, Step{0, 300, 0},
             }) {
            CAPTURE(s.name);
            CAPTURE(step.position);
            CAPTURE(step.tokens);
            const auto got = run_attention(instance.get(), *device, u, s, step.position, step.tokens, step.first);
            worst = std::max(worst, check_step(s, got, source, cache, step.position, step.tokens, step.first));
        }
    }
    MESSAGE("largest error, against its bound: " << worst);
}

TEST_CASE("a query decoded alone and prefilled among 300 gives the same bits") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    for (const Shape& s : {kQwen, kLlama, kGemma}) {
        CAPTURE(s.name);
        // 300 rows from 0 fold within each workgroup: 5 chunks × 300 rows
        // overflow the partial buffers.
        REQUIRE(kernels::key_chunks(0, 300, s.window, residency::kPrefillBlock).splits == 1);
        const auto prefilled = run_attention(instance.get(), *device, u, s, 0, 300, 0);
        const std::size_t width = std::size_t{s.query_heads} * s.d;
        for (const std::uint32_t r : {0u, 1u, 63u, 64u, 255u, 256u, 299u}) {
            CAPTURE(r);
            // Alone at its position: split across its chunks from 64 on.
            const auto decoded = run_attention(instance.get(), *device, u, s, r, 1, r);
            CHECK(std::equal(decoded.begin(), decoded.end(), prefilled.begin() + r * width));
        }
    }
}

TEST_CASE("a step's partial rows decide whether it splits, and split or folded its bits are the same") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_buffers(instance.get(), *device);
    for (const Shape& s : {kQwen, kLlama, kGemma}) {
        CAPTURE(s.name);
        // 33 rows at 990 reach 16 chunks of a full-attention layer, 528
        // partial rows: folded with 512, split with 1,024.
        const auto chunks = [&](std::uint32_t rows) { return kernels::key_chunks(990, 33, s.window, rows); };
        if (s.window >= 1023) {
            REQUIRE(chunks(512).splits == 1);
            REQUIRE(chunks(1024).splits == 16);
        }
        const auto folded = run_attention(instance.get(), *device, u, s, 990, 33, 40, 512);
        const auto split = run_attention(instance.get(), *device, u, s, 990, 33, 40, 1024);
        CHECK(std::equal(folded.begin(), folded.end(), split.begin()));
    }
}
