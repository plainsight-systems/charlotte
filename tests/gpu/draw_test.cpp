// The draw on the GPU (sampler/sampler.h): its uniform bit for bit against
// the reference Philox, each draw's token against the reference steps in
// f64 given that uniform, the settings at their edges, a non-finite top
// candidate, and the frequencies of many draws.

#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "core/kernels/program.h"
#include "core/residency/upload.h"
#include "core/sampler/sampler.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/sampler_reference.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

struct Drawing {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange candidates{}, sampled{};
    std::unique_ptr<kernels::Program> program;
};

std::unique_ptr<Drawing> make(WGPUInstance instance, const gpu::Device& device) {
    auto d = std::make_unique<Drawing>();
    d->model.bytes = load_gguf_fixture("rope_rows");   // any file: no weight is read
    gguf::MemoryByteSource source{d->model.bytes};
    REQUIRE(gguf::read_index(source, d->model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = d->model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    // The candidates in a buffer the test writes from the CPU, a weight
    // pool's; the record in a working buffer, as the plan places it.
    plan.buffers.push_back({residency::Pool::Weights, 64 * 8});
    d->candidates = {residency::BufferIndex{1}, 0, 64 * 8};
    plan.buffers.push_back({residency::Pool::Scratch, 16});
    d->sampled = {residency::BufferIndex{2}, 0, 16};
    Ready ready;
    residency::Upload::begin(device, d->model.index, plan, d->model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    d->upload = std::move(ready.upload);
    REQUIRE(stream(instance, *d->upload, d->model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *d->upload) == residency::UploadError::Ok);
    d->program = build_program(instance, *d->upload, {sampler::draw_launch(d->candidates, d->sampled)});
    return d;
}

void write_candidates(const gpu::Device& device, const Drawing& d, const std::vector<Candidate>& c) {
    std::vector<std::uint32_t> words;
    for (const Candidate& x : c) {
        words.push_back(std::bit_cast<std::uint32_t>(x.logit));
        words.push_back(x.token);
    }
    wgpuQueueWriteBuffer(device.queue(), d.upload->buffer(d.candidates.buffer), 0, words.data(), words.size() * 4);
}

struct Drawn {
    std::uint32_t token;
    std::uint32_t failed;
    float u;
};

Drawn draw(WGPUInstance instance, const gpu::Device& device, const Drawing& d, const policy::SamplingSettings& s,
           std::uint64_t seed, std::uint32_t position, std::uint32_t tokens = 1) {
    kernels::Step step{};
    step.position = position;
    step.tokens = tokens;
    step.logits = 1;
    sampler::apply(s, policy::Seed{seed}, step);
    const StepOutcome ran = try_step(instance, *d.program, step);
    REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
    const auto words = read_floats(instance, device, d.upload->buffer(d.sampled.buffer), 0, 4);
    return {std::bit_cast<std::uint32_t>(words[0]), std::bit_cast<std::uint32_t>(words[1]), words[2]};
}

// 64 candidates, sorted, logits spread over about 8 nats, tokens scattered.
std::vector<Candidate> candidates(std::uint32_t seed) {
    std::vector<Candidate> c(64);
    std::uint32_t state = seed;
    float logit = 12.0f;
    for (std::uint32_t i = 0; i < 64; ++i) {
        state = state * 1664525u + 1013904223u;
        logit -= static_cast<float>((state >> 8) % 1000u) / 4000.0f;
        c[i] = {logit, (state >> 4) % 151'936u};
    }
    return c;
}

}  // namespace

TEST_CASE("the draw's uniform is the reference Philox's, bit for bit") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto d = make(instance.get(), *device);
    write_candidates(*device, *d, candidates(1));
    for (const std::uint64_t seed : {0ull, 1ull, 0x0123456789ABCDEFull}) {
        for (const std::uint32_t position : {0u, 1u, 2u, 1000u, 40'959u, 16'777'215u}) {
            CAPTURE(seed);
            CAPTURE(position);
            // A one-token step at `position` draws the token at position + 1.
            CHECK(std::bit_cast<std::uint32_t>(draw(instance.get(), *device, *d, {}, seed, position).u) ==
                  std::bit_cast<std::uint32_t>(uniform(seed, position + 1)));
        }
    }
    // A step of many tokens draws the token after its last: the same uniform
    // however the tokens before it were stepped.
    for (const std::uint32_t tokens : {2u, 17u, 512u}) {
        CAPTURE(tokens);
        CHECK(std::bit_cast<std::uint32_t>(draw(instance.get(), *device, *d, {}, 7, 100, tokens).u) ==
              std::bit_cast<std::uint32_t>(uniform(7, 100 + tokens)));
    }
}

TEST_CASE("each draw is the reference's, given its uniform, across settings and positions") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto d = make(instance.get(), *device);
    const auto c = candidates(2);
    write_candidates(*device, *d, c);
    int compared = 0, near = 0;
    for (const policy::SamplingSettings& s : {
             policy::SamplingSettings{},                       // llama.cpp's defaults
             policy::SamplingSettings{0.6f, 20, 0.95f, 0.0f},  // Qwen3's card, thinking
             policy::SamplingSettings{1.0f, 64, 0.95f, 0.0f},  // Gemma 3's card
             policy::SamplingSettings{1.5f, 64, 1.0f, 0.01f},
             policy::SamplingSettings{0.3f, 8, 0.5f, 0.2f},
         }) {
        for (std::uint32_t position = 0; position < 200; ++position) {
            const Drawn got = draw(instance.get(), *device, *d, s, 42, position);
            const Reference want = reference_draw(c, s, got.u);
            CHECK(got.failed == 0);
            if (want.margin < 1e-5) {   // within f32's rounding of a decision
                ++near;
                continue;
            }
            ++compared;
            CHECK(got.token == c[want.chosen].token);
        }
    }
    MESSAGE(compared << " draws compared, " << near << " within rounding of a decision");
    CHECK(near * 100 <= compared);
}

TEST_CASE("the settings at their edges draw as the reference does") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto d = make(instance.get(), *device);
    const auto c = candidates(3);
    write_candidates(*device, *d, c);
    for (std::uint32_t position = 0; position < 20; ++position) {
        // Temperature 0, and a top_k of 1: the first candidate.
        CHECK(draw(instance.get(), *device, *d, {0.0f, 40, 0.95f, 0.05f}, 9, position).token == c[0].token);
        CHECK(draw(instance.get(), *device, *d, {1.0f, 1, 1.0f, 0.0f}, 9, position).token == c[0].token);
        // A top_p below the first candidate's probability: the first alone.
        CHECK(draw(instance.get(), *device, *d, {1.0f, 64, 0.01f, 0.0f}, 9, position).token == c[0].token);
        // A min_p that only the first passes.
        CHECK(draw(instance.get(), *device, *d, {1.0f, 64, 1.0f, 0.99f}, 9, position).token == c[0].token);
    }
    // A top_k of 2 with a high temperature: only the first two are drawn.
    for (std::uint32_t position = 0; position < 50; ++position) {
        const auto token = draw(instance.get(), *device, *d, {100.0f, 2, 1.0f, 0.0f}, 9, position).token;
        CHECK((token == c[0].token || token == c[1].token));
    }
}

TEST_CASE("a non-finite top candidate fails the draw, which writes a real token") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto d = make(instance.get(), *device);
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto c = candidates(4);
        c[0].logit = bad;
        write_candidates(*device, *d, c);
        const Drawn got = draw(instance.get(), *device, *d, {}, 1, 0);
        CHECK(got.failed == 1);
        CHECK(got.token == 0);
    }
}

TEST_CASE("the frequencies of 2,000 draws fit the reference's probabilities") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto d = make(instance.get(), *device);
    const auto c = candidates(5);
    write_candidates(*device, *d, c);
    const policy::SamplingSettings s{1.0f, 16, 0.95f, 0.0f};
    const Reference r = reference_draw(c, s, 0.5);
    constexpr int kDraws = 2000;
    std::vector<int> seen(64, 0);
    for (std::uint32_t position = 0; position < kDraws; ++position) {
        const auto token = draw(instance.get(), *device, *d, s, 77, position).token;
        const auto at = std::find_if(c.begin(), c.end(), [&](const Candidate& x) { return x.token == token; });
        REQUIRE(at != c.end());
        ++seen[at - c.begin()];
    }
    // Pearson's chi-squared over the candidates expected 5 times or more, the
    // rest pooled; against the 0.1% critical value (Wilson–Hilferty).
    double chi2 = 0, pooled_seen = 0, pooled_expected = 0;
    int bins = 0;
    for (std::size_t i = 0; i < 64; ++i) {
        const double expected = r.probability[i] * kDraws;
        if (r.probability[i] == 0) CHECK(seen[i] == 0);   // truncated away: never drawn
        if (expected >= 5) {
            chi2 += (seen[i] - expected) * (seen[i] - expected) / expected;
            ++bins;
        } else {
            pooled_seen += seen[i];
            pooled_expected += expected;
        }
    }
    if (pooled_expected >= 5) {
        chi2 += (pooled_seen - pooled_expected) * (pooled_seen - pooled_expected) / pooled_expected;
        ++bins;
    }
    const double df = bins - 1, z = 3.09;
    const double critical = df * std::pow(1 - 2 / (9 * df) + z * std::sqrt(2 / (9 * df)), 3);
    MESSAGE("chi-squared " << chi2 << " over " << df << " degrees of freedom, critical " << critical);
    CHECK(chi2 < critical);
}
