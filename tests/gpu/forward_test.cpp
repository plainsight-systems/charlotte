// The forward pass on the GPU (graph/graph.h): a generated two-layer
// Qwen3-shaped model, uploaded and run as the harness runs one, and the last
// token's logits the same bits however its prompt is stepped.

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <set>
#include <thread>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/kernels/program.h"
#include "core/sampler/sampler.h"
#include "core/residency/upload.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

// The model, uploaded through the capability table's formats, and its
// graph built into a program.
struct Running {
    Model model;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::unique_ptr<kernels::Program> program;
    residency::BufferRange logits{};
};

Running run_model(WGPUInstance instance, const gpu::Device& device) {
    Running r;
    r.model = load("forward_model");
    std::string_view name;
    REQUIRE(r.model.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    const arch::Architecture* architecture = capability::find_architecture(name);
    REQUIRE(architecture->describe(r.model.index, r.description).ok());

    Ready ready;
    residency::Upload::begin(device, r.model.index, r.model.plan, r.model.bytes.size(), capability::find_format, {},
                             kChunk, on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    r.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *r.upload, r.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *r.upload) == residency::UploadError::Ok);

    const residency::ResidencyPlan& plan = r.upload->plan();
    std::vector<kernels::Launch> launches;
    const graph::GraphResult built =
        architecture->graph(r.description, plan, *capability::find_format(plan.cache_type), launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);
    residency::BufferRange sampled{};
    for (const residency::PlannedScratch& s : plan.scratch) {
        if (s.purpose == "logits") r.logits = s.range;
        if (s.purpose == "sampled") sampled = s.range;
    }
    // The draw's record read back from each step that asks for logits.
    r.program = build_program(instance, *r.upload, std::move(launches),
                              kernels::Binding{sampled.buffer, sampled.offset, sizeof(sampler::SampledRecord)});
    return r;
}

// The prompt, pseudo-random identifiers below the vocabulary.
std::vector<std::uint32_t> prompt(std::size_t length, std::uint32_t vocabulary) {
    std::vector<std::uint32_t> ids;
    std::uint32_t state = 0x2545F491;
    for (std::size_t i = 0; i < length; ++i) {
        state = state * 1664525 + 1013904223;
        ids.push_back((state >> 8) % vocabulary);
    }
    return ids;
}

// Runs `ids` in steps of `steps` tokens from position 0, logits asked of the
// last step alone, and returns the last token's logits. Each run uploads
// the model afresh, so its buffers start zeroed and nothing an earlier run
// wrote — logits, cache, working buffers — can stand in for work this one
// omits.
std::vector<float> stepped(WGPUInstance instance, const gpu::Device& device, std::span<const std::uint32_t> ids,
                           std::span<const std::uint32_t> steps) {
    Running r = run_model(instance, device);
    std::uint32_t position = 0;
    for (std::size_t s = 0; s < steps.size(); ++s) {
        const bool last = s + 1 == steps.size();
        run_step(instance, *r.program, steps[s], ids.subspan(position, steps[s]), position, last);
        position += steps[s];
    }
    REQUIRE(position == ids.size());
    return read_floats(instance, device, r.upload->buffer(r.logits.buffer), r.logits.offset,
                       r.description.vocabulary_size);
}

}  // namespace

TEST_CASE("a token's logits are the same bits however its prompt is stepped") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    // 300 tokens: its keys span two 256-key chunks.
    const auto ids = prompt(300, 256);

    const std::vector<std::uint32_t> whole{300};
    const auto want = stepped(instance.get(), *device, ids, whole);
    REQUIRE(want.size() == 256);
    CHECK(std::all_of(want.begin(), want.end(), [](float f) { return std::isfinite(f); }));
    CHECK(*std::max_element(want.begin(), want.end()) > *std::min_element(want.begin(), want.end()));

    // Steps of 5 and 12, in the 8- and 16-token tiles, then 240 and 43 in
    // the 32-token tile, the last split across the two chunks.
    const std::vector<std::uint32_t> tiles{5, 12, 240, 43};
    // A prefill, then a decode step, split across the chunks.
    const std::vector<std::uint32_t> then_decode{299, 1};
    // A token at a time, every step past position 255 split.
    const std::vector<std::uint32_t> decoded(300, 1);
    for (const auto* steps : {&tiles, &then_decode, &decoded}) {
        CAPTURE(steps->size());
        const auto got = stepped(instance.get(), *device, ids, *steps);
        CHECK(std::equal(got.begin(), got.end(), want.begin(), want.end()));
    }
}

namespace {

// A step's report, for the decoding below.
struct Decoded {
    std::vector<std::uint32_t> tokens;   // in the order reported
    bool failed = false;
};

struct Pending {
    Decoded* decoded;
};

void on_decoded(kernels::ProgramError e, std::string_view message, std::span<const std::byte> bytes, void* userdata) {
    Decoded& d = *static_cast<Pending*>(userdata)->decoded;
    if (e != kernels::ProgramError::Ok || bytes.size() != sizeof(sampler::SampledRecord)) {
        FAIL_CHECK("a step failed: " << message);
        d.failed = true;
        return;
    }
    sampler::SampledRecord record;
    std::memcpy(&record, bytes.data(), sizeof record);
    CHECK(record.failed == 0);
    d.tokens.push_back(record.token);
}

void pump_for(WGPUInstance instance, const Decoded& d, std::size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while (d.tokens.size() < count && !d.failed) {
        wgpuInstanceProcessEvents(instance);
        if (std::chrono::steady_clock::now() > deadline) FAIL("timed out waiting for token " << count);
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

kernels::Step prompt_step(std::span<const std::uint32_t> ids, const policy::SamplingSettings& s) {
    kernels::Step step{};
    step.position = 0;
    step.tokens = static_cast<std::uint32_t>(ids.size());
    step.logits = 1;
    std::copy(ids.begin(), ids.end(), step.ids.begin());
    sampler::apply(s, policy::Seed{0x5EED}, step);
    return step;
}

kernels::Step decode_step(std::uint32_t position, std::optional<std::uint32_t> token,
                          const policy::SamplingSettings& s) {
    kernels::Step step{};
    step.position = position;
    step.tokens = 1;
    step.logits = 1;
    step.fed = token ? 0 : 1;
    if (token) step.ids[0] = *token;
    sampler::apply(s, policy::Seed{0x5EED}, step);
    return step;
}

// Decodes `count` tokens after `ids`, each step fed its token on the GPU and
// run before the last step's token is read back: two steps outstanding.
std::vector<std::uint32_t> decode_pipelined(WGPUInstance instance, const gpu::Device& device,
                                            std::span<const std::uint32_t> ids, std::size_t count,
                                            const policy::SamplingSettings& s) {
    Running r = run_model(instance, device);
    Decoded d;
    Pending pending{&d};
    const auto prompt_length = static_cast<std::uint32_t>(ids.size());
    r.program->run(prompt_step(ids, s), on_decoded, &pending);
    // Token i is drawn by the step at position prompt_length + i − 1; each
    // decode step embeds the last's draw from the GPU.
    for (std::uint32_t i = 1; i < count; ++i) {
        r.program->run(decode_step(prompt_length + i - 1, std::nullopt, s), on_decoded, &pending);
        pump_for(instance, d, i);   // the one before it reported: one left outstanding
    }
    pump_for(instance, d, count);
    return d.tokens;
}

// The same, a step at a time: each step waits for the last's token and is
// given it by identifier.
std::vector<std::uint32_t> decode_step_at_a_time(WGPUInstance instance, const gpu::Device& device,
                                                 std::span<const std::uint32_t> ids, std::size_t count,
                                                 const policy::SamplingSettings& s) {
    Running r = run_model(instance, device);
    Decoded d;
    Pending pending{&d};
    const auto prompt_length = static_cast<std::uint32_t>(ids.size());
    r.program->run(prompt_step(ids, s), on_decoded, &pending);
    pump_for(instance, d, 1);
    for (std::uint32_t i = 1; i < count; ++i) {
        r.program->run(decode_step(prompt_length + i - 1, d.tokens.back(), s), on_decoded, &pending);
        pump_for(instance, d, i + 1);
    }
    return d.tokens;
}

}  // namespace

TEST_CASE("decoding fed on the GPU, two steps outstanding, draws the tokens decoding a step at a time does") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto ids = prompt(20, 256);
    for (const policy::SamplingSettings& s : {policy::SamplingSettings{0.0f, 40, 0.95f, 0.05f},
                                              policy::SamplingSettings{0.8f, 40, 0.95f, 0.05f}}) {
        CAPTURE(s.temperature);
        const auto pipelined = decode_pipelined(instance.get(), *device, ids, 30, s);
        const auto stepped = decode_step_at_a_time(instance.get(), *device, ids, 30, s);
        REQUIRE(pipelined.size() == 30);
        CHECK(pipelined == stepped);
        // Sampling, not one token over and over: the draw reached the model's
        // logits and its seed.
        if (s.temperature > 0) CHECK(std::set<std::uint32_t>(stepped.begin(), stepped.end()).size() > 1);
    }
}
