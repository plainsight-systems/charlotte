// The forward pass on the GPU (graph/graph.h): a generated two-layer
// Qwen3-shaped model, uploaded and run as the harness runs one, and the last
// token's logits the same bits however its prompt is stepped.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/kernels/program.h"
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
    r.program = build_program(instance, *r.upload, std::move(launches));
    for (const residency::PlannedScratch& s : plan.scratch) {
        if (s.purpose == "logits") r.logits = s.range;
    }
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
// last step alone, and returns the last token's logits.
std::vector<float> stepped(WGPUInstance instance, const gpu::Device& device, Running& r,
                           std::span<const std::uint32_t> ids, std::span<const std::uint32_t> steps) {
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
    Running r = run_model(instance.get(), *device);
    // 300 tokens: its keys span two 256-key chunks.
    const auto ids = prompt(300, r.description.vocabulary_size);

    const std::vector<std::uint32_t> whole{300};
    const auto want = stepped(instance.get(), *device, r, ids, whole);
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
        const auto got = stepped(instance.get(), *device, r, ids, *steps);
        CHECK(std::equal(got.begin(), got.end(), want.begin(), want.end()));
    }
}
