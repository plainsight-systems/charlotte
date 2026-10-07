#pragma once

// Test-only: a fixture model uploaded through the capability table's formats
// and its graph built into a program reading back the draw's record, as the
// harness runs one — for the GPU tests of the forward pass and the runtime.

#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/kernels/program.h"
#include "core/policy/policy.h"
#include "core/residency/upload.h"
#include "core/sampler/sampler.h"
#include "support/program.h"
#include "support/upload.h"

namespace bllm::testing {

struct Running {
    Model model;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::unique_ptr<kernels::Program> program;
    residency::BufferRange logits{};
    residency::BufferRange candidates{};
};

inline Running run_model(WGPUInstance instance, const gpu::Device& device, const std::string& fixture = "forward_model",
                         const policy::LoadPolicy& policy = {}) {
    Running r;
    r.model = load(fixture, policy);
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
        if (s.purpose == "candidates") r.candidates = s.range;
        if (s.purpose == "sampled") sampled = s.range;
    }
    // The draw's record read back from each step that asks for logits.
    r.program = build_program(instance, *r.upload, std::move(launches),
                              kernels::Binding{sampled.buffer, sampled.offset, sizeof(sampler::SampledRecord)});
    return r;
}

}  // namespace bllm::testing
