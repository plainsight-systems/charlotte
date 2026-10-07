// A profiled step (kernels/program.h), in a diagnostic build: the same
// results as run(), its GPU timestamps ordered, prefixes, two in flight, and
// each refusal.

#include <doctest/doctest.h>

#include "core/diagnostics.h"

#if BLLM_DIAGNOSTICS_ENABLED

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/kernels/program.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/running.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

struct Profiled {
    kernels::ProgramError error = kernels::ProgramError::Step;
    std::string message;
    std::vector<std::byte> readback;
    std::uint64_t begin_ns = 0;
    std::uint64_t end_ns = 0;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> launches;
    bool done = false;
};

void on_profiled(kernels::ProgramError e, std::string_view message, std::span<const std::byte> readback,
                 const kernels::Program::Timestamps& times, void* userdata) {
    auto& p = *static_cast<Profiled*>(userdata);
    p.error = e;
    p.message = std::string(message);
    p.readback.assign(readback.begin(), readback.end());
    p.begin_ns = times.begin_ns;
    p.end_ns = times.end_ns;
    p.launches.assign(times.launches.begin(), times.launches.end());
    p.done = true;
}

kernels::Step step_of(std::span<const std::uint32_t> ids, std::uint32_t position) {
    kernels::Step step{};
    step.position = position;
    step.tokens = static_cast<std::uint32_t>(ids.size());
    step.logits = 1;
    std::copy(ids.begin(), ids.end(), step.ids.begin());
    return step;
}

Profiled profile(WGPUInstance instance, kernels::Program& program, const kernels::Step& step, std::uint32_t launches) {
    Profiled p;
    program.run_profiled(step, launches, on_profiled, &p);
    pump_until(instance, p.done, "a profiled step");
    return p;
}

std::vector<std::uint32_t> bits(const std::vector<float>& f) {
    std::vector<std::uint32_t> out;
    for (const float x : f) out.push_back(std::bit_cast<std::uint32_t>(x));
    return out;
}

std::vector<std::uint32_t> ids_of(std::size_t length) {
    std::vector<std::uint32_t> ids;
    for (std::size_t i = 0; i < length; ++i) ids.push_back(static_cast<std::uint32_t>((i * 37 + 11) % 256));
    return ids;
}

}  // namespace

TEST_CASE("a profiled step leaves run()'s results, and its timestamps are in order") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get(), gpu::DiagnosticRequest{.timestamps = true});
    Running r = run_model(instance.get(), *device);
    const std::uint32_t all = static_cast<std::uint32_t>(r.launches_count);
    const auto ids = ids_of(40);
    for (const std::uint32_t tokens : {40u, 1u}) {
        CAPTURE(tokens);
        const auto step = step_of(std::span{ids}.first(tokens), 0);
        const StepOutcome plain = try_step(instance.get(), *r.program, step);
        REQUIRE(plain.error == kernels::ProgramError::Ok);
        const auto want = read_floats(instance.get(), *device, r.upload->buffer(r.logits.buffer), r.logits.offset, 256);
        const Profiled p = profile(instance.get(), *r.program, step, all);
        REQUIRE_MESSAGE(p.error == kernels::ProgramError::Ok, p.message);
        CHECK(p.readback == plain.readback);
        CHECK(bits(read_floats(instance.get(), *device, r.upload->buffer(r.logits.buffer), r.logits.offset, 256)) ==
              bits(want));
        CHECK(p.begin_ns > 0);
        CHECK(p.end_ns >= p.begin_ns);
        if (device->timestamps_inside_passes()) {
            REQUIRE(!p.launches.empty());
            for (std::size_t i = 0; i < p.launches.size(); ++i) {
                CHECK(p.launches[i].second >= p.begin_ns);
                CHECK(p.launches[i].second <= p.end_ns);
                if (i > 0) CHECK(p.launches[i].first > p.launches[i - 1].first);
            }
        } else {
            CHECK(p.launches.empty());
        }
    }
    // A prefix — none of the launches, then the first ten — runs and is timed.
    for (const std::uint32_t prefix : {0u, 10u}) {
        const Profiled p = profile(instance.get(), *r.program, step_of(ids, 0), prefix);
        REQUIRE_MESSAGE(p.error == kernels::ProgramError::Ok, p.message);
        CHECK(p.end_ns >= p.begin_ns);
    }
}

TEST_CASE("profiled steps pipeline, two in flight, reported in the order run") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get(), gpu::DiagnosticRequest{.timestamps = true});
    Running r = run_model(instance.get(), *device);
    const std::uint32_t all = static_cast<std::uint32_t>(r.launches_count);
    const auto ids = ids_of(8);
    run_step(instance.get(), *r.program, 8, ids, 0, true);
    Profiled first, second;
    kernels::Step a = step_of(std::span{ids}.first(1), 8);
    kernels::Step b = step_of(std::span{ids}.first(1), 9);
    r.program->run_profiled(a, all, on_profiled, &first);
    r.program->run_profiled(b, all, on_profiled, &second);
    pump_until(instance.get(), second.done, "two profiled steps");
    REQUIRE(first.done);
    REQUIRE(first.error == kernels::ProgramError::Ok);
    REQUIRE(second.error == kernels::ProgramError::Ok);
    CHECK(second.begin_ns >= first.begin_ns);
    CHECK(second.end_ns >= second.begin_ns);
}

TEST_CASE("a profiled step is refused beside an unprofiled one, and on a device without timestamps") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    {
        const auto device = acquire(instance.get(), gpu::DiagnosticRequest{.timestamps = true});
        Running r = run_model(instance.get(), *device);
        const std::uint32_t all = static_cast<std::uint32_t>(r.launches_count);
        const auto ids = ids_of(4);
        // An unprofiled step outstanding: a profiled one is refused at once.
        bool plain_done = false;
        r.program->run(step_of(ids, 0),
                       [](kernels::ProgramError, std::string_view, std::span<const std::byte>, void* ud) {
                           *static_cast<bool*>(ud) = true;
                       },
                       &plain_done);
        Profiled refused;
        r.program->run_profiled(step_of(ids, 0), all, on_profiled, &refused);
        CHECK(refused.done);
        CHECK(refused.error == kernels::ProgramError::Step);
        CHECK(refused.message == "a profiled step beside an unprofiled one");
        pump_until(instance.get(), plain_done, "the unprofiled step");
        // A profiled step outstanding: an unprofiled one is refused at once.
        Profiled outstanding;
        r.program->run_profiled(step_of(ids, 0), all, on_profiled, &outstanding);
        const StepOutcome beside = try_step(instance.get(), *r.program, step_of(ids, 0));
        CHECK(beside.error == kernels::ProgramError::Step);
        CHECK(beside.message == "a step beside a profiled one");
        pump_until(instance.get(), outstanding.done, "the profiled step");
        CHECK(outstanding.error == kernels::ProgramError::Ok);
    }
    // The harness's own device grants no timestamps.
    const auto device = acquire(instance.get());
    Running r = run_model(instance.get(), *device);
    const Profiled p = profile(instance.get(), *r.program, step_of(ids_of(4), 0), 1);
    CHECK(p.error == kernels::ProgramError::Step);
    CHECK(p.message == "the device did not grant timestamp queries");
}

#endif
