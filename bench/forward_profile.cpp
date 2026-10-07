// The forward pass's GPU time, launch by launch: make profile.
//
// A DIAGNOSTIC tool (core/diagnostics.h). It is built only in the native-diag
// preset, and its numbers say where a step's time goes; they are never
// quoted as throughput (TLM.6). A clean build's throughput is measured where
// the page runs.
//
// Why natively: there the GPU's timestamps are unrounded, to its counter's
// resolution — WebGPU does not report it, so the report states the
// smallest interval it observed, a bound on it — where Chrome rounds them
// to 100 µs,
// longer than most of a step's dispatches (WASM.12); and a step is timed
// launch by launch. What it measures is native evidence: Dawn here is the
// release emdawnwebgpu is built from, not Chrome's own Dawn and Tint, so the
// Metal they generate, and the time it takes, may differ. Its findings are
// relative — which launch takes the step's time, against what floor — and
// each is confirmed in Chrome by the step's whole time there, the page's
// prefill and decode rates, before a change is built on it. Both
// toolchains' versions head the report.
//
//     charlotte_profile_forward <model.gguf> [--csv <file>]
//
// It loads the model as the harness does — plan, upload, graph, program —
// on a device that granted timestamp queries (gpu/device.h's
// DiagnosticRequest), then times three workloads:
//   1. Decode steps at positions 0, 1,024 and 8,192, those within the
//      context offered, the cache first filled by prefill to that position.
//   2. Prefill steps of 1, 8, 16, 32, 64, 128 and 512 tokens at position 0:
//      the products change form between decode and prefill, and with the
//      tile a step's token count selects (kernels/matmul/matmul.h).
//   3. Pipelined decode: 128 whole steps fed on the GPU, two outstanding, as
//      the runtime runs them (runtime/runtime.h), each profiled, so the run
//      is timed as it runs: the GPU idled for the run's span on its own
//      clock — the last step's end less the first's beginning, less the
//      steps' times — and, apart, on the CPU's, the interval between
//      reports. WebGPU offers no calibrated pair of CPU and GPU timestamps,
//      so the two clocks are never aligned or subtracted (TLM.11). GPU.10's
//      first question, whether the GPU waits on the CPU, before any kernel's.
// A launch's time, in workloads 1 and 2, in the one pass a step runs in
// (kernels/program.h):
//   - where the device grants timestamps inside a pass, the step's own, a
//     launch at a time;
//   - otherwise, as on the target, a prefix's: the step run through its
//     first k launches takes T(k), and launch k's time is T(k) − T(k − 1),
//     what adding it costs the step in the shape it runs in. A prefix costs
//     a step's time up to it, so the prefixes run are those through the
//     embedding, through the first layer of each kind the graph builds —
//     Gemma 3's window and global layers are two — and through the output
//     block; every other layer repeats a first layer's launches over the
//     same shapes. The check that they do: the step's whole time less its
//     time through those first layers, over the layers left, against a
//     first layer's, reported, and a failure named past 10%.
// Each time is the median of 5 runs after 2 discarded warm-ups — the GPU
// clocks up under load, and a run waiting on the last would time it cold, so
// the runs go back to back, two outstanding — reported with the least and
// the most. The conditions head every report: the build, the adapter and its
// backend, Dawn's commit, the smallest interval observed, runs and warm-ups;
// make profile prints the model file's SHA-256 before it (WASM.11).
//
// Reported, for each step timed:
//   - Its whole GPU time, from its pass's two timestamps.
//   - By role and entry point (graph/graph.h): the launches, the median time
//     a launch, the total, and its share of the step; and, for a launch that
//     reads its bindings once each — a product's weights, a norm, the gather
//     — the bytes it binds over its time, against the target's 400 GB/s
//     (graph.h), so a launch far below the device's bandwidth is named.
//   - The floors the headers derive, beside the times measured: a decode
//     step's weights and cache read once, 0.94 ms + 0.29 µs × p for Qwen3;
//     a 512-token prefill step's arithmetic, about 32 ms at the M3 Max's
//     peak (graph.h); each kernel's own (its header).
//   - Every launch's time, by index, role, layer and entry point, to the CSV
//     file when one is named, so a change's effect is a diff of two files.
// What it finds is recorded in docs/research/, with the commit and the
// conditions, as the readback round trip's was.
//
// Verification, before any time is reported, each a failure named if it does
// not hold:
//   - A profiled step at either grain draws the same record and leaves the
//     same logits, bit for bit, as run() over the same step: the instrument
//     changes no result.
//   - A step's timestamps in order: its pass's end at or after its
//     beginning, and any timestamps inside it, one for each launch it
//     dispatched, in the graph's order, between the two.
//   - The repetition check above, on the target.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     Per.6  Don't make claims about performance without measurements — the
//            headers' floors are derived; this measures what the GPU spends
//            against them.
//   C++ performance guidelines
//     GPU.10 Profile with GPU timelines and counters before optimizing — GPU
//            timestamps in the step's own pass, each launch named, and the
//            idle share first.
//     TLM.6  Diagnostic mode is not benchmark mode — a diagnostic build's
//            tool, its findings confirmed in Chrome before they are built on.
//     WASM.12 Keep the compute core natively buildable so it can be
//            profiled — the same kernels, timed natively.
//     WASM.11 State the measurement conditions — they head every report.
//     TLM.11 GPU timestamps require calibration against the CPU's — none is
//            offered, so none is claimed.
//   C++ Core Guidelines
//     E.28   Avoid error handling based on global state — each check's
//            failure is returned, not set aside.

#include "core/diagnostics.h"

#if !BLLM_DIAGNOSTICS_ENABLED
#error "charlotte_profile_forward is a diagnostic tool: build it in the native-diag preset"
#endif

#include <webgpu/webgpu.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/gpu/device.h"
#include "core/kernels/program.h"
#include "core/policy/policy.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/sampler/sampler.h"

using namespace bllm;

namespace {

// The target's figures the headers derive against (graph/graph.h,
// kernels/matmul/matmul.h).
constexpr double kBandwidthBytesPerNs = 400.0;     // 400 GB/s
constexpr double kPeakFlopsPerNs = 14'000.0;       // about 14 f32 TFLOPS, a third-party figure
constexpr int kRuns = 5;
constexpr int kWarmUps = 2;
// A step longer than this is timed 3 times after 1 warm-up, so a prefix
// sweep of a 512-token prefill ends in minutes.
constexpr double kLongStepNs = 500e6;
constexpr std::uint64_t kUploadChunk = 16u << 20;

[[noreturn]] void die(const std::string& why) {
    std::fprintf(stderr, "charlotte_profile_forward: %s\n", why.c_str());
    std::exit(1);
}

void pump(WGPUInstance instance, const bool& done) {
    while (!done) wgpuInstanceProcessEvents(instance);
}

// --- Loading, as the harness does ------------------------------------------

struct Loaded {
    std::string bytes;
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    std::unique_ptr<kernels::Program> program;
    residency::BufferRange logits{};
};

std::unique_ptr<gpu::Device> acquire(WGPUInstance instance) {
    struct Got {
        std::unique_ptr<gpu::Device> device;
        std::string error;
        bool done = false;
    } got;
    gpu::Device::request(
        instance,
        [](std::unique_ptr<gpu::Device> device, const char* error, void* ud) {
            auto& g = *static_cast<Got*>(ud);
            g.device = std::move(device);
            if (error != nullptr) g.error = error;
            g.done = true;
        },
        &got, nullptr, gpu::DiagnosticRequest{.timestamps = true});
    pump(instance, got.done);
    if (got.device == nullptr) die("no device: " + got.error);
    return std::move(got.device);
}

std::unique_ptr<Loaded> load(WGPUInstance instance, const gpu::Device& device, const char* path) {
    auto l = std::make_unique<Loaded>();
    std::ifstream file(path, std::ios::binary);
    if (!file) die(std::string("cannot open ") + path);
    l->bytes.assign(std::istreambuf_iterator<char>(file), {});
    gguf::MemoryByteSource source{std::as_bytes(std::span{l->bytes}), l->bytes.size()};
    if (gguf::read_index(source, l->index).error != gguf::ReadError::Ok) die("the file's index does not read");
    const auto& granted = device.limits();
    const residency::DeviceLimits limits{granted.max_buffer_size, granted.max_storage_buffer_binding_size,
                                         granted.min_storage_buffer_offset_alignment};
    residency::ResidencyPlan plan;
    if (const std::string stop = preflight::plan_load(l->index, limits, policy::LoadPolicy{}, l->description, plan);
        !stop.empty()) {
        die("the model does not load: " + stop);
    }
    struct Ready {
        std::unique_ptr<residency::Upload> upload;
        std::string error;
        bool done = false;
    } ready;
    residency::Upload::begin(device, l->index, plan, l->bytes.size(), capability::find_format, {}, kUploadChunk,
                             [](std::unique_ptr<residency::Upload> u, residency::UploadError e, const char* subject,
                                void* ud) {
                                 auto& r = *static_cast<Ready*>(ud);
                                 r.upload = std::move(u);
                                 if (e != residency::UploadError::Ok) {
                                     r.error = std::string(residency::to_string(e)) + " " + (subject ? subject : "");
                                 }
                                 r.done = true;
                             },
                             &ready);
    pump(device.instance(), ready.done);
    if (ready.upload == nullptr) die("the buffers: " + ready.error);
    l->upload = std::move(ready.upload);
    struct Step {
        residency::UploadError error = residency::UploadError::Ok;
        bool done = false;
    };
    const auto upload_callback = [](residency::UploadError e, void* ud) {
        auto& st = *static_cast<Step*>(ud);
        st.error = e;
        st.done = true;
    };
    const auto all = std::as_bytes(std::span{l->bytes});
    for (std::uint64_t at = 0; at < all.size(); at += kUploadChunk) {
        Step accepted;
        l->upload->write(at, all.subspan(at, std::min<std::uint64_t>(kUploadChunk, all.size() - at)),
                         upload_callback, &accepted);
        pump(instance, accepted.done);
        if (accepted.error != residency::UploadError::Ok) die("a chunk was refused");
    }
    Step finished;
    l->upload->finish(upload_callback, &finished);
    pump(instance, finished.done);
    if (finished.error != residency::UploadError::Ok) die("the upload did not finish");

    const residency::ResidencyPlan& planned = l->upload->plan();
    std::string_view name;
    (void)l->index.read_string("general.architecture", name);
    const formats::Format* cache_format = capability::find_format(planned.cache_type);
    if (cache_format == nullptr) die("the cache's format is not one this build writes");
    const graph::GraphResult built =
        capability::find_architecture(name)->graph(l->description, planned, *cache_format, l->launches);
    if (!built.ok()) die("the graph does not build: " + built.subject);
    std::optional<kernels::Binding> sampled;
    for (const residency::PlannedScratch& sc : planned.scratch) {
        if (sc.purpose == "logits") l->logits = sc.range;
        if (sc.purpose == "sampled") {
            sampled = kernels::Binding{sc.range.buffer, sc.range.offset, sizeof(sampler::SampledRecord)};
        }
    }
    struct Built {
        std::unique_ptr<kernels::Program> program;
        std::string error;
        bool done = false;
    } b;
    kernels::Program::build(*l->upload, l->launches, sampled,
                            [](std::unique_ptr<kernels::Program> p, kernels::ProgramError, std::string_view message,
                               void* ud) {
                                auto& x = *static_cast<Built*>(ud);
                                x.program = std::move(p);
                                x.error = std::string(message);
                                x.done = true;
                            },
                            &b);
    pump(instance, b.done);
    if (b.program == nullptr) die("the program did not build: " + b.error);
    l->program = std::move(b.program);
    return l;
}

// --- Running and timing -----------------------------------------------------

kernels::Step step_at(std::uint32_t position, std::uint32_t tokens) {
    kernels::Step step{};
    step.position = position;
    step.tokens = tokens;
    step.logits = 1;
    for (std::uint32_t i = 0; i < tokens; ++i) step.ids[i] = (i * 37 + 11) % 1000;
    return step;
}

void run(WGPUInstance instance, kernels::Program& program, const kernels::Step& step) {
    struct Ran {
        bool ok = false;
        bool done = false;
    } ran;
    program.run(step,
                [](kernels::ProgramError e, std::string_view, std::span<const std::byte>, void* ud) {
                    auto& r = *static_cast<Ran*>(ud);
                    r.ok = e == kernels::ProgramError::Ok;
                    r.done = true;
                },
                &ran);
    pump(instance, ran.done);
    if (!ran.ok) die("a step failed");
}

struct Profiled {
    bool ok = false;
    bool done = false;
    std::uint64_t ns = 0;
    std::vector<std::byte> readback;
};

Profiled profile_once(WGPUInstance instance, kernels::Program& program, const kernels::Step& step,
                      std::uint32_t launches) {
    Profiled p;
    program.run_profiled(step, launches,
                         [](kernels::ProgramError e, std::string_view message, std::span<const std::byte> readback,
                            const kernels::Program::Timestamps& t, void* ud) {
                             auto& q = *static_cast<Profiled*>(ud);
                             q.ok = e == kernels::ProgramError::Ok;
                             if (!q.ok) std::fprintf(stderr, "profiled step: %.*s\n", int(message.size()), message.data());
                             if (q.ok && t.end_ns < t.begin_ns) die("a pass ended before it began");
                             q.ns = t.end_ns - t.begin_ns;
                             q.readback.assign(readback.begin(), readback.end());
                             q.done = true;
                         },
                         &p);
    pump(instance, p.done);
    if (!p.ok) die("a profiled step failed");
    return p;
}

struct Timed {
    double median_ns, least_ns, most_ns;
};

// The step, or its first `launches` launches, run warm-up and timed runs
// back to back, two outstanding, as the runtime runs steps: a GPU left idle
// between runs clocks down, and each run would then time a cold GPU.
Timed time(WGPUInstance instance, kernels::Program& program, const kernels::Step& step, std::uint32_t launches,
           bool long_step) {
    struct Runs {
        kernels::Program* program;
        const kernels::Step* step;
        std::uint32_t launches;
        int total, submitted = 0, reported = 0;
        std::vector<double> ns;
        bool failed = false;
    } r{&program, &step, launches, (long_step ? 1 : kWarmUps) + (long_step ? 3 : kRuns)};
    const int warm = long_step ? 1 : kWarmUps;
    const auto submit = [](Runs& q) {
        ++q.submitted;
        q.program->run_profiled(*q.step, q.launches,
                                [](kernels::ProgramError e, std::string_view, std::span<const std::byte>,
                                   const kernels::Program::Timestamps& t, void* ud) {
                                    auto& x = *static_cast<Runs*>(ud);
                                    ++x.reported;
                                    if (e != kernels::ProgramError::Ok || t.end_ns < t.begin_ns) {
                                        x.failed = true;
                                        return;
                                    }
                                    x.ns.push_back(static_cast<double>(t.end_ns - t.begin_ns));
                                },
                                &q);
    };
    submit(r);
    submit(r);
    while (r.reported < r.total && !r.failed) {
        const int before = r.reported;
        wgpuInstanceProcessEvents(instance);
        for (int i = before; i < r.reported && r.submitted < r.total; ++i) submit(r);
    }
    if (r.failed) die("a timed step failed");
    std::vector<double> ns(r.ns.begin() + warm, r.ns.end());
    std::sort(ns.begin(), ns.end());
    return {ns[ns.size() / 2], ns.front(), ns.back()};
}

bool dispatched(const kernels::Launch& l, const kernels::Step& step) {
    return kernels::workgroups_for({l.rows, l.invocations_per_row, l.rows_per_tile, l.key_split, l.window, l.tokens},
                                   l.workgroup_size, step.position, step.tokens, step.logits != 0) > 0;
}

// The weights a launch binds — read once each by a product, the gather or a
// norm — not its working buffers, which are sized for a whole prefill block
// whatever the step uses of them.
std::uint64_t weight_bytes_of(const kernels::Launch& l, const residency::ResidencyPlan& plan) {
    std::uint64_t bytes = 0;
    for (const kernels::Binding& b : l.bindings) {
        if (plan.buffers[static_cast<std::size_t>(b.buffer)].pool == residency::Pool::Weights) bytes += b.size;
    }
    return bytes;
}

std::vector<float> read_logits(WGPUInstance instance, const gpu::Device& device, const Loaded& l) {
    const std::uint64_t bytes = std::uint64_t{l.description.vocabulary_size} * 4;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    desc.size = bytes;
    const gpu::Buffer target(wgpuDeviceCreateBuffer(device.handle(), &desc));
    const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(device.handle(), nullptr));
    wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), l.upload->buffer(l.logits.buffer), l.logits.offset,
                                         target.get(), 0, bytes);
    const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
    WGPUCommandBuffer raw = commands.get();
    wgpuQueueSubmit(device.queue(), 1, &raw);
    bool done = false;
    WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* ud, void*) { *static_cast<bool*>(ud) = true; };
    info.userdata1 = &done;
    wgpuBufferMapAsync(target.get(), WGPUMapMode_Read, 0, bytes, info);
    pump(instance, done);
    std::vector<float> out(l.description.vocabulary_size);
    std::memcpy(out.data(), wgpuBufferGetConstMappedRange(target.get(), 0, bytes), bytes);
    wgpuBufferUnmap(target.get());
    return out;
}

// --- One step's profile ------------------------------------------------------

struct Row {
    std::string workload;
    std::uint32_t launch;
    std::string_view role;
    std::uint32_t layer;
    std::string_view entry;
    double ns;
};

// What a run observed, for the report's conditions.
struct Observed {
    std::uint64_t smallest_ns = ~std::uint64_t{0};   // the smallest nonzero interval
};

// Profiles one step; false when its repetition check fails.
bool profile_step(WGPUInstance instance, const gpu::Device& device, Loaded& l, const std::string& workload,
                  const kernels::Step& step, std::vector<Row>& csv, Observed& observed) {
    kernels::Program& program = *l.program;
    const auto all = static_cast<std::uint32_t>(l.launches.size());

    // The instrument changes no result: run() and a profiled step, the same
    // record and the same logits.
    run(instance, program, step);
    const std::vector<float> plain = read_logits(instance, device, l);
    const Profiled once = profile_once(instance, program, step, all);
    const std::vector<float> profiled = read_logits(instance, device, l);
    if (std::memcmp(plain.data(), profiled.data(), plain.size() * 4) != 0) {
        die(workload + ": a profiled step left logits run() did not");
    }

    const Timed whole = time(instance, program, step, all, false);
    const bool long_step = whole.median_ns > kLongStepNs;

    // The layers' kinds, by attention window, and the first of each.
    std::map<std::uint32_t, std::uint32_t> first_of_kind;   // window -> first layer
    std::vector<std::uint32_t> kind_of(l.description.layers.size());
    for (std::uint32_t i = 0; i < l.description.layers.size(); ++i) {
        const std::uint32_t window = l.description.layers[i].attention_window;
        kind_of[i] = window;
        first_of_kind.emplace(window, i);
    }
    std::set<std::uint32_t> firsts;
    for (const auto& [window, layer] : first_of_kind) firsts.insert(layer);

    // The launches timed by prefix: those of the embedding, the first layer
    // of each kind, and the output block, that this step dispatches.
    std::vector<std::uint32_t> chosen;
    std::uint32_t embed_end = 0, output_start = all;
    for (std::uint32_t i = 0; i < all; ++i) {
        const kernels::Launch& x = l.launches[i];
        if (x.role == "embed") embed_end = i + 1;
        if (x.role.starts_with("output.") && output_start == all) output_start = i;
        if (!dispatched(x, step)) continue;
        if (x.layer == kernels::kNoLayer || firsts.contains(x.layer)) chosen.push_back(i);
    }
    std::map<std::uint32_t, double> prefix;   // limit -> median ns
    const auto prefix_ns = [&](std::uint32_t limit) {
        if (auto it = prefix.find(limit); it != prefix.end()) return it->second;
        const double ns = limit == 0 ? 0.0 : time(instance, program, step, limit, long_step).median_ns;
        prefix[limit] = ns;
        return ns;
    };
    std::fprintf(stderr, "%s: %zu launches timed by prefix\n", workload.c_str(), chosen.size());
    std::map<std::uint32_t, double> marginal;   // launch -> ns
    for (const std::uint32_t i : chosen) marginal[i] = prefix_ns(i + 1) - prefix_ns(i);

    // Repetition check: the layers' time, against their first layers'.
    const double layers_ns = prefix_ns(output_start) - prefix_ns(embed_end);
    std::map<std::uint32_t, double> first_layer_ns;   // layer -> its launches' sum
    for (const auto& [i, ns] : marginal) {
        if (l.launches[i].layer != kernels::kNoLayer) first_layer_ns[l.launches[i].layer] += ns;
    }
    double predicted = 0;
    for (std::uint32_t layer = 0; layer < kind_of.size(); ++layer) {
        predicted += first_layer_ns[first_of_kind[kind_of[layer]]];
    }
    const double off = predicted > 0 ? std::abs(layers_ns - predicted) / predicted : 1.0;

    std::printf("\n== %s: position %u, %u tokens\n", workload.c_str(), step.position, step.tokens);
    std::printf("   step %.3f ms (least %.3f, most %.3f)\n", whole.median_ns / 1e6, whole.least_ns / 1e6,
                whole.most_ns / 1e6);
    std::printf("   layers %.3f ms, from their first layers %.3f ms: %.1f%% apart%s\n", layers_ns / 1e6,
                predicted / 1e6, off * 100, off > 0.10 ? "  FAILED: over 10%" : "");

    // By role and entry point, each first layer's launches standing for the
    // layers of its kind.
    struct Group {
        int launches = 0;
        double ns = 0;
        std::uint64_t bytes = 0;
    };
    std::map<std::string, Group> groups;
    for (const auto& [i, ns] : marginal) {
        const kernels::Launch& x = l.launches[i];
        std::uint32_t copies = 1;
        if (x.layer != kernels::kNoLayer) {
            copies = static_cast<std::uint32_t>(
                std::count(kind_of.begin(), kind_of.end(), kind_of[x.layer]));
        }
        Group& g = groups[std::string(x.role) + " / " + std::string(x.entry_point)];
        g.launches += static_cast<int>(copies);
        g.ns += ns * copies;
        g.bytes += weight_bytes_of(x, l.upload->plan()) * copies;
        csv.push_back({workload, i, x.role, x.layer, x.entry_point, ns});
    }
    std::vector<std::pair<std::string, Group>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    std::printf("   %-32s %8s %12s %12s %7s %12s\n", "role / entry", "launches", "ms a launch", "ms total", "share",
                "weights GB/s");
    for (const auto& [name, g] : sorted) {
        std::printf("   %-32s %8d %12.4f %12.3f %6.1f%% %12.1f\n", name.c_str(), g.launches,
                    g.ns / g.launches / 1e6, g.ns / 1e6, 100 * g.ns / whole.median_ns,
                    g.ns > 0 ? static_cast<double>(g.bytes) / g.ns : 0.0);
    }

    // The headers' floors.
    std::uint64_t weight_bytes = 0;
    for (const residency::PlannedBuffer& b : l.upload->plan().buffers) {
        if (b.pool == residency::Pool::Weights) weight_bytes += b.size;
    }
    double cache_bytes = 0;
    for (const model::LayerDescription& layer : l.description.layers) {
        const std::uint32_t keys =
            std::min(step.position + step.tokens, layer.attention_window == 0 ? ~0u : layer.attention_window);
        cache_bytes += double(keys) * layer.key_value_heads * layer.head_dimension * 2 * 2;   // keys, values, F16
    }
    double weight_elements = 0;
    for (const gguf::TensorEntry& t : l.index.tensors()) {
        if (t.dimension_count == 2) weight_elements += double(t.element_count);
    }
    std::printf("   floors: weights and cache read once %.3f ms; products' arithmetic at peak %.3f ms\n",
                (double(weight_bytes) + cache_bytes) / kBandwidthBytesPerNs / 1e6,
                2 * weight_elements * step.tokens / kPeakFlopsPerNs / 1e6);
    if (once.ns > 0) observed.smallest_ns = std::min(observed.smallest_ns, once.ns);
    for (const auto& [limit, ns] : prefix) {
        if (ns > 0) observed.smallest_ns = std::min<std::uint64_t>(observed.smallest_ns, static_cast<std::uint64_t>(ns));
    }
    return off <= 0.10;
}

// --- Pipelined decode ---------------------------------------------------------

void profile_pipeline(WGPUInstance instance, Loaded& l) {
    kernels::Program& program = *l.program;
    const auto all = static_cast<std::uint32_t>(l.launches.size());
    constexpr std::uint32_t kPrompt = 64, kSteps = 128;
    run(instance, program, step_at(0, kPrompt));
    struct Pipeline {
        kernels::Program* program;
        std::uint32_t all;
        std::uint32_t submitted = 0, reported = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> gpu;
        std::vector<double> cpu_ms;   // when each report arrived
        std::chrono::steady_clock::time_point origin;
        bool failed = false;
    } p{&program, all};
    p.origin = std::chrono::steady_clock::now();
    const auto submit = [](Pipeline& q) {
        kernels::Step step{};
        step.position = kPrompt + q.submitted;
        step.tokens = 1;
        step.logits = 1;
        step.fed = 1;
        ++q.submitted;
        q.program->run_profiled(
            step, q.all,
            [](kernels::ProgramError e, std::string_view, std::span<const std::byte>,
               const kernels::Program::Timestamps& t, void* ud) {
                auto& r = *static_cast<Pipeline*>(ud);
                ++r.reported;
                if (e != kernels::ProgramError::Ok) {
                    r.failed = true;
                    return;
                }
                r.gpu.emplace_back(t.begin_ns, t.end_ns);
                r.cpu_ms.push_back(
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r.origin).count());
            },
            &q);
    };
    submit(p);
    submit(p);
    while (p.reported < kSteps && !p.failed) {
        const std::uint32_t before = p.reported;
        wgpuInstanceProcessEvents(instance);
        for (std::uint32_t i = before; i < p.reported && p.submitted < kSteps; ++i) submit(p);
    }
    if (p.failed) die("a pipelined step failed");
    double busy = 0;
    for (const auto& [b, e] : p.gpu) busy += double(e - b);
    const double span = double(p.gpu.back().second - p.gpu.front().first);
    std::printf("\n== pipelined decode: %u steps fed on the GPU, two outstanding, from position %u\n", kSteps,
                kPrompt);
    std::printf("   GPU: span %.3f ms, busy %.3f ms, idle %.1f%%; %.3f ms a step busy\n", span / 1e6, busy / 1e6,
                100 * (span - busy) / span, busy / kSteps / 1e6);
    std::printf("   CPU: a report every %.3f ms (first to last, %u reports)\n",
                (p.cpu_ms.back() - p.cpu_ms.front()) / (kSteps - 1), kSteps);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("usage: charlotte_profile_forward <model.gguf> [--csv <file>]");
    const char* csv_path = argc >= 4 && std::string_view(argv[2]) == "--csv" ? argv[3] : nullptr;

    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    auto loaded = load(instance.get(), *device, argv[1]);
    const auto& info = device->adapter_info();
    std::printf("DIAGNOSTIC — charlotte_profile_forward: build native-diag (Release, diagnostics on), not a "
                "throughput figure (TLM.6)\n");
    std::printf("adapter: %s %s (%s); Dawn %s; timestamps inside a pass: %s\n", info.backend.c_str(),
                info.description.c_str(), info.vendor.c_str(), BLLM_DAWN_COMMIT,
                device->timestamps_inside_passes() ? "yes" : "no — launches timed by prefix");
    std::printf("model: %s, %zu bytes; context offered %u; runs %d after %d warm-ups (3 after 1 past %.0f ms)\n",
                argv[1], loaded->bytes.size(), loaded->upload->plan().context_offered, kRuns, kWarmUps,
                kLongStepNs / 1e6);

    std::vector<Row> csv;
    Observed observed;
    bool checks_held = true;
    const std::uint32_t context = loaded->upload->plan().context_offered;
    for (const std::uint32_t position : {0u, 1024u, 8192u}) {
        if (position + 1 > context) continue;
        // The cache filled to `position` by prefill, unprofiled.
        for (std::uint32_t at = 0; at < position; at += 512) {
            run(instance.get(), *loaded->program, step_at(at, std::min(512u, position - at)));
        }
        kernels::Step decode = step_at(position, 1);
        checks_held = profile_step(instance.get(), *device, *loaded, "decode@" + std::to_string(position), decode, csv,
                                   observed) && checks_held;
    }
    for (const std::uint32_t tokens : {1u, 8u, 16u, 32u, 64u, 128u, 512u}) {
        checks_held = profile_step(instance.get(), *device, *loaded, "prefill" + std::to_string(tokens),
                                   step_at(0, tokens), csv, observed) && checks_held;
    }
    profile_pipeline(instance.get(), *loaded);

    if (csv_path != nullptr) {
        std::FILE* out = std::fopen(csv_path, "w");
        if (out == nullptr) die(std::string("cannot write ") + csv_path);
        std::fprintf(out, "workload,launch,role,layer,entry,ns\n");
        for (const Row& r : csv) {
            std::fprintf(out, "%s,%u,%.*s,%d,%.*s,%.0f\n", r.workload.c_str(), r.launch, int(r.role.size()),
                         r.role.data(), r.layer == kernels::kNoLayer ? -1 : int(r.layer), int(r.entry.size()),
                         r.entry.data(), r.ns);
        }
        std::fclose(out);
    }
    std::printf("\nsmallest interval observed: %llu ns, a bound on the timestamps' resolution\n",
                static_cast<unsigned long long>(observed.smallest_ns));
    if (!checks_held) {
        std::fprintf(stderr, "charlotte_profile_forward: a repetition check failed; see FAILED above\n");
        return 2;
    }
    return 0;
}
