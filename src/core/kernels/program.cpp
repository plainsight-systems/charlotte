#include "core/kernels/program.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "bllm/shaders_generated.h"
#include "core/gpu/callback_mode.h"
#include "core/gpu/userdata.h"
#include "core/kernels/order.h"
#include "core/kernels/schedule.h"

namespace bllm::kernels {

// What every callback of one Program shares: its own references, so a
// callback that lands after the Program is gone still has what it touches
// (R.20, R.21).
struct Program::State {
    gpu::Instance instance;
    gpu::DeviceHandle device;
    gpu::Queue queue;
    std::vector<gpu::ComputePipeline> pipelines;
    struct Bound {
        std::size_t pipeline;
        gpu::BindGroup group;
        Geometry geometry;
        std::uint32_t workgroup_size;
    };
    std::vector<Bound> launches;
    // The launches each step can dispatch, planned once the launches are
    // bound (kernels/schedule.h).
    Schedules schedules{std::span<const Geometry>{}};
    gpu::Buffer constants;   // UNIFORM | COPY_DST: every launch's, on kLaunchConstantsAlignment
    gpu::Buffer step;        // UNIFORM | COPY_DST: sizeof(Step)
    std::uint32_t max_workgroups = 0;
    bool cancelled = false;

    // What a step reads back, and the two slots steps alternate between,
    // MAP_READ | COPY_DST, made at build; none where the graph reads nothing.
    struct Source {
        WGPUBuffer buffer;   // borrowed: the upload outlives the program
        std::uint64_t offset;
        std::uint64_t size;
    };
    std::optional<Source> readback;
    std::array<gpu::Buffer, 2> slots;

    // A step outstanding: at most two (program.h), step n in steps[n % 2].
    // Its callbacks carry this State and its own record as userdata, and
    // `in_flight` keeps the State alive until the last outstanding step is
    // reported, so a step allocates nothing (MEM.9).
    struct Outstanding {
        std::uint64_t number = 0;  // its place in the order steps ran
        std::size_t pending = 0;   // callbacks still to land
        ProgramError error = ProgramError::Ok;
        // The first failure's message, reserved at build and truncated to
        // that capacity, so reporting a failure allocates nothing either.
        std::string message;
        bool reads_back = false;
        std::array<std::byte, kMaxReadback> bytes{};
        StepCallback done = nullptr;
        void* userdata = nullptr;
    };
    std::array<Outstanding, Order::kOutstanding> steps;
    Order order;                      // reports steps in the order they ran (order.h)
    std::shared_ptr<State> in_flight;
};

namespace {

using gpu::hand_off;
using gpu::take_back;

std::string to_string(WGPUStringView view) {
    if (view.data == nullptr) return {};
    return view.length == WGPU_STRLEN ? std::string(view.data) : std::string(view.data, view.length);
}

WGPUStringView view_of(std::string_view s) { return WGPUStringView{s.data(), s.size()}; }

ProgramError from_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, ProgramError failure) {
    if (status == WGPUPopErrorScopeStatus_CallbackCancelled) return ProgramError::Cancelled;
    if (status != WGPUPopErrorScopeStatus_Success) return failure;
    return type == WGPUErrorType_NoError ? ProgramError::Ok : failure;
}

ProgramError from_work_done(WGPUQueueWorkDoneStatus status) {
    switch (status) {
        case WGPUQueueWorkDoneStatus_Success: return ProgramError::Ok;
        case WGPUQueueWorkDoneStatus_CallbackCancelled: return ProgramError::Cancelled;
        default: return ProgramError::Step;
    }
}

constexpr std::array<WGPUErrorFilter, 3> kBuildScopes = {WGPUErrorFilter_OutOfMemory, WGPUErrorFilter_Validation,
                                                         WGPUErrorFilter_Internal};
constexpr std::array<WGPUErrorFilter, 2> kStepScopes = {WGPUErrorFilter_Validation, WGPUErrorFilter_Internal};
// Room for WebGPU's message on a failed step; a longer one is cut there. Dawn's
// validation messages run to a few hundred bytes.
constexpr std::size_t kStepMessageCapacity = 1024;

template <std::size_t N>
void push_scopes(WGPUDevice device, const std::array<WGPUErrorFilter, N>& filters) {
    for (const WGPUErrorFilter f : filters) wgpuDevicePushErrorScope(device, f);
}

// Pops `count` scopes; job->settled(error, message) runs once for each.
template <typename Job>
void pop_scopes(WGPUDevice device, std::size_t count, const std::shared_ptr<Job>& job, ProgramError failure) {
    struct Pop {
        std::shared_ptr<Job> job;
        ProgramError failure;
    };
    for (std::size_t i = 0; i < count; ++i) {
        WGPUPopErrorScopeCallbackInfo info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message,
                           void* userdata, void*) {
            const auto pop = take_back<Pop>(userdata);
            pop->job->settled(from_scope(status, type, pop->failure), to_string(message));
        };
        info.userdata1 = hand_off(std::make_unique<Pop>(Pop{job, failure}));
        wgpuDevicePopErrorScope(device, info);
    }
}

// A bound range of one of upload's buffers.
struct Resolved {
    WGPUBuffer buffer;   // borrowed: the upload outlives the program
    std::uint64_t offset;
    std::uint64_t size;
};

// Building, from the first section to its last callback. Section one makes
// the buffers and requests the pipelines; once every pipeline and its scopes
// have settled, section two makes the bind groups; once its scopes have
// settled, the program is handed over.
struct Build : std::enable_shared_from_this<Build> {
    std::shared_ptr<Program::State> state;
    std::unique_ptr<Program> program;              // handed over once built
    std::vector<Launch> launches;
    std::vector<std::vector<Resolved>> bindings;   // for each launch
    std::vector<std::size_t> pipeline_of;          // for each launch
    std::vector<std::uint64_t> constants_at;       // for each launch: its slots' offset
    std::vector<std::uint64_t> constants_size;     // and their bytes, a whole number of slots
    std::size_t pending = 0;                       // callbacks still to land in this section
    bool bound = false;                            // section two has run
    ProgramError error = ProgramError::Ok;
    std::string message;
    BuildCallback done;
    void* userdata;

    // Every failure's message is kept, not only the first: a shader that
    // fails to compile fails its pipeline too, and the pipeline's message,
    // which may land first, only says the module was invalid.
    void settled(ProgramError e, std::string m) {
        if (e != ProgramError::Ok) {
            if (error == ProgramError::Ok) error = e;
            if (!m.empty()) message += (message.empty() ? "" : "\n") + m;
        }
        if (--pending == 0) next();
    }

    void next();
    void bind();
};

void Build::bind() {
    Program::State& s = *state;
    WGPUDevice device = s.device.get();
    bound = true;
    pending = kBuildScopes.size();
    push_scopes(device, kBuildScopes);
    // A layout is the pipeline's, not the launch's: asked for once a
    // pipeline, about a dozen calls rather than one a launch.
    std::vector<gpu::BindGroupLayout> layouts;
    layouts.reserve(s.pipelines.size());
    for (const gpu::ComputePipeline& pipeline : s.pipelines) {
        layouts.emplace_back(wgpuComputePipelineGetBindGroupLayout(pipeline.get(), kBindGroup));
    }
    for (std::size_t i = 0; i < launches.size(); ++i) {
        const std::size_t p = pipeline_of[i];
        const gpu::BindGroupLayout& layout = layouts[p];
        std::vector<WGPUBindGroupEntry> entries;
        WGPUBindGroupEntry step = WGPU_BIND_GROUP_ENTRY_INIT;
        step.binding = kStepBinding;
        step.buffer = s.step.get();
        step.size = sizeof(Step);
        entries.push_back(step);
        WGPUBindGroupEntry constants = WGPU_BIND_GROUP_ENTRY_INIT;
        constants.binding = kLaunchBinding;
        constants.buffer = s.constants.get();
        constants.offset = constants_at[i];
        constants.size = constants_size[i];
        entries.push_back(constants);
        for (std::size_t k = 0; k < bindings[i].size(); ++k) {
            WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
            e.binding = kFirstWeightBinding + static_cast<std::uint32_t>(k);
            e.buffer = bindings[i][k].buffer;
            e.offset = bindings[i][k].offset;
            e.size = bindings[i][k].size;
            entries.push_back(e);
        }
        WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        desc.layout = layout.get();
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        s.launches.push_back({p, gpu::BindGroup(wgpuDeviceCreateBindGroup(device, &desc)),
                              Geometry{launches[i].rows, launches[i].invocations_per_row, launches[i].rows_per_tile,
                                       launches[i].key_split, launches[i].window, launches[i].tokens},
                              launches[i].workgroup_size});
    }
    std::vector<Geometry> geometries;
    geometries.reserve(s.launches.size());
    for (const auto& b : s.launches) geometries.push_back(b.geometry);
    s.schedules = Schedules(geometries);
    pop_scopes(device, kBuildScopes.size(), shared_from_this(), ProgramError::Build);
}

void Build::next() {
    if (error != ProgramError::Ok) {
        done(nullptr, error, message, userdata);
    } else if (!bound) {
        bind();
    } else {
        done(std::move(program), ProgramError::Ok, {}, userdata);
    }
}

// A pipeline's override constants, in name order, the program's own among
// them: workgroup_size always, last_token for a launch of the last token.
using Constants = std::vector<std::pair<std::string_view, double>>;

Constants constants_of(const Launch& launch) {
    Constants c;
    c.emplace_back("workgroup_size", launch.workgroup_size);
    if (launch.rows == Rows::LastToken) c.emplace_back("last_token", 1.0);
    for (const Override& o : launch.overrides) c.emplace_back(o.name, o.value);
    std::sort(c.begin(), c.end());
    return c;
}

// A distinct pipeline: the kernel's text, the format it unpacks, the format
// it packs, its constants. The text is compared, not its address: an
// embedded string may lie at a different address in each translation unit
// that names it.
using Key = std::tuple<std::string_view, std::string_view, const formats::Format*, const formats::Format*, Constants>;

Key key_of(const Launch& launch) {
    return {launch.kernel, launch.entry_point, launch.format, launch.pack_format, constants_of(launch)};
}

// The bytes a launch's constants take in the constants buffer: whole
// slots, at least one, since binding 1 is bound whatever a kernel declares.
std::uint64_t slot_bytes(const Launch& launch) {
    const std::uint64_t size = std::max<std::uint64_t>(launch.constants.size(), 1);
    return (size + kLaunchConstantsAlignment - 1) / kLaunchConstantsAlignment * kLaunchConstantsAlignment;
}

}  // namespace

void Program::build(const residency::Upload& upload, std::vector<Launch> launches,
                    std::optional<Binding> readback, BuildCallback done, void* userdata) {
    auto build = std::make_shared<Build>();
    build->launches = std::move(launches);
    build->done = done;
    build->userdata = userdata;

    // What the descriptions promise, checked before any GPU call: a refusal
    // here names the launch at fault.
    for (std::size_t i = 0; i < build->launches.size(); ++i) {
        const Launch& launch = build->launches[i];
        const std::string at = "launch " + std::to_string(i) + ": ";
        if (launch.constants.size() > kMaxLaunchConstants) {
            done(nullptr, ProgramError::Build, at + "constants larger than " + std::to_string(kMaxLaunchConstants),
                 userdata);
            return;
        }
        if (launch.pack_format != nullptr && launch.pack_format->pack_wgsl().empty()) {
            done(nullptr, ProgramError::Build, at + "its pack format has no pack", userdata);
            return;
        }
        if (launch.workgroup_size == 0 || launch.invocations_per_row == 0) {
            done(nullptr, ProgramError::Build, at + "no invocations", userdata);
            return;
        }
        if (launch.rows_per_tile != 0 &&
            (launch.rows == Rows::LastToken ||
             std::uint64_t{launch.rows_per_tile} * launch.invocations_per_row % launch.workgroup_size != 0)) {
            done(nullptr, ProgramError::Build, at + "a tile is not a whole number of workgroups of every token",
                 userdata);
            return;
        }
        if (launch.key_split != KeySplit::None && launch.window == 0) {
            done(nullptr, ProgramError::Build, at + "split by key chunks with no window", userdata);
            return;
        }
        for (std::size_t o = 0; o < launch.overrides.size(); ++o) {
            const std::string_view name = launch.overrides[o].name;
            const bool repeated = std::any_of(launch.overrides.begin(), launch.overrides.begin() + o,
                                              [&](const Override& earlier) { return earlier.name == name; });
            if (name == "workgroup_size" || name == "last_token" || repeated) {
                done(nullptr, ProgramError::Build,
                     at + "override " + std::string(name) + (repeated ? " given twice" : " is the program's to set"),
                     userdata);
                return;
            }
        }
        std::vector<Resolved> resolved;
        for (const Binding& b : launch.bindings) {
            WGPUBuffer buffer = upload.buffer(b.buffer);
            if (buffer == nullptr) {
                done(nullptr, ProgramError::Build,
                     at + "binds buffer " + std::to_string(static_cast<std::uint32_t>(b.buffer)) +
                         ", which upload did not create",
                     userdata);
                return;
            }
            resolved.push_back({buffer, b.offset, b.size});
        }
        build->bindings.push_back(std::move(resolved));
    }

    auto state = std::make_shared<State>();
    state->instance = gpu::retain(upload.instance());
    state->device = gpu::retain(upload.device());
    WGPUDevice device = state->device.get();
    state->queue = gpu::Queue(wgpuDeviceGetQueue(device));
    WGPULimits limits = WGPU_LIMITS_INIT;
    wgpuDeviceGetLimits(device, &limits);
    state->max_workgroups = limits.maxComputeWorkgroupsPerDimension;
    for (State::Outstanding& o : state->steps) o.message.reserve(kStepMessageCapacity);
    if (readback) {
        WGPUBuffer buffer = upload.buffer(readback->buffer);
        if (buffer == nullptr || readback->size == 0 || readback->size > kMaxReadback || readback->size % 4 != 0) {
            done(nullptr, ProgramError::Build, "the readback range is not a buffer upload created, of 4 to 16 bytes",
                 userdata);
            return;
        }
        state->readback = State::Source{buffer, readback->offset, readback->size};
    }
    build->state = state;
    build->program = std::unique_ptr<Program>(new Program());
    build->program->state_ = state;

    // Each launch's constants at the offset its slots begin, worked out once
    // (MEM.11).
    std::uint64_t constants_bytes = 0;
    for (const Launch& launch : build->launches) {
        build->constants_at.push_back(constants_bytes);
        build->constants_size.push_back(slot_bytes(launch));
        constants_bytes += build->constants_size.back();
    }

    // Section one: the buffers, the constants written, the pipelines asked for.
    push_scopes(device, kBuildScopes);
    WGPUBufferDescriptor constants_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    constants_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    constants_desc.size = std::max<std::uint64_t>(constants_bytes, kLaunchConstantsAlignment);
    state->constants = gpu::Buffer(wgpuDeviceCreateBuffer(device, &constants_desc));
    WGPUBufferDescriptor step_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    step_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    step_desc.size = sizeof(Step);
    state->step = gpu::Buffer(wgpuDeviceCreateBuffer(device, &step_desc));
    if (state->readback) {
        WGPUBufferDescriptor slot_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        slot_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        slot_desc.size = state->readback->size;
        for (gpu::Buffer& slot : state->slots) slot = gpu::Buffer(wgpuDeviceCreateBuffer(device, &slot_desc));
    }

    // Optimization (browser): every launch's constants in one write (WASM.2).
    std::vector<std::byte> packed(constants_desc.size);
    for (std::size_t i = 0; i < build->launches.size(); ++i) {
        const auto& c = build->launches[i].constants;
        std::copy(c.begin(), c.end(), packed.begin() + static_cast<std::ptrdiff_t>(build->constants_at[i]));
    }
    wgpuQueueWriteBuffer(state->queue.get(), state->constants.get(), 0, packed.data(), packed.size());

    // Each distinct pipeline's index and the first launch that asks for it,
    // found in one pass over the launches: one key a launch.
    struct Distinct {
        std::size_t index;
        std::size_t first_launch;
    };
    std::map<Key, Distinct> distinct;
    for (std::size_t i = 0; i < build->launches.size(); ++i) {
        const auto [it, added] = distinct.try_emplace(key_of(build->launches[i]), Distinct{distinct.size(), i});
        build->pipeline_of.push_back(it->second.index);
    }
    state->pipelines.resize(distinct.size());
    build->pending = distinct.size() + kBuildScopes.size();

    struct Requested {
        std::shared_ptr<Build> build;
        std::size_t index;
    };
    for (const auto& [key, found] : distinct) {
        const std::size_t index = found.index;
        const Launch& launch = build->launches[found.first_launch];
        // Optimization (practice): each distinct kernel is composed and
        // compiled once, whatever its launches, and every pipeline is asked
        // for at once, so the browser compiles them together (WASM.7).
        // The step's declaration first, binding 0 of every kernel
        // (step.wgsl), then the kernel, then the formats it composes with.
        std::string source(shaders::step);
        source += '\n';
        source += launch.kernel;
        if (launch.format != nullptr) {
            source += '\n';
            source += launch.format->unpack_wgsl();
        }
        if (launch.pack_format != nullptr) {
            source += '\n';
            source += launch.pack_format->pack_wgsl();
        }
        WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgsl.code = view_of(source);
        WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        module_desc.nextInChain = &wgsl.chain;
        const gpu::ShaderModule module(wgpuDeviceCreateShaderModule(device, &module_desc));

        std::vector<WGPUConstantEntry> entries;
        for (const auto& [name, value] : std::get<4>(key)) {
            WGPUConstantEntry e = WGPU_CONSTANT_ENTRY_INIT;
            e.key = view_of(name);
            e.value = value;
            entries.push_back(e);
        }
        WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        pipeline_desc.compute.module = module.get();
        pipeline_desc.compute.entryPoint = view_of(launch.entry_point);
        pipeline_desc.compute.constantCount = entries.size();
        pipeline_desc.compute.constants = entries.data();

        WGPUCreateComputePipelineAsyncCallbackInfo info = WGPU_CREATE_COMPUTE_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline, WGPUStringView message,
                           void* userdata1, void*) {
            const auto r = take_back<Requested>(userdata1);
            Build& b = *r->build;
            if (status == WGPUCreatePipelineAsyncStatus_Success) {
                b.state->pipelines[r->index] = gpu::ComputePipeline(pipeline);
                b.settled(ProgramError::Ok, {});
            } else {
                if (pipeline != nullptr) wgpuComputePipelineRelease(pipeline);
                b.settled(status == WGPUCreatePipelineAsyncStatus_CallbackCancelled ? ProgramError::Cancelled
                                                                                      : ProgramError::Build,
                          to_string(message));
            }
        };
        info.userdata1 = hand_off(std::make_unique<Requested>(Requested{build, index}));
        wgpuDeviceCreateComputePipelineAsync(device, &pipeline_desc, info);
    }
    pop_scopes(device, kBuildScopes.size(), build, ProgramError::Build);
}

namespace {

// Reports every settled step whose turn it is, in the order they were run.
// `done` may run another step or destroy the Program: the State is held
// meanwhile, and a step is marked reported before its callback.
void report(Program::State& s) {
    const std::shared_ptr<Program::State> keep = s.in_flight;
    while (const std::optional<std::uint64_t> number = s.order.next()) {
        Program::State::Outstanding& o = s.steps[*number % Order::kOutstanding];
        if (s.order.idle()) s.in_flight.reset();
        const ProgramError error = s.cancelled ? ProgramError::Cancelled : o.error;
        const std::span<const std::byte> bytes =
            error == ProgramError::Ok && o.reads_back ? std::span<const std::byte>(o.bytes.data(), s.readback->size)
                                                      : std::span<const std::byte>{};
        o.done(error, o.message, bytes, o.userdata);
    }
}

// One of a step's callbacks has landed: its two scopes, and its work done or
// its readback mapped, in any order. The last settles the step.
void settle(Program::State& s, Program::State::Outstanding& o, ProgramError e, WGPUStringView message) {
    if (o.error == ProgramError::Ok && e != ProgramError::Ok) {
        o.error = e;
        const std::size_t length = message.data == nullptr ? 0
                                   : message.length == WGPU_STRLEN ? std::strlen(message.data)
                                                                   : message.length;
        o.message.assign(message.data == nullptr ? "" : message.data, std::min(length, o.message.capacity()));
    }
    if (--o.pending > 0) return;
    s.order.settle(o.number);
    report(s);
}

}  // namespace

void Program::run(const Step& step, StepCallback done, void* userdata) {
    State& s = *state_;
    // Two outstanding: a third would copy into a slot still mapped.
    if (s.order.full()) {
        done(ProgramError::Step, "a third step while two are outstanding", {}, userdata);
        return;
    }
    WGPUDevice device = s.device.get();
    const std::uint64_t number = s.order.run();
    State::Outstanding& o = s.steps[number % Order::kOutstanding];
    WGPUBuffer slot = s.slots[number % Order::kOutstanding].get();
    o.number = number;
    o.pending = kStepScopes.size();
    o.error = ProgramError::Ok;
    o.message.clear();
    o.reads_back = s.readback.has_value() && step.logits != 0;
    o.done = done;
    o.userdata = userdata;
    s.in_flight = state_;

    // The scopes cover the step's write too, so a write WebGPU refuses fails
    // the step rather than leaving it to run on the last step's parameters.
    push_scopes(device, kStepScopes);
    // Optimization (browser): the step's head and only the identifier words
    // it uses, in one write (interface.h).
    const std::size_t bytes = kStepHead + (step.fed == 1 ? 0 : 16 * ((std::size_t{step.tokens} + 3) / 4));
    wgpuQueueWriteBuffer(s.queue.get(), s.step.get(), 0, &step, bytes);

    // Past kMaxPositions a position is not exact as an f32 and the chunk
    // arithmetic may wrap (interface.h): refused, before any launch.
    bool fits = std::uint64_t{step.position} + step.tokens <= kMaxPositions;
    if (!fits) o.message.assign("a step past position 2^24");
    if (fits) {
        const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(device, nullptr));
        {
            const gpu::ComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr));
            std::size_t current = s.pipelines.size();
            for (const std::uint32_t index : s.schedules.of(step.tokens, step.logits != 0)) {
                const State::Bound& launch = s.launches[index];
                const std::uint64_t workgroups =
                    workgroups_for(launch.geometry, launch.workgroup_size, step.position, step.tokens, step.logits != 0);
                if (workgroups == 0) continue;   // a combine, the step unsplit
                if (workgroups > s.max_workgroups) {
                    fits = false;
                    break;
                }
                if (launch.pipeline != current) {
                    wgpuComputePassEncoderSetPipeline(pass.get(), s.pipelines[launch.pipeline].get());
                    current = launch.pipeline;
                }
                wgpuComputePassEncoderSetBindGroup(pass.get(), kBindGroup, launch.group.get(), 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass.get(), static_cast<std::uint32_t>(workgroups), 1, 1);
            }
            wgpuComputePassEncoderEnd(pass.get());
        }
        // The readback copied after the pass, into this step's own slot.
        if (fits && o.reads_back) {
            wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), s.readback->buffer, s.readback->offset, slot, 0,
                                                 s.readback->size);
        }
        const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
        if (fits) {
            WGPUCommandBuffer raw = commands.get();
            wgpuQueueSubmit(s.queue.get(), 1, &raw);
        }
    }
    if (fits && o.reads_back) {
        // Its mapping settles the step: it cannot land before the work.
        ++o.pending;
        WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
            auto& state = *static_cast<State*>(userdata1);
            auto& step = *static_cast<State::Outstanding*>(userdata2);
            WGPUBuffer mapped = state.slots[&step - state.steps.data()].get();
            ProgramError e = ProgramError::Ok;
            if (status == WGPUMapAsyncStatus_Success) {
                const void* range = wgpuBufferGetConstMappedRange(mapped, 0, state.readback->size);
                std::memcpy(step.bytes.data(), range, state.readback->size);
                wgpuBufferUnmap(mapped);
            } else {
                e = status == WGPUMapAsyncStatus_CallbackCancelled ? ProgramError::Cancelled : ProgramError::Step;
            }
            settle(state, step, e, message);
        };
        info.userdata1 = &s;
        info.userdata2 = &o;
        wgpuBufferMapAsync(slot, WGPUMapMode_Read, 0, s.readback->size, info);
    } else if (fits) {
        ++o.pending;
        WGPUQueueWorkDoneCallbackInfo info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
            settle(*static_cast<State*>(userdata1), *static_cast<State::Outstanding*>(userdata2),
                   from_work_done(status), message);
        };
        info.userdata1 = &s;
        info.userdata2 = &o;
        wgpuQueueOnSubmittedWorkDone(s.queue.get(), info);
    } else {
        // A step past kMaxPositions, or too large for one dispatch's
        // workgroups: refused, not truncated.
        o.error = ProgramError::Step;
        if (o.message.empty()) o.message.assign("a launch needs more workgroups than one dispatch allows");
    }
    for (std::size_t i = 0; i < kStepScopes.size(); ++i) {
        WGPUPopErrorScopeCallbackInfo info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message,
                           void* userdata1, void* userdata2) {
            settle(*static_cast<State*>(userdata1), *static_cast<State::Outstanding*>(userdata2),
                   from_scope(status, type, ProgramError::Step), message);
        };
        info.userdata1 = &s;
        info.userdata2 = &o;
        wgpuDevicePopErrorScope(device, info);
    }
}

Program::~Program() {
    if (state_) state_->cancelled = true;
}

}  // namespace bllm::kernels
