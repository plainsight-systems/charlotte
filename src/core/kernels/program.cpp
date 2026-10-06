#include "core/kernels/program.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "core/gpu/callback_mode.h"
#include "core/gpu/dispatch_math.h"
#include "core/gpu/userdata.h"

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
        std::uint32_t invocations_per_row;
        std::uint32_t workgroup_size;
    };
    std::vector<Bound> launches;
    gpu::Buffer constants;   // UNIFORM | COPY_DST: every launch's, on kLaunchConstantsAlignment
    gpu::Buffer step;        // UNIFORM | COPY_DST: sizeof(Step)
    std::uint32_t max_workgroups = 0;
    bool cancelled = false;
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
    std::size_t pending = 0;                       // callbacks still to land in this section
    bool bound = false;                            // section two has run
    ProgramError error = ProgramError::Ok;
    std::string message;
    BuildCallback done;
    void* userdata;

    void settled(ProgramError e, std::string m) {
        if (error == ProgramError::Ok && e != ProgramError::Ok) {
            error = e;
            message = std::move(m);
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
    for (std::size_t i = 0; i < launches.size(); ++i) {
        const std::size_t p = pipeline_of[i];
        const gpu::BindGroupLayout layout(wgpuComputePipelineGetBindGroupLayout(s.pipelines[p].get(), kBindGroup));
        std::vector<WGPUBindGroupEntry> entries;
        WGPUBindGroupEntry step = WGPU_BIND_GROUP_ENTRY_INIT;
        step.binding = kStepBinding;
        step.buffer = s.step.get();
        step.size = sizeof(Step);
        entries.push_back(step);
        WGPUBindGroupEntry constants = WGPU_BIND_GROUP_ENTRY_INIT;
        constants.binding = kLaunchBinding;
        constants.buffer = s.constants.get();
        constants.offset = std::uint64_t{kLaunchConstantsAlignment} * i;
        constants.size = kLaunchConstantsAlignment;
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
                              launches[i].invocations_per_row, launches[i].workgroup_size});
    }
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

// A distinct pipeline: the kernel, the format it unpacks, its workgroup size.
using Key = std::tuple<const char*, const formats::Format*, std::uint32_t>;

}  // namespace

void Program::build(const residency::Upload& upload, std::vector<Launch> launches, BuildCallback done,
                    void* userdata) {
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
        if (launch.workgroup_size == 0 || launch.invocations_per_row == 0) {
            done(nullptr, ProgramError::Build, at + "no invocations", userdata);
            return;
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
    build->state = state;
    build->program = std::unique_ptr<Program>(new Program());
    build->program->state_ = state;

    // Section one: the buffers, the constants written, the pipelines asked for.
    push_scopes(device, kBuildScopes);
    const std::uint64_t slots = std::max<std::size_t>(build->launches.size(), 1);
    WGPUBufferDescriptor constants_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    constants_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    constants_desc.size = slots * kLaunchConstantsAlignment;
    state->constants = gpu::Buffer(wgpuDeviceCreateBuffer(device, &constants_desc));
    WGPUBufferDescriptor step_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    step_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    step_desc.size = sizeof(Step);
    state->step = gpu::Buffer(wgpuDeviceCreateBuffer(device, &step_desc));

    // Optimization (browser): every launch's constants in one write (WASM.2).
    std::vector<std::byte> packed(constants_desc.size);
    for (std::size_t i = 0; i < build->launches.size(); ++i) {
        const auto& c = build->launches[i].constants;
        std::memcpy(packed.data() + i * kLaunchConstantsAlignment, c.data(), c.size());
    }
    wgpuQueueWriteBuffer(state->queue.get(), state->constants.get(), 0, packed.data(), packed.size());

    std::map<Key, std::size_t> distinct;
    for (const Launch& launch : build->launches) {
        const Key key{launch.kernel.data(), launch.format, launch.workgroup_size};
        const auto [it, added] = distinct.try_emplace(key, distinct.size());
        build->pipeline_of.push_back(it->second);
    }
    state->pipelines.resize(distinct.size());
    build->pending = distinct.size() + kBuildScopes.size();

    struct Requested {
        std::shared_ptr<Build> build;
        std::size_t index;
    };
    for (const auto& [key, index] : distinct) {
        const Launch& launch = *std::find_if(build->launches.begin(), build->launches.end(), [&](const Launch& l) {
            return Key{l.kernel.data(), l.format, l.workgroup_size} == key;
        });
        // Optimization (practice): each distinct kernel is composed and
        // compiled once, whatever its launches, and every pipeline is asked
        // for at once, so the browser compiles them together (WASM.7).
        std::string source(launch.kernel);
        if (launch.format != nullptr) {
            source += '\n';
            source += launch.format->unpack_wgsl();
        }
        WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgsl.code = view_of(source);
        WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        module_desc.nextInChain = &wgsl.chain;
        const gpu::ShaderModule module(wgpuDeviceCreateShaderModule(device, &module_desc));

        WGPUConstantEntry size = WGPU_CONSTANT_ENTRY_INIT;
        size.key = view_of("workgroup_size");
        size.value = launch.workgroup_size;
        WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        pipeline_desc.compute.module = module.get();
        pipeline_desc.compute.entryPoint = view_of("main");
        pipeline_desc.compute.constantCount = 1;
        pipeline_desc.compute.constants = &size;

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

// One step: its two scopes and its work done, settled in any order.
struct Running {
    std::shared_ptr<Program::State> state;
    std::size_t pending;
    ProgramError error = ProgramError::Ok;
    StepCallback done;
    void* userdata;

    void settled(ProgramError e, const std::string& = {}) {
        if (error == ProgramError::Ok && e != ProgramError::Ok) error = e;
        if (--pending == 0) done(state->cancelled ? ProgramError::Cancelled : error, userdata);
    }
};

}  // namespace

void Program::run(const Step& step, StepCallback done, void* userdata) {
    State& s = *state_;
    WGPUDevice device = s.device.get();

    // Optimization (browser): the step's head and only the identifier words
    // it uses, in one write (interface.h).
    const std::size_t bytes = 16 + 16 * ((std::size_t{step.tokens} + 3) / 4);
    wgpuQueueWriteBuffer(s.queue.get(), s.step.get(), 0, &step, bytes);

    push_scopes(device, kStepScopes);
    auto running = std::make_shared<Running>(Running{state_, kStepScopes.size(), ProgramError::Ok, done, userdata});
    bool fits = true;
    {
        const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(device, nullptr));
        {
            const gpu::ComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr));
            std::size_t current = s.pipelines.size();
            for (const State::Bound& launch : s.launches) {
                const gpu::DispatchResult d = gpu::dispatch_count(
                    std::uint64_t{step.tokens} * launch.invocations_per_row, launch.workgroup_size, s.max_workgroups);
                if (d.status != gpu::DispatchStatus::Ok) {
                    fits = false;
                    break;
                }
                if (launch.pipeline != current) {
                    wgpuComputePassEncoderSetPipeline(pass.get(), s.pipelines[launch.pipeline].get());
                    current = launch.pipeline;
                }
                wgpuComputePassEncoderSetBindGroup(pass.get(), kBindGroup, launch.group.get(), 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass.get(), d.workgroup_count, 1, 1);
            }
            wgpuComputePassEncoderEnd(pass.get());
        }
        const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
        if (fits) {
            WGPUCommandBuffer raw = commands.get();
            wgpuQueueSubmit(s.queue.get(), 1, &raw);
        }
    }
    if (fits) {
        ++running->pending;
        struct Done {
            std::shared_ptr<Running> running;
        };
        WGPUQueueWorkDoneCallbackInfo info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
        info.mode = gpu::kCallbackMode;
        info.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView, void* userdata1, void*) {
            const auto d = take_back<Done>(userdata1);
            d->running->settled(from_work_done(status));
        };
        info.userdata1 = hand_off(std::make_unique<Done>(Done{running}));
        wgpuQueueOnSubmittedWorkDone(s.queue.get(), info);
    } else {
        // A step too large for one dispatch's workgroups: refused, not
        // truncated (gpu/dispatch_math.h).
        running->error = ProgramError::Step;
    }
    pop_scopes(device, kStepScopes.size(), running, ProgramError::Step);
}

Program::~Program() {
    if (state_) state_->cancelled = true;
}

}  // namespace bllm::kernels
