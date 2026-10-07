#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/diagnostics.h"
#include "core/gpu/wgpu_handles.h"
#include "core/kernels/interface.h"
#include "core/residency/upload.h"

namespace bllm::kernels {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Carries out a graph's launches (interface.h) on the device: built once at
// load, then run once a step. It holds the GPU objects the launch
// descriptions name, as upload holds the buffers the plan names.
//
//   - Build, at load: each distinct kernel — its WGSL, the unpack of the
//     format it reads, the pack of the format it writes, its workgroup size
//     and override constants — is composed and compiled once,
//     with createComputePipelineAsync, every pipeline requested at once so
//     the browser compiles them together; a bind group is made for every
//     launch; every launch's constants are packed into one uniform buffer,
//     each in whole slots of kLaunchConstantsAlignment, and written with one
//     write; a launch whose pack format has no pack is refused; the
//     step's uniform buffer is created, and, where the graph names a range
//     to read back — the draw's record (sampler/sampler.h) — two mappable
//     readback slots of its size. The buffers bound are upload's, named by
//     plan index, so the program allocates only those.
//   - Run, once a step: one write of the step's parameters, then one command
//     encoder holding one compute pass that sets each launch's bind group
//     and dispatches its workgroups, setting a pipeline only where it
//     changes from the launch before, then one submit. It walks only the
//     launches the step's token count and logits can run, planned at build
//     (kernels/schedule.h). Launches run in the
//     order given, which is the graph's; WebGPU orders dispatches in a pass
//     by their buffer use, so a launch sees what the one before it wrote.
//   - Read back: a step that asks for logits copies the readback range into
//     one of the two slots, after its pass, in its own command buffer, and
//     maps the slot once submitted; its callback carries the bytes. Steps
//     alternate between the slots.
//   - Two steps may be outstanding — one running and the next queued — so
//     the runtime submits the next before the last's readback maps (GPU.7).
//     WebGPU orders a queue's writes after the submits before them, so one
//     step uniform serves both. It does not order mappings of two buffers, so
//     each step carries a sequence number, and one that settles before an
//     earlier step is held until that one has been reported: callbacks come
//     in the order the steps were run (kernels/order.h). A third run while two are
//     outstanding is refused at once, without running, so a slot is never
//     copied into while mapped.
//   - Failure is visible, as upload's is. Build runs inside out-of-memory,
//     validation and internal error scopes, and a pipeline that fails to
//     compile reports WebGPU's message; a step runs inside validation and
//     internal scopes popped with the step's completion, and any of them
//     failing fails the step with the first failure's message. A step reported done is not proof the device
//     ran it: a lost device resolves queued work as done and scopes clean
//     (gpu/device.h). What a step computed is trusted only through a
//     completed mapping, which a lost device refuses — the sampler's
//     readback of the step's drawn token, as upload's witness is its.
//   - It borrows: the upload, and so its buffers, outlives the program
//     (I.11). Callbacks keep their state alive on their own, as upload's do,
//     so destroying a program with a step in flight reports Cancelled.
//
// Profiling, in a diagnostic build alone (core/diagnostics.h; TLM.1): the
// clean build compiles none of it, and its module has no symbol of it
// (TLM.8). A profiled step runs as run() does, its launches in the graph's
// order over the same pipelines and bind groups, with GPU timestamps written
// at one of two grains:
//   - Step: one compute pass, timestamps at its beginning and end — the
//     step's GPU time as run() spends it, but for those two writes.
//   - Launch: each launch that dispatches in a compute pass of its own,
//     timestamps at its beginning and end — what each launch takes. A pass
//     a launch perturbs the step: each boundary may drain the GPU's work
//     before the next begins, where one pass lets dispatches overlap as
//     their buffer use allows. So the launches' sum against a Step-grain
//     run of the same step is the instrument's cost, and every profile
//     reports both (TLM.6).
// The timestamps are resolved into a buffer, copied to a mappable one and
// read back with the step; WebGPU gives them in nanoseconds. A query set of
// two entries a launch, its resolve buffer and its readback buffer are made
// at the first profiled step, outside any step a throughput is quoted from.
// A profiled step runs only with no step outstanding, and on a device that
// granted the timestamp-query feature (gpu/device.h's DiagnosticRequest);
// otherwise it is refused at once, named. Its results are the same bits as
// run()'s: timestamps write no buffer a kernel reads.
//
// What a step costs, counted: calls into WebGPU are 1 writeBuffer of 48
// bytes for a fed step and 48 + 16 × ceil(tokens / 4) for another, 1
// createCommandEncoder, 1 beginComputePass, for each launch that runs 1
// setBindGroup, 1 dispatchWorkgroups and 1 setPipeline where the kernel
// changes, then end, finish, submit, two scopes' pushes and pops, and the
// release of the three single-use objects WebGPU makes a step — the command
// encoder, the pass encoder and the command buffer; then onSubmittedWorkDone,
// or, for a step that reads back, a copyBufferToBuffer, a mapAsync, a
// getConstMappedRange and an unmap: 14 calls out of the module, 17 reading
// back, and 2 or 3 a launch; and three callbacks back in, the two scopes'
// and the queue's or the mapping's. Everything else —
// buffers, pipelines, bind groups — is made at load. The GPU adds about
// 1.5 µs a dispatch (interface.h). Build costs a pipeline and a bind group
// layout per distinct kernel — 28 for Qwen3 (graph/graph.h) — one key a launch
// to find them, and a bind group per launch, once.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — the upload is
//            borrowed, and stated so; GPU objects are RAII handles (R.1).
//     E.27   Use error codes systematically — ProgramError.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — one pass, one submit, a step.
//     GPU.9  Suballocate — one constants buffer for every launch.
//     MEM.11 Plan offsets once from known sizes — each launch's constants
//            offset, at build.
//     WASM.2 Batch work across the JS boundary — one write a step.
//     WASM.7 Budget startup — pipelines compile together, asynchronously.
//     MEM.9  Allocate at init — buffers, pipelines and bind groups at load;
//            a step makes only the single-use encoders WebGPU requires, and
//            its callbacks carry the program's own state, so it allocates
//            nothing on the heap.
//     GPU.10 Profile with GPU timelines before optimizing — a profiled step,
//            timed by the GPU's own timestamps, a launch at a time, each named.
//     TLM.1  Compile telemetry out by default — the profiled step is the
//            diagnostic build's alone.
//     TLM.6  Diagnostic mode is not benchmark mode — Launch grain's cost is
//            reported beside it, against Step grain.

enum class ProgramError {
    Ok,
    // A pipeline failed to compile, or building the bind groups or buffers
    // failed validation. The message is WebGPU's.
    Build,
    // A step failed validation, or the device reported an internal error.
    Step,
    DeviceLost,
    // The program was destroyed with work in flight.
    Cancelled,
};

class Program;

using BuildCallback = void (*)(std::unique_ptr<Program> program, ProgramError error, std::string_view message,
                               void* userdata);
// `message` is WebGPU's for a failed step, at most 1 KiB, and empty on
// success; `readback` the step's copy of the readback range, empty for a
// step that read none or failed. Both are valid only during the call.
using StepCallback = void (*)(ProgramError error, std::string_view message, std::span<const std::byte> readback,
                              void* userdata);

// The most bytes a step reads back: the draw's record.
inline constexpr std::uint64_t kMaxReadback = 16;

class Program {
public:
    // Builds the program for `launches`, in the graph's order, over
    // `upload`'s buffers, reading back `readback` from each step that asks
    // for logits, where given. `done` is called once, with the program or a
    // failure. Preconditions: `upload` has finished and outlives the
    // program; every launch's bindings name buffers upload created;
    // `readback` names one of them, a working buffer, and is at most
    // kMaxReadback bytes.
    static void build(const residency::Upload& upload, std::vector<Launch> launches,
                      std::optional<Binding> readback, BuildCallback done, void* userdata);

#if BLLM_DIAGNOSTICS_ENABLED
    enum class ProfileGrain { Step, Launch };

    // The GPU time of a launch — its index in the graph's launches — or, at
    // Step grain, of the whole step, whose index is the launches' count.
    struct Timed {
        std::uint32_t launch;
        std::uint64_t begin_ns;
        std::uint64_t end_ns;
    };
    // `times`, in the order run, is valid only during the call; empty for a
    // step refused or failed.
    using ProfileCallback = void (*)(ProgramError error, std::string_view message, std::span<const Timed> times,
                                     void* userdata);

    // Runs one step as run() does, timed at `grain` (above). `done` is
    // called once. Preconditions: run()'s, and no step outstanding.
    void run_profiled(const Step& step, ProfileGrain grain, ProfileCallback done, void* userdata);
#endif

    // Runs one step. `done` is called once: when the queue has finished it,
    // or its readback has mapped, or it has failed; a step past
    // kMaxPositions fails without running, and a third while two are
    // outstanding is refused at once. Preconditions: 1 <= step.tokens <=
    // kPrefillBlock, every identifier below the vocabulary.
    void run(const Step& step, StepCallback done, void* userdata);

    ~Program();
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    // What the program and its in-flight callbacks share; defined only where
    // the program is.
    struct State;

private:
    Program() = default;
    std::shared_ptr<State> state_;   // shared with in-flight callbacks
};

}  // namespace bllm::kernels
