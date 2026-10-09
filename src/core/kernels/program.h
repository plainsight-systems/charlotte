#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
//     readback of the step's drawn token, as upload's witness is its. A step
//     that ran and fails once the device has said it is lost is reported
//     DeviceLost — a run refused at once, unrun, is not;
//     WebGPU does not order the lost callback before the failed mapping, so a
//     failure that comes first is reported Step, and a later one DeviceLost.
//   - It borrows: the upload, and so its buffers, outlives the program
//     (I.11). Callbacks keep their state alive on their own, as upload's do,
//     so destroying a program with a step in flight reports Cancelled.
//
// Profiling, in a diagnostic build alone (core/diagnostics.h; TLM.1): the
// clean build compiles none of it, and its module has no symbol of it
// (TLM.8). A profiled step runs as run() does — its launches in the graph's
// order, in its one compute pass, over the same pipelines and bind groups —
// but may stop after its first `launches` launches, and is timed by the GPU:
//   - a timestamp at the pass's beginning and one at its end, always — the
//     time the step, or its prefix, takes in the shape run() gives it;
//   - and, on a device that grants timestamps inside a pass, one after each
//     launch it dispatches, so each launch's time in that same pass. The
//     target's Metal adapter does not offer them: there a launch's time is
//     what adding it to a prefix adds (bench/forward_profile.cpp).
// No pass is split: a pass a launch would drain the GPU at every boundary,
// timing each launch in a shape run() never gives it, and add three calls
// into Dawn a launch.
// The timestamps are resolved into a buffer and copied to a mappable one
// with the step's readback, and reported with it; WebGPU gives them in
// nanoseconds, at the resolution of the GPU's counter. Each of the two slots
// a step may be outstanding in has its own query set, resolve buffer and
// readback buffer, made at the first profiled step — so profiled steps
// pipeline as run()'s do, and a run of them is timed as it runs. A profiled
// step runs only on a device that granted the timestamp-query feature
// (gpu/device.h's DiagnosticRequest), and not beside an unprofiled one;
// otherwise it is refused at once, named. A whole profiled step's results
// are run()'s: no timestamp writes a buffer a kernel reads. A prefix leaves
// the step's later launches undone — the draw among them, so it reads
// nothing back, and the cache holds the step's positions for the layers it
// ran alone — so a step at those positions runs whole before any result is
// read from them.
//
// What a step of run() costs, counted: calls into WebGPU are 1 writeBuffer of 48
// bytes for a fed step and 48 + 16 × ceil(tokens / 4) for another, 1
// createCommandEncoder, 1 beginComputePass, for each launch that runs 1
// setBindGroup, 1 dispatchWorkgroups and 1 setPipeline where the kernel
// changes, then end, finish, submit, two scopes' pushes and pops, and the
// release of the three single-use objects WebGPU makes a step — the command
// encoder, the pass encoder and the command buffer; then onSubmittedWorkDone,
// or, for a step that reads back, a copyBufferToBuffer, a mapAsync, a
// getConstMappedRange and an unmap: 14 calls out of the module, 17 reading
// back, and 2 or 3 a launch; and three callbacks back in, the two scopes'
// and the queue's or the mapping's. A profiled step, in diagnostic builds,
// adds to run()'s the resolve of its timestamps, their copy, and their
// buffer's mapAsync, getConstMappedRange and unmap — 22 calls a whole step,
// 19 a prefix, which reads nothing back — a fourth callback, the timestamps'
// mapping, and, where timestamps inside a pass are granted, a writeTimestamp
// a launch; the device's capability is read once, at build. Everything else —
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
//            timed by the GPU's own timestamps in the shape run() gives it.
//     TLM.1  Compile telemetry out by default — the profiled step is the
//            diagnostic build's alone.

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

// A kernel as build() compiles it, for a launch: the step's declaration
// (step.wgsl), then the kernel, then the unpack of the format it reads and
// the pack of the format it writes, with its entry point and its override
// constants in name order — workgroup_size always, last_token for a launch
// of the last token, then the launch's own. Two launches with the same
// composition share a pipeline. Exposed so a tool can hand a browser
// exactly what the program compiles, to find which kernel its shader
// compiler refuses (bench/kernel_dump.cpp).
struct Composed {
    std::string source;
    std::string_view entry_point;
    std::vector<std::pair<std::string_view, double>> constants;
};
[[nodiscard]] Composed compose(const Launch& launch);

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
    // A profiled step's GPU timestamps, in nanoseconds: its pass's beginning
    // and end, and, where the device grants timestamps inside a pass, the
    // end of each launch dispatched, by the launch's index in the graph.
    struct Timestamps {
        std::uint64_t begin_ns;
        std::uint64_t end_ns;
        std::span<const std::pair<std::uint32_t, std::uint64_t>> launches;
    };
    // `times` is valid only during the call; zero for a step refused or
    // failed.
    using ProfileCallback = void (*)(ProgramError error, std::string_view message, std::span<const std::byte> readback,
                                     const Timestamps& times, void* userdata);

    // Runs one step as run() does, but only its first `launches` launches
    // in the graph's order — all of them for a whole step — timed (above).
    // `done` is called once, in run order with other profiled steps.
    // Preconditions: run()'s; launches <= the graph's; no unprofiled step
    // outstanding.
    void run_profiled(const Step& step, std::uint32_t launches, ProfileCallback done, void* userdata);

    // The graph's launches: a whole step's bound for run_profiled.
    [[nodiscard]] std::uint32_t launch_count() const noexcept;

    // Whether the device granted timestamp queries, read once at build:
    // without them run_profiled refuses every step.
    [[nodiscard]] bool can_profile() const noexcept;
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
