#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <webgpu/webgpu.h>

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
//     format it reads, its workgroup size — is composed and compiled once,
//     with createComputePipelineAsync, every pipeline requested at once so
//     the browser compiles them together; a bind group is made for every
//     launch; every launch's constants are packed into one uniform buffer,
//     each on kLaunchConstantsAlignment, and written with one write; the
//     step's uniform buffer is created. The buffers bound are upload's,
//     named by plan index, so the program allocates only those two.
//   - Run, once a step: one write of the step's parameters, then one command
//     encoder holding one compute pass that sets each launch's bind group
//     and dispatches its workgroups, setting a pipeline only where it
//     changes from the launch before, then one submit. Launches run in the
//     order given, which is the graph's; WebGPU orders dispatches in a pass
//     by their buffer use, so a launch sees what the one before it wrote.
//   - Failure is visible, as upload's is. Build runs inside out-of-memory,
//     validation and internal error scopes, and a pipeline that fails to
//     compile reports WebGPU's message; a step runs inside validation and
//     internal scopes popped with the step's completion, and any of them
//     failing fails the step. A step reported done is not proof the device
//     ran it: a lost device resolves queued work as done and scopes clean
//     (gpu/device.h). What a step computed is trusted only through a
//     completed mapping, which a lost device refuses — the sampler's
//     readback of the step's candidates, as upload's witness is its.
//   - It borrows: the upload, and so its buffers, outlives the program
//     (I.11). Callbacks keep their state alive on their own, as upload's do,
//     so destroying a program with a step in flight reports Cancelled.
//
// What a step costs, counted: calls into WebGPU are 1 writeBuffer of 16 +
// 16 × ceil(tokens / 4) bytes, 1 createCommandEncoder, 1 beginComputePass,
// for each launch 1 setBindGroup, 1 dispatchWorkgroups and 1 setPipeline
// where the kernel changes, then end, finish, submit, two scopes' pushes and
// pops, onSubmittedWorkDone, and the release of the three single-use objects
// WebGPU makes a step — the command encoder, the pass encoder and the command
// buffer: 14 calls out of the module, and 2 or 3 a launch; and three
// callbacks back in, the two scopes' and the queue's. Everything else —
// buffers, pipelines, bind groups — is made at load. The GPU adds about
// 1.5 µs a dispatch (interface.h). Build costs a pipeline per distinct
// kernel — about a dozen for a model — and a bind group per launch, once.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.11   Never transfer ownership by a raw pointer — the upload is
//            borrowed, and stated so; GPU objects are RAII handles (R.1).
//     E.27   Use error codes systematically — ProgramError.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — one pass, one submit, a step.
//     GPU.9  Suballocate — one constants buffer for every launch.
//     WASM.2 Batch work across the JS boundary — one write a step.
//     WASM.7 Budget startup — pipelines compile together, asynchronously.
//     MEM.9  Allocate at init — buffers, pipelines and bind groups at load;
//            a step makes only the single-use encoders WebGPU requires, and
//            its callbacks carry the program's own state, so it allocates
//            nothing on the heap.

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
using StepCallback = void (*)(ProgramError error, void* userdata);

class Program {
public:
    // Builds the program for `launches`, in the graph's order, over
    // `upload`'s buffers. `done` is called once, with the program or a
    // failure. Preconditions: `upload` has finished and outlives the
    // program; every launch's bindings name buffers upload created.
    static void build(const residency::Upload& upload, std::vector<Launch> launches, BuildCallback done,
                      void* userdata);

    // Runs one step. `done` is called once, when the queue has finished it
    // or it has failed. Preconditions: 1 <= step.tokens <= kPrefillBlock,
    // every identifier below the vocabulary, and no other step in flight.
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
