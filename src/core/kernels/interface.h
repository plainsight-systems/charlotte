#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/formats/format.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Contract 7: kernel launch.
//
// Every kernel binds the same way, so one piece of code launches all of them
// and changing a kernel never changes how the others are called. In bind
// group 0:
//
//   binding 0   the step's parameters, shared by every launch and written
//               once per step: its first position, its token count, and its
//               tokens' identifiers (Step below)
//   binding 1   this launch's constants, written once at load
//   binding 2…  the kernel's weights, in the order it declares them, then its
//               activations
//
//   - A kernel launcher is pure: from the plan's weight views and working
//     buffers it describes its launches (Launch below) — which kernel, which
//     format it unpacks, its constants, its bindings, its geometry — and
//     holds no GPU object, so what it launches is tested without a device.
//     The program (kernels/program.h) carries the descriptions out, as
//     upload carries out the routes.
//   - Pipelines and bind groups are built at load, one bind group per launch
//     in the graph. No bind group, pipeline or buffer is created per token,
//     so a token costs one uniform write and its dispatches.
//   - A kernel that reads a weight composes with that weight's format's
//     unpack, which reads the binding named `weights` (format.h); a launch
//     binds one piece of a weight, so a split weight is one launch per piece.
//   - Geometry: a launch runs a fixed number of invocations for each token
//     row of the step, in workgroups of a size the launcher chooses and gives
//     the kernel as an override constant, so the size is stated once. The
//     program works out each step's workgroups from its token count with
//     core/gpu/dispatch_math. Workgroup size belongs to each kernel: the right
//     value differs per kernel, and a shared constant would couple them.
//   - The regime is chosen per step from its token count.
//   - The step's token identifiers are below the vocabulary: the runtime
//     checks them before it writes a step, since a kernel cannot tell an
//     identifier past the table from one in another piece of it.
//
// What a launch costs, measured on the target (Chrome 152, Apple M3 Max,
// Metal; 300 one-workgroup dispatches a pass, medians of 20 runs of 40
// passes): the GPU spends about 1.7 µs on every dispatch on top of its own
// work, whether or not dispatches share a buffer; encoding from JavaScript
// is about 0.03 µs a launch, pipeline changes included, a lower bound for
// the module's path, which adds a wasm-to-browser crossing a call; and
// submitting one pass and hearing it done takes about 0.9 ms. So a graph of
// 300 launches pays about 0.5 ms a pass in dispatch overhead, about half the
// 0.95 ms floor for reading Qwen3 0.6B's 380 MB of weights at the M3 Max's
// 400 GB/s (its published bandwidth). That is not small, and WebGPU has no
// captured compute sequence to replay, so the lever is fusion: each kernel's
// header states its launches a pass, and the graph keeps the total low
// (GPU.6). The module's own encoding cost, and where a real pass's time goes,
// are measured once the program runs (GPU.10).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     I.4    Make interfaces precisely and strongly typed — a launch names
//            planned buffers through Binding, never a raw offset alone.
//   C++ performance guidelines
//     GPU.6  Batch tiny GPU work — the measurement above, and fusion as the
//            response; each kernel records its launches.
//     GPU.10 Profile before optimizing — the module's path is measured once
//            it runs, not assumed from the JavaScript figure.
//     GPU.9  Suballocate — every launch's constants in one buffer.
//     WASM.2 Batch work across the JS boundary — the step's parameters and
//            identifiers in one write; constants in one write, at load.
//     MEM.9  Allocate at init, not in steady state — every pipeline, bind
//            group and buffer at load.

enum class Regime {
    // One token: matrix times vector, bound by weight bandwidth.
    Decode,
    // Many tokens: matrix times block, bound by arithmetic.
    Prefill,
};

inline constexpr std::uint32_t kBindGroup = 0;
inline constexpr std::uint32_t kStepBinding = 0;
inline constexpr std::uint32_t kLaunchBinding = 1;
inline constexpr std::uint32_t kFirstWeightBinding = 2;

[[nodiscard]] Regime regime_for(std::uint32_t tokens_in_step) noexcept;

// Binding 0, a uniform buffer. As WGSL declares it:
//
//   struct Step { position: u32, tokens: u32, ids: array<vec4<u32>, 128> }
//
// with two words of padding before `ids`, which a uniform array needs on a
// 16-byte stride; token i's identifier is ids[i / 4][i % 4]. A step writes
// its 16-byte head and only the identifier words it uses: 32 bytes for a
// decode step.
// Optimization (browser): the identifiers ride in the uniform every launch
// already binds, so a step is one write, not one for its parameters and one
// for its tokens (WASM.2).
struct Step {
    std::uint32_t position;   // the first token's position in the context
    std::uint32_t tokens;     // 1 .. kPrefillBlock
    std::uint32_t padding[2];
    std::array<std::uint32_t, residency::kPrefillBlock> ids;
};
static_assert(sizeof(Step) == 16 + 4 * residency::kPrefillBlock);

// Each launch's constants lie at their own offset in one uniform buffer,
// on the alignment WebGPU's default limits require for a uniform binding's
// offset, and no larger than this.
inline constexpr std::uint32_t kLaunchConstantsAlignment = 256;   // minUniformBufferOffsetAlignment
inline constexpr std::uint32_t kMaxLaunchConstants = 256;

// A bound range of a buffer the residency plan names: a piece of a weight,
// a working buffer, a layer's cache.
struct Binding {
    residency::BufferIndex buffer;
    std::uint64_t offset;
    std::uint64_t size;
};

// One launch, as a kernel launcher describes it.
struct Launch {
    // The kernel's WGSL entry point and helpers, embedded at build time.
    std::string_view kernel;
    // The format whose unpack the kernel composes with, or null for a kernel
    // that reads no weight.
    const formats::Format* format;
    // Binding 1: this launch's constants, as the kernel's WGSL struct lays
    // them out. At most kMaxLaunchConstants bytes.
    std::vector<std::byte> constants;
    // Bindings 2 onward: weights, then activations.
    std::vector<Binding> bindings;
    std::uint32_t invocations_per_row;
    std::uint32_t workgroup_size;
};

}  // namespace bllm::kernels
