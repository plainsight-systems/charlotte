#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
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
//     A kernel that writes the cache composes with the cache format's pack
//     (format.h), which reads no binding, so it composes beside an unpack.
//   - Geometry: a launch runs a fixed number of invocations for each row it
//     covers — every token of the step, or only its last, which is all the
//     final norm and the output head need — in workgroups of a size the
//     launcher chooses and gives the kernel as an override constant, so the
//     size is stated once. The program works out each step's workgroups from
//     its position and token count with workgroups_for below. Workgroup size belongs to
//     each kernel: the right value differs per kernel, and a shared constant
//     would couple them. A launch that covers only the last token is one
//     row's workgroups, and the program sets its override constant
//     `last_token` to true; a kernel that can cover only the last token
//     declares `override last_token: bool = false;` and then works on row
//     tokens - 1 of each buffer. Rows alone decides both, so they cannot
//     disagree. A launch may instead cover tiles of rows — attention's
//     query tiles — and may multiply its workgroups by the step's key
//     chunks or run only when the step splits them, and may run in one
//     range of token counts — a regime, or part of one (key_chunks,
//     regime_for, TokenRange, workgroups_for below); a launch that does not
//     run in a step is not dispatched. A module may hold several entry
//     points, and a launch names its own.
//   - Variants: a kernel's other override constants select among its forms —
//     the norm with or without the residual add — so one WGSL source serves
//     them. The program compiles each distinct kernel, unpack and pack
//     format, workgroup size and set of overrides once, and passes every override to the pipeline;
//     `workgroup_size` and `last_token` are the program's to set, and a
//     launch naming either, or one name twice, is refused at build.
//   - The regime is chosen per step from its token count.
//   - The step's token identifiers are below the vocabulary: the runtime
//     checks them before it writes a step, since a kernel cannot tell an
//     identifier past the table from one in another piece of it.
//
// What a launch costs, measured on the target (Chrome 152, Apple M3 Max,
// Metal). Holding a pass's work constant at 300 workgroups and splitting it
// into 1, 3, 10, 30, 100 and 300 dispatches, each extra dispatch adds about
// 1.5 µs of GPU time (20 µs a pass for one dispatch, 478 µs for 300; medians
// of 16 runs of 40 passes). Encoding from JavaScript is about 0.03 µs a
// launch, pipeline changes included, a lower bound for the module's path,
// which adds a wasm-to-browser crossing a call. So a graph of 300 launches
// pays about 0.45 ms a pass in dispatch overhead, about half the 0.95 ms
// floor for reading Qwen3 0.6B's 380 MB of weights at the M3 Max's 400 GB/s
// (its published bandwidth). That is not small, and WebGPU has no captured
// compute sequence to replay, so the lever is fusion: each kernel's header
// states its launches a pass, and the graph keeps the total low (GPU.6). The
// module's own encoding cost, and where a real pass's time goes, are
// measured once the program runs (GPU.10).
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

// A step's regime, from its token count alone, never from anything else
// about the step: one token decodes, more prefill.
[[nodiscard]] constexpr Regime regime_for(std::uint32_t tokens_in_step) noexcept {
    return tokens_in_step == 1 ? Regime::Decode : Regime::Prefill;
}

// The steps a launch runs in, by token count, both ends included: a
// regime's, or a part of prefill's, as a matrix product's tile widths
// take (kernels/matmul/matmul.h).
struct TokenRange {
    std::uint32_t least = 1;
    std::uint32_t most = UINT32_MAX;
    friend bool operator==(const TokenRange&, const TokenRange&) = default;
};

[[nodiscard]] constexpr TokenRange tokens_of(Regime r) noexcept {
    return r == Regime::Decode ? TokenRange{1, 1} : TokenRange{2, UINT32_MAX};
}

inline constexpr std::uint32_t kBindGroup = 0;
inline constexpr std::uint32_t kStepBinding = 0;
inline constexpr std::uint32_t kLaunchBinding = 1;
inline constexpr std::uint32_t kFirstWeightBinding = 2;

// Binding 0, a uniform buffer, declared once in WGSL, kernels/step.wgsl,
// which the program composes before every kernel, as it composes a format's
// unpack (program.h):
//
//   struct Step { position: u32, tokens: u32, logits: u32, ids: array<vec4<u32>, 128> }
//
// and one word of padding after `logits`, since a uniform array needs a
// 16-byte stride; `logits` is the program's, read by no kernel. Token i's
// identifier is ids[i / 4][i % 4]. A step
// writes its 16-byte head and only the identifier words it uses: 32 bytes
// for a decode step.
// Optimization (browser): the identifiers ride in the uniform every launch
// already binds, so a step is one write, not one for its parameters and one
// for its tokens (WASM.2).
struct Step {
    std::uint32_t position;   // the first token's position in the context
    std::uint32_t tokens;     // 1 .. kPrefillBlock
    // 1 when the step's last token's logits are sampled; 0 for a prefill
    // step that does not end the prompt, which runs no launch over the last
    // token alone — the final norm and the head (graph/graph.h).
    std::uint32_t logits;
    std::uint32_t padding;
    std::array<std::uint32_t, residency::kPrefillBlock> ids;
};
static_assert(sizeof(Step) == 16 + 4 * residency::kPrefillBlock);

// Each launch's constants lie at their own offset in one uniform buffer, in
// a whole number of slots on the alignment WebGPU's default limits require
// for a uniform binding's offset — one for most kernels, three for rope's
// 528 bytes — and no larger than this.
inline constexpr std::uint32_t kLaunchConstantsAlignment = 256;   // minUniformBufferOffsetAlignment
inline constexpr std::uint32_t kMaxLaunchConstants = 1024;

// A bound range of a buffer the residency plan names: a piece of a weight,
// a working buffer, a layer's cache.
struct Binding {
    residency::BufferIndex buffer;
    std::uint64_t offset;
    std::uint64_t size;
};

// The rows a launch covers.
enum class Rows {
    EveryToken,
    LastToken,
};

// The positions a step may reach: position + tokens <= 2^24. Rope forms a
// position as an f32, exact only below 2^24 (kernels/rope/rope.h), and
// below it every intermediate of the chunk arithmetic — position + 1,
// tokens × count, at most 512 × 65,537 — stays far below 2^32. Every listed
// model's context, at most 131,072, is well inside. The program refuses a
// step past it.
inline constexpr std::uint32_t kMaxPositions = 1u << 24;

// Keys a chunk holds, fixed by position: chunk c is positions 256c ..
// 256c + 255 (kernels/attention/attention.h). And the query rows a split
// step's partial buffers hold, the prefill block's.
inline constexpr std::uint32_t kChunkKeys = 256;
inline constexpr std::uint32_t kPartialRows = residency::kPrefillBlock;

// A layer's chunks of keys for a step, and how it splits them: from the
// chunk holding its first row's earliest key, position − window + 1, to the
// chunk holding its last row; split one workgroup a chunk when every row's
// partials fit, else not at all. The program dispatches by it and the
// attention kernel computes the same in WGSL.
struct KeyChunks {
    std::uint32_t first;
    std::uint32_t count;
    std::uint32_t splits;   // count, or 1
};

// Preconditions: tokens >= 1, window >= 1, position + tokens <=
// kMaxPositions.
[[nodiscard]] constexpr KeyChunks key_chunks(std::uint32_t position, std::uint32_t tokens,
                                             std::uint32_t window) noexcept {
    const std::uint32_t earliest = position + 1 > window ? position + 1 - window : 0;
    const std::uint32_t first = earliest / kChunkKeys;
    const std::uint32_t count = (position + tokens - 1) / kChunkKeys - first + 1;
    return {first, count, tokens * count <= kPartialRows ? count : 1};
}

// How a launch uses the step's key chunks.
enum class KeySplit {
    // Not at all.
    None,
    // Its workgroups are multiplied by the step's splits.
    PerChunk,
    // It runs only when the step splits: the combine after a split.
    WhenSplit,
};

// What decides how many invocations a launch runs in a step.
struct Geometry {
    Rows rows;
    std::uint32_t invocations_per_row;
    std::uint32_t rows_per_tile;   // 0: a launch over rows
    KeySplit key_split;
    std::uint32_t window;
    // The steps it runs in, by token count; every step by default.
    TokenRange tokens = {};
};

// The workgroups a launch of `workgroup_size` runs in a step of `tokens`
// from `position`; 0 when it does not run: in a step outside its token
// range, over the last token alone in a step that asks for no `logits`, or
// a combine when the step does not split. A split runs its rows' or
// tiles' workgroups once for each chunk, each a whole number of
// workgroups. Preconditions: tokens >= 1; workgroup_size >= 1; a key
// split's window >= 1; position + tokens <= kMaxPositions.
[[nodiscard]] constexpr std::uint64_t workgroups_for(const Geometry& g, std::uint32_t workgroup_size,
                                                     std::uint32_t position, std::uint32_t tokens,
                                                     bool logits) noexcept {
    if (tokens < g.tokens.least || tokens > g.tokens.most) return 0;
    if (g.rows == Rows::LastToken && !logits) return 0;
    const std::uint64_t rows = g.rows == Rows::LastToken ? 1 : tokens;
    const std::uint64_t covered =
        g.rows_per_tile == 0 ? rows * g.invocations_per_row
                             : (rows + g.rows_per_tile - 1) / g.rows_per_tile * g.rows_per_tile *
                                   g.invocations_per_row;
    const std::uint64_t workgroups = (covered + workgroup_size - 1) / workgroup_size;
    if (g.key_split == KeySplit::None) return workgroups;
    const std::uint32_t splits = key_chunks(position, tokens, g.window).splits;
    if (g.key_split == KeySplit::WhenSplit) return splits > 1 ? workgroups : 0;
    return workgroups * splits;
}

// A WGSL override constant and its value.
struct Override {
    std::string_view name;
    double value;
    friend bool operator==(const Override&, const Override&) = default;
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
    Rows rows = Rows::EveryToken;
    // Override constants beyond workgroup_size, which every kernel declares.
    std::vector<Override> overrides;
    // The format whose pack the kernel composes with — the cache's, for a
    // kernel that writes the cache (formats/format.h) — or null. Pack reads
    // no binding, so it composes beside an unpack.
    const formats::Format* pack_format = nullptr;
    // When nonzero, the launch covers tiles of this many rows: the program
    // dispatches ceil(rows / rows_per_tile) tiles, each rows_per_tile ×
    // invocations_per_row invocations, a whole number of workgroups.
    std::uint32_t rows_per_tile = 0;
    // Its use of the step's key chunks, for a layer of this window.
    KeySplit key_split = KeySplit::None;
    std::uint32_t window = 0;
    // The WGSL entry point: one module may hold two roles, which share its
    // functions and not each other's workgroup memory.
    std::string_view entry_point = "main";
    // The steps the launch runs in, by token count, every step by default:
    // a matrix product's decode form runs in tokens_of(Regime::Decode), its
    // prefill tiles in parts of prefill's.
    TokenRange tokens = {};
};

}  // namespace bllm::kernels
