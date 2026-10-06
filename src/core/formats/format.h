#pragma once

#include <cstdint>
#include <string_view>

#include "core/formats/device_layout.h"
#include "core/gguf/types.h"

namespace bllm::formats {

// Contract: what every weight format supplies.
//
// Each formats/<format>/ provides one Format, and the capability table lists
// it under the GGUF type it implements. A format is the only code that knows
// its block layout's meaning. It supplies:
//
//   - unpack, as WGSL a kernel composes with. No kernel knows a format; adding
//     one adds a file here and a row in the capability table, and touches no
//     kernel.
//   - pack, as WGSL, for a format the KV cache stores in (below). Packing and
//     unpacking are one piece of knowledge, whether the data is a weight or a
//     cached key.
//   - its device layout: how its blocks' fields lie on the device, as streams
//     (device_layout.h). Upload writes a weight that way; unpack reads it.
//
// A Format cannot be built wrong. It exists only at compile time (each
// formats/<format>/ defines one as a constexpr constant), it takes its type
// from its layout, so the two cannot name different types, and it has no
// state without a layout. Its constructor refuses, at compile time, a layout
// that breaks the promise device_layout.h states, and an empty unpack: a
// format with nothing to read its weights is not a format this build runs. The capability table then
// checks at compile time that every row's type is its Format's type. So a
// format the table lists always has a layout that matches it and keeps its
// promise, and preflight and upload cannot disagree on how a format runs.
//
// Unpack is one WGSL function, the same for every format:
//
//   // declared by the kernel, under this name, at the binding
//   // kernels/interface.h gives the weight a launch reads:
//   @group(0) @binding(2) var<storage, read> weights: array<u32>;
//   // supplied by the format, in formats/<format>/<format>.wgsl:
//   fn unpack(blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8>
//
//   - It returns the 32 consecutive weights of `group` in one piece: a whole
//     block of Q4_0, Q4_1 or Q8_0, one of the eight 32-weight groups of a
//     Q6_K super-block, 32 floats of F32. Every kernel steps through a row 32
//     weights at a time, whatever the format, so no kernel knows one.
//   - `blocks_in_piece` locates each stream: a stream starts after every
//     earlier stream's fields for the whole piece (device_layout.h). Kernels
//     pass it as a uniform.
//   - Weights decode in the kernel's load path, into registers, and are never
//     written back out expanded (GDSA.18).
//   - A kernel asks only for groups within a row, so every row must be a whole
//     number of groups: steps_by_groups says whether a tensor's are. A tensor
//     whose rows are not — possible only for F32, whose block is one weight —
//     is refused by routes (RowNotSteppable) and reported by preflight, so no
//     kernel is handed one.
//   - Whether adjacent lanes read adjacent addresses depends on how a kernel
//     maps lanes to groups; the layout makes it possible, and each kernel
//     states its own mapping.
//   Optimization (practice): the 32 weights come as eight vec4s, so a kernel
//   takes dot products four lanes at a time.
//   Optimization (browser): half-precision scales are read from u32 words
//   with unpack2x16float, so no format needs the shader-f16 extension, which
//   is optional in WebGPU and not every browser offers; for cross-browser
//   compatibility the harness requires no optional feature
//   (gpu/device_requirements.h).
//
// Each unpack is tested on the GPU against a CPU reference that mirrors
// ggml's dequantize_row_* for the format, in its order of operations, over
// blocks of edge and random finite values; the references live in
// tests/support, as test oracles only. Every weight must equal the
// reference's bit for bit, with two exceptions WGSL allows:
//   - a zero may lose its sign; a dot product cannot tell, so zeros compare
//     by value;
//   - where the format's decode is a multiply-add (Q4_1), WGSL and ggml alike
//     may fuse it; there a weight must equal it rounded twice or fused.
// WGSL also lets unpack2x16float flush an fp16 subnormal scale to zero, which
// ggml never does; it would turn those blocks' weights, each under 4e-4, to
// zero. The tests include such scales and require them kept, so a backend
// that flushes fails them rather than decoding silently otherwise; the
// backends they run on, Metal and Vulkan, keep them. That is the contract
// on those backends; WGSL leaves rounding and reassociation to each, so one
// that decodes otherwise fails the tests visibly. NaN and infinity are not
// inputs: no file this harness lists stores them.
//
// Pack is one WGSL function, for a format the KV cache stores in. It reads and
// writes no binding, so a kernel composes it whatever it binds, and writes
// the words it returns itself (kernels/rope/rope.h):
//
//   // supplied by the format, in formats/<format>/<format>_pack.wgsl:
//   fn pack(values: vec4<f32>) -> vec2<u32>
//
//   - It returns the two words that store four consecutive values, so a
//     cache format stores each value in 16 bits, on its own. F16 is that
//     format (formats/f16): each value saturated to ±65,504, then
//     pack2x16float. WGSL leaves converting a value outside f16's range
//     indeterminate, and the saturation makes it the nearest finite value,
//     where llama.cpp's conversion gives infinity; WGSL lets the conversion
//     round to either neighbour, and the tests require nearest-even, as for
//     unpack's scales above.
//   - A block-scaled format cannot pack four values on their own: Q8_0's
//     scale is its 32 values' largest, a reduction across invocations. So Q8_0
//     has no pack, BF16 has no format, and the graph refuses a cache
//     precision whose format has no pack, naming it, rather than run.
//   - Reading the cache back is attention's, and attention states how.
//
// Block sizes belong to the file format and are read from core/gguf; a format
// does not restate them. There is no CPU dequantizer: production never
// materialises a dequantized weight, and the CPU reference used to check
// unpack lives under tests/.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking to run-time checking — a wrong
//            pairing of type and layout does not compile.
//     C.41   A constructor should create a fully initialized object — there
//            is no Format without a layout.
//   C++ performance guidelines
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path —
//            unpack decodes into registers; nothing expands a weight in
//            memory.
//     GPU.2  Shape data for coalesced lane access — the layout makes it
//            possible; each kernel's lane mapping claims it or not.

// The weights unpack returns at a time: every kernel's step along a row.
inline constexpr std::uint64_t kUnpackGroup = 32;

// Whether a kernel can step through `tensor`'s rows, a group at a time.
[[nodiscard]] constexpr bool steps_by_groups(const gguf::TensorEntry& tensor) noexcept {
    return tensor.dimensions[0] % kUnpackGroup == 0;
}

namespace detail {
// Not constexpr: reached only when a Format would be built wrong, which makes
// its construction fail to compile.
void layout_breaks_its_promise();
void format_has_no_unpack();
}  // namespace detail

class Format {
public:
    // `pack_wgsl` is empty for a format the cache never stores in.
    consteval Format(const DeviceLayout& layout, std::string_view unpack_wgsl,
                     std::string_view pack_wgsl = {})
        : layout_(&layout), unpack_wgsl_(unpack_wgsl), pack_wgsl_(pack_wgsl) {
        if (!keeps_its_promise(layout)) detail::layout_breaks_its_promise();
        if (unpack_wgsl.empty()) detail::format_has_no_unpack();
    }

    [[nodiscard]] constexpr gguf::TensorType type() const noexcept { return layout_->type; }
    [[nodiscard]] constexpr const DeviceLayout& layout() const noexcept { return *layout_; }
    [[nodiscard]] constexpr std::string_view unpack_wgsl() const noexcept { return unpack_wgsl_; }
    [[nodiscard]] constexpr std::string_view pack_wgsl() const noexcept { return pack_wgsl_; }

private:
    const DeviceLayout* layout_;   // never null: the constructor takes a reference
    std::string_view unpack_wgsl_;
    std::string_view pack_wgsl_;
};

}  // namespace bllm::formats
