#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/gguf/types.h"

namespace bllm::formats {

// Axis B: changes with a new weight format.
//
// How a format's blocks lie on the device. A file stores each block whole:
// a Q4_0 block is a 2-byte scale and 16 bytes of nibbles, 18 bytes in all.
// On the device a piece of a weight holds its blocks' fields as streams
// instead: every block's nibbles, in block order, then every block's scales.
// Upload writes a piece that way (residency/piece_writer.h) and a format's
// unpack reads it that way.
//
// Most formats' streams are fields of the stored block, copied as they are.
// A format whose stored bits do not group the way unpack reads them repacks
// each block first, at upload, and its streams are fields of the repacked
// block, of the same size; repack writes them straight into their streams.
// Q6_K is the one that does: its stored block keeps each weight's low four
// bits in a byte shared with another 32-weight group and its high two bits in
// a byte shared with three others, so an unpack of one group read 18 words,
// 72 bytes, for 26.25 bytes of weights. Repacked, each group's bits lie
// together — 16 bytes of low nibbles, 8 of high bit pairs — and it reads 8
// words, 32 bytes.
// Optimization (practice): CDSA.32 — the transform is done once, as the
// block is written, so every token's read of the weight costs less. The
// repack moves eight bytes at a time with shifts and masks, no call per
// field, and costs upload nothing measurable (q6_k_repack.cpp).
//
// A layout is part of its format: each Format names its own (format.h), and
// the capability table, which lists the formats this build runs, is the one
// place a layout is found. This file holds the layouts' shape, the layouts
// the listed models need, and the compile-time check that each keeps its
// promise; it is not a second table of what is supported.
//
// Optimization (browser): WGSL reads a storage buffer in aligned 32-bit
// words. An 18-byte block straddles them, so reading one stored whole takes
// two loads and a shift for every field that crosses a word; as streams,
// every field of every block starts on the alignment its width needs.
// Optimization (practice): adjacent invocations read adjacent blocks, and
// with each field a stream they touch adjacent addresses, the access GPU.2
// asks for. llama.cpp's OpenCL backend converts Q4_0 and Q8_0 weights to the
// same struct of arrays as it uploads them (GGML_OPENCL_SOA_Q).
//
// A layout's streams cover its block exactly, each byte in one stream. At most
// the last stream's width may leave a stream unaligned, so a piece of n blocks
// takes exactly its stored bytes rounded up to 4 — the length the residency
// plan binds it at (plan.h) — whatever n is, odd or even. That is checked at
// compile time below, for every layout here (P.5).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking to run-time checking — each layout's
//            promise is a static_assert, not a test.
//     Con.5  Use constexpr for values that can be computed at compile time —
//            the layouts are constexpr constants.
//   C++ performance guidelines
//     GPU.2  Shape data for coalesced lane access before tuning the kernel —
//            the streams.
//     CACHE.4 Choose AoS, SoA, or AoSoA by which fields the hot loop touches —
//            a piece is SoA: an unpack's lanes read one field across many
//            blocks.
//     CDSA.32 Transform static data once into the layout its consumer reads —
//            once, at upload, never per token; each layout a named constant,
//            checked against the stored block it is transformed from. It asks
//            for a version on a persisted copy; none is kept — the transformed
//            bytes live in device memory for one page, so the build that
//            writes a layout is the build that reads it. The layout is a
//            contract with one consumer, its format's unpack: each unpack is
//            tested, as it lands, against ggml's dequantization of the same
//            stored blocks. Coalescing itself is not claimed: a timeline
//            shows end to end time, not memory transactions, so what is
//            claimed is a measured end-to-end difference, named as that.
//     GDSA.18 Store numbers as block-scaled codes decoded in the load path —
//            the codes stay as the file holds them and an unpack decodes them
//            as it loads, never into a full-precision buffer; the scales are a
//            stream of their own, where lanes fetch them; and a layout is
//            named apart from its numeric type, which Format pairs with it.
//     EMB.6  Push computation to compile time with constexpr and consteval —
//            the check is consteval, so it cannot drift to run time.

// The bounds every layout here keeps, which the piece writer sizes what it
// holds between chunks by: one block, and a few bytes of each stream.
inline constexpr std::size_t kMaxBlockBytes = 210;   // Q6_K's
inline constexpr std::size_t kMaxStreams = 4;        // Q6_K's

// One field of a block — the stored block's, or, for a layout that repacks,
// the repacked block's: where it lies, and how wide it is.
struct Stream {
    std::uint16_t offset;
    std::uint16_t width;
};

// Repacks whole blocks, `blocks` holding a whole number of them, writing
// block i's stream s at streams[s] + i × that stream's width.
using Repack = void (*)(std::span<const std::byte> blocks, const std::array<std::byte*, kMaxStreams>& streams);

struct DeviceLayout {
    gguf::TensorType type;
    std::uint32_t block_bytes;
    std::span<const Stream> streams;   // in device order
    Repack repack = nullptr;           // null: streams are the stored block's fields
};

// The layouts of the formats the listed models use: F32 for norms, Q4_0 and
// Q4_1 for most weights, Q8_0 and Q6_K for embeddings, F16 for the KV cache
// and the weights some files store at half precision. ggml's block structs
// (ggml-common.h) give each field's place in the stored block. Each format's
// Format names its own, once it is implemented.
namespace detail {
inline constexpr Stream kF32[] = {{0, 4}};
inline constexpr Stream kF16[] = {{0, 2}};
inline constexpr Stream kQ4_0[] = {{2, 16}, {0, 2}};                     // qs, then d
inline constexpr Stream kQ4_1[] = {{4, 16}, {0, 4}};                     // qs, then d and m
inline constexpr Stream kQ8_0[] = {{2, 32}, {0, 2}};                     // qs, then d
// Of the repacked block: each group's low nibbles, its high bit pairs, then
// the scales and d as stored (formats/q6_k/q6_k_repack.cpp).
inline constexpr Stream kQ6_K[] = {{0, 128}, {128, 64}, {192, 16}, {208, 2}};
void repack_q6_k(std::span<const std::byte> blocks, const std::array<std::byte*, kMaxStreams>& streams);
}  // namespace detail

inline constexpr DeviceLayout kF32Layout{gguf::TensorType::F32, 4, detail::kF32};
inline constexpr DeviceLayout kF16Layout{gguf::TensorType::F16, 2, detail::kF16};
inline constexpr DeviceLayout kQ4_0Layout{gguf::TensorType::Q4_0, 18, detail::kQ4_0};
inline constexpr DeviceLayout kQ4_1Layout{gguf::TensorType::Q4_1, 20, detail::kQ4_1};
inline constexpr DeviceLayout kQ8_0Layout{gguf::TensorType::Q8_0, 34, detail::kQ8_0};
inline constexpr DeviceLayout kQ6_KLayout{gguf::TensorType::Q6_K, 210, detail::kQ6_K, detail::repack_q6_k};

// Whether `layout` keeps the promise above: its streams cover the block, its
// block size is the file format's, only its last stream may be unaligned, and
// it is within the bounds the piece writer holds.
[[nodiscard]] consteval bool keeps_its_promise(const DeviceLayout& layout) {
    if (layout.block_bytes > kMaxBlockBytes || layout.streams.size() > kMaxStreams) return false;
    std::uint32_t covered = 0;
    bool seen[256] = {};
    for (std::size_t i = 0; i < layout.streams.size(); ++i) {
        const Stream s = layout.streams[i];
        if (i + 1 < layout.streams.size() && s.width % 4 != 0) return false;
        for (std::uint32_t b = s.offset; b < std::uint32_t{s.offset} + s.width; ++b) {
            if (b >= layout.block_bytes || b >= 256 || seen[b]) return false;
            seen[b] = true;
        }
        covered += s.width;
    }
    // The file format's block size, looked up by value: GCC does not treat a
    // pointer comparison here as a constant expression under sanitizers.
    std::uint32_t stored_bytes = 0;
    for (const gguf::FormatLayout& stored : gguf::kFormatLayouts) {
        if (stored.type == layout.type) stored_bytes = stored.block_bytes;
    }
    return covered == layout.block_bytes && stored_bytes == layout.block_bytes;
}

static_assert(keeps_its_promise(kF32Layout) && keeps_its_promise(kF16Layout) && keeps_its_promise(kQ4_0Layout) &&
              keeps_its_promise(kQ4_1Layout) && keeps_its_promise(kQ8_0Layout) &&
              keeps_its_promise(kQ6_KLayout));

}  // namespace bllm::formats
