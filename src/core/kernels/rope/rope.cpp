#include "core/kernels/rope/rope.h"

#include <array>
#include <cmath>
#include <string_view>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;

// The workgroup: 64 invocations, or fewer where a row is not a whole number
// of 64 — a vision layer's 36 heads of 64 are 288 invocations a row — so
// every invocation of a workgroup still works on one token, and its heads
// still all rotate or are all value heads (rope.h): the largest power of two
// from 64 down to a head's d / 8 lanes that divides both a row and its
// rotating heads.
std::uint32_t workgroup_for(const model::LayerDescription& layer) {
    const std::uint32_t lanes = layer.head_dimension / 8;
    const std::uint32_t row = (layer.query_heads + 2 * layer.key_value_heads) * lanes;
    const std::uint32_t rotating = (layer.query_heads + layer.key_value_heads) * lanes;
    std::uint32_t size = kWorkgroupSize;
    while (size > lanes && (row % size != 0 || rotating % size != 0)) size /= 2;
    return size;
}
constexpr std::size_t kMaxPairs = 256;   // d / 2 for d up to 512

// Binding 1, as rope.wgsl's Rope lays it out: turns at offset 16.
struct Constants {
    std::uint32_t slots;
    float epsilon;
    std::uint32_t padding[2];
    std::array<float, kMaxPairs> turns;
};
static_assert(sizeof(Constants) == 1040);

// Two embedded sources, one after the other, joined at compile time: rope.wgsl,
// what both forms share, then a form's entry point (rope.h). The program
// compares kernels by their text, so each form is one pipeline per shape.
template <std::size_t N>
struct Joined {
    std::array<char, N> chars{};
    [[nodiscard]] constexpr std::string_view view() const { return {chars.data(), N}; }
};

template <std::size_t N>
constexpr Joined<N> join(std::string_view shared, std::string_view form) {
    Joined<N> out;
    std::size_t at = 0;
    for (const char c : shared) out.chars[at++] = c;
    for (const char c : form) out.chars[at++] = c;
    return out;
}

constexpr auto kCacheForm = join<shaders::rope.size() + shaders::rope_cache.size()>(shaders::rope, shaders::rope_cache);
constexpr auto kEncodeForm =
    join<shaders::rope.size() + shaders::rope_encode.size()>(shaders::rope, shaders::rope_encode);

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

// The launch's constants: its slots, ε, and each pair's θ_k / 2π, in f64,
// rounded once to f32: WGSL bounds pow only loosely (rope.h). Axial, each
// half of a head is rotated as a head of d / 2 by its own position, so its
// d / 4 pairs' frequencies are base^(−2k / (d / 2)), shared by both halves.
std::vector<std::byte> constants_of(const model::LayerDescription& layer, model::RotaryPairing pairing,
                                    std::uint32_t slots, float epsilon) {
    const std::uint32_t d = layer.head_dimension;
    const std::uint32_t rotated = pairing == model::RotaryPairing::Axial ? d / 2 : d;
    Constants constants{slots, epsilon, {}, {}};
    for (std::uint32_t k = 0; k < rotated / 2; ++k) {
        const double theta = std::pow(static_cast<double>(layer.rope_base), -2.0 * k / rotated);
        constants.turns[k] = static_cast<float>(theta / (2 * std::numbers::pi));
    }
    const auto bytes = std::as_bytes(std::span(&constants, 1));
    return {bytes.begin(), bytes.end()};
}

// The overrides both forms share: the shape, the pairing, which norms and
// factors apply, and the angle tables' size, d / 2.
std::vector<Override> shape_of(const model::LayerDescription& layer, model::RotaryPairing pairing, bool qk_norm,
                               bool factors, bool value_norm) {
    const std::uint32_t d = layer.head_dimension;
    return {{"head_dimension", static_cast<double>(d)},
            {"query_heads", static_cast<double>(layer.query_heads)},
            {"key_value_heads", static_cast<double>(layer.key_value_heads)},
            {"halves", pairing == model::RotaryPairing::Halves ? 1.0 : 0.0},
            {"axial", pairing == model::RotaryPairing::Axial ? 1.0 : 0.0},
            {"qk_norm", qk_norm ? 1.0 : 0.0},
            {"factors", factors ? 1.0 : 0.0},
            {"value_norm", value_norm ? 1.0 : 0.0},
            {"pairs_per_head", static_cast<double>(d / 2)}};
}

}  // namespace

Launch rope_launch(const RopeLaunch& r) {
    const model::LayerDescription& layer = r.layer;
    const std::uint32_t d = layer.head_dimension;

    // Without QK-norm or factors, their bindings take the key buffer, which
    // this kernel only reads, so no buffer is bound writable and read-only.
    const Binding key = whole(r.key);
    const bool qk_norm = r.query_norm != nullptr;
    return Launch{
        kCacheForm.view(),
        nullptr,
        constants_of(layer, r.pairing, r.cache.slots, r.epsilon),
        {qk_norm ? whole(*r.query_norm) : key, qk_norm ? whole(*r.key_norm) : key,
         r.factors != nullptr ? whole(*r.factors) : key, whole(r.query), key, whole(r.value),
         whole(r.cache.keys), whole(r.cache.values)},
        (layer.query_heads + 2 * layer.key_value_heads) * (d / 8),
        workgroup_for(layer),
        Rows::EveryToken,
        shape_of(layer, r.pairing, qk_norm, r.factors != nullptr, false),
        r.cache_format,
    };
}

Launch rope_encode_launch(const RopeEncodeLaunch& r) {
    const model::LayerDescription& layer = r.layer;
    const std::uint32_t d = layer.head_dimension;
    // Without factors, their binding takes the query gain, read-only there
    // and here: every buffer the form writes is bound writable once.
    const Binding query_gain = whole(*r.query_norm);
    return Launch{
        kEncodeForm.view(),
        nullptr,
        constants_of(layer, r.pairing, 1, r.epsilon),
        {query_gain, whole(*r.key_norm), r.factors != nullptr ? whole(*r.factors) : query_gain, whole(r.query),
         whole(r.key), whole(r.value)},
        (layer.query_heads + 2 * layer.key_value_heads) * (d / 8),
        workgroup_for(layer),
        Rows::EveryToken,
        shape_of(layer, r.pairing, true, r.factors != nullptr, r.value_norm),
    };
}

}  // namespace bllm::kernels
