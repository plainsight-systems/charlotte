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
// rounded once to f32: WGSL bounds pow only loosely (rope.h).
std::vector<std::byte> constants_of(const model::LayerDescription& layer, std::uint32_t slots, float epsilon) {
    const std::uint32_t d = layer.head_dimension;
    Constants constants{slots, epsilon, {}, {}};
    for (std::uint32_t k = 0; k < d / 2; ++k) {
        const double theta = std::pow(static_cast<double>(layer.rope_base), -2.0 * k / d);
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
        constants_of(layer, r.cache.slots, r.epsilon),
        {qk_norm ? whole(*r.query_norm) : key, qk_norm ? whole(*r.key_norm) : key,
         r.factors != nullptr ? whole(*r.factors) : key, whole(r.query), key, whole(r.value),
         whole(r.cache.keys), whole(r.cache.values)},
        (layer.query_heads + 2 * layer.key_value_heads) * (d / 8),
        kWorkgroupSize,
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
        constants_of(layer, 1, r.epsilon),
        {query_gain, whole(*r.key_norm), r.factors != nullptr ? whole(*r.factors) : query_gain, whole(r.query),
         whole(r.key), whole(r.value)},
        (layer.query_heads + 2 * layer.key_value_heads) * (d / 8),
        kWorkgroupSize,
        Rows::EveryToken,
        shape_of(layer, r.pairing, true, r.factors != nullptr, r.value_norm),
    };
}

}  // namespace bllm::kernels
