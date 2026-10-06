#include "core/kernels/rope/rope.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::size_t kMaxPairs = 128;   // d / 2 for d up to 256

// Binding 1, as rope.wgsl's Rope lays it out: turns at offset 16.
struct Constants {
    std::uint32_t slots;
    float epsilon;
    std::uint32_t padding[2];
    std::array<float, kMaxPairs> turns;
};
static_assert(sizeof(Constants) == 528);

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

Launch rope_launch(const RopeLaunch& r) {
    const model::LayerDescription& layer = r.layer;
    const std::uint32_t d = layer.head_dimension;

    Constants constants{r.cache.slots, r.epsilon, {}, {}};
    // Each pair's θ_k / 2π, in f64, rounded once to f32: WGSL bounds pow
    // only loosely (rope.h).
    for (std::uint32_t k = 0; k < d / 2; ++k) {
        const double theta = std::pow(static_cast<double>(layer.rope_base), -2.0 * k / d);
        constants.turns[k] = static_cast<float>(theta / (2 * std::numbers::pi));
    }
    const auto bytes = std::as_bytes(std::span(&constants, 1));

    // Without QK-norm or factors, their bindings take the key buffer, which
    // this kernel only reads, so no buffer is bound writable and read-only.
    const Binding key = whole(r.key);
    const bool qk_norm = r.query_norm != nullptr;
    return Launch{
        shaders::rope,
        nullptr,
        std::vector<std::byte>(bytes.begin(), bytes.end()),
        {qk_norm ? whole(*r.query_norm) : key, qk_norm ? whole(*r.key_norm) : key,
         r.factors != nullptr ? whole(*r.factors) : key, whole(r.query), key, whole(r.value),
         whole(r.cache.keys), whole(r.cache.values)},
        (layer.query_heads + 2 * layer.key_value_heads) * (d / 8),
        kWorkgroupSize,
        Rows::EveryToken,
        {{"head_dimension", static_cast<double>(d)},
         {"query_heads", static_cast<double>(layer.query_heads)},
         {"key_value_heads", static_cast<double>(layer.key_value_heads)},
         {"halves", r.pairing == model::RotaryPairing::Halves ? 1.0 : 0.0},
         {"qk_norm", qk_norm ? 1.0 : 0.0},
         {"factors", r.factors != nullptr ? 1.0 : 0.0}},
        r.cache_format,
    };
}

}  // namespace bllm::kernels
