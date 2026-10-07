#include "core/sampler/sampler.h"

#include <cmath>
#include <cstring>

#include "bllm/shaders_generated.h"
#include "core/kernels/topk/topk.h"

namespace bllm::sampler {
namespace {

// Binding 1, as draw.wgsl's Draw lays it out.
struct Constants {
    std::uint32_t count;
};

kernels::Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

SettingsResult refuse(std::string subject) { return {false, std::move(subject)}; }

}  // namespace

SettingsResult check(const policy::SamplingSettings& s) {
    if (s.top_k < 1 || s.top_k > kernels::kCandidates) {
        return refuse("top_k " + std::to_string(s.top_k) + " is outside 1 to " + std::to_string(kernels::kCandidates));
    }
    if (!std::isfinite(s.temperature) || s.temperature < 0) {
        return refuse("temperature " + std::to_string(s.temperature) + " is not a finite number of 0 or more");
    }
    if (!(s.top_p > 0 && s.top_p <= 1)) {
        return refuse("top_p " + std::to_string(s.top_p) + " is outside (0, 1]");
    }
    if (!(s.min_p >= 0 && s.min_p < 1)) {
        return refuse("min_p " + std::to_string(s.min_p) + " is outside [0, 1)");
    }
    return {};
}

kernels::Launch draw_launch(const residency::BufferRange& candidates, const residency::BufferRange& sampled) {
    const Constants constants{kernels::kCandidates};
    std::vector<std::byte> bytes(sizeof constants);
    std::memcpy(bytes.data(), &constants, sizeof constants);
    return kernels::Launch{shaders::draw,
                           nullptr,
                           std::move(bytes),
                           {whole(candidates), whole(sampled)},
                           kernels::kCandidates,
                           kernels::kCandidates,
                           kernels::Rows::LastToken,
                           {}};
}

}  // namespace bllm::sampler
