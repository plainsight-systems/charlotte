#include "core/media/video.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace bllm::media {

std::uint32_t frame_count(double duration) noexcept {
    if (!std::isfinite(duration) || duration <= 0) return 0;
    // Google's max(1, int(duration × fps)), truncated as Python's int().
    const double frames = std::floor(duration * kFramesPerSecond);
    if (frames >= static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    return std::max<std::uint32_t>(1, static_cast<std::uint32_t>(frames));
}

ImageGrid frame_grid(std::uint32_t width, std::uint32_t height, std::uint32_t patch, std::uint32_t pooling,
                     std::uint32_t tokens) noexcept {
    // Google's get_aspect_ratio_preserving_size, in f64 as Python computes
    // it: the sides' product and the budget's are integers exact in f64, so
    // their quotient rounds once, as Python's true division does.
    const std::uint64_t side = std::uint64_t{patch} * pooling;
    const std::uint64_t max_patches = std::uint64_t{tokens} * pooling * pooling;
    const double target = static_cast<double>(max_patches * patch * patch);
    const double total = static_cast<double>(std::uint64_t{width} * height);
    const double factor = std::sqrt(target / total);
    // Each ideal side at most about 72,700 px, at 16,384 × 1, so the floors
    // fit (ES.46).
    const auto floor_to_side = [side](double ideal) {
        return static_cast<std::uint64_t>(std::floor(ideal / static_cast<double>(side))) * side;
    };
    std::uint64_t h = floor_to_side(factor * height);
    std::uint64_t w = floor_to_side(factor * width);
    const std::uint64_t max_side = std::uint64_t{tokens} * side;
    if (h == 0) {
        h = side;
        w = std::min(static_cast<std::uint64_t>(std::floor(static_cast<double>(width) / height)) * side, max_side);
    } else if (w == 0) {
        w = side;
        h = std::min(static_cast<std::uint64_t>(std::floor(static_cast<double>(height) / width)) * side, max_side);
    }
    ImageGrid grid{};
    grid.width = static_cast<std::uint32_t>(w);
    grid.height = static_cast<std::uint32_t>(h);
    grid.columns = grid.width / patch;
    grid.rows = grid.height / patch;
    grid.tokens = (grid.columns / pooling) * (grid.rows / pooling);
    return grid;
}

ImageFit frame_fit(const ImageGrid& grid) noexcept { return {grid.width, grid.height, 0, 0}; }

std::uint32_t frames_per_clip(std::uint32_t tokens, std::uint32_t room) noexcept {
    return std::min(kClipFrames, room / (tokens + 2));
}

}  // namespace bllm::media
