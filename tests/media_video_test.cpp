// A video's frame rules (media/video.h) against EmbeddingGemma 2's video
// processor: Google's get_aspect_ratio_preserving_size, transcribed here
// from its definition in Python's arithmetic, over sizes from a phone's to
// a panorama past Google's edge case, with sizes worked by hand beside it;
// the frames a duration gives; the frames a clip holds; and a frame filling
// its grid where an image would be letterboxed.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/media/image.h"
#include "core/media/video.h"
#include "fixtures/video/scenes.h"

namespace {

using namespace bllm;

constexpr std::uint32_t kPatch = 16;
constexpr std::uint32_t kPooling = 3;

struct Size {
    std::uint32_t height;
    std::uint32_t width;
};

// Google's rule as its Python reads, line for line: ints where it has ints,
// floats where it divides and roots.
Size google(std::uint32_t height, std::uint32_t width, std::uint32_t max_patches) {
    const double total_px = static_cast<double>(height) * width;
    const double target_px = static_cast<double>(max_patches) * kPatch * kPatch;
    const double factor = std::sqrt(target_px / total_px);
    const double ideal_height = factor * height;
    const double ideal_width = factor * width;
    const long side_mult = kPooling * kPatch;
    long target_height = static_cast<long>(std::floor(ideal_height / side_mult)) * side_mult;
    long target_width = static_cast<long>(std::floor(ideal_width / side_mult)) * side_mult;
    REQUIRE_FALSE((target_height == 0 && target_width == 0));
    const long max_side_length = (max_patches / (kPooling * kPooling)) * side_mult;
    if (target_height == 0) {
        target_height = side_mult;
        target_width = std::min(static_cast<long>(std::floor(static_cast<double>(width) / height)) * side_mult,
                                max_side_length);
    } else if (target_width == 0) {
        target_width = side_mult;
        target_height = std::min(static_cast<long>(std::floor(static_cast<double>(height) / width)) * side_mult,
                                 max_side_length);
    }
    REQUIRE(static_cast<double>(target_height) * target_width <= target_px);
    return {static_cast<std::uint32_t>(target_height), static_cast<std::uint32_t>(target_width)};
}

}  // namespace

TEST_CASE("a frame's grid is Google's, at most 1,260 patches, enlarged or shrunk alike") {
    const std::vector<Size> sizes = {{720, 1280}, {360, 640},  {1080, 1920}, {1280, 720}, {1024, 1024},
                                     {480, 640},  {2160, 3840}, {48, 48},     {1, 1},      {60, 4000},
                                     {60, 16384}, {16384, 60}, {240, 320},   {1, 16384},  {16384, 16384}};
    for (const Size& s : sizes) {
        CAPTURE(s.width);
        CAPTURE(s.height);
        const media::ImageGrid grid = media::frame_grid(s.width, s.height, kPatch, kPooling, media::kFrameTokens);
        const Size want = google(s.height, s.width, media::kFrameTokens * kPooling * kPooling);
        CHECK(grid.width == want.width);
        CHECK(grid.height == want.height);
        CHECK(grid.width % 48 == 0);
        CHECK(grid.height % 48 == 0);
        CHECK(grid.columns == grid.width / kPatch);
        CHECK(grid.rows == grid.height / kPatch);
        CHECK(grid.columns * grid.rows <= 1260);
        CHECK(grid.tokens == (grid.columns / kPooling) * (grid.rows / kPooling));
        CHECK(grid.tokens <= media::kFrameTokens);
    }
    // Worked by hand: 16:9 at any size, 720 × 384, 120 tokens; a square,
    // 528 × 528, 121; 4:3, 624 × 480, 130; a 1 × 1 frame enlarged to the
    // square's; past about 140 to 1, 48 high and the budget's side long.
    const auto grid = [](std::uint32_t w, std::uint32_t h) {
        return media::frame_grid(w, h, kPatch, kPooling, media::kFrameTokens);
    };
    for (const auto& [w, h] : {std::pair{1280u, 720u}, {640u, 360u}, {3840u, 2160u}}) {
        CHECK(grid(w, h).width == 720);
        CHECK(grid(w, h).height == 384);
        CHECK(grid(w, h).tokens == 120);
    }
    CHECK(grid(1024, 1024).width == 528);
    CHECK(grid(1024, 1024).tokens == 121);
    CHECK(grid(640, 480).width == 624);
    CHECK(grid(640, 480).height == 480);
    CHECK(grid(640, 480).tokens == 130);
    CHECK(grid(1, 1).width == 528);
    CHECK(grid(16384, 60).width == 6720);
    CHECK(grid(16384, 60).height == 48);
    CHECK(grid(16384, 60).tokens == 140);
    CHECK(grid(60, 16384).width == 48);
    CHECK(grid(60, 16384).height == 6720);
}

TEST_CASE("a frame fills its grid, stretched, where an image is letterboxed") {
    const media::ImageGrid grid = media::frame_grid(1280, 720, kPatch, kPooling, media::kFrameTokens);
    const media::ImageFit fit = media::frame_fit(grid);
    CHECK(fit.width == grid.width);
    CHECK(fit.height == grid.height);
    CHECK(fit.x == 0);
    CHECK(fit.y == 0);

    // The red ball's first frame, a white field: filled, its left column is
    // white; letterboxed as image_fit places it, black.
    const auto& scene = fixtures::video::kScenes[0];
    std::vector<std::uint8_t> rgba(std::size_t{scene.width} * scene.height * 4);
    for (std::uint32_t y = 0; y < scene.height; ++y) {
        for (std::uint32_t x = 0; x < scene.width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                rgba[(std::size_t{y} * scene.width + x) * 4 + c] = fixtures::video::pixel(scene, 0, x, y, c);
            }
        }
    }
    const auto left_column = [&](const media::ImageFit& placed) {
        media::ImageResampler r;
        REQUIRE(r.begin(scene.width, scene.height, grid, placed, kPatch, media::kFrameTokens) ==
                media::ImageError::Ok);
        REQUIRE(r.accept(rgba, 0) == media::ImageError::Ok);
        std::vector<std::uint8_t> canvas(std::size_t{grid.width} * grid.height * 3);
        REQUIRE(r.canvas(canvas) == media::ImageError::Ok);
        return canvas[(std::size_t{grid.height / 2} * grid.width) * 3];
    };
    CHECK(left_column(fit) == 255);
    const media::ImageFit boxed = media::image_fit(scene.width, scene.height, grid);
    REQUIRE(boxed.x > 0);
    CHECK(left_column(boxed) == 0);
}

TEST_CASE("a duration gives a frame a second, at least one; none for no duration") {
    CHECK(media::frame_count(0.0) == 0);
    CHECK(media::frame_count(-1.0) == 0);
    CHECK(media::frame_count(std::numeric_limits<double>::quiet_NaN()) == 0);
    CHECK(media::frame_count(std::numeric_limits<double>::infinity()) == 0);
    CHECK(media::frame_count(0.5) == 1);
    CHECK(media::frame_count(1.0) == 1);
    CHECK(media::frame_count(4.99) == 4);
    CHECK(media::frame_count(5.0) == 5);
    CHECK(media::frame_count(300.0) == 300);
}

TEST_CASE("a clip holds the frames whose tokens and two ends fit the room, at most four") {
    CHECK(media::frames_per_clip(120, 510) == 4);
    CHECK(media::frames_per_clip(121, 510) == 4);
    CHECK(media::frames_per_clip(130, 510) == 3);
    CHECK(media::frames_per_clip(140, 510) == 3);
    CHECK(media::frames_per_clip(10, 510) == 4);
    CHECK(media::frames_per_clip(140, 100) == 0);
}
