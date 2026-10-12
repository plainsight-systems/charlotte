// An image's preparation (media/image.h) against llama.cpp's: the grid its
// rule gives for each size, and the bytes its resize leaves, byte for byte
// by hash, for the images of tests/fixtures/vision/scenes.h, kept by
// tools/make_reference_vision.sh. Then the bands, the patches and the
// refusals.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "core/media/image.h"
#include "fixtures/vision/scenes.h"

namespace {

#include "fixtures/reference/vision_prepare.inc"

using namespace bllm;
using fixtures::vision::fnv1a;
using fixtures::vision::kScenes;
using fixtures::vision::pixel;
using fixtures::vision::Scene;

constexpr std::uint32_t kPatch = 16;
constexpr std::uint32_t kPooling = 3;
constexpr media::ImageBudget kBudget{70, 280};

media::ImageGrid grid_of(std::uint32_t w, std::uint32_t h) {
    return media::image_grid(w, h, kPatch, kPooling, kBudget);
}

std::vector<std::uint8_t> rgba_rows(const Scene& scene, std::uint32_t first, std::uint32_t rows) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(scene.width) * rows * 4);
    for (std::uint32_t y = 0; y < rows; ++y) {
        for (std::uint32_t x = 0; x < scene.width; ++x) {
            std::uint8_t* p = &out[(static_cast<std::size_t>(y) * scene.width + x) * 4];
            for (std::uint32_t c = 0; c < 3; ++c) p[c] = pixel(scene, x, first + y, c);
            p[3] = static_cast<std::uint8_t>((x * 7 + y) & 0xff);   // alpha, which is dropped
        }
    }
    return out;
}

// Prepares `scene` in bands of `band` rows; the resampler, ready for its
// canvas and patches.
media::ImageResampler resample(const Scene& scene, std::uint32_t band) {
    media::ImageResampler r;
    const media::ImageGrid grid = grid_of(scene.width, scene.height);
    REQUIRE(r.begin(scene.width, scene.height, grid, media::image_fit(scene.width, scene.height, grid), kPatch,
                    kBudget.most) == media::ImageError::Ok);
    for (std::uint32_t row = 0; row < scene.height; row += band) {
        const std::uint32_t rows = std::min(band, scene.height - row);
        const auto bytes = rgba_rows(scene, row, rows);
        REQUIRE(r.accept(bytes, row) == media::ImageError::Ok);
    }
    return r;
}

std::uint64_t canvas_hash(media::ImageResampler& r, const media::ImageGrid& grid) {
    std::vector<std::uint8_t> canvas(static_cast<std::size_t>(grid.width) * grid.height * 3);
    REQUIRE(r.canvas(canvas) == media::ImageError::Ok);
    return fnv1a(canvas.data(), canvas.size());
}

const Scene& scene_named(std::string_view name) {
    for (const Scene& s : kScenes) {
        if (s.name == name) return s;
    }
    FAIL("no scene " << name);
    return kScenes[0];
}

}  // namespace

TEST_CASE("an image's grid is llama.cpp's for every size") {
    for (const PreparedGrid& g : kGrids) {
        CAPTURE(g.source_width);
        CAPTURE(g.source_height);
        const media::ImageGrid grid = grid_of(g.source_width, g.source_height);
        CHECK(grid.width == g.width);
        CHECK(grid.height == g.height);
        CHECK(grid.columns == g.width / kPatch);
        CHECK(grid.rows == g.height / kPatch);
        CHECK(grid.tokens == (g.width / 48) * (g.height / 48));
    }
}

TEST_CASE("an image prepared is llama.cpp's, byte for byte") {
    for (const PreparedImage& p : kPrepared) {
        CAPTURE(p.name);
        const Scene& scene = scene_named(p.name);
        const media::ImageGrid grid = grid_of(scene.width, scene.height);
        if (p.width == 0) {
            // llama.cpp refuses it; so does the resampler, past the budget.
            CHECK(grid.tokens > kBudget.most);
            media::ImageResampler r;
            CHECK(r.begin(scene.width, scene.height, grid, media::image_fit(scene.width, scene.height, grid), kPatch, kBudget.most) == media::ImageError::OverBudget);
            continue;
        }
        REQUIRE(grid.width == p.width);
        REQUIRE(grid.height == p.height);
        // Bands of every row at once, of one row, and of seven, give the same
        // bytes; the large images in the page's 4 MiB bands alone.
        const std::uint32_t page_band = std::max<std::uint32_t>(1, (4u << 20) / (4 * scene.width));
        std::vector<std::uint32_t> bands{page_band};
        if (static_cast<std::uint64_t>(scene.width) * scene.height <= 1'000'000) {
            bands.push_back(scene.height);
            bands.push_back(1);
            bands.push_back(7);
        }
        for (const std::uint32_t band : bands) {
            CAPTURE(band);
            auto r = resample(scene, band);
            CHECK(canvas_hash(r, grid) == p.hash);
        }
    }
}

TEST_CASE("the patches are the canvas cut in the patch embedding's layout") {
    const Scene& scene = scene_named("red_circle");
    const media::ImageGrid grid = grid_of(scene.width, scene.height);
    auto r = resample(scene, 16);
    std::vector<std::uint8_t> canvas(static_cast<std::size_t>(grid.width) * grid.height * 3);
    REQUIRE(r.canvas(canvas) == media::ImageError::Ok);
    const std::size_t per_patch = 3 * kPatch * kPatch;
    std::vector<float> patches(static_cast<std::size_t>(grid.columns) * grid.rows * per_patch);
    REQUIRE(r.patches(patches) == media::ImageError::Ok);
    std::size_t mismatched = 0;
    for (std::uint32_t py = 0; py < grid.rows; ++py) {
        for (std::uint32_t px = 0; px < grid.columns; ++px) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                for (std::uint32_t ky = 0; ky < kPatch; ++ky) {
                    for (std::uint32_t kx = 0; kx < kPatch; ++kx) {
                        const std::size_t x = px * kPatch + kx;
                        const std::size_t y = py * kPatch + ky;
                        const std::uint8_t u = canvas[(y * grid.width + x) * 3 + c];
                        const float expected = static_cast<float>(u) / 255.0f * 2.0f - 1.0f;
                        const float got = patches[(static_cast<std::size_t>(py) * grid.columns + px) * per_patch +
                                                  (c * kPatch + ky) * kPatch + kx];
                        if (got != expected) ++mismatched;
                    }
                }
            }
        }
    }
    CHECK(mismatched == 0);
    // The letterbox: the red circle's 4:3 fitted inside 480 × 384 leaves
    // black rows, −1, above and below; its corners are white, 1.
    const media::ImageFit fit = media::image_fit(scene.width, scene.height, grid);
    CHECK(fit.width == 480);
    CHECK(fit.height == 360);
    CHECK(fit.y == 12);
    CHECK(patches[0] == -1.0f);
}

TEST_CASE("an aspect past the budget's most to 1 is refused, at it is not") {
    const media::ImageGrid at = grid_of(13440, 48);
    CHECK(at.tokens == 280);
    media::ImageResampler r;
    CHECK(r.begin(13440, 48, at, media::image_fit(13440, 48, at), kPatch, kBudget.most) == media::ImageError::Ok);
    const media::ImageGrid past = grid_of(14000, 48);
    CHECK(past.tokens == 285);
    CHECK(r.begin(14000, 48, past, media::image_fit(14000, 48, past), kPatch, kBudget.most) == media::ImageError::OverBudget);
}

TEST_CASE("each refusal is named") {
    media::ImageResampler r;
    const media::ImageGrid grid = grid_of(10, 10);
    // A source refused before its fit is read.
    CHECK(r.begin(0, 10, grid, media::ImageFit{}, kPatch, kBudget.most) == media::ImageError::Empty);
    CHECK(r.begin(media::kMaxSide + 1, 10, grid, media::ImageFit{}, kPatch, kBudget.most) ==
          media::ImageError::TooLarge);

    REQUIRE(r.begin(10, 10, grid, media::image_fit(10, 10, grid), kPatch, kBudget.most) == media::ImageError::Ok);
    std::vector<float> patches(static_cast<std::size_t>(grid.columns) * grid.rows * 3 * kPatch * kPatch);
    CHECK(r.patches(patches) == media::ImageError::Incomplete);
    const std::vector<std::uint8_t> row(10 * 4, 200);
    const std::vector<std::uint8_t> ragged(10 * 4 + 1, 200);
    CHECK(r.accept(row, 1) == media::ImageError::BadBand);      // not the next row
    CHECK(r.accept(ragged, 0) == media::ImageError::BadBand);   // not whole rows
    for (std::uint32_t y = 0; y < 10; ++y) REQUIRE(r.accept(row, y) == media::ImageError::Ok);
    CHECK(r.accept(row, 10) == media::ImageError::BadBand);     // past the last row
    CHECK(r.patches(patches) == media::ImageError::Ok);
    CHECK(r.accept(row, 10) == media::ImageError::BadBand);     // after the patches
}
