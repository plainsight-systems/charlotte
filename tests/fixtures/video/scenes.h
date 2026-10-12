#pragma once

// The videos the vision reference tool gives llama.cpp
// (tools/reference_vision) and the tests give Charlotte
// (tests/media_video_test.cpp, tests/gpu/video_embed_test.cpp), each frame
// computed from its coordinates and its second, so both see the same pixels
// with no video file, decoder or licence between them, as the image scenes
// are (tests/fixtures/vision/scenes.h). Four scenes plain enough to name in
// a query, each something moving or changing across its frames, each at
// 1,280 × 720, a 16:9 frame Google's rule shrinks to 720 × 384, 120 tokens
// (media/video.h): a red ball crossing, a blue square growing, green stripes
// scrolling, a black line sweeping. Frame t is the frame at t s, 1 a second,
// t from 0 to frames − 1. Integer arithmetic only, so every machine draws
// the same frame.

#include <array>
#include <cstdint>
#include <string_view>

namespace bllm::fixtures::video {

enum class Kind {
    RedBall,
    BlueSquare,
    GreenStripes,
    BlackLine,
};

struct Scene {
    std::string_view name;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t frames;   // seconds, at 1 a second
    Kind kind;
};

inline constexpr std::array<Scene, 4> kScenes{{
    {"red_ball", 1280, 720, 4, Kind::RedBall},
    {"blue_square", 1280, 720, 4, Kind::BlueSquare},
    {"green_stripes", 1280, 720, 4, Kind::GreenStripes},
    {"black_line", 1280, 720, 4, Kind::BlackLine},
}};

// Channel c, 0 red to 2 blue, of pixel (x, y) of frame t of `scene`.
constexpr std::uint8_t pixel(const Scene& scene, std::uint32_t t, std::uint32_t x, std::uint32_t y,
                             std::uint32_t c) {
    const auto sx = static_cast<std::int64_t>(x);
    const auto sy = static_cast<std::int64_t>(y);
    const auto st = static_cast<std::int64_t>(t);
    switch (scene.kind) {
        case Kind::RedBall: {
            // A disc of radius 90 crossing left to right, 290 px a second.
            const std::int64_t dx = sx - (200 + 290 * st);
            const std::int64_t dy = sy - 360;
            constexpr std::array<std::uint8_t, 3> red{220, 30, 30};
            return dx * dx + dy * dy <= 90 * 90 ? red[c] : std::uint8_t{255};
        }
        case Kind::BlueSquare: {
            // A square centred on the frame, its half side 60 + 50 t.
            const std::int64_t half = 60 + 50 * st;
            const bool in = sx >= 640 - half && sx < 640 + half && sy >= 360 - half && sy < 360 + half;
            constexpr std::array<std::uint8_t, 3> blue{30, 60, 220};
            constexpr std::array<std::uint8_t, 3> yellow{250, 220, 40};
            return in ? blue[c] : yellow[c];
        }
        case Kind::GreenStripes: {
            // Vertical stripes 50 px wide, moving right 40 px a second.
            constexpr std::array<std::uint8_t, 3> green{40, 170, 60};
            return ((sx + 1280 - 40 * st) / 50) % 2 == 0 ? green[c] : std::uint8_t{255};
        }
        case Kind::BlackLine: {
            // A vertical line 12 px wide sweeping right, 320 px a second.
            const std::int64_t d = sx - (160 + 320 * st);
            return d >= -6 && d < 6 ? std::uint8_t{0} : std::uint8_t{255};
        }
    }
    return 0;
}

}  // namespace bllm::fixtures::video
