#pragma once

// The images the vision reference tool gives llama.cpp
// (tools/reference_vision) and the tests give Charlotte
// (tests/media_image_test.cpp), each computed from its coordinates, so both
// see the same pixels with no image file, decoder or licence between them.
// Four are scenes plain enough to name in a query — a red circle, a blue
// square, green stripes, a black diagonal — for the embedding and ranking
// checks; the rest are noise at sizes that exercise the resampler: shrinking,
// enlarging, one axis only, a single pixel, a wide strip, a photo's size, and
// one past the budget's aspect.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace bllm::fixtures::vision {

enum class Kind {
    RedCircle,
    BlueSquare,
    GreenStripes,
    BlackDiagonal,
    Noise,
};

struct Scene {
    std::string_view name;
    std::uint32_t width;
    std::uint32_t height;
    Kind kind;
};

inline constexpr std::array<Scene, 11> kScenes{{
    {"red_circle", 320, 240, Kind::RedCircle},
    {"blue_square", 240, 320, Kind::BlueSquare},
    {"green_stripes", 400, 200, Kind::GreenStripes},
    {"black_diagonal", 300, 300, Kind::BlackDiagonal},
    {"noise_97x61", 97, 61, Kind::Noise},
    {"noise_1200x900", 1200, 900, Kind::Noise},
    {"noise_2000x37", 2000, 37, Kind::Noise},
    {"noise_1x1", 1, 1, Kind::Noise},
    {"noise_48x48", 48, 48, Kind::Noise},
    {"noise_4032x3024", 4032, 3024, Kind::Noise},
    {"noise_16384x1", 16384, 1, Kind::Noise},
}};

// Channel c, 0 red to 2 blue, of pixel (x, y) of `scene`.
constexpr std::uint8_t pixel(const Scene& scene, std::uint32_t x, std::uint32_t y, std::uint32_t c) {
    switch (scene.kind) {
        case Kind::RedCircle: {
            const std::int64_t dx = static_cast<std::int64_t>(x) - 160;
            const std::int64_t dy = static_cast<std::int64_t>(y) - 120;
            const bool in = dx * dx + dy * dy <= 80 * 80;
            constexpr std::array<std::uint8_t, 3> red{220, 30, 30};
            return in ? red[c] : std::uint8_t{255};
        }
        case Kind::BlueSquare: {
            const bool in = x >= 60 && x < 180 && y >= 100 && y < 220;
            constexpr std::array<std::uint8_t, 3> blue{30, 60, 220};
            constexpr std::array<std::uint8_t, 3> yellow{250, 220, 40};
            return in ? blue[c] : yellow[c];
        }
        case Kind::GreenStripes: {
            constexpr std::array<std::uint8_t, 3> green{40, 170, 60};
            return (x / 25) % 2 == 0 ? green[c] : std::uint8_t{255};
        }
        case Kind::BlackDiagonal: {
            const std::int64_t d = static_cast<std::int64_t>(x) - static_cast<std::int64_t>(y);
            return d >= -6 && d <= 6 ? std::uint8_t{0} : std::uint8_t{255};
        }
        case Kind::Noise: {
            std::uint32_t h = x * 73856093u ^ y * 19349663u ^ c * 83492791u ^ scene.width * 2654435761u;
            h ^= h >> 13;
            h *= 0x5bd1e995u;
            h ^= h >> 15;
            return static_cast<std::uint8_t>(h & 0xffu);
        }
    }
    return 0;
}

// FNV-1a, 64 bits, over bytes: the fixture's check of a prepared image,
// whose bytes are too many to keep.
constexpr std::uint64_t fnv1a(const std::uint8_t* data, std::size_t length) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < length; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

}  // namespace bllm::fixtures::vision
