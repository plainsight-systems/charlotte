#include "core/media/image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace bllm::media {

namespace {

// Pillow's fixed point: 32 bits less 8 for a pixel and 2 of headroom
// (llama.cpp's resize_pillow, PRECISION_BITS).
constexpr int kPrecisionBits = 22;
constexpr std::int32_t kHalf = std::int32_t{1} << (kPrecisionBits - 1);

// Bicubic, a = −0.5, Pillow's kernel; support 2.
double bicubic(double x) noexcept {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

std::uint8_t clip8(std::int32_t v) noexcept {
    return static_cast<std::uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
}

// One axis's taps from `in` pixels to `out`, as llama.cpp's
// precompute_weights computes them: f64 weights normalized to sum to 1, then
// fixed point, rounded half away from zero by Pillow's +/- 0.5 and a
// truncation.
template <typename Taps>
void precompute(std::uint32_t in, std::uint32_t out, Taps& taps) {
    const double scale = static_cast<double>(in) / out;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * filterscale;
    const auto size = static_cast<std::uint32_t>(std::ceil(support)) * 2 + 1;
    taps.size = size;
    taps.first.resize(out);
    taps.count.resize(out);
    taps.weights.assign(static_cast<std::size_t>(out) * size, 0);
    std::vector<double> pre(size);
    const double fixed = std::ldexp(1.0, kPrecisionBits);
    for (std::uint32_t o = 0; o < out; ++o) {
        const double center = (o + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int first = static_cast<int>(center - support + 0.5);
        if (first < 0) first = 0;
        int last = static_cast<int>(center + support + 0.5);
        if (last > static_cast<int>(in)) last = static_cast<int>(in);
        const int count = last - first;
        double sum = 0.0;
        for (int x = 0; x < count; ++x) {
            const double w = bicubic((x + first - center + 0.5) * ss);
            pre[static_cast<std::size_t>(x)] = w;
            sum += w;
        }
        for (int x = 0; x < count; ++x) {
            if (sum != 0.0) pre[static_cast<std::size_t>(x)] /= sum;
        }
        std::int32_t* weights = &taps.weights[static_cast<std::size_t>(o) * size];
        for (int x = 0; x < count; ++x) {
            const double w = pre[static_cast<std::size_t>(x)];
            weights[x] = static_cast<std::int32_t>(w * fixed + (w < 0 ? -0.5 : 0.5));
        }
        taps.first[o] = static_cast<std::uint32_t>(first);
        taps.count[o] = static_cast<std::uint32_t>(count);
    }
}

// (u / 255) × 2 − 1 for every byte: llama.cpp's x / 255, then its graph's
// scale by 2 and bias by −1. Both rounded in f32; × 2 is exact, so a fused
// multiply-add gives the same bits.
const std::array<float, 256>& patch_values() {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> t{};
        for (int u = 0; u < 256; ++u) {
            const float x = static_cast<float>(u) / 255.0f;
            t[static_cast<std::size_t>(u)] = x * 2.0f - 1.0f;
        }
        return t;
    }();
    return table;
}

}  // namespace

ImageGrid image_grid(std::uint32_t width, std::uint32_t height, std::uint32_t patch, std::uint32_t pooling,
                     ImageBudget budget) noexcept {
    // llama.cpp's calc_size_preserved_ratio, its arithmetic as it is: ints,
    // and f32 where it divides, roots and rounds.
    const int align = static_cast<int>(patch * pooling);
    const int min_pixels = static_cast<int>(budget.least) * align * align;
    const int max_pixels = static_cast<int>(budget.most) * align * align;
    const auto round_by = [align](float x) {
        return static_cast<int>(std::round(x / static_cast<float>(align))) * align;
    };
    const auto ceil_by = [align](float x) {
        return static_cast<int>(std::ceil(x / static_cast<float>(align))) * align;
    };
    const auto floor_by = [align](float x) {
        return static_cast<int>(std::floor(x / static_cast<float>(align))) * align;
    };
    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);
    int w_bar = std::max(align, round_by(static_cast<float>(w)));
    int h_bar = std::max(align, round_by(static_cast<float>(h)));
    if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
        const float beta = std::sqrt(static_cast<float>(h) * static_cast<float>(w) / static_cast<float>(max_pixels));
        h_bar = std::max(align, floor_by(static_cast<float>(h) / beta));
        w_bar = std::max(align, floor_by(static_cast<float>(w) / beta));
    } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
        const float beta =
            std::sqrt(static_cast<float>(min_pixels) / (static_cast<float>(h) * static_cast<float>(w)));
        h_bar = ceil_by(static_cast<float>(h) * beta);
        w_bar = ceil_by(static_cast<float>(w) * beta);
    }
    const auto columns = static_cast<std::uint32_t>(w_bar) / patch;
    const auto rows = static_cast<std::uint32_t>(h_bar) / patch;
    return {static_cast<std::uint32_t>(w_bar), static_cast<std::uint32_t>(h_bar), columns, rows,
            (columns / pooling) * (rows / pooling)};
}

ImageFit image_fit(std::uint32_t width, std::uint32_t height, const ImageGrid& grid) noexcept {
    // llama.cpp's resize under PAD_CEIL: fitted inside, aspect kept, centred.
    const float scale_w = static_cast<float>(grid.width) / static_cast<float>(width);
    const float scale_h = static_cast<float>(grid.height) / static_cast<float>(height);
    const float scale = std::min(scale_w, scale_h);
    const auto fitted_w = std::min(static_cast<std::uint32_t>(std::ceil(static_cast<float>(width) * scale)), grid.width);
    const auto fitted_h =
        std::min(static_cast<std::uint32_t>(std::ceil(static_cast<float>(height) * scale)), grid.height);
    return {fitted_w, fitted_h, (grid.width - fitted_w) / 2, (grid.height - fitted_h) / 2};
}

ImageError ImageResampler::begin(std::uint32_t source_width, std::uint32_t source_height, const ImageGrid& grid,
                                 std::uint32_t patch, std::uint32_t most) {
    source_width_ = 0;
    if (source_width == 0 || source_height == 0) return ImageError::Empty;
    if (source_width > kMaxSide || source_height > kMaxSide) return ImageError::TooLarge;
    if (grid.tokens > most) return ImageError::OverBudget;
    grid_ = grid;
    fit_ = image_fit(source_width, source_height, grid);
    patch_ = patch;
    source_width_ = source_width;
    source_height_ = source_height;
    next_row_ = 0;
    finished_ = false;
    horizontal_pass_ = fit_.width != source_width;
    vertical_pass_ = fit_.height != source_height;
    if (horizontal_pass_) precompute(source_width, fit_.width, horizontal_);
    if (vertical_pass_) precompute(source_height, fit_.height, vertical_);
    intermediate_.assign(static_cast<std::size_t>(fit_.width) * source_height * 3, 0);
    fitted_.clear();
    return ImageError::Ok;
}

ImageError ImageResampler::accept(std::span<const std::uint8_t> rgba, std::uint32_t first_row) {
    const std::size_t stride = static_cast<std::size_t>(source_width_) * 4;
    if (source_width_ == 0 || finished_ || first_row != next_row_ || rgba.empty() || rgba.size() % stride != 0) {
        return ImageError::BadBand;
    }
    const std::size_t rows = rgba.size() / stride;
    if (rows > source_height_ - next_row_) return ImageError::BadBand;
    const std::size_t out_stride = static_cast<std::size_t>(fit_.width) * 3;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* src = rgba.data() + r * stride;
        std::uint8_t* dst = intermediate_.data() + (next_row_ + r) * out_stride;
        if (!horizontal_pass_) {
            for (std::uint32_t x = 0; x < source_width_; ++x) {
                dst[x * 3 + 0] = src[x * 4 + 0];
                dst[x * 3 + 1] = src[x * 4 + 1];
                dst[x * 3 + 2] = src[x * 4 + 2];
            }
            continue;
        }
        for (std::uint32_t o = 0; o < fit_.width; ++o) {
            const std::int32_t* k = &horizontal_.weights[static_cast<std::size_t>(o) * horizontal_.size];
            const std::uint8_t* p = src + static_cast<std::size_t>(horizontal_.first[o]) * 4;
            std::int32_t s0 = kHalf;
            std::int32_t s1 = kHalf;
            std::int32_t s2 = kHalf;
            for (std::uint32_t x = 0; x < horizontal_.count[o]; ++x) {
                s0 += p[0] * k[x];
                s1 += p[1] * k[x];
                s2 += p[2] * k[x];
                p += 4;
            }
            dst[o * 3 + 0] = clip8(s0 >> kPrecisionBits);
            dst[o * 3 + 1] = clip8(s1 >> kPrecisionBits);
            dst[o * 3 + 2] = clip8(s2 >> kPrecisionBits);
        }
    }
    next_row_ += static_cast<std::uint32_t>(rows);
    return ImageError::Ok;
}

void ImageResampler::finish() {
    if (finished_) return;
    finished_ = true;
    if (!vertical_pass_) {
        fitted_ = std::move(intermediate_);
        intermediate_ = {};
        return;
    }
    const std::size_t row = static_cast<std::size_t>(fit_.width) * 3;
    fitted_.assign(row * fit_.height, 0);
    std::vector<std::int32_t> acc(row);
    for (std::uint32_t o = 0; o < fit_.height; ++o) {
        const std::int32_t* k = &vertical_.weights[static_cast<std::size_t>(o) * vertical_.size];
        std::fill(acc.begin(), acc.end(), kHalf);
        for (std::uint32_t y = 0; y < vertical_.count[o]; ++y) {
            const std::uint8_t* src = intermediate_.data() + (vertical_.first[o] + y) * row;
            const std::int32_t w = k[y];
            for (std::size_t i = 0; i < row; ++i) acc[i] += src[i] * w;
        }
        std::uint8_t* dst = fitted_.data() + o * row;
        for (std::size_t i = 0; i < row; ++i) dst[i] = clip8(acc[i] >> kPrecisionBits);
    }
    intermediate_ = {};
}

ImageError ImageResampler::canvas(std::span<std::uint8_t> out) {
    if (source_width_ == 0 || next_row_ != source_height_) return ImageError::Incomplete;
    finish();
    std::fill(out.begin(), out.end(), std::uint8_t{0});
    const std::size_t row = static_cast<std::size_t>(fit_.width) * 3;
    for (std::uint32_t y = 0; y < fit_.height; ++y) {
        const std::uint8_t* src = fitted_.data() + y * row;
        std::copy(src, src + row,
                  out.begin() + static_cast<std::ptrdiff_t>(((fit_.y + y) * static_cast<std::size_t>(grid_.width) +
                                                             fit_.x) * 3));
    }
    return ImageError::Ok;
}

ImageError ImageResampler::patches(std::span<float> out) {
    if (source_width_ == 0 || next_row_ != source_height_) return ImageError::Incomplete;
    finish();
    const auto& value = patch_values();
    const float black = value[0];
    const std::uint32_t p = patch_;
    const std::size_t per_patch = static_cast<std::size_t>(3) * p * p;
    for (std::uint32_t py = 0; py < grid_.rows; ++py) {
        for (std::uint32_t px = 0; px < grid_.columns; ++px) {
            float* dst = out.data() + (static_cast<std::size_t>(py) * grid_.columns + px) * per_patch;
            for (std::uint32_t c = 0; c < 3; ++c) {
                for (std::uint32_t ky = 0; ky < p; ++ky) {
                    const std::uint32_t y = py * p + ky;
                    const bool row_in = y >= fit_.y && y < fit_.y + fit_.height;
                    for (std::uint32_t kx = 0; kx < p; ++kx) {
                        const std::uint32_t x = px * p + kx;
                        float v = black;
                        if (row_in && x >= fit_.x && x < fit_.x + fit_.width) {
                            v = value[fitted_[((static_cast<std::size_t>(y - fit_.y) * fit_.width) + (x - fit_.x)) *
                                                  3 +
                                              c]];
                        }
                        dst[(c * p + ky) * p + kx] = v;
                    }
                }
            }
        }
    }
    return ImageError::Ok;
}

}  // namespace bllm::media
