#include "core/graph/audio.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>

#include "core/kernels/matmul/matmul.h"

namespace bllm::graph {
namespace {

// The 13 relative positions a layer's table holds: distances 0 to the
// window (graph/audio.h).
std::uint64_t relative_rows(const model::AudioDescription& a) { return std::uint64_t{a.window} + 1; }

}  // namespace

residency::WorkingSet audio_working_set(const model::ModelDescription& model, std::uint32_t rows) {
    const model::AudioDescription& a = *model.audio;
    const std::uint64_t r = rows;
    const std::uint64_t frames = 4 * r;                  // T: 4 R at most
    const std::uint64_t w = a.width;
    // The second convolution's rows, flattened: 32 positions × 32 channels.
    const std::uint64_t flat = std::uint64_t{a.channels[1]} * (a.mel_bins / 4);
    // The first's: positions in time 2 R, × 64 in frequency × 128 channels.
    const std::uint64_t first = std::uint64_t{a.channels[0]} * (a.mel_bins / 2);
    // Range sums for the products the gate splits, the most of theirs, over a
    // step of R rows: each layer's — up and down, Q, K, V and the output,
    // the second pointwise projection — the input projection, the output's
    // two. The first pointwise projection is gated, which never splits.
    using kernels::Epilogue;
    const auto need = [](std::uint64_t outputs, std::uint64_t columns) {
        return kernels::range_sum_floats(Epilogue::Write, outputs, columns, kernels::kSplitGate);
    };
    const std::uint64_t range_sums =
        std::max({std::uint64_t{1}, need(a.feed_forward, w), need(w, a.feed_forward), need(w, w), need(w, flat),
                  need(a.output_width, w), need(a.projection_width, a.output_width)});
    residency::WorkingSet set{{
                                  {"mel", frames, a.mel_bins},
                                  {"subsampled", 2 * r, first},
                                  {"hidden", r, w},
                                  // Also the second convolution's rows, before the
                                  // input projection reads them (graph/audio.h).
                                  {"normed", r, std::max(w, flat)},
                                  // Also the light convolution's output.
                                  {"query", r, w},
                                  {"key", r, w},
                                  {"value", r, w},
                                  // Also the GLU's rows.
                                  {"attention", r, w},
                                  {"output", r, w},
                                  // Also the output's norm, 1,536 wide.
                                  {"activation", r, std::max<std::uint64_t>(a.feed_forward, a.output_width)},
                                  {"projected", r, a.output_width},
                                  {"soft_tokens", r, a.projection_width},
                                  {"positions", relative_rows(a), w},
                                  {"relative", a.layers.size() * relative_rows(a), w},
                                  {"range_sums", 1, range_sums},
                              },
                              false,
                              0};
    set.step_rows = rows;
    return set;
}

}  // namespace bllm::graph
