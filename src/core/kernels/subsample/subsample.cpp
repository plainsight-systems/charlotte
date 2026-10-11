#include "core/kernels/subsample/subsample.h"

#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

// Binding 1, as subsample.wgsl's Subsample lays it out.
struct Constants {
    float epsilon;
};

// One workgroup a step row: the first's 64 frequencies, the second's 32.
constexpr std::uint32_t kFirstInvocations = 64;
constexpr std::uint32_t kSecondInvocations = 32;

Binding whole(const residency::WeightView& view) {
    const residency::WeightPiece& piece = view.pieces().front();
    return {piece.buffer, piece.offset, piece.length};
}

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

Launch one_row(std::string_view entry_point, std::uint32_t invocations, const residency::WeightView& weights,
               const residency::WeightView& gain, const residency::BufferRange& in,
               const residency::BufferRange& out, float epsilon) {
    const Constants constants{epsilon};
    std::vector<std::byte> bytes(sizeof constants);
    std::memcpy(bytes.data(), &constants, sizeof constants);
    Launch launch{shaders::subsample,
                  nullptr,
                  std::move(bytes),
                  {whole(weights), whole(gain), whole(in), whole(out)},
                  invocations,
                  invocations,
                  Rows::EveryToken,
                  {}};
    launch.entry_point = entry_point;
    return launch;
}

}  // namespace

Launch subsample_first_launch(const residency::WeightView& weights, const residency::WeightView& gain,
                              const residency::BufferRange& mel, const residency::BufferRange& subsampled,
                              float epsilon) {
    return one_row("first", kFirstInvocations, weights, gain, mel, subsampled, epsilon);
}

Launch subsample_second_launch(const residency::WeightView& weights, const residency::WeightView& gain,
                               const residency::BufferRange& subsampled, const residency::BufferRange& rows,
                               float epsilon) {
    return one_row("second", kSecondInvocations, weights, gain, subsampled, rows, epsilon);
}

}  // namespace bllm::kernels
