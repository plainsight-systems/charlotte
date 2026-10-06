#include "core/kernels/gather/gather.h"

#include <cstring>

#include "bllm/shaders_generated.h"
#include "core/capability/capability.h"
#include "core/gguf/types.h"

namespace bllm::kernels {
namespace {

// Binding 1, as gather.wgsl's Gather lays it out.
struct Constants {
    std::uint32_t first_row;
    std::uint32_t row_count;
    std::uint32_t groups_per_row;
    std::uint32_t blocks_in_piece;
    float scale;
};
static_assert(sizeof(Constants) == 20);

constexpr std::uint32_t kWorkgroupSize = 64;

std::vector<std::byte> bytes_of(const Constants& c) {
    std::vector<std::byte> out(sizeof c);
    std::memcpy(out.data(), &c, sizeof c);
    return out;
}

}  // namespace

std::vector<Launch> gather_launches(const residency::WeightView& table, const residency::BufferRange& hidden,
                                    float scale) {
    const formats::Format* format = capability::find_format(table.format());
    const std::uint64_t width = table.shape().dimensions[0];
    const auto groups_per_row = static_cast<std::uint32_t>(width / formats::kUnpackGroup);
    const std::uint64_t block_elements = gguf::format_layout(table.format())->block_elements;

    std::vector<Launch> launches;
    launches.reserve(table.pieces().size());
    for (const residency::WeightPiece& piece : table.pieces()) {
        const Constants constants{
            static_cast<std::uint32_t>(piece.first_row),
            static_cast<std::uint32_t>(piece.row_count),
            groups_per_row,
            static_cast<std::uint32_t>(piece.row_count * width / block_elements),
            scale,
        };
        launches.push_back(Launch{
            shaders::gather,
            format,
            bytes_of(constants),
            {Binding{piece.buffer, piece.offset, piece.length}, Binding{hidden.buffer, hidden.offset, hidden.length}},
            groups_per_row,
            kWorkgroupSize,
        });
    }
    return launches;
}

}  // namespace bllm::kernels
