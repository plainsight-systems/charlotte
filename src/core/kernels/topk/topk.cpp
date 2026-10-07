#include "core/kernels/topk/topk.h"

#include <cstring>

#include "bllm/shaders_generated.h"

namespace bllm::kernels {
namespace {

// Binding 1, as topk.wgsl's TopK lays it out.
struct Constants {
    std::uint32_t count;
};

constexpr std::uint32_t kWorkgroupSize = 256;   // 4 entries an invocation, a tile of 1,024

Binding whole(const residency::BufferRange& range) { return {range.buffer, range.offset, range.length}; }

}  // namespace

std::vector<Launch> topk_launches(const residency::BufferRange& logits, std::uint32_t vocabulary,
                                  const residency::BufferRange& partials_a, const residency::BufferRange& partials_b,
                                  const residency::BufferRange& candidates) {
    std::vector<Launch> out;
    residency::BufferRange input = logits;
    std::uint32_t count = vocabulary;
    for (bool first = true;; first = false) {
        // A ceiling without the addition that would wrap near UINT32_MAX.
        const std::uint32_t tiles =
            count / residency::kSelectionTile + (count % residency::kSelectionTile != 0 ? 1 : 0);
        // The last pass, one workgroup, writes the candidates; the others
        // alternate, never reading the buffer they write.
        const residency::BufferRange& output =
            tiles == 1 ? candidates : (out.size() % 2 == 0 ? partials_a : partials_b);
        const Constants constants{count};
        std::vector<std::byte> bytes(sizeof constants);
        std::memcpy(bytes.data(), &constants, sizeof constants);
        Launch pass{shaders::topk,
                    nullptr,
                    std::move(bytes),
                    {whole(input), whole(output)},
                    tiles * kWorkgroupSize,
                    kWorkgroupSize,
                    Rows::LastToken,
                    {}};
        pass.entry_point = first ? "first" : "merge";
        out.push_back(std::move(pass));
        if (tiles == 1) break;
        input = output;
        count = tiles * kCandidates;
    }
    return out;
}

}  // namespace bllm::kernels
