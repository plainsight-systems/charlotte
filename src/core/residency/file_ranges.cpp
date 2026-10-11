#include "core/residency/file_ranges.h"

namespace bllm::residency {

RangesResult check_ranges(std::span<const FileRange> ranges, std::uint64_t file_size,
                          std::uint64_t index_end) noexcept {
    if (ranges.empty()) return {RangesError::NoRanges, 0};
    if (ranges[0].begin != 0 || ranges[0].end < index_end) return {RangesError::MissesHeader, 0};
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].end <= ranges[i].begin) return {RangesError::Empty, i};
        if (i > 0 && ranges[i].begin <= ranges[i - 1].end) return {RangesError::OutOfOrder, i};
        if (ranges[i].end > file_size) return {RangesError::PastEnd, i};
    }
    return {};
}

bool within(std::span<const FileRange> ranges, std::uint64_t begin, std::uint64_t end) noexcept {
    for (const FileRange& r : ranges) {
        if (begin >= r.begin && end <= r.end) return true;
    }
    return false;
}

}  // namespace bllm::residency
