#include "core/graph/vision.h"

#include <algorithm>
#include <cstdint>

namespace bllm::graph {

residency::WorkingSet vision_working_set(const model::ModelDescription& model, std::uint32_t rows) {
    std::uint64_t query = 0, key_value = 0, feed_forward = 0;
    for (const model::LayerDescription& l : model.layers) {
        query = std::max<std::uint64_t>(query, std::uint64_t{l.query_heads} * l.head_dimension);
        key_value = std::max<std::uint64_t>(key_value, std::uint64_t{l.key_value_heads} * l.head_dimension);
        feed_forward = std::max<std::uint64_t>(feed_forward, l.feed_forward_width);
    }
    const model::VisionDescription& v = *model.vision;
    const std::uint64_t patch = std::uint64_t{3} * v.patch_size * v.patch_size;
    const std::uint64_t pooled_rows = rows / (std::uint64_t{v.pooling} * v.pooling);
    const std::uint64_t w = model.embedding_width;
    residency::WorkingSet set{{
                                  {"pixels", rows, patch},
                                  {"hidden", rows, w},
                                  // Also the pooled rows, after the last layer's
                                  // feed-forward has read it (graph/vision.h).
                                  {"normed", rows, w},
                                  {"query", rows, query},
                                  {"key", rows, key_value},
                                  {"value", rows, key_value},
                                  {"attention", rows, query},
                                  {"output", rows, w},
                                  {"activation", rows, feed_forward},
                                  {"soft_tokens", pooled_rows, v.projection_width},
                                  // One float: no product of a vision step splits
                                  // along K (graph/vision.h).
                                  {"range_sums", 1, 1},
                              },
                              false,
                              0};
    set.step_rows = rows;
    return set;
}

}  // namespace bllm::graph
