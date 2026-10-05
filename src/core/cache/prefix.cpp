#include "core/cache/prefix.h"

#include <algorithm>

namespace bllm::cache {

std::size_t longest_common_prefix(std::span<const tokenizer::TokenId> cached,
                                  std::span<const tokenizer::TokenId> incoming) noexcept {
    const auto [end, ignored] = std::mismatch(cached.begin(), cached.end(), incoming.begin(), incoming.end());
    return static_cast<std::size_t>(end - cached.begin());
}

}  // namespace bllm::cache
