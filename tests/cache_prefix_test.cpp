#include <doctest/doctest.h>

#include <vector>

#include "core/cache/prefix.h"

using bllm::cache::longest_common_prefix;
using bllm::tokenizer::TokenId;

namespace {

std::vector<TokenId> ids(std::initializer_list<std::uint32_t> values) {
    std::vector<TokenId> out;
    for (const std::uint32_t v : values) out.push_back(TokenId{v});
    return out;
}

}  // namespace

TEST_CASE("the common prefix runs to the first difference, or the shorter sequence's end") {
    CHECK(longest_common_prefix(ids({1, 2, 3}), ids({1, 2, 3})) == 3);
    CHECK(longest_common_prefix(ids({1, 2, 3}), ids({1, 2, 3, 4, 5})) == 3);   // the new turn appends
    CHECK(longest_common_prefix(ids({1, 2, 3, 4, 5}), ids({1, 2, 3})) == 3);   // the new turn is shorter
    CHECK(longest_common_prefix(ids({1, 2, 3}), ids({9, 2, 3})) == 0);
    CHECK(longest_common_prefix({}, ids({1, 2})) == 0);
    CHECK(longest_common_prefix(ids({1, 2}), {}) == 0);
}

TEST_CASE("tokens that match again after a difference are not counted: their keys saw the old history") {
    // A template that rewrites an earlier turn: the tail is the same tokens,
    // but every key after the change was computed from different text.
    CHECK(longest_common_prefix(ids({1, 2, 3, 4, 5, 6}), ids({1, 2, 7, 4, 5, 6})) == 2);
}
