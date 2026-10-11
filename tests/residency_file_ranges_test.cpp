// The ranges a load of part of a file reads (residency/file_ranges.h): each
// rule's refusal, and which bytes lie within them.

#include <doctest/doctest.h>

#include <vector>

#include "core/residency/file_ranges.h"

using namespace bllm::residency;

TEST_CASE("ranges begin at 0 with the header, in order, apart, and within the file") {
    const std::vector<FileRange> good{{0, 100}, {200, 300}, {301, 400}};
    CHECK(check_ranges(good, 400, 64).ok());
    CHECK(check_ranges({}, 400, 64).error == RangesError::NoRanges);
    const std::vector<FileRange> late{{1, 100}};
    CHECK(check_ranges(late, 400, 64).error == RangesError::MissesHeader);
    const std::vector<FileRange> short_header{{0, 63}};
    CHECK(check_ranges(short_header, 400, 64).error == RangesError::MissesHeader);
    const std::vector<FileRange> empty{{0, 100}, {200, 200}};
    const auto e = check_ranges(empty, 400, 64);
    CHECK(e.error == RangesError::Empty);
    CHECK(e.range == 1);
    const std::vector<FileRange> touching{{0, 100}, {100, 200}};
    CHECK(check_ranges(touching, 400, 64).error == RangesError::OutOfOrder);
    const std::vector<FileRange> backwards{{0, 100}, {200, 300}, {150, 160}};
    CHECK(check_ranges(backwards, 400, 64).error == RangesError::OutOfOrder);
    const std::vector<FileRange> past{{0, 100}, {200, 401}};
    CHECK(check_ranges(past, 400, 64).error == RangesError::PastEnd);
}

TEST_CASE("bytes lie within the ranges only wholly inside one") {
    const std::vector<FileRange> r{{0, 100}, {200, 300}};
    CHECK(within(r, 0, 100));
    CHECK(within(r, 250, 300));
    CHECK_FALSE(within(r, 90, 210));
    CHECK_FALSE(within(r, 100, 101));
    CHECK_FALSE(within(r, 299, 301));
}
