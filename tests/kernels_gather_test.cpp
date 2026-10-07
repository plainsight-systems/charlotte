#include <doctest/doctest.h>

#include <cstring>
#include <vector>

#include "bllm/shaders_generated.h"
#include "core/capability/capability.h"
#include "core/kernels/gather/gather.h"

using namespace bllm;

namespace {

struct Constants {
    std::uint32_t first_row, row_count, groups_per_row, blocks_in_piece;
    float scale;
};

Constants constants_of(const kernels::Launch& launch) {
    Constants c{};
    REQUIRE(launch.constants.size() == sizeof c);
    std::memcpy(&c, launch.constants.data(), sizeof c);
    return c;
}

residency::WeightView table(gguf::TensorType type, std::uint64_t width, std::vector<residency::WeightPiece> pieces) {
    gguf::TensorShape shape{};
    shape.dimension_count = 2;
    shape.dimensions[0] = width;
    std::uint64_t rows = 0;
    for (const auto& p : pieces) rows += p.row_count;
    shape.dimensions[1] = rows;
    shape.element_count = width * rows;
    return residency::WeightView(type, shape, std::move(pieces));
}

}  // namespace

TEST_CASE("the gather launches once a piece, each binding its piece then the hidden buffer") {
    const residency::BufferRange hidden{residency::BufferIndex{9}, 512, 4096};
    const auto view = table(gguf::TensorType::Q6_K, 1024,
                            {{residency::BufferIndex{0}, 0, 3360, 0, 4},
                             {residency::BufferIndex{1}, 256, 2520, 4, 3},
                             {residency::BufferIndex{1}, 3072, 840, 7, 1}});
    const residency::BufferRange sampled{residency::BufferIndex{10}, 0, 16};
    const auto launches = kernels::gather_launches(view, hidden, sampled, 32.0f);
    REQUIRE(launches.size() == 3);
    const std::uint32_t first[] = {0, 4, 7};
    const std::uint32_t rows[] = {4, 3, 1};
    for (std::size_t i = 0; i < launches.size(); ++i) {
        CAPTURE(i);
        const kernels::Launch& l = launches[i];
        CHECK(l.kernel == shaders::gather);
        CHECK(l.format == capability::find_format(gguf::TensorType::Q6_K));
        const Constants c = constants_of(l);
        CHECK(c.first_row == first[i]);
        CHECK(c.row_count == rows[i]);
        CHECK(c.groups_per_row == 32);
        CHECK(c.blocks_in_piece == rows[i] * 4);   // four 256-weight super-blocks a row
        CHECK(c.scale == 32.0f);
        REQUIRE(l.bindings.size() == 3);
        CHECK(l.bindings[2].buffer == sampled.buffer);
        CHECK(l.bindings[0].buffer == view.pieces()[i].buffer);
        CHECK(l.bindings[0].offset == view.pieces()[i].offset);
        CHECK(l.bindings[0].size == view.pieces()[i].length);
        CHECK(l.bindings[1].buffer == hidden.buffer);
        CHECK(l.bindings[1].offset == hidden.offset);
        CHECK(l.bindings[1].size == hidden.length);
        CHECK(l.invocations_per_row == 32);
        CHECK(l.workgroup_size == 64);
    }
}

TEST_CASE("a format of one weight a block counts its blocks as weights") {
    const auto view = table(gguf::TensorType::F32, 64, {{residency::BufferIndex{0}, 0, 1536, 0, 6}});
    const auto launches =
        kernels::gather_launches(view, {residency::BufferIndex{1}, 0, 256}, {residency::BufferIndex{2}, 0, 16}, 1.0f);
    REQUIRE(launches.size() == 1);
    CHECK(constants_of(launches[0]).blocks_in_piece == 6 * 64);
    CHECK(constants_of(launches[0]).groups_per_row == 2);
    CHECK(launches[0].format == capability::find_format(gguf::TensorType::F32));
}
