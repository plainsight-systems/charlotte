#include <doctest/doctest.h>

#include <cstring>

#include "bllm/shaders_generated.h"
#include "core/kernels/subsample/subsample.h"

using namespace bllm;

namespace {

residency::WeightView f32(std::uint32_t buffer, std::uint64_t offset, std::uint64_t count) {
    gguf::TensorShape shape{};
    shape.dimension_count = 1;
    shape.dimensions[0] = count;
    shape.element_count = count;
    return residency::WeightView(gguf::TensorType::F32, shape,
                                 {{residency::BufferIndex{buffer}, offset, count * 4, 0, 1}});
}

}  // namespace

TEST_CASE("the subsampling launches: a workgroup a row, the weights, gain, input and output, and epsilon") {
    const auto first_weights = f32(0, 256, 128 * 9);
    const auto first_gain = f32(0, 8192, 128);
    const auto second_weights = f32(0, 9216, 32 * 128 * 9);
    const auto second_gain = f32(0, 157184, 32);
    const residency::BufferRange mel{residency::BufferIndex{2}, 0, 100}, subsampled{residency::BufferIndex{3}, 0, 200},
        rows{residency::BufferIndex{4}, 0, 300};

    const kernels::Launch first = kernels::subsample_first_launch(first_weights, first_gain, mel, subsampled, 1e-6f);
    const kernels::Launch second =
        kernels::subsample_second_launch(second_weights, second_gain, subsampled, rows, 1e-6f);
    CHECK(first.entry_point == "first");
    CHECK(second.entry_point == "second");
    CHECK(first.invocations_per_row == 64);
    CHECK(first.workgroup_size == 64);
    CHECK(second.invocations_per_row == 32);
    CHECK(second.workgroup_size == 32);
    for (const kernels::Launch* l : {&first, &second}) {
        CHECK(l->kernel == shaders::subsample);
        CHECK(l->format == nullptr);
        CHECK(l->rows == kernels::Rows::EveryToken);
        CHECK(l->overrides.empty());
        REQUIRE(l->constants.size() == 4);
        float epsilon = 0;
        std::memcpy(&epsilon, l->constants.data(), 4);
        CHECK(epsilon == 1e-6f);
        REQUIRE(l->bindings.size() == 4);
    }
    CHECK(first.bindings[0].offset == 256);
    CHECK(first.bindings[0].size == 128 * 9 * 4);
    CHECK(first.bindings[1].offset == 8192);
    CHECK(first.bindings[2].buffer == mel.buffer);
    CHECK(first.bindings[3].buffer == subsampled.buffer);
    CHECK(second.bindings[0].offset == 9216);
    CHECK(second.bindings[0].size == 32 * 128 * 9 * 4);
    CHECK(second.bindings[1].offset == 157184);
    CHECK(second.bindings[2].buffer == subsampled.buffer);
    CHECK(second.bindings[3].buffer == rows.buffer);
}
