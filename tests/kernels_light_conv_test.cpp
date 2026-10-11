#include <doctest/doctest.h>

#include <cstring>

#include "bllm/shaders_generated.h"
#include "core/kernels/light_conv/light_conv.h"

using namespace bllm;

namespace {

residency::WeightView f32(std::uint64_t offset, std::uint64_t count) {
    gguf::TensorShape shape{};
    shape.dimension_count = 1;
    shape.dimensions[0] = count;
    shape.element_count = count;
    return residency::WeightView(gguf::TensorType::F32, shape, {{residency::BufferIndex{0}, offset, count * 4, 0, 1}});
}

}  // namespace

TEST_CASE("a light convolution launch: a workgroup of 256 a row, its width, taps and epsilon") {
    const auto depthwise = f32(256, 5 * 1024);
    const auto gain = f32(20736, 1024);
    const residency::BufferRange input{residency::BufferIndex{2}, 0, 100}, output{residency::BufferIndex{3}, 0, 200};
    const kernels::Launch l = kernels::light_conv_launch(depthwise, gain, input, output, 1024, 5, 1e-6f);
    CHECK(l.kernel == shaders::light_conv);
    CHECK(l.format == nullptr);
    CHECK(l.rows == kernels::Rows::EveryToken);
    CHECK(l.invocations_per_row == 256);
    CHECK(l.workgroup_size == 256);
    REQUIRE(l.overrides.size() == 1);
    CHECK(l.overrides[0] == kernels::Override{"width", 1024.0});
    REQUIRE(l.bindings.size() == 4);
    CHECK(l.bindings[0].offset == 256);
    CHECK(l.bindings[0].size == 5 * 1024 * 4);
    CHECK(l.bindings[1].offset == 20736);
    CHECK(l.bindings[2].buffer == input.buffer);
    CHECK(l.bindings[3].buffer == output.buffer);
    REQUIRE(l.constants.size() == 8);
    float epsilon = 0;
    std::uint32_t taps = 0;
    std::memcpy(&epsilon, l.constants.data(), 4);
    std::memcpy(&taps, l.constants.data() + 4, 4);
    CHECK(epsilon == 1e-6f);
    CHECK(taps == 5);
}
