#include <doctest/doctest.h>

#include <cstring>

#include "bllm/shaders_generated.h"
#include "core/kernels/local_attention/local_attention.h"

using namespace bllm;

TEST_CASE("a local attention launch: a workgroup of 128 a row, its six bindings, and its four constants") {
    gguf::TensorShape shape{};
    shape.dimension_count = 1;
    shape.dimensions[0] = 128;
    shape.element_count = 128;
    const residency::WeightView scale(gguf::TensorType::F32, shape, {{residency::BufferIndex{0}, 512, 512, 0, 1}});
    const residency::BufferRange relative{residency::BufferIndex{2}, 53248, 53248},
        query{residency::BufferIndex{3}, 0, 100}, key{residency::BufferIndex{4}, 0, 200},
        value{residency::BufferIndex{5}, 0, 300}, attention{residency::BufferIndex{6}, 0, 400};
    const kernels::Launch l =
        kernels::local_attention_launch({&scale, relative, query, key, value, attention, 0.5f, 1.5f, 50.0f, 12});
    CHECK(l.kernel == shaders::local_attention);
    CHECK(l.format == nullptr);
    CHECK(l.rows == kernels::Rows::EveryToken);
    CHECK(l.invocations_per_row == 128);
    CHECK(l.workgroup_size == 128);
    CHECK(l.overrides.empty());
    REQUIRE(l.bindings.size() == 6);
    CHECK(l.bindings[0].offset == 512);
    CHECK(l.bindings[1].buffer == relative.buffer);
    CHECK(l.bindings[1].offset == 53248);
    CHECK(l.bindings[2].buffer == query.buffer);
    CHECK(l.bindings[3].buffer == key.buffer);
    CHECK(l.bindings[4].buffer == value.buffer);
    CHECK(l.bindings[5].buffer == attention.buffer);
    REQUIRE(l.constants.size() == 16);
    float f[3];
    std::uint32_t window = 0;
    std::memcpy(f, l.constants.data(), 12);
    std::memcpy(&window, l.constants.data() + 12, 4);
    CHECK(f[0] == 0.5f);
    CHECK(f[1] == 1.5f);
    CHECK(f[2] == 50.0f);
    CHECK(window == 12);
}
