#include <doctest/doctest.h>

#include <cstring>
#include <vector>

#include "bllm/shaders_generated.h"
#include "core/kernels/norm/norm.h"

using namespace bllm;

namespace {

residency::WeightView gain(std::uint32_t buffer, std::uint64_t offset, std::uint64_t width) {
    gguf::TensorShape shape{};
    shape.dimension_count = 1;
    shape.dimensions[0] = width;
    shape.element_count = width;
    return residency::WeightView(gguf::TensorType::F32, shape,
                                 {{residency::BufferIndex{buffer}, offset, width * 4, 0, 1}});
}

double override_of(const kernels::Launch& l, std::string_view name) {
    for (const auto& o : l.overrides) {
        if (o.name == name) return o.value;
    }
    FAIL("no override " << name);
    return -1;
}

}  // namespace

TEST_CASE("a norm launch binds both gains, the output, hidden and normed, and selects its variant") {
    const auto pre = gain(0, 256, 1152);
    const auto post = gain(0, 5120, 1152);
    const residency::BufferRange output{residency::BufferIndex{3}, 0, 100}, hidden{residency::BufferIndex{4}, 0, 200},
        normed{residency::BufferIndex{5}, 0, 300};

    const auto with_post = kernels::norm_launch({&pre, &post, true, output, hidden, normed, 1e-6f, kernels::Rows::EveryToken});
    CHECK(with_post.kernel == shaders::norm);
    CHECK(with_post.format == nullptr);
    REQUIRE(with_post.bindings.size() == 6);
    CHECK(with_post.bindings[0].offset == 256);
    CHECK(with_post.bindings[1].offset == 5120);
    CHECK(with_post.bindings[2].buffer == output.buffer);
    CHECK(with_post.bindings[3].buffer == hidden.buffer);
    CHECK(with_post.bindings[4].buffer == normed.buffer);
    CHECK(with_post.bindings[5].offset == 256);   // no scale: the gain, read-only, again
    CHECK(override_of(with_post, "add") == 1.0);
    CHECK(override_of(with_post, "post_norm") == 1.0);
    CHECK(override_of(with_post, "width") == 1152.0);
    REQUIRE(with_post.constants.size() == 4);
    float epsilon = 0;
    std::memcpy(&epsilon, with_post.constants.data(), 4);
    CHECK(epsilon == 1e-6f);
    CHECK(with_post.invocations_per_row == 256);   // one workgroup a row
    CHECK(with_post.workgroup_size == 256);

    // No post-norm: its binding is the norm's own gain, read-only, again.
    const auto plain = kernels::norm_launch({&pre, nullptr, false, output, hidden, normed, 1e-5f, kernels::Rows::LastToken});
    CHECK(plain.bindings[1].offset == plain.bindings[0].offset);
    CHECK(override_of(plain, "add") == 0.0);
    CHECK(override_of(plain, "post_norm") == 0.0);
    CHECK(plain.rows == kernels::Rows::LastToken);
    CHECK(override_of(plain, "scaled") == 0.0);
    CHECK(override_of(plain, "normalize") == 1.0);
    CHECK(override_of(plain, "rows_per_token") == 1.0);
}

TEST_CASE("an encoder's norm binds its scale, and covers its rows a token") {
    const auto pre = gain(0, 256, 512);
    const auto scale = gain(0, 4096, 1);
    const residency::BufferRange output{residency::BufferIndex{3}, 0, 100}, hidden{residency::BufferIndex{4}, 0, 200},
        normed{residency::BufferIndex{5}, 0, 300};
    kernels::NormLaunch n{&pre, nullptr, true, output, hidden, normed, 1e-6f, kernels::Rows::EveryToken};
    n.scale = &scale;
    const auto scaled = kernels::norm_launch(n);
    CHECK(scaled.bindings[5].offset == 4096);
    CHECK(override_of(scaled, "scaled") == 1.0);

    kernels::NormLaunch m{&pre, nullptr, false, output, hidden, normed, 5.12e-4f, kernels::Rows::EveryToken};
    m.rows_per_token = 24;
    m.normalize = true;
    const auto rows = kernels::norm_launch(m);
    CHECK(rows.invocations_per_row == 24 * 256);   // a workgroup a row, 24 rows a token
    CHECK(override_of(rows, "rows_per_token") == 24.0);
}
