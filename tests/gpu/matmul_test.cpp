// The matrix products on Dawn, against f64 over the format's CPU-decoded
// weights (kernels/matmul/matmul.h): each listed format, decode and
// prefill, at widths whose ranges are one group, unequal and three groups;
// a weight in pieces; QKV into its three buffers; the gated activation for
// SiLU and GELU; the head's last token in a prefill step; and a token's
// outputs the same bits decoded alone and prefilled.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/matmul/matmul.h"
#include "support/acquire.h"
#include "support/fill.h"
#include "support/program.h"
#include "support/q4_0_reference.h"
#include "support/q4_1_reference.h"
#include "support/q6_k_reference.h"
#include "support/q8_0_reference.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kTokens = 64;
constexpr std::uint64_t kWidest = 3072;
constexpr std::uint64_t kMostRows = 4096;   // w_set4's

// Test-only: copies rows first .. first + tokens of the source rows into
// the input buffer, so a step's rows are any of the source's.
constexpr std::string_view kCopy = R"(
struct Copy { vec4s: u32, first: u32 }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> copy: Copy;
@group(0) @binding(2) var<storage, read> copy_from: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> copy_to: array<vec4<f32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < step.tokens * copy.vec4s) {
        copy_to[id.x] = copy_from[copy.first * copy.vec4s + id.x];
    }
}
)";

// The fixture's weights in one buffer in file order, as the plan places
// them — or, with `split`, w_q4_0 in two pieces, rows 0 .. 71 and 72 .. 139,
// each in a buffer of its own, as the plan splits a weight wider than a
// binding; source rows of each width, filled on the device; the input and
// three output buffers.
struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange source{}, input{}, out[3]{};

    [[nodiscard]] const residency::WeightView& view(const std::string& name) const {
        for (const auto& t : model.plan.tensors) {
            if (model.index.tensor(t.tensor).name == name) return t.view;
        }
        FAIL("no tensor " << name);
        return model.plan.tensors.front().view;
    }

    [[nodiscard]] const gguf::TensorEntry& entry(const std::string& name) const {
        for (const auto& t : model.index.tensors()) {
            if (t.name == name) return t;
        }
        FAIL("no tensor " << name);
        return model.index.tensors().front();
    }
};

Uploaded upload_weights(WGPUInstance instance, const gpu::Device& device, bool split = false) {
    Uploaded u;
    u.model.bytes = load_gguf_fixture("matmul_rows");
    gguf::MemoryByteSource source{u.model.bytes};
    REQUIRE(gguf::read_index(source, u.model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = u.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    for (std::size_t i = 0; i < u.model.index.tensors().size(); ++i) {
        const gguf::TensorEntry& t = u.model.index.tensors()[i];
        const std::uint64_t offset = (plan.buffers[0].size + 255) / 256 * 256;
        gguf::TensorShape shape{};
        shape.dimension_count = t.dimension_count;
        shape.dimensions[0] = t.dimensions[0];
        shape.dimensions[1] = t.dimension_count > 1 ? t.dimensions[1] : 1;
        shape.element_count = t.element_count;
        const std::uint64_t rows = t.element_count / t.dimensions[0];
        if (split && t.name == "w_q4_0") {
            const std::uint64_t row_bytes = t.data_length / rows;
            std::vector<residency::WeightPiece> pieces;
            for (const auto [first, count] : {std::pair<std::uint64_t, std::uint64_t>{0, 72}, {72, 68}}) {
                plan.buffers.push_back({residency::Pool::Weights, count * row_bytes});
                pieces.push_back({static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, count * row_bytes,
                                  first, count});
            }
            plan.tensors.push_back({gguf::TensorId{static_cast<std::uint32_t>(i)},
                                    residency::WeightView(t.type, shape, std::move(pieces)), std::nullopt});
            continue;
        }
        plan.tensors.push_back({gguf::TensorId{static_cast<std::uint32_t>(i)},
                                residency::WeightView(t.type, shape,
                                                      {{residency::BufferIndex{0}, offset, (t.data_length + 3) / 4 * 4,
                                                        0, rows}}),
                                std::nullopt});
        plan.buffers[0].size = offset + (t.data_length + 3) / 4 * 4;
    }
    const auto alone = [&](std::uint64_t bytes) {
        plan.buffers.push_back({residency::Pool::Scratch, bytes});
        return residency::BufferRange{static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, bytes};
    };
    u.source = alone(kTokens * kWidest * 4);
    u.input = alone(kTokens * kWidest * 4);
    for (auto& o : u.out) o = alone(kTokens * kMostRows * 4);
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == residency::UploadError::Ok);
    const auto program = build_program(instance, *u.upload, {fill_launch(u.source, 7, 1.0f, false)});
    run_step(instance, *program, 1);
    return u;
}

// A tensor's rows, decoded on the CPU as the format's reference does.
std::vector<float> decoded(const Uploaded& u, const std::string& name) {
    const gguf::TensorEntry& t = u.entry(name);
    const auto bytes = std::span(reinterpret_cast<const std::uint8_t*>(u.model.bytes.data()) + t.data_offset,
                                 t.data_length);
    switch (t.type) {
        case gguf::TensorType::Q4_0: return test::dequantize_q4_0(bytes);
        case gguf::TensorType::Q4_1: return test::dequantize_q4_1(bytes).rounded;
        case gguf::TensorType::Q8_0: return test::dequantize_q8_0(bytes);
        case gguf::TensorType::Q6_K: return test::dequantize_q6_k(bytes);
        default: FAIL("no reference for " << name); return {};
    }
}

// A step of `tokens` source rows from `first`, through `launches`: what
// the outputs hold after it, `widths[o]` floats a row of output o.
std::vector<std::vector<float>> run(WGPUInstance instance, const gpu::Device& device, const Uploaded& u,
                                    std::uint32_t columns, std::vector<kernels::Launch> launches,
                                    std::uint32_t tokens, std::uint32_t first, std::vector<std::uint32_t> widths) {
    std::vector<std::byte> copy(8);
    const std::uint32_t words[2] = {columns / 4, first};
    std::memcpy(copy.data(), words, 8);
    launches.insert(launches.begin(),
                    kernels::Launch{kCopy, nullptr, copy,
                                    {{u.source.buffer, 0, u.source.length}, {u.input.buffer, 0, u.input.length}},
                                    columns / 4, 64});
    const auto program = build_program(instance, *u.upload, std::move(launches));
    run_step(instance, *program, tokens);
    std::vector<std::vector<float>> out;
    for (std::size_t o = 0; o < widths.size(); ++o) {
        out.push_back(read_floats(instance, device, u.upload->buffer(u.out[o].buffer), 0, std::size_t{tokens} * widths[o]));
    }
    return out;
}

std::vector<float> source_rows(WGPUInstance instance, const gpu::Device& device, const Uploaded& u) {
    return read_floats(instance, device, u.upload->buffer(u.source.buffer), 0, kTokens * kWidest);
}

// γ(n) = n u / (1 − n u), u = 2⁻²⁴ (matmul.h).
double gamma(std::uint64_t n) {
    const double nu = static_cast<double>(n) * std::ldexp(1.0, -24);
    return nu / (1 - nu);
}

// Output (token, row) in f64, and its bound: γ(2K) Σ |w x|.
struct Want {
    double value, bound;
};

Want reference(std::span<const float> w, std::span<const float> source, std::uint32_t columns, std::uint32_t row,
               std::uint32_t source_row) {
    double sum = 0, absolute = 0;
    for (std::uint32_t k = 0; k < columns; ++k) {
        const double p = static_cast<double>(w[std::size_t{row} * columns + k]) *
                         source[std::size_t{source_row} * columns + k];
        sum += p;
        absolute += std::abs(p);
    }
    return {sum, gamma(2 * std::uint64_t{columns}) * absolute};
}

kernels::MatmulLaunch single(const Uploaded& u, const residency::WeightView& w, kernels::Rows rows) {
    return {{&w, nullptr, nullptr}, 1, u.input, {u.out[0], u.out[1], u.out[2]},
            kernels::Epilogue::Write, model::FeedForwardActivation::SiLU, rows};
}

}  // namespace

TEST_CASE("each format's product is within its bound, decoded and prefilled, and the same bits either way") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_weights(instance.get(), *device);
    const auto source = source_rows(instance.get(), *device, u);
    double worst = 0;
    // Then narrow weights whose decode sets take 2, 3 and 4 rows.
    for (const char* name : {"w_q4_0", "w_q4_1_1152", "w_q4_1_3072", "w_q8_0", "w_q6_k", "w_set2", "w_set3", "w_set4"}) {
        CAPTURE(name);
        const residency::WeightView& w = u.view(name);
        const auto columns = static_cast<std::uint32_t>(w.shape().dimensions[0]);
        const auto rows = static_cast<std::uint32_t>(w.shape().dimensions[1]);
        // Source rows of this width: the fill covers the widest; a row of
        // `columns` floats starts every `columns` floats.
        const auto launches = kernels::matmul_launches(single(u, w, kernels::Rows::EveryToken));
        REQUIRE(launches.size() == 4);
        const auto weights = decoded(u, name);
        // Prefill steps of 2 and 8 rows, in the 8-token tile, 9 and 16, in
        // the 16-token, and 17, 33 and 64, in the 32-token, from row 0.
        std::vector<std::vector<float>> prefilled;
        for (const std::uint32_t tokens : {2u, 8u, 9u, 16u, 17u, 33u, 64u}) {
            CAPTURE(tokens);
            const auto got = run(instance.get(), *device, u, columns, launches, tokens, 0, {rows});
            for (std::uint32_t t = 0; t < tokens; ++t) {
                for (std::uint32_t o = 0; o < rows; ++o) {
                    const Want want = reference(weights, source, columns, o, t);
                    const double error = std::abs(got[0][std::size_t{t} * rows + o] - want.value);
                    worst = std::max(worst, error / want.bound);
                    if (!(error <= want.bound)) {   // NaN fails too
                        FAIL_CHECK("token " << t << " row " << o << " off by " << error);
                        break;
                    }
                }
            }
            prefilled.push_back(got[0]);
        }
        // Decoded alone, rows at each tile's edges: the same bits as every
        // prefill that held them.
        for (const std::uint32_t t : {0u, 1u, 7u, 8u, 15u, 16u, 32u, 63u}) {
            CAPTURE(t);
            const auto alone = run(instance.get(), *device, u, columns, launches, 1, t, {rows});
            for (std::size_t s = 0; s < prefilled.size(); ++s) {
                const std::size_t tokens = prefilled[s].size() / rows;
                if (t >= tokens) continue;
                CHECK(std::equal(alone[0].begin(), alone[0].end(), prefilled[s].begin() + t * rows));
            }
        }
    }
    MESSAGE("largest error, against its bound: " << worst);
}

TEST_CASE("a weight in pieces is four launches a piece, each writing its rows") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded whole = upload_weights(instance.get(), *device);
    const Uploaded split = upload_weights(instance.get(), *device, true);
    const auto pieces = kernels::matmul_launches(single(split, split.view("w_q4_0"), kernels::Rows::EveryToken));
    REQUIRE(pieces.size() == 8);
    const auto one = kernels::matmul_launches(single(whole, whole.view("w_q4_0"), kernels::Rows::EveryToken));
    // Every row's products in the same order whichever piece holds it.
    for (const std::uint32_t tokens : {1u, 33u}) {
        CAPTURE(tokens);
        const auto in_pieces = run(instance.get(), *device, split, 1024, pieces, tokens, 0, {140});
        const auto in_one = run(instance.get(), *device, whole, 1024, one, tokens, 0, {140});
        CHECK(std::equal(in_pieces[0].begin(), in_pieces[0].end(), in_one[0].begin()));
    }
}

TEST_CASE("the head covers the step's last token alone, in a prefill step too") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_weights(instance.get(), *device);
    const residency::WeightView& w = u.view("w_q6_k");
    const auto launches = kernels::matmul_launches(single(u, w, kernels::Rows::LastToken));
    REQUIRE(launches.size() == 1);
    const auto last = run(instance.get(), *device, u, 1024, launches, 5, 0, {40});
    const auto alone = run(instance.get(), *device, u, 1024, launches, 1, 4, {40});
    CHECK(std::equal(alone[0].begin(), alone[0].begin() + 40, last[0].begin()));
}

TEST_CASE("Q, K and V as one product land in their three buffers, the same bits decoded and prefilled") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_weights(instance.get(), *device);
    const auto source = source_rows(instance.get(), *device, u);
    const kernels::MatmulLaunch m{{&u.view("qkv_q"), &u.view("qkv_k"), &u.view("qkv_v")},
                                  3,
                                  u.input,
                                  {u.out[0], u.out[1], u.out[2]},
                                  kernels::Epilogue::QKV,
                                  model::FeedForwardActivation::SiLU,
                                  kernels::Rows::EveryToken};
    const auto launches = kernels::matmul_launches(m);
    REQUIRE(launches.size() == 4);
    const std::vector<std::uint32_t> widths{64, 32, 32};
    const char* names[] = {"qkv_q", "qkv_k", "qkv_v"};
    const auto prefilled = run(instance.get(), *device, u, 1024, launches, 33, 0, widths);
    for (std::size_t o = 0; o < 3; ++o) {
        CAPTURE(names[o]);
        const auto weights = decoded(u, names[o]);
        for (std::uint32_t t = 0; t < 33; ++t) {
            for (std::uint32_t r = 0; r < widths[o]; ++r) {
                const Want want = reference(weights, source, 1024, r, t);
                CHECK(std::abs(prefilled[o][std::size_t{t} * widths[o] + r] - want.value) <= want.bound);
            }
        }
    }
    for (const std::uint32_t t : {0u, 20u, 32u}) {
        CAPTURE(t);
        const auto alone = run(instance.get(), *device, u, 1024, launches, 1, t, widths);
        for (std::size_t o = 0; o < 3; ++o) {
            CHECK(std::equal(alone[o].begin(), alone[o].end(), prefilled[o].begin() + t * widths[o]));
        }
    }
    // Steps of 5 and 12, in the 8- and 16-token tiles: the same bits as the
    // 32-token tile's.
    for (const std::uint32_t tokens : {5u, 12u}) {
        CAPTURE(tokens);
        const auto narrow = run(instance.get(), *device, u, 1024, launches, tokens, 0, widths);
        for (std::size_t o = 0; o < 3; ++o) {
            CHECK(std::equal(narrow[o].begin(), narrow[o].end(), prefilled[o].begin()));
        }
    }
}

TEST_CASE("gate and up as one product write activation(gate) x up, for SiLU and GELU") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_weights(instance.get(), *device);
    const auto source = source_rows(instance.get(), *device, u);
    const auto gate = decoded(u, "ffn_gate"), up = decoded(u, "ffn_up");
    for (const auto activation : {model::FeedForwardActivation::SiLU, model::FeedForwardActivation::GeluTanh}) {
        const bool gelu = activation == model::FeedForwardActivation::GeluTanh;
        CAPTURE(gelu);
        const kernels::MatmulLaunch m{{&u.view("ffn_gate"), &u.view("ffn_up"), nullptr},
                                      2,
                                      u.input,
                                      {u.out[0], u.out[1], u.out[2]},
                                      kernels::Epilogue::GatedActivation,
                                      activation,
                                      kernels::Rows::EveryToken};
        const auto launches = kernels::matmul_launches(m);
        const auto prefilled = run(instance.get(), *device, u, 1024, launches, 33, 0, {72});
        for (std::uint32_t t = 0; t < 33; ++t) {
            for (std::uint32_t r = 0; r < 72; ++r) {
                const Want g = reference(gate, source, 1024, r, t), v = reference(up, source, 1024, r, t);
                const double x = g.value;
                const double act = gelu ? 0.5 * x * (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * (x + 0.044715 * x * x * x)))
                                        : x / (1 + std::exp(-x));
                // Each input's own bound carried through: the activation's
                // slope is at most 1.13 for either, and its f32 evaluation
                // within 2⁻¹⁶ of its value.
                const double bound = 1.13 * g.bound * std::abs(v.value) + std::abs(act) * v.bound +
                                     std::ldexp(std::abs(act * v.value), -16) + std::ldexp(1.0, -30);
                CHECK(std::abs(prefilled[0][std::size_t{t} * 72 + r] - act * v.value) <= bound);
            }
        }
        for (const std::uint32_t t : {0u, 32u}) {
            CAPTURE(t);
            const auto alone = run(instance.get(), *device, u, 1024, launches, 1, t, {72});
            CHECK(std::equal(alone[0].begin(), alone[0].end(), prefilled[0].begin() + t * 72));
        }
        for (const std::uint32_t tokens : {5u, 12u}) {
            CAPTURE(tokens);
            const auto narrow = run(instance.get(), *device, u, 1024, launches, tokens, 0, {72});
            CHECK(std::equal(narrow[0].begin(), narrow[0].end(), prefilled[0].begin()));
        }
    }
}
