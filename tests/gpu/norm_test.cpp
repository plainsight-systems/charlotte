// The norm on Dawn, against the same computation in f64 (kernels/norm/
// norm.h): every variant at each listed model's hidden width, rows of
// ordinary, large and tiny size, a row's result alike in a step of four rows
// and of one, and the final norm touching only the last row.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/norm/norm.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kRows = 4;
constexpr std::uint64_t kMaxWidth = 2048;
constexpr float kEpsilon = 1e-6f;

// Test-only: copies a tensor's rows into a working buffer, so a step starts
// from the fixture's activations. Working buffers are written only by
// kernels.
constexpr std::string_view kCopy = R"(
struct Copy { vec4s: u32 }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> copy: Copy;
@group(0) @binding(2) var<storage, read> copy_from: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> copy_to: array<vec4<f32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < step.tokens * copy.vec4s) {
        copy_to[id.x] = copy_from[id.x];
    }
}
)";

// The fixture's every tensor in one weights buffer, and the three working
// buffers a norm uses, each four rows of the widest width.
struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange hidden{}, output{}, normed{};

    [[nodiscard]] const residency::WeightView& view(const std::string& name) const {
        for (const auto& t : model.plan.tensors) {
            if (model.index.tensor(t.tensor).name == name) return t.view;
        }
        FAIL("no tensor " << name);
        return model.plan.tensors.front().view;
    }

    [[nodiscard]] std::vector<float> floats(const std::string& name) const {
        for (const gguf::TensorEntry& t : model.index.tensors()) {
            if (t.name != name) continue;
            std::vector<float> out(t.element_count);
            std::memcpy(out.data(), model.bytes.data() + t.data_offset, t.data_length);
            return out;
        }
        FAIL("no tensor " << name);
        return {};
    }
};

Uploaded upload_rows(WGPUInstance instance, const gpu::Device& device) {
    Uploaded u;
    u.model.bytes = load_gguf_fixture("norm_rows");
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
        shape.dimensions[1] = t.dimensions[1];
        shape.element_count = t.element_count;
        plan.tensors.push_back({gguf::TensorId{static_cast<std::uint32_t>(i)},
                                residency::WeightView(t.type, shape,
                                                      {{residency::BufferIndex{0}, offset, t.data_length, 0,
                                                        t.element_count / t.dimensions[0]}}),
                                std::nullopt});
        plan.buffers[0].size = offset + t.data_length;
    }
    const std::uint64_t bytes = kRows * kMaxWidth * 4;
    for (auto* range : {&u.hidden, &u.output, &u.normed}) {
        plan.buffers.push_back({residency::Pool::Scratch, bytes});
        *range = {static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, bytes};
    }
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == UploadError::Ok);
    return u;
}

kernels::Launch copy_launch(const residency::WeightView& from, const residency::BufferRange& to, std::uint32_t width) {
    const std::uint32_t vec4s = width / 4;
    std::vector<std::byte> bytes(4);
    std::memcpy(bytes.data(), &vec4s, 4);
    const auto& piece = from.pieces().front();
    return {kCopy, nullptr, std::move(bytes),
            {{piece.buffer, piece.offset, piece.length}, {to.buffer, to.offset, to.length}}, vec4s, 64};
}

struct Variant {
    bool add;
    bool post;
    kernels::Rows rows = kernels::Rows::EveryToken;
};

// A step of `tokens` rows: x copied into hidden, y into output, then the norm.
// Returns hidden and normed afterwards, the step's rows of each.
struct Result {
    std::vector<float> hidden, normed;
};

Result run_norm(WGPUInstance instance, const gpu::Device& device, const Uploaded& u, std::uint32_t width, Variant v,
                std::uint32_t tokens, bool logits = true) {
    const std::string w = std::to_string(width);
    std::vector<kernels::Launch> launches;
    launches.push_back(copy_launch(u.view("x_" + w), u.hidden, width));
    launches.push_back(copy_launch(u.view("y_" + w), u.output, width));
    launches.push_back(kernels::norm_launch({&u.view("gain_" + w), v.post ? &u.view("post_" + w) : nullptr, v.add,
                                             u.output, u.hidden, u.normed, kEpsilon, v.rows}));
    const auto program = build_program(instance, *u.upload, std::move(launches));
    run_step(instance, *program, tokens, {}, 0, logits);
    return {read_floats(instance, device, u.upload->buffer(u.hidden.buffer), 0, tokens * width),
            read_floats(instance, device, u.upload->buffer(u.normed.buffer), 0, tokens * width)};
}

// The norm in f64, for one row: what hidden and normed should hold.
struct Reference {
    std::vector<double> hidden, normed;
};

Reference reference(const Uploaded& u, std::uint32_t width, Variant v, std::uint32_t row) {
    const std::string w = std::to_string(width);
    const auto x = u.floats("x_" + w), y = u.floats("y_" + w), g = u.floats("gain_" + w), pg = u.floats("post_" + w);
    const auto rms_scale = [&](const std::vector<double>& r) {
        double s = 0;
        for (const double e : r) s += e * e;
        return 1.0 / std::sqrt(s / width + kEpsilon);
    };
    Reference ref;
    for (std::uint32_t i = 0; i < width; ++i) ref.hidden.push_back(x[row * width + i]);
    if (v.add) {
        std::vector<double> yr(y.begin() + row * width, y.begin() + (row + 1) * width);
        if (v.post) {
            const double r = rms_scale(yr);
            for (std::uint32_t i = 0; i < width; ++i) yr[i] = yr[i] * r * pg[i];
        }
        for (std::uint32_t i = 0; i < width; ++i) ref.hidden[i] += yr[i];
    }
    const double r = rms_scale(ref.hidden);
    for (std::uint32_t i = 0; i < width; ++i) ref.normed.push_back(ref.hidden[i] * r * g[i]);
    return ref;
}

// Every output within 2⁻¹⁸ of the f64 result, against its row's largest
// (norm.h); returns the largest error seen, in those terms.
double check_row(std::span<const float> got, const std::vector<double>& want) {
    double largest = 0;
    for (const double e : want) largest = std::max(largest, std::abs(e));
    double worst = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double error = std::abs(got[i] - want[i]) / largest;
        if (!(error <= std::ldexp(1.0, -18))) {   // NaN fails too, which std::max would drop
            FAIL_CHECK("value " << i << " off by " << error << " of its row's largest");
            return error;
        }
        worst = std::max(worst, error);
    }
    return worst;
}

}  // namespace

TEST_CASE("every variant of the norm writes each row normalized, at every listed width") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    double worst = 0;
    for (const std::uint32_t width : {1024u, 1152u, 2048u}) {
        for (const Variant v : {Variant{false, false}, Variant{true, false}, Variant{true, true}}) {
            CAPTURE(width);
            CAPTURE(v.add);
            CAPTURE(v.post);
            const Result got = run_norm(instance.get(), *device, u, width, v, kRows);
            for (std::uint32_t row = 0; row < kRows; ++row) {
                CAPTURE(row);
                const Reference want = reference(u, width, v, row);
                const auto at = [&](const std::vector<float>& all) {
                    return std::span(all).subspan(row * width, width);
                };
                worst = std::max(worst, check_row(at(got.normed), want.normed));
                if (!v.post) {
                    // x, or x + y rounded once: exactly the f32 sum.
                    for (std::uint32_t i = 0; i < width; ++i) {
                        if (at(got.hidden)[i] != static_cast<float>(want.hidden[i])) {
                            FAIL_CHECK("hidden " << i << " is not x" << (v.add ? " + y" : ""));
                            break;
                        }
                    }
                } else {
                    worst = std::max(worst, check_row(at(got.hidden), want.hidden));
                }
            }
        }
    }
    MESSAGE("largest error, against its row's largest output: 2^" << std::log2(worst));
}

TEST_CASE("a row's norm is the same in a step of four rows and of one") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    for (const Variant v : {Variant{true, false}, Variant{true, true}}) {
        const Result four = run_norm(instance.get(), *device, u, 1152, v, 4);
        const Result one = run_norm(instance.get(), *device, u, 1152, v, 1);
        CHECK(std::equal(one.normed.begin(), one.normed.end(), four.normed.begin()));
        CHECK(std::equal(one.hidden.begin(), one.hidden.end(), four.hidden.begin()));
    }
}

TEST_CASE("the final norm adds and normalizes only the step's last row") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    const std::uint32_t width = 1024;
    const Variant v{true, false, kernels::Rows::LastToken};
    const Result got = run_norm(instance.get(), *device, u, width, v, kRows);
    const auto x = u.floats("x_1024");
    for (std::uint32_t row = 0; row + 1 < kRows; ++row) {
        CAPTURE(row);
        CHECK(std::equal(got.hidden.begin() + row * width, got.hidden.begin() + (row + 1) * width,
                         x.begin() + row * width));
        CHECK(std::all_of(got.normed.begin() + row * width, got.normed.begin() + (row + 1) * width,
                          [](float f) { return f == 0.0f; }));
    }
    const Reference want = reference(u, width, v, kRows - 1);
    check_row(std::span(got.normed).subspan((kRows - 1) * width, width), want.normed);
}

TEST_CASE("the final norm runs no row in a step that asks for no logits") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    const std::uint32_t width = 1024;
    const Result got =
        run_norm(instance.get(), *device, u, width, {true, false, kernels::Rows::LastToken}, kRows, false);
    const auto x = u.floats("x_1024");
    CHECK(std::equal(got.hidden.begin(), got.hidden.end(), x.begin()));
    CHECK(std::all_of(got.normed.begin(), got.normed.end(), [](float f) { return f == 0.0f; }));
}

// Why the plan gives each working buffer a buffer of its own
// (residency/plan.h): WebGPU refuses a dispatch that binds one buffer both
// writable and read-only, even in ranges that do not overlap. The step fails,
// and says why.
TEST_CASE("a step binding one buffer writable and read-only fails with WebGPU's message") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    const std::uint64_t half = u.hidden.length / 2;
    const residency::BufferRange hidden{u.hidden.buffer, 0, half};
    const residency::BufferRange output{u.hidden.buffer, half, half};   // same buffer, read-only
    std::vector<kernels::Launch> launches;
    launches.push_back(kernels::norm_launch(
        {&u.view("gain_1024"), nullptr, true, output, hidden, u.normed, kEpsilon, kernels::Rows::EveryToken}));
    const auto program = build_program(instance.get(), *u.upload, std::move(launches));
    const StepOutcome ran = try_step(instance.get(), *program, 1);
    CHECK(ran.error == kernels::ProgramError::Step);
    CHECK_MESSAGE(ran.message.find("writable usage") != std::string::npos, ran.message);
}

TEST_CASE("a launch setting an override the program owns, or one override twice, is refused by name") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    const auto norm = kernels::norm_launch(
        {&u.view("gain_1024"), nullptr, false, u.output, u.hidden, u.normed, kEpsilon, kernels::Rows::EveryToken});
    for (const kernels::Override extra :
         {kernels::Override{"workgroup_size", 64}, kernels::Override{"last_token", 1}, kernels::Override{"add", 1}}) {
        CAPTURE(extra.name);
        auto launch = norm;
        launch.overrides.push_back(extra);
        struct Built {
            kernels::ProgramError error = kernels::ProgramError::Ok;
            std::string message;
            bool done = false;
        } built;
        kernels::Program::build(
            *u.upload, {launch}, std::nullopt,
            [](std::unique_ptr<kernels::Program> p, kernels::ProgramError e, std::string_view m, void* userdata) {
                auto& b = *static_cast<Built*>(userdata);
                CHECK(p == nullptr);
                b.error = e;
                b.message = m;
                b.done = true;
            },
            &built);
        pump_until(instance.get(), built.done, "the refusal");
        CHECK(built.error == kernels::ProgramError::Build);
        CHECK(built.message.find(std::string(extra.name)) != std::string::npos);
    }
}
