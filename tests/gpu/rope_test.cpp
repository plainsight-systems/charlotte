// QK-norm, RoPE and the cache append on Dawn, against the same computation in
// f64 from the launcher's turns (kernels/rope/rope.h): each listed model's
// shape at positions from 0 to the end of its context; values in the cache
// bit for bit; a step whose slots wrap a ring; and the same tokens prefilled
// as one step and decoded one at a time, writing the same bits.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "core/formats/f16/f16.h"
#include "core/gguf/reader.h"
#include "core/kernels/rope/rope.h"
#include "core/quant/q4_0.h"
#include "support/acquire.h"
#include "support/f16_reference.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kRows = 4;
constexpr std::uint32_t kMaxSlots = 64;
constexpr float kEpsilon = 1e-6f;

// One listed model's attention shape.
struct Shape {
    const char* name;
    std::uint32_t query_heads, key_value_heads, d;
    model::RotaryPairing pairing;
    bool qk_norm, factors;
    float base;
};

constexpr Shape kQwen{"qwen", 16, 8, 128, model::RotaryPairing::Halves, true, false, 1e6f};
constexpr Shape kLlama{"llama", 32, 8, 64, model::RotaryPairing::Adjacent, false, true, 5e5f};
constexpr Shape kGemma{"gemma", 4, 1, 256, model::RotaryPairing::Halves, true, false, 1e4f};

// Test-only: copies rows first .. first + tokens of a tensor into a working
// buffer, so a step starts from the fixture's activations.
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

// The fixture's tensors in one weights buffer; the query, key and value
// working buffers alone, as the plan places them; one buffer holding a
// layer's keys, then its values, as the plan's cache pool does — made in the
// scratch pool, whose buffers can be read back, which the cache pool's are
// not (residency/upload.h).
struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange query{}, key{}, value{};
    residency::PlannedCacheLayer cache{};

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
    u.model.bytes = load_gguf_fixture("rope_rows");
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
    const std::uint64_t rows = kRows * 2048 * 4;   // every shape's widest, 16 × 128 = 32 × 64 = 2,048
    for (auto* range : {&u.query, &u.key, &u.value}) {
        plan.buffers.push_back({residency::Pool::Scratch, rows});
        *range = {static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, rows};
    }
    const std::uint64_t layer = kMaxSlots * 8 * 128 * 2;   // the most slots of the widest key-value heads
    plan.buffers.push_back({residency::Pool::Scratch, 2 * layer});
    const auto cache = static_cast<residency::BufferIndex>(plan.buffers.size() - 1);
    u.cache = {{cache, 0, layer}, {cache, layer, layer}, kMaxSlots};
    Ready ready;
    residency::Upload::begin(device, u.model.index, plan, u.model.bytes.size(), capability::find_format, {}, kChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    u.upload = std::move(ready.upload);
    REQUIRE(stream(instance, *u.upload, u.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *u.upload) == residency::UploadError::Ok);
    return u;
}

kernels::Launch copy_launch(const residency::WeightView& from, const residency::BufferRange& to, std::uint32_t width,
                            std::uint32_t first) {
    const std::uint32_t words[2] = {width / 4, first};
    std::vector<std::byte> bytes(sizeof words);
    std::memcpy(bytes.data(), words, sizeof words);
    const auto& piece = from.pieces().front();
    return {kCopy, nullptr, std::move(bytes),
            {{piece.buffer, piece.offset, piece.length}, {to.buffer, to.offset, to.length}}, width / 4, 64};
}

kernels::Launch launch_for(const Uploaded& u, const Shape& s, std::uint32_t slots) {
    const std::string n = s.name;
    residency::PlannedCacheLayer cache = u.cache;
    cache.slots = slots;
    return kernels::rope_launch({model::LayerDescription{s.query_heads, s.key_value_heads, s.d, 0, 0, s.base, {}},
                                 s.pairing, s.qk_norm ? &u.view("qn_" + n) : nullptr,
                                 s.qk_norm ? &u.view("kn_" + n) : nullptr, s.factors ? &u.view("factors_" + n) : nullptr,
                                 kEpsilon, u.query, u.key, u.value, cache, &formats::kF16});
}

// What a step leaves: its rows of the query buffer, and the cache's halves.
struct Result {
    std::vector<float> query;
    std::vector<std::uint16_t> keys, values;
};

std::vector<std::uint16_t> read_halves(WGPUInstance instance, const gpu::Device& device, const Uploaded& u,
                                       const residency::BufferRange& range) {
    const auto words = read_floats(instance, device, u.upload->buffer(range.buffer), range.offset, range.length / 4);
    std::vector<std::uint16_t> halves(words.size() * 2);
    std::memcpy(halves.data(), words.data(), words.size() * 4);
    return halves;
}

// A step of `tokens` rows of the fixture from row `first`, at `position`,
// into a cache of `slots`.
Result run_rope(WGPUInstance instance, const gpu::Device& device, const Uploaded& u, const Shape& s,
                std::uint32_t position, std::uint32_t tokens, std::uint32_t first, std::uint32_t slots) {
    const std::string n = s.name;
    std::vector<kernels::Launch> launches;
    launches.push_back(copy_launch(u.view("q_" + n), u.query, s.query_heads * s.d, first));
    launches.push_back(copy_launch(u.view("k_" + n), u.key, s.key_value_heads * s.d, first));
    launches.push_back(copy_launch(u.view("v_" + n), u.value, s.key_value_heads * s.d, first));
    launches.push_back(launch_for(u, s, slots));
    const auto program = build_program(instance, *u.upload, std::move(launches));
    run_step(instance, *program, tokens, {}, position);
    return {read_floats(instance, device, u.upload->buffer(u.query.buffer), 0, tokens * s.query_heads * s.d),
            read_halves(instance, device, u, u.cache.keys), read_halves(instance, device, u, u.cache.values)};
}

// One head of one token in f64: normalized where the shape has QK-norm,
// then rotated by the launcher's turns. `t_of` is each pair's t, for the
// bound.
struct Head {
    std::vector<double> values;       // normalized and rotated: what the kernel should write
    std::vector<double> normalized;   // before rotation: the pair the bound is stated in
    std::vector<double> t_of_pair;
    double largest;   // the normalized head's largest magnitude
};

Head reference_head(const Shape& s, std::span<const float> x, std::span<const float> gain,
                    std::span<const float> factors, std::span<const float> turns, std::uint32_t p, bool rotate) {
    Head h{std::vector<double>(x.begin(), x.end()), {}, std::vector<double>(s.d / 2), 0};
    if (s.qk_norm && rotate) {
        double squares = 0;
        for (const double e : h.values) squares += e * e;
        const double r = 1.0 / std::sqrt(squares / s.d + static_cast<double>(kEpsilon));
        for (std::uint32_t j = 0; j < s.d; ++j) h.values[j] = h.values[j] * r * gain[j];
    }
    for (const double e : h.values) h.largest = std::max(h.largest, std::abs(e));
    h.normalized = h.values;
    if (!rotate) return h;
    std::vector<double> out(h.values);
    for (std::uint32_t k = 0; k < s.d / 2; ++k) {
        double t = static_cast<double>(p) * turns[k];
        if (s.factors) t /= factors[k];
        h.t_of_pair[k] = t;
        const double angle = 2 * std::numbers::pi * (t - std::round(t));
        const std::uint32_t i = s.pairing == model::RotaryPairing::Halves ? k : 2 * k;
        const std::uint32_t j = s.pairing == model::RotaryPairing::Halves ? k + s.d / 2 : 2 * k + 1;
        out[i] = h.values[i] * std::cos(angle) - h.values[j] * std::sin(angle);
        out[j] = h.values[i] * std::sin(angle) + h.values[j] * std::cos(angle);
    }
    h.values = out;
    return h;
}

// The bound rope.h states for one rotated value: (|x| + |y|) × (2⁻¹¹ + 2π ×
// 2⁻²⁴ × |t|), where t's division by a factor adds WGSL's 2.5 units of a
// division, 5 × 2⁻²⁴ × |t| more; and the norm's margin against the head's
// largest, twice, since a rotated value mixes two normalized ones.
double bound(const Shape& s, const Head& h, std::uint32_t i) {
    const std::uint32_t k = s.pairing == model::RotaryPairing::Halves ? i % (s.d / 2) : i / 2;
    const std::uint32_t mate = s.pairing == model::RotaryPairing::Halves ? (i + s.d / 2) % s.d : i ^ 1u;
    const double pair = std::abs(h.normalized[i]) + std::abs(h.normalized[mate]);
    const double t = std::abs(h.t_of_pair[k]);
    const double units_of_t = s.factors ? 6 : 1;
    return pair * (std::ldexp(1.0, -11) + 2 * std::numbers::pi * units_of_t * std::ldexp(1.0, -24) * t) +
           h.largest * std::ldexp(1.0, -17);
}

float half_value(std::uint16_t h) { return quant::fp16_to_fp32(h); }

// Checks one step's queries and cache against f64, and returns the largest
// error seen against its bound.
double check_step(const Uploaded& u, const Shape& s, const Result& got, std::uint32_t position, std::uint32_t tokens,
                  std::uint32_t first, std::uint32_t slots, std::span<const float> turns) {
    const std::string n = s.name;
    const auto q = u.floats("q_" + n), k = u.floats("k_" + n), v = u.floats("v_" + n);
    const auto qn = s.qk_norm ? u.floats("qn_" + n) : std::vector<float>(s.d, 1.0f);
    const auto kn = s.qk_norm ? u.floats("kn_" + n) : std::vector<float>(s.d, 1.0f);
    const auto factors = s.factors ? u.floats("factors_" + n) : std::vector<float>(s.d / 2, 1.0f);
    double worst = 0;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const std::uint32_t p = position + t, row = first + t, slot = p % slots;
        CAPTURE(p);
        for (std::uint32_t h = 0; h < s.query_heads; ++h) {
            const auto x = std::span(q).subspan((row * s.query_heads + h) * s.d, s.d);
            const Head want = reference_head(s, x, qn, factors, turns, p, true);
            for (std::uint32_t i = 0; i < s.d; ++i) {
                const double error = std::abs(got.query[(t * s.query_heads + h) * s.d + i] - want.values[i]);
                worst = std::max(worst, error / bound(s, want, i));
                if (!(error <= bound(s, want, i))) {   // NaN fails too
                    FAIL_CHECK("query head " << h << " value " << i << " off by " << error);
                    return worst;
                }
            }
        }
        for (std::uint32_t h = 0; h < s.key_value_heads; ++h) {
            const std::size_t at = (row * s.key_value_heads + h) * s.d;
            const std::size_t stored = (std::size_t{slot} * s.key_value_heads + h) * s.d;
            const Head want = reference_head(s, std::span(k).subspan(at, s.d), kn, factors, turns, p, true);
            for (std::uint32_t i = 0; i < s.d; ++i) {
                // Rounded once more, to f16: half a unit in its last place.
                const double allowed = bound(s, want, i) + std::abs(want.values[i]) * std::ldexp(1.0, -11) +
                                       std::ldexp(1.0, -25);
                const double error = std::abs(half_value(got.keys[stored + i]) - want.values[i]);
                worst = std::max(worst, error / allowed);
                if (!(error <= allowed)) {   // NaN fails too
                    FAIL_CHECK("key head " << h << " value " << i << " off by " << error);
                    return worst;
                }
                // Values: the nearest f16 of the input, bit for bit.
                if (got.values[stored + i] != test::fp32_to_fp16_saturating(v[at + i])) {
                    FAIL_CHECK("value head " << h << " value " << i << " is not its input's f16");
                    return worst;
                }
            }
        }
    }
    return worst;
}

std::vector<float> turns_of(const kernels::Launch& launch, std::uint32_t d) {
    std::vector<float> turns(d / 2);
    std::memcpy(turns.data(), launch.constants.data() + 16, turns.size() * 4);
    return turns;
}

}  // namespace

TEST_CASE("each listed shape's queries and cache are normalized and rotated, from position 0 to the context's end") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    struct Case {
        Shape shape;
        std::uint32_t last;   // the context's last position
    };
    double worst = 0;
    for (const Case& c : {Case{kQwen, 40959}, Case{kLlama, 131071}, Case{kGemma, 32767}}) {
        for (const std::uint32_t position : {0u, c.last + 1 - kRows}) {
            CAPTURE(c.shape.name);
            CAPTURE(position);
            const Result got = run_rope(instance.get(), *device, u, c.shape, position, kRows, 0, kMaxSlots);
            const auto turns = turns_of(launch_for(u, c.shape, kMaxSlots), c.shape.d);
            worst = std::max(worst, check_step(u, c.shape, got, position, kRows, 0, kMaxSlots, turns));
        }
    }
    MESSAGE("largest error, against its bound: " << worst);
}

TEST_CASE("a step whose slots wrap a ring writes each token at its position mod the slots") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    // Gemma 3's window layers hold a ring; positions 32,764 .. 32,767 in 6
    // slots take slots 4, 5, 0 and 1.
    constexpr std::uint32_t kSlots = 6;
    const Result got = run_rope(instance.get(), *device, u, kGemma, 32764, kRows, 0, kSlots);
    const auto turns = turns_of(launch_for(u, kGemma, kSlots), kGemma.d);
    check_step(u, kGemma, got, 32764, kRows, 0, kSlots, turns);
}

TEST_CASE("tokens prefilled as one step and decoded one at a time write the same bits") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_rows(instance.get(), *device);
    for (const Shape& s : {kQwen, kLlama, kGemma}) {
        CAPTURE(s.name);
        constexpr std::uint32_t kPosition = 1000;
        const Result prefilled = run_rope(instance.get(), *device, u, s, kPosition, kRows, 0, kMaxSlots);
        const std::size_t per_token = std::size_t{s.query_heads} * s.d;
        for (std::uint32_t t = 0; t < kRows; ++t) {
            CAPTURE(t);
            const Result decoded = run_rope(instance.get(), *device, u, s, kPosition + t, 1, t, kMaxSlots);
            CHECK(std::equal(decoded.query.begin(), decoded.query.end(), prefilled.query.begin() + t * per_token));
            const std::size_t slot = (kPosition + t) % kMaxSlots;
            const std::size_t width = std::size_t{s.key_value_heads} * s.d;
            CHECK(std::equal(decoded.keys.begin() + slot * width, decoded.keys.begin() + (slot + 1) * width,
                             prefilled.keys.begin() + slot * width));
            CHECK(std::equal(decoded.values.begin() + slot * width, decoded.values.begin() + (slot + 1) * width,
                             prefilled.values.begin() + slot * width));
        }
    }
}
