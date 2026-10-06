// The embedding gather on Dawn: a table uploaded as the harness uploads one,
// launched by the program, each row read back and compared with the format's
// CPU reference applied to the same stored blocks (kernels/gather/gather.h).

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/gpu/callback_mode.h"
#include "core/kernels/gather/gather.h"
#include "core/kernels/program.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/pump.h"
#include "support/q4_0_reference.h"
#include "support/q4_1_reference.h"
#include "support/q6_k_reference.h"
#include "support/q8_0_reference.h"
#include "support/unpack.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;
using kernels::Program;
using namespace bllm::test;
using kernels::ProgramError;

namespace {

constexpr std::uint64_t kAlign = 256;

std::uint64_t round_up(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

// The fixture's table, uploaded with its rows split into pieces of the
// given sizes — each at the next 256-byte boundary, a new buffer started
// where `new_buffer_at` says — and a hidden buffer of 512 rows.
struct Uploaded {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange hidden{};
    std::uint64_t width = 0;
    std::uint64_t row_bytes = 0;
    gguf::TensorType type{};
};

Uploaded upload_table(WGPUInstance instance, const gpu::Device& device, const std::string& fixture,
                      std::vector<std::uint64_t> piece_rows, std::size_t new_buffer_at = SIZE_MAX) {
    Uploaded u;
    u.model.bytes = load_gguf_fixture(fixture);
    gguf::MemoryByteSource source{u.model.bytes};
    REQUIRE(gguf::read_index(source, u.model.index).error == gguf::ReadError::Ok);
    const gguf::TensorEntry& t = u.model.index.tensor(gguf::TensorId{0});
    u.type = t.type;
    u.width = t.dimensions[0];
    const gguf::FormatLayout* layout = gguf::format_layout(t.type);
    u.row_bytes = u.width / layout->block_elements * layout->block_bytes;

    residency::ResidencyPlan& plan = u.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    std::vector<residency::WeightPiece> pieces;
    std::uint64_t first = 0;
    for (std::size_t i = 0; i < piece_rows.size(); ++i) {
        if (i == new_buffer_at) plan.buffers.push_back({residency::Pool::Weights, 0});
        auto& buffer = plan.buffers.back();
        const std::uint64_t offset = round_up(buffer.size, kAlign);
        const std::uint64_t length = round_up(piece_rows[i] * u.row_bytes, 4);
        pieces.push_back({static_cast<residency::BufferIndex>(plan.buffers.size() - 1), offset, length, first,
                          piece_rows[i]});
        buffer.size = offset + length;
        first += piece_rows[i];
    }
    REQUIRE(first == t.dimensions[1]);
    gguf::TensorShape shape{};
    shape.dimension_count = 2;
    shape.dimensions[0] = t.dimensions[0];
    shape.dimensions[1] = t.dimensions[1];
    shape.element_count = t.element_count;
    plan.tensors.push_back({gguf::TensorId{0}, residency::WeightView(t.type, shape, std::move(pieces)), std::nullopt});
    const std::uint64_t hidden_bytes = residency::kPrefillBlock * u.width * 4;
    plan.buffers.push_back({residency::Pool::Scratch, hidden_bytes});
    u.hidden = {static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, hidden_bytes};
    plan.scratch.push_back({"hidden", u.hidden});

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

std::unique_ptr<Program> build(WGPUInstance instance, const Uploaded& u, float scale) {
    return build_program(instance, *u.upload, kernels::gather_launches(u.model.plan.tensors[0].view, u.hidden, scale));
}

void run(WGPUInstance instance, Program& program, std::span<const std::uint32_t> ids) {
    run_step(instance, program, static_cast<std::uint32_t>(ids.size()), ids);
}

// The first `rows` rows of the hidden buffer.
std::vector<float> read_hidden(WGPUInstance instance, const gpu::Device& device, const Uploaded& u, std::size_t rows) {
    return read_floats(instance, device, u.upload->buffer(u.hidden.buffer), u.hidden.offset, rows * u.width);
}

// What token `id`'s row should hold: the format's CPU reference applied to
// the stored blocks, times `scale`. Q4_1 may be rounded twice or fused
// (format.h), so it has a second answer.
struct Expected {
    std::vector<float> primary;
    std::vector<float> alternate;
};

Expected expected_row(const Uploaded& u, std::uint32_t id, float scale) {
    const gguf::TensorEntry& t = u.model.index.tensor(gguf::TensorId{0});
    const auto* at = reinterpret_cast<const std::uint8_t*>(u.model.bytes.data()) + t.data_offset + id * u.row_bytes;
    const std::span<const std::uint8_t> row(at, u.row_bytes);
    Expected e;
    switch (u.type) {
        case gguf::TensorType::F32:
            e.primary.resize(u.width);
            std::memcpy(e.primary.data(), at, u.row_bytes);
            break;
        case gguf::TensorType::Q4_0: e.primary = dequantize_q4_0(row); break;
        case gguf::TensorType::Q4_1: {
            auto r = dequantize_q4_1(row);
            e.primary = std::move(r.rounded);
            e.alternate = std::move(r.fused);
            break;
        }
        case gguf::TensorType::Q8_0: e.primary = dequantize_q8_0(row); break;
        case gguf::TensorType::Q6_K: e.primary = dequantize_q6_k(row); break;
        default: FAIL("a format the test has no reference for");
    }
    for (float& w : e.primary) w *= scale;
    for (float& w : e.alternate) w *= scale;
    return e;
}

// How many representable floats lie between two finite floats of one sign.
std::uint32_t ulps(float a, float b) {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return static_cast<std::uint32_t>(x > y ? x - y : y - x);
}

// Every row of the step is its token's expected row; the row after the
// step's last is untouched. With a scale of 1 every weight is the
// reference's, bit for bit (format.h); with another, within `scaled_ulps`.
void check_rows(const Uploaded& u, std::span<const std::uint32_t> ids, std::span<const float> got, float scale,
                std::uint32_t scaled_ulps = 0) {
    std::uint32_t worst = 0;
    for (std::size_t r = 0; r < ids.size(); ++r) {
        const Expected want = expected_row(u, ids[r], scale);
        std::size_t differing = 0;
        for (std::size_t i = 0; i < u.width; ++i) {
            const float g = got[r * u.width + i];
            if (scale != 1.0f && !same_weight(g, want.primary[i])) worst = std::max(worst, ulps(g, want.primary[i]));
            const bool ok = same_weight(g, want.primary[i]) ||
                            (!want.alternate.empty() && same_weight(g, want.alternate[i])) ||
                            (scale != 1.0f && ulps(g, want.primary[i]) <= scaled_ulps);
            if (!ok && differing++ < 4) {
                CAPTURE(r);
                CAPTURE(ids[r]);
                CAPTURE(i);
                CHECK(g == want.primary[i]);
            }
        }
        CHECK(differing == 0);
    }
    if (scale != 1.0f) MESSAGE("largest difference from decode-then-scale: " << worst << " ulp");
    for (std::size_t i = 0; i < u.width; ++i) {
        if (got[ids.size() * u.width + i] != 0.0f) {
            FAIL_CHECK("the row past the step was written");
            break;
        }
    }
}

}  // namespace

TEST_CASE("every row of a step is its token's row of the table, decoded, in every listed format") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const std::uint32_t ids[] = {3, 0, 6, 3, 1};   // a token twice, the first and last rows
    for (const char* fixture : {"embedding_f32", "embedding_q4_0", "embedding_q4_1", "embedding_q8_0", "embedding_q6_k"}) {
        CAPTURE(fixture);
        const Uploaded u = upload_table(instance.get(), *device, fixture, {7});
        const auto program = build(instance.get(), u, 1.0f);
        run(instance.get(), *program, ids);
        check_rows(u, ids, read_hidden(instance.get(), *device, u, std::size(ids) + 1), 1.0f);
    }
}

TEST_CASE("a table split into pieces is gathered by each piece's launch, either side of every boundary") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const std::uint32_t ids[] = {2, 3, 4, 5, 0, 6, 4};
    for (const char* fixture : {"embedding_q6_k", "embedding_q8_0"}) {
        CAPTURE(fixture);
        // Rows 0 .. 2, 3 .. 4 and 5 .. 6; the last piece in a second buffer.
        const Uploaded u = upload_table(instance.get(), *device, fixture, {3, 2, 2}, 2);
        const auto program = build(instance.get(), u, 1.0f);
        run(instance.get(), *program, ids);
        check_rows(u, ids, read_hidden(instance.get(), *device, u, std::size(ids) + 1), 1.0f);
    }
}

TEST_CASE("the scale multiplies every weight, and a decode step writes its one row") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const float scale = std::sqrt(1152.0f);   // Gemma 3 1B's: the square root of its hidden width
    const Uploaded u = upload_table(instance.get(), *device, "embedding_q4_0", {7});
    const auto program = build(instance.get(), u, scale);
    const std::uint32_t ids[] = {5};
    run(instance.get(), *program, ids);
    // Within 2 units in the last place of decode-then-scale (gather.h).
    check_rows(u, ids, read_hidden(instance.get(), *device, u, 2), scale, 2);
}

TEST_CASE("one program runs step after step, each writing its own rows") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_table(instance.get(), *device, "embedding_q8_0", {7});
    const auto program = build(instance.get(), u, 1.0f);
    const std::vector<std::vector<std::uint32_t>> steps = {{6, 5, 4}, {1}, {0, 2}};
    for (const auto& ids : steps) {
        run(instance.get(), *program, ids);
        // Only the step's rows are compared: a longer earlier step's rows
        // remain past them.
        const auto got = read_hidden(instance.get(), *device, u, ids.size());
        for (std::size_t r = 0; r < ids.size(); ++r) {
            const Expected want = expected_row(u, ids[r], 1.0f);
            for (std::size_t i = 0; i < u.width; ++i) {
                if (!same_weight(got[r * u.width + i], want.primary[i])) {
                    FAIL_CHECK("row " << r << " is not token " << ids[r] << "'s");
                    break;
                }
            }
        }
    }
}

TEST_CASE("destroying the program with a step in flight reports Cancelled, once") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Uploaded u = upload_table(instance.get(), *device, "embedding_q4_0", {7});
    auto program = build(instance.get(), u, 1.0f);
    kernels::Step step{};
    step.tokens = 1;
    step.ids[0] = 3;
    struct Ran {
        ProgramError error = ProgramError::Ok;
        int calls = 0;
        bool done = false;
    } ran;
    program->run(step,
                 [](ProgramError e, std::string_view, void* userdata) {
                     auto& r = *static_cast<Ran*>(userdata);
                     r.error = e;
                     ++r.calls;
                     r.done = true;
                 },
                 &ran);
    program.reset();
    pump_until(instance.get(), ran.done, "the cancelled step");
    CHECK(ran.calls == 1);
    CHECK(ran.error == ProgramError::Cancelled);
}
