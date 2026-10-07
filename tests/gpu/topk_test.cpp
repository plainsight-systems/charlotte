// Top-k selection on the GPU (kernels/topk/topk.h): the 64 kept, bit for bit
// and token for token, against a CPU sort under the same order.

#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "core/kernels/program.h"
#include "core/kernels/topk/topk.h"
#include "core/residency/upload.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

constexpr std::uint32_t kLargest = 262'144;   // Gemma 3's vocabulary

struct Buffers {
    Model model;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange logits{}, a{}, b{}, candidates{};
};

Buffers make_buffers(WGPUInstance instance, const gpu::Device& device) {
    Buffers u;
    u.model.bytes = load_gguf_fixture("rope_rows");   // any file: no weight is read
    gguf::MemoryByteSource source{u.model.bytes};
    REQUIRE(gguf::read_index(source, u.model.index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan& plan = u.model.plan;
    plan.buffers.push_back({residency::Pool::Weights, 0});
    const auto alone = [&](std::uint64_t bytes) {
        plan.buffers.push_back({residency::Pool::Scratch, bytes});
        return residency::BufferRange{static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, bytes};
    };
    const std::uint64_t partials = (kLargest + 1023) / 1024 * 64 * 8;
    // The logits in a buffer the test writes from the CPU: a weight pool's,
    // which allows it, where a working buffer's does not.
    plan.buffers.push_back({residency::Pool::Weights, std::uint64_t{kLargest} * 4});
    u.logits = {static_cast<residency::BufferIndex>(plan.buffers.size() - 1), 0, std::uint64_t{kLargest} * 4};
    u.a = alone(partials);
    u.b = alone(partials);
    u.candidates = alone(64 * 8);
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

// The selection's order, as the kernel keys it.
std::uint32_t key_of(float f) {
    std::uint32_t b = std::bit_cast<std::uint32_t>(f);
    if (b == 0x80000000u) b = 0;
    if (std::isnan(f)) return 0xFFFFFFFFu;
    return (b & 0x80000000u) != 0 ? ~b : (b | 0x80000000u);
}

struct Pair {
    std::uint32_t bits;
    std::uint32_t token;
    bool operator==(const Pair&) const = default;
};

// The first 64 of `logits` by the larger logit, then the lower token; each
// as its logit's bits, −0 as +0 and a NaN as the canonical quiet NaN.
std::vector<Pair> reference(const std::vector<float>& logits) {
    std::vector<std::uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](std::uint32_t i, std::uint32_t j) {
        const std::uint32_t a = key_of(logits[i]), b = key_of(logits[j]);
        return a > b || (a == b && i < j);
    });
    std::vector<Pair> out;
    for (std::size_t k = 0; k < 64; ++k) {
        const float f = logits[order[k]];
        std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
        if (bits == 0x80000000u) bits = 0;
        if (std::isnan(f)) bits = 0x7FC00000u;
        out.push_back({bits, order[k]});
    }
    return out;
}

std::vector<Pair> select(WGPUInstance instance, const gpu::Device& device, const Buffers& u,
                         const std::vector<float>& logits) {
    wgpuQueueWriteBuffer(device.queue(), u.upload->buffer(u.logits.buffer), 0, logits.data(), logits.size() * 4);
    const auto vocabulary = static_cast<std::uint32_t>(logits.size());
    const auto program = build_program(
        instance, *u.upload,
        kernels::topk_launches({u.logits.buffer, 0, std::uint64_t{vocabulary} * 4}, vocabulary, u.a, u.b, u.candidates));
    run_step(instance, *program, 1);
    const auto words = read_floats(instance, device, u.upload->buffer(u.candidates.buffer), 0, 128);
    std::vector<Pair> out;
    for (std::size_t k = 0; k < 64; ++k) {
        out.push_back({std::bit_cast<std::uint32_t>(words[2 * k]), std::bit_cast<std::uint32_t>(words[2 * k + 1])});
    }
    return out;
}

std::vector<float> random_logits(std::uint32_t count, std::uint32_t seed) {
    std::vector<float> out(count);
    std::uint32_t state = seed;
    for (float& f : out) {
        state = state * 1664525u + 1013904223u;
        f = static_cast<float>((state >> 8) % 2000001u) / 100000.0f - 10.0f;
    }
    return out;
}

void check_selects(WGPUInstance instance, const gpu::Device& device, const Buffers& u,
                   const std::vector<float>& logits) {
    const auto got = select(instance, device, u, logits);
    const auto want = reference(logits);
    for (std::size_t k = 0; k < 64; ++k) {
        CAPTURE(k);
        CHECK(got[k].token == want[k].token);
        CHECK(got[k].bits == want[k].bits);
    }
}

}  // namespace

TEST_CASE("selection keeps the first 64 of each listed vocabulary and of a part-full tile") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers u = make_buffers(instance.get(), *device);
    for (const std::uint32_t count : {151'936u, 128'256u, 262'144u, 70'001u, 1'025u, 64u}) {
        CAPTURE(count);
        check_selects(instance.get(), *device, u, random_logits(count, count));
    }
}

TEST_CASE("selection breaks ties by the lower token, treats the zeros as equal, and puts a NaN first") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Buffers u = make_buffers(instance.get(), *device);
    // Five values over 5,000 tokens: the 64th place falls among equals.
    std::vector<float> ties(5000);
    for (std::size_t i = 0; i < ties.size(); ++i) ties[i] = static_cast<float>((i * 7919) % 5);
    check_selects(instance.get(), *device, u, ties);
    // One value everywhere: the first 64 tokens.
    check_selects(instance.get(), *device, u, std::vector<float>(3000, 1.5f));
    // −0 and +0 at the top, equal: the lower token first, whichever zero.
    std::vector<float> zeros(4096, -std::numeric_limits<float>::infinity());
    zeros[3000] = 0.0f;
    zeros[17] = -0.0f;
    check_selects(instance.get(), *device, u, zeros);
    // A NaN with its sign set reaches the top.
    auto nan = random_logits(20'000, 7);
    nan[12'345] = -std::numeric_limits<float>::quiet_NaN();
    const auto got = select(instance.get(), *device, u, nan);
    CHECK(got[0].token == 12'345);
    CHECK(got[0].bits == 0x7FC00000u);
}
