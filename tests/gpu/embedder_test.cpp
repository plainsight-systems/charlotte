// The embedder on the GPU (runtime/embedder.h), over EmbeddingGemma 2 at Q8_0,
// fetched as test data, pinned by SHA-256 (tests/fixtures/external.json),
// uploaded and run as the harness runs it.
//
//   - Against the reference (tools/make_reference_embeddings.sh): for each
//     pinned text — a short query, a passage of a few hundred tokens, one of
//     exactly 512 tokens with its ends, text in three scripts — its tokens,
//     ends included, are llama.cpp's; its embedding is within the largest
//     difference llama.cpp's own CPU and Metal backends show on that text of
//     Metal's, component by component, and its cosine with Metal's at least
//     the CPU's with Metal's — Metal the reference, as the logits' reference
//     test takes it (reference_test.cpp). Our sums are not llama.cpp's, in
//     order or in kernels, so bits are not compared (GDSA.2); its own two
//     backends' disagreement is the tolerance, and a text outside it is a
//     fault to find, never a tolerance to widen. Not against the CPU's as
//     well: a vector near Metal's lies about the spread from the CPU's, on
//     either side of it. The check discriminates: the same model with its
//     RoPE pairing swapped falls outside on every text.
//   - The same text embedded twice gives the same bits, and an encode after
//     a refused one the bits a fresh embedder gives.
//   - Each refusal named: Busy, Text, TooManyTokens by the least count and
//     by the count an admitted text made; a step that reads back no vector
//     is Step; a lost device DeviceLost, and every later encode refused so;
//     an embedder destroyed mid-encode reports Cancelled.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/program.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/runtime/embedder.h"
#include "core/runtime/ends.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/pump.h"
#include "support/test_data.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;
using runtime::EmbedError;

namespace {

// One text's reference, as tools/reference_embeddings writes it.
struct ReferenceEmbedding {
    std::string_view name;
    std::string_view text;
    std::vector<std::uint32_t> tokens;   // with the file's ends
    std::vector<float> metal;            // unit length
    std::vector<float> cpu;
};

#include "fixtures/reference/embeddinggemma-2-q8_0.inc"

constexpr std::size_t kUploadChunk = 16u << 20;

// EmbeddingGemma 2, planned within the default budget, uploaded, with its
// encoder's launches and the embedding's range.
struct Model {
    std::string bytes;
    gguf::TensorIndex index;
    const arch::Architecture* architecture = nullptr;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    kernels::Binding embedding{};
};

std::unique_ptr<Model> upload_model(WGPUInstance instance, const gpu::Device& device) {
    auto m = std::make_unique<Model>();
    m->bytes = load_test_data("models/embeddinggemma-2-q8_0.gguf");
    gguf::MemoryByteSource source{std::as_bytes(std::span{m->bytes}), m->bytes.size()};
    REQUIRE(gguf::read_index(source, m->index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan plan;
    const std::string stop = preflight::plan_load(m->index, residency::DeviceLimits{256ull << 20, 128ull << 20, 256},
                                                  policy::LoadPolicy{}, m->description, plan);
    REQUIRE_MESSAGE(stop.empty(), stop);
    Ready ready;
    residency::Upload::begin(device, m->index, plan, m->bytes.size(), capability::find_format, {}, kUploadChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    m->upload = std::move(ready.upload);
    const auto all = std::as_bytes(std::span{m->bytes});
    for (std::size_t at = 0; at < all.size(); at += kUploadChunk) {
        Reported accepted;
        m->upload->write(at, all.subspan(at, std::min(kUploadChunk, all.size() - at)), on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.error == residency::UploadError::Ok);
    }
    REQUIRE(finish(instance, *m->upload) == residency::UploadError::Ok);
    const residency::ResidencyPlan& planned = m->upload->plan();
    std::string_view name;
    REQUIRE(m->index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    m->architecture = capability::find_architecture(name);
    const graph::GraphResult built = arch::encoder_graph(*m->architecture)(m->description, planned, m->launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);
    for (const residency::PlannedScratch& s : planned.scratch) {
        if (s.purpose == "embedding") {
            m->embedding = {s.range.buffer, s.range.offset, std::uint64_t{m->description.encoder->embedding_length} *
                                                                sizeof(float)};
        }
    }
    REQUIRE(m->embedding.size == 768 * sizeof(float));
    return m;
}

std::unique_ptr<tokenizer::Tokenizer> tokenizer_of(const Model& m) {
    gguf::MemoryByteSource source{std::as_bytes(std::span{m.bytes}), m.bytes.size()};
    std::unique_ptr<tokenizer::Tokenizer> t;
    const std::string stop = preflight::load_tokenizer(source, m.index, t);
    REQUIRE_MESSAGE(stop.empty(), stop);
    return t;
}

runtime::Ends ends_of(const Model& m, const tokenizer::Tokenizer& t) {
    runtime::Ends ends;
    const runtime::EndsResult resolved = runtime::resolve_ends(m.index, t.vocabulary(), ends);
    REQUIRE_MESSAGE(resolved.error == runtime::EndsError::Ok, resolved.subject);
    return ends;
}

// An embedder over the model, a program of its own over `launches` reading
// back `readback`.
std::unique_ptr<runtime::Embedder> embedder_of(WGPUInstance instance, const Model& m,
                                               std::vector<kernels::Launch> launches,
                                               std::optional<kernels::Binding> readback) {
    auto t = tokenizer_of(m);
    const runtime::Ends ends = ends_of(m, *t);
    return std::make_unique<runtime::Embedder>(std::move(t),
                                               build_program(instance, *m.upload, std::move(launches), readback), ends,
                                               m.upload->plan().context_offered,
                                               m.description.encoder->embedding_length);
}

std::unique_ptr<runtime::Embedder> embedder_of(WGPUInstance instance, const Model& m) {
    return embedder_of(instance, m, m.launches, m.embedding);
}

// How an encode ended, as its callback reported it.
struct Embedded {
    std::optional<runtime::EmbedResult> result;
    std::string message;
    std::vector<float> vector;
    int calls = 0;
    bool done = false;
};

void on_embedded(const runtime::EmbedResult& result, std::span<const float> embedding, void* userdata) {
    auto& e = *static_cast<Embedded*>(userdata);
    e.result = result;
    e.message = result.message;
    e.result->message = {};
    e.vector.assign(embedding.begin(), embedding.end());
    ++e.calls;
    e.done = true;
}

void pump_embedded(WGPUInstance instance, const Embedded& e) { pump_until(instance, e.done, "the encode"); }

// `text` embedded; the encode must be accepted.
Embedded embed(WGPUInstance instance, runtime::Embedder& embedder, std::string_view text) {
    Embedded e;
    const runtime::EmbedResult started = embedder.start(text, on_embedded, &e);
    REQUIRE(started.error == EmbedError::Ok);
    pump_embedded(instance, e);
    CHECK(e.calls == 1);
    return e;
}

double dot(std::span<const float> a, std::span<const float> b) {
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
    return sum;
}

double largest_difference(std::span<const float> a, std::span<const float> b) {
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
    return d;
}

}  // namespace

TEST_CASE("each pinned text's tokens are llama.cpp's, its embedding within llama.cpp's own spread, and a wrong pairing's not") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_model(instance.get(), *device);
    const auto t = tokenizer_of(*m);
    const runtime::Ends ends = ends_of(*m, *t);
    auto embedder = embedder_of(instance.get(), *m);
    CHECK(embedder->context() == 512);
    CHECK(embedder->dimensions() == 768);
    for (const ReferenceEmbedding& ref : kReferenceEmbeddings) {
        CAPTURE(ref.name);
        // The tokens, before the GPU.
        std::vector<tokenizer::TokenId> tokens;
        if (ends.bos) tokens.push_back(*ends.bos);
        REQUIRE(t->encode(ref.text, embedder->context(), tokens).error == tokenizer::EncodeError::Ok);
        if (ends.eos) tokens.push_back(*ends.eos);
        std::vector<std::uint32_t> ids;
        for (const tokenizer::TokenId id : tokens) ids.push_back(static_cast<std::uint32_t>(id));
        CHECK(ids == ref.tokens);

        const Embedded e = embed(instance.get(), *embedder, ref.text);
        REQUIRE(e.result->error == EmbedError::Ok);
        CHECK(e.result->tokens == ref.tokens.size());
        REQUIRE(e.vector.size() == 768);
        const double spread = largest_difference(ref.metal, ref.cpu);
        const double theirs = dot(ref.metal, ref.cpu);
        const double from_metal = largest_difference(e.vector, ref.metal);
        const double from_cpu = largest_difference(e.vector, ref.cpu);
        MESSAGE(ref.name << ": " << ref.tokens.size() << " tokens; llama.cpp's spread " << spread << ", cosine "
                         << theirs << "; ours from Metal " << from_metal << ", from the CPU " << from_cpu
                         << "; cosine " << dot(e.vector, ref.metal) << ", " << dot(e.vector, ref.cpu));
        CHECK(from_metal <= spread);
        CHECK(dot(e.vector, ref.metal) >= theirs);
        CHECK(std::abs(dot(e.vector, e.vector) - 1.0) < 1e-5);
    }

    // The check discriminates: RoPE pairing the wrong dimensions.
    model::ModelDescription swapped = m->description;
    swapped.rotary_pairing = model::RotaryPairing::Adjacent;
    std::vector<kernels::Launch> launches;
    REQUIRE(arch::encoder_graph(*m->architecture)(swapped, m->upload->plan(), launches).ok());
    auto wrong = embedder_of(instance.get(), *m, std::move(launches), m->embedding);
    for (const ReferenceEmbedding& ref : kReferenceEmbeddings) {
        CAPTURE(ref.name);
        const Embedded e = embed(instance.get(), *wrong, ref.text);
        REQUIRE(e.result->error == EmbedError::Ok);
        const double spread = largest_difference(ref.metal, ref.cpu);
        MESSAGE(ref.name << " with the other pairing: from Metal " << largest_difference(e.vector, ref.metal)
                         << ", cosine " << dot(e.vector, ref.metal));
        CHECK(largest_difference(e.vector, ref.metal) > spread);
        CHECK(dot(e.vector, ref.metal) < dot(ref.metal, ref.cpu));
    }
}

TEST_CASE("the same text gives the same bits: twice, and after a refused encode, as a fresh embedder") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_model(instance.get(), *device);
    const std::string_view text = kReferenceEmbeddings[0].text;
    auto embedder = embedder_of(instance.get(), *m);
    const Embedded first = embed(instance.get(), *embedder, text);
    REQUIRE(first.result->error == EmbedError::Ok);
    const Embedded second = embed(instance.get(), *embedder, text);
    CHECK(second.vector == first.vector);
    Embedded refused;
    REQUIRE(embedder->start(std::string(5000, 'a'), on_embedded, &refused).error == EmbedError::TooManyTokens);
    const Embedded after = embed(instance.get(), *embedder, text);
    CHECK(after.vector == first.vector);
    auto fresh = embedder_of(instance.get(), *m);
    CHECK(embed(instance.get(), *fresh, text).vector == first.vector);
    CHECK(refused.calls == 0);
}

TEST_CASE("the embedder refuses what it cannot take, each by name, and never calls back for it") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_model(instance.get(), *device);
    const auto t = tokenizer_of(*m);
    auto embedder = embedder_of(instance.get(), *m);
    Embedded never;

    // Not UTF-8.
    const runtime::EmbedResult invalid = embedder->start(std::string_view{"\xC3(", 2}, on_embedded, &never);
    CHECK(invalid.error == EmbedError::Text);
    CHECK(invalid.encode.error == tokenizer::EncodeError::InvalidUtf8);

    // More than the context at least: refused before any merge, by the least
    // count, ends included.
    const std::string huge(5000, 'a');
    const runtime::EmbedResult least = embedder->start(huge, on_embedded, &never);
    CHECK(least.error == EmbedError::TooManyTokens);
    CHECK(least.tokens > 512);
    CHECK(least.context == 512);

    // Admitted by its least count, but making more than the context once
    // merged: refused after the encode, with the count it made.
    const std::string over = std::string(kReferenceEmbeddings[2].text) + " And the night went on, quietly.";
    std::vector<tokenizer::TokenId> made;
    REQUIRE(t->encode(over, 4096, made).error == tokenizer::EncodeError::Ok);
    REQUIRE(made.size() + 2 > 512);
    std::vector<tokenizer::TokenId> limited;
    REQUIRE(t->encode(over, 510, limited).error == tokenizer::EncodeError::Ok);   // admitted
    const runtime::EmbedResult counted = embedder->start(over, on_embedded, &never);
    CHECK(counted.error == EmbedError::TooManyTokens);
    CHECK(counted.tokens == made.size() + 2);

    // A second while one runs.
    Embedded running;
    REQUIRE(embedder->start(kReferenceEmbeddings[0].text, on_embedded, &running).error == EmbedError::Ok);
    CHECK(embedder->start(kReferenceEmbeddings[0].text, on_embedded, &never).error == EmbedError::Busy);
    pump_embedded(instance.get(), running);
    CHECK(running.result->error == EmbedError::Ok);
    CHECK(never.calls == 0);
}

TEST_CASE("a step that reads back no vector is Step, and a destroyed embedder ends its encode Cancelled") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_model(instance.get(), *device);
    {
        auto blind = embedder_of(instance.get(), *m, m->launches, std::nullopt);
        const Embedded e = embed(instance.get(), *blind, kReferenceEmbeddings[0].text);
        CHECK(e.result->error == EmbedError::Step);
        CHECK(e.message == "the step read back no embedding");
        CHECK(e.vector.empty());
    }
    auto embedder = embedder_of(instance.get(), *m);
    Embedded e;
    REQUIRE(embedder->start(kReferenceEmbeddings[0].text, on_embedded, &e).error == EmbedError::Ok);
    embedder.reset();
    pump_embedded(instance.get(), e);
    CHECK(e.calls == 1);
    CHECK(e.result->error == EmbedError::Cancelled);
    CHECK(e.vector.empty());
}

TEST_CASE("a lost device is DeviceLost, and the embedder then refuses every encode") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_model(instance.get(), *device);
    auto embedder = embedder_of(instance.get(), *m);
    const auto status = device->status();
    wgpuDeviceDestroy(device->handle());
    pump_until(instance.get(), status->lost, "the device-lost callback");
    const Embedded e = embed(instance.get(), *embedder, kReferenceEmbeddings[0].text);
    CHECK(e.result->error == EmbedError::DeviceLost);
    CHECK(e.vector.empty());
    Embedded again;
    CHECK(embedder->start(kReferenceEmbeddings[0].text, on_embedded, &again).error == EmbedError::DeviceLost);
}
