// An image embedded on the GPU (runtime/vision.h), EmbeddingGemma 2's text
// model and its vision encoder at Q8_0, fetched as test data, pinned by
// SHA-256 (tests/fixtures/external.json), uploaded on one device as the
// harness uploads them: the text model whole, the mmproj by its header and
// its vision span alone (residency/file_ranges.h).
//
//   - Against the reference (tools/make_reference_vision.sh): each fixture
//     image of tests/fixtures/vision/scenes.h, alone and with a caption
//     around it, and four text queries — its tokens and rows llama.cpp's,
//     its embedding within the largest difference llama.cpp's own CPU and
//     Metal backends show on it of Metal's, component by component, and its
//     cosine with Metal's at least the CPU's with Metal's: the text
//     embedder's check (embedder_test.cpp), Metal the reference.
//   - The check discriminates: the encoder run with SiLU in its
//     feed-forward falls outside on every image.
//   - Each query ranks its own image first among the four scenes.
//   - The same image embedded twice gives the same bits; refusals named.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/arch/gemma4v/gemma4v.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/graph/vision.h"
#include "core/kernels/program.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/runtime/embedder.h"
#include "core/runtime/ends.h"
#include "core/runtime/vision.h"
#include "fixtures/vision/scenes.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/pump.h"
#include "support/test_data.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;
using runtime::EmbedError;
using fixtures::vision::kScenes;
using fixtures::vision::pixel;
using fixtures::vision::Scene;

namespace {

// One item's reference, as tools/reference_vision writes it.
struct VisionReference {
    std::string_view name;
    std::string_view scene;   // empty for a text alone
    std::string_view before;
    std::string_view after;
    std::vector<std::uint32_t> tokens_before;   // the ends and <|image> among them
    std::uint32_t rows;
    std::vector<std::uint32_t> tokens_after;
    std::vector<float> metal;
    std::vector<float> cpu;
};

#include "fixtures/reference/vision_embeddings.inc"

constexpr std::size_t kChunk16 = 16u << 20;
constexpr residency::DeviceLimits kLimits{256ull << 20, 128ull << 20, 256};
constexpr std::uint64_t kMmprojSize = 554'821'024;
constexpr residency::FileRange kVisionRanges[] = {{0, 58'656}, {328'492'448, kMmprojSize}};
constexpr media::ImageBudget kBudget{70, 280};

struct Text {
    std::string bytes;
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    kernels::Binding embedding{};
    residency::BufferRange hidden{};
};

std::unique_ptr<Text> upload_text(WGPUInstance instance, const gpu::Device& device) {
    auto m = std::make_unique<Text>();
    m->bytes = load_test_data("models/embeddinggemma-2-q8_0.gguf");
    gguf::MemoryByteSource source{std::as_bytes(std::span{m->bytes}), m->bytes.size()};
    REQUIRE(gguf::read_index(source, m->index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan plan;
    const std::string stop = preflight::plan_load(m->index, kLimits, policy::LoadPolicy{}, m->description, plan);
    REQUIRE_MESSAGE(stop.empty(), stop);
    Ready ready;
    residency::Upload::begin(device, m->index, plan, m->bytes.size(), capability::find_format, {}, kChunk16, on_ready,
                             &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    m->upload = std::move(ready.upload);
    const auto all = std::as_bytes(std::span{m->bytes});
    for (std::size_t at = 0; at < all.size(); at += kChunk16) {
        Reported accepted;
        m->upload->write(at, all.subspan(at, std::min(kChunk16, all.size() - at)), on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.error == residency::UploadError::Ok);
    }
    REQUIRE(finish(instance, *m->upload) == residency::UploadError::Ok);
    const residency::ResidencyPlan& planned = m->upload->plan();
    REQUIRE(arch::encoder_graph(*capability::find_architecture("gemma-embedding2"))(m->description, planned,
                                                                                   m->launches)
                .ok());
    for (const residency::PlannedScratch& s : planned.scratch) {
        if (s.purpose == "embedding") m->embedding = {s.range.buffer, s.range.offset, 768 * sizeof(float)};
        if (s.purpose == "hidden") m->hidden = s.range;
    }
    return m;
}

struct VisionModel {
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    residency::BufferRange pixels{}, soft_tokens{};
};

// The mmproj, read and written by its two ranges alone.
std::unique_ptr<VisionModel> upload_vision(WGPUInstance instance, const gpu::Device& device,
                                           model::FeedForwardActivation activation) {
    auto v = std::make_unique<VisionModel>();
    const std::string path = std::string(BLLM_TEST_DATA_DIR) + "/models/mmproj-embeddinggemma-2-q8_0.gguf";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing test data: " << path << " (run make test-data)");
    std::string header(kVisionRanges[0].end, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    gguf::MemoryByteSource source{std::as_bytes(std::span{header}), kMmprojSize};
    REQUIRE(gguf::read_index(source, v->index).error == gguf::ReadError::Ok);
    REQUIRE(arch::kGemma4Vision.describe(v->index, v->description).ok());
    v->description.activation = activation;
    residency::ResidencyPlan plan;
    REQUIRE(residency::plan_residency(v->index, v->description, kLimits, policy::LoadPolicy{},
                                      graph::vision_working_set(v->description, kBudget.most * 9), plan, kVisionRanges)
                .ok());
    Ready ready;
    residency::Upload::begin(device, v->index, plan, kMmprojSize, capability::find_format, {}, kChunk16, on_ready,
                             &ready, kVisionRanges);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    v->upload = std::move(ready.upload);
    std::vector<char> chunk(kChunk16);
    for (const residency::FileRange& r : kVisionRanges) {
        for (std::uint64_t at = r.begin; at < r.end; at += kChunk16) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk16, r.end - at));
            in.seekg(static_cast<std::streamoff>(at));
            in.read(chunk.data(), static_cast<std::streamsize>(n));
            REQUIRE(in.gcount() == static_cast<std::streamsize>(n));
            Reported accepted;
            v->upload->write(at, std::as_bytes(std::span(chunk).first(n)), on_reported, &accepted);
            pump_until(instance, accepted.done, "a chunk's acceptance");
            REQUIRE(accepted.error == residency::UploadError::Ok);
        }
    }
    REQUIRE(finish(instance, *v->upload) == residency::UploadError::Ok);
    REQUIRE(arch::kGemma4Vision.graph(v->description, v->upload->plan(), v->launches).ok());
    for (const residency::PlannedScratch& s : v->upload->plan().scratch) {
        if (s.purpose == "pixels") v->pixels = s.range;
        if (s.purpose == "soft_tokens") v->soft_tokens = s.range;
    }
    return v;
}

std::unique_ptr<tokenizer::Tokenizer> tokenizer_of(const Text& m) {
    gguf::MemoryByteSource source{std::as_bytes(std::span{m.bytes}), m.bytes.size()};
    std::unique_ptr<tokenizer::Tokenizer> t;
    REQUIRE(preflight::load_tokenizer(source, m.index, t).empty());
    return t;
}

runtime::MediaTokens image_tokens_of(const tokenizer::Tokenizer& t) {
    const auto placeholder = t.vocabulary().find("<|image|>");
    const auto begin = t.vocabulary().find("<|image>");
    const auto end = t.vocabulary().find("<image|>");
    REQUIRE(placeholder);
    REQUIRE(begin);
    REQUIRE(end);
    return {*placeholder, *begin, *end};
}

// The embedder's image stage.
runtime::Vision& vision_of(runtime::Embedder& embedder) {
    auto* vision = dynamic_cast<runtime::Vision*>(embedder.stage(arch::MediaKind::Image));
    REQUIRE(vision != nullptr);
    return *vision;
}

// The text embedder with the vision encoder attached.
std::unique_ptr<runtime::Embedder> embedder_of(WGPUInstance instance, const Text& m, const VisionModel& v) {
    auto t = tokenizer_of(m);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(m.index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    const runtime::MediaTokens tokens = image_tokens_of(*t);
    auto embedder = std::make_unique<runtime::Embedder>(std::move(t), build_program(instance, *m.upload, m.launches,
                                                                                    m.embedding),
                                                        ends, m.upload->plan().context_offered, 768);
    auto link = std::make_unique<kernels::ProgramLink>(
        *v.upload, kernels::Binding{v.soft_tokens.buffer, v.soft_tokens.offset, v.soft_tokens.length}, *m.upload,
        kernels::Binding{m.hidden.buffer, m.hidden.offset, m.hidden.length});
    embedder->attach(arch::MediaKind::Image, std::make_unique<runtime::Vision>(build_program(instance, *v.upload, v.launches), std::move(link),
                                                       kernels::Binding{v.pixels.buffer, v.pixels.offset,
                                                                        v.pixels.length},
                                                       kBudget, v.description.vision->patch_size,
                                                       v.description.vision->pooling),
                     tokens, m.description.embedding_width);
    return embedder;
}

const Scene& scene_named(std::string_view name) {
    for (const Scene& s : kScenes) {
        if (s.name == name) return s;
    }
    FAIL("no scene " << name);
    return kScenes[0];
}

// The scene prepared in the vision encoder's slot, in bands of 37 rows.
void prepare(runtime::Vision& vision, const Scene& scene) {
    REQUIRE(vision.begin(scene.width, scene.height) == media::ImageError::Ok);
    for (std::uint32_t row = 0; row < scene.height; row += 37) {
        const std::uint32_t rows = std::min(37u, scene.height - row);
        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(scene.width) * rows * 4, 255);
        for (std::uint32_t y = 0; y < rows; ++y) {
            for (std::uint32_t x = 0; x < scene.width; ++x) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    rgba[(static_cast<std::size_t>(y) * scene.width + x) * 4 + c] = pixel(scene, x, row + y, c);
                }
            }
        }
        REQUIRE(vision.rows(row, rgba) == media::ImageError::Ok);
    }
}

struct Embedded {
    std::optional<runtime::EmbedResult> result;
    std::string message;
    std::vector<float> vector;
    bool done = false;
};

void on_embedded(const runtime::EmbedResult& result, std::span<const float> embedding, void* userdata) {
    auto& e = *static_cast<Embedded*>(userdata);
    e.result = result;
    e.message = result.message;
    e.result->message = {};
    e.vector.assign(embedding.begin(), embedding.end());
    e.done = true;
}

// The item embedded: its scene prepared and placed, or its text alone.
Embedded embed(WGPUInstance instance, runtime::Embedder& embedder, const VisionReference& r) {
    const bool image = !r.scene.empty();
    if (image) {
        prepare(vision_of(embedder), scene_named(r.scene));
        REQUIRE(vision_of(embedder).ready_rows() == r.rows);
    }
    const std::string text = std::string(r.before) + (image ? "<|image|>" : "") + std::string(r.after);
    Embedded e;
    const runtime::EmbedResult started =
        embedder.start(text, image ? std::optional{arch::MediaKind::Image} : std::nullopt, on_embedded, &e);
    REQUIRE(started.error == EmbedError::Ok);
    CHECK(started.tokens == r.tokens_before.size() + r.rows + r.tokens_after.size());
    pump_until(instance, e.done, "the encode");
    REQUIRE_MESSAGE(e.result->error == EmbedError::Ok, e.message);
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

// Whether `got` is within llama.cpp's own spread of Metal's.
bool within_spread(const VisionReference& r, std::span<const float> got) {
    return largest_difference(got, r.metal) <= largest_difference(r.cpu, r.metal) &&
           dot(got, r.metal) >= dot(r.cpu, r.metal);
}

}  // namespace

TEST_CASE("each image and query is embedded within llama.cpp's own spread, its tokens and rows llama.cpp's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision = upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh);
    const auto embedder = embedder_of(instance.get(), *text, *vision);

    // The tokens, before the GPU: the text with its placeholder, encoded and
    // wrapped as the embedder does, the placeholder replaced.
    const auto t = tokenizer_of(*text);
    const runtime::MediaTokens image = image_tokens_of(*t);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(text->index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    for (const VisionReference& r : kVisionReferences) {
        CAPTURE(r.name);
        std::vector<tokenizer::TokenId> tokens;
        if (ends.bos) tokens.push_back(*ends.bos);
        const std::string with = std::string(r.before) + (r.scene.empty() ? "" : "<|image|>") + std::string(r.after);
        REQUIRE(t->encode(with, 512, tokens).error == tokenizer::EncodeError::Ok);
        if (ends.eos) tokens.push_back(*ends.eos);
        std::vector<std::uint32_t> ids;
        for (const tokenizer::TokenId id : tokens) ids.push_back(static_cast<std::uint32_t>(id));
        const auto id_of = [](tokenizer::TokenId id) { return static_cast<std::uint32_t>(id); };
        const std::vector<std::uint32_t> before(r.tokens_before.begin(), r.tokens_before.end());
        const std::vector<std::uint32_t> after(r.tokens_after.begin(), r.tokens_after.end());
        if (!r.scene.empty()) {
            const auto at = std::find(ids.begin(), ids.end(), id_of(image.placeholder));
            REQUIRE(at != ids.end());
            CHECK(std::vector<std::uint32_t>(ids.begin(), at) == std::vector<std::uint32_t>(before.begin(), before.end() - 1));
            CHECK(before.back() == id_of(image.begin));
            CHECK(after.front() == id_of(image.end));
            CHECK(std::vector<std::uint32_t>(at + 1, ids.end()) == std::vector<std::uint32_t>(after.begin() + 1, after.end()));
        } else {
            std::vector<std::uint32_t> whole = before;
            whole.insert(whole.end(), after.begin(), after.end());   // the end token
            CHECK(ids == whole);
        }
    }

    for (const VisionReference& r : kVisionReferences) {
        CAPTURE(r.name);
        const Embedded e = embed(instance.get(), *embedder, r);
        MESSAGE(r.name << ": largest difference " << largest_difference(e.vector, r.metal) << " against llama.cpp's "
                       << largest_difference(r.cpu, r.metal) << "; cosine " << dot(e.vector, r.metal) << " against "
                       << dot(r.cpu, r.metal));
        CHECK(within_spread(r, e.vector));
    }
}

TEST_CASE("the check discriminates: a feed-forward run with SiLU falls outside on every image") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision = upload_vision(instance.get(), *device, model::FeedForwardActivation::SiLU);
    const auto embedder = embedder_of(instance.get(), *text, *vision);
    for (const VisionReference& r : kVisionReferences) {
        if (r.scene.empty()) continue;
        CAPTURE(r.name);
        CHECK_FALSE(within_spread(r, embed(instance.get(), *embedder, r).vector));
    }
}

TEST_CASE("each query ranks its own image first; an image embedded twice gives the same bits; refusals named") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision = upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh);
    const auto embedder = embedder_of(instance.get(), *text, *vision);

    std::vector<std::pair<std::string_view, std::vector<float>>> images;
    for (const VisionReference& r : kVisionReferences) {
        if (r.scene.empty() || r.name != r.scene || r.name.starts_with("noise")) continue;
        images.emplace_back(r.name, embed(instance.get(), *embedder, r).vector);
    }
    REQUIRE(images.size() == 4);
    for (const VisionReference& r : kVisionReferences) {
        if (!r.scene.empty()) continue;
        CAPTURE(r.name);
        const auto query = embed(instance.get(), *embedder, r).vector;
        const auto best = std::max_element(images.begin(), images.end(), [&](const auto& a, const auto& b) {
            return dot(query, a.second) < dot(query, b.second);
        });
        CHECK(std::string("query_") + std::string(best->first) == r.name);
    }

    // The same image twice: the same bits.
    const VisionReference& circle = kVisionReferences[0];
    CHECK(embed(instance.get(), *embedder, circle).vector == embed(instance.get(), *embedder, circle).vector);

    // Refusals: no image prepared; a placeholder without an image, or twice.
    Embedded unused;
    constexpr auto kImage = std::optional{arch::MediaKind::Image};
    CHECK(embedder->start("<|image|>", kImage, on_embedded, &unused).error == EmbedError::MediaNotReady);
    CHECK(embedder->start("a <|image|> here", std::nullopt, on_embedded, &unused).error == EmbedError::Placeholder);
    // A kind with no stage attached has nothing prepared.
    CHECK(embedder->start("<|audio|>", std::optional{arch::MediaKind::Audio}, on_embedded, &unused).error ==
          EmbedError::MediaNotReady);
    prepare(vision_of(*embedder), scene_named("red_circle"));
    CHECK(embedder->start("<|image|> and <|image|>", kImage, on_embedded, &unused).error == EmbedError::Placeholder);
    CHECK(embedder->start("no placeholder", kImage, on_embedded, &unused).error == EmbedError::Placeholder);
    CHECK_FALSE(unused.done);
}
