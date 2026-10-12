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
//   - An encode delivers only once every step has reported: the image's
//     step held until after the text step has reported, then released Ok,
//     gives the bits it gives unheld; released failed, no vector.

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

#include "core/arch/gemma4v/gemma4v.h"
#include "core/kernels/program.h"
#include "core/runtime/embedder.h"
#include "core/runtime/ends.h"
#include "core/runtime/vision.h"
#include "fixtures/vision/scenes.h"
#include "support/acquire.h"
#include "support/companions.h"
#include "support/program.h"
#include "support/pump.h"

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

constexpr media::ImageBudget kBudget{70, 280};

runtime::MediaTokens image_tokens_of(const tokenizer::Tokenizer& t) {
    return media_tokens_of(t, arch::kGemma4Vision.tokens);
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
    embedder->attach(arch::MediaKind::Image,
                     std::make_unique<runtime::Vision>(build_program(instance, *v.upload, v.launches),
                                                       link_of(m, *v.upload, v.soft_tokens), binding_of(v.pixels),
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

// Test-only: the real image stage behind the embedder's interface, its
// step's report held until the test releases it — so the text step reports
// first, the order two programs' reports may take on any queue.
class HeldVision final : public runtime::MediaStage {
public:
    explicit HeldVision(std::unique_ptr<runtime::Vision> vision) : vision_(std::move(vision)) {}

    runtime::Vision& vision() noexcept { return *vision_; }

    [[nodiscard]] std::uint32_t ready_rows() const noexcept override { return vision_->ready_rows(); }

    void run(kernels::StepCallback done, void* userdata) override {
        done_ = done;
        userdata_ = userdata;
        reported_ = false;
        vision_->run(on_vision, this);
    }

    [[nodiscard]] const kernels::ProgramLink& link() const noexcept override { return vision_->link(); }

    // Whether the image's step has reported, held.
    [[nodiscard]] const bool& reported() const noexcept { return reported_; }

    // The held report passed on, or `failure` in its place.
    void release(kernels::ProgramError failure = kernels::ProgramError::Ok) {
        REQUIRE(reported_);
        const kernels::ProgramError error = failure != kernels::ProgramError::Ok ? failure : error_;
        done_(error, failure != kernels::ProgramError::Ok ? "held and failed" : std::string_view(message_), {},
              userdata_);
    }

private:
    static void on_vision(kernels::ProgramError error, std::string_view message, std::span<const std::byte>,
                          void* userdata) {
        auto& self = *static_cast<HeldVision*>(userdata);
        self.error_ = error;
        self.message_.assign(message);
        self.reported_ = true;
    }

    std::unique_ptr<runtime::Vision> vision_;
    kernels::StepCallback done_ = nullptr;
    void* userdata_ = nullptr;
    kernels::ProgramError error_ = kernels::ProgramError::Ok;
    std::string message_;
    bool reported_ = false;
};

}  // namespace

TEST_CASE("each image and query is embedded within llama.cpp's own spread, its tokens and rows llama.cpp's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
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
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::SiLU, kBudget.most * 9);
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
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
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

TEST_CASE("an encode delivers once every step has reported: the image's report held past the text step's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const VisionReference& circle = kVisionReferences[0];
    REQUIRE(!circle.scene.empty());
    const std::vector<float> unheld = embed(instance.get(), *embedder_of(instance.get(), *text, *vision), circle).vector;

    auto t = tokenizer_of(*text);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(text->index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    const runtime::MediaTokens tokens = image_tokens_of(*t);
    runtime::Embedder embedder(std::move(t), build_program(instance.get(), *text->upload, text->launches, text->embedding),
                               ends, text->upload->plan().context_offered, 768);
    auto owned = std::make_unique<HeldVision>(std::make_unique<runtime::Vision>(
        build_program(instance.get(), *vision->upload, vision->launches), link_of(*text, *vision->upload, vision->soft_tokens),
        binding_of(vision->pixels), kBudget, vision->description.vision->patch_size,
        vision->description.vision->pooling));
    HeldVision& held = *owned;
    embedder.attach(arch::MediaKind::Image, std::move(owned), tokens, text->description.embedding_width);
    // A step of the text model's own on a program of its own, submitted
    // after the encode's: once it has reported, so has the encode's text step.
    const auto probe = build_program(instance.get(), *text->upload, text->launches, text->embedding);

    const std::string with = std::string(circle.before) + "<|image|>" + std::string(circle.after);
    for (const kernels::ProgramError failure : {kernels::ProgramError::Ok, kernels::ProgramError::Step}) {
        CAPTURE(static_cast<int>(failure));
        prepare(held.vision(), scene_named(circle.scene));
        Embedded e;
        REQUIRE(embedder.start(with, std::optional{arch::MediaKind::Image}, on_embedded, &e).error == EmbedError::Ok);
        pump_until(instance.get(), held.reported(), "the image's step");
        run_step(instance.get(), *probe, 4);
        CHECK_FALSE(e.done);   // the text step has reported; the image's is held
        held.release(failure);
        pump_until(instance.get(), e.done, "the encode");
        if (failure == kernels::ProgramError::Ok) {
            REQUIRE_MESSAGE(e.result->error == EmbedError::Ok, e.message);
            CHECK(e.vector == unheld);
        } else {
            CHECK(e.result->error == EmbedError::Step);
            CHECK(e.message == "held and failed");
            CHECK(e.vector.empty());
        }
    }
}
