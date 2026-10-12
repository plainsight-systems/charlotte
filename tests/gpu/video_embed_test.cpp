// A video's clip embedded on the GPU (runtime/video.h): EmbeddingGemma 2's
// text model and its vision encoder at Q8_0, as vision_embed_test.cpp loads
// them, the image stage and the video stage attached over one program.
//
//   - A clip of four computed frames (tests/fixtures/video/scenes.h) is
//     four segments, each `<|image>`, 120 rows and `<image|>`.
//   - The same clip embedded twice gives the same bits, and the same again
//     when the next clip is prepared while it runs, in the other set.
//   - An image's encode after a clip's gives the bits a fresh embedder's
//     does, and a clip's after an image's.
//   - Each refusal named.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/gemma4v/gemma4v.h"
#include "core/kernels/program.h"
#include "core/media/video.h"
#include "core/runtime/embedder.h"
#include "core/runtime/ends.h"
#include "core/runtime/video.h"
#include "core/runtime/vision.h"
#include "fixtures/video/scenes.h"
#include "fixtures/vision/scenes.h"
#include "support/acquire.h"
#include "support/companions.h"
#include "support/program.h"
#include "support/pump.h"

using namespace bllm;
using namespace bllm::testing;
using runtime::EmbedError;
using runtime::VideoError;

namespace {

constexpr media::ImageBudget kBudget{70, 280};
constexpr auto kVideo = std::optional{arch::MediaKind::Video};
constexpr auto kImage = std::optional{arch::MediaKind::Image};

// The text embedder with the image and video stages attached, the vision
// program shared between them.
std::unique_ptr<runtime::Embedder> embedder_of(WGPUInstance instance, const Text& m, const VisionModel& v) {
    auto t = tokenizer_of(m);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(m.index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    const runtime::MediaTokens image = media_tokens_of(*t, arch::kGemma4Vision.tokens);
    const runtime::MediaTokens video = media_tokens_of(*t, arch::kGemma4Vision.video_tokens);
    auto embedder = std::make_unique<runtime::Embedder>(
        std::move(t), build_program(instance, *m.upload, m.launches, m.embedding), ends,
        m.upload->plan().context_offered, 768);
    const std::shared_ptr<kernels::Program> program = build_program(instance, *v.upload, v.launches);
    const std::uint32_t patch = v.description.vision->patch_size;
    const std::uint32_t pooling = v.description.vision->pooling;
    embedder->attach(arch::MediaKind::Image,
                     std::make_unique<runtime::Vision>(program, link_of(m, *v.upload, v.soft_tokens),
                                                       binding_of(v.pixels), kBudget, patch, pooling),
                     image, m.description.embedding_width);
    embedder->attach(arch::MediaKind::Video,
                     std::make_unique<runtime::Video>(program, link_of(m, *v.upload, v.soft_tokens),
                                                      binding_of(v.pixels), patch, pooling, embedder->room()),
                     video, m.description.embedding_width);
    return embedder;
}

runtime::Video& video_of(runtime::Embedder& embedder) {
    auto* video = dynamic_cast<runtime::Video*>(embedder.stage(arch::MediaKind::Video));
    REQUIRE(video != nullptr);
    return *video;
}

runtime::Vision& vision_of(runtime::Embedder& embedder) {
    auto* vision = dynamic_cast<runtime::Vision*>(embedder.stage(arch::MediaKind::Image));
    REQUIRE(vision != nullptr);
    return *vision;
}

const fixtures::video::Scene& scene_named(std::string_view name) {
    for (const auto& s : fixtures::video::kScenes) {
        if (s.name == name) return s;
    }
    FAIL("no scene " << name);
    return fixtures::video::kScenes[0];
}

// Frame `t` of `scene`, RGBA, rows [first, first + rows).
std::vector<std::uint8_t> band_of(const fixtures::video::Scene& scene, std::uint32_t t, std::uint32_t first,
                                  std::uint32_t rows) {
    std::vector<std::uint8_t> rgba(std::size_t{scene.width} * rows * 4, 255);
    for (std::uint32_t y = 0; y < rows; ++y) {
        for (std::uint32_t x = 0; x < scene.width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                rgba[(std::size_t{y} * scene.width + x) * 4 + c] = fixtures::video::pixel(scene, t, x, first + y, c);
            }
        }
    }
    return rgba;
}

// The scene's frames prepared as one clip, in bands of 97 rows.
void prepare(runtime::Video& video, const fixtures::video::Scene& scene) {
    REQUIRE(video.begin(scene.width, scene.height, scene.frames).error == VideoError::Ok);
    for (std::uint32_t t = 0; t < scene.frames; ++t) {
        for (std::uint32_t row = 0; row < scene.height; row += 97) {
            const std::uint32_t rows = std::min(97u, scene.height - row);
            REQUIRE(video.rows(t, row, band_of(scene, t, row, rows)).error == VideoError::Ok);
        }
        CHECK(video.ready_segments().empty() == (t + 1 < scene.frames));
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

// `text` embedded with the prepared medium of `kind`; `during`, if given,
// called once the encode is submitted.
template <typename During>
Embedded embed(WGPUInstance instance, runtime::Embedder& embedder, std::string_view text,
               std::optional<arch::MediaKind> kind, std::uint64_t& tokens, During during) {
    Embedded e;
    const runtime::EmbedResult started = embedder.start(text, kind, on_embedded, &e);
    REQUIRE(started.error == EmbedError::Ok);
    tokens = started.tokens;
    during();
    pump_until(instance, e.done, "the encode");
    REQUIRE_MESSAGE(e.result->error == EmbedError::Ok, e.message);
    return e;
}

Embedded embed(WGPUInstance instance, runtime::Embedder& embedder, std::string_view text,
               std::optional<arch::MediaKind> kind) {
    std::uint64_t tokens = 0;
    return embed(instance, embedder, text, kind, tokens, [] {});
}

}  // namespace

TEST_CASE("a clip is four frames, each between its own tokens; its bits repeat, the next clip prepared alongside") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *vision);
    runtime::Video& video = video_of(*embedder);

    const auto& ball = scene_named("red_ball");
    runtime::VideoPlan plan{};
    REQUIRE(video.plan(ball.width, ball.height, ball.frames, plan).error == VideoError::Ok);
    CHECK(plan.frames == 4);
    CHECK(plan.grid.width == 720);
    CHECK(plan.grid.height == 384);
    CHECK(plan.grid.tokens == 120);
    CHECK(plan.clip_frames == 4);

    prepare(video, ball);
    REQUIRE(video.ready_segments().size() == 4);
    for (const std::uint32_t rows : video.ready_segments()) CHECK(rows == 120);
    std::uint64_t tokens = 0;
    const Embedded first = embed(instance.get(), *embedder, "<|video|>", kVideo, tokens, [] {});
    // The ends and four frames of 120 rows, each between its two tokens.
    CHECK(tokens == 2 + 4 * (120 + 2));
    CHECK(video.ready_segments().empty());

    // Again, with the stripes prepared while it runs, in the other set; then
    // the stripes, which that preparation left ready.
    prepare(video, ball);
    const Embedded again =
        embed(instance.get(), *embedder, "<|video|>", kVideo, tokens, [&] { prepare(video, scene_named("green_stripes")); });
    CHECK(again.vector == first.vector);
    REQUIRE(video.ready_segments().size() == 4);
    const Embedded stripes = embed(instance.get(), *embedder, "<|video|>", kVideo);
    CHECK(stripes.vector != first.vector);
    prepare(video, scene_named("green_stripes"));
    CHECK(embed(instance.get(), *embedder, "<|video|>", kVideo).vector == stripes.vector);
}

TEST_CASE("an image's encode around a clip's gives its own bits, and a clip's around an image's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *vision);
    const auto& circle = fixtures::vision::kScenes[0];
    const auto prepare_image = [&](runtime::Embedder& e) {
        runtime::Vision& v = vision_of(e);
        REQUIRE(v.begin(circle.width, circle.height) == media::ImageError::Ok);
        std::vector<std::uint8_t> rgba(std::size_t{circle.width} * circle.height * 4, 255);
        for (std::uint32_t y = 0; y < circle.height; ++y) {
            for (std::uint32_t x = 0; x < circle.width; ++x) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    rgba[(std::size_t{y} * circle.width + x) * 4 + c] = fixtures::vision::pixel(circle, x, y, c);
                }
            }
        }
        REQUIRE(v.rows(0, rgba) == media::ImageError::Ok);
    };
    const auto& line = scene_named("black_line");

    const auto fresh = embedder_of(instance.get(), *text, *vision);
    prepare_image(*fresh);
    const auto image_alone = embed(instance.get(), *fresh, "<|image|>", kImage).vector;
    prepare(video_of(*fresh), line);
    const auto clip_alone = embed(instance.get(), *fresh, "<|video|>", kVideo).vector;

    prepare(video_of(*embedder), line);
    CHECK(embed(instance.get(), *embedder, "<|video|>", kVideo).vector == clip_alone);
    prepare_image(*embedder);
    CHECK(embed(instance.get(), *embedder, "<|image|>", kImage).vector == image_alone);
    prepare(video_of(*embedder), line);
    CHECK(embed(instance.get(), *embedder, "<|video|>", kVideo).vector == clip_alone);
}

TEST_CASE("a video's refusals are named") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *vision);
    runtime::Video& video = video_of(*embedder);

    runtime::VideoPlan plan{};
    CHECK(video.plan(0, 720, 4.0, plan).image == media::ImageError::Empty);
    CHECK(video.plan(media::kMaxSide + 1, 720, 4.0, plan).image == media::ImageError::TooLarge);
    CHECK(video.plan(1280, 720, 0.0, plan).error == VideoError::NoFrames);
    CHECK(video.plan(640, 480, 300.0, plan).error == VideoError::Ok);
    CHECK(plan.frames == 300);
    CHECK(plan.clip_frames == 3);   // a 4:3 frame's 130 tokens

    CHECK(video.begin(1280, 720, 0).error == VideoError::TooManyFrames);
    CHECK(video.begin(1280, 720, 5).error == VideoError::TooManyFrames);
    CHECK(video.begin(640, 480, 4).error == VideoError::TooManyFrames);
    CHECK(video.begin(0, 720, 1).image == media::ImageError::Empty);
    REQUIRE(video.begin(1280, 720, 2).error == VideoError::Ok);
    const auto& ball = scene_named("red_ball");
    CHECK(video.rows(2, 0, band_of(ball, 0, 0, 1)).error == VideoError::BadFrame);
    CHECK(video.rows(0, 5, band_of(ball, 0, 5, 1)).image == media::ImageError::BadBand);

    Embedded unused;
    CHECK(embedder->start("<|video|>", kVideo, on_embedded, &unused).error == EmbedError::MediaNotReady);
    CHECK(embedder->start("a <|video|> here", std::nullopt, on_embedded, &unused).error == EmbedError::Placeholder);
    prepare(video, ball);
    CHECK(embedder->start("<|video|> and <|video|>", kVideo, on_embedded, &unused).error == EmbedError::Placeholder);
    CHECK(embedder->start("<|image|> <|video|>", kVideo, on_embedded, &unused).error == EmbedError::Placeholder);
    CHECK_FALSE(unused.done);
}
