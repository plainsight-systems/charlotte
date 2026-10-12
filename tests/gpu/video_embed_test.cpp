// A video's clip embedded on the GPU (runtime/video.h): EmbeddingGemma 2's
// text model and its vision encoder at Q8_0, as vision_embed_test.cpp loads
// them, the image stage and the video stage attached over one program.
//
//   - Against the reference (tools/make_reference_vision.sh's video mode):
//     each clip of tests/fixtures/video/scenes.h, alone and one with a
//     caption, and four text queries — its tokens and rows llama.cpp's
//     sequence's, built as EmbeddingGemma 2's processor builds a video, and
//     its embedding within llama.cpp's own CPU-to-Metal spread of Metal's,
//     as the image's check (vision_embed_test.cpp).
//   - The check discriminates: the frames letterboxed as an image is, or
//     given in reverse order, fall outside on every clip.
//   - Each query ranks its own scene's clip first among the four.
//   - A clip of four computed frames is
//     four segments, each `<|image>`, 120 rows and `<image|>`.
//   - The same clip embedded twice gives the same bits, and the same again
//     when the next clip is prepared while it runs, in the other set.
//   - An image's encode after a clip's gives the bits a fresh embedder's
//     does, and a clip's after an image's.
//   - Each refusal named.

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

// One item's reference, as tools/reference_vision's video mode writes it.
struct VideoReference {
    std::string_view name;
    std::string_view scene;   // empty for a text alone
    std::string_view before;
    std::string_view after;
    std::vector<std::uint32_t> tokens_before;   // the ends and the text before the clip
    std::uint32_t frames;
    std::uint32_t rows;   // each frame's
    std::uint32_t frame_begin;
    std::uint32_t frame_end;
    std::vector<std::uint32_t> tokens_after;
    std::vector<float> metal;
    std::vector<float> cpu;
};

#include "fixtures/reference/video_embeddings.inc"

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

// The scene's frames prepared as one clip, in bands of 97 rows; in reverse
// order where `reversed`, the discriminator's.
void prepare(runtime::Video& video, const fixtures::video::Scene& scene, bool reversed = false) {
    REQUIRE(video.begin(scene.width, scene.height, scene.frames).error == VideoError::Ok);
    for (std::uint32_t t = 0; t < scene.frames; ++t) {
        const std::uint32_t shown = reversed ? scene.frames - 1 - t : t;
        for (std::uint32_t row = 0; row < scene.height; row += 97) {
            const std::uint32_t rows = std::min(97u, scene.height - row);
            REQUIRE(video.rows(t, row, band_of(scene, shown, row, rows)).error == VideoError::Ok);
        }
        CHECK(video.ready_segments().empty() == (t + 1 < scene.frames));
    }
}

// Test-only: a clip's frames letterboxed into their grid as an image is
// (media/image.h's image_fit), where the video stage stretches them — the
// discriminator's. Run on the video stage's program, one frame a segment.
class Letterboxed final : public runtime::MediaStage {
public:
    Letterboxed(std::shared_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
                kernels::Binding pixels, std::uint32_t patch, std::uint32_t pooling,
                const fixtures::video::Scene& scene)
        : program_(std::move(program)), link_(std::move(link)), pixels_(pixels) {
        grid_ = media::frame_grid(scene.width, scene.height, patch, pooling, media::kFrameTokens);
        const media::ImageFit fit = media::image_fit(scene.width, scene.height, grid_);
        REQUIRE(fit.x + fit.y > 0);
        for (std::uint32_t t = 0; t < scene.frames; ++t) {
            media::ImageResampler r;
            REQUIRE(r.begin(scene.width, scene.height, grid_, fit, patch, media::kFrameTokens) == media::ImageError::Ok);
            REQUIRE(r.accept(band_of(scene, t, 0, scene.height), 0) == media::ImageError::Ok);
            patches_.emplace_back(std::size_t{grid_.columns} * grid_.rows * 3 * patch * patch);
            REQUIRE(r.patches(patches_.back()) == media::ImageError::Ok);
            segments_.push_back(grid_.tokens);
        }
    }

    [[nodiscard]] std::span<const std::uint32_t> ready_segments() const noexcept override {
        return taken_ ? std::span<const std::uint32_t>{} : std::span<const std::uint32_t>(segments_);
    }

    void run(std::uint32_t segment, const runtime::SegmentCopy& out, kernels::StepCallback done,
             void* userdata) override {
        taken_ = true;
        kernels::Step step{};
        step.tokens = grid_.columns * grid_.rows;
        step.columns = grid_.columns;
        const kernels::RunInput input{pixels_, std::as_bytes(std::span(patches_[segment]))};
        const kernels::RunCopy copy{link_.get(), 0, out.to_offset, out.bytes, kernels::CopyAt::AfterPass};
        program_->run(step, std::span(&input, 1), std::span(&copy, 1), done, userdata);
    }

private:
    std::shared_ptr<kernels::Program> program_;
    std::unique_ptr<kernels::ProgramLink> link_;
    kernels::Binding pixels_;
    media::ImageGrid grid_{};
    std::vector<std::vector<float>> patches_;
    std::vector<std::uint32_t> segments_;
    bool taken_ = false;
};

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
bool within_spread(const VideoReference& r, std::span<const float> got) {
    return largest_difference(got, r.metal) <= largest_difference(r.cpu, r.metal) &&
           dot(got, r.metal) >= dot(r.cpu, r.metal);
}

std::string text_of(const VideoReference& r) {
    return std::string(r.before) + (r.scene.empty() ? "" : "<|video|>") + std::string(r.after);
}

// The item embedded: its clip prepared, or its text alone; its tokens the
// reference's count.
Embedded embed(WGPUInstance instance, runtime::Embedder& embedder, const VideoReference& r) {
    const bool clip = !r.scene.empty();
    if (clip) prepare(video_of(embedder), scene_named(r.scene));
    std::uint64_t tokens = 0;
    Embedded e = embed(instance, embedder, text_of(r), clip ? kVideo : std::nullopt, tokens, [] {});
    CHECK(tokens == r.tokens_before.size() + std::size_t{r.frames} * (r.rows + 2) + r.tokens_after.size());
    return e;
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

TEST_CASE("each clip and query is embedded within llama.cpp's own spread, its tokens and rows llama.cpp's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *vision);

    // The tokens, before the GPU: the text with its placeholder, encoded and
    // wrapped as the embedder does; the frames' tokens and rows the plan's.
    const auto t = tokenizer_of(*text);
    const runtime::MediaTokens video = media_tokens_of(*t, arch::kGemma4Vision.video_tokens);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(text->index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    for (const VideoReference& r : kVideoReferences) {
        CAPTURE(r.name);
        std::vector<tokenizer::TokenId> tokens;
        if (ends.bos) tokens.push_back(*ends.bos);
        REQUIRE(t->encode(text_of(r), 512, tokens).error == tokenizer::EncodeError::Ok);
        if (ends.eos) tokens.push_back(*ends.eos);
        std::vector<std::uint32_t> ids;
        for (const tokenizer::TokenId id : tokens) ids.push_back(static_cast<std::uint32_t>(id));
        if (r.scene.empty()) {
            std::vector<std::uint32_t> whole = r.tokens_before;
            whole.insert(whole.end(), r.tokens_after.begin(), r.tokens_after.end());
            CHECK(ids == whole);
            continue;
        }
        const auto at = std::find(ids.begin(), ids.end(), static_cast<std::uint32_t>(video.placeholder));
        REQUIRE(at != ids.end());
        CHECK(std::vector<std::uint32_t>(ids.begin(), at) == r.tokens_before);
        CHECK(std::vector<std::uint32_t>(at + 1, ids.end()) == r.tokens_after);
        CHECK(r.frame_begin == static_cast<std::uint32_t>(video.begin));
        CHECK(r.frame_end == static_cast<std::uint32_t>(video.end));
        runtime::VideoPlan plan{};
        const auto& scene = scene_named(r.scene);
        REQUIRE(video_of(*embedder).plan(scene.width, scene.height, scene.frames, plan).error == VideoError::Ok);
        CHECK(plan.frames == r.frames);
        CHECK(plan.grid.tokens == r.rows);
    }

    for (const VideoReference& r : kVideoReferences) {
        CAPTURE(r.name);
        const Embedded e = embed(instance.get(), *embedder, r);
        MESSAGE(r.name << ": largest difference " << largest_difference(e.vector, r.metal) << " against llama.cpp's "
                       << largest_difference(r.cpu, r.metal) << "; cosine " << dot(e.vector, r.metal) << " against "
                       << dot(r.cpu, r.metal));
        CHECK(within_spread(r, e.vector));
    }
}

TEST_CASE("the check discriminates: frames letterboxed, or in reverse order, fall outside on every clip") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *vision);
    const std::shared_ptr<kernels::Program> program = build_program(instance.get(), *vision->upload, vision->launches);
    const runtime::MediaTokens tokens = media_tokens_of(*tokenizer_of(*text), arch::kGemma4Vision.video_tokens);
    for (const VideoReference& r : kVideoReferences) {
        if (r.scene.empty()) continue;
        CAPTURE(r.name);
        const auto& scene = scene_named(r.scene);

        prepare(video_of(*embedder), scene, true);
        const Embedded reversed = embed(instance.get(), *embedder, text_of(r), kVideo);
        MESSAGE(r.name << " reversed: largest difference " << largest_difference(reversed.vector, r.metal)
                       << " against " << largest_difference(r.cpu, r.metal));
        CHECK_FALSE(within_spread(r, reversed.vector));

        runtime::Embedder boxed(tokenizer_of(*text),
                                build_program(instance.get(), *text->upload, text->launches, text->embedding),
                                [&] {
                                    runtime::Ends ends;
                                    REQUIRE(runtime::resolve_ends(text->index, tokenizer_of(*text)->vocabulary(), ends)
                                                .error == runtime::EndsError::Ok);
                                    return ends;
                                }(),
                                text->upload->plan().context_offered, 768);
        boxed.attach(arch::MediaKind::Video,
                     std::make_unique<Letterboxed>(program, link_of(*text, *vision->upload, vision->soft_tokens),
                                                   binding_of(vision->pixels), vision->description.vision->patch_size,
                                                   vision->description.vision->pooling, scene),
                     tokens, text->description.embedding_width);
        const Embedded letterboxed = embed(instance.get(), boxed, text_of(r), kVideo);
        MESSAGE(r.name << " letterboxed: largest difference " << largest_difference(letterboxed.vector, r.metal)
                       << " against " << largest_difference(r.cpu, r.metal));
        CHECK_FALSE(within_spread(r, letterboxed.vector));
    }
}

TEST_CASE("each query ranks its own scene's clip first among the four") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *vision);
    std::vector<std::pair<std::string_view, std::vector<float>>> clips;
    for (const VideoReference& r : kVideoReferences) {
        if (r.scene.empty() || r.name != r.scene) continue;
        clips.emplace_back(r.name, embed(instance.get(), *embedder, r).vector);
    }
    REQUIRE(clips.size() == 4);
    for (const VideoReference& r : kVideoReferences) {
        if (!r.scene.empty()) continue;
        CAPTURE(r.name);
        const auto query = embed(instance.get(), *embedder, r).vector;
        const auto best = std::max_element(clips.begin(), clips.end(), [&](const auto& a, const auto& b) {
            return dot(query, a.second) < dot(query, b.second);
        });
        for (const auto& [name, clip] : clips) MESSAGE(r.name << " · " << name << ": " << dot(query, clip));
        CHECK(std::string("query_") + std::string(best->first) == r.name);
    }
}
