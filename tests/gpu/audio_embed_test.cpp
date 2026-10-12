// A clip embedded on the GPU (runtime/audio.h), EmbeddingGemma 2's text
// model and its audio encoder at Q8_0, fetched as test data, pinned by
// SHA-256 (tests/fixtures/external.json), uploaded on one device as the
// harness uploads them: the text model whole, the mmproj by its audio range
// alone (residency/file_ranges.h).
//
//   - Against the reference (tools/make_reference_audio.sh): the computed
//     sounds and the recordings of tests/fixtures/audio, alone and one with a
//     caption around it, and four text queries — its tokens and rows
//     llama.cpp's, its embedding within the largest difference llama.cpp's
//     own CPU and Metal backends show on it of Metal's, component by
//     component, and its cosine with Metal's at least the CPU's with Metal's:
//     the text embedder's check (embedder_test.cpp), Metal the reference.
//   - The check discriminates: the spectrogram in llama.cpp's floor form
//     falls outside on the tone and the siren; each layer's relative rows
//     written unshifted, so distance j reads the row of distance 12 − j,
//     fall outside on every recording.
//   - Each query ranks its own recording first among the four.
//   - The same clip embedded twice gives the same bits, and the same when the
//     next clip is prepared while it encodes; a text after a clip gives the
//     bits a fresh embedder gives; with both encoders attached, an image
//     after a clip gives the image's bits, and a clip after an image the
//     clip's; refusals named.

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

#include "core/arch/gemma4a/gemma4a.h"
#include "core/kernels/program.h"
#include "core/media/audio.h"
#include "core/runtime/audio.h"
#include "core/runtime/embedder.h"
#include "core/runtime/ends.h"
#include "core/runtime/vision.h"
#include "fixtures/audio/sounds.h"
#include "fixtures/audio/wav.h"
#include "fixtures/vision/scenes.h"
#include "support/acquire.h"
#include "support/companions.h"
#include "support/program.h"
#include "support/pump.h"

using namespace bllm;
using namespace bllm::testing;
using runtime::EmbedError;
namespace fx = bllm::fixtures::audio;

namespace {

// One item's reference, as tools/reference_audio writes it.
struct AudioReference {
    std::string_view name;
    std::string_view clip;   // a sound of sounds.h, a recording beside it, or empty for a text alone
    std::string_view before;
    std::string_view after;
    std::vector<std::uint32_t> tokens_before;   // the ends and <|audio> among them
    std::uint32_t rows;
    std::vector<std::uint32_t> tokens_after;
    std::vector<float> metal;
    std::vector<float> cpu;
};

#include "fixtures/reference/audio_embeddings.inc"

constexpr std::uint32_t kMostRows = 508;
constexpr std::uint32_t kPositions = 13;

// How the setup writes each layer's relative rows: as the encoder reads
// them, row p the position 12 − p; or unshifted, row p the position p, so
// distance j reads the row of distance 12 − j — the discriminator.
enum class Relative {
    Shifted,
    Unshifted,
};

struct Ran {
    kernels::ProgramError error = kernels::ProgramError::Step;
    std::string message;
    bool done = false;
};

void on_ran(kernels::ProgramError error, std::string_view message, std::span<const std::byte>, void* userdata) {
    auto& r = *static_cast<Ran*>(userdata);
    r.error = error;
    r.message = message;
    r.done = true;
}

void run_setup(WGPUInstance instance, const AudioModel& a, Relative relative) {
    const auto setup = build_program(instance, *a.upload, a.setup);
    Ran ran;
    const std::uint32_t width = a.description.audio->width;
    if (relative == Relative::Shifted) {
        runtime::run_audio_setup(*setup, binding_of(a.positions), width, on_ran, &ran);
    } else {
        // The same sinusoids as runtime/audio.cpp's, row p the position p.
        const std::uint32_t timescales = width / 2;
        const float increment = static_cast<float>(std::log(10000.0)) / static_cast<float>(timescales - 1);
        std::vector<float> rows(std::size_t{kPositions} * width);
        for (std::uint32_t p = 0; p < kPositions; ++p) {
            for (std::uint32_t i = 0; i < timescales; ++i) {
                const auto inverse =
                    static_cast<float>(std::exp(static_cast<double>(-static_cast<float>(i) * increment)));
                const float scaled = static_cast<float>(p) * inverse;
                rows[std::size_t{p} * width + i] = static_cast<float>(std::sin(static_cast<double>(scaled)));
                rows[std::size_t{p} * width + i + timescales] =
                    static_cast<float>(std::cos(static_cast<double>(scaled)));
            }
        }
        kernels::Step step{};
        step.tokens = kPositions;
        const kernels::RunInput input{binding_of(a.positions), std::as_bytes(std::span(rows))};
        setup->run(step, std::span(&input, 1), {}, on_ran, &ran);
    }
    pump_until(instance, ran.done, "the setup");
    REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
}

// Test-only: a stage that runs the audio encoder's real program over a
// spectrogram given whole, rather than one Audio prepares — the floor
// discriminator's, its spectrogram in llama.cpp's floor form. Behind the
// embedder's MediaStage interface, as Audio is.
class GivenSpectrogram final : public runtime::MediaStage {
public:
    GivenSpectrogram(std::unique_ptr<kernels::Program> program, std::unique_ptr<kernels::ProgramLink> link,
                     kernels::Binding mel)
        : program_(std::move(program)), link_(std::move(link)), mel_(mel) {}

    void give(const media::AudioClip& clip, std::vector<float> mel) {
        clip_ = clip;
        spectrogram_ = std::move(mel);
        ready_ = true;
    }

    [[nodiscard]] std::uint32_t ready_rows() const noexcept override { return ready_ ? clip_.rows : 0; }

    void run(kernels::StepCallback done, void* userdata) override {
        kernels::Step step{};
        step.tokens = clip_.rows;
        step.columns = clip_.frames;
        const kernels::RunInput input{mel_, std::as_bytes(std::span(spectrogram_))};
        program_->run(step, std::span(&input, 1), {}, done, userdata);
        ready_ = false;
    }

    [[nodiscard]] const kernels::ProgramLink& link() const noexcept override { return *link_; }

private:
    std::unique_ptr<kernels::Program> program_;
    std::unique_ptr<kernels::ProgramLink> link_;
    kernels::Binding mel_;
    media::AudioClip clip_{};
    std::vector<float> spectrogram_;
    bool ready_ = false;
};

runtime::MediaTokens audio_tokens_of(const tokenizer::Tokenizer& t) {
    return media_tokens_of(t, arch::kGemma4Audio.tokens);
}

std::unique_ptr<runtime::Embedder> text_embedder(WGPUInstance instance, const Text& m) {
    auto t = tokenizer_of(m);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(m.index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    return std::make_unique<runtime::Embedder>(std::move(t),
                                               build_program(instance, *m.upload, m.launches, m.embedding), ends,
                                               m.upload->plan().context_offered, 768);
}

// The text embedder with the audio encoder attached, its setup run first.
std::unique_ptr<runtime::Embedder> embedder_of(WGPUInstance instance, const Text& m, const AudioModel& a,
                                               Relative relative = Relative::Shifted) {
    run_setup(instance, a, relative);
    auto embedder = text_embedder(instance, m);
    const runtime::MediaTokens tokens = audio_tokens_of(*tokenizer_of(m));
    embedder->attach(arch::MediaKind::Audio,
                     std::make_unique<runtime::Audio>(build_program(instance, *a.upload, a.launches),
                                                      link_of(m, *a.upload, a.soft_tokens), binding_of(a.mel),
                                                      kMostRows),
                     tokens, m.description.embedding_width);
    return embedder;
}

runtime::Audio& audio_of(runtime::Embedder& embedder) {
    auto* audio = dynamic_cast<runtime::Audio*>(embedder.stage(arch::MediaKind::Audio));
    REQUIRE(audio != nullptr);
    return *audio;
}

std::vector<float> samples_of(std::string_view clip) {
    for (const fx::Sound& sound : fx::kSounds) {
        if (sound.name != clip) continue;
        std::vector<float> out(sound.samples);
        for (std::uint32_t i = 0; i < sound.samples; ++i) out[i] = fx::sample(sound, i);
        return out;
    }
    std::string error;
    const std::string path = std::string(BLLM_AUDIO_FIXTURE_DIR) + "/" + std::string(clip) + ".wav";
    std::vector<float> out = fx::read_wav(path, error);
    REQUIRE_MESSAGE(!out.empty(), error);
    return out;
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

std::string text_of(const AudioReference& r) {
    return std::string(r.before) + (r.clip.empty() ? "" : "<|audio|>") + std::string(r.after);
}

// Starts the item's encode, its clip already prepared where it has one.
void start(runtime::Embedder& embedder, const AudioReference& r, Embedded& e) {
    const runtime::EmbedResult started = embedder.start(
        text_of(r), r.clip.empty() ? std::nullopt : std::optional{arch::MediaKind::Audio}, on_embedded, &e);
    REQUIRE(started.error == EmbedError::Ok);
    CHECK(started.tokens == r.tokens_before.size() + r.rows + r.tokens_after.size());
}

Embedded finish_encode(WGPUInstance instance, Embedded& e) {
    pump_until(instance, e.done, "the encode");
    REQUIRE_MESSAGE(e.result->error == EmbedError::Ok, e.message);
    return std::move(e);
}

// The item embedded: its clip prepared and placed, or its text alone.
Embedded embed(WGPUInstance instance, runtime::Embedder& embedder, const AudioReference& r) {
    if (!r.clip.empty()) {
        REQUIRE(audio_of(embedder).prepare(samples_of(r.clip)) == media::AudioError::Ok);
        REQUIRE(audio_of(embedder).ready_rows() == r.rows);
    }
    Embedded e;
    start(embedder, r, e);
    return finish_encode(instance, e);
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
bool within_spread(const AudioReference& r, std::span<const float> got) {
    return largest_difference(got, r.metal) <= largest_difference(r.cpu, r.metal) &&
           dot(got, r.metal) >= dot(r.cpu, r.metal);
}

const AudioReference& reference_named(std::string_view name) {
    for (const AudioReference& r : kAudioReferences) {
        if (r.name == name) return r;
    }
    FAIL("no reference " << name);
    return kAudioReferences[0];
}

bool recording(const AudioReference& r) {
    return !r.clip.empty() && r.name == r.clip &&
           std::none_of(fx::kSounds.begin(), fx::kSounds.end(), [&](const fx::Sound& s) { return s.name == r.clip; });
}

}  // namespace

TEST_CASE("each clip and query is embedded within llama.cpp's own spread, its tokens and rows llama.cpp's") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto audio = upload_audio(instance.get(), *device, kMostRows);
    const auto embedder = embedder_of(instance.get(), *text, *audio);

    // The tokens, before the GPU: the text with its placeholder, encoded and
    // wrapped as the embedder does, the placeholder replaced.
    const auto t = tokenizer_of(*text);
    const runtime::MediaTokens clip_tokens = audio_tokens_of(*t);
    runtime::Ends ends;
    REQUIRE(runtime::resolve_ends(text->index, t->vocabulary(), ends).error == runtime::EndsError::Ok);
    const auto id_of = [](tokenizer::TokenId id) { return static_cast<std::uint32_t>(id); };
    for (const AudioReference& r : kAudioReferences) {
        CAPTURE(r.name);
        std::vector<tokenizer::TokenId> tokens;
        if (ends.bos) tokens.push_back(*ends.bos);
        REQUIRE(t->encode(text_of(r), 512, tokens).error == tokenizer::EncodeError::Ok);
        if (ends.eos) tokens.push_back(*ends.eos);
        std::vector<std::uint32_t> ids;
        for (const tokenizer::TokenId id : tokens) ids.push_back(id_of(id));
        const std::vector<std::uint32_t> before(r.tokens_before.begin(), r.tokens_before.end());
        const std::vector<std::uint32_t> after(r.tokens_after.begin(), r.tokens_after.end());
        if (!r.clip.empty()) {
            const auto at = std::find(ids.begin(), ids.end(), id_of(clip_tokens.placeholder));
            REQUIRE(at != ids.end());
            CHECK(std::vector<std::uint32_t>(ids.begin(), at) ==
                  std::vector<std::uint32_t>(before.begin(), before.end() - 1));
            CHECK(before.back() == id_of(clip_tokens.begin));
            CHECK(after.front() == id_of(clip_tokens.end));
            CHECK(std::vector<std::uint32_t>(at + 1, ids.end()) ==
                  std::vector<std::uint32_t>(after.begin() + 1, after.end()));
        } else {
            std::vector<std::uint32_t> whole = before;
            whole.insert(whole.end(), after.begin(), after.end());
            CHECK(ids == whole);
        }
    }

    for (const AudioReference& r : kAudioReferences) {
        CAPTURE(r.name);
        const Embedded e = embed(instance.get(), *embedder, r);
        MESSAGE(r.name << ": largest difference " << largest_difference(e.vector, r.metal) << " against llama.cpp's "
                       << largest_difference(r.cpu, r.metal) << "; cosine " << dot(e.vector, r.metal) << " against "
                       << dot(r.cpu, r.metal));
        CHECK(within_spread(r, e.vector));
    }
}

TEST_CASE("the check discriminates: llama.cpp's floor on the tone and siren, unshifted relative rows on recordings") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto audio = upload_audio(instance.get(), *device, kMostRows);

    SUBCASE("the spectrogram in llama.cpp's floor form, log(max(x, floor))") {
        run_setup(instance.get(), *audio, Relative::Shifted);
        auto embedder = text_embedder(instance.get(), *text);
        auto stage = std::make_unique<GivenSpectrogram>(
            build_program(instance.get(), *audio->upload, audio->launches),
            link_of(*text, *audio->upload, audio->soft_tokens), binding_of(audio->mel));
        GivenSpectrogram& given = *stage;
        embedder->attach(arch::MediaKind::Audio, std::move(stage), audio_tokens_of(*tokenizer_of(*text)),
                         text->description.embedding_width);
        media::Spectrogram spectrogram;
        const double floor = static_cast<double>(media::kMelFloor);
        for (const std::string_view name : {"tone", "siren"}) {
            CAPTURE(name);
            const AudioReference& r = reference_named(name);
            const std::vector<float> samples = samples_of(r.clip);
            media::AudioClip clip{};
            REQUIRE(media::audio_clip(samples.size(), kMostRows, clip) == media::AudioError::Ok);
            std::vector<float> mel(std::size_t{clip.frames} * media::kMelBins);
            spectrogram.compute(samples, clip, mel);
            for (float& v : mel) {
                const double x = std::exp(static_cast<double>(v)) - floor;
                v = static_cast<float>(std::log(std::max(x, floor)));
            }
            given.give(clip, std::move(mel));
            Embedded e;
            start(*embedder, r, e);
            CHECK_FALSE(within_spread(r, finish_encode(instance.get(), e).vector));
        }
    }

    SUBCASE("each layer's relative rows unshifted") {
        const auto embedder = embedder_of(instance.get(), *text, *audio, Relative::Unshifted);
        for (const AudioReference& r : kAudioReferences) {
            if (!recording(r)) continue;
            CAPTURE(r.name);
            CHECK_FALSE(within_spread(r, embed(instance.get(), *embedder, r).vector));
        }
    }
}

TEST_CASE("each query ranks its own recording first; a clip's bits repeat, prepared alongside too; refusals named") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto audio = upload_audio(instance.get(), *device, kMostRows);
    const auto embedder = embedder_of(instance.get(), *text, *audio);

    std::vector<std::pair<std::string_view, std::vector<float>>> recordings;
    for (const AudioReference& r : kAudioReferences) {
        if (recording(r)) recordings.emplace_back(r.name, embed(instance.get(), *embedder, r).vector);
    }
    REQUIRE(recordings.size() == 4);
    for (const AudioReference& r : kAudioReferences) {
        if (!r.clip.empty()) continue;
        CAPTURE(r.name);
        const auto query = embed(instance.get(), *embedder, r).vector;
        const auto best = std::max_element(recordings.begin(), recordings.end(), [&](const auto& a, const auto& b) {
            return dot(query, a.second) < dot(query, b.second);
        });
        CHECK(std::string("query_") + std::string(best->first) == r.name);
    }

    // The same clip twice: the same bits; and the same with the next clip
    // prepared into the slot while it encodes.
    const AudioReference& kennedy = reference_named("kennedy");
    const auto alone = embed(instance.get(), *embedder, kennedy).vector;
    CHECK(embed(instance.get(), *embedder, kennedy).vector == alone);
    REQUIRE(audio_of(*embedder).prepare(samples_of("kennedy")) == media::AudioError::Ok);
    Embedded first;
    start(*embedder, kennedy, first);
    REQUIRE(audio_of(*embedder).prepare(samples_of("sputnik")) == media::AudioError::Ok);
    CHECK(finish_encode(instance.get(), first).vector == alone);
    const AudioReference& sputnik = reference_named("sputnik");
    Embedded second;
    start(*embedder, sputnik, second);
    CHECK(finish_encode(instance.get(), second).vector == embed(instance.get(), *embedder, sputnik).vector);

    // A text after a clip: a fresh embedder's bits.
    const AudioReference& query = reference_named("query_kennedy");
    const auto after = embed(instance.get(), *embedder, query).vector;
    const auto fresh = text_embedder(instance.get(), *text);
    CHECK(embed(instance.get(), *fresh, query).vector == after);

    // Refusals: a clip too short or too long; no clip prepared; a
    // placeholder without a clip, or twice; a clip and text past the context.
    runtime::Audio& stage = audio_of(*embedder);
    CHECK(stage.prepare(std::vector<float>(160, 0.0f)) == media::AudioError::TooShort);
    CHECK(stage.prepare(std::vector<float>(stage.most_samples() + 1, 0.0f)) == media::AudioError::TooLong);
    CHECK(stage.ready_rows() == 0);
    Embedded unused;
    constexpr auto kAudio = std::optional{arch::MediaKind::Audio};
    CHECK(embedder->start("<|audio|>", kAudio, on_embedded, &unused).error == EmbedError::MediaNotReady);
    CHECK(embedder->start("a <|audio|> here", std::nullopt, on_embedded, &unused).error == EmbedError::Placeholder);
    REQUIRE(stage.prepare(samples_of("kennedy")) == media::AudioError::Ok);
    CHECK(embedder->start("<|audio|> and <|audio|>", kAudio, on_embedded, &unused).error == EmbedError::Placeholder);
    CHECK(embedder->start("no placeholder", kAudio, on_embedded, &unused).error == EmbedError::Placeholder);
    CHECK(embedder->start("a longer caption than the context leaves room for beside it <|audio|>", kAudio, on_embedded,
                          &unused)
              .error == EmbedError::TooManyTokens);
    CHECK_FALSE(unused.done);
}

TEST_CASE("with both encoders attached, an image after a clip gives its own bits, and a clip after an image") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto text = upload_text(instance.get(), *device);
    const auto audio = upload_audio(instance.get(), *device, kMostRows);
    constexpr media::ImageBudget kBudget{70, 280};
    const auto vision =
        upload_vision(instance.get(), *device, model::FeedForwardActivation::GeluTanh, kBudget.most * 9);
    const auto embedder = embedder_of(instance.get(), *text, *audio);
    embedder->attach(arch::MediaKind::Image,
                     std::make_unique<runtime::Vision>(build_program(instance.get(), *vision->upload, vision->launches),
                                                       link_of(*text, *vision->upload, vision->soft_tokens),
                                                       binding_of(vision->pixels), kBudget,
                                                       vision->description.vision->patch_size,
                                                       vision->description.vision->pooling),
                     media_tokens_of(*tokenizer_of(*text), arch::kGemma4Vision.tokens),
                     text->description.embedding_width);
    auto* image_stage = dynamic_cast<runtime::Vision*>(embedder->stage(arch::MediaKind::Image));
    REQUIRE(image_stage != nullptr);

    // The red circle, prepared whole in one band, and embedded alone.
    const fixtures::vision::Scene& scene = fixtures::vision::kScenes[0];
    const auto embed_image = [&] {
        REQUIRE(image_stage->begin(scene.width, scene.height) == media::ImageError::Ok);
        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(scene.width) * scene.height * 4, 255);
        for (std::uint32_t y = 0; y < scene.height; ++y) {
            for (std::uint32_t x = 0; x < scene.width; ++x) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    const std::size_t at = (static_cast<std::size_t>(y) * scene.width + x) * 4 + c;
                    rgba[at] = fixtures::vision::pixel(scene, x, y, c);
                }
            }
        }
        REQUIRE(image_stage->rows(0, rgba) == media::ImageError::Ok);
        Embedded e;
        REQUIRE(embedder->start("<|image|>", std::optional{arch::MediaKind::Image}, on_embedded, &e).error ==
                EmbedError::Ok);
        return finish_encode(instance.get(), e).vector;
    };
    // The image first, alone; then a clip; then the image again.
    const AudioReference& sputnik = reference_named("sputnik");
    const auto image = embed_image();
    const auto clip = embed(instance.get(), *embedder, sputnik).vector;
    CHECK(embed_image() == image);
    // The clip after an image: the bits an embedder with no image stage
    // gives it.
    const auto alone = embedder_of(instance.get(), *text, *audio);
    CHECK(embed(instance.get(), *alone, sputnik).vector == clip);
    // A text naming the other kind's placeholder is refused.
    Embedded unused;
    REQUIRE(audio_of(*embedder).prepare(samples_of("sputnik")) == media::AudioError::Ok);
    CHECK(embedder->start("<|audio|> <|image|>", std::optional{arch::MediaKind::Audio}, on_embedded, &unused).error ==
          EmbedError::Placeholder);
    CHECK_FALSE(unused.done);
}
