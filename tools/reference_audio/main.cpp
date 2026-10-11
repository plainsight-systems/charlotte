// llama.cpp's preparation of audio for EmbeddingGemma 2's audio encoder,
// written as the fixtures the audio tests read. A development tool, run by
// tools/make_reference_audio.sh; never part of the harness.
//
//     reference_audio decode <model.gguf> <mmproj.gguf> <in> <out.wav> <seconds>
//     reference_audio prepare <mmproj.gguf> <fixtures dir> <llama.cpp commit> <mmproj sha256>
//
// decode: a recording through llama.cpp's own decoder — the multimodal
// helper's, miniaudio, to 16 kHz mono f32 as llama.cpp hands audio to its
// encoder — its first `seconds` kept and written as 16-bit PCM, each sample
// rounded from x × 32,768 and clamped. The tests and the page read the WAV
// back as its integers over 32,768 (tests/fixtures/audio/wav.h), and every
// reference is made from those samples, so all of them see the same input.
//
// prepare: the gemma4a preprocessor llama.cpp itself runs
// (mtmd_audio_preprocessor_gemma4a), with tools/reference_audio/
// mel_floor.patch applied so its log takes Google's floor: every frame of
// half a second of the tone, the siren (tests/fixtures/audio/sounds.h) and
// the countdown recording, frame-major; and, over a table of lengths, the
// frames it keeps and the rows gemma4a's encoder makes of them
// (clip_n_output_tokens).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "clip-impl.h"
#include "clip.h"
#include "llama.h"
#include "mtmd-audio.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "tests/fixtures/audio/sounds.h"
#include "tests/fixtures/audio/wav.h"

namespace {

namespace fx = bllm::fixtures::audio;

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "reference_audio: %s\n", message.c_str());
    std::exit(1);
}

void write_wav(const std::string& path, const std::vector<float>& samples) {
    std::vector<std::int16_t> pcm(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const float scaled = std::round(samples[i] * 32768.0f);
        pcm[i] = static_cast<std::int16_t>(std::clamp(scaled, -32768.0f, 32767.0f));
    }
    const std::uint32_t data = static_cast<std::uint32_t>(pcm.size() * 2);
    std::ofstream out(path, std::ios::binary);
    const auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    const auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    out.write("RIFF", 4);
    u32(36 + data);
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);                  // PCM
    u16(1);                  // mono
    u32(16000);
    u32(32000);              // bytes a second
    u16(2);                  // bytes a frame
    u16(16);
    out.write("data", 4);
    u32(data);
    out.write(reinterpret_cast<const char*>(pcm.data()), data);
    if (!out) fail("could not write " + path);
}

int decode(const char* model_path, const char* mmproj, const char* in, const char* out, double seconds) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) fail("could not load the model");
    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = false;
    vp.warmup = false;
    mtmd_context* media = mtmd_init_from_file(mmproj, model, vp);
    if (media == nullptr) fail("could not load the mmproj");
    std::ifstream file(in, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
    mtmd_bitmap* bitmap =
        mtmd_helper_bitmap_init_from_buf(media, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default()).bitmap;
    if (bitmap == nullptr || !mtmd_bitmap_is_audio(bitmap)) fail(std::string("could not decode ") + in);
    const auto* data = reinterpret_cast<const float*>(mtmd_bitmap_get_data(bitmap));
    std::vector<float> samples(data, data + mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float));
    samples.resize(std::min(samples.size(), static_cast<std::size_t>(seconds * fx::kRate)));
    write_wav(out, samples);
    std::fprintf(stderr, "%s: %zu samples\n", out, samples.size());
    mtmd_bitmap_free(bitmap);
    mtmd_free(media);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

clip_ctx* load_audio(const char* path) {
    clip_context_params params{};
    params.use_gpu = false;
    params.device = nullptr;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;
    params.no_alloc = true;   // the preprocessor reads hyperparameters, not weights
    const clip_init_result loaded = clip_init(path, params);
    if (loaded.ctx_a == nullptr) fail(std::string("no audio encoder in ") + path);
    return loaded.ctx_a;
}

// llama.cpp's one spectrogram of `samples`: its frames, frame-major, and
// how many.
std::vector<float> spectrogram(const mtmd_audio_preprocessor_gemma4a& preprocessor, const std::vector<float>& samples,
                               std::int64_t& frames) {
    std::vector<mtmd_audio_mel> chunks;
    if (!preprocessor.preprocess(samples.data(), samples.size(), chunks) || chunks.size() != 1) {
        fail("the preprocessor did not make one spectrogram");
    }
    const mtmd_audio_mel& mel = chunks.front();
    frames = mel.n_len;
    std::vector<float> out(static_cast<std::size_t>(frames) * mel.n_mel);
    for (std::int64_t t = 0; t < frames; ++t) {
        for (std::int64_t m = 0; m < mel.n_mel; ++m) out[t * mel.n_mel + m] = mel.data[m * mel.n_len + t];
    }
    return out;
}

int rows_of(clip_ctx* ctx, std::int64_t frames) {
    clip_image_f32 mel;
    mel.set_size({static_cast<int>(frames), 128}, true, true);
    return clip_n_output_tokens(ctx, &mel);
}

int prepare(const char* mmproj, const std::string& dir, const char* commit, const char* sha256) {
    clip_ctx* ctx = load_audio(mmproj);
    mtmd_audio_preprocessor_gemma4a preprocessor(ctx);
    preprocessor.initialize();

    std::string error;
    const std::vector<float> countdown = fx::read_wav(dir + "/countdown.wav", error);
    if (countdown.empty()) fail(error);
    const auto sound = [](std::string_view name) {
        for (const fx::Sound& s : fx::kSounds) {
            if (s.name == name) return s;
        }
        fail("no sound named " + std::string(name));
    };
    constexpr std::uint32_t kHalf = fx::kRate / 2;
    struct Clip {
        const char* name;
        std::vector<float> samples;
    };
    std::vector<Clip> clips;
    for (const char* name : {"tone", "siren"}) {
        Clip c{name, std::vector<float>(kHalf)};
        for (std::uint32_t i = 0; i < kHalf; ++i) c.samples[i] = fx::sample(sound(name), i);
        clips.push_back(std::move(c));
    }
    clips.push_back({"countdown", std::vector<float>(countdown.begin(), countdown.begin() + kHalf)});

    std::printf("// Generated by tools/make_reference_audio.sh: llama.cpp %s's gemma4a preprocessor,\n", commit);
    std::printf("// its log taking Google's floor (tools/reference_audio/mel_floor.patch), for mmproj\n");
    std::printf("// sha256 %s. Do not edit.\n\n", sha256);
    for (const Clip& c : clips) {
        std::int64_t frames = 0;
        const std::vector<float> mel = spectrogram(preprocessor, c.samples, frames);
        std::printf("// %s: half a second, %lld frames of 128 bins, frame-major.\n", c.name,
                    static_cast<long long>(frames));
        std::printf("inline const float kMel_%s[] = {", c.name);
        for (std::size_t i = 0; i < mel.size(); ++i) {
            const char* separator = i == 0 ? "\n    " : i % 8 != 0 ? ", " : ",\n    ";
            std::printf("%s%af", separator, mel[i]);
        }
        std::printf("\n};\n\n");
    }
    std::printf("struct AudioMelReference { std::string_view name; std::span<const float> mel; };\n");
    std::printf("inline const AudioMelReference kMelReferences[] = {\n");
    for (const Clip& c : clips) std::printf("    {\"%s\", kMel_%s},\n", c.name, c.name);
    std::printf("};\n\n");

    // Lengths: none, the first frame, the edges of a 508-row clip.
    constexpr std::uint32_t kLengths[] = {160, 161, 320, 321, 16000, 48000, 325280, 325281};
    std::printf("// Samples, the frames the preprocessor keeps, the rows gemma4a makes of them.\n");
    std::printf("struct AudioClipReference { std::uint32_t samples; std::uint32_t frames; std::uint32_t rows; };\n");
    std::printf("inline constexpr AudioClipReference kClipReferences[] = {\n");
    for (const std::uint32_t n : kLengths) {
        std::vector<mtmd_audio_mel> chunks;
        const std::vector<float> zeros(n, 0.0f);
        std::int64_t frames = 0;
        if (preprocessor.preprocess(zeros.data(), zeros.size(), chunks) && chunks.size() == 1) {
            frames = chunks.front().n_len;
        } else if (!chunks.empty()) {
            fail("more than one spectrogram for " + std::to_string(n) + " samples");
        }
        std::printf("    {%u, %lld, %d},\n", n, static_cast<long long>(frames), frames > 0 ? rows_of(ctx, frames) : 0);
    }
    std::printf("};\n");
    clip_free(ctx);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 7 && std::string(argv[1]) == "decode") return decode(argv[2], argv[3], argv[4], argv[5], std::atof(argv[6]));
    if (argc == 6 && std::string(argv[1]) == "prepare") return prepare(argv[2], argv[3], argv[4], argv[5]);
    std::fprintf(stderr, "usage: reference_audio decode <model.gguf> <mmproj.gguf> <in> <out.wav> <seconds>\n"
                         "       reference_audio prepare <mmproj.gguf> <fixtures dir> <commit> <mmproj sha256>\n");
    return 2;
}
