// llama.cpp's preparation of audio for EmbeddingGemma 2's audio encoder,
// written as the fixtures the audio tests read. A development tool, run by
// tools/make_reference_audio.sh; never part of the harness.
//
//     reference_audio decode <model.gguf> <mmproj.gguf> <in> <out.wav> <seconds>
//     reference_audio prepare <mmproj.gguf> <fixtures dir> <llama.cpp commit> <mmproj sha256>
//     reference_audio plain <model.gguf> <mmproj.gguf> <fixtures dir> <out>
//     reference_audio embed <model.gguf> <mmproj.gguf> <fixtures dir> <plain> <commit> <model sha256>
//                           <mmproj sha256>
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
//
// embed: each item below embedded by EmbeddingGemma 2 with its clip's rows
// in place of its placeholder, as runtime/audio.h's encode puts them — the
// ends, the text before, <|audio>, the rows, <audio|>, the text after —
// on the CPU and on Metal, with the floor patch applied; written as
// tests/fixtures/reference/audio_embeddings.inc for
// tests/gpu/audio_embed_test.cpp, each with its cosine against `plain`'s.
//
// plain: the same items' embeddings on Metal from a build without the
// patch, unchanged llama.cpp, written raw to a file `embed` reads.

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

// One item to embed: a text, and where it holds a clip, the clip — a sound
// of sounds.h or a recording beside it — the text before it and after.
struct Item {
    const char* name;
    const char* clip;   // "" for a text alone
    const char* before;
    const char* after;
};

constexpr Item kItems[] = {
    {"tone", "tone", "", ""},
    {"siren", "siren", "", ""},
    {"noise", "noise", "", ""},
    {"clicks", "clicks", "", ""},
    {"silence", "silence", "", ""},
    {"kennedy", "kennedy", "", ""},
    {"christmas", "christmas", "", ""},
    {"countdown", "countdown", "", ""},
    {"sputnik", "sputnik", "", ""},
    {"captioned", "christmas", "title: none | text: From lunar orbit, ", " A message from Apollo 8."},
    {"query_kennedy", "", "task: search result | query: a president explaining why we go to the moon", ""},
    {"query_christmas", "", "task: search result | query: a christmas greeting from astronauts orbiting the moon", ""},
    {"query_countdown", "", "task: search result | query: a countdown to a rocket launch", ""},
    {"query_sputnik", "", "task: search result | query: the beeping signal of the first satellite", ""},
};

constexpr int kWidth = 512;   // the text model's, the rows' and the vector's

std::vector<float> clip_samples(const std::string& dir, std::string_view clip) {
    for (const fx::Sound& sound : fx::kSounds) {
        if (sound.name != clip) continue;
        std::vector<float> out(sound.samples);
        for (std::uint32_t i = 0; i < sound.samples; ++i) out[i] = fx::sample(sound, i);
        return out;
    }
    std::string error;
    std::vector<float> out = fx::read_wav(dir + "/" + std::string(clip) + ".wav", error);
    if (out.empty()) fail(error);
    return out;
}

struct Backend {
    llama_model* model;
    llama_context* context;
    mtmd_context* media;
};

Backend open_backend(const char* model_path, const char* mmproj, bool cpu) {
    llama_model_params mp = llama_model_default_params();
    static ggml_backend_dev_t no_devices[] = {nullptr};
    if (cpu) {
        mp.devices = no_devices;
        mp.n_gpu_layers = 0;
    } else {
        mp.n_gpu_layers = -1;
    }
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) fail("could not load the model");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.embeddings = true;
    if (cpu) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    llama_context* context = llama_init_from_model(model, cp);
    if (context == nullptr) fail("could not make a context");
    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = !cpu;
    vp.warmup = false;
    mtmd_context* media = mtmd_init_from_file(mmproj, model, vp);
    if (media == nullptr) fail("could not load the mmproj");
    return {model, context, media};
}

std::vector<llama_token> tokens_of(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> tokens(text.size() + 8);
    const int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(),
                                 static_cast<int>(tokens.size()), false, true);
    if (n < 0) fail("could not tokenize " + text);
    tokens.resize(n);
    return tokens;
}

// The clip's rows from the audio encoder: its audio chunk encoded, one chunk
// for a clip of at most 30 s.
std::vector<float> clip_rows(mtmd_context* media, const std::vector<float>& samples, std::uint32_t& rows) {
    mtmd_bitmap* bitmap = mtmd_bitmap_init_from_audio(samples.size(), samples.data());
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[] = {bitmap};
    const char* marker = mtmd_get_marker(media);
    mtmd_input_text text{marker, std::strlen(marker), false, true};
    if (mtmd_tokenize(media, chunks, &text, bitmaps, 1) != 0) fail("could not tokenize the clip");
    std::vector<float> out;
    std::size_t audio = 0;
    for (std::size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) continue;
        ++audio;
        if (mtmd_encode_chunk(media, chunk) != 0) fail("could not encode the clip");
        rows = static_cast<std::uint32_t>(mtmd_input_chunk_get_n_tokens(chunk));
        const float* embd = mtmd_get_output_embd(media);
        out.assign(embd, embd + static_cast<std::size_t>(rows) * kWidth);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    if (audio != 1 || rows == 0) fail("the clip did not make one audio chunk");
    return out;
}

struct Embedding {
    std::vector<llama_token> before;   // the ends and the text before the clip's rows
    std::uint32_t rows = 0;            // the clip's rows
    std::vector<llama_token> after;
    std::vector<float> vector;
};

Embedding embed_item(const Backend& b, const Item& item, const std::string& dir) {
    const llama_vocab* vocab = llama_model_get_vocab(b.model);
    Embedding e;
    if (llama_vocab_get_add_bos(vocab)) e.before.push_back(llama_vocab_bos(vocab));
    for (const llama_token t : tokens_of(vocab, item.before)) e.before.push_back(t);
    std::vector<float> rows;
    const bool clip = item.clip[0] != '\0';
    if (clip) {
        for (const llama_token t : tokens_of(vocab, "<|audio>")) e.before.push_back(t);
        rows = clip_rows(b.media, clip_samples(dir, item.clip), e.rows);
        for (const llama_token t : tokens_of(vocab, "<audio|>")) e.after.push_back(t);
    }
    for (const llama_token t : tokens_of(vocab, item.after)) e.after.push_back(t);
    if (llama_vocab_get_add_eos(vocab)) e.after.push_back(llama_vocab_eos(vocab));

    llama_memory_clear(llama_get_memory(b.context), true);
    llama_batch_ext* batch = llama_batch_ext_init(b.context);
    llama_pos pos = 0;
    const auto place = [&](int32_t index) {
        if (index < 0) fail("could not add to the batch");
        llama_batch_ext_set_pos(batch, index, &pos);
        llama_batch_ext_set_output_embd(batch, index, true);
        ++pos;
    };
    for (const llama_token t : e.before) place(llama_batch_ext_add_token(batch, 0, t));
    for (std::uint32_t r = 0; r < e.rows; ++r) {
        place(llama_batch_ext_add_embd(batch, 0, llama_embd{rows.data() + static_cast<std::size_t>(r) * kWidth, 1, kWidth}));
    }
    for (const llama_token t : e.after) place(llama_batch_ext_add_token(batch, 0, t));
    if (llama_process(b.context, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) fail(std::string("could not embed ") + item.name);
    const int n = llama_model_n_embd_out(b.model);
    const float* pooled = llama_get_embeddings_seq(b.context, 0);
    if (pooled == nullptr) fail("no pooled embedding");
    e.vector.assign(pooled, pooled + n);
    llama_batch_ext_free(batch);
    double sum = 0;
    for (const float v : e.vector) sum += static_cast<double>(v) * v;
    const float scale = sum > 0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0f;
    for (float& v : e.vector) v *= scale;
    return e;
}

void close_backend(const Backend& b) {
    mtmd_free(b.media);
    llama_free(b.context);
    llama_model_free(b.model);
}

int plain(const char* model, const char* mmproj, const std::string& dir, const char* out_path) {
    llama_backend_init();
    const Backend metal = open_backend(model, mmproj, false);
    std::ofstream out(out_path, std::ios::binary);
    for (const Item& item : kItems) {
        if (item.clip[0] == '\0') continue;
        const Embedding e = embed_item(metal, item, dir);
        out.write(reinterpret_cast<const char*>(e.vector.data()),
                  static_cast<std::streamsize>(e.vector.size() * sizeof(float)));
    }
    if (!out) fail(std::string("could not write ") + out_path);
    close_backend(metal);
    llama_backend_free();
    return 0;
}

void print_floats(const std::vector<float>& v) {
    std::printf("{");
    for (std::size_t i = 0; i < v.size(); ++i) std::printf("%s%af", i ? ", " : "", v[i]);
    std::printf("}");
}

void print_tokens(const std::vector<llama_token>& t) {
    std::printf("{");
    for (std::size_t i = 0; i < t.size(); ++i) std::printf("%s%d", i ? ", " : "", t[i]);
    std::printf("}");
}

int embed(const char* model, const char* mmproj, const std::string& dir, const char* plain_path, const char* commit,
          const char* model_sha, const char* mmproj_sha) {
    std::ifstream plain_file(plain_path, std::ios::binary);
    const std::vector<char> plain_bytes((std::istreambuf_iterator<char>(plain_file)), {});
    std::size_t plain_at = 0;
    llama_backend_init();
    const Backend cpu = open_backend(model, mmproj, true);
    const Backend metal = open_backend(model, mmproj, false);

    std::printf("// Generated by tools/make_reference_audio.sh. Do not edit.\n");
    std::printf("// llama.cpp %s with tools/reference_audio/mel_floor.patch, CPU and Metal, the model\n", commit);
    std::printf("// of SHA-256 %s and the mmproj of SHA-256\n// %s. Each item: its name, its clip or\n", model_sha,
                mmproj_sha);
    std::printf("// none, its text before and after the clip, the tokens before the clip's rows (the\n");
    std::printf("// ends and <|audio> among them), its rows, the tokens after, and its unit-length\n");
    std::printf("// embedding on Metal and on the CPU.\n");
    std::printf("inline const AudioReference kAudioReferences[] = {\n");
    for (const Item& item : kItems) {
        const Embedding m = embed_item(metal, item, dir), c = embed_item(cpu, item, dir);
        if (m.before != c.before || m.after != c.after || m.rows != c.rows) {
            std::fprintf(stderr, "%s: the backends built different sequences\n", item.name);
            return 1;
        }
        if (item.clip[0] != '\0') {
            const std::size_t bytes = m.vector.size() * sizeof(float);
            if (plain_at + bytes > plain_bytes.size()) fail("the plain embeddings end early");
            std::vector<float> p(m.vector.size());
            std::memcpy(p.data(), plain_bytes.data() + plain_at, bytes);
            plain_at += bytes;
            double dot = 0;
            for (std::size_t i = 0; i < m.vector.size(); ++i) dot += static_cast<double>(m.vector[i]) * p[i];
            std::printf("    // %s: cosine with unchanged llama.cpp's floor, on Metal: %.6f\n", item.name, dot);
        }
        std::printf("    {\"%s\", \"%s\", R\"REF(%s)REF\", R\"REF(%s)REF\", ", item.name, item.clip, item.before,
                    item.after);
        print_tokens(m.before);
        std::printf(", %u, ", m.rows);
        print_tokens(m.after);
        std::printf(", ");
        print_floats(m.vector);
        std::printf(", ");
        print_floats(c.vector);
        std::printf("},\n");
        std::fprintf(stderr, "%s: %zu tokens, %u rows\n", item.name, m.before.size() + m.after.size(), m.rows);
    }
    std::printf("};\n");
    if (plain_at != plain_bytes.size()) fail("the plain embeddings hold more items than these");
    close_backend(cpu);
    close_backend(metal);
    llama_backend_free();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 7 && std::string(argv[1]) == "decode") return decode(argv[2], argv[3], argv[4], argv[5], std::atof(argv[6]));
    if (argc == 6 && std::string(argv[1]) == "prepare") return prepare(argv[2], argv[3], argv[4], argv[5]);
    if (argc == 6 && std::string(argv[1]) == "plain") return plain(argv[2], argv[3], argv[4], argv[5]);
    if (argc == 9 && std::string(argv[1]) == "embed") {
        return embed(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
    }
    std::fprintf(stderr, "usage: reference_audio decode <model.gguf> <mmproj.gguf> <in> <out.wav> <seconds>\n"
                         "       reference_audio prepare <mmproj.gguf> <fixtures dir> <commit> <mmproj sha256>\n"
                         "       reference_audio plain <model.gguf> <mmproj.gguf> <fixtures dir> <out>\n"
                         "       reference_audio embed <model.gguf> <mmproj.gguf> <fixtures dir> <plain> <commit>"
                         " <model sha256> <mmproj sha256>\n");
    return 2;
}
