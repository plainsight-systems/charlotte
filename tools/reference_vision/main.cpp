// llama.cpp's preparation of images for EmbeddingGemma 2's vision encoder,
// written as the C++ fixture tests/media_image_test.cpp reads. A
// development tool, run by tools/make_reference_vision.sh; never part of the
// harness.
//
//     reference_vision prepare <mmproj.gguf> <llama.cpp commit> <mmproj sha256>
//     reference_vision embed <model.gguf> <mmproj, clip.use_gelu set> <mmproj as published>
//                            <llama.cpp commit> <model sha256> <mmproj sha256>
//     reference_vision video <model.gguf> <mmproj, clip.use_gelu set>
//                            <llama.cpp commit> <model sha256> <mmproj sha256>
//     reference_vision frames <model.gguf> <mmproj, clip.use_gelu set> <manifest>
//
// The mmproj is loaded as llama.cpp loads it, with the card's default
// budget, 280 tokens at most, and llama.cpp's own least, 70. Each image of
// tests/fixtures/vision/scenes.h is computed from its coordinates and handed
// to the gemma4v preprocessor llama.cpp itself runs
// (mtmd_image_preprocessor_dyn_size): the size its rule gives, its resize
// with the padding gemma4v uses, and its floats, x / 255. The bytes are
// recovered from the floats, exactly, and kept as their FNV-1a hash with
// the grid. A table of further sizes keeps the grid alone.
//
// embed: each item's unit-length embedding on llama.cpp's CPU and on Metal,
// mean-pooled and L2-normalized, as the text reference's are. An image's
// rows come from the mmproj llama.cpp is given with clip.use_gelu set, so it
// computes the activation the model was trained with (arch/gemma4v); the
// sequence is built as llama.cpp's multimodal path builds it for gemma4v —
// the file's ends, the text before the image, `<|image>`, the image's rows,
// `<image|>`, the text after — and run as one batch of tokens and rows
// mixed, so every row attends to every other, as its server runs one. On
// Metal, each image is embedded again from the mmproj as published, and the
// cosine between the two written beside it: what llama.cpp's default
// activation costs.
//
// video: each clip of tests/fixtures/video/scenes.h embedded as
// EmbeddingGemma 2's video processor writes a video, not as llama.cpp's
// video helper does (runtime/video.h): its frames, one a second, each
// resized to the grid Google's rule gives at 140 tokens — stretched to it,
// by llama.cpp's own resampler, img_tool, copied from the pinned source by
// CMakeLists.txt — and encoded by llama.cpp's vision encoder at that size,
// which its preprocessor leaves as it is; the sequence the file's ends, the
// text before, for each frame `<|image>`, its rows and `<image|>`, then the
// text after, run as one batch, as embed runs an image's. On the CPU and on
// Metal, from the mmproj with clip.use_gelu set.
//
// frames: the same for clips whose frames are files — the NASA videos of
// tests/fixtures/video/, decoded by FFmpeg (tools/make_reference_vision.sh) —
// and the score of each query against each clip, for the page's check.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "clip-impl.h"
#include "clip-model.h"
#include "clip.h"
#include "llama.h"
#include "mtmd-image.h"
#include "mtmd.h"
#include "tests/fixtures/video/scenes.h"
#include "tests/fixtures/vision/scenes.h"

// llama.cpp's resize, in a namespace of its own beside the library's copy.
namespace llama_image {
#include "img_tool.inc"
}  // namespace llama_image

namespace {

using bllm::fixtures::vision::fnv1a;
using bllm::fixtures::vision::kScenes;
using bllm::fixtures::vision::pixel;
using bllm::fixtures::vision::Scene;

constexpr int kLeast = 70;
constexpr int kMost = 280;

// Sizes whose grid alone is kept: the shapes a page meets, and the edges of
// the rule.
constexpr std::uint32_t kGridSizes[][2] = {
    {640, 480},   {1920, 1080}, {1080, 1920}, {4032, 3024}, {3024, 4032}, {100, 3000}, {3000, 100},
    {5000, 5000}, {300, 7},     {30, 30},     {48, 48},     {47, 49},     {912, 672},  {13440, 48},
    {14000, 48},  {2, 1000},
};

clip_ctx* load(const char* path) {
    clip_context_params params{};
    params.use_gpu = false;
    params.device = nullptr;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.image_min_tokens = kLeast;
    params.image_max_tokens = kMost;
    params.warmup = false;
    params.no_alloc = true;   // the preprocessor reads hyperparameters, not weights
    const clip_init_result loaded = clip_init(path, params);
    if (loaded.ctx_v == nullptr) {
        std::fprintf(stderr, "reference_vision: could not load %s\n", path);
        std::exit(1);
    }
    return loaded.ctx_v;
}

clip_image_u8 image_of(std::uint32_t width, std::uint32_t height, const Scene* scene) {
    clip_image_u8 img;
    img.set_size({static_cast<int>(width), static_cast<int>(height)}, false);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * 3);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                bytes[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                    scene != nullptr ? pixel(*scene, x, y, c) : std::uint8_t{128};
            }
        }
    }
    img.cpy_buf(bytes);
    return img;
}

const clip_image_f32& prepared(const mtmd_image_preprocessor_dyn_size& preprocessor, const clip_image_u8& img,
                               mtmd_image_preproc_out& out) {
    out = preprocessor.preprocess(img);
    if (out.entries.size() != 1) {
        std::fprintf(stderr, "reference_vision: expected one prepared image, got %zu\n", out.entries.size());
        std::exit(1);
    }
    return out.entries[0];
}

int prepare(const char* mmproj, const char* commit, const char* sha256) {
    clip_ctx* ctx = load(mmproj);
    const mtmd_image_preprocessor_dyn_size preprocessor(ctx);

    std::printf("// Generated by tools/make_reference_vision.sh: llama.cpp %s's preparation of\n", commit);
    std::printf("// the images of tests/fixtures/vision/scenes.h for mmproj sha256 %s,\n", sha256);
    std::printf("// budget %d to %d tokens. Do not edit.\n\n", kLeast, kMost);
    std::printf("struct PreparedImage { std::string_view name; std::uint32_t width; std::uint32_t height; "
                "std::uint64_t hash; };\n");
    std::printf("inline constexpr PreparedImage kPrepared[] = {\n");
    for (const Scene& scene : kScenes) {
        mtmd_image_preproc_out out;
        const clip_image_f32* prepared_image = nullptr;
        try {
            prepared_image = &prepared(preprocessor, image_of(scene.width, scene.height, &scene), out);
        } catch (const std::exception& e) {
            // llama.cpp refuses it too: kept as a grid of 0 × 0.
            std::printf("    {\"%.*s\", 0, 0, 0},   // llama.cpp: %s\n", static_cast<int>(scene.name.size()),
                        scene.name.data(), e.what());
            continue;
        }
        const clip_image_f32& f = *prepared_image;
        const std::vector<float>& floats = f.get_ro_buf();
        std::vector<std::uint8_t> bytes(floats.size());
        for (std::size_t i = 0; i < floats.size(); ++i) {
            const float u = floats[i] * 255.0f;
            const long r = std::lround(u);
            if (r < 0 || r > 255 || static_cast<float>(r) / 255.0f != floats[i]) {
                std::fprintf(stderr, "reference_vision: %.*s float %zu is not a byte over 255\n",
                             static_cast<int>(scene.name.size()), scene.name.data(), i);
                return 1;
            }
            bytes[i] = static_cast<std::uint8_t>(r);
        }
        std::printf("    {\"%.*s\", %d, %d, 0x%016llxull},\n", static_cast<int>(scene.name.size()),
                    scene.name.data(), f.nx(), f.ny(),
                    static_cast<unsigned long long>(fnv1a(bytes.data(), bytes.size())));
    }
    std::printf("};\n\n");

    std::printf("struct PreparedGrid { std::uint32_t source_width; std::uint32_t source_height; "
                "std::uint32_t width; std::uint32_t height; };\n");
    std::printf("inline constexpr PreparedGrid kGrids[] = {\n");
    for (const auto& size : kGridSizes) {
        mtmd_image_preproc_out out;
        const clip_image_f32& f = prepared(preprocessor, image_of(size[0], size[1], nullptr), out);
        std::printf("    {%u, %u, %d, %d},\n", size[0], size[1], f.nx(), f.ny());
    }
    std::printf("};\n");
    clip_free(ctx);
    return 0;
}

// One item to embed: a text, and where it holds an image, the scene, the text
// before it and the text after.
struct Item {
    const char* name;
    const char* scene;    // nullptr for a text alone
    const char* before;
    const char* after;
};

constexpr Item kItems[] = {
    {"red_circle", "red_circle", "", ""},
    {"blue_square", "blue_square", "", ""},
    {"green_stripes", "green_stripes", "", ""},
    {"black_diagonal", "black_diagonal", "", ""},
    {"noise_1200x900", "noise_1200x900", "", ""},
    {"captioned", "red_circle", "title: none | text: A drawing ", " on white paper."},
    {"query_red_circle", nullptr, "task: search result | query: a red circle", ""},
    {"query_blue_square", nullptr, "task: search result | query: a blue square on yellow", ""},
    {"query_green_stripes", nullptr, "task: search result | query: green and white stripes", ""},
    {"query_black_diagonal", nullptr, "task: search result | query: a black diagonal line", ""},
};

struct Embedding {
    std::vector<llama_token> before;   // the ends and the text before the image's rows
    std::uint32_t rows = 0;            // the image's rows
    std::vector<llama_token> after;
    std::vector<float> vector;
};

struct Backend {
    llama_model* model;
    llama_context* context;
    mtmd_context* vision;
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
    if (model == nullptr) std::exit(1);
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
    if (context == nullptr) std::exit(1);
    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = !cpu;
    vp.warmup = false;
    vp.image_min_tokens = kLeast;
    vp.image_max_tokens = kMost;
    mtmd_context* vision = mtmd_init_from_file(mmproj, model, vp);
    if (vision == nullptr) std::exit(1);
    return {model, context, vision};
}

std::vector<llama_token> tokens_of(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> tokens(text.size() + 8);
    const int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(),
                                 static_cast<int>(tokens.size()), false, true);
    if (n < 0) std::exit(1);
    tokens.resize(n);
    return tokens;
}

// The scene's rows from the vision encoder: its image chunk encoded.
std::vector<float> image_rows(mtmd_context* vision, const Scene& scene, std::uint32_t& rows) {
    std::vector<unsigned char> rgb(static_cast<std::size_t>(scene.width) * scene.height * 3);
    for (std::uint32_t y = 0; y < scene.height; ++y) {
        for (std::uint32_t x = 0; x < scene.width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) rgb[(static_cast<std::size_t>(y) * scene.width + x) * 3 + c] = pixel(scene, x, y, c);
        }
    }
    mtmd_bitmap* bitmap = mtmd_bitmap_init(scene.width, scene.height, rgb.data());
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[] = {bitmap};
    const char* marker = mtmd_get_marker(vision);
    mtmd_input_text text{marker, std::strlen(marker), false, true};
    if (mtmd_tokenize(vision, chunks, &text, bitmaps, 1) != 0) std::exit(1);
    std::vector<float> out;
    for (std::size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(vision, chunk) != 0) std::exit(1);
        rows = static_cast<std::uint32_t>(mtmd_input_chunk_get_n_tokens(chunk));
        const float* embd = mtmd_get_output_embd(vision);
        out.assign(embd, embd + static_cast<std::size_t>(rows) * 512);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    if (rows == 0) std::exit(1);
    return out;
}

const Scene* scene_named(const char* name) {
    for (const Scene& s : kScenes) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

Embedding embed_item(const Backend& b, const Item& item) {
    const llama_vocab* vocab = llama_model_get_vocab(b.model);
    Embedding e;
    if (llama_vocab_get_add_bos(vocab)) e.before.push_back(llama_vocab_bos(vocab));
    for (const llama_token t : tokens_of(vocab, item.before)) e.before.push_back(t);
    std::vector<float> rows;
    if (item.scene != nullptr) {
        for (const llama_token t : tokens_of(vocab, "<|image>")) e.before.push_back(t);
        rows = image_rows(b.vision, *scene_named(item.scene), e.rows);
        for (const llama_token t : tokens_of(vocab, "<image|>")) e.after.push_back(t);
    }
    for (const llama_token t : tokens_of(vocab, item.after)) e.after.push_back(t);
    if (llama_vocab_get_add_eos(vocab)) e.after.push_back(llama_vocab_eos(vocab));

    llama_memory_clear(llama_get_memory(b.context), true);
    llama_batch_ext* batch = llama_batch_ext_init(b.context);
    llama_pos pos = 0;
    const auto place = [&](int32_t index) {
        if (index < 0) std::exit(1);
        llama_batch_ext_set_pos(batch, index, &pos);
        llama_batch_ext_set_output_embd(batch, index, true);
        ++pos;
    };
    for (const llama_token t : e.before) place(llama_batch_ext_add_token(batch, 0, t));
    for (std::uint32_t r = 0; r < e.rows; ++r) {
        place(llama_batch_ext_add_embd(batch, 0, llama_embd{rows.data() + static_cast<std::size_t>(r) * 512, 1, 512}));
    }
    for (const llama_token t : e.after) place(llama_batch_ext_add_token(batch, 0, t));
    if (llama_process(b.context, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) std::exit(1);
    const int n = llama_model_n_embd_out(b.model);
    const float* pooled = llama_get_embeddings_seq(b.context, 0);
    if (pooled == nullptr) std::exit(1);
    e.vector.assign(pooled, pooled + n);
    llama_batch_ext_free(batch);
    double sum = 0;
    for (const float v : e.vector) sum += static_cast<double>(v) * v;
    const float scale = sum > 0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0f;
    for (float& v : e.vector) v *= scale;
    return e;
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

int embed(const char* model, const char* patched, const char* published, const char* commit, const char* model_sha,
          const char* mmproj_sha) {
    llama_backend_init();
    const Backend cpu = open_backend(model, patched, true);
    const Backend metal = open_backend(model, patched, false);
    const Backend quick = open_backend(model, published, false);

    std::printf("// Generated by tools/make_reference_vision.sh. Do not edit.\n");
    std::printf("// llama.cpp %s, CPU and Metal, the model of SHA-256 %s and the mmproj of SHA-256\n", commit, model_sha);
    std::printf("// %s with clip.use_gelu set; budget %d to %d tokens an image. Each item: its\n", mmproj_sha, kLeast, kMost);
    std::printf("// name, its scene or none, its text before and after the image, the tokens before the\n");
    std::printf("// image's rows (the ends and <|image> among\n");
    std::printf("// them), its rows, the tokens after, and its unit-length embedding on Metal and on the CPU.\n");
    std::printf("inline const VisionReference kVisionReferences[] = {\n");
    for (const Item& item : kItems) {
        const Embedding m = embed_item(metal, item), c = embed_item(cpu, item);
        if (m.before != c.before || m.after != c.after || m.rows != c.rows) {
            std::fprintf(stderr, "%s: the backends built different sequences\n", item.name);
            return 1;
        }
        if (item.scene != nullptr) {
            const Embedding q = embed_item(quick, item);
            double dot = 0;
            for (std::size_t i = 0; i < m.vector.size(); ++i) dot += static_cast<double>(m.vector[i]) * q.vector[i];
            std::printf("    // %s: cosine with llama.cpp's default activation, GELU-quick, on Metal: %.6f\n", item.name, dot);
        }
        std::printf("    {\"%s\", \"%s\", R\"REF(%s)REF\", R\"REF(%s)REF\", ", item.name,
                    item.scene != nullptr ? item.scene : "", item.before, item.after);
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
    for (const Backend* b : {&cpu, &metal, &quick}) {
        mtmd_free(b->vision);
        llama_free(b->context);
        llama_model_free(b->model);
    }
    llama_backend_free();
    return 0;
}

// Google's get_aspect_ratio_preserving_size, as its Python reads, at
// `tokens` a frame: the size a frame is resized to.
clip_image_size google_size(int width, int height, int patch, int pooling, int tokens) {
    const double total_px = static_cast<double>(height) * width;
    const double target_px = static_cast<double>(tokens) * pooling * pooling * patch * patch;
    const double factor = std::sqrt(target_px / total_px);
    const int side = patch * pooling;
    int h = static_cast<int>(std::floor(factor * height / side)) * side;
    int w = static_cast<int>(std::floor(factor * width / side)) * side;
    if (h == 0 && w == 0) std::exit(1);
    if (h == 0) {
        h = side;
        w = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * side, tokens * side);
    } else if (w == 0) {
        w = side;
        h = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * side, tokens * side);
    }
    return {w, h};
}

constexpr int kFrameTokens = 140;

// One clip item: a text, and where it holds a clip, the scene, the text
// before it and the text after.
struct VideoItem {
    const char* name;
    const char* scene;   // nullptr for a text alone
    const char* before;
    const char* after;
};

constexpr VideoItem kVideoItems[] = {
    {"red_ball", "red_ball", "", ""},
    {"blue_square", "blue_square", "", ""},
    {"green_stripes", "green_stripes", "", ""},
    {"black_line", "black_line", "", ""},
    {"captioned", "red_ball", "title: none | text: A clip of ", " rolling across white."},
    {"query_red_ball", nullptr, "task: search result | query: a red ball moving across", ""},
    {"query_blue_square", nullptr, "task: search result | query: a blue square growing on yellow", ""},
    {"query_green_stripes", nullptr, "task: search result | query: green and white stripes scrolling", ""},
    {"query_black_line", nullptr, "task: search result | query: a black line sweeping across white", ""},
};

const bllm::fixtures::video::Scene* video_scene_named(const char* name) {
    for (const auto& s : bllm::fixtures::video::kScenes) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

// A frame of `source`'s size, stretched by llama.cpp's resampler to the
// grid Google's rule gives it.
clip_image_u8 stretched(const clip_image_u8& source, resize_algo algo) {
    const clip_image_size from = source.get_size();
    const clip_image_size size = google_size(from.width, from.height, 16, 3, kFrameTokens);
    clip_image_u8 frame;
    llama_image::img_tool::resize(source, frame, size, algo, PAD_NONE);
    return frame;
}

// Frame `t` of `scene`, stretched to its grid.
clip_image_u8 frame_of(const bllm::fixtures::video::Scene& scene, std::uint32_t t, resize_algo algo) {
    clip_image_u8 source;
    source.set_size({static_cast<int>(scene.width), static_cast<int>(scene.height)}, false);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(scene.width) * scene.height * 3);
    for (std::uint32_t y = 0; y < scene.height; ++y) {
        for (std::uint32_t x = 0; x < scene.width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                bytes[(static_cast<std::size_t>(y) * scene.width + x) * 3 + c] =
                    bllm::fixtures::video::pixel(scene, t, x, y, c);
            }
        }
    }
    source.cpy_buf(bytes);
    return stretched(source, algo);
}

// A frame's rows from the vision encoder: the frame, already at its grid,
// handed over as an image, which llama.cpp's preprocessor leaves at that size.
std::vector<float> frame_rows(mtmd_context* vision, const clip_image_u8& frame, std::uint32_t& rows) {
    const clip_image_size size = frame.get_size();
    std::vector<unsigned char> rgb(frame.get_ro_buf().begin(), frame.get_ro_buf().end());
    mtmd_bitmap* bitmap = mtmd_bitmap_init(size.width, size.height, rgb.data());
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[] = {bitmap};
    const char* marker = mtmd_get_marker(vision);
    mtmd_input_text text{marker, std::strlen(marker), false, true};
    if (mtmd_tokenize(vision, chunks, &text, bitmaps, 1) != 0) std::exit(1);
    std::vector<float> out;
    rows = 0;
    for (std::size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(vision, chunk) != 0) std::exit(1);
        rows = static_cast<std::uint32_t>(mtmd_input_chunk_get_n_tokens(chunk));
        const float* embd = mtmd_get_output_embd(vision);
        out.assign(embd, embd + static_cast<std::size_t>(rows) * 512);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    // Left at its size: its grid's 48-pixel cells, one row each.
    if (rows != static_cast<std::uint32_t>((size.width / 48) * (size.height / 48))) {
        std::fprintf(stderr, "reference_vision: a %dx%d frame made %u rows\n", size.width, size.height, rows);
        std::exit(1);
    }
    return out;
}

struct VideoEmbedding {
    std::vector<llama_token> before;   // the ends and the text before the clip
    std::uint32_t frames = 0;
    std::uint32_t rows = 0;            // each frame's
    llama_token frame_begin = 0;
    llama_token frame_end = 0;
    std::vector<llama_token> after;
    std::vector<float> vector;
};

llama_token one_token(const llama_vocab* vocab, const char* text) {
    const std::vector<llama_token> t = tokens_of(vocab, text);
    if (t.size() != 1) std::exit(1);
    return t[0];
}

// The text before, each frame's rows between `<|image>` and `<image|>`, the
// text after, the file's ends around them, embedded as one batch.
VideoEmbedding embed_frames(const Backend& b, const char* before, const std::vector<clip_image_u8>& stretched_frames,
                            const char* after) {
    const llama_vocab* vocab = llama_model_get_vocab(b.model);
    VideoEmbedding e;
    e.frame_begin = one_token(vocab, "<|image>");
    e.frame_end = one_token(vocab, "<image|>");
    if (llama_vocab_get_add_bos(vocab)) e.before.push_back(llama_vocab_bos(vocab));
    for (const llama_token t : tokens_of(vocab, before)) e.before.push_back(t);
    std::vector<std::vector<float>> frames;
    for (const clip_image_u8& frame : stretched_frames) {
        std::uint32_t rows = 0;
        frames.push_back(frame_rows(b.vision, frame, rows));
        if (e.rows != 0 && rows != e.rows) std::exit(1);
        e.rows = rows;
    }
    e.frames = static_cast<std::uint32_t>(frames.size());
    for (const llama_token t : tokens_of(vocab, after)) e.after.push_back(t);
    if (llama_vocab_get_add_eos(vocab)) e.after.push_back(llama_vocab_eos(vocab));

    llama_memory_clear(llama_get_memory(b.context), true);
    llama_batch_ext* batch = llama_batch_ext_init(b.context);
    llama_pos pos = 0;
    const auto place = [&](int32_t index) {
        if (index < 0) std::exit(1);
        llama_batch_ext_set_pos(batch, index, &pos);
        llama_batch_ext_set_output_embd(batch, index, true);
        ++pos;
    };
    for (const llama_token t : e.before) place(llama_batch_ext_add_token(batch, 0, t));
    for (const std::vector<float>& rows : frames) {
        place(llama_batch_ext_add_token(batch, 0, e.frame_begin));
        for (std::uint32_t r = 0; r < e.rows; ++r) {
            place(llama_batch_ext_add_embd(batch, 0, llama_embd{rows.data() + static_cast<std::size_t>(r) * 512, 1, 512}));
        }
        place(llama_batch_ext_add_token(batch, 0, e.frame_end));
    }
    for (const llama_token t : e.after) place(llama_batch_ext_add_token(batch, 0, t));
    if (llama_process(b.context, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) std::exit(1);
    const int n = llama_model_n_embd_out(b.model);
    const float* pooled = llama_get_embeddings_seq(b.context, 0);
    if (pooled == nullptr) std::exit(1);
    e.vector.assign(pooled, pooled + n);
    llama_batch_ext_free(batch);
    double sum = 0;
    for (const float v : e.vector) sum += static_cast<double>(v) * v;
    const float scale = sum > 0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0f;
    for (float& v : e.vector) v *= scale;
    return e;
}

VideoEmbedding embed_video_item(const Backend& b, const VideoItem& item, resize_algo algo) {
    std::vector<clip_image_u8> frames;
    if (item.scene != nullptr) {
        const auto& scene = *video_scene_named(item.scene);
        for (std::uint32_t t = 0; t < scene.frames; ++t) frames.push_back(frame_of(scene, t, algo));
    }
    return embed_frames(b, item.before, frames, item.after);
}

// A binary PPM, P6 at 255, as FFmpeg writes one.
clip_image_u8 read_ppm(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "reference_vision: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    int width = 0, height = 0, most = 0;
    if (std::fscanf(f, "P6 %d %d %d", &width, &height, &most) != 3 || most != 255 || std::fgetc(f) == EOF) {
        std::fprintf(stderr, "reference_vision: %s is not a binary PPM at 255\n", path.c_str());
        std::exit(1);
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * 3);
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) std::exit(1);
    std::fclose(f);
    clip_image_u8 image;
    image.set_size({width, height}, false);
    image.cpy_buf(bytes);
    return image;
}

// frames: a manifest of lines `clip <name> <frame.ppm> ...` and `query
// <text>`, each query's text as written, its task prompt included, and each
// clip alone, as the page embeds `<|video|>`. Writes, as JSON, each query's
// score against each clip on Metal and on the CPU.
int frames_mode(const char* model, const char* patched, const char* manifest) {
    llama_backend_init();
    clip_ctx* hparams_of = load(patched);
    const resize_algo algo = clip_get_hparams(hparams_of)->image_resize_algo;
    const Backend cpu = open_backend(model, patched, true);
    const Backend metal = open_backend(model, patched, false);
    std::FILE* f = std::fopen(manifest, "r");
    if (f == nullptr) std::exit(1);
    std::vector<std::pair<std::string, std::vector<clip_image_u8>>> clips;
    std::vector<std::string> queries;
    char line[8192];
    while (std::fgets(line, sizeof line, f) != nullptr) {
        std::string text(line);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        if (text.rfind("query ", 0) == 0) {
            queries.push_back(text.substr(6));
        } else if (text.rfind("clip ", 0) == 0) {
            std::vector<std::string> words;
            std::size_t at = 5;
            while (at < text.size()) {
                const std::size_t space = text.find(' ', at);
                words.push_back(text.substr(at, space == std::string::npos ? std::string::npos : space - at));
                at = space == std::string::npos ? text.size() : space + 1;
            }
            std::vector<clip_image_u8> frames;
            for (std::size_t i = 1; i < words.size(); ++i) frames.push_back(stretched(read_ppm(words[i]), algo));
            clips.emplace_back(words[0], std::move(frames));
        }
    }
    std::fclose(f);
    const auto scores = [&](const Backend& b) {
        std::vector<std::vector<float>> clip_vectors, query_vectors;
        for (const auto& [name, frames] : clips) clip_vectors.push_back(embed_frames(b, "", frames, "").vector);
        for (const std::string& q : queries) query_vectors.push_back(embed_frames(b, q.c_str(), {}, "").vector);
        std::vector<std::vector<double>> out;
        for (const auto& q : query_vectors) {
            std::vector<double> row;
            for (const auto& c : clip_vectors) {
                double dot = 0;
                for (std::size_t i = 0; i < q.size(); ++i) dot += static_cast<double>(q[i]) * c[i];
                row.push_back(dot);
            }
            out.push_back(row);
        }
        return out;
    };
    const auto on_metal = scores(metal), on_cpu = scores(cpu);
    std::printf("{\n  \"clips\": [");
    for (std::size_t c = 0; c < clips.size(); ++c) std::printf("%s\"%s\"", c ? ", " : "", clips[c].first.c_str());
    std::printf("],\n  \"queries\": [\n");
    for (std::size_t q = 0; q < queries.size(); ++q) {
        std::printf("    {\"query\": \"%s\", \"metal\": [", queries[q].c_str());
        for (std::size_t c = 0; c < clips.size(); ++c) std::printf("%s%.6f", c ? ", " : "", on_metal[q][c]);
        std::printf("], \"cpu\": [");
        for (std::size_t c = 0; c < clips.size(); ++c) std::printf("%s%.6f", c ? ", " : "", on_cpu[q][c]);
        std::printf("]}%s\n", q + 1 < queries.size() ? "," : "");
    }
    std::printf("  ]\n}\n");
    for (const Backend* b : {&cpu, &metal}) {
        mtmd_free(b->vision);
        llama_free(b->context);
        llama_model_free(b->model);
    }
    clip_free(hparams_of);
    llama_backend_free();
    return 0;
}

int video(const char* model, const char* patched, const char* commit, const char* model_sha, const char* mmproj_sha) {
    llama_backend_init();
    clip_ctx* hparams_of = load(patched);
    const resize_algo algo = clip_get_hparams(hparams_of)->image_resize_algo;
    const Backend cpu = open_backend(model, patched, true);
    const Backend metal = open_backend(model, patched, false);

    std::printf("// Generated by tools/make_reference_vision.sh. Do not edit.\n");
    std::printf("// llama.cpp %s, CPU and Metal, the model of SHA-256 %s and the mmproj of SHA-256\n", commit, model_sha);
    std::printf("// %s with clip.use_gelu set. Each clip's frames, one a second, stretched by\n", mmproj_sha);
    std::printf("// llama.cpp's resampler to the grid Google's rule gives at %d tokens, each frame's\n", kFrameTokens);
    std::printf("// rows between <|image> and <image|>, as EmbeddingGemma 2's processor writes a video.\n");
    std::printf("// Each item: its name, its scene or none, its text before and after the clip, the\n");
    std::printf("// tokens before the clip (the ends among them), its frames, each frame's rows, the\n");
    std::printf("// tokens around each frame, the tokens after, and its unit-length embedding on Metal\n");
    std::printf("// and on the CPU.\n");
    std::printf("inline const VideoReference kVideoReferences[] = {\n");
    for (const VideoItem& item : kVideoItems) {
        const VideoEmbedding m = embed_video_item(metal, item, algo), c = embed_video_item(cpu, item, algo);
        if (m.before != c.before || m.after != c.after || m.rows != c.rows || m.frames != c.frames) {
            std::fprintf(stderr, "%s: the backends built different sequences\n", item.name);
            return 1;
        }
        std::printf("    {\"%s\", \"%s\", R\"REF(%s)REF\", R\"REF(%s)REF\", ", item.name,
                    item.scene != nullptr ? item.scene : "", item.before, item.after);
        print_tokens(m.before);
        std::printf(", %u, %u, %d, %d, ", m.frames, m.rows, m.frame_begin, m.frame_end);
        print_tokens(m.after);
        std::printf(", ");
        print_floats(m.vector);
        std::printf(", ");
        print_floats(c.vector);
        std::printf("},\n");
        std::fprintf(stderr, "%s: %u frames of %u rows\n", item.name, m.frames, m.rows);
    }
    std::printf("};\n");
    for (const Backend* b : {&cpu, &metal}) {
        mtmd_free(b->vision);
        llama_free(b->context);
        llama_model_free(b->model);
    }
    clip_free(hparams_of);
    llama_backend_free();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "prepare") return prepare(argv[2], argv[3], argv[4]);
    if (argc == 8 && std::string(argv[1]) == "embed") return embed(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    if (argc == 7 && std::string(argv[1]) == "video") return video(argv[2], argv[3], argv[4], argv[5], argv[6]);
    if (argc == 5 && std::string(argv[1]) == "frames") return frames_mode(argv[2], argv[3], argv[4]);
    std::fprintf(stderr, "usage: reference_vision prepare <mmproj.gguf> <llama.cpp commit> <mmproj sha256>\n"
                         "       reference_vision embed <model.gguf> <patched mmproj> <published mmproj> <commit>"
                         " <model sha256> <mmproj sha256>\n"
                         "       reference_vision video <model.gguf> <patched mmproj> <commit> <model sha256>"
                         " <mmproj sha256>\n"
                         "       reference_vision frames <model.gguf> <patched mmproj> <manifest>\n");
    return 2;
}
