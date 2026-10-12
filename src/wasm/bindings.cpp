// The only Emscripten-aware translation unit in this repository.
//
// Its job is translation, not behavior: it starts the device request, forwards
// the result to JavaScript as JSON, and holds the session's state across
// crossings. Product behavior belongs in src/core.
//
// The session's state is what outlives one crossing, held in one place here,
// at the boundary, because the module has one page and one device:
//   - the device the self-check acquired and checked, kept rather than
//     released, so the model loads onto the device that was checked;
//   - a load: the index read from the prefix begin was given, the plan made
//     for it, the Upload carrying it out (residency/upload.h), and one chunk
//     buffer, allocated at begin and reused for every chunk (WASM.1); and,
//     made at begin from the same prefix, the description, the tokenizer and
//     the stop set — or, for a model that encodes, its ends (runtime/ends.h);
//   - the loaded model: its Upload, which owns the weights' buffers, and the
//     generator over it (runtime/generator.h), which owns the tokenizer, the
//     program and the cache — or, for a model that encodes, the embedder
//     (runtime/embedder.h), which owns the tokenizer and the program; either
//     is released first, since its program borrows the Upload.
// Every decision about that state is core's; this file only holds it and
// translates.
//
// The device the self-check requests reports its uncaptured errors and its
// loss as they happen (gpu/device.h's DeviceEvents), each crossing once to
// globalThis.bllmOnDeviceEvent as JSON: {"kind":"uncaptured","type",
// "message"} with the type validation, out-of-memory or internal, or
// {"kind":"lost","reason","message"} with the reason unknown, destroyed,
// callback-cancelled or failed-creation. They are what no call returns — a
// device lost between turns, an error outside every scope — so the page can
// show and keep them (web/diagnostics.js). The browser's event loop runs the
// sink; it holds no state.
//
// Diagnostic builds only, for a page served with ?profile from the
// diagnostic site (web/dev/step_profile.js), never the deployed one's:
//   bllm_run_profile_check()
//     The self-check, on a device asked for timestamp queries
//     (gpu/device.h's DiagnosticRequest); the failure named when the
//     adapter grants none.
//   bllm_observe_steps(on)
//     Sets or clears the loaded runtime's step observer
//     (runtime/runtime.h), between turns. The observer appends each step's
//     record — prefill, position, tokens, its GPU begin and end, and
//     emscripten_get_now() at the report — to a buffer reserved at the
//     context offered plus a prefill step per block, so no step allocates;
//     the turn's records cross once, as doubles, with its result: GPU times
//     exact below 2^53 ns, about 104 days. A step's only cost at the
//     boundary is the clock's import call (WASM.2).
//
// Contract 11, the boundary: this file and web/worker.js are the only two
// places JavaScript and C++ meet. The crossings are preflight a header prefix;
// begin a load, load a chunk of the file, and finish the load (web/load.js);
// generate from a rendered prompt and the turn's policy; cancel; and, for a
// model that encodes, embed a text, with an image where its companion has
// loaded. A
// prompt the tokenizer refuses as too long to encode is answered with code
// "prompt-too-large", its bytes — normalized, or raw where it was refused
// before normalizing — and kMaxEncodeBytes (tokenizer/tokenizer.h), which the
// page shows by name: dropping old messages does not shorten a message that
// is itself too large. Text goes
// back one crossing per piece the reply completes (WASM.2).
//
// The model's load policy (policy/policy.h) crosses with preflight and with
// begin, the same at both, so Fit is judged against the plan the load makes:
// the cache precision as its index in CachePrecision, or −1; the memory
// budget as a double, an exact integer, or NaN; the rollback reserve, or
// 2^32 − 1; and the stop texts as one buffer of their UTF-8 bytes with an
// array of their lengths, so a text may hold any byte. −1, NaN and 2^32 − 1
// stand for the field unset — an unmeasured model's, which runs on the
// default.
//
//   bllm_preflight(request, prefix, prefix_length, file_size, limits...,
//                  policy...)
//     Reads the index from `prefix` and answers with how far this build
//     takes the model, judged under the model's load policy — its tokenizer
//     loaded from the prefix and its stop set resolved, each failure a Run
//     blocker (preflight.h) — or with the bytes the reader still needs.
//
//   bllm_load_begin(request, prefix, prefix_length, file_size, confirmed,
//                   confirmed_count, max_chunk, policy...)
//     Reads the index from `prefix` — the bytes preflight read — describes
//     the model, plans it for the kept device, and begins the Upload with the
//     duplicates the page confirmed (web/duplicates.js). Then, while the
//     device creates the buffers, it loads the tokenizer from the prefix,
//     where its vocabulary and merges lie (preflight.h's load_tokenizer), and
//     resolves the stop set from the file and the policy's stop texts
//     (runtime/stops.h) — or, for a model that encodes, its ends
//     (runtime/ends.h), since it never samples: 20 to 56 ms of the CPU's
//     for the listed models, overlapped with the GPU's work rather than
//     ahead of it. Answers once
//     the device holds the buffers: the chunk buffer's address and size, or
//     the failure, named — the tokenizer's or the stop set's included, the
//     buffers then released. A model already loaded is released first — its
//     generator, so a turn running finishes cancelled, then its buffers — so
//     two models are never on the device together.
//   bllm_load_chunk(request, file_offset, length)
//     The page has copied `length` bytes at `file_offset` into the chunk
//     buffer. Answers when the page may send the next (residency/upload.h).
//   bllm_load_finish(request)
//     Once every write has completed, shown by the witness: builds the
//     architecture's graph over the plan, builds its program — every
//     pipeline compiled together, asynchronously (kernels/program.h) —
//     reading back the draw's record, and makes the cache at the context
//     offered and the generator. Answers then, with the context offered, or
//     with the failure that stopped any of it, named. For a model that
//     encodes: the encoder's graph, its program reading back the step's
//     embedding, and the embedder; answering with the context offered and
//     the embedding's length.
// A load already begun is refused, by name, until it finishes or fails; a
// chunk or finish without a load is refused the same way.
//
//   bllm_generate(request, text, text_length, sampled, temperature, top_k,
//                 top_p, min_p, seed, max_tokens)
//     Starts a turn over `text`, the rendered conversation as UTF-8 bytes in
//     the module's memory, which the page frees once the call returns: the
//     generator encodes them before it does. Sampling settings are the
//     model's for the turn's mode when `sampled` is 1, else the defaults;
//     the seed is a double, an exact integer below 2^53; max_tokens 2^32 − 1
//     asks for no limit but the context. A refusal answers at once, named:
//     no model loaded, a turn running, text that does not encode, a prompt
//     longer than the context — with code "prompt-too-long" and both counts,
//     so the page can drop its oldest messages and render again
//     (logical-overview.md) — or a setting out of range. Otherwise each
//     piece of text goes back through bllm_text(request, pointer, length), a
//     view of the module's memory valid for that call, and the turn answers
//     once: its stop reason — "stop", "limit", "context" or "cancelled" —
//     the reply's tokens, the prompt's, and how many of those the cache
//     already held; or its failure, named: the draw's logits not finite, a
//     step's failure with WebGPU's message, or the device lost, after which
//     every turn is refused.
//   bllm_cancel(request)
//     Ends the turn `request` started, at its next report; its answer then
//     comes as any turn's, stop reason "cancelled". Answers at once whether
//     that turn was running.
// One turn at a time: the page waits for a turn's answer before the next.
// A model that encodes refuses generate, by name.
//
//   bllm_embed(request, text, text_length, image)
//     Starts an encode of `text`, UTF-8 bytes in the module's memory which
//     the page frees once the call returns: the embedder encodes them before
//     it does. A refusal answers at once, named: no model loaded, a model
//     that generates, an encode running, text that does not encode, the
//     device lost, or text longer than the context — with code
//     "text-too-long" and both counts, ends included, so the page can cut
//     it and embed the pieces. Otherwise the vector goes back once through
//     bllm_vector(request, pointer, length), a Float32Array view of the
//     module's memory valid for that call — the vector's one crossing
//     (WASM.2) — and the encode answers with the tokens the text made; or
//     with its failure, named: the vector not finite, a step's failure with
//     WebGPU's message, the device lost, or the encode cancelled by a new
//     load. One encode at a time. With `image` 1, the image the slot holds
//     goes where `text` writes <|image|>, once (runtime/vision.h); refused,
//     by name, when no image is prepared or the placeholder is not there
//     once — and with `image` 0, when it is there at all.
//
// A model that encodes may have a companion: the vision encoder, in a file
// of its own, read in part (residency/file_ranges.h) and loaded beside it.
// Its ranges cross as pairs of doubles, [begin, end), each an exact
// integer; its image budget as the least and most tokens an image becomes;
// the memory budget as the model's load policy writes it, NaN for unset.
//   bllm_companion_preflight(request, resident, resident_length, file_size,
//                            ranges, range_count, least, most, limits...,
//                            memory_budget)
//     Reads the companion's index and answers with how far this build takes
//     it beside the loaded model (preflight.h's preflight_companion), or
//     with the bytes the reader still needs. Answers unreadable, by name,
//     when no model that encodes is loaded.
//   bllm_companion_load_begin(request, prefix, prefix_length, file_size,
//                             ranges, range_count, max_chunk, least, most,
//                             memory_budget)
//   bllm_companion_load_chunk(request, file_offset, length)
//   bllm_companion_load_finish(request)
//     As a model's load, over the ranges alone: finish builds the vision
//     encoder's graph and program, links its rows into the text program's
//     hidden buffer, and attaches the vision stage to the embedder,
//     answering with the most tokens an image becomes. A companion loaded
//     before is released at begin; a new model's load releases it too.
//   bllm_image_begin(request, width, height)
//     Starts an image in the vision stage's one slot, discarding what it
//     held: answers with its grid — tokens, and the size it is resampled
//     to — or its refusal, named.
//   bllm_image_rows(request, first_row, pixels, length)
//     The image's next band, whole RGBA rows from `first_row`, which the
//     page frees once the call returns. Answers whether the image is ready
//     to embed, or the band's refusal.
//
// The companion crossings take a kind after the request, 0 image or 1
// audio (runtime/audio.h), each kind's companion held and released apart;
// one loading at a time. An audio companion's finish runs its setup step
// once and answers with the most samples a clip may hold. Then:
//   bllm_audio_prepare(request, samples, count)
//     A clip's samples, f32 at 16 kHz mono, which the page frees once the
//     call returns: its spectrogram computed into the audio stage's slot.
//     Answers with its frames and rows, or its refusal, named — too short,
//     too long, or no audio encoder loaded.
//   bllm_embed's `image` becomes `media`: 0 none, 1 the image stage's, 2 the
//   audio stage's, 3 the video stage's.
//
// Video, designed (runtime/video.h): no companion of its own. An image
// companion whose encoder names video tokens (arch/media_encoder.h) attaches
// a video stage beside its image stage at finish, the two sharing the
// encoder's program; its finish answers as an image companion's does, and
// whether video is attached. Then:
//   bllm_video_plan(request, width, height, duration)
//     What a video of that size and duration becomes, changing nothing:
//     its frames at 1 a second, the grid each frame takes — its tokens and
//     its size — and the frames a clip holds; or the refusal, named — a
//     frame size the encoder cannot take, or no video stage attached.
//   bllm_video_begin(request, width, height, frames)
//     Starts a clip of `frames` frames in the video stage's free slot set,
//     discarding what that set held; answers with its tokens a frame, or
//     the refusal — too many frames or none, the set still in use, the
//     size's.
//   bllm_video_rows(request, frame, first_row, pixels, length)
//     Frame `frame`'s next band, whole RGBA rows from `first_row`, which the
//     page frees once the call returns; answers whether the clip is ready,
//     as bllm_image_rows answers for an image, or the band's refusal.
//
// A diagnostic build adds the check of a finished load, the same three
// crossings over the file streamed a second time (residency/upload_check.h);
// the clean build compiles none of them, so its module has no such exports:
//   bllm_check_begin(request, max_chunk)
//     Begins checking the loaded model; answers with the chunk buffer.
//   bllm_check_chunk(request, file_offset, length)
//     Answers once the chunk's ranges are compared.
//   bllm_check_finish(request)
//     Answers with every mismatch, by buffer, offset and tensor, or the
//     failure that stopped the check.

#include <emscripten.h>
#include <emscripten/eventloop.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/arch/media_encoder.h"
#include "core/cache/kv.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/gpu/device.h"
#include "core/preflight/preflight.h"
#include "core/kernels/program.h"
#include "core/residency/upload.h"
#include "core/run_guard.h"
#include "core/runtime/audio.h"
#include "core/runtime/embedder.h"
#include "core/runtime/ends.h"
#include "core/runtime/generator.h"
#include "core/runtime/stops.h"
#include "core/runtime/vision.h"
#include "core/sampler/sampler.h"
#include "core/tokenizer/tokenizer.h"
#include "core/gpu/self_check.h"
#include "core/diagnostics.h"
#if BLLM_DIAGNOSTICS_ENABLED
#include "core/gpu/readback_bench.h"
#include "core/residency/upload_check.h"
#endif

namespace {

// Element count for the toolchain self-check. Large enough to span many
// workgroups at the shader's 64-wide group, small enough to stay far inside
// any device's buffer limits.
constexpr std::size_t kSelfCheckElements = 4096;

// A device that has not answered in this long is not going to. WebGPU gives
// no cancellation, so this bounds how long the module can appear busy, not
// how long the request actually runs.
constexpr double kRunTimeoutMs = 15000.0;

// Round trips measured by the spike. Enough for a stable median.
constexpr std::size_t kBenchIterations = 200;

// Generations ride through WebGPU's void* userdata. A 32-bit generation is
// used so this holds on wasm32, where a pointer is 4 bytes.
static_assert(sizeof(std::uint32_t) <= sizeof(void*),
              "generation must fit in a userdata pointer");

void* to_userdata(std::uint32_t generation) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(generation));
}

std::uint32_t to_generation(void* userdata) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(userdata));
}

// Escapes only what a JSON string requires. Adapter descriptions come from the
// driver, so they are not assumed to be free of quotes or backslashes.
std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (const char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    // Cast before formatting: a plain char would sign-extend
                    // if this branch ever widened past the control range.
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// A device event, as JSON, to the worker (the header above).
EM_JS(void, bllm_device_event, (const char* json), {
    const text = UTF8ToString(json);
    if (typeof globalThis.bllmOnDeviceEvent === 'function') {
        globalThis.bllmOnDeviceEvent(JSON.parse(text));
    } else {
        console.error('bllm: no device event handler registered', text);
    }
});

const char* error_type_name(int type) {
    switch (type) {
        case WGPUErrorType_Validation: return "validation";
        case WGPUErrorType_OutOfMemory: return "out-of-memory";
        case WGPUErrorType_Internal: return "internal";
        default: return "unknown";
    }
}

const char* lost_reason_name(int reason) {
    switch (reason) {
        case WGPUDeviceLostReason_Destroyed: return "destroyed";
        case WGPUDeviceLostReason_CallbackCancelled: return "callback-cancelled";
        case WGPUDeviceLostReason_FailedCreation: return "failed-creation";
        default: return "unknown";
    }
}

void on_device_event(const bllm::gpu::DeviceEvent& event, void*) {
    using Kind = bllm::gpu::DeviceEvent::Kind;
    const std::string message = json_escape(std::string(event.message));
    const std::string json = event.kind == Kind::Uncaptured
        ? std::string("{\"kind\":\"uncaptured\",\"type\":\"") + error_type_name(event.code) +
              "\",\"message\":\"" + message + "\"}"
        : std::string("{\"kind\":\"lost\",\"reason\":\"") + lost_reason_name(event.code) +
              "\",\"message\":\"" + message + "\"}";
    bllm_device_event(json.c_str());
}

constexpr bllm::gpu::DeviceEvents kDeviceEvents{.sink = on_device_event, .userdata = nullptr};

// Delivers a JSON result to the page. Defined in JS because the page owns
// presentation; C++ owns only what happened.
EM_JS(void, bllm_deliver, (const char* json), {
    const text = UTF8ToString(json);
    if (typeof globalThis.bllmOnResult === 'function') {
        globalThis.bllmOnResult(JSON.parse(text));
    } else {
        console.error('bllm: no result handler registered', text);
    }
});

void report_failure(const std::string& stage, const std::string& error) {
    const std::string json = std::string("{\"ok\":false,\"stage\":\"") + stage +
                             "\",\"error\":\"" + json_escape(error) + "\"}";
    bllm_deliver(json.c_str());
}

// Answers one request from the worker. Requests carry an id so an answer
// that arrives later, from a callback, still reaches the request it belongs to.
EM_JS(void, bllm_reply, (std::uint32_t request, const char* json), {
    globalThis.bllmOnReply(request, JSON.parse(UTF8ToString(json)));
});

std::string json_string(std::string_view text) {
    return "\"" + json_escape(std::string(text)) + "\"";
}

// A token's text, looked up by the id stored under `id_key`, as JSON; null if
// the file declares no such token.
std::string token_json(bllm::gguf::ByteSource& source, const bllm::gguf::TensorIndex& index,
                       std::string_view id_key) {
    std::uint32_t id = 0;
    bllm::gguf::ArrayLocation tokens{};
    std::string text;
    if (index.read_u32(id_key, id) != bllm::gguf::MetadataError::Ok ||
        index.read_array("tokenizer.ggml.tokens", tokens) != bllm::gguf::MetadataError::Ok ||
        bllm::gguf::read_string_element(source, tokens, id, text).error !=
            bllm::gguf::ReadError::Ok) {
        return "null";
    }
    return json_string(text);
}

// What the page needs to render a conversation for this model: the chat
// template the file carries, and the text of the tokens it refers to.
std::string chat_json(bllm::gguf::ByteSource& source, const bllm::gguf::TensorIndex& index) {
    std::string_view chat_template;
    const bool has_template = index.read_string("tokenizer.chat_template", chat_template) ==
                              bllm::gguf::MetadataError::Ok;
    return "{\"template\":" + (has_template ? json_string(chat_template) : std::string("null")) +
           ",\"bosToken\":" + token_json(source, index, "tokenizer.ggml.bos_token_id") +
           ",\"eosToken\":" + token_json(source, index, "tokenizer.ggml.eos_token_id") + "}";
}

// The plan's duplicate candidates, with both byte ranges, for the page to
// compare before a load (web/duplicates.js).
std::string duplicates_json(const std::vector<bllm::preflight::DuplicateCandidate>& duplicates) {
    std::string json = "[";
    for (std::size_t i = 0; i < duplicates.size(); ++i) {
        const auto& d = duplicates[i];
        json += i == 0 ? "{" : ",{";
        json += "\"tensor\":" + std::to_string(static_cast<std::uint32_t>(d.tensor)) +
                ",\"copies\":" + std::to_string(static_cast<std::uint32_t>(d.copies)) +
                ",\"offset\":" + std::to_string(d.offset) +
                ",\"copiesOffset\":" + std::to_string(d.copies_offset) +
                ",\"length\":" + std::to_string(d.length) + "}";
    }
    return json + "]";
}

// What the residency plan found, as JSON; null if the model did not fit.
std::string fit_json(const std::optional<bllm::preflight::FitSummary>& fit) {
    if (!fit) return "null";
    return "{\"weightBytes\":" + std::to_string(fit->weight_bytes) +
           ",\"cacheBytes\":" + std::to_string(fit->cache_bytes) +
           ",\"scratchBytes\":" + std::to_string(fit->scratch_bytes) +
           ",\"totalBytes\":" + std::to_string(fit->total_bytes) +
           ",\"memoryBudget\":" + std::to_string(fit->memory_budget) +
           ",\"contextOffered\":" + std::to_string(fit->context_offered) +
           ",\"trainedContext\":" + std::to_string(fit->trained_context) +
           ",\"bufferCount\":" + std::to_string(fit->buffer_count) +
           ",\"duplicates\":" + duplicates_json(fit->duplicates) + "}";
}

// The preflight answer: bytes the reader still needs, a file that cannot be
// read, or the verdict on a file that can.
std::string preflight_json(bllm::gguf::ByteSource& source, const bllm::gguf::ReadResult& read,
                           const bllm::gguf::TensorIndex& index,
                           const bllm::residency::DeviceLimits& limits, const bllm::policy::LoadPolicy& policy) {
    using bllm::gguf::ReadError;
    if (read.error == ReadError::NeedMoreBytes) {
        return "{\"status\":\"need-bytes\",\"bytesNeeded\":" +
               std::to_string(read.bytes_needed) + "}";
    }
    if (read.error != ReadError::Ok) {
        return "{\"status\":\"unreadable\",\"error\":" +
               json_string(bllm::gguf::to_string(read.error)) + "}";
    }
    const auto verdict = bllm::preflight::preflight(source, index, limits, policy);
    std::string_view architecture;
    const bool named = index.read_string("general.architecture", architecture) ==
                       bllm::gguf::MetadataError::Ok;

    std::string json = "{\"status\":\"read\",\"architecture\":";
    json += named ? json_string(architecture) : "null";
    json += ",\"tensorCount\":" + std::to_string(index.tensors().size());
    json += ",\"chat\":" + chat_json(source, index);
    json += ",\"reached\":" + json_string(bllm::preflight::to_string(verdict.reached()));
    json += ",\"fit\":" + fit_json(verdict.fit);
    json += ",\"blockers\":[";
    for (std::size_t i = 0; i < verdict.blockers.size(); ++i) {
        const auto& b = verdict.blockers[i];
        json += i == 0 ? "" : ",";
        json += "{\"stage\":" + json_string(bllm::preflight::to_string(b.stage)) +
                ",\"detail\":" + json_string(b.detail) + "}";
    }
    return json + "]}";
}

// Serialises runs and identifies late callbacks. Logic lives in core and is
// tested natively; this file only supplies the clock.
bllm::RunGuard& guard() {
    static bllm::RunGuard g;
    return g;
}

std::uint32_t& pending_generation() {
    static std::uint32_t g = 0;
    return g;
}

int& timeout_id() {
    static int id = 0;
    return id;
}

void disarm_timeout() {
    if (timeout_id() != 0) {
        emscripten_clear_timeout(timeout_id());
        timeout_id() = 0;
    }
}

// The session's state across crossings (see the top of this file).
struct Load {
    std::uint32_t begin_request = 0;
    std::unique_ptr<bllm::residency::Upload> upload;   // null until begin's answer
    std::vector<std::byte> chunk;                      // the chunk buffer
    std::uint64_t file_size = 0;
    bool settled = false;   // finished or failed: the next begin replaces it
    // What begin made from the prefix, for finish: the description, the
    // tokenizer and the stop set, or why the tokenizer could not be made.
    bllm::model::ModelDescription description;
    const bllm::arch::Architecture* architecture = nullptr;
    bllm::policy::CachePrecision precision = bllm::policy::CachePrecision::F16;
    std::unique_ptr<bllm::tokenizer::Tokenizer> tokenizer;
    bllm::runtime::StopSet stops;   // a generator's
    bllm::runtime::Ends ends;       // an embedder's
    std::string tokenizer_failure;
    // Begin answers once both the buffers' answer and the tokenizer are in.
    bool prepared = false;
    bool ready = false;
    std::string ready_failure;   // the buffers' failure, when `ready` and no upload
};

#if BLLM_DIAGNOSTICS_ENABLED
// A check of the loaded model, which must outlive it (upload_check.h).
struct Check {
    std::unique_ptr<bllm::residency::UploadCheck> check;
    std::vector<std::byte> chunk;
};
#endif

// A companion's load (runtime/media_stage.h): its kind, its Upload of the
// ranges it reads, one chunk buffer, and what begin made of its header for
// finish.
struct CompanionLoad {
    bllm::arch::MediaKind kind = bllm::arch::MediaKind::Image;
    std::uint32_t begin_request = 0;
    std::unique_ptr<bllm::residency::Upload> upload;
    std::vector<std::byte> chunk;
    bool settled = false;
    bllm::model::ModelDescription description;
    const bllm::arch::MediaEncoder* encoder = nullptr;
    bllm::media::ImageBudget budget{};
    // An audio companion's: the most rows a clip makes, its plan's; its
    // step's launches, built once its setup has run; and the setup's
    // program, held until the step's is built.
    std::uint32_t most_rows = 0;
    std::vector<bllm::kernels::Launch> step_launches;
    std::unique_ptr<bllm::kernels::Program> setup;
};

constexpr std::size_t kMediaKinds = 2;

constexpr std::size_t index_of(bllm::arch::MediaKind kind) noexcept { return static_cast<std::size_t>(kind); }

struct Session {
    std::unique_ptr<bllm::gpu::Device> device;          // the checked device, kept
    std::unique_ptr<Load> load;
    std::unique_ptr<bllm::residency::Upload> loaded;    // the model a load finished
    std::uint64_t loaded_file_size = 0;
    // What a companion's verdict pairs against: the loaded model's
    // description and architecture (preflight.h's CompanionHost).
    bllm::model::ModelDescription loaded_description;
    std::string loaded_architecture;
    // The companion loading, of either kind; one at a time.
    std::unique_ptr<CompanionLoad> companion;
    // Each kind's companion a load finished: its buffers, which that kind's
    // stage the embedder holds borrows. Declared before the embedder, so
    // released after it.
    std::array<std::unique_ptr<bllm::residency::Upload>, kMediaKinds> companion_loaded;
    // Over `loaded`, whose buffers their programs borrow: declared after it,
    // so released before it. One or the other, as the model generates or
    // encodes.
    std::unique_ptr<bllm::runtime::Generator> generator;
    std::unique_ptr<bllm::runtime::Embedder> embedder;
    std::uint32_t turn_request = 0;   // the generate whose turn runs, or 0
#if BLLM_DIAGNOSTICS_ENABLED
    std::unique_ptr<Check> check;   // declared after `loaded`, so released before it
    // The running turn's steps, when they are observed (bllm_observe_steps):
    // reserved at load for the most steps a turn runs, so no step allocates.
    struct StepRecord {
        double prefill, position, tokens, begin_ns, end_ns, report_ms;
    };
    std::vector<StepRecord> steps;
    bool observing = false;
#endif
};

Session& session() {
    static Session s;
    return s;
}

std::string failure_json(std::string_view error, std::string_view subject = {}) {
    return "{\"ok\":false,\"error\":" + json_string(error) + ",\"subject\":" + json_string(subject) + "}";
}

constexpr double kMaxExactInteger = 9007199254740992.0;   // 2^53

bool exact_integer(double v) {
    return v >= 0 && v <= kMaxExactInteger && v == static_cast<double>(static_cast<std::uint64_t>(v));
}

// The model's load policy as the page writes it (see the top of this file):
// a field's stand-in for unset leaves the default. Empty on success, else
// what is wrong with it.
std::string read_load_policy(std::int32_t precision, double memory_budget, std::uint32_t rollback_reserve,
                             const char* stop_bytes, const std::uint32_t* stop_lengths, std::uint32_t stop_count,
                             bllm::policy::LoadPolicy& out) {
    if (precision != -1) {
        if (precision < 0 || precision > static_cast<std::int32_t>(bllm::policy::CachePrecision::Q8_0)) {
            return "the cache precision passed in is not one this build names";
        }
        out.cache_precision = static_cast<bllm::policy::CachePrecision>(precision);
    }
    if (!std::isnan(memory_budget)) {
        if (!exact_integer(memory_budget)) return "the memory budget passed in is not valid";
        out.memory_budget = static_cast<std::uint64_t>(memory_budget);
    }
    if (rollback_reserve != UINT32_MAX) out.rollback_reserve = rollback_reserve;
    std::size_t at = 0;
    for (std::uint32_t i = 0; i < stop_count; ++i) {
        out.stop.emplace_back(stop_bytes + at, stop_lengths[i]);
        at += stop_lengths[i];
    }
    return {};
}

// Begin's answer, once both halves are in: the buffers, which the device
// creates, and the tokenizer and stop set, made meanwhile.
void answer_begin(Load& load) {
    if (!load.prepared || !load.ready) return;
    if (!load.ready_failure.empty() || !load.tokenizer_failure.empty()) {
        load.settled = true;
        load.upload.reset();
        bllm_reply(load.begin_request,
                   failure_json(load.ready_failure.empty() ? load.tokenizer_failure : load.ready_failure).c_str());
        return;
    }
    const std::string json = "{\"ok\":true,\"chunkPointer\":" +
                             std::to_string(reinterpret_cast<std::uintptr_t>(load.chunk.data())) +
                             ",\"chunkBytes\":" + std::to_string(load.chunk.size()) + "}";
    bllm_reply(load.begin_request, json.c_str());
}

// The load an answer belongs to may have been replaced by a later begin;
// the request id carries it, so the answer still reaches its request.
void on_load_ready(std::unique_ptr<bllm::residency::Upload> upload, bllm::residency::UploadError error,
                   const char* subject, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Load* load = session().load.get();
    if (load == nullptr || load->begin_request != request) {
        bllm_reply(request, failure_json("the load was replaced by another").c_str());
        return;
    }
    load->ready = true;
    if (upload == nullptr) {
        const std::string named(bllm::residency::to_string(error));
        load->ready_failure = subject != nullptr && *subject != '\0' ? named + " (" + subject + ")" : named;
    }
    load->upload = std::move(upload);
    answer_begin(*load);
}

void on_load_accepted(bllm::residency::UploadError error, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    if (error != bllm::residency::UploadError::Ok) {
        if (session().load != nullptr) session().load->settled = true;
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    bllm_reply(request, "{\"ok\":true}");
}

// The program is built: the cache at the context offered, the runtime and
// the generator, and finish's answer.
void on_program_built(std::unique_ptr<bllm::kernels::Program> program, bllm::kernels::ProgramError error,
                      std::string_view message, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    Load* load = s.load.get();
    if (load == nullptr || load->settled || s.loaded == nullptr) {
        bllm_reply(request, failure_json("the load was replaced by another").c_str());
        return;
    }
    load->settled = true;
    if (error != bllm::kernels::ProgramError::Ok) {
        s.loaded.reset();
        bllm_reply(request, failure_json("the model's kernels did not build", message).c_str());
        return;
    }
    const bllm::residency::ResidencyPlan& plan = s.loaded->plan();
    bllm::cache::KvCache cache{load->description, plan, load->precision, plan.context_offered};
    auto runtime = std::make_unique<bllm::runtime::Runtime>(std::move(program), std::move(cache), load->stops);
    s.generator = std::make_unique<bllm::runtime::Generator>(std::move(load->tokenizer), std::move(runtime));
#if BLLM_DIAGNOSTICS_ENABLED
    // A turn runs at most a decode step a token of context and a prefill
    // step a block of it.
    s.observing = false;
    s.steps.clear();
    s.steps.reserve(std::size_t{plan.context_offered} + plan.context_offered / bllm::residency::kPrefillBlock + 2);
#endif
    bllm_reply(request, ("{\"ok\":true,\"contextOffered\":" + std::to_string(plan.context_offered) + "}").c_str());
}

// The encoder's program is built: the embedder, and finish's answer.
void on_encoder_built(std::unique_ptr<bllm::kernels::Program> program, bllm::kernels::ProgramError error,
                      std::string_view message, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    Load* load = s.load.get();
    if (load == nullptr || load->settled || s.loaded == nullptr) {
        bllm_reply(request, failure_json("the load was replaced by another").c_str());
        return;
    }
    load->settled = true;
    if (error != bllm::kernels::ProgramError::Ok) {
        s.loaded.reset();
        bllm_reply(request, failure_json("the model's kernels did not build", message).c_str());
        return;
    }
    const std::uint32_t context = s.loaded->plan().context_offered;
    const std::uint32_t dimensions = load->description.encoder->embedding_length;
    s.embedder = std::make_unique<bllm::runtime::Embedder>(std::move(load->tokenizer), std::move(program), load->ends,
                                                           context, dimensions);
    bllm_reply(request, ("{\"ok\":true,\"contextOffered\":" + std::to_string(context) +
                         ",\"embeddingLength\":" + std::to_string(dimensions) + "}")
                            .c_str());
}

// The encoder's graph over the plan, and its program, reading back the
// step's embedding (graph/encoder.h).
void build_encoder(std::uint32_t request, Session& s, Load& load, bllm::arch::EncodeFn graph) {
    const bllm::residency::ResidencyPlan& plan = s.loaded->plan();
    std::vector<bllm::kernels::Launch> launches;
    const bllm::graph::GraphResult built = graph(load.description, plan, launches);
    std::optional<bllm::kernels::Binding> embedding;
    for (const bllm::residency::PlannedScratch& scratch : plan.scratch) {
        if (scratch.purpose == "embedding") {
            embedding = bllm::kernels::Binding{scratch.range.buffer, scratch.range.offset,
                                               std::uint64_t{load.description.encoder->embedding_length} *
                                                   sizeof(float)};
        }
    }
    if (!built.ok() || !embedding) {
        load.settled = true;
        s.loaded.reset();
        bllm_reply(request, failure_json("the model's graph does not build", built.subject).c_str());
        return;
    }
    bllm::kernels::Program::build(*s.loaded, std::move(launches), embedding, on_encoder_built, to_userdata(request));
}

void on_load_finished(bllm::residency::UploadError error, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    Load* load = s.load.get();
    if (error != bllm::residency::UploadError::Ok || load == nullptr) {
        if (load != nullptr) load->settled = true;
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    // The finished Upload owns the model's buffers: it stays as the loaded
    // model. The chunk buffer goes with the load.
    s.loaded = std::move(load->upload);
    s.loaded_file_size = load->file_size;
    // The graph over the plan, and its program, reading back the draw's
    // record; every pipeline compiles together (kernels/program.h).
    if (const bllm::arch::EncodeFn encode = bllm::arch::encoder_graph(*load->architecture); encode != nullptr) {
        build_encoder(request, s, *load, encode);
        return;
    }
    const bllm::residency::ResidencyPlan& plan = s.loaded->plan();
    const bllm::arch::GraphFn graph = bllm::arch::decoder_graph(*load->architecture);
    std::vector<bllm::kernels::Launch> launches;
    const bllm::graph::GraphResult built =
        graph(load->description, plan, *bllm::capability::find_format(plan.cache_type), launches);
    std::optional<bllm::kernels::Binding> sampled;
    for (const bllm::residency::PlannedScratch& scratch : plan.scratch) {
        if (scratch.purpose == "sampled") {
            sampled = bllm::kernels::Binding{scratch.range.buffer, scratch.range.offset,
                                             sizeof(bllm::sampler::SampledRecord)};
        }
    }
    if (!built.ok() || !sampled) {
        load->settled = true;
        s.loaded.reset();
        bllm_reply(request, failure_json("the model's graph does not build", built.subject).c_str());
        return;
    }
    bllm::kernels::Program::build(*s.loaded, std::move(launches), sampled, on_program_built, to_userdata(request));
}

// A piece of a reply's text, read from the module's memory during the call.
// One decoder for every piece, made at the first.
EM_JS(void, bllm_text, (std::uint32_t request, const char* text, std::uint32_t length), {
    globalThis.bllmTextDecoder ??= new TextDecoder();
    globalThis.bllmOnText(request, globalThis.bllmTextDecoder.decode(HEAPU8.subarray(text, text + length)));
});

void on_turn_text(std::string_view text, void* userdata) {
    bllm_text(to_generation(userdata), text.data(), static_cast<std::uint32_t>(text.size()));
}

std::string_view stop_reason(bllm::runtime::TurnEnd end) {
    switch (end) {
        case bllm::runtime::TurnEnd::Stop: return "stop";
        case bllm::runtime::TurnEnd::Limit: return "limit";
        case bllm::runtime::TurnEnd::Context: return "context";
        case bllm::runtime::TurnEnd::Cancelled: return "cancelled";
    }
    return "cancelled";
}

// The observed turn's steps, as a JSON member, emptied for the next turn;
// nothing when the turn was not observed, or in a clean build.
std::string observed_steps(Session& s) {
#if BLLM_DIAGNOSTICS_ENABLED
    if (!s.observing) return {};
    std::string json = ",\"steps\":[";
    for (std::size_t i = 0; i < s.steps.size(); ++i) {
        const auto& r = s.steps[i];
        char row[160];
        std::snprintf(row, sizeof row, "%s[%.0f,%.0f,%.0f,%.0f,%.0f,%.4f]", i == 0 ? "" : ",", r.prefill, r.position,
                      r.tokens, r.begin_ns, r.end_ns, r.report_ms);
        json += row;
    }
    s.steps.clear();
    return json + "]";
#else
    (void)s;
    return {};
#endif
}

void on_turn_end(const bllm::runtime::TurnResult& result, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    if (s.turn_request == request) s.turn_request = 0;
    const std::string steps = observed_steps(s);   // taken whichever way the turn ended
    using bllm::runtime::TurnFailure;
    switch (result.failure) {
        case TurnFailure::None:
            bllm_reply(request, ("{\"ok\":true,\"stopReason\":" + json_string(stop_reason(result.end)) +
                                 ",\"tokens\":" + std::to_string(result.emitted) +
                                 ",\"promptTokens\":" + std::to_string(result.prompt) +
                                 ",\"reusedTokens\":" + std::to_string(result.reused) + steps + "}")
                                    .c_str());
            return;
        case TurnFailure::NonFinite:
            bllm_reply(request, failure_json("the model's logits were not finite",
                                             "after " + std::to_string(result.emitted) + " tokens")
                                    .c_str());
            return;
        case TurnFailure::Step:
            bllm_reply(request, failure_json("a step failed on the GPU", result.message).c_str());
            return;
        case TurnFailure::DeviceLost:
            bllm_reply(request, failure_json("the GPU device was lost", result.message).c_str());
            return;
    }
}

// An encode's vector, read from the module's memory during the call: a view,
// which the runtime copies at once (web/wasm_runtime.js).
EM_JS(void, bllm_vector, (std::uint32_t request, const float* vector, std::uint32_t length), {
    globalThis.bllmOnVector(request, HEAPF32.subarray(vector >> 2, (vector >> 2) + length));
});

void on_embedded(const bllm::runtime::EmbedResult& result, std::span<const float> embedding, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    using bllm::runtime::EmbedError;
    switch (result.error) {
        case EmbedError::Ok:
            bllm_vector(request, embedding.data(), static_cast<std::uint32_t>(embedding.size()));
            bllm_reply(request, ("{\"ok\":true,\"tokens\":" + std::to_string(result.tokens) + "}").c_str());
            return;
        case EmbedError::NonFinite:
            bllm_reply(request, failure_json("the model's embedding was not finite").c_str());
            return;
        case EmbedError::DeviceLost:
            bllm_reply(request, failure_json("the GPU device was lost", result.message).c_str());
            return;
        case EmbedError::Cancelled:
            bllm_reply(request, failure_json("the encode was cancelled by another load").c_str());
            return;
        default:
            bllm_reply(request, failure_json("a step failed on the GPU", result.message).c_str());
            return;
    }
}

// The device arrives inside the result and is released when it goes out of
// scope here, so this reports the device it actually measured rather than
// re-reading whatever is current.
void on_self_check(bllm::gpu::SelfCheckResult result, void* userdata) {
    // A run already closed by the timeout must not report a second result.
    // The device still arrives here and is released as this returns.
    if (!guard().complete(to_generation(userdata))) {
        return;
    }
    disarm_timeout();

    if (!result.ok) {
        report_failure("self_check", result.error);
        return;
    }
    if (result.device == nullptr) {
        report_failure("self_check", "internal: result carried no device");
        return;
    }
    const auto& device = *result.device;
    const auto& info = device.adapter_info();
    const auto& limits = device.limits();
    const auto& maxima = device.adapter_maxima();

    std::string json = "{\"ok\":true,\"adapter\":{";
    json += "\"vendor\":\"" + json_escape(info.vendor) + "\",";
    json += "\"architecture\":\"" + json_escape(info.architecture) + "\",";
    json += "\"device\":\"" + json_escape(info.device) + "\",";
    json += "\"description\":\"" + json_escape(info.description) + "\",";
    json += "\"backend\":\"" + json_escape(info.backend) + "\"";
    json += "},\"limits\":{";
    json += "\"maxBufferSize\":" + std::to_string(limits.max_buffer_size) + ",";
    json += "\"maxStorageBufferBindingSize\":" +
            std::to_string(limits.max_storage_buffer_binding_size) + ",";
    json += "\"maxComputeWorkgroupsPerDimension\":" +
            std::to_string(limits.max_compute_workgroups_per_dimension) + ",";
    json += "\"maxComputeInvocationsPerWorkgroup\":" +
            std::to_string(limits.max_compute_invocations_per_workgroup) + ",";
    json += "\"maxStorageBuffersPerShaderStage\":" +
            std::to_string(limits.max_storage_buffers_per_shader_stage) + ",";
    json += "\"minStorageBufferOffsetAlignment\":" +
            std::to_string(limits.min_storage_buffer_offset_alignment) + "},";
    json += "\"adapterMaxima\":{";
    json += "\"maxBufferSize\":" + std::to_string(maxima.max_buffer_size) + ",";
    json += "\"maxStorageBufferBindingSize\":" +
            std::to_string(maxima.max_storage_buffer_binding_size) + "},";
    json += "\"selfCheck\":{\"elements\":" + std::to_string(result.elements) +
            ",\"mismatches\":" + std::to_string(result.mismatches) + "}}";
    // Kept, not released: a model loads onto the device that was checked.
    session().device = std::move(result.device);
    bllm_deliver(json.c_str());
}

#if BLLM_DIAGNOSTICS_ENABLED
void on_check_accepted(bllm::residency::CheckError error, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    if (error != bllm::residency::CheckError::Ok) {
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    bllm_reply(request, "{\"ok\":true}");
}
#endif

#if BLLM_DIAGNOSTICS_ENABLED
// --- readback measurement spike -------------------------------------------
// Answers one question before the decode loop is designed: what does a
// serialized GPU round trip cost? Run on request only; not on the normal path.

double now_ms() { return emscripten_get_now(); }

void on_bench(bllm::gpu::ReadbackBenchResult result, void*) {
    if (!guard().complete(pending_generation())) {
        return;
    }
    disarm_timeout();

    if (!result.ok) {
        report_failure("bench", result.error);
        return;
    }
    auto seq = result.sequential_ms;
    std::sort(seq.begin(), seq.end());
    const auto pick = [&seq](double q) {
        if (seq.empty()) return 0.0;
        const auto i = static_cast<std::size_t>(q * static_cast<double>(seq.size() - 1));
        return seq[i];
    };
    double total = 0.0;
    for (const double v : seq) total += v;

    std::string json = "{\"ok\":true,\"bench\":{";
    json += "\"iterations\":" + std::to_string(result.iterations) + ",";
    json += "\"seqMinMs\":" + std::to_string(pick(0.0)) + ",";
    json += "\"seqMedianMs\":" + std::to_string(pick(0.5)) + ",";
    json += "\"seqP95Ms\":" + std::to_string(pick(0.95)) + ",";
    json += "\"seqMaxMs\":" + std::to_string(pick(1.0)) + ",";
    json += "\"seqMeanMs\":" +
            std::to_string(seq.empty() ? 0.0 : total / static_cast<double>(seq.size())) + ",";
    json += "\"batchedTotalMs\":" + std::to_string(result.batched_total_ms) + "}}";
    bllm_deliver(json.c_str());
}

void on_device_for_bench(std::unique_ptr<bllm::gpu::Device> device, const char* error,
                         void* userdata) {
    const std::uint32_t generation = to_generation(userdata);
    if (device == nullptr) {
        if (guard().complete(generation)) {
            disarm_timeout();
            report_failure("device", error != nullptr ? error : "unknown error");
        }
        return;
    }
    if (!guard().active() || guard().generation() != generation) {
        return;
    }
    bllm::gpu::run_readback_bench(std::move(device), kBenchIterations, now_ms,
                                  on_bench, nullptr);
}

#endif  // BLLM_DIAGNOSTICS_ENABLED

void on_device(std::unique_ptr<bllm::gpu::Device> device, const char* error,
               void* userdata) {
    const std::uint32_t generation = to_generation(userdata);

    if (device == nullptr) {
        if (guard().complete(generation)) {
            disarm_timeout();
            report_failure("device", error != nullptr ? error : "unknown error");
        }
        return;
    }
    // A device that arrives after the run timed out is released here rather
    // than starting work nobody is waiting for.
    if (!guard().active() || guard().generation() != generation) {
        return;
    }
    // Ownership passes into the check and comes back in the result.
    bllm::gpu::run_self_check(std::move(device), kSelfCheckElements,
                              on_self_check, to_userdata(generation));
}

void on_timeout(void* userdata) {
    timeout_id() = 0;
    if (!guard().complete(to_generation(userdata))) {
        return;   // the run already finished; nothing to report
    }
    report_failure("timeout", "the GPU did not respond within " +
                                  std::to_string(static_cast<int>(kRunTimeoutMs / 1000)) +
                                  " seconds; the request may still be pending");
}

}  // namespace

namespace {

// `count` ranges as pairs of doubles, [begin, end), each an exact integer.
bool read_ranges(const double* pairs, std::uint32_t count, std::vector<bllm::residency::FileRange>& out) {
    out.clear();
    for (std::uint32_t i = 0; i < 2 * count; ++i) {
        if (!exact_integer(pairs[i]) || pairs[i] < 0) return false;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        out.push_back({static_cast<std::uint64_t>(pairs[2 * i]), static_cast<std::uint64_t>(pairs[2 * i + 1])});
    }
    return true;
}

// Where a file's header and index end, as its ranges must hold them
// (residency/file_ranges.h): its tensor data's first byte, or the file's end
// for a file of no tensors.
std::uint64_t index_end_of(const bllm::gguf::TensorIndex& index, std::uint64_t file_size) {
    std::uint64_t end = file_size;
    for (const bllm::gguf::TensorEntry& t : index.tensors()) end = std::min(end, t.data_offset);
    return end;
}

// The loaded model a companion of `kind` pairs against, and the bytes the
// plans it shares the budget with hold: the text model's and the other
// kind's companion's — that kind's own is replaced by the load. Null when
// no model encodes.
std::optional<bllm::preflight::CompanionHost> host_of(const Session& s, bllm::arch::MediaKind kind) {
    if (s.embedder == nullptr || s.loaded == nullptr) return std::nullopt;
    std::uint64_t resident = s.loaded->plan().total_bytes;
    for (std::size_t k = 0; k < kMediaKinds; ++k) {
        if (k != index_of(kind) && s.companion_loaded[k] != nullptr) resident += s.companion_loaded[k]->plan().total_bytes;
    }
    return bllm::preflight::CompanionHost{s.loaded_description, s.loaded_architecture,
                                          s.embedder->tokenizer().vocabulary(), resident,
                                          s.embedder->most_media_rows()};
}

// A kind as the page writes it: 0 image, 1 audio.
std::optional<bllm::arch::MediaKind> kind_of(std::uint32_t kind) {
    if (kind == 0) return bllm::arch::MediaKind::Image;
    if (kind == 1) return bllm::arch::MediaKind::Audio;
    return std::nullopt;
}

// The vision stage attached, or null.
bllm::runtime::Vision* vision_of(Session& s) {
    if (s.embedder == nullptr) return nullptr;
    // C.146: the stage is held by its interface; the image kind's is a Vision.
    return dynamic_cast<bllm::runtime::Vision*>(s.embedder->stage(bllm::arch::MediaKind::Image));
}

// The audio stage attached, or null.
bllm::runtime::Audio* audio_of(Session& s) {
    if (s.embedder == nullptr) return nullptr;
    // C.146: the stage is held by its interface; the audio kind's is an Audio.
    return dynamic_cast<bllm::runtime::Audio*>(s.embedder->stage(bllm::arch::MediaKind::Audio));
}

std::string verdict_json(const bllm::preflight::Verdict& verdict) {
    std::string json = "{\"status\":\"read\",\"reached\":" + json_string(bllm::preflight::to_string(verdict.reached()));
    json += ",\"fit\":" + fit_json(verdict.fit);
    json += ",\"blockers\":[";
    for (std::size_t i = 0; i < verdict.blockers.size(); ++i) {
        const auto& b = verdict.blockers[i];
        json += i == 0 ? "" : ",";
        json += "{\"stage\":" + json_string(bllm::preflight::to_string(b.stage)) + ",\"detail\":" +
                json_string(b.detail) + "}";
    }
    return json + "]}";
}

void on_companion_ready(std::unique_ptr<bllm::residency::Upload> upload, bllm::residency::UploadError error,
                        const char* subject, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    CompanionLoad* load = session().companion.get();
    if (load == nullptr || load->begin_request != request) {
        bllm_reply(request, failure_json("the companion's load was replaced by another").c_str());
        return;
    }
    if (upload == nullptr) {
        load->settled = true;
        const std::string named(bllm::residency::to_string(error));
        bllm_reply(request,
                   failure_json(subject != nullptr && *subject != '\0' ? named + " (" + subject + ")" : named).c_str());
        return;
    }
    load->upload = std::move(upload);
    bllm_reply(request, ("{\"ok\":true,\"chunkPointer\":" +
                         std::to_string(reinterpret_cast<std::uintptr_t>(load->chunk.data())) +
                         ",\"chunkBytes\":" + std::to_string(load->chunk.size()) + "}")
                            .c_str());
}

void on_companion_accepted(bllm::residency::UploadError error, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    if (error != bllm::residency::UploadError::Ok) {
        if (session().companion != nullptr) session().companion->settled = true;
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    bllm_reply(request, "{\"ok\":true}");
}

// A working buffer of a plan, by purpose, as a binding; empty when the plan
// has none.
bllm::kernels::Binding scratch(const bllm::residency::ResidencyPlan& plan, std::string_view purpose) {
    for (const auto& p : plan.scratch) {
        if (p.purpose == purpose) return bllm::kernels::Binding{p.range.buffer, p.range.offset, p.range.length};
    }
    return bllm::kernels::Binding{};
}

// Whether `load` is still the session's, its buffers kept and the embedder
// it attaches to still there: a later load or release replaces it.
bool current(const Session& s, const CompanionLoad* load) {
    return load != nullptr && load == s.companion.get() && !load->settled &&
           s.companion_loaded[index_of(load->kind)] != nullptr && s.embedder != nullptr;
}

// A companion's program is built: the link to the text program and its
// kind's stage, attached to the embedder.
void on_media_built(std::unique_ptr<bllm::kernels::Program> program, bllm::kernels::ProgramError error,
                     std::string_view message, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    CompanionLoad* load = s.companion.get();
    if (!current(s, load)) {
        bllm_reply(request, failure_json("the companion's load was replaced by another").c_str());
        return;
    }
    load->settled = true;
    load->setup.reset();
    auto& loaded = s.companion_loaded[index_of(load->kind)];
    if (error != bllm::kernels::ProgramError::Ok) {
        loaded.reset();
        bllm_reply(request, failure_json("the encoder's kernels did not build", message).c_str());
        return;
    }
    const auto& encoder_plan = loaded->plan();
    auto link = std::make_unique<bllm::kernels::ProgramLink>(*loaded, scratch(encoder_plan, "soft_tokens"),
                                                             *s.loaded, scratch(s.loaded->plan(), "hidden"));
    // The encoder's tokens, which preflight found in the vocabulary.
    const auto& vocabulary = s.embedder->tokenizer().vocabulary();
    const auto& texts = load->encoder->tokens;
    const bllm::runtime::MediaTokens tokens{*vocabulary.find(texts[0]), *vocabulary.find(texts[1]),
                                            *vocabulary.find(texts[2])};
    if (load->kind == bllm::arch::MediaKind::Audio) {
        auto audio = std::make_unique<bllm::runtime::Audio>(std::move(program), std::move(link),
                                                            scratch(encoder_plan, "mel"), load->most_rows);
        const std::uint32_t most = audio->most_samples();
        s.embedder->attach(load->kind, std::move(audio), tokens, s.loaded_description.embedding_width);
        bllm_reply(request, ("{\"ok\":true,\"mostSamples\":" + std::to_string(most) + "}").c_str());
        return;
    }
    const bllm::model::VisionDescription& v = *load->description.vision;
    s.embedder->attach(load->kind,
                       std::make_unique<bllm::runtime::Vision>(std::move(program), std::move(link),
                                                               scratch(encoder_plan, "pixels"), load->budget,
                                                               v.patch_size, v.pooling),
                       tokens, s.loaded_description.embedding_width);
    bllm_reply(request, ("{\"ok\":true,\"imageTokens\":" + std::to_string(load->budget.most) + "}").c_str());
}

// The setup's step has run: the step's program built next.
void on_setup_ran(bllm::kernels::ProgramError error, std::string_view message, std::span<const std::byte>,
                  void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    CompanionLoad* load = s.companion.get();
    if (!current(s, load)) {
        bllm_reply(request, failure_json("the companion's load was replaced by another").c_str());
        return;
    }
    if (error != bllm::kernels::ProgramError::Ok) {
        load->settled = true;
        s.companion_loaded[index_of(load->kind)].reset();
        bllm_reply(request, failure_json("the encoder's setup did not run", message).c_str());
        return;
    }
    bllm::kernels::Program::build(*s.companion_loaded[index_of(load->kind)], std::move(load->step_launches),
                                  std::nullopt, on_media_built, userdata);
}

// The setup's program is built: its one step run, its positions written.
void on_setup_built(std::unique_ptr<bllm::kernels::Program> program, bllm::kernels::ProgramError error,
                    std::string_view message, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    CompanionLoad* load = s.companion.get();
    if (!current(s, load)) {
        bllm_reply(request, failure_json("the companion's load was replaced by another").c_str());
        return;
    }
    if (error != bllm::kernels::ProgramError::Ok) {
        load->settled = true;
        s.companion_loaded[index_of(load->kind)].reset();
        bllm_reply(request, failure_json("the encoder's setup kernels did not build", message).c_str());
        return;
    }
    load->setup = std::move(program);
    const auto& plan = s.companion_loaded[index_of(load->kind)]->plan();
    bllm::runtime::run_audio_setup(*load->setup, scratch(plan, "positions"), load->description.audio->width,
                                   on_setup_ran, userdata);
}

void on_companion_finished(bllm::residency::UploadError error, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    CompanionLoad* load = s.companion.get();
    if (error != bllm::residency::UploadError::Ok || load == nullptr) {
        if (load != nullptr) load->settled = true;
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    auto& loaded = s.companion_loaded[index_of(load->kind)];
    loaded = std::move(load->upload);
    std::vector<bllm::kernels::Launch> launches;
    const bllm::graph::GraphResult built = load->encoder->graph(load->description, loaded->plan(), launches);
    if (!built.ok()) {
        load->settled = true;
        loaded.reset();
        bllm_reply(request, failure_json("the encoder's graph does not build", built.subject).c_str());
        return;
    }
    if (load->encoder->setup == nullptr) {
        bllm::kernels::Program::build(*loaded, std::move(launches), std::nullopt, on_media_built, to_userdata(request));
        return;
    }
    // An encoder with a setup: its setup's program built and run once, then
    // its step's (runtime/audio.h).
    std::vector<bllm::kernels::Launch> setup;
    const bllm::graph::GraphResult made = load->encoder->setup(load->description, loaded->plan(), setup);
    if (!made.ok()) {
        load->settled = true;
        loaded.reset();
        bllm_reply(request, failure_json("the encoder's setup graph does not build", made.subject).c_str());
        return;
    }
    load->step_launches = std::move(launches);
    bllm::kernels::Program::build(*loaded, std::move(setup), std::nullopt, on_setup_built, to_userdata(request));
}

// A memory budget as the page writes it: an exact integer, or NaN for the
// model's policy unset.
bool budget_or_unset(double memory_budget) { return std::isnan(memory_budget) || exact_integer(memory_budget); }

bool read_budget(std::uint32_t least, std::uint32_t most, bllm::media::ImageBudget& out) {
    if (least == 0 || least > most) return false;
    out = {least, most};
    return true;
}

}  // namespace

extern "C" {

// Entry point called from the worker once the module is instantiated.
EMSCRIPTEN_KEEPALIVE void bllm_run_self_check() {
    // One run at a time. A second would acquire another device while one is
    // still live and report two results the page cannot tell apart.
    const std::uint32_t generation = guard().begin();
    if (generation == bllm::RunGuard::kNoRun) {
        report_failure("request", "a self-check is already running");
        return;
    }
    // Armed before the request, so a request that never calls back is still
    // bounded. WebGPU cannot be cancelled, so this frees the module rather
    // than the GPU.
    pending_generation() = generation;
    timeout_id() = emscripten_set_timeout(on_timeout, kRunTimeoutMs,
                                          to_userdata(generation));
    bllm::gpu::Device::request(on_device, to_userdata(generation), kDeviceEvents);
}

// Reads the index from the front of a model file and answers with the
// preflight verdict. `resident` bytes are the file's first bytes; `file_size`
// is the whole file's length. Sizes arrive as doubles because JavaScript
// numbers are exact to 2^53 and nothing here approaches that. The limits are
// the device's granted ones, all zero when no device was acquired.
EMSCRIPTEN_KEEPALIVE void bllm_preflight(std::uint32_t request, const std::byte* resident,
                                         std::uint32_t resident_length, double file_size,
                                         double max_buffer_size, double max_binding_size,
                                         std::uint32_t offset_alignment, std::int32_t precision,
                                         double memory_budget, std::uint32_t rollback_reserve,
                                         const char* stop_bytes, const std::uint32_t* stop_lengths,
                                         std::uint32_t stop_count) {
    // Checked before any conversion: casting a NaN, a negative or a
    // fractional double to an integer is undefined or lossy, and the values
    // come from outside C++. The resident prefix cannot exceed the file.
    if (!exact_integer(file_size) || file_size < resident_length || !exact_integer(max_buffer_size) ||
        !exact_integer(max_binding_size)) {
        bllm_reply(request, "{\"status\":\"unreadable\",\"error\":\"the sizes passed in are not valid\"}");
        return;
    }
    bllm::policy::LoadPolicy policy;
    if (const std::string wrong = read_load_policy(precision, memory_budget, rollback_reserve, stop_bytes,
                                                   stop_lengths, stop_count, policy);
        !wrong.empty()) {
        bllm_reply(request, ("{\"status\":\"unreadable\",\"error\":" + json_string(wrong) + "}").c_str());
        return;
    }
    bllm::gguf::MemoryByteSource source{{resident, resident_length},
                                        static_cast<std::uint64_t>(file_size)};
    const bllm::residency::DeviceLimits limits{static_cast<std::uint64_t>(max_buffer_size),
                                               static_cast<std::uint64_t>(max_binding_size),
                                               offset_alignment};
    bllm::gguf::TensorIndex index;
    const auto read = bllm::gguf::read_index(source, index);
    bllm_reply(request, preflight_json(source, read, index, limits, policy).c_str());
}

// Begins a load: see the top of this file. `confirmed` holds `confirmed_count`
// tensor ids; `max_chunk` is the largest chunk the page will send.
EMSCRIPTEN_KEEPALIVE void bllm_load_begin(std::uint32_t request, const std::byte* prefix,
                                          std::uint32_t prefix_length, double file_size,
                                          const std::uint32_t* confirmed, std::uint32_t confirmed_count,
                                          std::uint32_t max_chunk, std::int32_t precision, double memory_budget,
                                          std::uint32_t rollback_reserve, const char* stop_bytes,
                                          const std::uint32_t* stop_lengths, std::uint32_t stop_count) {
    Session& s = session();
    if (s.device == nullptr) {
        bllm_reply(request, failure_json("no checked GPU device to load onto").c_str());
        return;
    }
    if (s.load != nullptr && !s.load->settled) {
        bllm_reply(request, failure_json("a load is already in progress").c_str());
        return;
    }
    if (!exact_integer(file_size) || file_size < prefix_length || max_chunk == 0) {
        bllm_reply(request, failure_json("the sizes passed in are not valid").c_str());
        return;
    }
    bllm::policy::LoadPolicy policy;
    if (const std::string wrong = read_load_policy(precision, memory_budget, rollback_reserve, stop_bytes,
                                                   stop_lengths, stop_count, policy);
        !wrong.empty()) {
        bllm_reply(request, failure_json(wrong).c_str());
        return;
    }
    const auto size = static_cast<std::uint64_t>(file_size);
    bllm::gguf::MemoryByteSource source{{prefix, prefix_length}, size};
    bllm::gguf::TensorIndex index;
    if (const auto read = bllm::gguf::read_index(source, index); read.error != bllm::gguf::ReadError::Ok) {
        bllm_reply(request, failure_json("the index does not read from the bytes given",
                                         bllm::gguf::to_string(read.error)).c_str());
        return;
    }
    const auto& granted = s.device->limits();
    const bllm::residency::DeviceLimits limits{granted.max_buffer_size, granted.max_storage_buffer_binding_size,
                                               granted.min_storage_buffer_offset_alignment};
    bllm::model::ModelDescription description;
    bllm::residency::ResidencyPlan plan;
    if (const std::string stop = bllm::preflight::plan_load(index, limits, policy, description, plan);
        !stop.empty()) {
        bllm_reply(request, failure_json(stop).c_str());
        return;
    }
    std::vector<bllm::gguf::TensorId> duplicates;
    for (std::uint32_t i = 0; i < confirmed_count; ++i) duplicates.push_back(bllm::gguf::TensorId{confirmed[i]});

    std::string_view architecture;
    (void)index.read_string("general.architecture", architecture);

    // A new load replaces the model loaded before it, releasing its buffers
    // first so the two are never on the device together: its generator or
    // embedder first, so a turn or an encode running finishes cancelled and
    // the program borrowing the buffers goes; any check of it too, since it
    // must not outlive what it checks.
    s.generator.reset();
    s.embedder.reset();
#if BLLM_DIAGNOSTICS_ENABLED
    s.check.reset();
#endif
    // A companion goes with the model it served: its stage went with the
    // embedder, its buffers now.
    s.companion.reset();
    for (auto& loaded : s.companion_loaded) loaded.reset();
    s.loaded.reset();
    s.loaded_description = description;
    s.loaded_architecture = std::string(architecture);
    s.load = std::make_unique<Load>();
    Load& load = *s.load;
    load.begin_request = request;
    load.chunk.resize(max_chunk);
    load.file_size = size;
    load.description = description;
    load.architecture = bllm::capability::find_architecture(architecture);
    load.precision = policy.cache_precision;
    bllm::residency::Upload::begin(*s.device, index, plan, size, bllm::capability::find_format, duplicates,
                                   max_chunk, on_load_ready, to_userdata(request));
    // While the device creates the buffers, the tokenizer from the prefix and
    // the stop set over its vocabulary; begin answers once both are in.
    // Optimization (browser): 20 to 56 ms of the CPU's for the listed models,
    // overlapped with the GPU's work rather than ahead of it.
    load.tokenizer_failure = bllm::preflight::load_tokenizer(source, index, load.tokenizer);
    if (load.tokenizer_failure.empty() && load.architecture != nullptr &&
        bllm::arch::encoder_graph(*load.architecture) != nullptr) {
        const bllm::runtime::EndsResult ends =
            bllm::runtime::resolve_ends(index, load.tokenizer->vocabulary(), load.ends);
        if (ends.error != bllm::runtime::EndsError::Ok) {
            load.tokenizer_failure = "the file's ends cannot be resolved: " + ends.subject;
        }
    } else if (load.tokenizer_failure.empty()) {
        const bllm::runtime::StopsResult stops =
            bllm::runtime::resolve_stops(index, load.tokenizer->vocabulary(), policy.stop, load.stops);
        if (stops.error != bllm::runtime::StopsError::Ok) {
            load.tokenizer_failure = "the stop tokens cannot be resolved: " + stops.subject;
        }
    }
    load.prepared = true;
    answer_begin(load);
}

// The page has copied `length` bytes at `file_offset` into the chunk buffer.
EMSCRIPTEN_KEEPALIVE void bllm_load_chunk(std::uint32_t request, double file_offset, std::uint32_t length) {
    Load* load = session().load.get();
    if (load == nullptr || load->settled || load->upload == nullptr) {
        bllm_reply(request, failure_json("no load is in progress").c_str());
        return;
    }
    if (!(file_offset >= 0 && file_offset == static_cast<double>(static_cast<std::uint64_t>(file_offset))) ||
        length > load->chunk.size()) {
        bllm_reply(request, failure_json("the chunk passed in is not valid").c_str());
        return;
    }
    load->upload->write(static_cast<std::uint64_t>(file_offset), std::span(load->chunk.data(), length),
                        on_load_accepted, to_userdata(request));
}

EMSCRIPTEN_KEEPALIVE void bllm_load_finish(std::uint32_t request) {
    Load* load = session().load.get();
    if (load == nullptr || load->settled || load->upload == nullptr) {
        bllm_reply(request, failure_json("no load is in progress").c_str());
        return;
    }
    load->upload->finish(on_load_finished, to_userdata(request));
}

// Starts a turn: see the top of this file.
EMSCRIPTEN_KEEPALIVE void bllm_generate(std::uint32_t request, const char* text, std::uint32_t text_length,
                                        std::uint32_t sampled, float temperature, std::uint32_t top_k, float top_p,
                                        float min_p, double seed, std::uint32_t max_tokens) {
    Session& s = session();
    if (s.generator == nullptr) {
        bllm_reply(request, failure_json(s.embedder != nullptr ? "the model embeds; it does not generate"
                                                               : "no model is loaded")
                                .c_str());
        return;
    }
    if (!exact_integer(seed) || seed >= kMaxExactInteger) {
        bllm_reply(request, failure_json("the seed passed in is not valid").c_str());
        return;
    }
    bllm::policy::TurnPolicy policy;
    if (sampled == 1) policy.sampling = {temperature, top_k, top_p, min_p};
    policy.seed = bllm::policy::Seed{static_cast<std::uint64_t>(seed)};
    policy.max_tokens = max_tokens;
    const bllm::runtime::GenerateResult started = s.generator->start(
        {text, text_length}, policy, on_turn_text, on_turn_end, to_userdata(request));
    using bllm::tokenizer::EncodeError;
    if (started.encode.error == EncodeError::TooLong) {
        // Its normalized bytes where it was normalized; else its raw bytes,
        // refused before.
        const bool normalized = started.encode.normalized_bytes != 0;
        const std::string subject =
            "it is " + std::to_string(normalized ? started.encode.normalized_bytes : started.encode.raw_bytes) +
            (normalized ? " bytes once normalized" : " bytes") + ", and a prompt may be " +
            std::to_string(bllm::tokenizer::kMaxEncodeBytes) + " normalized";
        bllm_reply(request, ("{\"ok\":false,\"error\":\"the prompt is too large to encode\""
                             ",\"subject\":" + json_string(subject) +
                             ",\"code\":\"prompt-too-large\",\"promptBytes\":" +
                             std::to_string(normalized ? started.encode.normalized_bytes : started.encode.raw_bytes) +
                             ",\"limitBytes\":" + std::to_string(bllm::tokenizer::kMaxEncodeBytes) + "}")
                                .c_str());
        return;
    }
    if (started.encode.error != EncodeError::Ok) {
        bllm_reply(request, failure_json("the prompt is not valid UTF-8").c_str());
        return;
    }
    using bllm::runtime::StartError;
    switch (started.start.error) {
        case StartError::Ok:
            s.turn_request = request;
            return;
        case StartError::PromptTooLong:
            bllm_reply(request, ("{\"ok\":false,\"error\":\"the conversation is longer than the context\""
                                 ",\"code\":\"prompt-too-long\",\"subject\":" +
                                 json_string(started.start.subject) +
                                 ",\"promptTokens\":" + std::to_string(started.start.prompt_tokens) +
                                 ",\"contextTokens\":" + std::to_string(started.start.context) + "}")
                                    .c_str());
            return;
        case StartError::Busy:
            bllm_reply(request, failure_json("a turn is running", started.start.subject).c_str());
            return;
        case StartError::DeviceLost:
            bllm_reply(request, failure_json("the GPU device was lost", started.start.subject).c_str());
            return;
        case StartError::EmptyPrompt:
        case StartError::Settings:
            bllm_reply(request, failure_json("the turn cannot start", started.start.subject).c_str());
            return;
    }
}

// Ends the turn `target` started: see the top of this file.
EMSCRIPTEN_KEEPALIVE void bllm_cancel(std::uint32_t request, std::uint32_t target) {
    Session& s = session();
    const bool running = s.generator != nullptr && target != 0 && s.turn_request == target;
    if (running) s.generator->cancel();
    bllm_reply(request, running ? "{\"ok\":true,\"running\":true}" : "{\"ok\":true,\"running\":false}");
}

// Starts an encode: see the top of this file.
EMSCRIPTEN_KEEPALIVE void bllm_embed(std::uint32_t request, const char* text, std::uint32_t text_length,
                                     std::uint32_t media) {
    Session& s = session();
    // 0 none, 1 the image stage's, 2 the audio stage's.
    if (media > 2) {
        bllm_reply(request, failure_json("the medium passed in is not one this build names").c_str());
        return;
    }
    const std::optional<bllm::arch::MediaKind> kind =
        media == 0 ? std::nullopt : kind_of(media - 1);
    if (s.embedder == nullptr) {
        bllm_reply(request, failure_json(s.generator != nullptr ? "the model generates; it does not embed"
                                                                : "no model is loaded")
                                .c_str());
        return;
    }
    const bllm::runtime::EmbedResult started =
        s.embedder->start({text, text_length}, kind, on_embedded, to_userdata(request));
    using bllm::runtime::EmbedError;
    switch (started.error) {
        case EmbedError::Ok:
            return;
        case EmbedError::TooManyTokens:
            bllm_reply(request, ("{\"ok\":false,\"error\":\"the text is longer than the context\""
                                 ",\"code\":\"text-too-long\",\"textTokens\":" +
                                 std::to_string(started.tokens) +
                                 ",\"contextTokens\":" + std::to_string(started.context) + "}")
                                    .c_str());
            return;
        case EmbedError::Text:
            bllm_reply(request, failure_json(started.encode.error == bllm::tokenizer::EncodeError::TooLong
                                                 ? "the text is too large to encode"
                                                 : "the text is not valid UTF-8")
                                    .c_str());
            return;
        case EmbedError::EmptyText:
            bllm_reply(request, failure_json("the text makes no tokens").c_str());
            return;
        case EmbedError::Busy:
            bllm_reply(request, failure_json("an encode is running").c_str());
            return;
        case EmbedError::DeviceLost:
            bllm_reply(request, failure_json("the GPU device was lost").c_str());
            return;
        case EmbedError::MediaNotReady:
            bllm_reply(request, failure_json(media == 1 ? "no image is prepared" : "no audio clip is prepared").c_str());
            return;
        case EmbedError::Placeholder:
            bllm_reply(request,
                       failure_json(media == 0 ? "the text holds a medium's placeholder and no medium"
                                               : "the text must hold its medium's placeholder once, and no other's")
                           .c_str());
            return;
        default:
            bllm_reply(request, failure_json("the encode cannot start").c_str());
            return;
    }
}

// Judges a companion beside the loaded model: see the top of this file.
EMSCRIPTEN_KEEPALIVE void bllm_companion_preflight(std::uint32_t request, std::uint32_t kind_index,
                                                   const std::byte* resident,
                                                   std::uint32_t resident_length, double file_size,
                                                   const double* ranges, std::uint32_t range_count,
                                                   std::uint32_t least_tokens,
                                                   std::uint32_t most_tokens, double max_buffer_size,
                                                   double max_binding_size, std::uint32_t offset_alignment,
                                                   double memory_budget) {
    Session& s = session();
    const auto kind = kind_of(kind_index);
    if (!kind) {
        bllm_reply(request, "{\"status\":\"unreadable\",\"error\":\"the kind passed in is not one this build names\"}");
        return;
    }
    const auto host = host_of(s, *kind);
    if (!host) {
        bllm_reply(request, "{\"status\":\"unreadable\",\"error\":\"no model that embeds is loaded\"}");
        return;
    }
    std::vector<bllm::residency::FileRange> parsed;
    bllm::media::ImageBudget budget{};
    if (!exact_integer(file_size) || file_size < resident_length || !exact_integer(max_buffer_size) ||
        !exact_integer(max_binding_size) || !budget_or_unset(memory_budget) ||
        !read_ranges(ranges, range_count, parsed) ||
        (*kind == bllm::arch::MediaKind::Image && !read_budget(least_tokens, most_tokens, budget))) {
        bllm_reply(request, "{\"status\":\"unreadable\",\"error\":\"the sizes passed in are not valid\"}");
        return;
    }
    bllm::gguf::MemoryByteSource source{{resident, resident_length}, static_cast<std::uint64_t>(file_size)};
    bllm::gguf::TensorIndex index;
    const auto read = bllm::gguf::read_index(source, index);
    if (read.error == bllm::gguf::ReadError::NeedMoreBytes) {
        bllm_reply(request,
                   ("{\"status\":\"need-bytes\",\"bytesNeeded\":" + std::to_string(read.bytes_needed) + "}").c_str());
        return;
    }
    if (read.error != bllm::gguf::ReadError::Ok) {
        bllm_reply(request,
                   ("{\"status\":\"unreadable\",\"error\":" + json_string(bllm::gguf::to_string(read.error)) + "}")
                       .c_str());
        return;
    }
    bllm::policy::LoadPolicy policy;
    if (!std::isnan(memory_budget)) policy.memory_budget = static_cast<std::uint64_t>(memory_budget);
    const bllm::residency::DeviceLimits limits{static_cast<std::uint64_t>(max_buffer_size),
                                               static_cast<std::uint64_t>(max_binding_size), offset_alignment};
    const auto size = static_cast<std::uint64_t>(file_size);
    const bllm::preflight::Verdict verdict = bllm::preflight::preflight_companion(
        *kind, index, parsed, size, index_end_of(index, size), limits, policy, budget, *host);
    bllm_reply(request, verdict_json(verdict).c_str());
}

// Begins a companion's load beside the loaded model: see the top of this file.
EMSCRIPTEN_KEEPALIVE void bllm_companion_load_begin(std::uint32_t request, std::uint32_t kind_index,
                                                    const std::byte* prefix,
                                                    std::uint32_t prefix_length, double file_size,
                                                    const double* ranges, std::uint32_t range_count,
                                                    std::uint32_t max_chunk, std::uint32_t least_tokens,
                                                    std::uint32_t most_tokens, double memory_budget) {
    Session& s = session();
    const auto kind = kind_of(kind_index);
    if (!kind) {
        bllm_reply(request, failure_json("the kind passed in is not one this build names").c_str());
        return;
    }
    const auto host = host_of(s, *kind);
    if (s.device == nullptr || !host) {
        bllm_reply(request, failure_json("no model that embeds is loaded to load a companion beside").c_str());
        return;
    }
    if (s.companion != nullptr && !s.companion->settled) {
        bllm_reply(request, failure_json("a companion's load is already in progress").c_str());
        return;
    }
    std::vector<bllm::residency::FileRange> parsed;
    bllm::media::ImageBudget budget{};
    if (!exact_integer(file_size) || file_size < prefix_length || max_chunk == 0 || !budget_or_unset(memory_budget) ||
        !read_ranges(ranges, range_count, parsed) ||
        (*kind == bllm::arch::MediaKind::Image && !read_budget(least_tokens, most_tokens, budget))) {
        bllm_reply(request, failure_json("the sizes passed in are not valid").c_str());
        return;
    }
    const auto size = static_cast<std::uint64_t>(file_size);
    bllm::gguf::MemoryByteSource source{{prefix, prefix_length}, size};
    bllm::gguf::TensorIndex index;
    if (const auto read = bllm::gguf::read_index(source, index); read.error != bllm::gguf::ReadError::Ok) {
        bllm_reply(request, failure_json("the index does not read from the bytes given",
                                         bllm::gguf::to_string(read.error)).c_str());
        return;
    }
    if (const auto r = bllm::residency::check_ranges(parsed, size, index_end_of(index, size)); !r.ok()) {
        bllm_reply(request, failure_json("the ranges are not ones a load can read").c_str());
        return;
    }
    const auto& granted = s.device->limits();
    const bllm::residency::DeviceLimits limits{granted.max_buffer_size, granted.max_storage_buffer_binding_size,
                                               granted.min_storage_buffer_offset_alignment};
    bllm::policy::LoadPolicy policy;
    if (!std::isnan(memory_budget)) policy.memory_budget = static_cast<std::uint64_t>(memory_budget);
    bllm::model::ModelDescription description;
    bllm::residency::ResidencyPlan plan;
    if (const std::string stop =
            bllm::preflight::plan_companion(*kind, index, parsed, limits, policy, budget, *host, description, plan);
        !stop.empty()) {
        bllm_reply(request, failure_json(stop).c_str());
        return;
    }
    std::string_view projector;
    (void)index.read_string(bllm::arch::projector_key(*kind), projector);
    // A companion of this kind loaded before is replaced: its stage first,
    // then its buffers. The other kind's stays.
    if (s.embedder != nullptr) s.embedder->attach(*kind, nullptr, {}, s.loaded_description.embedding_width);
    s.companion_loaded[index_of(*kind)].reset();
    s.companion = std::make_unique<CompanionLoad>();
    CompanionLoad& load = *s.companion;
    load.kind = *kind;
    load.begin_request = request;
    load.chunk.resize(max_chunk);
    load.description = description;
    load.encoder = bllm::capability::find_media_encoder(*kind, projector);
    load.budget = budget;
    load.most_rows = host->most_media_rows;
    bllm::residency::Upload::begin(*s.device, index, plan, size, bllm::capability::find_format, {}, max_chunk,
                                   on_companion_ready, to_userdata(request), parsed);
}

EMSCRIPTEN_KEEPALIVE void bllm_companion_load_chunk(std::uint32_t request, double file_offset, std::uint32_t length) {
    CompanionLoad* load = session().companion.get();
    if (load == nullptr || load->settled || load->upload == nullptr) {
        bllm_reply(request, failure_json("no companion's load is in progress").c_str());
        return;
    }
    if (!(file_offset >= 0 && file_offset == static_cast<double>(static_cast<std::uint64_t>(file_offset))) ||
        length > load->chunk.size()) {
        bllm_reply(request, failure_json("the chunk passed in is not valid").c_str());
        return;
    }
    load->upload->write(static_cast<std::uint64_t>(file_offset), std::span(load->chunk.data(), length),
                        on_companion_accepted, to_userdata(request));
}

EMSCRIPTEN_KEEPALIVE void bllm_companion_load_finish(std::uint32_t request) {
    CompanionLoad* load = session().companion.get();
    if (load == nullptr || load->settled || load->upload == nullptr) {
        bllm_reply(request, failure_json("no companion's load is in progress").c_str());
        return;
    }
    load->upload->finish(on_companion_finished, to_userdata(request));
}

// Starts an image in the vision stage's slot: its grid, or the refusal.
EMSCRIPTEN_KEEPALIVE void bllm_image_begin(std::uint32_t request, std::uint32_t width, std::uint32_t height) {
    bllm::runtime::Vision* vision = vision_of(session());
    if (vision == nullptr) {
        bllm_reply(request, failure_json("no vision encoder is loaded").c_str());
        return;
    }
    using bllm::media::ImageError;
    switch (vision->begin(width, height)) {
        case ImageError::Ok: {
            const bllm::media::ImageGrid& g = vision->slot().grid;
            bllm_reply(request, ("{\"ok\":true,\"tokens\":" + std::to_string(g.tokens) + ",\"width\":" +
                                 std::to_string(g.width) + ",\"height\":" + std::to_string(g.height) + "}")
                                    .c_str());
            return;
        }
        case ImageError::Empty: bllm_reply(request, failure_json("the image has no pixels").c_str()); return;
        case ImageError::TooLarge:
            bllm_reply(request, failure_json("a side of the image is over 16,384 pixels").c_str());
            return;
        case ImageError::OverBudget:
            bllm_reply(request, failure_json("the image is too wide or too tall for the token budget").c_str());
            return;
        default: bllm_reply(request, failure_json("the image cannot start").c_str()); return;
    }
}

// The next band of the slot's image, `length` RGBA bytes at `pixels`.
EMSCRIPTEN_KEEPALIVE void bllm_image_rows(std::uint32_t request, std::uint32_t first_row, const std::uint8_t* pixels,
                                          std::uint32_t length) {
    bllm::runtime::Vision* vision = vision_of(session());
    if (vision == nullptr) {
        bllm_reply(request, failure_json("no vision encoder is loaded").c_str());
        return;
    }
    if (vision->rows(first_row, {pixels, length}) != bllm::media::ImageError::Ok) {
        bllm_reply(request, failure_json("the band is out of order or not whole rows").c_str());
        return;
    }
    bllm_reply(request, vision->ready_rows() > 0 ? "{\"ok\":true,\"ready\":true}" : "{\"ok\":true,\"ready\":false}");
}

// A clip's `count` samples at `samples`, prepared in the audio stage's slot:
// its frames and rows, or the refusal.
EMSCRIPTEN_KEEPALIVE void bllm_audio_prepare(std::uint32_t request, const float* samples, std::uint32_t count) {
    bllm::runtime::Audio* audio = audio_of(session());
    if (audio == nullptr) {
        bllm_reply(request, failure_json("no audio encoder is loaded").c_str());
        return;
    }
    using bllm::media::AudioError;
    switch (audio->prepare({samples, count})) {
        case AudioError::Ok: {
            const bllm::media::AudioClip& c = audio->slot().clip;
            bllm_reply(request, ("{\"ok\":true,\"frames\":" + std::to_string(c.frames) + ",\"rows\":" +
                                 std::to_string(c.rows) + "}")
                                    .c_str());
            return;
        }
        case AudioError::TooShort:
            bllm_reply(request, failure_json("the clip is shorter than 161 samples, about 10 ms").c_str());
            return;
        case AudioError::TooLong:
            bllm_reply(request, failure_json("the clip is longer than " + std::to_string(audio->most_samples()) +
                                             " samples")
                                    .c_str());
            return;
        default: bllm_reply(request, failure_json("the clip cannot be prepared").c_str()); return;
    }
}

#if BLLM_DIAGNOSTICS_ENABLED
// Measurement spike. Present only in a diagnostic build; the clean build does
// not compile it, so the symbol is absent from the shipped module.
// Begins checking the loaded model: see the top of this file.
EMSCRIPTEN_KEEPALIVE void bllm_check_begin(std::uint32_t request, std::uint32_t max_chunk) {
    Session& s = session();
    if (s.loaded == nullptr) {
        bllm_reply(request, failure_json("no model is loaded to check").c_str());
        return;
    }
    if (max_chunk == 0) {
        bllm_reply(request, failure_json("the sizes passed in are not valid").c_str());
        return;
    }
    s.check.reset();
    s.check = std::make_unique<Check>();
    s.check->chunk.resize(max_chunk);
    s.check->check = std::make_unique<bllm::residency::UploadCheck>(*s.loaded, max_chunk);
    const std::string json = "{\"ok\":true,\"chunkPointer\":" +
                             std::to_string(reinterpret_cast<std::uintptr_t>(s.check->chunk.data())) +
                             ",\"chunkBytes\":" + std::to_string(s.check->chunk.size()) + "}";
    bllm_reply(request, json.c_str());
}

// The page has copied `length` bytes at `file_offset` into the check's buffer.
EMSCRIPTEN_KEEPALIVE void bllm_check_chunk(std::uint32_t request, double file_offset, std::uint32_t length) {
    Check* check = session().check.get();
    if (check == nullptr) {
        bllm_reply(request, failure_json("no check is in progress").c_str());
        return;
    }
    if (!(file_offset >= 0 && file_offset == static_cast<double>(static_cast<std::uint64_t>(file_offset))) ||
        length > check->chunk.size()) {
        bllm_reply(request, failure_json("the chunk passed in is not valid").c_str());
        return;
    }
    check->check->check(static_cast<std::uint64_t>(file_offset), std::span(check->chunk.data(), length),
                        on_check_accepted, to_userdata(request));
}

EMSCRIPTEN_KEEPALIVE void bllm_check_finish(std::uint32_t request) {
    Session& s = session();
    if (s.check == nullptr) {
        bllm_reply(request, failure_json("no check is in progress").c_str());
        return;
    }
    const auto error = s.check->check->finish(s.loaded_file_size);
    if (error != bllm::residency::CheckError::Ok) {
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    std::string json = "{\"ok\":true,\"mismatches\":[";
    const auto found = s.check->check->mismatches();
    for (std::size_t i = 0; i < found.size(); ++i) {
        json += i == 0 ? "{" : ",{";
        json += "\"buffer\":" + std::to_string(static_cast<std::uint32_t>(found[i].buffer)) +
                ",\"offset\":" + std::to_string(found[i].offset) + ",\"tensor\":" + json_string(found[i].tensor) +
                "}";
    }
    bllm_reply(request, (json + "]}").c_str());
}

// A step reported: its record, stamped with the module's clock — after the
// runtime has run its next steps (runtime/runtime.h).
void record_step(const bllm::runtime::StepTimes& step, void*) {
    Session& s = session();
    if (s.steps.size() == s.steps.capacity()) return;   // never past the reserve
    s.steps.push_back({step.prefill ? 1.0 : 0.0, static_cast<double>(step.position), static_cast<double>(step.tokens),
                       static_cast<double>(step.begin_ns), static_cast<double>(step.end_ns), emscripten_get_now()});
}

// The self-check, on a device asked for timestamp queries: see the top of
// this file.
EMSCRIPTEN_KEEPALIVE void bllm_run_profile_check() {
    const std::uint32_t generation = guard().begin();
    if (generation == bllm::RunGuard::kNoRun) {
        report_failure("request", "a self-check is already running");
        return;
    }
    pending_generation() = generation;
    timeout_id() = emscripten_set_timeout(on_timeout, kRunTimeoutMs, to_userdata(generation));
    const bllm::gpu::Instance instance{wgpuCreateInstance(nullptr)};
    bllm::gpu::Device::request(instance.get(), on_device, to_userdata(generation), nullptr,
                               bllm::gpu::DiagnosticRequest{.timestamps = true}, kDeviceEvents);
}

// Sets the loaded model's step observer, or clears it: 1 done, 0 refused —
// no model loaded, a turn running, or a device without timestamp queries.
EMSCRIPTEN_KEEPALIVE int bllm_observe_steps(int on) {
    Session& s = session();
    if (s.generator == nullptr || s.device == nullptr || !s.device->timestamps()) return 0;
    if (!s.generator->observe_steps(on != 0 ? record_step : nullptr, nullptr)) return 0;
    s.observing = on != 0;
    s.steps.clear();
    return 1;
}

EMSCRIPTEN_KEEPALIVE void bllm_run_readback_bench() {
    const std::uint32_t generation = guard().begin();
    if (generation == bllm::RunGuard::kNoRun) {
        report_failure("request", "a run is already in progress");
        return;
    }
    pending_generation() = generation;
    timeout_id() = emscripten_set_timeout(on_timeout, kRunTimeoutMs,
                                          to_userdata(generation));
    bllm::gpu::Device::request(on_device_for_bench, to_userdata(generation));
}
#endif  // BLLM_DIAGNOSTICS_ENABLED

}  // extern "C"

int main() {
    // Nothing runs at load. The page decides when to start, so a failure is
    // attributable to an explicit request rather than to module instantiation.
    return 0;
}
