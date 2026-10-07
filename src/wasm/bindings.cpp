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
//     the stop set;
//   - the loaded model: its Upload, which owns the weights' buffers, and the
//     generator over it (runtime/generator.h), which owns the tokenizer, the
//     program and the cache; the generator is released first, since its
//     program borrows the Upload.
// Every decision about that state is core's; this file only holds it and
// translates.
//
// Contract 11, the boundary: this file and web/worker.js are the only two
// places JavaScript and C++ meet. The crossings are preflight a header prefix;
// begin a load, load a chunk of the file, and finish the load (web/load.js);
// generate from a rendered prompt and the turn's policy; and cancel. Text goes
// back one crossing per piece the reply completes (WASM.2).
//
// The model's load policy (policy/policy.h) crosses with preflight and with
// begin, the same at both, so Fit is judged against the plan the load makes:
// the cache precision as its index in CachePrecision, or −1; the memory
// budget as a double, an exact integer, or NaN; the rollback reserve, or
// 2^32 − 1; and the stop texts as one buffer, each text followed by a NUL,
// which no token's text holds. −1, NaN and 2^32 − 1 stand for the field
// unset — an unmeasured model's, which runs on the default.
//
//   bllm_preflight(request, prefix, prefix_length, file_size, limits...,
//                  policy...)
//     Reads the index from `prefix` and answers with how far this build
//     takes the model, judged under the model's load policy, or with the
//     bytes the reader still needs.
//
//   bllm_load_begin(request, prefix, prefix_length, file_size, confirmed,
//                   confirmed_count, max_chunk, policy...)
//     Reads the index from `prefix` — the bytes preflight read — describes
//     the model, plans it for the kept device, loads its tokenizer from the
//     prefix, where its vocabulary and merges lie (preflight.h's
//     load_tokenizer), and resolves its stop set from the file and the
//     policy's stop texts (runtime/stops.h); then begins the Upload with the
//     duplicates the page confirmed (web/duplicates.js). Answers once the
//     device holds the buffers: the chunk buffer's address and size, or the
//     failure, named. A model already loaded is released first — its
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
//     with the failure that stopped any of it, named.
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
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/gpu/device.h"
#include "core/preflight/preflight.h"
#include "core/residency/upload.h"
#include "core/run_guard.h"
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
                           const bllm::residency::DeviceLimits& limits) {
    using bllm::gguf::ReadError;
    if (read.error == ReadError::NeedMoreBytes) {
        return "{\"status\":\"need-bytes\",\"bytesNeeded\":" +
               std::to_string(read.bytes_needed) + "}";
    }
    if (read.error != ReadError::Ok) {
        return "{\"status\":\"unreadable\",\"error\":" +
               json_string(bllm::gguf::to_string(read.error)) + "}";
    }
    // Every listed model is unmeasured, so each runs on the load policy's
    // defaults.
    const auto verdict = bllm::preflight::preflight(index, limits, bllm::policy::LoadPolicy{});
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
    std::unique_ptr<bllm::residency::Upload> upload;   // null until begin's answer
    std::vector<std::byte> chunk;                      // the chunk buffer
    std::uint64_t file_size = 0;
    bool settled = false;   // finished or failed: the next begin replaces it
};

#if BLLM_DIAGNOSTICS_ENABLED
// A check of the loaded model, which must outlive it (upload_check.h).
struct Check {
    std::unique_ptr<bllm::residency::UploadCheck> check;
    std::vector<std::byte> chunk;
};
#endif

struct Session {
    std::unique_ptr<bllm::gpu::Device> device;          // the checked device, kept
    std::unique_ptr<Load> load;
    std::unique_ptr<bllm::residency::Upload> loaded;    // the model a load finished
    std::uint64_t loaded_file_size = 0;
#if BLLM_DIAGNOSTICS_ENABLED
    std::unique_ptr<Check> check;   // declared after `loaded`, so released before it
#endif
};

Session& session() {
    static Session s;
    return s;
}

std::string failure_json(std::string_view error, std::string_view subject = {}) {
    return "{\"ok\":false,\"error\":" + json_string(error) + ",\"subject\":" + json_string(subject) + "}";
}

// The load an answer belongs to may have been replaced by a later begin;
// the request id carries it, so the answer still reaches its request.
void on_load_ready(std::unique_ptr<bllm::residency::Upload> upload, bllm::residency::UploadError error,
                   const char* subject, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Load* load = session().load.get();
    if (upload == nullptr || load == nullptr) {
        if (load != nullptr) load->settled = true;
        bllm_reply(request, failure_json(bllm::residency::to_string(error), subject).c_str());
        return;
    }
    load->upload = std::move(upload);
    const std::string json = "{\"ok\":true,\"chunkPointer\":" +
                             std::to_string(reinterpret_cast<std::uintptr_t>(load->chunk.data())) +
                             ",\"chunkBytes\":" + std::to_string(load->chunk.size()) + "}";
    bllm_reply(request, json.c_str());
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

void on_load_finished(bllm::residency::UploadError error, void* userdata) {
    const std::uint32_t request = to_generation(userdata);
    Session& s = session();
    if (s.load != nullptr) s.load->settled = true;
    if (error != bllm::residency::UploadError::Ok) {
        bllm_reply(request, failure_json(bllm::residency::to_string(error)).c_str());
        return;
    }
    // The finished Upload owns the model's buffers: it stays as the loaded
    // model. The chunk buffer goes with the load.
    if (s.load != nullptr) {
        s.loaded = std::move(s.load->upload);
        s.loaded_file_size = s.load->file_size;
    }
    bllm_reply(request, "{\"ok\":true}");
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
    bllm::gpu::Device::request(on_device, to_userdata(generation));
}

// Reads the index from the front of a model file and answers with the
// preflight verdict. `resident` bytes are the file's first bytes; `file_size`
// is the whole file's length. Sizes arrive as doubles because JavaScript
// numbers are exact to 2^53 and nothing here approaches that. The limits are
// the device's granted ones, all zero when no device was acquired.
EMSCRIPTEN_KEEPALIVE void bllm_preflight(std::uint32_t request, const std::byte* resident,
                                         std::uint32_t resident_length, double file_size,
                                         double max_buffer_size, double max_binding_size,
                                         std::uint32_t offset_alignment) {
    // Checked before any conversion: casting a NaN, a negative or a
    // fractional double to an integer is undefined or lossy, and the values
    // come from outside C++. The resident prefix cannot exceed the file.
    constexpr double kMaxExactInteger = 9007199254740992.0;   // 2^53
    const auto exact = [](double v) {
        return v >= 0 && v <= kMaxExactInteger && v == static_cast<double>(static_cast<std::uint64_t>(v));
    };
    if (!exact(file_size) || file_size < resident_length || !exact(max_buffer_size) ||
        !exact(max_binding_size)) {
        bllm_reply(request, "{\"status\":\"unreadable\",\"error\":\"the sizes passed in are not valid\"}");
        return;
    }
    bllm::gguf::MemoryByteSource source{{resident, resident_length},
                                        static_cast<std::uint64_t>(file_size)};
    const bllm::residency::DeviceLimits limits{static_cast<std::uint64_t>(max_buffer_size),
                                               static_cast<std::uint64_t>(max_binding_size),
                                               offset_alignment};
    bllm::gguf::TensorIndex index;
    const auto read = bllm::gguf::read_index(source, index);
    bllm_reply(request, preflight_json(source, read, index, limits).c_str());
}

// Begins a load: see the top of this file. `confirmed` holds `confirmed_count`
// tensor ids; `max_chunk` is the largest chunk the page will send.
EMSCRIPTEN_KEEPALIVE void bllm_load_begin(std::uint32_t request, const std::byte* prefix,
                                          std::uint32_t prefix_length, double file_size,
                                          const std::uint32_t* confirmed, std::uint32_t confirmed_count,
                                          std::uint32_t max_chunk) {
    Session& s = session();
    if (s.device == nullptr) {
        bllm_reply(request, failure_json("no checked GPU device to load onto").c_str());
        return;
    }
    if (s.load != nullptr && !s.load->settled) {
        bllm_reply(request, failure_json("a load is already in progress").c_str());
        return;
    }
    constexpr double kMaxExactInteger = 9007199254740992.0;   // 2^53
    if (!(file_size >= prefix_length && file_size <= kMaxExactInteger &&
          file_size == static_cast<double>(static_cast<std::uint64_t>(file_size))) || max_chunk == 0) {
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
    const auto& granted = s.device->limits();
    const bllm::residency::DeviceLimits limits{granted.max_buffer_size, granted.max_storage_buffer_binding_size,
                                               granted.min_storage_buffer_offset_alignment};
    bllm::model::ModelDescription description;
    bllm::residency::ResidencyPlan plan;
    if (const std::string stop = bllm::preflight::plan_load(index, limits, bllm::policy::LoadPolicy{},
                                                            description, plan);
        !stop.empty()) {
        bllm_reply(request, failure_json(stop).c_str());
        return;
    }
    std::vector<bllm::gguf::TensorId> duplicates;
    for (std::uint32_t i = 0; i < confirmed_count; ++i) duplicates.push_back(bllm::gguf::TensorId{confirmed[i]});

    // A new load replaces the model loaded before it, releasing its buffers
    // first so the two are never on the device together; any check of it
    // goes first, since it must not outlive what it checks.
#if BLLM_DIAGNOSTICS_ENABLED
    s.check.reset();
#endif
    s.loaded.reset();
    s.load = std::make_unique<Load>();
    s.load->chunk.resize(max_chunk);
    s.load->file_size = size;
    bllm::residency::Upload::begin(*s.device, index, plan, size, bllm::capability::find_format, duplicates,
                                   max_chunk, on_load_ready, to_userdata(request));
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
