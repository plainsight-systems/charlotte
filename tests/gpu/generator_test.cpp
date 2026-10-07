// The generator on the GPU (runtime/generator.h), over Qwen3's file: text in,
// text out, against the same turn run through the runtime alone, its tokens
// decoded by the tokenizer.

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/arch/architecture.h"
#include "core/cache/kv.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/program.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/runtime/generator.h"
#include "core/runtime/stops.h"
#include "core/sampler/sampler.h"
#include "core/tokenizer/vocabulary.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/test_data.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;
using tokenizer::TokenId;

namespace {

constexpr std::size_t kUploadChunk = 16u << 20;

// Qwen3, uploaded once, planned within 512 MiB — a context near a thousand
// tokens — with its graph's launches and the draw's record.
struct Model {
    std::string bytes;
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    kernels::Binding sampled{};
};

std::unique_ptr<Model> upload_qwen3(WGPUInstance instance, const gpu::Device& device) {
    auto m = std::make_unique<Model>();
    m->bytes = load_test_data("models/qwen3-0.6b-q4_0.gguf");
    gguf::MemoryByteSource source{std::as_bytes(std::span{m->bytes}), m->bytes.size()};
    REQUIRE(gguf::read_index(source, m->index).error == gguf::ReadError::Ok);
    policy::LoadPolicy policy;
    policy.memory_budget = 512ull << 20;
    residency::ResidencyPlan plan;
    const std::string stop = preflight::plan_load(m->index, residency::DeviceLimits{256ull << 20, 128ull << 20, 256},
                                                  policy, m->description, plan);
    REQUIRE_MESSAGE(stop.empty(), stop);
    Ready ready;
    residency::Upload::begin(device, m->index, plan, m->bytes.size(), capability::find_format, {}, kUploadChunk,
                             on_ready, &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    m->upload = std::move(ready.upload);
    const auto all = std::as_bytes(std::span{m->bytes});
    for (std::size_t at = 0; at < all.size(); at += kUploadChunk) {
        Reported accepted;
        m->upload->write(at, all.subspan(at, std::min(kUploadChunk, all.size() - at)), on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.error == residency::UploadError::Ok);
    }
    REQUIRE(finish(instance, *m->upload) == residency::UploadError::Ok);
    const residency::ResidencyPlan& planned = m->upload->plan();
    std::string_view name;
    REQUIRE(m->index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    const graph::GraphResult built = capability::find_architecture(name)->graph(
        m->description, planned, *capability::find_format(planned.cache_type), m->launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);
    for (const residency::PlannedScratch& s : planned.scratch) {
        if (s.purpose == "sampled") m->sampled = {s.range.buffer, s.range.offset, sizeof(sampler::SampledRecord)};
    }
    return m;
}

std::unique_ptr<tokenizer::Tokenizer> tokenizer_of(const Model& m) {
    gguf::MemoryByteSource source{std::as_bytes(std::span{m.bytes}), m.bytes.size()};
    std::unique_ptr<tokenizer::Tokenizer> t;
    const std::string stop = preflight::load_tokenizer(source, m.index, t);
    REQUIRE_MESSAGE(stop.empty(), stop);
    return t;
}

// A runtime over the model, a program of its own, the cache empty.
std::unique_ptr<runtime::Runtime> runtime_of(WGPUInstance instance, const Model& m, const tokenizer::Tokenizer& t) {
    gguf::MemoryByteSource source{std::as_bytes(std::span{m.bytes}), m.bytes.size()};
    runtime::StopSet stops;
    REQUIRE(runtime::resolve_stops(m.index, t.vocabulary(), {}, stops).error == runtime::StopsError::Ok);
    const residency::ResidencyPlan& plan = m.upload->plan();
    cache::KvCache cache{m.description, plan, policy::CachePrecision::F16, plan.context_offered};
    return std::make_unique<runtime::Runtime>(build_program(instance, *m.upload, m.launches, m.sampled),
                                              std::move(cache), stops);
}

struct Streamed {
    std::vector<std::string> pieces;
    std::optional<runtime::TurnResult> result;
    std::vector<TokenId> tokens;   // through the runtime alone
};

void on_text(std::string_view text, void* userdata) {
    static_cast<Streamed*>(userdata)->pieces.emplace_back(text);
}
void on_token(TokenId token, void* userdata) { static_cast<Streamed*>(userdata)->tokens.push_back(token); }
void on_end(const runtime::TurnResult& result, void* userdata) {
    auto& s = *static_cast<Streamed*>(userdata);
    s.result = result;
    s.result->message = {};
}

void pump_end(WGPUInstance instance, const Streamed& s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{300};
    while (!s.result) {
        wgpuInstanceProcessEvents(instance);
        REQUIRE_MESSAGE(std::chrono::steady_clock::now() < deadline, "timed out waiting for the turn");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

const std::string kPrompt =
    "<|im_start|>user\nRepeat exactly, nothing else: 🙂 naïve 你好 🙂<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n";

policy::TurnPolicy greedy(std::uint32_t max_tokens) {
    return {{0.0f, 40, 0.95f, 0.05f}, policy::Seed{1}, max_tokens};
}

// The turn through the runtime alone: its tokens.
std::vector<TokenId> tokens_alone(WGPUInstance instance, const Model& m, std::uint32_t max_tokens) {
    auto t = tokenizer_of(m);
    auto r = runtime_of(instance, m, *t);
    std::vector<TokenId> prompt;
    REQUIRE(t->encode(kPrompt, prompt) == tokenizer::EncodeError::Ok);
    Streamed s;
    REQUIRE(r->start(prompt, greedy(max_tokens), on_token, on_end, &s).error == runtime::StartError::Ok);
    pump_end(instance, s);
    return s.tokens;
}

}  // namespace

TEST_CASE("the generator streams whole characters, the tokens' decoding, and ends a split one with U+FFFD") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_qwen3(instance.get(), *device);
    const auto decoder = tokenizer_of(*m);
    const std::vector<TokenId> tokens = tokens_alone(instance.get(), *m, 24);
    REQUIRE(!tokens.empty());

    // The tokens' decoding through one stream, ended after `limit` tokens,
    // and whether that ends inside a character.
    const auto decoded = [&](std::uint32_t limit, std::string& want) {
        std::string bytes;
        for (std::uint32_t i = 0; i < limit; ++i) decoder->decode(tokens[i], bytes);
        tokenizer::Utf8Stream whole;
        whole.push(bytes, want);
        return !whole.finish(want);
    };
    // The prompt's emoji, repeated, splits across byte tokens: the first
    // limit that ends inside a character.
    std::optional<std::uint32_t> split;
    for (std::uint32_t limit = 1; limit <= tokens.size() && !split; ++limit) {
        std::string ignored;
        if (decoded(limit, ignored)) split = limit;
    }
    REQUIRE(split);
    // The whole reply, then the turn ended inside a character: the text is
    // the tokens' decoding, each piece whole characters, the second's last
    // U+FFFD.
    for (const auto limit : {static_cast<std::uint32_t>(tokens.size()), *split}) {
        CAPTURE(limit);
        std::string want;
        const bool inside = decoded(limit, want);

        runtime::Generator g{tokenizer_of(*m), runtime_of(instance.get(), *m, *decoder)};
        Streamed s;
        const runtime::GenerateResult started = g.start(kPrompt, greedy(limit), on_text, on_end, &s);
        REQUIRE_MESSAGE(started.ok(), started.start.subject);
        pump_end(instance.get(), s);
        CHECK(s.result->emitted == limit);
        std::string got;
        for (const std::string& piece : s.pieces) {
            CHECK(!piece.empty());
            got += piece;
        }
        CHECK(got == want);
        if (inside) CHECK(s.pieces.back().ends_with("\xEF\xBF\xBD"));
    }
}

TEST_CASE("the generator refuses text it cannot take, and a destroyed one ends its turn Cancelled") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const auto m = upload_qwen3(instance.get(), *device);
    const auto decoder = tokenizer_of(*m);
    auto g = std::make_unique<runtime::Generator>(tokenizer_of(*m), runtime_of(instance.get(), *m, *decoder));
    Streamed s;
    // Not UTF-8.
    const auto invalid = g->start(std::string_view{"\xC3(", 2}, greedy(4), on_text, on_end, &s);
    CHECK(invalid.encode == tokenizer::EncodeError::InvalidUtf8);
    // Longer than the context × the longest token's bytes: refused unencoded.
    std::size_t longest = 0;
    for (std::size_t id = 0; id < decoder->vocabulary().size(); ++id) {
        longest = std::max(longest, decoder->vocabulary().text(static_cast<TokenId>(id)).size());
    }
    const std::string huge(std::size_t{m->upload->plan().context_offered} * longest + 1, 'a');
    const auto too_long = g->start(huge, greedy(4), on_text, on_end, &s);
    CHECK(too_long.start.error == runtime::StartError::PromptTooLong);
    CHECK(too_long.start.subject.find(std::to_string(huge.size()) + " bytes") != std::string::npos);
    CHECK(!s.result);   // neither called back
    CHECK(s.pieces.empty());
    // Destroyed mid-turn.
    REQUIRE(g->start(kPrompt, greedy(UINT32_MAX), on_text, on_end, &s).ok());
    g.reset();
    pump_end(instance.get(), s);
    CHECK(s.result->failure == runtime::TurnFailure::None);
    CHECK(s.result->end == runtime::TurnEnd::Cancelled);
}
