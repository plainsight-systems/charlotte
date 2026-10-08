// The forward pass against its reference (graph/graph.h): each listed model
// at Q4_0 — Qwen3 0.6B, Llama 3.2 1B and Gemma 3 1B — uploaded and run as
// the harness runs it, a pinned prompt decoded a token at a time, each
// position's log-probabilities against llama.cpp's on its CPU and Metal
// backends (tools/make_reference_logits.sh), one fixture a model, each in a
// namespace of its own.
//
//   - Accepted, at every position: each of Metal's 20 most likely tokens'
//     log-probability within the spread llama.cpp's own CPU and Metal show
//     there; the top token the same wherever Metal's leads its next by more
//     than that spread; every logit finite. Our sums are not llama.cpp's,
//     in order or in kernels, so bits are not compared (GDSA.2); its own two
//     backends' disagreement is the tolerance, and a model outside it is a
//     fault to find, never a tolerance to widen.
//   - The prompts: 64 tokens for Qwen3 and Llama 3.2; 600 for Gemma 3, so
//     its 512-key window layers (model/model_description.h) mask keys from
//     position 512 on, as a window layer must.
//   - The check discriminates, for each model: its RoPE pairing swapped
//     falls outside from the second position; and for Gemma 3, every layer
//     given the full context as its window falls outside past position 512.
//   - Each model is planned within a budget its weights and the prompt fit,
//     and its file is fetched by tools/fetch_test_data.py, pinned by SHA-256
//     (tests/fixtures/external.json): 382, 773 and 722 MB.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/kernels/program.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/pump.h"
#include "support/test_data.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

// One position of llama.cpp's logits, as tools/reference_logits writes it.
struct ReferencePosition {
    double metal_lse;
    double cpu_lse;
    std::uint32_t ids[20];   // Metal's most likely tokens, most likely first
    float metal[20];
    float cpu[20];
};

namespace qwen3 {
#include "fixtures/reference/qwen3-0.6b-q4_0.inc"
}
namespace llama32 {
#include "fixtures/reference/llama-3.2-1b-instruct-q4_0.inc"
}
namespace gemma3 {
#include "fixtures/reference/gemma-3-1b-it-q4_0.inc"
}

// A model's reference: its file, its fixture, and a budget its weights and
// the prompt fit.
struct Reference {
    std::string_view file;
    std::uint32_t vocabulary;
    std::span<const std::uint32_t> tokens;
    std::span<const ReferencePosition> positions;
    std::uint64_t budget;
};

const Reference kQwen3{"models/qwen3-0.6b-q4_0.gguf", qwen3::kReferenceVocabulary, qwen3::kReferenceTokens,
                       qwen3::kReferencePositions, 512ull << 20};
const Reference kLlama32{"models/llama-3.2-1b-instruct-q4_0.gguf", llama32::kReferenceVocabulary,
                         llama32::kReferenceTokens, llama32::kReferencePositions, 1ull << 30};
const Reference kGemma3{"models/gemma-3-1b-it-q4_0.gguf", gemma3::kReferenceVocabulary, gemma3::kReferenceTokens,
                        gemma3::kReferencePositions, 1ull << 30};

constexpr std::size_t kUploadChunk = 16u << 20;   // the file is written 16 MiB at a time

struct Running {
    std::string bytes;
    gguf::TensorIndex index;
    const arch::Architecture* architecture = nullptr;
    model::ModelDescription description{};
    residency::ResidencyPlan plan;
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange logits{};
};

// The model, read, described, planned within its reference's budget — a
// context room enough for the prompt — and uploaded.
Running upload_model(WGPUInstance instance, const gpu::Device& device, const Reference& ref) {
    Running r;
    r.bytes = load_test_data(std::string(ref.file));
    gguf::MemoryByteSource source{std::as_bytes(std::span{r.bytes}), r.bytes.size()};
    REQUIRE(gguf::read_index(source, r.index).error == gguf::ReadError::Ok);
    std::string_view name;
    REQUIRE(r.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    r.architecture = capability::find_architecture(name);
    REQUIRE(r.architecture->describe(r.index, r.description).ok());
    REQUIRE(r.description.vocabulary_size == ref.vocabulary);
    policy::LoadPolicy policy;
    policy.memory_budget = ref.budget;
    REQUIRE(residency::plan_residency(r.index, r.description, residency::DeviceLimits{256ull << 20, 128ull << 20, 256},
                                      policy, r.plan)
                .ok());
    REQUIRE(r.plan.context_offered >= ref.tokens.size());

    Ready ready;
    residency::Upload::begin(device, r.index, r.plan, r.bytes.size(), capability::find_format, {}, kUploadChunk, on_ready,
                             &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    r.upload = std::move(ready.upload);
    const auto all = std::as_bytes(std::span{r.bytes});
    for (std::size_t at = 0; at < all.size(); at += kUploadChunk) {
        Reported accepted;
        r.upload->write(at, all.subspan(at, std::min(kUploadChunk, all.size() - at)), on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.error == residency::UploadError::Ok);
    }
    REQUIRE(finish(instance, *r.upload) == residency::UploadError::Ok);
    for (const residency::PlannedScratch& s : r.upload->plan().scratch) {
        if (s.purpose == "logits") r.logits = s.range;
    }
    return r;
}

// The prompt's first `positions` tokens decoded a token at a time through
// the graph of `description`, and each position's logits.
std::vector<std::vector<float>> decode(WGPUInstance instance, const gpu::Device& device, const Running& r,
                                       const Reference& ref, const model::ModelDescription& description,
                                       std::size_t positions) {
    const residency::ResidencyPlan& plan = r.upload->plan();
    std::vector<kernels::Launch> launches;
    const graph::GraphResult built =
        r.architecture->graph(description, plan, *capability::find_format(plan.cache_type), launches);
    REQUIRE_MESSAGE(built.ok(), built.subject);
    const auto program = build_program(instance, *r.upload, std::move(launches));
    std::vector<std::vector<float>> logits;
    for (std::uint32_t p = 0; p < positions; ++p) {
        run_step(instance, *program, 1, ref.tokens.subspan(p, 1), p, true);
        logits.push_back(read_floats(instance, device, r.upload->buffer(r.logits.buffer), r.logits.offset,
                                     ref.vocabulary));
    }
    return logits;
}

// How far `ours` falls from llama.cpp's Metal logits, as a multiple of the
// difference llama.cpp's own CPU and Metal backends show at the same
// position, over Metal's 20 most likely tokens; and the positions where
// Metal's top token leads its next by more than that difference, and ours
// is another.
struct Agreement {
    double worst = 0;           // 1 or less: within llama.cpp's own spread everywhere
    double worst_nats = 0;
    int top_disagreements = 0;
    int non_finite = 0;         // logits that are NaN or infinite, any of which fails
};

// The larger of the two, and NaN when either is: std::max drops a NaN.
double worse(double so_far, double d) { return d > so_far || std::isnan(d) ? d : so_far; }

Agreement compare(const Reference& reference, const std::vector<std::vector<float>>& ours) {
    Agreement a;
    for (std::size_t p = 0; p < ours.size(); ++p) {
        const ReferencePosition& ref = reference.positions[p];
        const std::vector<float>& l = ours[p];
        a.non_finite += static_cast<int>(std::count_if(l.begin(), l.end(), [](float x) { return !std::isfinite(x); }));
        const double top = *std::max_element(l.begin(), l.end());
        double sum = 0;
        for (const float x : l) sum += std::exp(static_cast<double>(x) - top);
        const double lse = top + std::log(sum);
        double spread = 0;
        for (int k = 0; k < 20; ++k) {
            spread = std::max(spread, std::abs((ref.metal[k] - ref.metal_lse) - (ref.cpu[k] - ref.cpu_lse)));
        }
        for (int k = 0; k < 20; ++k) {
            const double d = std::abs((l[ref.ids[k]] - lse) - (ref.metal[k] - ref.metal_lse));
            a.worst = worse(a.worst, d / spread);
            a.worst_nats = worse(a.worst_nats, d);
        }
        const auto argmax = static_cast<std::uint32_t>(std::max_element(l.begin(), l.end()) - l.begin());
        if (ref.metal[0] - ref.metal[1] > spread && argmax != ref.ids[0]) ++a.top_disagreements;
    }
    return a;
}

// The model against its reference, then with its RoPE pairing swapped, which
// must fall outside.
void check_model(const Reference& ref) {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Running r = upload_model(instance.get(), *device, ref);

    const Agreement ours = compare(ref, decode(instance.get(), *device, r, ref, r.description, ref.tokens.size()));
    MESSAGE("worst deviation from llama.cpp's Metal logits: " << ours.worst_nats << " nats, " << ours.worst
                                                               << " of its own CPU-to-Metal spread");
    CHECK(ours.non_finite == 0);
    CHECK(ours.worst <= 1.0);   // NaN fails too
    CHECK(ours.top_disagreements == 0);

    // The check discriminates: RoPE pairing the wrong dimensions, from the
    // second position on, falls outside it.
    model::ModelDescription swapped = r.description;
    swapped.rotary_pairing = swapped.rotary_pairing == model::RotaryPairing::Halves ? model::RotaryPairing::Adjacent
                                                                                    : model::RotaryPairing::Halves;
    const Agreement wrong = compare(ref, decode(instance.get(), *device, r, ref, swapped, 16));
    MESSAGE("with the other pairing: " << wrong.worst_nats << " nats, " << wrong.worst << " of the spread");
    CHECK((wrong.worst > 1.0 || wrong.top_disagreements > 0));
}

}  // namespace

TEST_CASE("Qwen3 0.6B's log-probabilities are within llama.cpp's own spread, and a wrong pairing is not") {
    check_model(kQwen3);
}

TEST_CASE("Llama 3.2 1B's log-probabilities are within llama.cpp's own spread, and a wrong pairing is not") {
    check_model(kLlama32);
}

TEST_CASE("Gemma 3 1B's log-probabilities are within llama.cpp's own spread past its window, and a wrong pairing is not") {
    check_model(kGemma3);
    // The window is exercised: every layer given the whole context, from
    // position 512 on, falls outside.
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Running r = upload_model(instance.get(), *device, kGemma3);
    model::ModelDescription global = r.description;
    bool windowed = false;
    for (model::LayerDescription& l : global.layers) {
        windowed = windowed || l.attention_window < kGemma3.tokens.size();
        l.attention_window = global.trained_context;
    }
    REQUIRE(windowed);
    const Agreement wrong =
        compare(kGemma3, decode(instance.get(), *device, r, kGemma3, global, kGemma3.tokens.size()));
    MESSAGE("with every layer global: " << wrong.worst_nats << " nats, " << wrong.worst << " of the spread");
    CHECK((wrong.worst > 1.0 || wrong.top_disagreements > 0));
}
