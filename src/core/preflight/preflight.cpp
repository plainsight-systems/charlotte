#include "core/preflight/preflight.h"

#include <algorithm>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/arch/media_encoder.h"
#include "core/capability/capability.h"
#include "core/formats/format.h"
#include "core/graph/encoder.h"
#include "core/graph/graph.h"
#include "core/graph/audio.h"
#include "core/graph/vision.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/runtime/ends.h"
#include "core/runtime/stops.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::preflight {
namespace {

std::string quoted(std::string_view text) {
    return "\"" + std::string(text) + "\"";
}

// Why a required string key could not be read, for a rejection message.
std::string unreadable_key(gguf::MetadataError error, std::string_view key) {
    return error == gguf::MetadataError::MissingKey
               ? "the file does not declare " + std::string(key)
               : std::string(key) + " is not a string";
}

std::string_view to_string(arch::DescribeError error) noexcept {
    switch (error) {
        case arch::DescribeError::Ok: return "ok";
        case arch::DescribeError::MissingKey: return "a required key is missing";
        case arch::DescribeError::WrongKeyType: return "a key has the wrong type";
        case arch::DescribeError::MissingTensor: return "a required tensor is missing";
        case arch::DescribeError::ShapeMismatch: return "a tensor has the wrong shape";
        case arch::DescribeError::InvalidValue: return "a value no model could have";
        case arch::DescribeError::UnsupportedValue: return "a value in a form this build does not read";
    }
    return "unrecognised error";
}

// The architecture and the description it read, which Fit and Run need.
struct Described {
    const arch::Architecture* architecture;
    std::string_view name;
    model::ModelDescription description;
};

// Describe: the architecture is implemented and can read its numbers from
// this file.
std::optional<Described> check_architecture(const gguf::TensorIndex& index, Verdict& verdict) {
    constexpr std::string_view kKey = "general.architecture";
    std::string_view name;
    if (const auto e = index.read_string(kKey, name); e != gguf::MetadataError::Ok) {
        verdict.blockers.push_back({Stage::Describe, unreadable_key(e, kKey)});
        return std::nullopt;
    }
    const arch::Architecture* architecture = capability::find_architecture(name);
    if (architecture == nullptr) {
        verdict.blockers.push_back(
            {Stage::Describe, "architecture " + quoted(name) + " is not supported"});
        return std::nullopt;
    }
    model::ModelDescription description{};
    if (const auto r = architecture->describe(index, description); !r.ok()) {
        verdict.blockers.push_back(
            {Stage::Describe, "architecture " + quoted(name) + " cannot read this file: " +
                                  std::string(to_string(r.error)) + " (" + r.subject + ")"});
        return std::nullopt;
    }
    return Described{architecture, name, std::move(description)};
}

constexpr std::string_view to_string(graph::GraphError e) noexcept {
    switch (e) {
        case graph::GraphError::Ok: return "ok";
        case graph::GraphError::UngroupedFeedForward: return "a layer's gate and up are not one group";
        case graph::GraphError::UnsupportedShape: return "a layer's shape is outside the kernels";
    }
    return "unrecognised error";
}

// Run needs the architecture's graph to build over the plan Fit made, with
// a cache format a kernel can write (graph/graph.h).
void check_graph(const Described& d, const residency::ResidencyPlan& plan, Verdict& verdict) {
    std::vector<kernels::Launch> launches;
    // An encoder keeps no cache: its graph is given none.
    if (const arch::EncodeFn encode = arch::encoder_graph(*d.architecture)) {
        if (const auto r = encode(d.description, plan, launches); !r.ok()) {
            verdict.blockers.push_back({Stage::Run, "architecture " + quoted(d.name) + " cannot run this file: " +
                                                        std::string(to_string(r.error)) + " (" + r.subject + ")"});
        }
        return;
    }
    const formats::Format* cache = capability::find_format(plan.cache_type);
    if (cache == nullptr || cache->pack_wgsl().empty()) {
        verdict.blockers.push_back({Stage::Run, "the cache format " +
                                                    std::string(gguf::format_layout(plan.cache_type)->name) +
                                                    " cannot be written by this build"});
        return;
    }
    if (const auto r = arch::decoder_graph(*d.architecture)(d.description, plan, *cache, launches); !r.ok()) {
        verdict.blockers.push_back({Stage::Run, "architecture " + quoted(d.name) + " cannot run this file: " +
                                                    std::string(to_string(r.error)) + " (" + r.subject + ")"});
    }
}

std::string mebibytes(std::uint64_t bytes) {
    return std::to_string((bytes + (1u << 19)) >> 20) + " MiB";
}

std::string fit_failure(const residency::PlanResult& r, const residency::ResidencyPlan& plan,
                        const policy::LoadPolicy& policy) {
    using residency::PlanError;
    switch (r.error) {
        case PlanError::Ok: return "fits";
        case PlanError::ExceedsBudget:
            return "needs " + mebibytes(plan.total_bytes) + " for the shortest context worth offering (" +
                   std::to_string(residency::kPrefillBlock) + " tokens); the memory budget is " +
                   mebibytes(policy.memory_budget);
        case PlanError::RowExceedsBinding:
            return "a row of " + r.subject + " is wider than one storage binding";
        case PlanError::ScratchExceedsBinding:
            return "the " + r.subject + " working buffer is wider than one storage binding";
        case PlanError::UnsupportedCachePrecision:
            return "the cache precision cannot store the head dimension of " + r.subject;
        case PlanError::Overflow:
            return "sizes overflow (" + r.subject + ")";
        case PlanError::MissingTensorBytes:
            return r.subject + " lies outside the bytes the load reads";
        case PlanError::UnboundSpan:
            return r.subject + " and the bounds its product reads cannot share one storage binding";
    }
    return "unrecognised error";
}

// The working buffers the architecture's kind of graph binds: a decoder's,
// with its cache, or an encoder's, with none (graph/graph.h,
// graph/encoder.h).
residency::WorkingSet working_set_of(const Described& d) {
    return arch::encoder_graph(*d.architecture) != nullptr ? graph::encoder_working_set(d.description)
                                                            : graph::decoder_working_set(d.description);
}

// Fit: the residency plan fits the granted limits and the memory budget.
// `plan` is the plan judged, for a load to carry out.
void check_fit(const gguf::TensorIndex& index, const Described& d, const residency::DeviceLimits& limits,
               const policy::LoadPolicy& policy, Verdict& verdict, residency::ResidencyPlan& plan) {
    const model::ModelDescription& description = d.description;
    if (limits.max_buffer_size == 0 || limits.max_storage_binding_size == 0 ||
        limits.storage_offset_alignment == 0) {
        verdict.blockers.push_back({Stage::Fit, "no GPU device was acquired, so fit cannot be judged"});
        return;
    }
    if ((limits.storage_offset_alignment & (limits.storage_offset_alignment - 1)) != 0) {
        verdict.blockers.push_back({Stage::Fit, "the device's storage-offset alignment is not a power of two"});
        return;
    }
    if (const auto r = residency::plan_residency(index, description, limits, policy, working_set_of(d), plan);
        !r.ok()) {
        verdict.blockers.push_back({Stage::Fit, fit_failure(r, plan, policy)});
        return;
    }
    verdict.fit = FitSummary{plan.weight_bytes,    plan.cache_bytes,       plan.scratch_bytes,
                             plan.total_bytes,     policy.memory_budget,   plan.context_offered,
                             description.trained_context, plan.buffers.size(), {}};
    for (const residency::PlannedTensor& t : plan.tensors) {
        if (!t.candidate_duplicate_of) continue;
        const gguf::TensorEntry& copy = index.tensor(t.tensor);
        const gguf::TensorEntry& original = index.tensor(*t.candidate_duplicate_of);
        verdict.fit->duplicates.push_back(
            {t.tensor, *t.candidate_duplicate_of, copy.data_offset, original.data_offset, copy.data_length});
    }
}

// Upload needs every weight format: routes refuse a format the capability
// table does not list before any buffer is made. One blocker per unsupported
// format, naming how many tensors use it and the first of them.
void check_formats(const gguf::TensorIndex& index, Verdict& verdict,
                   std::span<const residency::FileRange> ranges = {}) {
    struct Unsupported {
        gguf::TensorType type;
        std::size_t users;
        std::string_view first_user;
    };
    std::vector<Unsupported> found;
    for (const gguf::TensorEntry& tensor : index.tensors()) {
        // A load of part of a file uploads only what lies in its ranges.
        if (!ranges.empty() && !residency::within(ranges, tensor.data_offset, tensor.data_offset + tensor.data_length)) {
            continue;
        }
        if (capability::find_format(tensor.type) != nullptr) continue;
        auto it = std::find_if(found.begin(), found.end(),
                               [&](const Unsupported& u) { return u.type == tensor.type; });
        if (it == found.end()) {
            found.push_back({tensor.type, 1, tensor.name});
        } else {
            ++it->users;
        }
    }
    for (const Unsupported& u : found) {
        verdict.blockers.push_back(
            {Stage::Upload, "format " + std::string(gguf::format_layout(u.type)->name) +
                                " is not supported (" + std::to_string(u.users) +
                                (u.users == 1 ? " tensor" : " tensors") + ", first " +
                                std::string(u.first_user) + ")"});
    }
}

// Upload needs every row a whole number of unpack's groups (format.h), as
// routes does; one blocker, naming how many tensors fall short and the first.
void check_rows(const gguf::TensorIndex& index, Verdict& verdict, std::span<const residency::FileRange> ranges = {}) {
    std::size_t short_rows = 0;
    const gguf::TensorEntry* first = nullptr;
    for (const gguf::TensorEntry& tensor : index.tensors()) {
        if (!ranges.empty() && !residency::within(ranges, tensor.data_offset, tensor.data_offset + tensor.data_length)) {
            continue;
        }
        if (formats::steps_by_groups(tensor) || formats::is_scalar(tensor)) continue;
        if (first == nullptr) first = &tensor;
        ++short_rows;
    }
    if (first == nullptr) return;
    verdict.blockers.push_back(
        {Stage::Upload, "rows must be a multiple of " + std::to_string(formats::kUnpackGroup) + " weights (" +
                         std::to_string(short_rows) + (short_rows == 1 ? " tensor" : " tensors") +
                         ", first " + first->name + ", rows of " + std::to_string(formats::row_elements(*first)) + ")"});
}

// Run needs the tokenizer, and the pre-tokenizer when the tokenizer splits
// text first. One that splits none ignores tokenizer.ggml.pre, as llama.cpp
// does: its converter writes "default" there for SentencePiece files.
void check_tokenizer(const gguf::TensorIndex& index, Verdict& verdict) {
    constexpr std::string_view kModelKey = "tokenizer.ggml.model";
    constexpr std::string_view kPreKey = "tokenizer.ggml.pre";

    std::string_view model_name;
    const tokenizer::Algorithm* algorithm = nullptr;
    if (const auto e = index.read_string(kModelKey, model_name); e != gguf::MetadataError::Ok) {
        verdict.blockers.push_back({Stage::Run, unreadable_key(e, kModelKey)});
    } else if (algorithm = capability::find_tokenizer(model_name); algorithm == nullptr) {
        verdict.blockers.push_back(
            {Stage::Run, "tokenizer " + quoted(model_name) + " is not supported"});
    }

    if (algorithm != nullptr && !algorithm->requires_pretokenizer) return;

    std::string_view pre_name;
    const auto pre = index.read_string(kPreKey, pre_name);
    if (pre == gguf::MetadataError::Ok) {
        if (capability::find_pretokenizer(pre_name) == nullptr) {
            verdict.blockers.push_back(
                {Stage::Run, "pre-tokenizer " + quoted(pre_name) + " is not supported"});
        }
    } else if (algorithm != nullptr && algorithm->requires_pretokenizer) {
        verdict.blockers.push_back(
            {Stage::Run, "tokenizer " + quoted(model_name) +
                                  " needs a pre-tokenizer, and " + unreadable_key(pre, kPreKey)});
    }
}

// Run needs the tokenizer to load from the header, and over its vocabulary
// a generator's stop set to resolve, or an embedder's ends (runtime/ends.h);
// checked only where the algorithm and the pre-tokenizer it needs are
// listed, the blockers above having named whichever is not. `encodes` is
// whether the architecture's graph encodes.
void check_tokenizer_loads(gguf::ByteSource& source, const gguf::TensorIndex& index,
                           const policy::LoadPolicy& policy, bool encodes, Verdict& verdict) {
    std::unique_ptr<tokenizer::Tokenizer> loaded;
    if (const std::string stop = load_tokenizer(source, index, loaded); !stop.empty()) {
        verdict.blockers.push_back({Stage::Run, stop});
        return;
    }
    if (encodes) {
        runtime::Ends ends;
        if (const runtime::EndsResult r = runtime::resolve_ends(index, loaded->vocabulary(), ends);
            r.error != runtime::EndsError::Ok) {
            verdict.blockers.push_back({Stage::Run, "the begin and end tokens cannot be resolved: " + r.subject});
        }
        return;
    }
    runtime::StopSet stops;
    if (const runtime::StopsResult r = runtime::resolve_stops(index, loaded->vocabulary(), policy.stop, stops);
        r.error != runtime::StopsError::Ok) {
        verdict.blockers.push_back({Stage::Run, "the stop tokens cannot be resolved: " + r.subject});
    }
}

}  // namespace

std::string load_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                           std::unique_ptr<tokenizer::Tokenizer>& out) {
    Verdict verdict;
    check_tokenizer(index, verdict);
    if (!verdict.blockers.empty()) return verdict.blockers.front().detail;
    std::string_view model_name, pre_name;
    (void)index.read_string("tokenizer.ggml.model", model_name);
    const tokenizer::Algorithm* algorithm = capability::find_tokenizer(model_name);
    const tokenizer::PreTokenizer* pretokenizer = nullptr;
    if (algorithm->requires_pretokenizer) {
        (void)index.read_string("tokenizer.ggml.pre", pre_name);
        pretokenizer = capability::find_pretokenizer(pre_name);
    }
    const tokenizer::LoadResult r = algorithm->load(source, index, pretokenizer, out);
    if (!r.ok()) return "tokenizer " + quoted(model_name) + " does not load from this file: " + r.subject;
    return {};
}

std::string plan_load(const gguf::TensorIndex& index, const residency::DeviceLimits& limits,
                      const policy::LoadPolicy& policy, model::ModelDescription& description,
                      residency::ResidencyPlan& plan) {
    Verdict verdict;
    const auto described = check_architecture(index, verdict);
    if (!described) return verdict.blockers.front().detail;
    check_fit(index, *described, limits, policy, verdict, plan);
    if (!verdict.fit) return verdict.blockers.front().detail;
    description = described->description;
    return {};
}

Verdict preflight(gguf::ByteSource& source, const gguf::TensorIndex& index, const residency::DeviceLimits& limits,
                  const policy::LoadPolicy& policy) {
    Verdict verdict;
    bool encodes = false;
    if (const auto described = check_architecture(index, verdict)) {
        encodes = arch::encoder_graph(*described->architecture) != nullptr;
        residency::ResidencyPlan plan;
        check_fit(index, *described, limits, policy, verdict, plan);
        if (verdict.fit) check_graph(*described, plan, verdict);
    }
    check_formats(index, verdict);
    check_rows(index, verdict);
    const std::size_t before = verdict.blockers.size();
    check_tokenizer(index, verdict);
    if (verdict.blockers.size() == before) check_tokenizer_loads(source, index, policy, encodes, verdict);

    for (int s = static_cast<int>(kImplementedThrough) + 1;
         s <= static_cast<int>(Stage::Run); ++s) {
        const auto stage = static_cast<Stage>(s);
        verdict.blockers.push_back(
            {stage, "the " + std::string(to_string(stage)) + " stage is not implemented in this build"});
    }
    return verdict;
}

namespace {

// What a kind of encoder is called in a blocker.
std::string_view kind_name(arch::MediaKind kind) {
    return kind == arch::MediaKind::Image ? "vision encoder" : "audio encoder";
}

// A companion's encoder of `kind`, found by its projector type and described.
struct DescribedEncoder {
    const arch::MediaEncoder* encoder;
    std::string_view projector;
    model::ModelDescription description;
};

std::optional<DescribedEncoder> check_encoder(arch::MediaKind kind, const gguf::TensorIndex& index,
                                              Verdict& verdict) {
    const std::string_view key = arch::projector_key(kind);
    std::string_view name;
    if (const auto e = index.read_string(key, name); e != gguf::MetadataError::Ok) {
        verdict.blockers.push_back({Stage::Describe, unreadable_key(e, key)});
        return std::nullopt;
    }
    const arch::MediaEncoder* encoder = capability::find_media_encoder(kind, name);
    if (encoder == nullptr) {
        verdict.blockers.push_back({Stage::Describe, std::string(kind_name(kind)) + " " + quoted(name) +
                                                         " is not supported"});
        return std::nullopt;
    }
    model::ModelDescription description{};
    if (const auto r = encoder->describe(index, description); !r.ok()) {
        verdict.blockers.push_back({Stage::Describe, std::string(kind_name(kind)) + " " + quoted(name) +
                                                         " cannot read this file: " + std::string(to_string(r.error)) +
                                                         " (" + r.subject + ")"});
        return std::nullopt;
    }
    return DescribedEncoder{encoder, name, std::move(description)};
}

// The budget a companion is planned in: what the plans loaded leave.
policy::LoadPolicy companion_policy(const policy::LoadPolicy& policy, const CompanionHost& host) {
    policy::LoadPolicy p = policy;
    p.memory_budget = policy.memory_budget > host.resident_bytes ? policy.memory_budget - host.resident_bytes : 0;
    return p;
}

// The working buffers a companion's step binds: its kind's, at its most
// rows. Null, with a blocker, for a kind this build has no working set for.
std::optional<residency::WorkingSet> working_set(const DescribedEncoder& d, media::ImageBudget budget,
                                                 const CompanionHost& host, Verdict& verdict) {
    if (d.description.vision) {
        const std::uint32_t pooling = d.description.vision->pooling;
        return graph::vision_working_set(d.description, budget.most * pooling * pooling);
    }
    if (d.description.audio) {
        if (host.most_media_rows == 0) {
            verdict.blockers.push_back({Stage::Fit, "the model loaded has no room in an encode for an audio clip"});
            return std::nullopt;
        }
        return graph::audio_working_set(d.description, host.most_media_rows);
    }
    verdict.blockers.push_back({Stage::Fit, std::string(kind_name(d.encoder->kind)) + " " + quoted(d.projector) +
                                                " has no working set in this build"});
    return std::nullopt;
}

void check_fit(const gguf::TensorIndex& index, const DescribedEncoder& d, std::span<const residency::FileRange> ranges,
               const residency::DeviceLimits& limits, const policy::LoadPolicy& policy, media::ImageBudget budget,
               const CompanionHost& host, Verdict& verdict, residency::ResidencyPlan& plan) {
    const auto set = working_set(d, budget, host, verdict);
    if (!set) return;
    if (const auto r = residency::plan_residency(index, d.description, limits, policy, *set, plan, ranges); !r.ok()) {
        verdict.blockers.push_back({Stage::Fit, r.error == residency::PlanError::ExceedsBudget
                                                    ? "needs " + mebibytes(plan.total_bytes) +
                                                          "; the budget left beside the model is " +
                                                          mebibytes(policy.memory_budget)
                                                    : fit_failure(r, plan, policy)});
        return;
    }
    verdict.fit = FitSummary{plan.weight_bytes,  plan.cache_bytes,         plan.scratch_bytes,
                             plan.total_bytes,   policy.memory_budget,     plan.context_offered,
                             0,                  plan.buffers.size(),      {}};
}

}  // namespace

std::string plan_companion(arch::MediaKind kind, const gguf::TensorIndex& index,
                           std::span<const residency::FileRange> ranges, const residency::DeviceLimits& limits,
                           const policy::LoadPolicy& policy, media::ImageBudget budget, const CompanionHost& host,
                           model::ModelDescription& description, residency::ResidencyPlan& plan) {
    Verdict verdict;
    const auto described = check_encoder(kind, index, verdict);
    if (!described) return verdict.blockers.front().detail;
    check_fit(index, *described, ranges, limits, companion_policy(policy, host), budget, host, verdict, plan);
    if (!verdict.fit) return verdict.blockers.front().detail;
    description = described->description;
    return {};
}

Verdict preflight_companion(arch::MediaKind kind, const gguf::TensorIndex& index,
                            std::span<const residency::FileRange> ranges, std::uint64_t file_size,
                            std::uint64_t index_end, const residency::DeviceLimits& limits,
                            const policy::LoadPolicy& policy, media::ImageBudget budget, const CompanionHost& host) {
    Verdict verdict;
    if (const residency::RangesResult r = residency::check_ranges(ranges, file_size, index_end); !r.ok()) {
        verdict.blockers.push_back({Stage::Download, "the catalog's ranges are not ones a load can read (range " +
                                                         std::to_string(r.range) + ")"});
        return verdict;
    }
    const auto described = check_encoder(kind, index, verdict);
    if (described) {
        const std::string named = std::string(kind_name(kind)) + " " + quoted(described->projector);
        residency::ResidencyPlan plan;
        check_fit(index, *described, ranges, limits, companion_policy(policy, host), budget, host, verdict, plan);
        if (verdict.fit) {
            for (const arch::MediaGraphFn graph : {described->encoder->graph, described->encoder->setup}) {
                if (graph == nullptr) continue;
                std::vector<kernels::Launch> launches;
                if (const auto r = graph(described->description, plan, launches); !r.ok()) {
                    verdict.blockers.push_back({Stage::Run, named + " cannot run this file: " +
                                                                std::string(to_string(r.error)) + " (" + r.subject +
                                                                ")"});
                    break;
                }
            }
        }
        if (const auto r = described->encoder->pairs(described->description, host.text, host.text_architecture);
            !r.ok()) {
            verdict.blockers.push_back({Stage::Run, named + " does not serve the model loaded (" + r.subject + ")"});
        }
        for (const std::string_view token : described->encoder->tokens) {
            if (!host.vocabulary.find(token)) {
                verdict.blockers.push_back({Stage::Run, "the model loaded has no " + quoted(token) + " token"});
            }
        }
    }
    check_formats(index, verdict, ranges);
    check_rows(index, verdict, ranges);
    return verdict;
}

}  // namespace bllm::preflight
