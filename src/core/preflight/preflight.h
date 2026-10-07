#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::preflight {

// Axis I: applies the capability table; it holds no criteria of its own.
//
// How far this build can take a model, judged from its tensor index before
// any weight byte is fetched. A model passes through ordered stages, each
// needing every stage before it:
//
//   Read       the file is GGUF and its index reads (decided by the reader;
//              a file that does not read has no verdict)
//   Download   the file can be fetched and cached — every readable file can
//   Describe   its architecture is implemented and can describe this file
//   Fit        its residency plan fits the granted limits and the budget
//   Upload     every weight's format is listed and its rows are a whole
//              number of unpack's groups, as routes require; its weights are
//              written to the GPU, every write confirmed on a live device
//              (residency/upload.h); a diagnostic build reads every byte back
//              (residency/upload_check.h)
//   Run        it generates: its architecture's graph builds
//              (arch/architecture.h), its tokenizer loads from the header
//              (load_tokenizer below), and its stop set resolves
//
// The verdict lists blockers, each naming the stage it stops and why: the
// architecture, each unsupported format with how many tensors use it and the
// first, the graph's refusal of a layer, a cache format no kernel writes, the
// tokenizer or pre-tokenizer, the tokenizer's load or the stop set's
// failure, or a stage this build does not implement. The graph is built only
// for a model that fits, over the plan Fit made; the tokenizer is loaded only
// where its algorithm and pre-tokenizer are listed, at 20 to 56 ms for the
// listed models, once a verdict. Every check that can run does, so the verdict says everything a
// model still needs, not only what stops the next stage. The stage reached is
// derived from the blockers, never stored beside them.

enum class Stage {
    Read,
    Download,
    Describe,
    Fit,
    Upload,
    Run,
};

[[nodiscard]] constexpr std::string_view to_string(Stage stage) noexcept {
    switch (stage) {
        case Stage::Read: return "read";
        case Stage::Download: return "download";
        case Stage::Describe: return "describe";
        case Stage::Fit: return "fit";
        case Stage::Upload: return "upload";
        case Stage::Run: return "run";
    }
    return "unknown";
}

// The furthest stage this build implements. Every later stage is blocked, by
// name, whatever the model: a verdict never claims a stage that does not
// exist. Advanced when the next stage is built.
inline constexpr Stage kImplementedThrough = Stage::Run;

// What stops `stage`. Never Read or Download: those depend only on the
// reader.
struct Blocker {
    Stage stage;
    std::string detail;
};

// A tensor the plan marks as possibly a byte-for-byte copy of another, with
// both byte ranges in the file, for the page to compare before upload
// (web/duplicates.js, residency/routes.h).
struct DuplicateCandidate {
    gguf::TensorId tensor;
    gguf::TensorId copies;
    std::uint64_t offset;
    std::uint64_t copies_offset;
    std::uint64_t length;
};

// What the residency plan found for a model that fits.
struct FitSummary {
    std::uint64_t weight_bytes;
    std::uint64_t cache_bytes;
    std::uint64_t scratch_bytes;
    std::uint64_t total_bytes;
    std::uint64_t memory_budget;
    std::uint32_t context_offered;
    std::uint32_t trained_context;
    std::size_t buffer_count;
    // Empty for a model whose output head reads its embedding, as every
    // listed model's does.
    std::vector<DuplicateCandidate> duplicates;
};

struct Verdict {
    std::vector<Blocker> blockers;
    // Set when the model reaches Fit.
    std::optional<FitSummary> fit;

    // The furthest stage reached: the one before the earliest stage blocked.
    [[nodiscard]] Stage reached() const noexcept {
        Stage earliest = Stage::Run;
        bool blocked = false;
        for (const Blocker& b : blockers) {
            if (!blocked || b.stage < earliest) earliest = b.stage;
            blocked = true;
        }
        if (!blocked) return Stage::Run;
        return static_cast<Stage>(static_cast<int>(earliest) - 1);
    }
};

// What a load carries out: the description and the plan preflight judged
// Fit, made again from the same index and limits. Returns what stops it,
// worded as preflight's blocker would be, or an empty string.
[[nodiscard]] std::string plan_load(const gguf::TensorIndex& index, const residency::DeviceLimits& limits,
                                    const policy::LoadPolicy& policy, model::ModelDescription& description,
                                    residency::ResidencyPlan& plan);

// The model's tokenizer: the algorithm the file's tokenizer.ggml.model names
// and the pre-tokenizer its tokenizer.ggml.pre names, found in the
// capability table, loaded from the bytes the index was read from — the
// vocabulary and merges lie in the header (tokenizer.h's LoadFn). Returns
// what stops it, worded as a Run blocker, or the load's own failure named; an
// empty string, and `out` set, otherwise. Preflight runs it on the header it
// was given, with the stop set resolved over the vocabulary it loads
// (runtime/stops.h), and names either failure as a Run blocker, so a file
// whose tokenizer or stop tokens cannot be made is refused before its
// weights are fetched; the load runs it again on the same bytes.
[[nodiscard]] std::string load_tokenizer(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                         std::unique_ptr<tokenizer::Tokenizer>& out);

// `source` is the bytes the index was read from: the header, where the
// tokenizer's arrays lie.
[[nodiscard]] Verdict preflight(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                const residency::DeviceLimits& limits,
                                const policy::LoadPolicy& policy);

}  // namespace bllm::preflight
