#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bllm::policy {

// Contract 12: policy.
//
// A model's measured configuration, from its entry in web/models.json. The
// values change when a model is measured (axis J); this header changes only
// when what can be configured changes.
//
// Two parts cross the boundary at different times. Cache precision, the
// memory budget and the stop tokens are fixed at load. Sampling settings
// depend on the mode a turn runs in, such as thinking or not, so the
// JavaScript side resolves them for the turn and passes them with each
// generate, together with the seed and the reply's token limit.
//
// An unmeasured model runs on the defaults below and is shown as unmeasured.
// The context offered is not policy: it is derived from the file, the memory
// budget and the cache precision.

enum class CachePrecision {
    F16,
    BF16,
    Q8_0,
};

struct LoadPolicy {
    CachePrecision cache_precision = CachePrecision::F16;
    // Bytes the harness may place on the device: weights, cache and
    // activations. WebGPU does not report device memory, so this is measured
    // per model on the devices it is listed for. The default covers a model of
    // about 1 GB with a working cache; a larger unmeasured model is rejected
    // at preflight's Fit stage, by name, rather than attempted.
    std::uint64_t memory_budget = 2ull * 1024 * 1024 * 1024;
    // Tokens a sliding-window layer's cache keeps beyond what one step needs,
    // so a turn can roll the cache back that far without restarting it:
    // enough to regenerate a long reply. A deeper rollback restarts the cache
    // from the first token.
    std::uint32_t rollback_reserve = 4096;
    // Stop tokens, by their text, beyond the end-of-generation tokens the
    // file names, where the model's generation config lists more than its
    // file carries (runtime/stops.h). Each must be one control token of the
    // vocabulary. None by default: an unmeasured model stops on its file's.
    std::vector<std::string> stop;
};

// Defaults are llama.cpp's, the most widely exercised settings for models
// without a card of their own.
struct SamplingSettings {
    float temperature = 0.8f;
    std::uint32_t top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
};

// The randomness of a run. Its own type, so a seed cannot be passed where a
// count or a position is meant.
enum class Seed : std::uint64_t {};

struct TurnPolicy {
    SamplingSettings sampling;
    Seed seed;
    // The most tokens the reply may draw, at least 1; the context offered
    // bounds it first, so the largest value asks for no other limit.
    std::uint32_t max_tokens = UINT32_MAX;
};

}  // namespace bllm::policy
