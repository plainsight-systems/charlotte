#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/index.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/file_ranges.h"
#include "core/residency/weight_view.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Contract 5: the residency plan. A pure function. Tensor index, model
// description, device limits and load policy in; buffers, each tensor's place
// in them, the cache, the working buffers, and the context offered, out. No
// GPU, no fetch, no browser. Preflight's Fit stage runs it before any weight
// byte is downloaded, and upload carries it out afterwards.
//
//   - Three pools, never sharing a buffer, because their lifetimes differ
//     (GPU.9): the weights, packed in file order into as few buffers as the
//     limits allow; the KV cache, keys and values per layer at the context
//     offered, for a graph that keeps one; and one set of working buffers
//     that every layer reuses.
//   - Which working buffers, and how large, is the graph's: the caller hands
//     the planner a WorkingSet its graph declared (graph/graph.h), and the
//     planner places each need in order, knowing no kernel by name. Which
//     buffers a graph binds changes when a graph or a kernel needs one, axis
//     A or E; where they lie changes with the limits, axis D
//     (docs/architecture/change-axes.md).
//     Optimization (practice): weights and cache in a few large buffers,
//     suballocated, rather than one allocation per tensor (GPU.9).
//   - Each working buffer is a buffer of its own. WebGPU tracks a dispatch's
//     use of a buffer across the whole buffer, not by range, and refuses a
//     buffer bound both writable and read-only in one dispatch; a kernel
//     reads one working buffer and writes another — the norm reads `output`
//     and writes `hidden` — so two working buffers in one buffer could not be
//     bound together. Weights are only read, and no dispatch both reads and
//     writes the cache, so those pools stay suballocated. Fifteen buffers
//     instead of one costs fifteen allocations at load and nothing a step; GPU.9 keeps
//     dedicated allocations for where an API requirement justifies them, and
//     this is one.
//   - Limits are the ones the device granted, never the adapter's advertised
//     maxima (WASM.10). Packing works at WebGPU's default limits.
//   - A weight larger than one storage binding is split by rows. Every offset
//     is aligned to the device's storage-offset alignment, and every range is
//     padded to a multiple of 4 bytes, as a storage binding requires.
//   - A layer's Q, K and V, and its gate and up, are each a fused group
//     when its members share a format and an input width and are one piece
//     each: the plan keeps the group in one buffer, opening a new buffer
//     before it when the open one cannot hold the span from its first
//     member to its last, and records the group, its members in output
//     order and that span, which a fused product binds
//     (kernels/matmul/matmul.h). The members stay where file order puts
//     them, with whatever the file puts between them, so upload is the
//     same; a span wider than a binding is not a group, and its members are
//     multiplied apart.
//   - The working buffers hold a prefill block of kPrefillBlock tokens at f32.
//     Attention never stores a block-by-context matrix of scores: at 512
//     tokens, 32 heads and a 40,000-token context that is 2.6 GB. Kernels work
//     within these buffers.
//     Optimization (browser): no 128 MiB binding could hold that matrix, nor
//     a 2 GiB budget; attention computes in tiles, as FlashAttention does.
//   - A layer's cache holds one slot per token of the context offered, or, for
//     a sliding-window layer, never more than its window, a prefill block and
//     the policy's rollback reserve. A step writes up to kPrefillBlock new
//     tokens while its first query still reads a window behind them, so a
//     ring of window + kPrefillBlock is the least a step needs; the reserve is
//     how far a turn can roll back. Position p lives in slot p mod slots.
//     Optimization (browser): Gemma 3's cache falls from 832 MiB to 238 MiB
//     under a budget the browser cannot measure; llama.cpp sizes its
//     sliding-window cache the same way.
//   - The context offered is the largest that fits the memory budget and the
//     binding limit, capped at the context the model was trained for.
//   - A load may read part of its file (file_ranges.h): a companion's
//     header and its encoder's span. Then only the tensors wholly inside
//     the ranges are placed; every other keeps its entry, not present, with
//     no place and no route, so the plan's tensors stay indexed by id; and a
//     tensor the description names that lies outside is refused,
//     MissingTensorBytes, before anything is placed. A fused group whose
//     span holds a tensor not read is not a group.
//   - A product span: a weight and the F32 tensors its product reads from
//     the weight's own binding — an audio encoder's clamp bounds, four one-
//     float tensors around each clipped weight, and its output projection's
//     bias (kernels/matmul/matmul.h) — kept in one buffer within one
//     binding, as a fused group is, opening a new buffer before the span
//     when the open one cannot hold it, and recorded: the weight, the
//     tensors by role, and the range from the first's first byte to the
//     last's last. Unlike a group's, its members' formats differ; the
//     weight is one piece. The description names the spans
//     (arch/gemma4a/gemma4a.h); the plan refuses, naming the weight, a span
//     wider than a binding or holding a tensor not read, since the product
//     could not bind its bounds otherwise. In EmbeddingGemma 2's file a
//     span is its weight and 4 × 32 bytes, the tensors' alignment.
//   - A row is every dimension but the last (formats/format.h), so a
//     convolution's weight splits, where it must, between the rows its
//     product reads.
//   - The fit counts every tensor. An output head stored as a byte-for-byte
//     copy of the token embedding — tied weights written twice — is marked a
//     candidate duplicate and given buffers of its own, which upload does not
//     create once it confirms the bytes match; a fit never depends on sharing
//     that has not been confirmed.

// The block of tokens one prefill step processes, and the shortest context
// worth offering.
inline constexpr std::uint32_t kPrefillBlock = 512;

// The keys an attention chunk holds (kernels/attention/attention.h), by
// which the partial buffers are sized.
inline constexpr std::uint32_t kChunkKeys = 64;

// The query rows a split step's partial buffers hold for a model of
// `trained_context`: a prefill block's, or every chunk of the context where
// that is more, so a decode step splits at every position any context the
// plan offers reaches.
[[nodiscard]] constexpr std::uint32_t partial_rows_for(std::uint32_t trained_context) noexcept {
    const std::uint32_t chunks = trained_context / kChunkKeys + (trained_context % kChunkKeys != 0 ? 1 : 0);
    return chunks > kPrefillBlock ? chunks : kPrefillBlock;
}

// The candidates top-k selection keeps for the draw, and the entries one of
// its workgroups reduces to them (kernels/topk/topk.h): what its working
// buffers are sized by.
inline constexpr std::uint32_t kCandidates = 64;
inline constexpr std::uint32_t kSelectionTile = 1024;

// A working buffer a graph binds: its purpose, which the graph finds it by,
// and its rows of `width` f32 values.
struct ScratchNeed {
    std::string_view purpose;
    std::uint64_t rows;
    std::uint64_t width;
};

// What a graph asks the planner to place beside the weights: its working
// buffers, in order; whether it keeps a KV cache; and, for one that splits
// attention by key chunks, the partial buffers' query rows
// (kernels/attention/attention.h), or 0.
struct WorkingSet {
    std::vector<ScratchNeed> needs;
    bool cache = true;
    std::uint32_t partial_rows = 0;
    // The most rows one step holds, where a graph's step is not a prefill
    // block of tokens: a vision encoder's patches (graph/vision.h). 0 for a
    // step of tokens.
    std::uint32_t step_rows = 0;
};

// The limits the device was granted. How many bytes the plan may place on the
// device is not among them — WebGPU does not report device memory — and comes
// from load policy instead.
struct DeviceLimits {
    std::uint64_t max_buffer_size;
    std::uint64_t max_storage_binding_size;
    std::uint32_t storage_offset_alignment;
};

enum class Pool {
    Weights,
    Cache,
    Scratch,
};

struct PlannedBuffer {
    Pool pool;
    std::uint64_t size;
};

// A bound range of one planned buffer. Its length is what is bound: the bytes
// it holds rounded up to 4, since a storage binding's size must be a multiple
// of 4 (WebGPU). A weight piece's bytes in the file are its rows times the
// row's bytes.
struct BufferRange {
    BufferIndex buffer;
    std::uint64_t offset;
    std::uint64_t length;
};

struct PlannedTensor {
    gguf::TensorId tensor;
    WeightView view;
    // Set when this tensor may be a byte-for-byte copy of an earlier one.
    std::optional<gguf::TensorId> candidate_duplicate_of;
    // False for a tensor whose bytes lie outside the ranges the load reads
    // (file_ranges.h): it has no place, no route, and no description names
    // it. Every tensor keeps an entry, so the plan's tensors stay indexed by
    // tensor id.
    bool present = true;
};

// One layer's cache storage. Position p lives in slot p mod slots; for a
// full-attention layer slots is the context offered, so no slot is reused.
struct PlannedCacheLayer {
    BufferRange keys;
    BufferRange values;
    std::uint32_t slots;
};

// A working buffer, named for what it holds.
struct PlannedScratch {
    std::string_view purpose;
    BufferRange range;
};

// A fused group: its members in output order — Q, K, V, or gate, up — and
// the range from the first member's first byte to the last member's last.
struct PlannedGroup {
    std::array<gguf::TensorId, 3> members;
    std::uint32_t count;
    BufferRange span;
};

struct ResidencyPlan {
    std::vector<PlannedBuffer> buffers;
    std::vector<PlannedTensor> tensors;
    std::vector<PlannedGroup> groups;
    std::vector<PlannedCacheLayer> cache;
    // The type the cache is stored in, from the policy's precision; set by
    // plan_residency before anything else, so a failed plan has it too.
    gguf::TensorType cache_type{};
    std::vector<PlannedScratch> scratch;
    // The partial buffers' query rows, as the working set gave them.
    std::uint32_t partial_rows = 0;
    std::uint32_t context_offered = 0;
    // Bytes in each pool, alignment padding included, and their sum. On
    // ExceedsBudget, total_bytes is what the shortest context worth offering
    // would need, so a rejection can state needed against available.
    std::uint64_t weight_bytes = 0;
    std::uint64_t cache_bytes = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t total_bytes = 0;
};

enum class PlanError {
    Ok,
    // Weights and working buffers leave too little of the budget for a
    // kPrefillBlock-token cache.
    ExceedsBudget,
    // One row of a weight is wider than a storage binding, so no split by
    // rows can place it. The subject names the tensor.
    RowExceedsBinding,
    // A working buffer is wider than a binding. The subject names it.
    ScratchExceedsBinding,
    // The cache precision stores whole blocks, and a layer's head dimension
    // is not a whole number of them. The subject names the layer.
    UnsupportedCachePrecision,
    // A size that overflows 64 bits: a file no device could hold.
    Overflow,
    // A tensor the description names lies outside the ranges the load reads
    // (file_ranges.h). The subject names it.
    MissingTensorBytes,
};

struct PlanResult {
    PlanError error = PlanError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == PlanError::Ok; }
};

// Plans `model` for `limits` and `policy`, placing the working set its
// graph declared. Without a cache the context offered is the smaller of
// kPrefillBlock and the trained context — the most one step holds — and the
// fit is the weights' and working buffers' bytes against the budget; a
// working set that gives its step's rows offers those, a vision encoder's
// patches. `ranges`, which check_ranges accepted, are the bytes the load reads: only
// the tensors wholly inside them are placed, and one the description names
// that is not is refused, MissingTensorBytes. Empty: the whole file.
[[nodiscard]] PlanResult plan_residency(const gguf::TensorIndex& index,
                                        const model::ModelDescription& model,
                                        const DeviceLimits& limits,
                                        const policy::LoadPolicy& policy,
                                        const WorkingSet& working,
                                        ResidencyPlan& out,
                                        std::span<const FileRange> ranges = {});

}  // namespace bllm::residency
