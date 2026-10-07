#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/index.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
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
//     offered; and one set of working buffers that every layer reuses.
//     Optimization (practice): weights and cache in a few large buffers,
//     suballocated, rather than one allocation per tensor (GPU.9).
//   - Each working buffer is a buffer of its own. WebGPU tracks a dispatch's
//     use of a buffer across the whole buffer, not by range, and refuses a
//     buffer bound both writable and read-only in one dispatch; a kernel
//     reads one working buffer and writes another — the norm reads `output`
//     and writes `hidden` — so two working buffers in one buffer could not be
//     bound together. Weights are only read, and no dispatch both reads and
//     writes the cache, so those pools stay suballocated. Ten buffers instead
//     of one costs ten allocations at load and nothing a step; GPU.9 keeps
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
//   - Selection and the draw have four working buffers of their own
//     (sampler/sampler.h): `partials_a` and `partials_b`, which its passes
//     alternate between, each kCandidates (logit, token) pairs of 8 bytes for
//     every kSelectionTile of the vocabulary — 76 KB for Qwen3, 131 KB for
//     Gemma 3; `candidates`, the kCandidates kept, 512 bytes; and `sampled`,
//     the draw's 16-byte record.
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
//   - The fit counts every tensor. An output head stored as a byte-for-byte
//     copy of the token embedding — tied weights written twice — is marked a
//     candidate duplicate and given buffers of its own, which upload does not
//     create once it confirms the bytes match; a fit never depends on sharing
//     that has not been confirmed.

// The block of tokens one prefill step processes, and the shortest context
// worth offering.
inline constexpr std::uint32_t kPrefillBlock = 512;

// The candidates top-k selection keeps for the draw, and the entries one of
// its workgroups reduces to them (kernels/topk/topk.h): what its working
// buffers are sized by.
inline constexpr std::uint32_t kCandidates = 64;
inline constexpr std::uint32_t kSelectionTile = 1024;

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
};

struct PlanResult {
    PlanError error = PlanError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == PlanError::Ok; }
};

[[nodiscard]] PlanResult plan_residency(const gguf::TensorIndex& index,
                                        const model::ModelDescription& model,
                                        const DeviceLimits& limits,
                                        const policy::LoadPolicy& policy,
                                        ResidencyPlan& out);

}  // namespace bllm::residency
