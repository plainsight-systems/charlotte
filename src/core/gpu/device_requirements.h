#pragma once

#include <cstdint>

// What this harness requires of a WebGPU device.
//
// This is a specification of our needs, not a report of any GPU's capability
// (WASM.10). Requesting the adapter's advertised maxima — which this code used
// to do — turns an explicit contract into an implicit one: the harness ends up
// depending on whatever the development machine happened to advertise, and
// fails elsewhere for reasons nobody can trace.
//
// Adapter limits remain useful as *diagnostics*. They are reported, never
// planned against. Planning uses the limits actually granted, read back after
// acquisition, because those are what validation enforces.
namespace bllm::gpu {

struct DeviceRequirements {
    std::uint64_t max_buffer_size;
    std::uint64_t max_storage_buffer_binding_size;
    std::uint32_t max_storage_buffers_per_shader_stage;
    std::uint32_t max_compute_invocations_per_workgroup;
    std::uint32_t max_compute_workgroups_per_dimension;
};

// The WebGPU spec defaults, deliberately and with every number traced to
// something the harness actually does.
//
// Requiring only the defaults is a product decision, for cross-browser
// compatibility: it runs anywhere WebGPU runs. For the same reason the device
// is requested with no optional feature — not `subgroups`, not `shader-f16` —
// though the target, Chrome on Apple silicon, offers both; each place that
// does without one says so. Raising any of these would exclude fully conformant devices, and should
// only happen against a measured benefit — see the sizing evidence below for
// why none of them currently binds.
inline constexpr DeviceRequirements kRequirements{
    // 320 MiB of Q4_0 weights across two suballocated buffers.
    .max_buffer_size = 256ull * 1024 * 1024,
    // Largest single bound range is the embedding's nibble stream, 74.2 MiB.
    .max_storage_buffer_binding_size = 128ull * 1024 * 1024,
    // A fused dequant-matmul binds nibbles, scales, input, output. Eight allows
    // two weight tensors per dispatch, which is more than the naive kernel uses.
    .max_storage_buffers_per_shader_stage = 8,
    // vector_add and the planned kernels use a 64-wide workgroup.
    .max_compute_invocations_per_workgroup = 256,
    .max_compute_workgroups_per_dimension = 65535,
};

// Sizing evidence for the numbers above, for the model this harness targets.
//
// These are compile-time checks on the *requirement*, not on any file. The
// loader separately verifies the actual model against the granted limits at
// runtime, because a different model is a different requirement.
namespace sizing {

inline constexpr std::uint64_t kVocabSize = 151936;
inline constexpr std::uint64_t kHiddenSize = 1024;
inline constexpr std::uint64_t kEmbeddingParams = kVocabSize * kHiddenSize;

// De-interleaved: nibbles are 4 bits per weight, scales 2 bytes per 32.
inline constexpr std::uint64_t kEmbeddingNibbleBytes = kEmbeddingParams / 2;
inline constexpr std::uint64_t kEmbeddingScaleBytes = (kEmbeddingParams / 32) * 2;

// The largest single bound range the harness will create. If a future model
// makes this exceed what we require, the requirement must change deliberately
// rather than the binding failing at load.
static_assert(kEmbeddingNibbleBytes <= kRequirements.max_storage_buffer_binding_size,
              "the largest bound range must fit inside the binding size we require");
static_assert(kEmbeddingScaleBytes <= kRequirements.max_storage_buffer_binding_size,
              "the scale stream must also fit the required binding size");

// A fused dequant-matmul binds: weight nibbles, weight scales, activations in,
// activations out.
inline constexpr std::uint32_t kBindingsPerMatmul = 4;
static_assert(kBindingsPerMatmul <= kRequirements.max_storage_buffers_per_shader_stage,
              "one matmul must be dispatchable within the required binding budget");

}  // namespace sizing

}  // namespace bllm::gpu
