#pragma once

// Test-only: EmbeddingGemma 2's text model and its mmproj's encoders at
// Q8_0, fetched as test data, pinned by SHA-256 (tests/fixtures/
// external.json), uploaded on one device as the harness uploads them: the
// text model whole, the mmproj by its header and one kind's ranges alone
// (residency/file_ranges.h). For the GPU tests that embed media.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/arch/gemma4a/gemma4a.h"
#include "core/arch/gemma4v/gemma4v.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/graph/audio.h"
#include "core/graph/vision.h"
#include "core/kernels/program.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/runtime/media_stage.h"
#include "support/pump.h"
#include "support/test_data.h"
#include "support/upload.h"

namespace bllm::testing {

inline constexpr std::size_t kChunk16 = 16u << 20;
inline constexpr residency::DeviceLimits kCompanionLimits{256ull << 20, 128ull << 20, 256};
inline constexpr std::uint64_t kMmprojSize = 554'821'024;
inline constexpr std::uint64_t kMmprojHeader = 58'656;
inline constexpr residency::FileRange kVisionRanges[] = {{0, kMmprojHeader}, {328'492'448, kMmprojSize}};
inline constexpr residency::FileRange kAudioRanges[] = {{0, 328'492'448}};

inline kernels::Binding binding_of(const residency::BufferRange& r) { return {r.buffer, r.offset, r.length}; }

struct Text {
    std::string bytes;
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    kernels::Binding embedding{};
    residency::BufferRange hidden{};
};

inline std::unique_ptr<Text> upload_text(WGPUInstance instance, const gpu::Device& device) {
    auto m = std::make_unique<Text>();
    m->bytes = load_test_data("models/embeddinggemma-2-q8_0.gguf");
    gguf::MemoryByteSource source{std::as_bytes(std::span{m->bytes}), m->bytes.size()};
    REQUIRE(gguf::read_index(source, m->index).error == gguf::ReadError::Ok);
    residency::ResidencyPlan plan;
    const std::string stop =
        preflight::plan_load(m->index, kCompanionLimits, policy::LoadPolicy{}, m->description, plan);
    REQUIRE_MESSAGE(stop.empty(), stop);
    Ready ready;
    residency::Upload::begin(device, m->index, plan, m->bytes.size(), capability::find_format, {}, kChunk16, on_ready,
                             &ready);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    m->upload = std::move(ready.upload);
    const auto all = std::as_bytes(std::span{m->bytes});
    for (std::size_t at = 0; at < all.size(); at += kChunk16) {
        Reported accepted;
        m->upload->write(at, all.subspan(at, std::min(kChunk16, all.size() - at)), on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.error == residency::UploadError::Ok);
    }
    REQUIRE(finish(instance, *m->upload) == residency::UploadError::Ok);
    const residency::ResidencyPlan& planned = m->upload->plan();
    REQUIRE(arch::encoder_graph(*capability::find_architecture("gemma-embedding2"))(m->description, planned,
                                                                                   m->launches)
                .ok());
    for (const residency::PlannedScratch& s : planned.scratch) {
        if (s.purpose == "embedding") m->embedding = {s.range.buffer, s.range.offset, 768 * sizeof(float)};
        if (s.purpose == "hidden") m->hidden = s.range;
    }
    return m;
}

inline std::unique_ptr<tokenizer::Tokenizer> tokenizer_of(const Text& m) {
    gguf::MemoryByteSource source{std::as_bytes(std::span{m.bytes}), m.bytes.size()};
    std::unique_ptr<tokenizer::Tokenizer> t;
    REQUIRE(preflight::load_tokenizer(source, m.index, t).empty());
    return t;
}

// An encoder's three tokens, by text, from the text model's vocabulary.
inline runtime::MediaTokens media_tokens_of(const tokenizer::Tokenizer& t,
                                            const std::array<std::string_view, 3>& texts) {
    const auto placeholder = t.vocabulary().find(texts[0]);
    const auto begin = t.vocabulary().find(texts[1]);
    const auto end = t.vocabulary().find(texts[2]);
    REQUIRE(placeholder);
    REQUIRE(begin);
    REQUIRE(end);
    return {*placeholder, *begin, *end};
}

// The mmproj's index, from its header.
inline void read_mmproj_index(std::ifstream& in, gguf::TensorIndex& index) {
    std::string header(kMmprojHeader, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    gguf::MemoryByteSource source{std::as_bytes(std::span{header}), kMmprojSize};
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
}

inline std::ifstream open_mmproj() {
    const std::string path = std::string(BLLM_TEST_DATA_DIR) + "/models/mmproj-embeddinggemma-2-q8_0.gguf";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing test data: " << path << " (run make test-data)");
    return in;
}

// The mmproj's upload by `ranges` alone, its plan `plan`.
inline std::unique_ptr<residency::Upload> upload_ranges(WGPUInstance instance, const gpu::Device& device,
                                                        std::ifstream& in, const gguf::TensorIndex& index,
                                                        const residency::ResidencyPlan& plan,
                                                        std::span<const residency::FileRange> ranges) {
    Ready ready;
    residency::Upload::begin(device, index, plan, kMmprojSize, capability::find_format, {}, kChunk16, on_ready, &ready,
                             ranges);
    pump_until(instance, ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == residency::UploadError::Ok, ready.subject);
    std::unique_ptr<residency::Upload> upload = std::move(ready.upload);
    std::vector<char> chunk(kChunk16);
    for (const residency::FileRange& r : ranges) {
        for (std::uint64_t at = r.begin; at < r.end; at += kChunk16) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk16, r.end - at));
            in.seekg(static_cast<std::streamoff>(at));
            in.read(chunk.data(), static_cast<std::streamsize>(n));
            REQUIRE(in.gcount() == static_cast<std::streamsize>(n));
            Reported accepted;
            upload->write(at, std::as_bytes(std::span(chunk).first(n)), on_reported, &accepted);
            pump_until(instance, accepted.done, "a chunk's acceptance");
            REQUIRE(accepted.error == residency::UploadError::Ok);
        }
    }
    REQUIRE(finish(instance, *upload) == residency::UploadError::Ok);
    return upload;
}

struct VisionModel {
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches;
    residency::BufferRange pixels{}, soft_tokens{};
};

// The vision encoder, planned for `patches` rows a step, its feed-forward's
// activation `activation`.
inline std::unique_ptr<VisionModel> upload_vision(WGPUInstance instance, const gpu::Device& device,
                                                  model::FeedForwardActivation activation, std::uint32_t patches) {
    auto v = std::make_unique<VisionModel>();
    std::ifstream in = open_mmproj();
    read_mmproj_index(in, v->index);
    REQUIRE(arch::kGemma4Vision.describe(v->index, v->description).ok());
    v->description.activation = activation;
    residency::ResidencyPlan plan;
    REQUIRE(residency::plan_residency(v->index, v->description, kCompanionLimits, policy::LoadPolicy{},
                                      graph::vision_working_set(v->description, patches), plan, kVisionRanges)
                .ok());
    v->upload = upload_ranges(instance, device, in, v->index, plan, kVisionRanges);
    REQUIRE(arch::kGemma4Vision.graph(v->description, v->upload->plan(), v->launches).ok());
    for (const residency::PlannedScratch& s : v->upload->plan().scratch) {
        if (s.purpose == "pixels") v->pixels = s.range;
        if (s.purpose == "soft_tokens") v->soft_tokens = s.range;
    }
    return v;
}

struct AudioModel {
    gguf::TensorIndex index;
    model::ModelDescription description{};
    std::unique_ptr<residency::Upload> upload;
    std::vector<kernels::Launch> launches, setup;
    residency::BufferRange mel{}, positions{}, soft_tokens{};
};

// The audio encoder, planned for `rows` rows a clip.
inline std::unique_ptr<AudioModel> upload_audio(WGPUInstance instance, const gpu::Device& device,
                                                std::uint32_t rows) {
    auto a = std::make_unique<AudioModel>();
    std::ifstream in = open_mmproj();
    read_mmproj_index(in, a->index);
    REQUIRE(arch::kGemma4Audio.describe(a->index, a->description).ok());
    residency::ResidencyPlan plan;
    REQUIRE(residency::plan_residency(a->index, a->description, kCompanionLimits, policy::LoadPolicy{},
                                      graph::audio_working_set(a->description, rows), plan, kAudioRanges)
                .ok());
    a->upload = upload_ranges(instance, device, in, a->index, plan, kAudioRanges);
    REQUIRE(arch::kGemma4Audio.graph(a->description, a->upload->plan(), a->launches).ok());
    REQUIRE(arch::kGemma4Audio.setup(a->description, a->upload->plan(), a->setup).ok());
    for (const residency::PlannedScratch& s : a->upload->plan().scratch) {
        if (s.purpose == "mel") a->mel = s.range;
        if (s.purpose == "positions") a->positions = s.range;
        if (s.purpose == "soft_tokens") a->soft_tokens = s.range;
    }
    return a;
}

// A link from an encoder's `soft_tokens` into the text program's `hidden`.
inline std::unique_ptr<kernels::ProgramLink> link_of(const Text& m, const residency::Upload& encoder,
                                                     const residency::BufferRange& soft_tokens) {
    return std::make_unique<kernels::ProgramLink>(encoder, binding_of(soft_tokens), *m.upload, binding_of(m.hidden));
}

}  // namespace bllm::testing
