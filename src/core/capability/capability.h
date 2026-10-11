#pragma once

#include <string_view>

#include "core/gguf/types.h"

namespace bllm::arch { struct Architecture; struct MediaEncoder; enum class MediaKind; }
namespace bllm::formats { class Format; }
namespace bllm::tokenizer { struct Algorithm; struct PreTokenizer; }

namespace bllm::capability {

// Axis I: changes when the harness gains or loses an implementation, and for
// no other reason.
//
// Contract 3. One table from each identifier a GGUF file carries to the
// implementation that runs it. Supported means present here. Preflight, format
// dispatch, tokenizer selection and graph selection all find implementations
// through these lookups, so what the picker reports and what the loader runs
// cannot disagree. There is no second list.
//
// Several identifiers may name one implementation (principle 2). A lookup
// returns the implementation, or null if the harness has none; null is not an
// error here, and preflight turns it into a blocker that names what is
// missing.

[[nodiscard]] const arch::Architecture* find_architecture(std::string_view general_architecture) noexcept;

// A companion file's media encoder of `kind`, by the value of that kind's
// projector key (arch/media_encoder.h): clip.vision.projector_type or
// clip.audio.projector_type, the identifier that names the encoder, where
// its general.architecture, "clip", names none. A type the table holds for
// the other kind is not found.
[[nodiscard]] const arch::MediaEncoder* find_media_encoder(arch::MediaKind kind,
                                                           std::string_view projector_type) noexcept;
[[nodiscard]] const formats::Format* find_format(gguf::TensorType type) noexcept;

[[nodiscard]] const tokenizer::Algorithm* find_tokenizer(std::string_view tokenizer_model) noexcept;

[[nodiscard]] const tokenizer::PreTokenizer* find_pretokenizer(std::string_view tokenizer_pre) noexcept;

}  // namespace bllm::capability
