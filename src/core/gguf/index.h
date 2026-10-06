#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/types.h"

namespace bllm::gguf {

// Contract 2: the tensor index.
//
// What the reader produces and every later stage reads.
//
//   - Immutable once read. read_index fills it, and every accessor is const.
//   - Complete. Every tensor in the file, including formats the harness cannot
//     run. Whether a format runs is the capability table's question; the index
//     records what the file says.
//   - Scalar and string metadata is decoded during the read. A missing key and
//     a key of the wrong type are different errors because they are different
//     defects: one is a file that lacks a field, the other a file that
//     misdeclares it.
//   - Arrays are located, not decoded. A vocabulary is ~150,000 strings and
//     only the tokenizer reads it, from the byte source.

// A tensor's position in the index. Its own type, so a tensor identifier
// cannot be passed where a layer or a token is meant (I.4).
enum class TensorId : std::uint32_t {};

enum class MetadataError {
    Ok,
    MissingKey,
    WrongType,
};

// Where an array value lives in the file: its elements, after the header
// that declares their type and count.
struct ArrayLocation {
    ValueType element_type;
    std::uint64_t element_count;
    ByteRange bytes;
};

// One metadata value as the file stores it. Integers widen to 64 bits and
// floats to double; `type` keeps what the file declared, so an accessor can
// refuse a key of the wrong type.
struct MetadataEntry {
    std::string key;
    ValueType type;
    std::variant<std::uint64_t, std::int64_t, double, bool, std::string, ArrayLocation> value;
};

class TensorIndex {
public:
    // An empty index. read_index fills one.
    TensorIndex() = default;

    [[nodiscard]] std::span<const TensorEntry> tensors() const noexcept { return tensors_; }

    // Precondition: `id` came from this index.
    [[nodiscard]] const TensorEntry& tensor(TensorId id) const noexcept {
        return tensors_[static_cast<std::size_t>(id)];
    }

    // The tensor named `name`: a binary search of the ids in name order, about
    // nine comparisons for a model's 340 tensors, where a scan compared names
    // with every tensor for each of describe's few hundred lookups.
    // Optimization (practice): CDSA.11 points lookups at a flat hash table,
    // which the standard library lacks; the sorted ids are contiguous
    // (CACHE.3), built once by read_index.
    [[nodiscard]] std::optional<TensorId> find(std::string_view name) const noexcept;

    [[nodiscard]] MetadataError read_u32(std::string_view key, std::uint32_t& out) const noexcept;
    [[nodiscard]] MetadataError read_u64(std::string_view key, std::uint64_t& out) const noexcept;
    [[nodiscard]] MetadataError read_f32(std::string_view key, float& out) const noexcept;
    [[nodiscard]] MetadataError read_bool(std::string_view key, bool& out) const noexcept;
    // `out` views storage owned by the index and lives as long as it does.
    [[nodiscard]] MetadataError read_string(std::string_view key, std::string_view& out) const noexcept;
    [[nodiscard]] MetadataError read_array(std::string_view key, ArrayLocation& out) const noexcept;

private:
    friend ReadResult read_index(ByteSource& source, TensorIndex& out);

    [[nodiscard]] const MetadataEntry* entry(std::string_view key) const noexcept;

    std::vector<TensorEntry> tensors_;
    std::vector<TensorId> by_name_;   // every tensor's id, in name order
    std::vector<MetadataEntry> metadata_;
};

}  // namespace bllm::gguf
