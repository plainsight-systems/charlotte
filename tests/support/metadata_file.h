#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/types.h"

// Builds, in memory, a GGUF file holding metadata arrays and no tensors —
// TEST SUPPORT ONLY. For tests whose data reads best written out in the test
// itself, such as a small vocabulary and its merges. Files that exercise the
// reader's handling of malformed bytes come from tools/make_fixture_gguf.py.
namespace bllm::testing {

class MetadataFile {
public:
    MetadataFile& strings(std::string_view key, std::initializer_list<std::string_view> values) {
        array_header(key, gguf::ValueType::String, values.size());
        for (const std::string_view v : values) string(v);
        ++entries_;
        return *this;
    }

    // The same, for a list built in the test.
    MetadataFile& strings(std::string_view key, std::span<const std::string> values) {
        array_header(key, gguf::ValueType::String, values.size());
        for (const std::string& v : values) string(v);
        ++entries_;
        return *this;
    }

    MetadataFile& int32s(std::string_view key, std::span<const std::int32_t> values) {
        array_header(key, gguf::ValueType::Int32, values.size());
        for (const std::int32_t v : values) put(&v, sizeof v);
        ++entries_;
        return *this;
    }

    MetadataFile& int32s(std::string_view key, std::initializer_list<std::int32_t> values) {
        array_header(key, gguf::ValueType::Int32, values.size());
        for (const std::int32_t v : values) put(&v, sizeof v);
        ++entries_;
        return *this;
    }

    MetadataFile& float32s(std::string_view key, std::span<const float> values) {
        array_header(key, gguf::ValueType::Float32, values.size());
        for (const float v : values) put(&v, sizeof v);
        ++entries_;
        return *this;
    }

    MetadataFile& uint32(std::string_view key, std::uint32_t value) {
        string(key);
        const auto type = static_cast<std::uint32_t>(gguf::ValueType::UInt32);
        put(&type, sizeof type);
        put(&value, sizeof value);
        ++entries_;
        return *this;
    }

    MetadataFile& boolean(std::string_view key, bool value) {
        string(key);
        const auto type = static_cast<std::uint32_t>(gguf::ValueType::Bool);
        const std::uint8_t byte = value ? 1 : 0;
        put(&type, sizeof type);
        put(&byte, sizeof byte);
        ++entries_;
        return *this;
    }

    // The whole file: the header, the entries, and padding to where tensor
    // data would start, which must lie inside the file.
    [[nodiscard]] std::vector<std::byte> bytes() const {
        std::vector<std::byte> out;
        const auto append = [&](const void* p, std::size_t n) {
            const auto* b = static_cast<const std::byte*>(p);
            out.insert(out.end(), b, b + n);
        };
        const std::uint32_t version = 3;
        const std::uint64_t tensors = 0;
        append("GGUF", 4);
        append(&version, sizeof version);
        append(&tensors, sizeof tensors);
        append(&entries_, sizeof entries_);
        out.insert(out.end(), body_.begin(), body_.end());
        out.resize((out.size() + gguf::kDefaultAlignment - 1) / gguf::kDefaultAlignment * gguf::kDefaultAlignment);
        return out;
    }

private:
    void put(const void* p, std::size_t n) {
        const auto* b = static_cast<const std::byte*>(p);
        body_.insert(body_.end(), b, b + n);
    }
    void string(std::string_view s) {
        const std::uint64_t length = s.size();
        put(&length, sizeof length);
        put(s.data(), s.size());
    }
    void array_header(std::string_view key, gguf::ValueType element, std::size_t count) {
        string(key);
        const auto array = static_cast<std::uint32_t>(gguf::ValueType::Array);
        const auto type = static_cast<std::uint32_t>(element);
        const std::uint64_t n = count;
        put(&array, sizeof array);
        put(&type, sizeof type);
        put(&n, sizeof n);
    }

    std::vector<std::byte> body_;
    std::uint64_t entries_ = 0;
};

}  // namespace bllm::testing
