#include "core/residency/piece_writer.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "core/gguf/checked.h"

namespace bllm::residency {
namespace {

using formats::DeviceLayout;

constexpr std::uint64_t round_up4(std::uint64_t n) { return (n + 3) / 4 * 4; }

// Where stream `s` of a piece of `blocks` blocks starts, from the piece's
// start: after every earlier stream's fields. Each earlier stream is a
// multiple of 4 wide (device_layout.h), so every stream starts on a word.
std::uint64_t stream_start(const DeviceLayout& layout, std::size_t s, std::uint64_t blocks) {
    std::uint64_t before = 0;
    for (std::size_t i = 0; i < s; ++i) before += layout.streams[i].width;
    return before * blocks;
}

// Copies one field of every block in `run` into `out`, back to back, and
// returns the bytes copied.
template <std::size_t Width>
std::size_t gather_fixed(std::span<const std::byte> run, std::size_t block_bytes, std::size_t offset,
                         std::span<std::byte> out) {
    std::byte* to = out.data();
    for (std::size_t b = 0; b < run.size(); b += block_bytes, to += Width) {
        std::memcpy(to, run.data() + b + offset, Width);
    }
    return static_cast<std::size_t>(to - out.data());
}

// Optimization (practice): every field a listed layout has is one of these
// widths, so its copy has a size the compiler knows and inlines as a few
// loads and stores. Copied at a width known only at run time, each field
// was a call to memcpy: 21 million calls a stream for a Q4_0 model's
// weights. GDSA.6 prices this stage against memcpy of the same bytes, which
// a gather of small fields can reach about half of. make bench
// (bench/piece_writer_bench.cpp), 360 MiB, native release, Apple M3 Max:
//   Q4_0  83.3 ms (4.5 GB/s)  ->  11.9 ms (31.8 GB/s)
//   Q4_1 101.4 ms (3.7 GB/s)  ->  11.1 ms (34.2 GB/s)
//   Q8_0  49.6 ms (7.6 GB/s)  ->  10.7 ms (35.4 GB/s)
//   Q6_K  33.2 ms (11.4 GB/s) ->  13.7 ms (27.6 GB/s), before Q6_K was
//         repacked instead (formats/q6_k/q6_k_repack.cpp)
//   memcpy of the same bytes: 63 GB/s
std::size_t gather(std::span<const std::byte> run, std::size_t block_bytes, formats::Stream stream,
                   std::span<std::byte> out) {
    // A stream that is the whole block (F32) is the run itself, in order.
    // std::copy, not memcpy: an empty run's data may be null.
    if (stream.offset == 0 && stream.width == block_bytes) {
        std::copy(run.begin(), run.end(), out.begin());
        return run.size();
    }
    switch (stream.width) {
        case 2: return gather_fixed<2>(run, block_bytes, stream.offset, out);
        case 4: return gather_fixed<4>(run, block_bytes, stream.offset, out);
        case 16: return gather_fixed<16>(run, block_bytes, stream.offset, out);
        case 32: return gather_fixed<32>(run, block_bytes, stream.offset, out);
        case 64: return gather_fixed<64>(run, block_bytes, stream.offset, out);
        case 128: return gather_fixed<128>(run, block_bytes, stream.offset, out);
        default: break;   // a width no listed layout has: copied as it comes
    }
    std::size_t at = 0;
    for (std::size_t b = 0; b < run.size(); b += block_bytes, at += stream.width) {
        std::memcpy(out.data() + at, run.data() + b + stream.offset, stream.width);
    }
    return at;
}

// Takes `n` bytes of staging past `staged`. The bound the staging is sized at
// is proved in piece_writer.h; exceeding it would mean the proof is wrong,
// which no caller can recover from, so it stops here rather than write past
// the area (E.26).
std::span<std::byte> take(std::span<std::byte> staging, std::size_t& staged, std::size_t n) {
    if (n > staging.size() - staged) std::abort();
    const auto area = staging.subspan(staged, n);
    staged += n;
    return area;
}

}  // namespace

PieceWriter::PieceWriter(std::span<const Route> routes, std::size_t max_chunk, std::span<const FileRange> ranges)
    : routes_(routes),
      max_chunk_(max_chunk),
      ranges_(ranges.begin(), ranges.end()),
      staging_(staging_bound(routes.size(), max_chunk)) {
    if (!ranges_.empty()) next_offset_ = ranges_[0].begin;
    while (route_ < routes_.size() && routes_[route_].blocks == 0) ++route_;
    // Where each buffer's pieces have reached, walking the routes in order.
    std::vector<std::uint64_t> reached;
    for (const Route& r : routes_) {
        const auto b = static_cast<std::size_t>(r.buffer);
        if (b >= reached.size()) reached.resize(b + 1, 0);
        if (r.buffer_offset < reached[b]) pieces_ascend_ = false;
        reached[b] = std::max(reached[b], r.buffer_offset + r.length);
    }
}

WriteError PieceWriter::accept(std::uint64_t file_offset, std::span<const std::byte> chunk,
                               std::vector<Write>& out) {
    if (failed_ != WriteError::Ok) return failed_;
    if (chunk.size() > max_chunk_) return failed_ = WriteError::ChunkTooLarge;
    std::uint64_t end = 0;
    if (!ranges_.empty() && next_offset_ == ranges_[range_].end && range_ + 1 < ranges_.size()) {
        // The last chunk ended its range: the next starts the next range.
        ++range_;
        next_offset_ = ranges_[range_].begin;
    }
    if (file_offset != next_offset_ || !gguf::checked_add(file_offset, chunk.size(), end) ||
        (!ranges_.empty() && end > ranges_[range_].end)) {
        return failed_ = WriteError::OutOfOrder;
    }

    std::size_t staged = 0;
    finished_ = nullptr;
    std::uint64_t pos = file_offset;
    while (route_ < routes_.size() && pos < end) {
        const Route& route = routes_[route_];
        const std::uint64_t block_bytes = route.layout->block_bytes;
        // The route's next byte; routes are in file order and apart, so the
        // chunk never starts past it. Bytes before it are outside every route.
        const std::uint64_t next = route.file_offset + blocks_done_ * block_bytes + held_count_;
        if (pos < next) {
            pos = std::min(end, next);
            continue;
        }
        const std::uint64_t stop = std::min(end, route.file_offset + route.blocks * block_bytes);
        auto bytes = chunk.subspan(pos - file_offset, stop - pos);
        pos = stop;

        // A block cut off at the last chunk's end is completed first.
        std::span<const std::byte> held;
        if (held_count_ > 0) {
            const auto fill = std::min<std::size_t>(block_bytes - held_count_, bytes.size());
            std::copy_n(bytes.begin(), fill, held_.begin() + held_count_);
            held_count_ += fill;
            bytes = bytes.subspan(fill);
            if (held_count_ < block_bytes) continue;   // the chunk ended inside it
            held = std::span<const std::byte>(held_).first(block_bytes);
        }
        const auto whole = bytes.first(bytes.size() / block_bytes * block_bytes);
        if (!held.empty() || !whole.empty()) write_blocks(held, whole, staged, out);
        held_count_ = 0;

        // What is left is less than a block, cut off at the chunk's end.
        const auto rest = bytes.subspan(whole.size());
        std::copy(rest.begin(), rest.end(), held_.begin());
        held_count_ = rest.size();
    }
    next_offset_ = end;
    return WriteError::Ok;
}

void PieceWriter::write_blocks(std::span<const std::byte> held, std::span<const std::byte> blocks,
                               std::size_t& staged, std::vector<Write>& out) {
    const Route& route = routes_[route_];
    const DeviceLayout& layout = *route.layout;
    const std::uint64_t block_bytes = layout.block_bytes;
    const std::uint64_t count = held.size() / block_bytes + blocks.size() / block_bytes;
    const bool finished = blocks_done_ + count == route.blocks;

    // The plan's padding before this piece, written as zeros so its first
    // run joins the piece this call finished before it.
    if (const auto padding = joinable_padding()) {
        const auto zeros = take(staging_, staged, *padding);
        std::fill(zeros.begin(), zeros.end(), std::byte{0});
        emit({route.buffer, route.buffer_offset - *padding, zeros}, out);
    }

    // Each stream's run: the bytes its tail held back, then its fields of
    // these blocks, in staging. The areas are taken in stream order, so a
    // stream's write can join the one before it.
    const std::size_t streams = layout.streams.size();
    std::array<std::span<std::byte>, formats::kMaxStreams> areas{};
    std::array<std::uint64_t, formats::kMaxStreams> devices{};
    std::array<std::uint64_t, formats::kMaxStreams> lengths{};
    for (std::size_t s = 0; s < streams; ++s) {
        const formats::Stream stream = layout.streams[s];
        const Tail& tail = tails_[s];
        // The tail holds the run's bytes past its last whole word, so the
        // write starts on the word they begin.
        devices[s] = route.buffer_offset + stream_start(layout, s, route.blocks) + blocks_done_ * stream.width -
                     tail.count;
        lengths[s] = tail.count + count * stream.width;
        areas[s] = take(staging_, staged, round_up4(lengths[s]));
        std::copy_n(tail.bytes.begin(), tail.count, areas[s].begin());
    }
    if (layout.repack != nullptr) {
        std::array<std::byte*, formats::kMaxStreams> to{};
        for (std::size_t s = 0; s < streams; ++s) to[s] = areas[s].data() + tails_[s].count;
        for (const auto run : {held, blocks}) {
            layout.repack(run, to);
            for (std::size_t s = 0; s < streams; ++s) to[s] += run.size() / block_bytes * layout.streams[s].width;
        }
    } else {
        for (std::size_t s = 0; s < streams; ++s) {
            std::size_t at = tails_[s].count;
            for (const auto run : {held, blocks}) at += gather(run, block_bytes, layout.streams[s], areas[s].subspan(at));
        }
    }

    for (std::size_t s = 0; s < streams; ++s) {
        Tail& tail = tails_[s];
        const auto area = areas[s];
        const std::uint64_t length = lengths[s];
        std::uint64_t written = length;
        if (finished) {
            // The run that ends the piece is padded to the word; the piece's
            // bound length covers it (device_layout.h).
            std::fill(area.begin() + length, area.end(), std::byte{0});
            written = area.size();
            tail.count = 0;
        } else {
            written = length / 4 * 4;
            tail.count = static_cast<std::uint8_t>(length - written);
            std::copy_n(area.begin() + written, tail.count, tail.bytes.begin());
        }
        if (written > 0) emit({route.buffer, devices[s], area.first(written)}, out);
    }

    blocks_done_ += count;
    if (finished) {
        finished_ = &route;
        blocks_done_ = 0;
        tails_ = {};
        do ++route_;
        while (route_ < routes_.size() && routes_[route_].blocks == 0);
    }
}

void PieceWriter::emit(const Write& w, std::vector<Write>& out) {
    // Every write of a call is staged back to back, so a write that
    // continues the last on the device also continues it in staging; the
    // last test is what makes the joined span hold both writes' bytes.
    if (!out.empty()) {
        Write& last = out.back();
        if (last.buffer == w.buffer && last.offset + last.bytes.size() == w.offset &&
            last.bytes.data() + last.bytes.size() == w.bytes.data()) {
            last.bytes = std::span(last.bytes.data(), last.bytes.size() + w.bytes.size());
            return;
        }
    }
    out.push_back(w);
}

std::optional<std::uint64_t> PieceWriter::joinable_padding() const {
    if (!pieces_ascend_ || finished_ == nullptr) return std::nullopt;
    const Route& route = routes_[route_];
    const std::uint64_t end = finished_->buffer_offset + finished_->length;
    // Pieces that ascend put route_'s at or past the one before it.
    if (finished_->buffer != route.buffer || route.buffer_offset - end > kMaxJoinedPadding) {
        return std::nullopt;
    }
    return route.buffer_offset - end;
}

WriteError PieceWriter::finish(std::uint64_t file_size) {
    if (failed_ != WriteError::Ok) return failed_;
    const std::uint64_t last = ranges_.empty() ? file_size : ranges_.back().end;
    if (route_ < routes_.size() || next_offset_ != last || (!ranges_.empty() && range_ + 1 != ranges_.size())) {
        return failed_ = WriteError::Unfinished;
    }
    return WriteError::Ok;
}

}  // namespace bllm::residency
