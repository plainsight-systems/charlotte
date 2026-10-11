#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/residency/piece_writer.h"
#include "core/residency/routes.h"

using namespace bllm;
using formats::DeviceLayout;
using residency::PieceWriter;
using residency::Route;
using residency::Write;
using residency::WriteError;

namespace {

// Test-only: a layout whose last stream is 3 bytes wide, so a run can end
// partway through a word and its tail, held back or padded, is exercised.
// No listed format has one; the writer's rules allow it.
constexpr formats::Stream kOddStreams[] = {{3, 4}, {7, 4}, {0, 3}};
constexpr DeviceLayout kOddLayout{gguf::TensorType::F32, 11, kOddStreams};
// Test-only: already in device order, as F32 is, but 3 bytes a block, so a
// run of whole blocks need not end on a word and cannot go straight out.
constexpr formats::Stream kOddIdentityStreams[] = {{0, 3}};
constexpr DeviceLayout kOddIdentity{gguf::TensorType::F32, 3, kOddIdentityStreams};

struct Piece {
    const DeviceLayout* layout;
    std::uint64_t blocks;
    std::uint64_t gap_before;   // file bytes outside every route, before it
    // Where the plan puts it: in a buffer of its own, or this many bytes past
    // the next 256-byte boundary — past kMaxJoinedPadding, which no plan's
    // padding reaches.
    bool new_buffer = false;
    std::uint64_t extra_padding = 0;
};

struct Fixture {
    std::vector<std::byte> file;
    std::vector<Route> routes;
    std::vector<std::uint64_t> buffer_sizes;
};

constexpr std::byte kUntouched{0xEE};

std::uint64_t round_up(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

// A file of deterministic pseudo-random bytes holding `pieces` in order, each
// routed into its buffer at the next 256-byte boundary, as a plan places them.
Fixture make(std::span<const Piece> pieces, std::uint64_t trailing = 5) {
    Fixture f;
    std::uint32_t state = 0x9E3779B9u;
    const auto bytes = [&](std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            f.file.push_back(static_cast<std::byte>(state >> 24));
        }
    };
    bytes(24);   // a header
    std::uint32_t tensor = 0;
    for (const Piece& p : pieces) {
        bytes(p.gap_before);
        if (f.buffer_sizes.empty() || p.new_buffer) f.buffer_sizes.push_back(0);
        std::uint64_t& size = f.buffer_sizes.back();
        size += p.extra_padding;
        const std::uint64_t stored = p.blocks * p.layout->block_bytes;
        f.routes.push_back({gguf::TensorId{tensor++}, f.file.size(), p.blocks, p.layout,
                            residency::BufferIndex{static_cast<std::uint32_t>(f.buffer_sizes.size() - 1)}, size,
                            round_up(stored, 4)});
        size = round_up(size + round_up(stored, 4), 256);
        bytes(stored);
    }
    bytes(trailing);
    return f;
}

using Buffers = std::vector<std::vector<std::byte>>;

// One block as the device holds it: as stored, or as its layout repacks it.
std::vector<std::byte> device_block(const DeviceLayout& layout, const std::byte* stored) {
    std::vector<std::byte> block(stored, stored + layout.block_bytes);
    if (layout.repack != nullptr) {
        std::array<std::byte*, formats::kMaxStreams> to{};
        for (std::size_t s = 0; s < layout.streams.size(); ++s) to[s] = block.data() + layout.streams[s].offset;
        layout.repack(std::span(stored, layout.block_bytes), to);
    }
    return block;
}

// What the device should hold: each route's blocks gathered by hand into its
// streams, the last run zero-padded to the route's length, nothing else
// written. The untransformed reference CDSA.32 asks the conversion be
// tested against.
Buffers expected(const Fixture& f) {
    Buffers device;
    for (const std::uint64_t size : f.buffer_sizes) device.emplace_back(size, kUntouched);
    for (const Route& r : f.routes) {
        auto& buffer = device[static_cast<std::size_t>(r.buffer)];
        std::fill_n(buffer.begin() + static_cast<std::ptrdiff_t>(r.buffer_offset), r.length, std::byte{0});
        std::uint64_t at = r.buffer_offset;
        for (const formats::Stream s : r.layout->streams) {
            for (std::uint64_t b = 0; b < r.blocks; ++b) {
                const auto block = device_block(*r.layout, &f.file[r.file_offset + b * r.layout->block_bytes]);
                for (std::uint64_t i = 0; i < s.width; ++i) buffer[at++] = block[s.offset + i];
            }
        }
    }
    return device;
}

// The device as the writes leave it, and how many writes reached each byte.
struct Device {
    Buffers bytes;
    std::vector<std::vector<int>> times;

    explicit Device(const Fixture& f) {
        for (const std::uint64_t size : f.buffer_sizes) {
            bytes.emplace_back(size, kUntouched);
            times.emplace_back(size, 0);
        }
    }

    // Applies one write, checking it is word-aligned and a whole number of
    // words.
    void apply(const Write& w) {
        CHECK(w.offset % 4 == 0);
        CHECK(w.bytes.size() % 4 == 0);
        auto& buffer = bytes[static_cast<std::size_t>(w.buffer)];
        REQUIRE(w.offset + w.bytes.size() <= buffer.size());
        std::copy(w.bytes.begin(), w.bytes.end(), buffer.begin() + static_cast<std::ptrdiff_t>(w.offset));
        for (std::uint64_t i = 0; i < w.bytes.size(); ++i) ++times[static_cast<std::size_t>(w.buffer)][w.offset + i];
    }
};

// Whether the device holds what it should: every byte of every route's
// binding written once, with `want`'s value; and any other byte written is
// padding between two routes of its buffer, written once, as zero — what
// WebGPU creates a buffer holding.
void check_settled(const Fixture& f, const Device& device, const Buffers& want) {
    for (std::size_t b = 0; b < want.size(); ++b) {
        std::vector<char> bound(want[b].size(), 0);
        std::uint64_t first = want[b].size();
        std::uint64_t last = 0;
        for (const Route& r : f.routes) {
            if (static_cast<std::size_t>(r.buffer) != b) continue;
            std::fill_n(bound.begin() + static_cast<std::ptrdiff_t>(r.buffer_offset), r.length, 1);
            first = std::min(first, r.buffer_offset);
            last = std::max(last, r.buffer_offset + r.length);
        }
        std::size_t wrong = 0;
        for (std::uint64_t i = 0; i < want[b].size(); ++i) {
            const int times = device.times[b][i];
            const std::byte got = device.bytes[b][i];
            const bool ok = bound[i] ? times == 1 && got == want[b][i]
                                     : times == 0 || (times == 1 && got == std::byte{0} && first < i && i < last);
            if (!ok && wrong++ < 4) {
                CAPTURE(b);
                CAPTURE(i);
                CAPTURE(times);
                CHECK(ok);
            }
        }
        CHECK(wrong == 0);
    }
}

// Streams the file through a writer in chunks of `chunk_size` and returns
// what the device would hold.
Device upload(const Fixture& f, std::size_t chunk_size) {
    PieceWriter writer(f.routes, chunk_size);
    Device device(f);
    for (std::size_t at = 0; at < f.file.size(); at += chunk_size) {
        const auto chunk = std::span(f.file).subspan(at, std::min(chunk_size, f.file.size() - at));
        std::vector<Write> writes;
        REQUIRE(writer.accept(at, chunk, writes) == WriteError::Ok);
        for (const Write& w : writes) device.apply(w);
    }
    CHECK(writer.finish(f.file.size()) == WriteError::Ok);
    return device;
}

// The writes the whole file, as one chunk, gives.
std::vector<Write> one_chunk(const Fixture& f, PieceWriter& writer) {
    std::vector<Write> writes;
    REQUIRE(writer.accept(0, f.file, writes) == WriteError::Ok);
    return writes;
}

const Piece kEveryLayout[] = {
    {&formats::kF32Layout, 7, 0},  {&formats::kQ4_0Layout, 5, 13}, {&formats::kQ4_1Layout, 3, 1},
    {&formats::kQ8_0Layout, 1, 0}, {&formats::kQ6_KLayout, 3, 30}, {&kOddLayout, 9, 2},
    {&formats::kQ4_0Layout, 1, 0}, {&kOddLayout, 1, 0},  {&kOddIdentity, 7, 0},
    {&kOddIdentity, 4, 3},
};

}  // namespace

TEST_CASE("the device holds every stored block's fields as its layout's streams, whatever the chunks") {
    const Fixture f = make(kEveryLayout);
    const auto want = expected(f);
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{17},
                                    std::size_t{18}, std::size_t{210}, std::size_t{211}, std::size_t{4096},
                                    f.file.size()}) {
        CAPTURE(chunk);
        check_settled(f, upload(f, chunk), want);
    }
}

TEST_CASE("the same chunks always give the same writes") {
    const Fixture f = make(kEveryLayout);
    const auto run = [&] {
        PieceWriter writer(f.routes, 64);
        std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> seen;
        for (std::size_t at = 0; at < f.file.size(); at += 64) {
            std::vector<Write> writes;
            const auto chunk = std::span(f.file).subspan(at, std::min<std::size_t>(64, f.file.size() - at));
            REQUIRE(writer.accept(at, chunk, writes) == WriteError::Ok);
            for (const Write& w : writes) seen.push_back({w.offset, {w.bytes.begin(), w.bytes.end()}});
        }
        return seen;
    };
    CHECK(run() == run());
}

TEST_CASE("a piece within one chunk takes one write, its streams joined") {
    for (const DeviceLayout* layout : {&formats::kQ4_0Layout, &formats::kQ6_KLayout, &kOddLayout}) {
        const Piece piece[] = {{layout, 6, 0}};
        const Fixture f = make(piece);
        PieceWriter writer(f.routes, f.file.size());
        const auto writes = one_chunk(f, writer);
        REQUIRE(writes.size() == 1);
        CHECK(writes[0].offset == 0);
        CHECK(writes[0].bytes.size() == f.routes[0].length);
    }
}

TEST_CASE("the pieces of one buffer within one chunk take one write, across the padding between them") {
    const Fixture f = make(kEveryLayout);
    PieceWriter writer(f.routes, f.file.size());
    const auto writes = one_chunk(f, writer);
    REQUIRE(writes.size() == 1);
    Device device(f);
    device.apply(writes[0]);
    check_settled(f, device, expected(f));
}

TEST_CASE("pieces in different buffers, or further apart than the plan's padding, are written apart") {
    // The second buffer's piece starts 8 bytes past where the first's ends,
    // as padding would, or exactly where it ends, as the next run would.
    const Piece padded[] = {{&formats::kQ4_0Layout, 3, 0}, {&formats::kQ4_0Layout, 3, 0, true, 64}};
    const Piece abutting[] = {{&formats::kQ4_0Layout, 3, 0}, {&formats::kQ4_0Layout, 3, 0, true, 56}};
    const Piece apart[] = {{&formats::kQ4_0Layout, 3, 0}, {&formats::kQ4_0Layout, 3, 0, false, 256}};
    for (const auto pieces :
         {std::span<const Piece>(padded), std::span<const Piece>(abutting), std::span<const Piece>(apart)}) {
        const Fixture f = make(pieces);
        PieceWriter writer(f.routes, f.file.size());
        const auto writes = one_chunk(f, writer);
        CHECK(writes.size() == 2);
        check_settled(f, upload(f, f.file.size()), expected(f));
    }
}

TEST_CASE("a piece a chunk begins is not joined to one the chunk before finished") {
    // The first chunk ends exactly where the first piece does.
    const Piece pieces[] = {{&formats::kQ4_0Layout, 3, 0}, {&formats::kQ4_0Layout, 3, 0}};
    const Fixture f = make(pieces);
    const std::size_t first = 24 + 54;
    PieceWriter writer(f.routes, f.file.size());
    std::vector<Write> writes;
    REQUIRE(writer.accept(0, std::span(f.file).first(first), writes) == WriteError::Ok);
    REQUIRE(writes.size() == 1);
    writes.clear();
    REQUIRE(writer.accept(first, std::span(f.file).subspan(first), writes) == WriteError::Ok);
    REQUIRE(writes.size() == 1);
    CHECK(writes[0].offset == f.routes[1].buffer_offset);
}

TEST_CASE("padding is joined only when every buffer's pieces lie in route order") {
    // The third piece lies in the padding between the first two: joining
    // them would write that padding as zeros, and the third piece over it.
    const Piece pieces[] = {{&formats::kF32Layout, 3, 0}, {&formats::kF32Layout, 3, 0}, {&formats::kF32Layout, 3, 0}};
    Fixture f = make(pieces);
    f.routes[2].buffer_offset = 128;
    PieceWriter writer(f.routes, f.file.size());
    CHECK(one_chunk(f, writer).size() == 3);
    check_settled(f, upload(f, f.file.size()), expected(f));
}

TEST_CASE("a piece a chunk cuts is never joined across the part a later chunk writes") {
    // Three Q4_0 blocks: the first chunk ends inside the third, so it
    // writes two blocks' nibbles and their scales, 16 bytes apart on the
    // device; the bytes between are the third block's nibbles, the next
    // chunk's.
    const Piece piece[] = {{&formats::kQ4_0Layout, 3, 0}};
    const Fixture f = make(piece, 0);
    PieceWriter writer(f.routes, f.file.size());
    std::vector<Write> writes;
    REQUIRE(writer.accept(0, std::span(f.file).first(24 + 36 + 5), writes) == WriteError::Ok);
    CHECK(writes.size() == 2);
    check_settled(f, upload(f, 24 + 36 + 5), expected(f));
}

TEST_CASE("a chunk out of order, or too large, is refused; out is untouched and nothing more is accepted") {
    const Fixture f = make(kEveryLayout);
    const auto chunk = std::span(f.file).first(64);

    PieceWriter skipped(f.routes, 64);
    std::vector<Write> writes{Write{}};
    CHECK(skipped.accept(1, chunk, writes) == WriteError::OutOfOrder);
    CHECK(writes.size() == 1);
    CHECK(skipped.accept(0, chunk, writes) == WriteError::OutOfOrder);
    CHECK(skipped.finish(f.file.size()) == WriteError::OutOfOrder);

    PieceWriter small(f.routes, 63);
    CHECK(small.accept(0, chunk, writes) == WriteError::ChunkTooLarge);
    CHECK(writes.size() == 1);
    CHECK(small.accept(0, chunk.first(10), writes) == WriteError::ChunkTooLarge);
}

TEST_CASE("the file ending before every route is filled, or short of its size, is Unfinished") {
    const Fixture f = make(kEveryLayout);
    std::vector<Write> writes;

    PieceWriter early(f.routes, f.file.size());
    REQUIRE(early.accept(0, std::span(f.file).first(f.file.size() - 40), writes) == WriteError::Ok);
    CHECK(early.finish(f.file.size() - 40) == WriteError::Unfinished);
    // Unfinished is sticky: the rest of the file is refused, not resumed.
    writes.clear();
    CHECK(early.accept(f.file.size() - 40, std::span(f.file).last(40), writes) == WriteError::Unfinished);
    CHECK(writes.empty());
    CHECK(early.finish(f.file.size()) == WriteError::Unfinished);

    // Every route filled, but the chunks stopped before the file's end.
    PieceWriter short_of_end(f.routes, f.file.size());
    REQUIRE(short_of_end.accept(0, std::span(f.file).first(f.file.size() - 1), writes) == WriteError::Ok);
    CHECK(short_of_end.finish(f.file.size()) == WriteError::Unfinished);
}

TEST_CASE("staging holds the case its bound is set by: a held block, then many padded one-block routes") {
    // Q6_K's 210-byte block cut by the first chunk's end, then 300 adjacent
    // one-block routes of the odd layout, each ending inside the full second
    // chunk with its last stream padded, and the plan's padding before it
    // joined.
    std::vector<Piece> pieces{{&formats::kQ6_KLayout, 2, 0}};
    for (int i = 0; i < 300; ++i) pieces.push_back({&kOddLayout, 1, 0});
    const Fixture f = make(pieces, 0);
    const std::size_t first = 24 + 210 + 100;   // ends inside the second Q6_K block
    const std::size_t max_chunk = f.file.size() - first;
    PieceWriter writer(f.routes, max_chunk);
    Device device(f);

    std::vector<Write> writes;
    REQUIRE(writer.accept(0, std::span(f.file).first(first), writes) == WriteError::Ok);
    for (const Write& w : writes) device.apply(w);

    writes.clear();
    const auto chunk = std::span(f.file).subspan(first);
    REQUIRE(chunk.size() == max_chunk);
    REQUIRE(writer.accept(first, chunk, writes) == WriteError::Ok);
    std::uint64_t staged = 0;
    for (const Write& w : writes) {
        staged += w.bytes.size();
        device.apply(w);
    }
    // The held Q6_K block's first three streams continue runs the last chunk
    // began, apart on the device; its last joins every route after it.
    CHECK(writes.size() == 3);
    CHECK(staged <= PieceWriter::staging_bound(f.routes.size(), max_chunk));
    // The padding joined, past what the bound counted before joins.
    CHECK(staged > max_chunk + formats::kMaxBlockBytes + f.routes.size() * formats::kMaxStreams * 6);
    CHECK(writer.finish(f.file.size()) == WriteError::Ok);
    check_settled(f, device, expected(f));
}

namespace {

// The fixture with only routes `first` .. `last_before_gap` and from
// `after_gap` on, and the ranges that hold them: the header and those routes'
// bytes, the routes between them never read — a companion file's header and
// its encoder's span (residency/file_ranges.h).
struct Ranged {
    Fixture fixture;
    std::vector<residency::FileRange> ranges;
};

Ranged ranged(const Fixture& f, std::size_t last_before_gap, std::size_t after_gap) {
    Ranged r{f, {}};
    const Route& before = f.routes[last_before_gap];
    const std::uint64_t first_end = before.file_offset + before.blocks * before.layout->block_bytes;
    const std::uint64_t second_begin = f.routes[after_gap].file_offset;
    r.ranges = {{0, first_end}, {second_begin, f.file.size()}};
    r.fixture.routes.clear();
    for (std::size_t i = 0; i < f.routes.size(); ++i) {
        if (i <= last_before_gap || i >= after_gap) r.fixture.routes.push_back(f.routes[i]);
    }
    return r;
}

// Streams each range in chunks of `chunk_size`, never across its end.
Device upload_ranges(const Ranged& r, std::size_t chunk_size) {
    PieceWriter writer(r.fixture.routes, chunk_size, r.ranges);
    Device device(r.fixture);
    for (const residency::FileRange& range : r.ranges) {
        for (std::uint64_t at = range.begin; at < range.end; at += chunk_size) {
            const auto chunk = std::span(r.fixture.file).subspan(at, std::min<std::uint64_t>(chunk_size, range.end - at));
            std::vector<Write> writes;
            REQUIRE(writer.accept(at, chunk, writes) == WriteError::Ok);
            for (const Write& w : writes) device.apply(w);
        }
    }
    CHECK(writer.finish(r.fixture.file.size()) == WriteError::Ok);
    return device;
}

}  // namespace

TEST_CASE("a load of ranges writes exactly what its routes hold, the gap never read") {
    const Ranged r = ranged(make(kEveryLayout), 3, 6);
    const Buffers want = expected(r.fixture);
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{7}, std::size_t{64}, r.fixture.file.size()}) {
        CAPTURE(chunk);
        check_settled(r.fixture, upload_ranges(r, chunk), want);
    }
}

TEST_CASE("a chunk in the gap, across a range's end or skipping a range is refused; the last range ends it") {
    const Ranged r = ranged(make(kEveryLayout), 3, 6);
    const auto& file = r.fixture.file;
    const auto first = r.ranges[0];
    const auto second = r.ranges[1];
    std::vector<Write> writes;

    PieceWriter across(r.fixture.routes, file.size(), r.ranges);
    CHECK(across.accept(0, std::span(file).first(first.end + 1), writes) == WriteError::OutOfOrder);

    PieceWriter in_gap(r.fixture.routes, file.size(), r.ranges);
    REQUIRE(in_gap.accept(0, std::span(file).first(first.end), writes) == WriteError::Ok);
    CHECK(in_gap.accept(first.end, std::span(file).subspan(first.end, 4), writes) == WriteError::OutOfOrder);

    PieceWriter skipping(r.fixture.routes, file.size(), r.ranges);
    CHECK(skipping.accept(second.begin, std::span(file).subspan(second.begin, 4), writes) == WriteError::OutOfOrder);

    PieceWriter short_of_last(r.fixture.routes, file.size(), r.ranges);
    REQUIRE(short_of_last.accept(0, std::span(file).first(first.end), writes) == WriteError::Ok);
    CHECK(short_of_last.finish(file.size()) == WriteError::Unfinished);

    PieceWriter whole(r.fixture.routes, file.size(), r.ranges);
    REQUIRE(whole.accept(0, std::span(file).first(first.end), writes) == WriteError::Ok);
    REQUIRE(whole.accept(second.begin, std::span(file).subspan(second.begin, second.end - second.begin), writes) ==
            WriteError::Ok);
    CHECK(whole.finish(file.size()) == WriteError::Ok);
}
