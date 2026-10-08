#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/residency/routes.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Turns the file, as it arrives in chunks, into the writes that fill the
// planned buffers. A deterministic state machine with no I/O: it holds no GPU
// object and says what to write, not how; upload issues each write as one
// wgpuQueueWriteBuffer (upload.h), and every rule here is tested without a
// device. The same chunks always give the same writes, so the diagnostic
// check runs it again to know what the device should hold (upload_check.h).
//
//   - Chunks arrive in order, of any size; a block, a row or a piece may
//     span chunks. The bytes of a block cut off at a chunk's end are held
//     until the next chunk completes it — never more than one block (210
//     bytes, Q6_K's), so what is held does not grow with the file.
//   - A chunk's whole blocks of a piece are laid out as the piece's device
//     layout says: each stream's run of fields, contiguous, the blocks
//     repacked first by a layout that repacks (device_layout.h). A piece in
//     a 16 MiB chunk takes one run per stream, not one per block.
//   - A chunk's writes are issued in device order, and a run that continues
//     the write before it, in the same buffer, joins it rather than starting
//     its own: the streams of a piece the chunk holds whole, and the pieces
//     the plan packs back to back into one buffer. Padding is joined only
//     if every buffer's pieces lie in route order, each after the one
//     before it — checked once, at construction — so that the gap between
//     two consecutive pieces of a buffer holds no other piece; and only
//     between a piece the chunk finished and the next piece's first run, at
//     most kMaxJoinedPadding bytes. The join writes the gap as zeros, which
//     is what WebGPU created it holding, so the buffer ends byte for byte as
//     it would without the join. A gap inside a piece, which a later chunk
//     fills, is never joined. A chunk takes a write for each buffer it
//     reaches, and one more for each stream of a piece it cuts, rather than
//     one for each stream of every piece: Qwen3 0.6B's 567 writes become 78.
//     Optimization (browser): every write crosses into the browser and is
//     validated there (WASM.2), about half a millisecond a call in the desktop
//     app's Chromium on the target whatever its length, so the writes scale with buffers and
//     cut pieces, not with streams or tensors. Qwen3 0.6B from the browser's
//     cache, warm, release module, the desktop app's Chromium 152, Apple M3 Max, medians of six
//     runs before and twelve after: the worker's load 174 -> 119 ms, the
//     page's 198 -> 140 ms.
//   - Every run is copied into staging, even one already in device order
//     (F32): a layout whose one stream is its whole block is copied as one
//     memcpy, which costs far less than the write it would otherwise take
//     on its own.
//   - Every write starts on a 4-byte boundary and is a multiple of 4 bytes
//     long, as writeBuffer requires. A stream's run that ends partway through
//     a word holds back those bytes for the next chunk; the run that ends the
//     piece is padded with zeros to the word, which the piece's bound length
//     covers.
//   - Writes are made from one staging area, allocated once and reused: the
//     heap holds a chunk and its staging, whatever the file (WASM.1, WASM.9,
//     MEM.9). Its size is proved from the routes, not guessed: what one chunk
//     stages is at most the chunk itself, plus one block held over from the
//     chunk before, plus, for each route the chunk reaches, up to 3 bytes
//     held back and 3 bytes of padding in each of its streams and the
//     padding joined before it. Every route could end inside one chunk — a
//     file of adjacent one-block tensors does — so the bound counts all of
//     them:
//       max_chunk + kMaxBlockBytes + routes × (kMaxStreams × 6 + kMaxJoinedPadding),
//     95,970 bytes beyond the chunk for Gemma 3's 342 routes. A native test
//     drives the case the bound is set by — a full chunk that completes a
//     held block, then many adjacent one-block routes, each padded and
//     joined — and checks what it staged against the bound.
//   - Bytes outside every route — the header, the padding between tensors —
//     are skipped. A chunk that does not start where the last one ended, or
//     the file ending with a route unfilled, is a named failure.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant — the members below, and
//            the invariant each keeps.
//     E.4    Design your error-handling strategy around invariants — after a
//            failure the writer accepts nothing, so no write follows a broken
//            invariant.
//     E.27   Use error codes systematically — WriteError.
//     R.14   Prefer span to pointer and count — chunks in, writes out.
//   C++ performance guidelines
//     WASM.1 Size linear memory to the real high-water mark — one chunk and
//            its staging, allocated once.
//     WASM.2 Batch work across the JS boundary — writes joined across
//            streams and pieces, never one per block. Its caveat, that a
//            batch blurs which item failed, is met for content only: the
//            diagnostic check names each piece whose bytes differ
//            (upload_check.h). A write the browser rejects is still
//            reported for its chunk as a whole.
//     WASM.9 Stream in bounded chunks — what is held is at most one block.
//     MEM.9  Allocate at init, not in steady state — the staging area.
//     CDSA.32 Transform static data once into the layout its consumer reads —
//            this is that transform. Its caveat, that a load-time transform
//            doubles peak memory while source and result coexist, is met by
//            transforming a chunk at a time; and its conversion is tested
//            against the untransformed reference: every stream's bytes,
//            gathered back, are the stored blocks' fields in block order.

// One write: `bytes` at `offset` in `buffer`. The span borrows, never owns:
// it points into the writer's staging, valid until the next call to accept.
// A caller that holds writes past the call — the diagnostic check does, until
// its comparison completes — makes no other call to accept until then (I.11:
// ownership stays with the caller).
struct Write {
    BufferIndex buffer;
    std::uint64_t offset;
    std::span<const std::byte> bytes;
};

enum class WriteError {
    Ok,
    OutOfOrder,         // a chunk did not start where the last one ended
    ChunkTooLarge,      // larger than the staging the writer was given
    Unfinished,         // the file ended before every route was filled
};

// The most padding a join writes: WebGPU's default storage-offset alignment,
// the largest a device may require, so the most bytes the plan leaves
// between two pieces of a buffer.
inline constexpr std::size_t kMaxJoinedPadding = 256;

class PieceWriter {
public:
    // The most one chunk's writes can stage, by the bound above: what the
    // writer allocates, and what a reader of its writes must hold.
    [[nodiscard]] static constexpr std::size_t staging_bound(std::size_t routes, std::size_t max_chunk) noexcept {
        return max_chunk + formats::kMaxBlockBytes + routes * (formats::kMaxStreams * 6 + kMaxJoinedPadding);
    }

    // `routes` in file order, as plan_routes gives them; `max_chunk` the
    // largest chunk accept will be given. Precondition: `routes` outlives
    // the writer.
    PieceWriter(std::span<const Route> routes, std::size_t max_chunk);

    // The writes the chunk at `file_offset` completes, appended to `out`. On
    // failure `out` is left as it was and the writer accepts nothing more.
    [[nodiscard]] WriteError accept(std::uint64_t file_offset, std::span<const std::byte> chunk,
                                    std::vector<Write>& out);

    // Called once the file has ended at `file_size`: Unfinished unless the
    // chunks reached it and every route was filled. Unfinished is a failure
    // like the others: from then on accept refuses every chunk.
    [[nodiscard]] WriteError finish(std::uint64_t file_size);

private:
    // Bytes of one stream's run held back because it ended partway through a
    // word; they lead that stream's next write.
    struct Tail {
        std::array<std::byte, 3> bytes{};
        std::uint8_t count = 0;
    };

    // Lays out route_'s next blocks — `held`, one block or none, then
    // `blocks` — as its streams, in staging_ past `staged`, which it
    // advances, and appends their runs to `out` through emit.
    void write_blocks(std::span<const std::byte> held, std::span<const std::byte> blocks,
                      std::size_t& staged, std::vector<Write>& out);

    // Appends `w` to `out`, joined to the last write when it continues it in
    // both the buffer and staging.
    static void emit(const Write& w, std::vector<Write>& out);

    // The padding between the piece this call last finished, whose write
    // was the last, and route_'s, when the join rules above allow it to be
    // written; otherwise none.
    [[nodiscard]] std::optional<std::uint64_t> joinable_padding() const;

    std::span<const Route> routes_;
    std::size_t max_chunk_;
    std::size_t route_ = 0;            // the first route not yet filled
    std::uint64_t blocks_done_ = 0;    // of route_, the blocks already written
    std::uint64_t next_offset_ = 0;    // where the next chunk must start
    // A block of route_ cut off at the last chunk's end: fewer than its
    // layout's block_bytes, so never more than kMaxBlockBytes - 1.
    std::array<std::byte, formats::kMaxBlockBytes> held_{};
    std::size_t held_count_ = 0;
    std::array<Tail, formats::kMaxStreams> tails_{};   // one for each of route_'s streams
    // Whether every buffer's pieces lie in route order, each after the one
    // before it: padding is joined only if so. Set at construction.
    bool pieces_ascend_ = true;
    // The route this call finished last; route_ follows it.
    const Route* finished_ = nullptr;
    // Where rearranged bytes are written from: allocated once, at
    // construction, at the bound above, and never grown (MEM.9).
    std::vector<std::byte> staging_;
    // Set by the first failure; from then on accept refuses every chunk.
    WriteError failed_ = WriteError::Ok;
};

}  // namespace bllm::residency
