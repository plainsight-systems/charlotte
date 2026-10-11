#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace bllm::residency {

// Axis D: changes with how a load reads its file.
//
// The bytes of a file a load reads, as ranges of its offsets. A model file
// is read whole: one range, from 0 to its size. A companion file is read in
// part (runtime/vision.h): its header and tensor index, and the span holding
// the tensors of the one encoder it serves — for EmbeddingGemma 2's mmproj,
// 0 .. 58,656 and 328,492,448 .. 554,821,024, the vision encoder, 226.4 MB of
// 554.8, the audio encoder's 328 MB before it never fetched, cached or
// uploaded (docs/research/2026-10-11-embeddinggemma2-vision.md).
//
// The ranges are the catalog's, pinned with the SHA-256 of their bytes in
// order (web/models.json), so a part of a file is verified as a whole file
// is: every byte the load reads was hashed against a pinned value before
// it reached the runtime. The ranges are offsets into the file as published,
// so the reader validates the index against the file's whole size, every
// tensor keeps its offset, and nothing renumbers. What changes is what a
// load may assume is present: only the bytes inside the ranges.
//
// The rules, checked by check_ranges before anything is planned:
//   - The first range begins at 0 and holds the whole header and tensor
//     index — the reader's index end at least (gguf/reader.h) — since
//     everything before Fetch reads only the header.
//   - Ranges are in order, disjoint and not touching (a gap between each),
//     non-empty, and end at or before the file's size.
// Refused by name: RangesError, with the range at fault.
//
// What it asks of the other contracts:
//   - residency/plan.h: planned against ranges, the plan places only the
//     tensors whose bytes lie wholly within a range — the encoder's — and
//     a tensor its description names that does not is refused,
//     MissingTensorBytes naming it; one lying partly within a range is
//     refused the same way, as the catalog's span has cut it. A whole
//     file's one range places every tensor, as now.
//   - residency/piece_writer.h: chunks arrive at their file offsets, in
//     order, contiguous within a range; between two ranges the next chunk
//     begins at the next range's start, and a chunk never crosses a gap.
//     OutOfOrder for any other offset; finish when the last range's end is
//     reached, every range delivered. A whole file's one range is the rule
//     that holds now: contiguous from 0 to the file's size.
//   - residency/routes.h and upload.h: begin with the ranges beside the
//     file's size; a route is only ever made for a placed tensor, so routes
//     lie within ranges by construction, which plan_routes checks.
//   - residency/upload_check.h: the check pass reads back the same ranges'
//     bytes, every byte, never the gaps.
//   - preflight/preflight.h: Download, for a load of ranges, is the ranges;
//     Fit plans against them.
//   - The boundary, contract 11: the load crossings carry the ranges, as
//     pairs of offsets, once at begin; each chunk keeps its file offset.
//   - web/: the cache holds the ranges' bytes in order, one file; the page
//     maps a chunk's place in that file to its offset in the published one
//     (web/load.js), and the catalog pins the ranges and their SHA-256
//     (web/catalog.js), computed by tools/pin_ranges.py from the file whose
//     own hash is the publisher's.
//
// What it costs: a check over at most a few ranges, once a load; a chunk's
// offset compared with its range's, once a chunk. Nothing a step.
// Optimization (browser): the audio encoder's 328 MB neither downloaded,
// cached in the origin's storage quota, nor uploaded, for a page that only
// embeds images — 59% of the companion's bytes (WASM.9's split by use).
//
// Verification the implementation is held to (tests/file_ranges_test.cpp,
// and the planner's, piece writer's and upload's own tests): each rule's
// refusal; a whole file as one range planning, routing and writing exactly
// as before, byte for byte; the mmproj's header planned against its vision
// ranges placing the 211 vision tensors and no audio tensor; a description
// naming an audio tensor refused, MissingTensorBytes; the piece writer
// across a gap, and refusing a chunk in it or across it.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     SL.io.2 When reading, always consider ill-formed input — the ranges
//            checked before a byte is planned.
//     F.24   Use a span to designate a half-open sequence — each range is
//            [begin, end).
//     E.27   Use error codes systematically — RangesError.
//   C++ performance guidelines
//     WASM.9 Stream assets and keep them compressed to their destination;
//            split by use — the encoder's span, not the file.

// Bytes [begin, end) of the file as published.
struct FileRange {
    std::uint64_t begin;
    std::uint64_t end;
};

enum class RangesError {
    Ok,
    NoRanges,
    MissesHeader,    // the first range does not begin at 0 or does not hold the index
    Empty,           // a range of no bytes
    OutOfOrder,      // a range before or touching the last
    PastEnd,         // a range ending after the file
};

struct RangesResult {
    RangesError error = RangesError::Ok;
    std::size_t range = 0;   // the range at fault
    [[nodiscard]] bool ok() const noexcept { return error == RangesError::Ok; }
};

// Checks `ranges` against a file of `file_size` bytes whose header and index
// end at `index_end`. Pure.
[[nodiscard]] RangesResult check_ranges(std::span<const FileRange> ranges, std::uint64_t file_size,
                                        std::uint64_t index_end) noexcept;

// Whether bytes [begin, end) lie wholly within one of `ranges`, which
// check_ranges accepted. Pure.
[[nodiscard]] bool within(std::span<const FileRange> ranges, std::uint64_t begin, std::uint64_t end) noexcept;

}  // namespace bllm::residency
