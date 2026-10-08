# Load performance audit: the floor, the path, and the rule we were missing

**Date:** 2026-10-03.
**Context:** Qwen3 0.6B (382 MB, Q4_0 / Q4_1 / Q6_K / F32), loaded from the
browser's cache onto the GPU. Apple M3 Max, the Claude desktop app's built-in Chromium 152 browser pane,
not Chrome, release module, warm
(the file in the operating system's cache), not cross-origin isolated (a
100 µs clock, ample at these durations). Best or median of three to twelve
runs, as marked. Cold loads were not measured: the browser cannot flush the
operating system's cache.

## Method

GDSA.6: write down the bytes each stage moves, measure each stage's ceiling
on its own on this machine, and compare the whole with that floor. The load
path, per 16 MiB chunk: read the cached file, hand it to the module, rearrange
each piece's blocks into streams, `writeBuffer` each stream, wait for the
previous chunk's queue work.

## Each stage alone

| Stage | Measured |
|---|---|
| Cache read, blob, one at a time | 122–128 ms (3.1 GB/s) |
| Cache read, blob, two / three / four in flight | 77 / 48 / 41 ms |
| Cache read, `FileSystemSyncAccessHandle` in a worker, into wasm memory | 23–26 ms (16 GB/s) |
| memcpy, native | 63 GB/s |
| Rearrangement, native, runtime-width copies (Q4_0) | 83 ms per 360 MiB (4.5 GB/s) |
| Rearrangement, native, fixed-width copies (Q4_0) | 11.9 ms per 360 MiB (31.8 GB/s) |
| Rearrangement, wasm in Node, fixed-width copies (Q4_0) | about 17 ms per 360 MiB |
| `writeBuffer` + GPU, 23 writes of 16 MiB, paced | 41–77 ms |
| The same bytes as 25 writes a chunk | 365–407 ms |

## The path, before and after

| | Load (warm, end to end) |
|---|---|
| Before the audit | about 298 ms |
| Fixed-width rearrangement (95a1c62) | about 230 ms |
| Worker reads the cache into the module (a70fb55) | 241 ms median of five, against 257 |
| Writes joined across streams and pieces | 140 ms median of twelve, against 198 median of six on the same page and day; 567 writes become 78 |

The reads gained less than their stage measurement promised: the page's reads
had overlapped the GPU process working through the previous chunk's writes.
What bound the load next was the number of `writeBuffer` calls. The module
issued 567 a load, one per stream of every piece; joined where they touch or
are separated only by the plan's alignment padding they are 78, and the
worker's side of the load went from 174 to 119 ms (medians of six runs
before and twelve after). A `writeBuffer` call costs about half a millisecond in that Chromium at
these sizes, whatever its length. The diagnostic check reads every byte back
after the joined writes and finds the model's bytes where the plan put them.

## Why the audit had to be asked for

Every bottleneck the audit found could have been found by walking the path
on paper and counting, with the guidelines already in the corpus. None of
it needed a measurement to discover:

| Stage | What a walk counts | What it shows |
|---|---|---|
| Rearrangement | Q4_0, 360 MiB: 21 M blocks, two fields each, so 42 M `memcpy` calls whose length is known only at run time, which the compiler cannot inline | A per-field call where memcpy of the same bytes is one pass: tens of milliseconds against about six |
| Writes | One `writeBuffer` per stream of every piece: 567 a load, against 23 chunks | Crossings out of the module that scale with pieces, where WASM.2 asks for one per phase |
| Reads | Blob to `ArrayBuffer`, then a copy into the module's memory, on the page | Two copies of the file and a crossing a chunk, where reading straight into the module's memory is one copy (GDSA.6) |

Each optimization in the load was reasoned about locally — one write per
stream rather than per block, F32 written straight from the chunk, two
chunks in flight — and each was right locally. What the design never did
was walk the whole path and write the counts down as functions of the
model. So nothing showed that the inner loop was a call per field, or that
the module's calls into WebGPU (567) dwarfed the page's calls into the
module (46), which were the only crossings the headers counted. Without the
counts, a guess that the load "isn't where the performance story is" went
unchallenged, and single smoke timings (0.33 s, 1.2 s) passed for evidence.

Measurement told us two things a walk could not: what a `writeBuffer` call
costs in that Chromium (about half a millisecond, whatever its size), and that the
page's reads already overlapped the GPU's work on the previous chunk, so
moving them saved 16 ms rather than the 100 the read stage alone promised.
Those are magnitudes and overlap. Measurement calibrates them; it did not
discover the bottlenecks, and it does not stand in for the walk.

## The rule

**A data path's cost is derived before it is built.** Its header walks every
stage and counts, as functions of the model: the bytes it moves and the
copies it makes (GDSA.6), the operations in its inner loop and whether each
is a call the compiler can inline, and the calls it makes across every
boundary, including the module's calls into browser APIs (WASM.2). A count
that scales with blocks or fields where it could scale with chunks, or a
crossing per piece where one per phase would do, is a bottleneck by
construction, and is designed out before code is written. Once the path
runs it is measured against that floor (GPU.10, WASM.11) to calibrate
per-call costs and overlap, and any stage well off its own ceiling is fixed
or named as the remaining gap.

## Levers not taken here

- Rearranging on the GPU instead: one raw write a chunk and a compute pass,
  which would remove the writes a cut piece still takes per stream; the
  joined writes leave about three a chunk.
- Wasm SIMD: no measurable effect on rearrangement, in Node or that Chromium.
