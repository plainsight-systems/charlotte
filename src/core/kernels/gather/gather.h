#pragma once

#include <vector>

#include "core/kernels/interface.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"

namespace bllm::kernels {

// Axis E: changes with a new or optimized kernel.
//
// The embedding: each token of the step becomes its row of the token
// embedding table, decoded and scaled, in the hidden working buffer. One
// kernel for both regimes — a step's rows are independent, so prefill and
// decode differ only in how many there are.
//
//   - Rows: row i of the hidden buffer is token i of the step, hidden-width
//     floats, 4-byte f32 (residency/plan.h).
//   - One invocation per 32-weight group of a row: it reads the step's
//     identifier for its row — or, in a fed step, the draw's token and not
//     the identifier — decodes its group through the table's format's
//     unpack (format.h), multiplies each weight by the launch's scale, and
//     writes the group's 32 floats. A row is hidden-width / 32 invocations;
//     every listed model's width is a whole number of groups, which routes
//     already requires of any tensor kernels read (format.h, steps_by_groups).
//   - The scale is the graph's: 1 for most architectures; Gemma 3 scales its
//     embedding by the square root of the hidden width, as llama.cpp's
//     build_inp_embd(tok_embd, sqrtf(n_embd)) does, in f32. With a scale of
//     1 every weight is its unpack's, bit for bit within format.h's
//     contract. With another, a weight is within 2 units in the last place
//     of llama.cpp's decode-then-scale: the GPU's compiler may fold the
//     scale into the block's own, multiplying the code by d × scale rather
//     than the decoded weight by scale — on Metal every weight was exactly
//     that — and each order rounds twice.
//   - A table split by rows (residency/plan.h) is one launch per piece, each
//     binding only its piece. Every launch covers every row of the step; an
//     invocation whose token lies in another piece returns once it has read
//     the identifier. Each row is written by exactly one launch, since the
//     pieces partition the table and every identifier is below the
//     vocabulary (interface.h). Qwen3 0.6B's table, Q6_K, 127.6 MB, is one
//     piece; Llama 3.2 1B's, Q6_K, 215.5 MB, two; Gemma 3 1B's, Q8_0,
//     320.9 MB, three, at the default 128 MiB binding.
//   - Constants (binding 1), as WGSL lays them out:
//       struct Gather { first_row: u32, row_count: u32, groups_per_row: u32,
//                       blocks_in_piece: u32, scale: f32 }
//     Bindings 2, 3 and 4: the piece of the table, the hidden buffer, and
//     the draw's record, read-only.
//   - A step that is `fed` (kernels/interface.h) embeds, as its one row, the
//     token the last step drew, the first word of the draw's record
//     (sampler/sampler.h), in place of the step's identifier: the next decode
//     step starts without the token having left the GPU. A drawn token is
//     below the vocabulary by construction.
//   - Workgroups of 64 invocations.
//
// What it costs, counted. Launches: one a piece, a step — 1 for Qwen3, 2 for
// Llama 3.2, 3 for Gemma 3 — at about 1.5 µs of GPU time each (interface.h). Bytes,
// a row: unpack's reads for every group of it — for Qwen3's repacked Q6_K,
// 8 words a group, 32 groups, 1 KiB, against 840 bytes stored — and 4 bytes a
// weight written: 4 KiB. A 512-row prefill step reads 512 KiB and writes
// 2 MiB; a decode step reads 1 KiB and writes 4 KiB, so decode's embedding
// costs about its one launch's 1.5 µs. Each invocation also reads its row's
// token, 4 bytes — logically 128 bytes a row a launch for Qwen3, 256 for
// Llama 3.2 and 144 for Gemma 3, every group of a row reading the same word;
// how many of those reach device memory is the GPU's, and WebGPU states
// nothing of it. Loaded once a workgroup instead, the token would need a
// workgroup variable and a barrier in a kernel that has none, every
// invocation waiting at it each step, to spare loads of one word. No row of
// the table that the step does not name is read.
// Optimization (practice): an invocation writes its group's 128 bytes
// contiguously and adjacent invocations take adjacent groups, so a
// workgroup's writes are one contiguous run (GPU.2).
//
// Verification the implementation is held to, on the GPU against the
// format's CPU reference (tests/support): every row of a step, for each
// listed format, bit for bit within format.h's contract; a table forced into
// several pieces, with identifiers on either side of each boundary; and a
// scale other than 1, within the 2 units above.
//
// Pieces are not bound together. One launch could bind every piece and pick
// one by identifier, saving about 1.5 µs a step for Llama 3.2 and 3 µs for
// Gemma 3, about a tenth to a third of a percent of a decode step; it would
// need an unpack for each binding, where every format's unpack reads the one
// binding named `weights` (format.h).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.20   Prefer return values to out parameters — the launches are
//            returned.
//   C++ performance guidelines
//     GDSA.18 Decode block-scaled codes in the load path — the table stays in
//            its format; only the rows named are decoded, into registers,
//            then written once as f32.
//     GPU.2  Shape data for coalesced lane access — the write pattern above.
//     GPU.6  Batch tiny GPU work — one launch a piece, counted above.

// The launches that embed a step's tokens: one for each piece of `table`,
// in order, writing rows of `hidden`, each weight multiplied by `scale`. The
// unpack is the one the capability table lists for the view's own format,
// so a view cannot be read with another format's. Preconditions: `table` is
// the token embedding, in a format the capability table lists — routes
// refused any other — with rows a whole number of 32-weight groups; `hidden`
// is the plan's hidden buffer; `sampled` its draw's record.
[[nodiscard]] std::vector<Launch> gather_launches(const residency::WeightView& table,
                                                  const residency::BufferRange& hidden,
                                                  const residency::BufferRange& sampled, float scale);

}  // namespace bllm::kernels
