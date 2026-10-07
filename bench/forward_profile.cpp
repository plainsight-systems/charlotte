// The forward pass's GPU time, launch by launch: make profile.
//
// A DIAGNOSTIC tool (core/diagnostics.h). It is built only in the native-diag
// preset, and its numbers say where a step's time goes; they are never
// quoted as throughput (TLM.6). A clean build's throughput is measured where
// the page runs.
//
// Why natively: there the GPU's timestamps are unrounded, to its counter's
// resolution, which the report states, where Chrome rounds them to 100 µs,
// longer than most of a step's dispatches (WASM.12); and a step is timed
// launch by launch. What it measures is native evidence: Dawn here is the
// release emdawnwebgpu is built from, not Chrome's own Dawn and Tint, so the
// Metal they generate, and the time it takes, may differ. Its findings are
// relative — which launch takes the step's time, against what floor — and
// each is confirmed in Chrome by the step's whole time there, the page's
// prefill and decode rates, before a change is built on it. Both
// toolchains' versions head the report.
//
//     charlotte_profile_forward <model.gguf> [--csv <file>]
//
// It loads the model as the harness does — plan, upload, graph, program —
// on a device that granted timestamp queries (gpu/device.h's
// DiagnosticRequest), then times three workloads:
//   1. Decode steps at positions 0, 1,024 and 8,192, those within the
//      context offered, the cache first filled by prefill to that position.
//   2. Prefill steps of 1, 8, 16, 32, 64, 128 and 512 tokens at position 0:
//      the products change form between decode and prefill, and with the
//      tile a step's token count selects (kernels/matmul/matmul.h).
//   3. Pipelined decode: 128 whole steps fed on the GPU, two outstanding, as
//      the runtime runs them (runtime/runtime.h), each profiled, so the run
//      is timed as it runs: the GPU idled for the run's span on its own
//      clock — the last step's end less the first's beginning, less the
//      steps' times — and, apart, on the CPU's, each step's submission and
//      its record's arrival after its end. GPU.10's first question, whether
//      the GPU waits on the CPU, before any kernel's.
// A launch's time, in workloads 1 and 2, in the one pass a step runs in
// (kernels/program.h):
//   - where the device grants timestamps inside a pass, the step's own, a
//     launch at a time;
//   - otherwise, as on the target, a prefix's: the step run through its
//     first k launches takes T(k), and launch k's time is T(k) − T(k − 1),
//     what adding it costs the step in the shape it runs in. A prefix costs
//     a step's time up to it, so the prefixes run are those through the
//     embedding, through the first layer of each kind the graph builds —
//     Gemma 3's window and global layers are two — and through the output
//     block; every other layer repeats a first layer's launches over the
//     same shapes. The check that they do: the step's whole time less its
//     time through those first layers, over the layers left, against a
//     first layer's, reported, and a failure named past 10%.
// Each time is the median of 5 runs after 2 discarded warm-ups — the GPU
// clocks up under load — reported with the least and the most. The
// conditions head every report: the build, the adapter and its backend, both
// toolchains' versions, the timestamp counter's resolution, the model
// file's SHA-256, runs and warm-ups (WASM.11).
//
// Reported, for each step timed:
//   - Its whole GPU time, from its pass's two timestamps.
//   - By role and entry point (graph/graph.h): the launches, the median time
//     a launch, the total, and its share of the step; and, for a launch that
//     reads its bindings once each — a product's weights, a norm, the gather
//     — the bytes it binds over its time, against the target's 400 GB/s
//     (graph.h), so a launch far below the device's bandwidth is named.
//   - The floors the headers derive, beside the times measured: a decode
//     step's weights and cache read once, 0.94 ms + 0.29 µs × p for Qwen3;
//     a 512-token prefill step's arithmetic, about 32 ms at the M3 Max's
//     peak (graph.h); each kernel's own (its header).
//   - Every launch's time, by index, role, layer and entry point, to the CSV
//     file when one is named, so a change's effect is a diff of two files.
// What it finds is recorded in docs/research/, with the commit and the
// conditions, as the readback round trip's was.
//
// Verification, before any time is reported, each a failure named if it does
// not hold:
//   - A profiled step at either grain draws the same record and leaves the
//     same logits, bit for bit, as run() over the same step: the instrument
//     changes no result.
//   - A step's timestamps in order: its pass's end at or after its
//     beginning, and any timestamps inside it, one for each launch it
//     dispatched, in the graph's order, between the two.
//   - The repetition check above, on the target.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     Per.6  Don't make claims about performance without measurements — the
//            headers' floors are derived; this measures what the GPU spends
//            against them.
//   C++ performance guidelines
//     GPU.10 Profile with GPU timelines and counters before optimizing — GPU
//            timestamps in the step's own pass, each launch named, and the
//            idle share first.
//     TLM.6  Diagnostic mode is not benchmark mode — a diagnostic build's
//            tool, its findings confirmed in Chrome before they are built on.
//     WASM.12 Keep the compute core natively buildable so it can be
//            profiled — the same kernels, timed natively.
//     WASM.11 State the measurement conditions — they head every report.
