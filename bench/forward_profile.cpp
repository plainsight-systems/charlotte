// The forward pass's GPU time, launch by launch: make profile.
//
// A DIAGNOSTIC tool (core/diagnostics.h). It is built only in the native-diag
// preset, and its numbers say where a step's time goes; they are never
// quoted as throughput (TLM.6). A clean build's throughput is measured where
// the page runs.
//
// Why natively: the kernels are the same — Tint compiles the same WGSL to
// Metal for Dawn here and for Chrome on the target — and here the GPU's
// timestamps are exact, where Chrome rounds them to 100 µs, longer than most
// of a step's dispatches (WASM.12). What the browser adds lies outside the
// GPU's time: its process boundary and its event loop.
//
//     charlotte_profile_forward <model.gguf> [--csv <file>]
//
// It loads the model as the harness does — plan, upload, graph, program —
// on a device that granted timestamp queries (gpu/device.h's
// DiagnosticRequest), then times three workloads:
//   1. Decode steps at positions 0, 1,024 and 8,192, those within the
//      context offered, the cache first filled by prefill to that position.
//   2. Prefill steps of 1, 8, 64, 128 and 512 tokens at position 0: the
//      products change form between decode and prefill, and with the tile a
//      step's token count selects (kernels/matmul/matmul.h).
//   3. Pipelined decode: 128 steps fed on the GPU, two outstanding, as the
//      runtime runs them (runtime/runtime.h). Their wall time against the sum
//      of their Step-grain GPU times is the share the GPU sat idle between
//      steps — whether the CPU's submission or the readback's round trip
//      leaves it waiting, GPU.10's first question, before any kernel's.
// The first two at both grains of a profiled step (kernels/program.h): Step,
// the step's GPU time as run() spends it, and Launch, each launch's. Each
// timed step runs 5 times after 2 discarded warm-ups, and the median is
// reported with the least and the most. The conditions head every report:
// the build, the adapter and its backend, Dawn's version, the model file's
// SHA-256, runs and warm-ups (WASM.11).
//
// Reported, for each step timed:
//   - Its Step-grain GPU time; the Launch grain's sum; and their ratio, the
//     instrument's cost (TLM.6).
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
//   - At Launch grain, one time for each launch the step dispatches, in the
//     graph's order, each named, each ending at or after it begins; at Step
//     grain, one.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     Per.6  Don't make claims about performance without measurements — the
//            headers' floors are derived; this measures what the GPU spends
//            against them.
//   C++ performance guidelines
//     GPU.10 Profile with GPU timelines and counters before optimizing — GPU
//            timestamps a launch, each named, and the idle share first.
//     TLM.6  Diagnostic mode is not benchmark mode — a diagnostic build's
//            tool, its instrument's cost reported.
//     WASM.12 Keep the compute core natively buildable so it can be
//            profiled — the same kernels, timed natively.
//     WASM.11 State the measurement conditions — they head every report.
