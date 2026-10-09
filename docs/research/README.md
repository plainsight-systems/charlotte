# Research

Measurements and investigations, one note each. A note is dated and names
the machine, browser, build and commit it ran on; its figures hold for
those. A note marked DIAGNOSTIC was taken with instrumentation compiled in:
it says where time goes, never how fast the page runs.

Newest first:

| Date | Note | What it found | Status |
|---|---|---|---|
| 2026-10-09 | [WebKit's shader compiler](2026-10-09-webkit-shader-compiler.md) | Three WebKit compiler faults kept every kernel off the iPhone; how they were found, fixed in the main path, and shown to cost nothing | current |
| 2026-10-09 | [Experimental phone support](2026-10-09-experimental-phone-support.md) | What it takes to run on Chrome for Android and Safari on iOS 26: the iPhone most likely died on memory, and a capture-first plan to find out | current |
| 2026-10-07 | [Chrome page benchmark](2026-10-07-chrome-page-benchmark.md) | Qwen3 0.6B in Chrome: 21 ms to a short prompt's first token, 270 tok/s decoding | current |
| 2026-10-07 | [Decode in the app's Chromium](2026-10-07-app-chromium-decode-profile.md) | Step by step in the page, the GPU is kept fed: 8% of its period outside passes | current |
| 2026-10-07 | [Forward-pass profile](2026-10-07-forward-pass-profile.md) | Where a step's GPU time goes, launch by launch, natively | current |
| 2026-10-03 | [Load performance audit](2026-10-03-load-performance-audit.md) | The load path against its floor, and the rule that was missing: derive a path's cost before building it | current |
| 2026-09-24 | [Model files and engine registries](2026-09-24-model-files-and-engine-registries.md) | Three models' GGUF headers and two engines' registries, read to test the design's principles | current |
| 2026-08-31 | [Measurement build configurations](2026-08-31-measurement-build-configurations.md) | Three builds for performance work; the diagnostic build is Release with instrumentation | current |
| 2026-08-31 | [Model port methodology](2026-08-31-model-port-methodology.md) | Porting a model to a new backend: a fixture ladder, smallest primitive first | current |
| 2026-08-31 | [GPU readback round trip](2026-08-31-gpu-readback-round-trip.md) | A serialized readback costs about 0.5 ms median | measurement stands; decision superseded 2026-10-07 by `src/core/sampler/sampler.h` |
| 2026-08-29 | [WebGPU limits observed](2026-08-29-webgpu-limits-observed.md) | Spec defaults, adapter maxima, and what a device is granted | measurements stand; strategy superseded 2026-08-31 |
