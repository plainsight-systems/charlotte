# WebKit's shader compiler: three faults that kept every model off the iPhone

**Date:** 2026-10-09
**Devices:** an iPhone on iOS 26.6.1, Chrome 154 (WebKit underneath, as every iOS browser); Apple M3 Max, macOS 26.6.2, Safari 26.6.2; the desktop app's built-in Chromium 152 for comparison; Tint from the pinned Dawn, `31e25af254ab572c77054edec4946d2244e184dd`.
**Code:** the failure at `5d1c2dc`; the tools at `ea514ed` (`bench/kernel_dump.cpp`) and in `tools/kernel_probe/`; the fix at `a9ffaae`.

## What the phone said

The page's diagnostics (`web/diagnostics.js`) caught the first iPhone load of
Qwen3 0.6B. The device check passed, the default 2 GiB of buffers were
created, all 382 MB uploaded, and then:

> load: the model's kernels did not build (Compute library failed creation ·
> Failed to evaluate override value · Compute library failed creation × 25)

So the first wall on the phone was not memory, as
[`2026-10-09-experimental-phone-support.md`](2026-10-09-experimental-phone-support.md)
expected, but WebKit's shader compiler. That buffers were created says
little about memory: the GPU may not commit them until they are used.

## Method

WebKit names no kernel in these errors and passes on no compiler message,
and the system log redacts the Metal compiler's warnings as `<private>`.
So the kernels were compiled one at a time, in Safari on the Mac, which
runs the same WGSL compiler as iOS:

1. `charlotte_dump_kernels <model.gguf>` writes each distinct kernel as the
   program compiles it — source, entry point, override constants
   (`kernels/program.h`'s `compose`) — planned at WebGPU's default limits,
   with no GPU. Qwen3 0.6B has 31 distinct kernels among 595 launches.
2. `tools/kernel_probe/index.html` compiles each with
   `createComputePipelineAsync` inside an error scope and lists pass or
   fail with the browser's messages. Opened in Safari from the terminal
   (`open -a Safari`) and read back with AppleScript (`text of current
   tab`), so no click was needed.
3. `tools/kernel_probe/reduce.html` shrinks one failing kernel to the
   fewest lines that still fail the same way: delta debugging by lines,
   then by whole blocks and single statements.

Safari reproduced the phone exactly: 26 of 31 kernels failed. Gather, the
three norms and rope passed; every matrix product, both attention entry
points, top-k's two passes and the draw failed.

Twenty single-construct shaders, each the shared step declaration plus one
thing the failing kernels use and the passing ones do not (hex literals,
`~`, `bitcast`, `exp`, `fma`, `vec2<u32>` storage, `while` with `break`,
scalar and 1,024-element workgroup arrays, a `bool` override), all passed.
The reducer then cut the 143-line draw kernel to four:

```wgsl
override workgroup_size: u32;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) t: u32) {
}
```

That compiles everywhere on its own. It failed because the pipeline was
given `last_token`, which the draw's module declares and its entry point
never reads.

## Three faults

| Construct | WebGPU | Safari 26.6.2 | Chromium 152 |
|---|---|---|---|
| A constant for an override the entry point uses | valid | builds | builds |
| A constant for an override the module declares and the entry point does not use | valid | **"Compute library failed creation"** | builds |
| The same override, no constant given | valid | builds | builds |
| The same, the entry point reading it with `_ = last_token;` | valid | builds | builds |
| A workgroup array sized by an expression of overrides, `8u * tile_tokens` | valid | **"failed to evaluate override expression"** | builds |
| A workgroup array sized by a plain override left at its default | valid | **"failed to evaluate override expression"** | builds |
| An override initialized from others, `1024u / head_dimension` | valid | **"Failed to evaluate override value"** | builds |

Charlotte met all three. Matmul's six entry points share one set of
overrides and each uses some; the program sets `last_token` on every
launch of the last token; matmul's prefill tile was `8u * tile_tokens`
long; and attention's tile shape was derived from its overrides in WGSL.
The second fault had been reported for Safari 26.6.2 by another project
([musetric #991](https://github.com/musetric/musetric/issues/991)), which
found a plain override builds — given a value, as the table shows it must
be; the first and third turned up in none of the sources the phone-support
research read.

## The fix

`a9ffaae`, in the main path, for every browser:

- `compose` begins each entry point with `_ = name;` for every constant
  its pipeline is given, which uses it and computes nothing.
- Matmul's `x_tile` is sized by `x_tile_vec4s`, and attention's `k_tile`
  and `p_tile` by `k_tile_words` and `p_tile_floats`; attention's derived
  shape — `queries_per_tile`, `keys_per_tile`, `group`, `rows_per_tile`,
  `lanes_per_query` — is computed by its launcher. Each is set on every
  launch, decode's matmul included, to the value the expression gave.

All 93 kernels of the three models, Qwen3 0.6B, Llama 3.2 1B and Gemma 3
1B, then build in Safari 26.6.2 and in Chromium 152.

## What it costs

**Generated code.** Tint compiled every kernel to Metal from the code
before (`ea514ed`) and after (`a9ffaae`). Once each side's temporaries are
renamed in order and constant locals that are assigned a literal and never
read are set aside, all 93 are identical. Before there are no such locals;
after there are 534, one for each constant injected. Metal's compiler
deletes an unused constant; that it does here is inferred, not observed.

**Throughput.** The page benchmark (`web/dev/benchmark.js`) on Qwen3 0.6B
in Chrome, both builds served side by side from one origin, alternated,
the page visible throughout, no run left out; load average 2 to 3:

| Pass | Build | First token, short | Decode, short, median (range) | First token, long | Decode, long |
|---|---|---|---|---|---|
| 1 | before | 19.8 ms | 251.0 tok/s (248.0–252.1) | 294.8 ms | 234.3 tok/s |
| 2 | after | 19.3 ms | 258.0 (256.3–259.1) | 285.8 ms | 241.1 |
| 3 | before | 19.3 ms | 258.5 (257.0–259.9) | 284.9 ms | 240.1 |
| 4 | after | 19.4 ms | 256.8 (255.7–257.8) | 286.4 ms | 239.8 |

The first pass, the before build, ran 3% slower than the same build two
passes later; passes 2 to 4 agree within 0.7%, the builds' ranges
overlapping and the differences favoring neither. The fix costs nothing
measurable. Decode is below the 270 tok/s of
[`2026-10-07-chrome-page-benchmark.md`](2026-10-07-chrome-page-benchmark.md)
for both builds alike: the machine's state, not the code.

## What this leaves

Building kernels was the first thing to fail on the phone, so nothing past
it has run there: whether the default budget's buffers survive use on an
iPhone, whether two steps in flight hang WebKit's queue, and what the
phone's GPU makes of the decode loop. The diagnostics will name the next
failure.
