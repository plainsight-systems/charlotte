# Change axes

One translation unit, one reason to change. This document names the reasons,
because without that list the rule cannot be applied — every split sounds
arguable and the argument is settled by taste.

It governs how [`logical-overview.md`](logical-overview.md)'s boxes become
files. It does not list the files; that is [`file-mapping.md`](file-mapping.md).

## The axes

| Axis | Triggered by | May change |
|---|---|---|
| **A** | a new architecture | its graph |
| **B** | a new weight format | its device layout, pack and unpack |
| **C** | a GGUF format revision, or our parse contract | reader, its types, its errors |
| **D** | the WebGPU surface, or granted device limits | device, planner, buffer writer |
| **E** | a new or optimized kernel, in one regime | that kernel only |
| **F** | a new sampling method | the sampler |
| **G** | a new cache or context mechanism | cache, prefix diff |
| **H** | product and interface | the JavaScript presentation layer |
| **I** | **what the code implements** | one capability table |
| **J** | **a model is measured** | its entry in the curated list |
| **K** | a new tokenization algorithm or pre-tokenizer | that algorithm, or that pre-tokenizer's split pattern |
| **L** | a browser network or storage API, or how we download | fetch transport, OPFS model cache |
| **M** | a new stage in the generation loop | the runtime that sequences diff, prefill, decode and sample |

Every file maps to exactly one row. Two rows means it splits. Two files that
always change together means they merge — unless they cannot, for a reason
recorded below.

The axes were found by reading real model files and then generalized. The files
are evidence; the axes do not depend on them.

**Mechanism is code; the values it runs with are policy.** F and G change when
a *method* is added — a new way to sample, a new way to manage the cache. The
settings a particular model runs with — its temperature, its cache precision —
change when that model is measured, which is J. Keeping them apart means
measuring a model never looks like a code change.

**Architecture and tokenizer are separate axes.** Adding an architecture that
reuses an existing tokenizer touches no tokenizer; adding a tokenizer under an
existing architecture touches no graph. That is the test below, and it is why
K is not part of A. Examples that happen to differ in both architecture and
tokenizer do not show that the two change together; they are the coincidence
the test exists to catch.

**B covers writing as well as reading.** Weights are only ever read, but a
quantized cache would be the first thing the harness *writes* in
a quantized format, and it reuses the format knowledge of any weight stored in
the same format. B owns both directions. Which format the cache uses is not B's
business; that is policy, on J.

## E is a family, not an axis

Optimizing one kernel is not the same reason as optimizing another, and
optimizing a kernel for decode is not the same reason as optimizing it for
prefill. Decode multiplies a matrix by a vector and is bound by weight
bandwidth; prefill multiplies a matrix by a block and is bound by arithmetic.
They want different tiling. So E is parameterized by kernel *and* regime —
E(matmul, decode), E(matmul, prefill), E(attention, decode),
E(attention, prefill) — plus one per kernel whose form does not depend on
regime.

Kernels include every operation that differs in kind. Each activation, each
norm type and each position encoding is its own kernel (principle 6 in the
logical overview), dispatched by whichever graph needs it. None belongs to an
architecture.

The test is mechanical: **optimizing a kernel for one regime must produce a
diff that touches no other kernel** — including the same kernel's other regime.
If it does, something is shared that should not be.

Three things would recouple them silently:

| Shared thing | If it lives inside kernels | Belongs to |
|---|---|---|
| the unpack function | every kernel changes when a weight format is added | axis B, its own file |
| bind-group and parameter convention | changing argument order touches all of them | one interface file |
| dispatch geometry arithmetic | retuning workgroup counts touches all of them | its own file |

Workgroup size is the opposite case. It belongs **per kernel**, because the
right value differs per kernel; a shared constant would put all of them back on
one axis.

The kernels are equal in structure and unequal in value. In decode the
projections are most of the dispatches and nearly all of the weight bandwidth —
for Qwen3 0.6B, four of each layer's eight launches and the output head,
113 of a pass's 231 ([`kernel-fusions.md`](kernel-fusions.md)) — and in prefill the
same holds for their block form. Most kernel files are correctness surface; the
two matmul regimes are the performance surface.

## The boxes that are already one axis

`Gates` (I — it applies the table, it does not contain criteria), `Jinja` (H —
it renders a template it does not interpret), `Graph` (A — one per
architecture), `Tokenize` (K — one per tokenization algorithm, with pre-tokenizer split
patterns selected by name), and each kernel
(E).

## The boxes that are compound

| Box | Axes it mixes | Splits into |
|---|---|---|
| **Upload** | D, B, D, L | planner, routes and piece writer · device layout · buffer writer · reading the cache and confirming duplicates |
| **Fetch** | L, L | transport · OPFS model cache — network and storage change independently |
| **Sample and emit** | F, K, H | sampler · detokenize · emit |
| **Diff and KV cache** | G, D | prefix diff · cache resources |
| **Pick** | H, J | picker interface · the curated list, which is measured policy |

**Diff and KV cache** is the split worth insisting on.
`longest_common_prefix(old, new) → length` is a pure function over two integer
sequences: no GPU, no model, no file, no browser. The cache is buffers and
counters. Fused, the most correctness-critical logic in the chat loop becomes
reachable only through a device.

**Upload rearranges layout, never values, and owns neither.** GGUF's
converters store weights in the convention the uniform kernels expect —
llama.cpp writes Gemma's norm weights as `1 + w` and permutes Llama's Q and K —
so upload changes no value. What it rearranges is where bytes lie: each
format's device layout splits its blocks into streams, and that layout belongs
to the format, which unpack reads it by. Upload never learns what a format's
fields mean.

**Upload** is also the box that carries the most risk if left whole. The
planner is pure arithmetic over a tensor index and a limit set, and preflight
depends on running it before any weight byte is fetched. Fusing it with the
writer would make the compatibility gate impossible.

## Cases that look wrong and are not

**The capability table has one owner and several readers.** The picker's gates,
the unpack dispatch, the tokenizer selector and the graph selector all consult
it. That is not a violation: the table changes for exactly one reason, and its
readers do not change when it does. Adding a weight format means one new
unpack file and one new row.

This is also what keeps the picker honest. If the gates and the loader consult
two lists, the lists will diverge, and a picker reporting "compatible" for a
model that then fails to load is worse than no picker. The single table is the
enforcement; review is not.

**The capability table and the curated list are two files, not one.** Both
describe what the harness can run, which makes merging them tempting. But the
capability table changes when code is written — a new unpack, a new
tokenizer or pre-tokenizer, a new graph — and the curated list changes when a model is measured: its cache
precision, its sampling settings, the controls it offers. Different reasons,
different files. Merging them would make measuring a model look like a code
change, and adding a format look like a policy change.

**Enumerating a format is not implementing it.** A type that lists every value
the file format defines — every weight format GGUF can name, with its block
size — is format knowledge, and changes only when the format does. Which of
those values the harness can run is the capability table's business. A reader
that knows the size of a format it cannot unpack is not claiming to support it;
it needs that size to validate the file, and to name the tensor it will reject.

**A kernel is one responsibility expressed as two files.** The WGSL and its C++
launcher change together — retile the shader and the dispatch geometry moves
with it. The strict rule says merge them; they are different languages, so the
coupling is a language artifact rather than a design choice. They stay a
co-located pair, and nothing should be introduced between them to make the
separation look intentional.

## How to apply this

Before adding to a file, ask which row of the table would cause this line to
change. If it is not the row that file already owns, it belongs somewhere else.

Before creating a file, name its row. A file whose row cannot be named is
either a utility bucket or a responsibility nobody has articulated, and both
tend to accumulate.

The decisive question for a split is not "are these different concerns" —
that is unfalsifiable — but "can I name a change that touches one and not the
other." If not, they are one file. And when the only evidence that two things
change together is that the examples at hand happened to differ in both, look
for an example where they do not.
