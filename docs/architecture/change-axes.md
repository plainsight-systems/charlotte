# Change axes

One translation unit, one reason to change. This document names the reasons,
because without that list the rule cannot be applied — every split sounds
arguable and the argument is settled by taste.

It governs how [`logical-overview.md`](logical-overview.md)'s boxes become
files. It does not list the files; that is [`file-mapping.md`](file-mapping.md).

## The axes

| Axis | Triggered by | May change |
|---|---|---|
| **A** | a new architecture | its graph, and the working buffers that graph binds |
| **B** | a new weight format | its device layout, pack and unpack |
| **C** | a GGUF format revision, or our parse contract | reader, its types, its errors |
| **D** | the WebGPU surface, or granted device limits | device, planner's placement, buffer writer |
| **E** | a new or optimized kernel, in one regime | that kernel only |
| **F** | a new sampling method | the sampler |
| **G** | a new cache or context mechanism | cache, prefix diff |
| **H** | product and interface | the JavaScript presentation layer |
| **I** | **what the code implements** | one capability table |
| **J** | **a model is measured** | its entry in the curated list |
| **K** | a new tokenization algorithm or pre-tokenizer | that algorithm, or that pre-tokenizer's split pattern |
| **L** | a browser network or storage API, or how we download | fetch transport, OPFS model cache |
| **M** | a new stage in a model's loop | the runtime that sequences it: diff, prefill, decode and sample for a generator; encode, one step and read back for an embedder |

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

## M is a family too

A generator and an embedder run different loops over the same kernels. A
generator's turn diffs the conversation against the cache, prefills, then
decodes and samples a token at a time; an embedder's encode tokenizes a text,
wraps it in the file's ends, runs one step and reads back a vector. Adding a
stage to one — speculative decoding, say, or batching several texts into one
encode — is not the same reason as adding one to the other. So M is
parameterized by loop: M(generate), the turn loop and its stop set, and
M(encode), the embedder and its ends.

The test is E's: **a stage added to one loop must produce a diff that touches
no other loop's files.** What the two share is below them — the program, the
tokenizer, the plan — each on its own axis.

## The boxes that are already one axis

`Gates` (I — it applies the table, it does not contain criteria), `Jinja` (H —
it renders a template it does not interpret), `Graph` (A — one per
architecture), `Tokenize` (K — one per tokenization algorithm, with pre-tokenizer split
patterns selected by name), `Wrap` (M(encode) — the file's ends around a
text), `Pool` (E), and each kernel (E).

## The boxes that are compound

| Box | Axes it mixes | Splits into |
|---|---|---|
| **Upload** | D, B, D, L | planner, routes and piece writer · device layout · buffer writer · reading the cache and confirming duplicates |
| **Fetch** | L, L | transport · OPFS model cache — network and storage change independently |
| **Sample and emit** | F, K, H | sampler · detokenize · emit |
| **Diff and KV cache** | G, D | prefix diff · cache resources |
| **Pick** | H, J | picker interface · the curated list, which is measured policy |
| **Working buffers** | A, D | each graph's working set — which buffers it binds, how large · the planner's placement of them |
| **Encode** | M, A, E, D | the embedder · the architecture's graph, composed from the encoder's blocks · the kernels · the program |

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

**Working buffers** split the same way the planner and the writer do. Which
buffers a graph binds, and at what size, changes when a graph or a kernel
needs one — attention's partials, selection's candidates, an encoder's
scores — which is A or E. Where they lie in device memory changes with the
limits a device grants, which is D. So each graph builder declares its working
set as a list of needs, and the planner places whatever list it is given,
knowing no kernel by name. A list kept in the planner would change for every
kernel added, and an encoder would make it branch on the model's kind.

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

**An encoder adds no regime.** Regime is the shape of a call, and an
encoder's one step over a whole text is prefill's shape: its matrix products
run their prefill forms, chosen by token count as a decoder's are. What an
encoder changes is what the model is: attention in both directions over the
step's own keys, with no cache. That differs in kind from attention against a
cache, so it is its own kernel, block attention, beside attention, not a
regime of it or a flag inside it.

**Rope's encode form is a variant; block attention is a kernel.** Rope does
the same operation for an encoder — QK-norm and rotation by position — and
only writes its keys and values to the step's buffers instead of packing them
into the cache, with a value norm where the model declares one. The same
operation, writing elsewhere, is an entry point of the same module, as a
matrix product's decode and prefill forms are. Attention in both directions
masks, tiles and folds differently from attention against a cache, so a
change to one should never re-verify the other.

**Two graph builders, not one with a flag.** A decoder's blocks end in logits
and the draw and bind a cache; an encoder's end in a pooled vector and bind
none. A builder serving both would change whenever either did. They share the
kernels and the residual convention, not code.

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
