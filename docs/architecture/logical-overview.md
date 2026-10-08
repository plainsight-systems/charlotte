# Logical overview

What the harness does, and the principles that decide how it handles the ways
models differ.

This is the **logical** view: phases, responsibilities, and principles. It
names no files, no types, and no interfaces. The rules that turn these boxes
into files are in [`change-axes.md`](change-axes.md), and the resulting
arrangement is in [`file-mapping.md`](file-mapping.md). Every claim here can be
checked against a model file.

The principles are general. Three real model files and two engines'
registries were read to test them
([`../research/2026-09-24-model-files-and-engine-registries.md`](../research/2026-09-24-model-files-and-engine-registries.md));
they are evidence, not the design.

## The flow

Shading is the same in every diagram: **coral** is selected by an
identifier in the file, **purple** is per-model policy, **teal** is a regime.
**Gray** is written once and serves every model.

```mermaid
flowchart LR
    pick["Pick<br/>list carries policy"] --> pf["Preflight<br/>header only"]
    pf --> gates["Gates<br/>stage by stage"]
    gates --> fetch["Fetch<br/>OPFS"]
    fetch --> upload["Upload<br/>plan, routes, device layout"]
    classDef default fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    classDef fork fill:#FAECE7,stroke:#D85A30,color:#993C1D
    classDef policy fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    class upload fork
    class pick policy
```

**Load** runs once per model. Everything before `Fetch` reads only the header,
so an incompatible model is rejected before its weights are downloaded.

```mermaid
flowchart LR
    jinja["Jinja<br/>template from the file"] --> tok["Tokenize<br/>algorithm and<br/>pre-tokenizer"]
    tok --> diff["Diff<br/>longest common prefix"]
    diff --> pfl["Prefill<br/>suffix as a block"]
    pfl --> dec["Decode<br/>one token"]
    dec --> smp["Sample<br/>settings per model"]
    smp -. next token .-> dec
    kv[("KV cache<br/>state, per layer<br/>precision per model")]
    diff -. truncate .-> kv
    pfl <-.->|append, read| kv
    dec <-.->|append, read| kv
    classDef default fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    classDef fork fill:#FAECE7,stroke:#D85A30,color:#993C1D
    classDef policy fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef regime fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class tok fork
    class smp,kv policy
    class pfl,dec regime
```

**Generate** runs per turn. The whole conversation is re-rendered and
re-tokenized every turn; the diff is what makes that cheap. The KV cache is
drawn as state rather than as a step, because that is what it is — see below.

```mermaid
flowchart TD
    sched["Graph<br/>per architecture"]
    sched --> mm["matmul<br/>GEMM · GEMV"]
    sched --> nrm["norm"]
    sched --> pos["position encoding"]
    sched --> att["attention<br/>block · against cache"]
    sched --> gat["gather"]
    sched --> act["activation"]
    classDef default fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    classDef fork fill:#FAECE7,stroke:#D85A30,color:#993C1D
    classDef regime fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class sched fork
    class mm,att regime
```

**Inside Prefill and Decode.** The graph is the only thing that knows the
architecture. It chooses which shared kernels run, in what order, with what
parameters. The kernels do not know which model they serve.

```mermaid
flowchart TD
    emb["Embed<br/>a row of the table per token"]
    subgraph layer["Each layer, in order"]
        att["Attention<br/>looks back at earlier tokens"] --> ffn["Feed-forward<br/>each row alone"]
    end
    emb --> att
    ffn --> fnorm["Final norm"]
    fnorm --> head["Output head<br/>a score per vocabulary entry"]
    head --> smp["Sample<br/>one token"]
    smp --> out["Detokenize<br/>each token as it comes"]
    smp -. next token .-> emb
    kv[("KV cache<br/>keys and values,<br/>per layer")]
    att <-.->|append, read| kv
    classDef default fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    classDef policy fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef regime fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class smp,kv policy
    class att,ffn,head regime
```

**One pass.** Prefill and Decode are the same pass with a different number of
rows: the tokens the diff did not find in the cache, at most a prefill block
at a time, or the one token just sampled. Each row is embedded, moves through
every layer, and only the last row reaches the output head. The sampled token
is fed back for the next pass, and is detokenized and streamed as it comes.
Within a layer every step works on each row alone, except attention: the only
place one token reads another, and only through the keys and values the cache
holds. Weight formats reach the embedding, every projection and the output
head — the steps that read weights.

```mermaid
flowchart TD
    x["Hidden state X<br/>one row per token in the pass"] --> n["RMSNorm<br/>of a copy"]
    n --> q["Q projection<br/>a query per head"]
    n --> k["K projection<br/>a key per shared head"]
    n --> v["V projection<br/>a value per shared head"]
    q --> rq["QK-norm where the file has it,<br/>then RoPE by position"]
    k --> rk["QK-norm where the file has it,<br/>then RoPE by position"]
    rk -->|append| kv[("KV cache")]
    v -->|append| kv
    rq --> s["Scores<br/>q · k / √head size"]
    kv -->|every key, cached and new| s
    s --> m["Causal mask, softmax<br/>row i: weights over 0..i"]
    m --> w["Weighted sum of values"]
    kv -->|every value, cached and new| w
    w --> o["Concatenate heads, × Wo<br/>back to the hidden width"]
    o --> add["Add to X"]
    x -. residual .-> add
    classDef default fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    classDef policy fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef regime fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class kv policy
    class q,k,v,s,m,w,o regime
```

**The attention block.** The norm works on a copy: the block's result is
added to X, never written over it, so information carries through every
layer. The projections give each row a query for every head, and a key and a
value for every shared key-value head; heads that share one read the same keys
and values. Position enters here and nowhere else: RoPE rotates each query and
key by its token's position, so a score depends on how far apart two tokens
are. The new keys and values are appended to the cache before the scores, so
each row scores itself and every earlier token, cached or new. The scores are
never stored whole; they are computed in tiles and combined as they go.
Concatenating heads moves nothing — each head writes its own columns — and
the output projection maps them back to the hidden width.

## Principles

These decide how any difference between models is handled. Each was tested
against real files, but none is specific to the files tested.

1. **Kernels never know the family.** A kernel is parameterized by shape,
   stride, weight format and regime. Knowledge of a family lives in exactly one
   place: the graph, which decides which kernels run in what order with what
   parameters. A kernel that branches on family must be re-verified for every
   family added.

2. **Each identifier in the file selects exactly one implementation.**
   `general.architecture` selects a graph. A tensor's type selects its device
   layout, pack and unpack. `tokenizer.ggml.model` selects a
   tokenization algorithm, and `tokenizer.ggml.pre`, where present, selects the
   pre-tokenizer that splits text before the algorithm runs. These are
   independent: families share algorithms without sharing pre-tokenizers,
   formats cross families, and any combination can appear in a file. The
   mapping is many-to-one — several identifier values may select the same
   implementation — but never one-to-many.

3. **The file supplies every number.** Layer count, head counts, head
   dimension, attention window, RoPE θ, epsilon and context length are read per
   file. None is a constant keyed on the architecture. Two conversions of the
   same model can disagree, and have.

4. **Normalize before the kernel, not in it.** When a family's convention is
   a reparametrization of its weights — a norm offset, a permutation, a folded
   scale — it is applied once, before any kernel reads the weight, so the
   kernel stays uniform. GGUF's converters apply them when they write the file,
   so upload writes what the file holds. Branch only for operations that differ in kind. The exception is a
   weight used in two places, such as an embedding tied to the output head,
   where a fold that is correct for one use is wrong for the other.

5. **Detect structure; never assume it.** Whether a tensor exists — a separate
   output head, a bias, an optional norm — is read from the tensor index at
   load. No architecture is assumed to have or to lack one.

6. **What differs in kind is a shared kernel.** A new activation, norm type or
   position encoding is a new kernel that any graph can dispatch. It is not a
   branch inside an existing kernel, and it does not belong to the family that
   first needed it.

7. **Support is declared, and judged stage by stage.** One capability table
   lists the graphs, weight formats, tokenization algorithms and
   pre-tokenizers implemented. Preflight reports how far the build can take a
   model — read, download, describe, fit, upload, run — and names what stops
   each stage it cannot reach: *"format Q4_1 is not supported (3 tensors,
   first blk.0.ffn_down.weight)"*, rather than failing somewhere during the
   load. A server engine can fall back to running a model's reference
   implementation, slowly; a browser has no reference implementation to run,
   so naming the miss is the only honest outcome. Judging each stage on its
   own keeps every stage testable against real files before the stages after
   it exist.

8. **Policy is measured per model, and an unmeasured model is labelled.** How
   to sample, what precision the cache uses, and which controls the interface
   offers are not in the file. They are recorded per model in the curated
   list. A model without an entry runs on defaults and is shown as unmeasured.

9. **Regime is orthogonal to model.** Prefill and decode are shapes of a call,
   not properties of a model. A kernel whose best form depends on that shape
   has one implementation per regime, and never one per family.

## The design space

What varies across decoder-only transformers, and which principle handles it.

| What varies | Examples | Handled as |
|---|---|---|
| Norm type | RMSNorm, LayerNorm | shared kernel (6) |
| Norm placement | before, after, or both | graph |
| Norm weight convention | `w`, `1 + w` | normalized by the file's converter (4) |
| Heads, head dimension, θ, ε, scale | numbers | file parameter (3) |
| Attention sharing | multi-head, grouped, multi-query | file parameter |
| Attention window | global, sliding, interleaved per layer | per-layer parameter |
| QK-norm, QKV biases | present or absent | detected (5) |
| Position encoding | RoPE and its pairing and scaling variants, ALiBi, none | shared kernel and parameters |
| Logit softcapping | present or absent | shared kernel and parameter |
| MLP | gated or plain, and which activation | graph and shared kernel |
| Mixture of experts | a router and experts | graph and new shared kernels |
| Embedding scale | none, √d | graph |
| Output head | tied to the embedding, or separate | detected |
| Tokenization algorithm | BPE, SentencePiece, WordPiece | its own implementation (2) |
| Pre-tokenizer | a split pattern, named per lineage | selected by name; the file names it but does not carry it (2) |
| Weight format | per tensor | pack and unpack (2) |

Two consequences are worth stating outright:

- **Weight format reaches only what reads weights.** That is the projections,
  the embedding gather and the output head. Attention proper — scores, softmax
  and the weighted sum — reads activations and the cache, never weights.
- **Which path is hot is decided by the file.** Bandwidth flows through the
  unpack of whatever format the largest tensors are stored in. That is a
  property of a particular file, not of its family, and it is why optimizing
  "the matmul" means every unpack path shipped.

## Two regimes

The diff-and-prefill design means the forward pass runs in two shapes, and they
want different kernels.

| | Decode | Prefill |
|---|---|---|
| Tokens per pass | one | the diffed suffix — up to thousands |
| Projections | matrix × vector | matrix × matrix |
| Attention | one query per head against the cache | a causal block over the suffix |
| Bound by | weight bandwidth | arithmetic |
| Runs when | every generated token | the first turn, a long system prompt, and every re-prefill after the template rewrites history or context overflows |

So the matmul is two kernels, and so is attention. Neither forks on family.

## The KV cache is state keyed on tokens

The cache is keyed on **the token sequence**, never on the message list, and it
is touched at two levels:

- **once per turn**, the diff truncates it to the longest common prefix — a
  length counter, with no data movement. A sliding-window layer keeps a ring,
  so a rollback deeper than the policy's reserve empties the cache instead;
- **in every layer, on every token**, both regimes append new K and V and read
  the cache back for attention.

Drawing it as a step in the turn pipeline would hide that most of its traffic
is inside the forward pass.

Keying on tokens is not a performance convenience. It is the only correct
option, because **a chat template may rewrite history retroactively**, and a
cache keyed on message boundaries then reuses a prefix the template has already
changed. The corruption is invisible; the output just gets worse. Qwen3's
template is a concrete example: it computes `last_query_index` by scanning
backwards for the most recent user turn, then

```jinja
{%- if loop.index0 > ns.last_query_index %}
    ... '<think>\n' + reasoning_content + '\n</think>\n\n' + content
{%- else %}
    ... content          {# think block dropped #}
{%- endif %}
```

so adding a user turn strips the reasoning block from an assistant message that
carried it the turn before.

Two consequences:

- **The JavaScript side is stateless with respect to the cache.** It renders
  everything and sends a string. It does not compute deltas, track what is
  cached, or know that token identifiers exist. One place can get this wrong,
  and it is the place holding the tokens.
- **Context overflow uses the same mechanism.** Drop the oldest messages,
  re-render, and the diff finds a short common prefix on its own. It is slower
  for that turn, correct by construction, and visible to the user.

The cache's mechanism is the same for every model. Its storage precision is
not — that is policy, below.

## Per-model policy

Some choices differ per model and are carried by neither the code nor the
file. We decide them — from the model card, or by measurement — and record
them per model (principle 8).

**Cache precision.** Storage is 16-bit or narrower; arithmetic is always f32,
because, for cross-browser compatibility, nothing here requires the
optional `shader-f16` feature, so the only error introduced is
storage rounding. Which format is right depends on how hard the cache binds
context length for that model. Where KV bytes per token are large against the
memory budget, narrower storage buys context; where they are small, there is
little to gain. f16 and bf16 are the same size with opposite trade-offs — f16
has three more mantissa bits, bf16 has f32's range — and a model trained in
bf16 has only ever produced values bf16 can hold. The choice sits on the
bounded-divergence side of the correctness gate, so it is made by measuring
logits against an f32-cache reference over real conversations, not by
argument. A model without a measurement uses f16.

**Sampling.** Model cards specify sampling settings, and they are not
interchangeable between models or even between modes of one model. Qwen3's
card, for example, warns *"DO NOT use greedy decoding, as it can lead to
performance degradation and endless repetitions."* So the product sampler is
stochastic, and its randomness is an injected seed rather than ambient state: a
fixed seed reproduces a run exactly. Comparing top-1 against a reference
remains a valid *test*; it is not the product sampler.

**Interface affordances.** Some models expose modes through template
variables, such as a thinking mode, and some expose none. A control that exists
for one model and not another is policy, not code.

**Context offered** is derived, not chosen: the smaller of the file's declared
context and the memory budget divided by KV bytes per token at the chosen
precision.

**Where policy lives.** The curated model list is a list of measured
configurations — file, hash, cache precision, memory budget, sampling per
mode, affordances — not a list of URLs. It changes when a model is measured. The capability table
changes when code is written. Different reasons, so they are two things. A
model loaded from a pasted URL gets defaults and is shown as unmeasured;
presenting an untested model as tuned would claim something nobody checked.

## Why preflight exists

Preflight judges a model **before** its weights are downloaded. The metadata
block and the tensor index sit at the front of the file and are a small
fraction of it, and they answer every compatibility question — including the
one only the tensor index can answer: which weight formats are actually
present.

That question cannot be answered from the filename. A file named for one
format routinely contains others: of the four files named `Q4_0` read for
the research note above, three carry other formats.

The gates are principle 7 applied, one per stage, each needing every stage
before it:

| Stage | Needs |
|---|---|
| Read | the file is GGUF and its index reads |
| Download | a readable file — so every model that reads can be cached |
| Describe | its architecture is implemented and can describe the file |
| Fit | the residency plan fits the **granted** device limits and the budget |
| Upload | its weights reach the GPU and read back identical |
| Run | the graph, every weight format, the tokenizer and pre-tokenizer |

Every check that can run does, so a verdict lists everything a model still
needs, each item tagged with the stage it stops; the stage reached is the one
before the earliest blocked. A stage the build does not implement is
blocked by name for every model, so no verdict claims more than exists. The
planner is a pure function of the tensor index and the limits, which is what
lets Fit be judged before a single weight byte is fetched.

A model already downloaded is judged from its copy on this device, not the
network, and a fresh download is read back from that copy before it is
offered.

The gates and the unpack dispatch consult **one** capability table. Two lists
will diverge, and a picker that reports "compatible" for a model that then
fails to load is worse than no picker.

## Platform constraints

Four properties of the target shape everything above. They are constraints, not
preferences, and each one closes off an option that would otherwise look
reasonable.

**The GPU is reached through `webgpu.h`, not JavaScript.** In the browser,
`--use-port=emdawnwebgpu` implements that C API on top of the browser's WebGPU.
Natively, the same API is implemented by Dawn — the same engine Chrome uses,
with the same Tint shader compiler. So identical kernel code can run under a
native test binary and in a browser, and native tests exercise the real stack
rather than a mock. Driving WebGPU from JavaScript would put the forward pass
on the wrong side of the WASM boundary.

**No threads.** GitHub Pages cannot set `Cross-Origin-Opener-Policy` or
`Cross-Origin-Embedder-Policy`, so `SharedArrayBuffer` is unavailable and
pthreads cannot be used. The same absence of cross-origin isolation clamps
`performance.now()` to 100 µs, which constrains how anything here can be
measured. Responsiveness comes from a plain Web Worker, which needs no shared
memory — acceptable because the GPU does the compute while WASM parses,
tokenizes, dispatches and samples.

**Weights are not in this repository and cannot be.** GitHub caps files at
100 MB and Pages does not serve Git LFS, against a several-hundred-megabyte
quantized model. They are fetched on first run, verified, and cached in OPFS.
The code is self-contained; the weights are not.

**No inference dependencies.** No llama.cpp, no ggml, no ONNX Runtime, no npm,
no bundler. Vendoring an inference stack would defeat the premise. Third-party
code is for things that are not the demonstration — which is why the chat
template, a full Jinja2 dialect, is rendered by a library on the JavaScript
side rather than reimplemented in C++.
