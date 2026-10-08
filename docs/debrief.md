# Debrief: what building Charlotte taught us

Charlotte went from an empty repository to three open-weight models chatting
in the browser in about 400 commits, between 2026-08-28 and 2026-10-07. This
is what we learned, what surprised us, and what we left on the table because
this is a demonstration, not a product. Hashes are commits in this
repository; their bodies carry the detail. Figures come from one machine, an
Apple M3 Max, and are labelled with the browser they ran in.

## How it went

- **A walking skeleton first (late August).** C++ compiled to WebAssembly,
  calling WebGPU, adding two vectors in Chrome with no mismatches over 4,096
  elements (`b635851`). Then the model decision — Qwen3 0.6B, Q4_0, GGUF — and
  a GGUF reader built to treat the file as hostile input (`cc2c2df`).
- **The design rewritten as principles (late September).** Planning that
  had run ahead of the code was deleted, and the design restated as
  principles, the reasons a file changes, and a map from parts to files
  (`1fefe74`, `e1f552a`). The page learned to judge a model from its header
  alone, download it into the browser's private storage with its SHA-256
  checked, and render each model's own chat template (`99e9f6a`, `ccdce2e`).
- **Tokenizers, weight formats and upload (October 1–3).** Byte-level and
  SentencePiece BPE matched against their reference implementations; five
  weight formats unpacked on the GPU and tested bit for bit against ggml;
  the GPU code tested natively through Dawn (`c2a8f02`); and a load audit
  that cut Qwen3's warm load from about 298 to 140 ms.
- **Every kernel of a forward pass (October 6).** Each was designed in its
  file header, reviewed, then built and tested against an f64 or CPU
  reference, with a deliberately broken kernel required to fail the test.
- **Generation, then speed (October 7).** Text generating in the page at
  125 tok/s (`e6a9bc7`); a GPU-timestamp profile of every launch; three
  kernel changes from it; 270 tok/s for Qwen3 in Chrome (`10ebc35`); and all
  three models within llama.cpp's own CPU-to-Metal spread (`ecd33ee`).

## Lessons

### Count the cost before building the path

The load path was built stage by stage, each stage right on its own terms,
and nobody walked the whole of it. An audit found it at three to four times
its floor, and every cause could have been counted on paper: 567
`writeBuffer` calls where 78 would do, about 21 million runtime-length
copies, and two copies of the file on every read (`b093506`,
[`research/2026-10-03-load-performance-audit.md`](research/2026-10-03-load-performance-audit.md)).
Since then every data path's header counts its bytes, copies, inner-loop
calls and boundary crossings before code is written, and measurement only
calibrates the count. The same habit changed kernel design the next day: a
fused Q/K/V weight layout was rejected because it would have added about
280 writes to the load (`e5f5a6b`).

### A decision is only as good as the number under it

On 2026-08-31 we measured a GPU readback at 0.5 ms median and decided to
read every sampled token back before the next step, assuming tokens of
20 to 50 ms (`5d93f6c`). Once the forward pass was counted, a decode step
was 1.3 to 2.1 ms, and that 0.5 ms would have added a quarter to a third of
every token. The token now stays on the GPU and is read back a step behind
(`71498da`). The measurement was right; the step time it was weighed against
was a guess.

### Know what you measured

- **Which browser.** Page figures had been called "Chrome" when they were
  taken in a desktop app's built-in Chromium pane, whose Dawn, Tint and
  scheduling may differ. They are relabelled, and a Chrome claim now needs a
  Chrome run with its version recorded (`a427e9e`, `dc856cf`).
- **Whether the GPU was awake.** Profiling runs that waited for each other
  let the GPU clock down: one shape took 2.99 ms in one workload and 20.9 ms
  in another. Profiles now run back to back, two steps outstanding
  (`2cb8a1d`).
- **What the instrument can see.** Metal on this machine gives no
  timestamps inside a pass, so a launch's time is the difference between two
  prefix runs, with their spread carried as uncertainty, and figures under
  the noise are not used (`a2c2236`, `4289903`).
- **What you don't know.** Two slow early runs were first blamed on their
  sampling settings; every run sampled that way, the fast one included, so
  the explanation was withdrawn and the cause recorded as unknown
  (`2dc54a3`).

### A test has to be shown to fail

Again and again a suite that looked thorough passed with the code broken.
The GGUF reader's tests still passed with its overflow-checked arithmetic
replaced by the naive kind (`cc2c2df`). Unicode's own NFC conformance suite
passed with an unstable sort, because no line has a long enough run of
combining marks (`caf3ef4`). Accuracy tests passed NaN, because
`error > bound` is false for NaN (`fa9b0c9`). So commits list the deliberate
breakages their tests catch, and each llama.cpp reference check asserts that
a real error fails it: swapping RoPE's pairing lands 33 to 66 times outside
the tolerance, and giving every Gemma 3 layer the whole context 53 times
(`ecd33ee`).

### Correctness constraints shape the fast path

Every kernel gives a token the same bits whether it was decoded alone or
prefilled in a batch — on this machine's shader compiler, which WGSL does
not promise, so the GPU tests pin it. That meant fixed-order reductions everywhere
(`e24b20c`, `5449727`), explicit `fma` and a polynomial `exp`, because the
compiler contracted `acc + w*x` in one pipeline and not the other
(`aa674d2`), and it ruled out speedups that would change the order of a sum
(`142c90f`). The constraint was in the design from the start, so the speed
work had to find wins inside it — and did.

### Plans that run ahead of the code rot

Acceptance criteria for code that did not exist, a 360-line work packet
never started, committed review records: all deleted (`e9f70f8`, `c3bb6c2`).
A retroactive packet was ruled out because it "only produces a document that
makes the process look followed" (`a1a7910`). Design moved into the headers
of the files it governs, reviewed before the code under them is written
(`72bf16f`).

### Point review at correctness and cost

An automated reviewer read every commit. Asked to apply governance
checklists, its findings leaned towards process: device matrices, budgets,
baselines. Asked two questions — is it correct, and could it be faster —
it caught bugs that would have produced silently wrong output
(`d18fbaa`, `b74e32b`):

- attention's scale set to 0 for every architecture, making attention
  uniform (`1237a7a`);
- the output head running only in decode, so a prompt would never produce
  logits (`bd1795b`);
- 15 of 16 attention partials never written (`d76c667`);
- a borrowed WebGPU handle released without a reference ever taken
  (`e461d01`);
- binding lengths not rounded to 4, so a 54-byte piece would be refused
  (`028c377`).

### Refuse by name rather than guess

Where a model file asks for something the harness does not implement —
partial rotary, RoPE scaling, logit soft-capping, an undeclared
`add_space_prefix` — the page says so by name instead of running it wrong
(`d690325`, `5da98fc`, `f524160`). A model either runs as its file
describes or is refused with the reason.

## Surprises

### WebGPU and Dawn

- **Limits are three numbers.** What the adapter advertises, what a device
  is granted and what you request differ, and it took four tries to stop
  conflating them. The harness now requests the spec's defaults exactly
  (`9da256b` to `51070c3`).
- **A lost device looks like success.** Queued work resolves and error
  scopes come back clean. Upload proves itself by mapping back four bytes it
  wrote last: a lost device refuses the mapping (`b8b20c5`).
- **A `writeBuffer` costs about half a millisecond whatever its size**, in
  the app's Chromium. Joining writes took the load from 198 to 140 ms
  (`2541539`).
- **Buffer usage is tracked across the whole buffer.** Packing every working
  buffer into one failed validation as read-only and read-write in one
  scope (`88c662c`).
- **A component write to a vec4 in workgroup memory may write all four.**
  The first prefill micro-tile raced with its neighbours; the GPU tests
  caught it, and staging moved to whole vec4s (`b061ca4`).
- **Dawn's tick-to-nanosecond factor changes within a process.** One step
  appeared to begin 341.6 s before the one ahead of it. Only a step's own
  two timestamps are comparable (`980df48`).
- **Dawn can hand out its Null backend.** In a sandbox the GPU self-check
  read back zeros; that adapter is now refused by name (`492f19a`).

### Model files

- **"Q4_0" files are mostly not pure Q4_0.** Three of the four read carry
  other formats; Google's Gemma 3 build keeps a 604 MB F16 embedding, 60% of
  the download (`6fa9227`).
- **Two conversions of one model disagree.** Google's Gemma 3 declares a
  1,024-token window, a community build of the same weights 512 (`6fa9227`).
- **Qwen3's head dimension is not width over heads**: 128, not 64
  (`439018e`).
- **Q4_0 interleaves its block halves.** Reading pairs in order keeps the
  shape and silently transposes the values (`fb71b22`).
- **Q6_K's layout fights a GPU unpack.** One 32-weight group read 72 bytes
  for 26 bytes of weights until a repack at upload (`e161937`).

### Tokenizers

- **SentencePiece GGUF files list no merges.** Derived from the token scores
  they come to 513,511, exactly Gemma 3's ordinary merges (`4899d1e`).
- **A typed "▁" splits all three references**: Hugging Face, llama.cpp and
  SentencePiece each tokenize it differently (`dddb1bb`).
- **Lookups dominated loading.** Bisecting the vocabulary cost about 155 ns
  a lookup, a hash table 20; Qwen3's tokenizer load fell from 101 to 19 ms
  (`8fda5c9`).
- **Thousands of Gemma 3's 6,414 special tokens start with "<"**, so a
  first-byte scan was slow until a byte trie took matching from 14 to
  0.82 ms (`f9eaec4`).
- **A prompt within the byte bound could exhaust the heap.** 12 MiB of spaces
  is one piece whose merge needs hundreds of MiB. Prompts are now admitted
  by a lower bound on their tokens before any merge (`ab669eb`, `b25d66a`).

### Performance

- **Prefill was 58 times its floor.** A 512-token step derived at 32 ms
  measured 1,851. Micro-tiles of 4 tokens by 4 outputs, after llama.cpp, MLC
  and ONNX Runtime, took it to 129 ms (`479b2d9`).
- **Attention, not the matrices, bound long decode**: 15% of a step at
  position 0, three quarters by 8,192. Chunks of 64 keys took a step at
  position 1,024 from 6.04 to 3.43 ms natively (`e465632`).
- **The browser keeps the GPU fed.** The GPU spends 8% of its period
  outside passes in the page, so a token costs its step's GPU time and
  little else
  ([`research/2026-10-07-app-chromium-decode-profile.md`](research/2026-10-07-app-chromium-decode-profile.md)).
- **Agreement with llama.cpp is far tighter than needed.** Qwen3's worst
  deviation is 0.012 nats, 6% of llama.cpp's own spread between its CPU and
  Metal backends; across all three models it is at most 8.5% (`b8ed92e`,
  `ecd33ee`).

## What we left on the table

These are deliberate, each with a stated reason; none is a correctness gap.

- **One machine.** Every figure is from one Apple M3 Max. No device matrix,
  per-device budget or regression baseline (`cf8aad0`).
- **Desktop Chrome and Edge only.** Safari's WebGPU is WebKit's own and
  would be a separate verification; mobile was excluded (`1b570cd`).
- **No optional WebGPU features.** Subgroups and shader-f16 would help —
  one subgroup sum against a workgroup tree of nine barriers — but are
  skipped so the page needs nothing optional (`666287b`).
- **No threads.** GitHub Pages cannot set the cross-origin isolation
  headers that shared memory needs, which also leaves the page's clock at
  100 µs (`b635851`).
- **Memory64 declined.** It costs about 10% and buys nothing here: the
  weights stream through the heap to the GPU, never stay in it (`bcf47d3`).
- **Small models.** Larger and mixture-of-experts models were rejected:
  MoE saves compute, and the browser's binding constraint is resident bytes
  (`439018e`).
- **Speed still parked**, each measured (see
  [`process/QUEUE.md`](process/QUEUE.md)): attention's combine at depth,
  prefill attention, the decode products of few rows, prefill for short
  prompts, and the re-prefill when a chat template rewrites history. Decode
  is still about three times its derived floor.
- **Unexplained results left unexplained**: long prefill varying up to 2.1
  times between runs, and the browser's step running about 10% longer than
  native, which was never measured as a matched pair (`93286e5`,
  `2dc54a3`).
- **Rollback keeps the cache or empties it.** A sliding-window ring cache
  cut Gemma 3's from 832 to 238 MiB at full context, at the cost of a full
  prefill when a rollback reaches past it (`e772215`, `7ae564b`).
- **Review skipped at the end.** The last stretch of commits was accepted on
  its tests and the browser runs without the per-commit review (`71ca9d3`).
- **Small wins declined, with their cost stated**: batching the text sent
  to JavaScript (about 30 µs a token against a 4 ms token), folding the
  draw into the selection's last pass (about 1.5 µs), sizing partial
  buffers to the offered context (12.4 MiB) (`8caaf53`, `7763426`,
  `a2a2fe2`).

## By the numbers

| | Figure | Where |
|---|---|---|
| Decode, Qwen3 0.6B, Chrome 151 | 270 tok/s; 251 at about 1,000 positions | [benchmark](research/2026-10-07-chrome-page-benchmark.md) |
| First token, ~20-token prompt | 21 ms | same |
| First token, 951-token prompt | 296 ms | same |
| Decode before the speed work, the app's Chromium | 125 tok/s | `e6a9bc7` |
| Prefill, 512 tokens, native | 1,851 → 129 ms | `479b2d9` |
| Warm load, Qwen3, the app's Chromium | about 298 → 140 ms | [load audit](research/2026-10-03-load-performance-audit.md) |
| Launches in a Qwen3 pass | 231 | [`kernel-fusions.md`](architecture/kernel-fusions.md) |
| Deviation from llama.cpp, worst of three models | 0.085 of its own spread | `ecd33ee` |
