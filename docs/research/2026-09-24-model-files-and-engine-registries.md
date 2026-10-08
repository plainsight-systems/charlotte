# Model files and engine registries: the evidence behind the principles

**Date:** 2026-09-24. GGUF headers read by range request on 2026-09-23 and
2026-09-24; vLLM's and llama.cpp's sources read 2026-09-24.

The principles these test are in
[`../architecture/logical-overview.md`](../architecture/logical-overview.md);
this note is the evidence, not the design. Three architectures were read to
test the principles, and the source of two production engines to check that
their boundaries match.

RoPE pairing comes from llama.cpp's `llama_model_rope_type`. Qwen3's sampling
settings come from its model card, read 2026-09-24.

| | Qwen3-0.6B | Gemma 3 1B | Llama 3.2 1B |
|---|---|---|---|
| Source | `ggml-org/Qwen3-0.6B-GGUF` | `google/…-qat-q4_0-gguf`; `unsloth/…` | `unsloth/Llama-3.2-1B-Instruct-GGUF` |
| `general.architecture` | `qwen3` | `gemma3` | `llama` |
| `tokenizer.ggml.model` | `gpt2` | `llama` | `gpt2` |
| `tokenizer.ggml.pre` | `qwen2` | absent | `llama-bpe` |
| Formats in the `Q4_0` file | Q4_0, F32 | Q4_0, F32, F16; or Q4_0, F32, Q8_0, Q4_1 | Q4_0, F32, Q5_K, Q4_1 |
| Tensors | 311 | 340 | 147 |
| Header ends at | 5.95 MB of 429.0 MB | 6.53 MB of 1,004 MB | 7.83 MB of 773.0 MB |
| Layers | 28 | 26 | 16 |
| KV heads × head dimension | 8 × 128 | 1 × 256 | 8 × 64 |
| Attention | all global | 5:1 local and global; window 1024 or 512 | all global |
| RoPE pairing | NEOX | NEOX | NORM |
| Separate `output.weight` | yes, byte-identical to the embedding | no | no |
| KV per token at f16 | 112 KiB | 26 KiB | 32 KiB |
| Full-context KV | 40,960 tokens → 4.4 GiB | 32,768 tokens → ~190 MB | 131,072 tokens → 4.0 GiB |

What each finding demonstrates:

- **Identifiers are independent (principle 2).** Qwen3 and Llama 3 are
  different architectures that share a tokenization algorithm, byte-level BPE —
  but not a tokenizer. Their pre-tokenizers differ, `qwen2` against
  `llama-bpe`, so the same text is split differently before a single merge is
  applied, and using one for the other produces wrong token IDs from a correct
  merge table. Gemma 3 uses a different algorithm entirely. The names cross
  over too: the tokenizer identifier `llama` means SentencePiece, and the
  `llama` architecture does not use it.
- **The file supplies every number (principle 3).** Google's Gemma 3 build
  declares `sliding_window = 1024`; a community build of the same weights
  declares `512`. A constant keyed on `gemma3` would be wrong for one of them.
- **Detect structure (principle 5).** Qwen3's file stores its tied output head
  twice, byte-identical — 87.5 MB of the download and of device memory that a
  planner detecting duplicates need not spend. The Gemma 3 and Llama 3.2 files
  store it once, so the head reads the embedding. Same concept, two layouts,
  and neither follows from the architecture.
- **Which path is hot is decided by the file.** In Google's Gemma 3 build the
  embedding is F16 at 604 MB. The head reads it on every decoded token, against
  roughly 400 MB for everything else, so about 60% of decode weight bandwidth
  is an F16 matmul in a file named `q4_0`.
- **Filenames are not evidence of format (principle 7).** Three of the four
  files named `Q4_0` carry other formats. Google's build is also 4.7× the
  default storage-binding limit in a single tensor, so it fails on device fit
  as well as on format.
- **Cache precision is per model (principle 8).** Full-context KV spans more
  than 20× across three small models. Per-token size alone does not predict
  it: Llama 3.2's is less than a third of Qwen3's, yet its declared 131k
  context brings it to 4.0 GiB. What makes Gemma 3 cheap is the sliding window.
  For two of the three the cache binds context length; for the third it barely
  registers.
- **Decode dispatch counts follow from the tensor index.** Qwen3's file holds
  198 Q4_0 tensors and 113 F32 tensors, which decompose exactly as seven
  projections × 28 layers plus the embedding and the output head, and four
  norms × 28 layers plus the final norm. In decode that is 197 matmuls and one
  gather per token, with 113 norms, 56 position encodings, and 28 each of
  attention and activation.

**Production engines draw the same boundaries.** From source, read 2026-09-24.

- vLLM's model registry, `vllm/model_executor/models/registry.py`, maps the
  `architectures` string in a model's `config.json` to an implementation. The
  mapping is many-to-one: `InternLM3ForCausalLM` and several others resolve to
  the `llama` implementation. An architecture not in the registry needs a new
  implementation, an out-of-tree registration, or vLLM's fallback,
  `TransformersForCausalLM`, which runs Hugging Face's reference modeling code.
- llama.cpp names 153 architectures in `llama-arch.cpp` and constructs their
  graphs in code keyed by architecture. It separately recognises 57
  pre-tokenizer types in `llama-vocab.cpp`, selected by `tokenizer.ggml.pre`,
  and refuses to load a model whose pre-tokenizer it does not know — even on a
  supported architecture.

In both, new weights for a known architecture need no code, and a new
architecture, weight format or tokenizer does — the boundaries of principles 2
and 3. The one difference is the fallback, which is why principle 7 rejects by
name instead.
