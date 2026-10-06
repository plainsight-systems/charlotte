#!/usr/bin/env python3
"""Builds small synthetic GGUF files for the reader tests.

Fixtures are generated rather than committed as opaque blobs so a reviewer can
see exactly what each byte is, and so a malformed case is one deliberate
mutation away from the valid one rather than a hand-edited binary nobody can
audit.

CI must never download the real 420 MB model, so every automated test runs
against these.
"""
import struct
import sys

MAGIC = b"GGUF"
VERSION = 3

# gguf_type
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# ggml_type
T_F32, T_F16, T_Q4_0 = 0, 1, 2
# A real GGUF format the harness has no kernel for. The reader must record a
# tensor in it, not reject the file.
T_Q6_K = 14
Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES = 256, 210
# A format no listed model uses, so the harness will never run it: what a
# test of an unsupported format reports stays unsupported.
T_Q5_0 = 6
Q5_0_BLOCK_BYTES = 22
T_Q4_1, T_Q8_0 = 3, 8


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def kv(key: bytes, vtype: int, payload: bytes) -> bytes:
    return gstr(key) + struct.pack("<I", vtype) + payload


def build(tensors, metadata=(), *, magic=MAGIC, version=VERSION,
          tensor_count=None, metadata_count=None, alignment=32):
    """tensors: list of (name, dims, ggml_type, data_bytes)."""
    md = b"".join(metadata)
    md_count = metadata_count if metadata_count is not None else len(metadata)
    t_count = tensor_count if tensor_count is not None else len(tensors)

    head = magic + struct.pack("<I", version)
    head += struct.pack("<QQ", t_count, md_count)
    head += md

    # Lay tensors out back to back, each padded to `alignment`.
    infos, blobs, offset = b"", b"", 0
    for name, dims, ttype, data in tensors:
        infos += gstr(name) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<q", d)
        infos += struct.pack("<I", ttype) + struct.pack("<Q", offset)
        pad = (-len(data)) % alignment
        blobs += data + b"\0" * pad
        offset += len(data) + pad

    body = head + infos
    body += b"\0" * ((-len(body)) % alignment)
    return body + blobs


def q4_0_blocks(n: int, scale_bits: int = 0x3C00, nibble_byte: int = 0x10) -> bytes:
    """n Q4_0 blocks: fp16 scale then 16 bytes of packed nibbles."""
    return (struct.pack("<H", scale_bits) + bytes([nibble_byte]) * 16) * n


def valid() -> bytes:
    return build(
        tensors=[
            (b"token_embd.weight", [64, 2], T_Q4_0, q4_0_blocks(4)),
            (b"output_norm.weight", [4], T_F32, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
        ],
        metadata=[
            kv(b"general.architecture", STRING, gstr(b"qwen3")),
            kv(b"general.alignment", U32, struct.pack("<I", 32)),
            kv(b"qwen3.block_count", U32, struct.pack("<I", 28)),
            kv(b"qwen3.attention.head_count", U32, struct.pack("<I", 16)),
            kv(b"tokenizer.ggml.tokens", ARRAY,
               struct.pack("<IQ", STRING, 3) + gstr(b"a") + gstr(b"bb") + gstr(b"ccc")),
        ],
    )


def token_arrays() -> bytes:
    """A vocabulary's three arrays: strings (one empty, one multi-byte), their
    int32 types, and float32 scores."""
    tokens = [b"", b"a", "h\u00e9llo".encode(), b"<|x|>"]
    return build(
        tensors=[],
        metadata=[
            kv(b"tokenizer.ggml.tokens", ARRAY,
               struct.pack("<IQ", STRING, len(tokens)) + b"".join(gstr(t) for t in tokens)),
            kv(b"tokenizer.ggml.token_type", ARRAY,
               struct.pack("<IQ", I32, 4) + struct.pack("<4i", 3, 1, 1, -4)),
            kv(b"tokenizer.ggml.scores", ARRAY,
               struct.pack("<IQ", F32, 4) + struct.pack("<4f", -1000.0, -1.5, 0.0, 2.25)),
        ],
    )


def _with_tensor_offset(bogus_offset: int) -> bytes:
    """A structurally valid file whose single tensor claims an absurd offset."""
    data = q4_0_blocks(1)
    head = MAGIC + struct.pack("<I", VERSION) + struct.pack("<QQ", 1, 0)
    info = gstr(b"t") + struct.pack("<I", 1) + struct.pack("<q", 32)
    info += struct.pack("<I", T_Q4_0) + struct.pack("<Q", bogus_offset)
    body = head + info
    body += b"\0" * ((-len(body)) % 32)
    return body + data + b"\0" * ((-len(data)) % 32)


def _one_tensor_with(*metadata):
    """A one-tensor file carrying exactly the given metadata, for the gates."""
    return build([(b"t", [4], T_F32, b"\0" * 16)], metadata=list(metadata))


# A tiny but complete transformer, in the shape a real converter writes:
# embedding 8, 2 query heads and 1 key/value head of width 4, feed-forward 16,
# a 6-token vocabulary. Enough for describe to check every key, name and shape.
E, H, KV, D, F, VOCAB = 32, 2, 1, 32, 64, 6
ROLE_SHAPES = {
    "attn_norm": [E], "attn_q": [E, H * D], "attn_k": [E, KV * D], "attn_v": [E, KV * D],
    "attn_q_norm": [D], "attn_k_norm": [D], "attn_output": [H * D, E],
    "post_attention_norm": [E], "ffn_norm": [E], "ffn_gate": [E, F], "ffn_up": [E, F],
    "ffn_down": [F, E], "post_ffw_norm": [E],
}
ARCH_ROLES = {
    "qwen3": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
              "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"],
    "llama": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
              "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"],
    "gemma3": ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
               "attn_output", "post_attention_norm", "ffn_norm", "ffn_gate", "ffn_up",
               "ffn_down", "post_ffw_norm"],
}


def f32_zeros(dims):
    n = 1
    for d in dims:
        n *= d
    return b"\0" * (4 * n)


def embedding(ggml_type, rows=7, width=256, seed=0x2545F491):
    """A token embedding table alone, `rows` tokens of `width` weights, of
    deterministic pseudo-random blocks: codes of every value, fp16 scales
    finite and of either sign. What a gather of it should write is the
    format's CPU reference applied to the same stored blocks."""
    state = seed

    def u32():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state

    def raw(n):
        return bytes((u32() >> 24) & 0xFF for _ in range(n))

    def half():
        return struct.pack("<e", ((u32() >> 8) % 2001 - 1000) / 500.0)

    blocks_per_row = {T_F32: width, T_Q4_0: width // 32, T_Q4_1: width // 32,
                      T_Q8_0: width // 32, T_Q6_K: width // Q6_K_BLOCK_ELEMENTS}[ggml_type]
    block = {
        T_F32: lambda: struct.pack("<f", ((u32() >> 8) % 20001 - 10000) / 2500.0),
        T_Q4_0: lambda: half() + raw(16),
        T_Q4_1: lambda: half() + half() + raw(16),
        T_Q8_0: lambda: half() + raw(32),
        T_Q6_K: lambda: raw(128) + raw(64) + raw(16) + half(),
    }[ggml_type]
    data = b"".join(block() for _ in range(rows * blocks_per_row))
    return build([(b"token_embd.weight", [width, rows], ggml_type, data)])


def norm_rows(widths=(1024, 1152, 2048), rows=4, seed=0x6C8E9CF5):
    """For each width a listed model has — Qwen3 0.6B's, Gemma 3 1B's, Llama
    3.2 1B's — a norm's gain and a post-norm's gain, and rows of a hidden
    state x and a block's output y, of deterministic pseudo-random F32: rows
    of ordinary size, of 10^4 and of 10^-4, so a norm is checked across
    magnitudes. Activations, held as tensors so a test can upload them."""
    state = seed

    def unit():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return ((state >> 8) % 2000001 - 1000000) / 1000000.0

    def floats(values):
        return b"".join(struct.pack("<f", v) for v in values)

    scales = [1.0, 1e4, 1e-4, 3.0]
    tensors = []
    for w in widths:
        tensors.append((f"gain_{w}".encode(), [w], T_F32, floats(1.0 + 0.5 * unit() for _ in range(w))))
        tensors.append((f"post_{w}".encode(), [w], T_F32, floats(1.0 + 0.5 * unit() for _ in range(w))))
        for name in ("x", "y"):
            data = b"".join(floats(scales[r] * unit() for _ in range(w)) for r in range(rows))
            tensors.append((f"{name}_{w}".encode(), [w, rows], T_F32, data))
    return build(tensors)


def tiny_model(arch, layers=2, *, extra=(), omit_key=None, omit_tensor=None, reshape=None,
               head_count_kv=KV, output_copy=False, context=64, extra_tensors=()):
    """A tiny `arch` model. `reshape` is (tensor name, dims) to break a shape."""
    keys = {
        "block_count": (U32, struct.pack("<I", layers)),
        "context_length": (U32, struct.pack("<I", context)),
        "embedding_length": (U32, struct.pack("<I", E)),
        "feed_forward_length": (U32, struct.pack("<I", F)),
        "attention.head_count": (U32, struct.pack("<I", H)),
        "attention.head_count_kv": (U32, struct.pack("<I", head_count_kv)),
        "attention.key_length": (U32, struct.pack("<I", D)),
        "attention.value_length": (U32, struct.pack("<I", D)),
        "attention.layer_norm_rms_epsilon": (F32, struct.pack("<f", 1e-6)),
        "rope.freq_base": (F32, struct.pack("<f", 1e6)),
    }
    metadata = [kv(b"general.architecture", STRING, gstr(arch.encode()))]
    for key, (vtype, payload) in keys.items():
        if key != omit_key:
            metadata.append(kv(f"{arch}.{key}".encode(), vtype, payload))
    tokens = b"".join(gstr(t) for t in [b"a", b"b", b"c", b"d", b"e", b"f"])
    metadata.append(kv(b"tokenizer.ggml.tokens", ARRAY, struct.pack("<IQ", STRING, VOCAB) + tokens))
    metadata.extend(extra)

    shapes = {"token_embd.weight": [E, VOCAB], "output_norm.weight": [E]}
    if output_copy:
        shapes["output.weight"] = [E, VOCAB]
    for layer in range(layers):
        for role in ARCH_ROLES[arch]:
            shapes[f"blk.{layer}.{role}.weight"] = ROLE_SHAPES[role]
    if reshape is not None:
        shapes[reshape[0]] = reshape[1]
    tensors = [(name.encode(), dims, T_F32, f32_zeros(dims))
               for name, dims in shapes.items() if name != omit_tensor]
    return build(list(extra_tensors) + tensors, metadata=metadata)


def _tensors_at(*placements, alignment=32):
    """A file whose tensors sit at chosen relative offsets: (name, offset)."""
    data = q4_0_blocks(1)
    head = MAGIC + struct.pack("<I", VERSION) + struct.pack("<QQ", len(placements), 0)
    info = b""
    for name, offset in placements:
        info += gstr(name) + struct.pack("<I", 1) + struct.pack("<q", 32)
        info += struct.pack("<I", T_Q4_0) + struct.pack("<Q", offset)
    body = head + info
    body += b"\0" * ((-len(body)) % alignment)
    return body + data + b"\0" * 64


def _with_alignment(vtype, payload, alignment=32):
    return build([(b"t", [4], T_F32, b"\0" * 16)],
                 metadata=[kv(b"general.alignment", vtype, payload)], alignment=alignment)


CASES = {
    "valid": valid,
    "token_arrays": token_arrays,
    "bad_magic": lambda: build([], magic=b"GGUX"),
    "bad_version": lambda: build([], version=2),
    "truncated_header": lambda: MAGIC + struct.pack("<I", VERSION) + b"\x01\x02",
    # Claims more tensors than the file can possibly contain.
    "tensor_count_too_large": lambda: build([], tensor_count=1 << 40),
    "metadata_count_lies": lambda: build([], metadata_count=5),
    # A tensor whose DECLARED offset puts its data past the end of the file.
    # Trimming trailing padding would not do this: the padding is not part of
    # any declared region, so removing it leaves every tensor still resident.
    "offset_past_eof": lambda: _with_tensor_offset(1 << 40),
    # An offset chosen so data_start + offset WRAPS past 2^64 rather than
    # merely exceeding the file. A naive `start + offset > size` test computes
    # a small number here and reports the region as in-bounds.
    "offset_overflow": lambda: _with_tensor_offset((1 << 64) - 64),
    # Truncated so the cut lands inside declared tensor data, not padding.
    "data_truncated": lambda: build(
        [(b"t", [64], T_Q4_0, q4_0_blocks(2))]
    )[:-40],
    "negative_dimension": lambda: build(
        [(b"t", [-4], T_F32, b"\0" * 16)]
    ),
    "unknown_tensor_type": lambda: build(
        [(b"t", [4], 999, b"\0" * 16)]
    ),
    # Not malformed: Q6_K is a real format. Whether it can run is decided by
    # the gates, so the reader records it like any other tensor.
    "q6_k_tensor": lambda: build(
        [(b"t", [Q6_K_BLOCK_ELEMENTS], T_Q6_K, b"\0" * Q6_K_BLOCK_BYTES)]
    ),
    # Two tensors share a format, so a gate that reports per format must
    # report it once, with both counted.
    "shared_format": lambda: build([
        (b"first.weight", [32], T_Q5_0, b"\0" * Q5_0_BLOCK_BYTES),
        (b"second.weight", [32], T_Q5_0, b"\0" * Q5_0_BLOCK_BYTES),
        (b"norm.weight", [4], T_F32, b"\0" * 16),
    ]),
    "not_block_aligned": lambda: build(
        # 33 elements is not a whole number of Q4_0 blocks.
        [(b"t", [33], T_Q4_0, q4_0_blocks(2))]
    ),
    "duplicate_tensor_name": lambda: build([
        (b"dup", [4], T_F32, b"\0" * 16),
        (b"dup", [4], T_F32, b"\0" * 16),
    ]),
    # Duplicates separated by another tensor, so a check that compares only
    # neighbours in file order would miss them.
    "duplicate_tensor_name_apart": lambda: build([
        (b"dup", [4], T_F32, b"\0" * 16),
        (b"other", [4], T_F32, b"\0" * 16),
        (b"dup", [4], T_F32, b"\0" * 16),
    ]),
    # A declared alignment other than the default is honoured.
    "alignment_64": lambda: build(
        [(b"a", [4], T_F32, b"\0" * 16), (b"b", [4], T_F32, b"\0" * 16)],
        metadata=[kv(b"general.alignment", U32, struct.pack("<I", 64))], alignment=64),
    # The alignment must be a power-of-two uint32; neither is a reason to
    # fall back to the default.
    "alignment_wrong_type": lambda: _with_alignment(U64, struct.pack("<Q", 32)),
    "alignment_not_power_of_two": lambda: _with_alignment(U32, struct.pack("<I", 48)),
    # The spec: every tensor offset is a multiple of the alignment.
    "misaligned_tensor_offset": lambda: _tensors_at((b"t", 16)),
    "overlapping_tensor_data": lambda: _tensors_at((b"a", 0), (b"b", 0)),
    "empty_metadata_key": lambda: build(
        [], metadata=[kv(b"", U32, struct.pack("<I", 1))]),
    "duplicate_metadata_key": lambda: build(
        [], metadata=[kv(b"general.name", STRING, gstr(b"a")),
                      kv(b"general.type", STRING, gstr(b"model")),
                      kv(b"general.name", STRING, gstr(b"b"))]),
    # The format's limit on a name is 64 bytes: exactly that is fine.
    "tensor_name_64_bytes": lambda: build([(b"n" * 64, [4], T_F32, b"\0" * 16)]),
    "tensor_name_65_bytes": lambda: build([(b"n" * 65, [4], T_F32, b"\0" * 16)]),
    # For the preflight gates: files that are valid GGUF but name their
    # architecture and tokenizer in the ways a gate must report.
    "no_architecture": lambda: _one_tensor_with(),
    "architecture_not_string": lambda: _one_tensor_with(
        kv(b"general.architecture", U32, struct.pack("<I", 3))),
    "tokenizer_named": lambda: _one_tensor_with(
        kv(b"general.architecture", STRING, gstr(b"qwen3")),
        kv(b"tokenizer.ggml.model", STRING, gstr(b"gpt2")),
        kv(b"tokenizer.ggml.pre", STRING, gstr(b"qwen2"))),
    "tokenizer_without_pre": lambda: _one_tensor_with(
        kv(b"general.architecture", STRING, gstr(b"qwen3")),
        kv(b"tokenizer.ggml.model", STRING, gstr(b"gpt2"))),
    "sentencepiece_default_pre": lambda: _one_tensor_with(
        kv(b"general.architecture", STRING, gstr(b"gemma3")),
        kv(b"tokenizer.ggml.model", STRING, gstr(b"llama")),
        kv(b"tokenizer.ggml.pre", STRING, gstr(b"default"))),
    "unknown_pretokenizer": lambda: _one_tensor_with(
        kv(b"general.architecture", STRING, gstr(b"qwen3")),
        kv(b"tokenizer.ggml.model", STRING, gstr(b"gpt2")),
        kv(b"tokenizer.ggml.pre", STRING, gstr(b"no-such-split"))),
    # Describe: one complete tiny model per architecture, and one file broken
    # in each way describe must name.
    "tiny_qwen3": lambda: tiny_model("qwen3"),
    "tiny_llama": lambda: tiny_model("llama"),
    # An output head stored as its own tensor, the same shape and format as
    # the token embedding: the case residency treats as a candidate duplicate.
    "tiny_qwen3_output_copy": lambda: tiny_model("qwen3", output_copy=True),
    # Three rows of one Q4_0 block, first in the file: 18 bytes a row, 54 in
    # all, no multiple of 4.
    "tiny_qwen3_odd_blocks": lambda: tiny_model(
        "qwen3", extra_tensors=[(b"extra.weight", [32, 3], T_Q4_0, q4_0_blocks(3))]),
    # A row of 33 F32 weights: a whole number of F32's one-weight blocks, but
    # not of the 32-weight groups unpack steps through.
    "tiny_qwen3_odd_row": lambda: tiny_model(
        "qwen3", extra_tensors=[(b"extra.norm", [33], T_F32, b"\0" * 132)]),
    # Seven layers: a run of six (five window layers, one global) and one more.
    "tiny_gemma3": lambda: tiny_model("gemma3", layers=7, extra=[
        kv(b"gemma3.attention.sliding_window", U32, struct.pack("<I", 16))]),
    "tiny_gemma3_no_window": lambda: tiny_model("gemma3", layers=7),
    # Trained for longer than a window layer's ring (16 + 512 + 4,096 slots),
    # so the ring, not the context, sizes the window layers' cache.
    "tiny_gemma3_long": lambda: tiny_model("gemma3", layers=7, context=8192, extra=[
        kv(b"gemma3.attention.sliding_window", U32, struct.pack("<I", 16))]),
    # A pattern longer than the model: every layer attends over the window.
    "tiny_gemma3_long_all_window": lambda: tiny_model("gemma3", layers=7, context=8192, extra=[
        kv(b"gemma3.attention.sliding_window", U32, struct.pack("<I", 16)),
        kv(b"gemma3.attention.sliding_window_pattern", U32, struct.pack("<I", 8))]),
    "tiny_gemma3_pattern_per_layer": lambda: tiny_model("gemma3", layers=7, extra=[
        kv(b"gemma3.attention.sliding_window", U32, struct.pack("<I", 16)),
        kv(b"gemma3.attention.sliding_window_pattern", ARRAY,
           struct.pack("<IQ", BOOL, 7) + bytes([1, 1, 1, 1, 1, 0, 1]))]),
    # Llama 3's rotary frequency factors: one F32 a pair; then a wrong width,
    # and a type the rope kernel does not bind.
    "tiny_llama_rope_freqs": lambda: tiny_model("llama", extra_tensors=[
        (b"rope_freqs.weight", [D // 2], T_F32, f32_zeros([D // 2]))]),
    "tiny_llama_rope_freqs_wrong_shape": lambda: tiny_model("llama", extra_tensors=[
        (b"rope_freqs.weight", [D], T_F32, f32_zeros([D]))]),
    "tiny_llama_rope_freqs_f16": lambda: tiny_model("llama", extra_tensors=[
        (b"rope_freqs.weight", [D // 2], T_F16, b"\0" * D)]),
    # Rotary keys a file may declare: whole heads and no scaling run; part of
    # each head, or scaled positions, are refused.
    "tiny_qwen3_rope_declared": lambda: tiny_model("qwen3", extra=[
        kv(b"qwen3.rope.dimension_count", U32, struct.pack("<I", D)),
        kv(b"qwen3.rope.scaling.type", STRING, gstr(b"none"))]),
    "tiny_qwen3_partial_rotation": lambda: tiny_model("qwen3", extra=[
        kv(b"qwen3.rope.dimension_count", U32, struct.pack("<I", D // 2))]),
    "tiny_qwen3_rope_scaling": lambda: tiny_model("qwen3", extra=[
        kv(b"qwen3.rope.scaling.type", STRING, gstr(b"linear"))]),
    "tiny_qwen3_missing_key": lambda: tiny_model("qwen3", omit_key="attention.head_count_kv"),
    "tiny_qwen3_missing_tensor": lambda: tiny_model("qwen3", omit_tensor="blk.1.ffn_up.weight"),
    "tiny_qwen3_wrong_shape": lambda: tiny_model("qwen3", reshape=("blk.1.attn_k.weight", [E, 2 * D])),
    "tiny_qwen3_heads_not_grouped": lambda: tiny_model("qwen3", head_count_kv=3),
    "tiny_qwen3_too_many_layers": lambda: build(
        [(b"token_embd.weight", [E, VOCAB], T_F32, f32_zeros([E, VOCAB]))],
        metadata=[kv(b"general.architecture", STRING, gstr(b"qwen3"))] + [
            kv(f"qwen3.{k}".encode(), t, p) for k, (t, p) in {
                "block_count": (U32, struct.pack("<I", 4_000_000_000)),
                "context_length": (U32, struct.pack("<I", 64)),
                "embedding_length": (U32, struct.pack("<I", E)),
                "feed_forward_length": (U32, struct.pack("<I", F)),
                "attention.head_count": (U32, struct.pack("<I", H)),
                "attention.head_count_kv": (U32, struct.pack("<I", KV)),
                "attention.layer_norm_rms_epsilon": (F32, struct.pack("<f", 1e-6)),
                "rope.freq_base": (F32, struct.pack("<f", 1e6)),
            }.items()]),
    # A token embedding table in each format a kernel reads weights in.
    "embedding_f32": lambda: embedding(T_F32),
    "embedding_q4_0": lambda: embedding(T_Q4_0),
    "embedding_q4_1": lambda: embedding(T_Q4_1),
    "embedding_q8_0": lambda: embedding(T_Q8_0),
    "embedding_q6_k": lambda: embedding(T_Q6_K),
    # Norm gains and activation rows at each listed model's hidden width.
    "norm_rows": lambda: norm_rows(),
    "nested_array": lambda: build(
        [], metadata=[kv(b"bad", ARRAY, struct.pack("<IQ", ARRAY, 1))]
    ),
    "unknown_value_type": lambda: build(
        [], metadata=[kv(b"bad", 99, b"\0" * 4)]
    ),
}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in CASES:
        print(f"usage: {sys.argv[0]} <case> <out>\ncases: {', '.join(CASES)}",
              file=sys.stderr)
        return 2
    with open(sys.argv[2], "wb") as f:
        f.write(CASES[sys.argv[1]]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
