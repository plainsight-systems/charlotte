"""Copies a GGUF file with one boolean metadata key added.

    python3 tools/gguf_add_bool.py <in.gguf> <out.gguf> <key> <true|false>

For the vision reference (tools/make_reference_vision.sh): llama.cpp chooses
a vision encoder's feed-forward activation from clip.use_gelu, which
EmbeddingGemma 2's mmproj lacks, and without it runs GELU-quick where Google
trained GELU's tanh form (docs/research/2026-10-11-embeddinggemma2-vision.md).
A copy with clip.use_gelu = true lets llama.cpp compute the activation the
model was trained with from the same weights.

The tensors' bytes are copied unchanged. Their offsets are relative to the
start of the data, which is re-aligned after the longer header, so no offset
changes. A key already present is refused rather than shadowed.
"""
import struct
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import gguf_header as g  # noqa: E402

BOOL = 7
CHUNK = 64 * 2**20


def main() -> int:
    if len(sys.argv) != 5 or sys.argv[4] not in ("true", "false"):
        print(__doc__, file=sys.stderr)
        return 2
    source, target, key, value = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "true"
    with open(source, "rb") as f:
        head = f.read(g.HEADER_BYTES)
    metadata, metadata_end = g.read_metadata(head)
    if key in metadata:
        print(f"{key} is already set", file=sys.stderr)
        return 1
    alignment = metadata.get("general.alignment", 32)

    # The tensor infos follow the metadata: walk them to find where they end.
    (tensor_count,) = struct.unpack_from("<Q", head, 8)
    pos = metadata_end
    for _ in range(tensor_count):
        (n,) = struct.unpack_from("<Q", head, pos)
        pos += 8 + n
        (dims,) = struct.unpack_from("<I", head, pos)
        pos += 4 + 8 * dims + 4 + 8
    infos_end = pos
    data_start = (infos_end + alignment - 1) // alignment * alignment

    (kv_count,) = struct.unpack_from("<Q", head, 16)
    name = key.encode()
    added = struct.pack("<Q", len(name)) + name + struct.pack("<I?", BOOL, value)
    header = head[:16] + struct.pack("<Q", kv_count + 1) + head[24:metadata_end] + added + head[metadata_end:infos_end]
    header += b"\0" * ((-len(header)) % alignment)

    with open(source, "rb") as f, open(target, "wb") as out:
        out.write(header)
        f.seek(data_start)
        while chunk := f.read(CHUNK):
            out.write(chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
