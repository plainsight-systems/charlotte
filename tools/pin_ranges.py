"""Pins the byte ranges of a GGUF file a load reads in part, and their hash.

    python3 tools/pin_ranges.py <file.gguf> <prefix>...

For a catalog entry's `vision` (web/catalog.js): a companion file holds more
than the page needs, as EmbeddingGemma 2's mmproj holds an audio encoder
beside the vision one, and the page downloads only the header and the
tensors whose names begin with one of the prefixes given. Prints the entry's
fields as JSON: the file's size; the ranges, [begin, end) pairs in order, the
first from 0 to the start of the tensor data, each kept tensor running to the
next tensor's offset or the end of the file, adjacent ones merged
(residency/file_ranges.h); and the SHA-256 of those ranges' bytes in order,
which the download checks in place of the whole file's (web/download.js).
"""
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gguf_header as g  # noqa: E402

CHUNK = 64 * 2**20


def tensors(head, metadata_end):
    """Each tensor's (offset, name), in the file's order, and where the infos end."""
    (count,) = struct.unpack_from("<Q", head, 8)
    pos = metadata_end
    out = []
    for _ in range(count):
        (n,) = struct.unpack_from("<Q", head, pos)
        name = head[pos + 8 : pos + 8 + n].decode()
        pos += 8 + n
        (dims,) = struct.unpack_from("<I", head, pos)
        pos += 4 + 8 * dims
        (offset,) = struct.unpack_from("<Q", head, pos + 4)
        pos += 12
        out.append((offset, name))
    return out, pos


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    path, prefixes = sys.argv[1], tuple(sys.argv[2:])
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(g.HEADER_BYTES)
    metadata, metadata_end = g.read_metadata(head)
    alignment = metadata.get("general.alignment", 32)
    found, infos_end = tensors(head, metadata_end)
    data_start = (infos_end + alignment - 1) // alignment * alignment

    by_offset = sorted(found)
    ranges = [[0, data_start]]
    for i, (offset, name) in enumerate(by_offset):
        if not name.startswith(prefixes):
            continue
        begin = data_start + offset
        end = data_start + by_offset[i + 1][0] if i + 1 < len(by_offset) else size
        if ranges[-1][1] == begin:
            ranges[-1][1] = end
        else:
            ranges.append([begin, end])
    if len(ranges) == 1:
        print("no tensor begins with the prefixes given", file=sys.stderr)
        return 1

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for begin, end in ranges:
            f.seek(begin)
            left = end - begin
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                digest.update(chunk)
                left -= len(chunk)
    print(json.dumps({"sizeBytes": size, "ranges": ranges, "rangesSha256": digest.hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
