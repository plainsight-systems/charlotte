"""Writes the four named scenes of tests/fixtures/vision/scenes.h as PNGs.

    python3 tools/make_vision_photos.py

For checking the search panel in a browser (web/search.js): the same pixels
the vision fixtures give llama.cpp and the GPU tests, as files a page can be
given. Each is 8-bit RGB with no colour profile, so the browser decodes the
pixels exactly (web/image_input.js). The formulas are scenes.h's.
"""
import os
import struct
import sys
import zlib

SCENES = {
    "red_circle": (320, 240),
    "blue_square": (240, 320),
    "green_stripes": (400, 200),
    "black_diagonal": (300, 300),
}


def pixel(name, x, y):
    if name == "red_circle":
        inside = (x - 160) ** 2 + (y - 120) ** 2 <= 80 * 80
        return (220, 30, 30) if inside else (255, 255, 255)
    if name == "blue_square":
        inside = 60 <= x < 180 and 100 <= y < 220
        return (30, 60, 220) if inside else (250, 220, 40)
    if name == "green_stripes":
        return (40, 170, 60) if (x // 25) % 2 == 0 else (255, 255, 255)
    if name == "black_diagonal":
        return (0, 0, 0) if -6 <= x - y <= 6 else (255, 255, 255)
    raise ValueError(name)


def png(width, height, rows):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\0" + bytes(row) for row in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main() -> int:
    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "vision")
    for name, (width, height) in SCENES.items():
        rows = [[c for x in range(width) for c in pixel(name, x, y)] for y in range(height)]
        with open(os.path.join(out, f"{name}.png"), "wb") as f:
            f.write(png(width, height, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
