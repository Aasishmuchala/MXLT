"""Minimal stdlib PNG reader — the zero-dependency floor for image stats inside Max.

3ds Max 2026's Python ships neither numpy nor Pillow, and the analytic exposure/WB solver
must ALWAYS have pixel stats for the loop renders (they're our own 8-bit RGB(A) PNGs written
by Max, non-interlaced). This decodes exactly that subset with zlib + struct; anything else
returns None and the caller falls through to Pillow/numpy (venv, sidecar) or asks the bridge
to transcode via Max's own bitmap I/O.

Returns rows of (r, g, b) 0-255 tuples, subsampled to at most ``max_dim`` on the long side —
stats don't need every pixel and pure-python must stay fast.
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Optional, Tuple

_SIG = b"\x89PNG\r\n\x1a\n"

# Decompression-bomb guards: a hostile "reference" can declare huge-but-legal geometry and
# inflate to gigabytes on Max's MAIN thread (loop stats can never fail — or freeze). Cap the
# declared dimensions and bound the decompressed payload BEFORE allocating anything.
_MAX_DIM = 16384
_MAX_RAW_BYTES = 256 * 1024 * 1024      # ceiling on height * (stride + 1)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png_rgb(path: str, max_dim: int = 160) -> Optional[List[List[Tuple[int, int, int]]]]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if not data.startswith(_SIG):
        return None
    pos = len(_SIG)
    width = height = 0
    bit_depth = color_type = interlace = -1
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # length + type + data + crc
        if ctype == b"IHDR":
            if length != 13 or len(chunk) != 13:    # truncated/malformed IHDR: not a
                return None                         # PNG we can trust
            try:
                width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                    ">IIBBBBB", chunk)
            except struct.error:            # unreachable after the length gate — belt
                return None
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
    if width <= 0 or height <= 0 or bit_depth != 8 or interlace != 0:
        return None
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        return None
    stride = width * channels
    expected = height * (stride + 1)        # exact decoded size of a valid stream
    if width > _MAX_DIM or height > _MAX_DIM or expected > _MAX_RAW_BYTES:
        return None
    try:
        dec = zlib.decompressobj()
        raw = dec.decompress(bytes(idat), expected)     # bounded: never allocates past cap
    except zlib.error:
        return None
    # more payload than the declared geometry needs (bomb) or less (corrupt) → reject
    if dec.unconsumed_tail or len(raw) < expected:
        return None

    # subsample factor before unfiltering rows we keep — but filters reference the PREVIOUS
    # row, so every row must still be unfiltered in order; we just skip the pixel extraction.
    # ceil-divide: floor made 480//256 == 1, so the 256..511px band (the DEFAULT loop
    # render width!) processed every pixel and the "at most max_dim" contract was a lie
    step = max(1, -(-max(width, height) // max(1, max_dim)))
    rows: List[List[Tuple[int, int, int]]] = []
    prev = bytearray(stride)
    offset = 0
    for y in range(height):
        ftype = raw[offset]
        line = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride
        if ftype == 1:      # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:    # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:    # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:    # Paeth
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                up_left = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, prev[i], up_left)) & 0xFF
        elif ftype != 0:
            return None
        prev = line
        if y % step == 0:
            row: List[Tuple[int, int, int]] = []
            for x in range(0, width, step):
                base = x * channels
                if channels >= 3:
                    row.append((line[base], line[base + 1], line[base + 2]))
                else:  # grayscale (+alpha)
                    g = line[base]
                    row.append((g, g, g))
            rows.append(row)
    return rows if rows else None
