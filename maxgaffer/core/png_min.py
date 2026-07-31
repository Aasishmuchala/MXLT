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


def _chunk(ctype: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def write_png_rgb(path: str, rows: List[List[Tuple[int, int, int]]]) -> Optional[str]:
    """Write rows of (r, g, b) 0-255 tuples as a non-interlaced 8-bit RGB PNG — the exact
    subset ``read_png_rgb`` decodes, so a read → transform → write round-trip stays inside
    this zero-dependency codec (Max's Python has no Pillow; the display-encode of OCIO-raw
    loop plates needs a writer as much as the stats need a reader). Filter 0 everywhere:
    the loop plates are small and zlib alone compresses them fine. → path, or None."""
    if not rows or not rows[0]:
        return None
    height, width = len(rows), len(rows[0])
    if width > _MAX_DIM or height > _MAX_DIM:
        return None
    raw = bytearray()
    for row in rows:
        if len(row) != width:
            return None                      # ragged input — not an image
        raw.append(0)                        # filter type 0 (None)
        for r, g, b in row:
            raw.append(r & 0xFF)
            raw.append(g & 0xFF)
            raw.append(b & 0xFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)   # 8-bit truecolor
    try:
        with open(path, "wb") as f:
            f.write(_SIG)
            f.write(_chunk(b"IHDR", ihdr))
            f.write(_chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
            f.write(_chunk(b"IEND", b""))
    except OSError:
        return None
    return path


def read_png_size(path: str) -> Optional[Tuple[int, int]]:
    """(width, height) from the IHDR chunk alone — NO zlib, no unfiltering, microseconds.

    Added 2026-07-31. ``compute_stats`` reports no dimensions at all, so until now nothing
    in the plugin could tell a 240×135 probe from a full-resolution frame that was saved
    at scene resolution because all three of ``render_frame``'s size spellings fell
    through (render.py:57-73). ``compute_stats`` downsamples to 256 px, so the stats of a
    wrong-sized plate look entirely normal and the critic ranks it happily — the same
    class of blindness as scoring a black frame.

    → None on a non-PNG, a truncated header, or an unreadable file. None reads as "could
    not verify" upstream (plate.validate WARNs and proceeds), never as "wrong size":
    plenty of fixtures and Max builds write formats this reader does not claim to know.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(len(_SIG) + 8 + 13)
    except OSError:
        return None
    if not head.startswith(_SIG) or len(head) < len(_SIG) + 8 + 13:
        return None
    pos = len(_SIG)
    (length,) = struct.unpack(">I", head[pos:pos + 4])
    if head[pos + 4:pos + 8] != b"IHDR" or length != 13:
        return None
    try:
        width, height = struct.unpack(">II", head[pos + 8:pos + 16])
    except struct.error:                    # unreachable after the length gate — belt
        return None
    if width <= 0 or height <= 0 or width > _MAX_DIM or height > _MAX_DIM:
        return None
    return int(width), int(height)


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
