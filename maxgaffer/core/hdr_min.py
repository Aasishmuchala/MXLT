"""Minimal stdlib Radiance .hdr (RGBE) codec — the dome-seed's output format.

png_min.py's sibling: 3ds Max's Python has no imageio, and the seeded dome must hand V-Ray
a real high-dynamic-range file (a PNG dome caps at 1.0 and can't carry a sun). Radiance
RGBE is the one HDR container simple enough to emit from pure stdlib: 4 bytes/pixel, a
shared exponent, no compression tables. VRayBitmap reads .hdr natively.

Writer emits new-style RLE scanlines for widths 8..32767 (the spec says scanlines in that
range SHOULD be new-style, and some strict loaders assume it) and flat scanlines otherwise.
Reader decodes both — it exists for round-trip tests and for re-ingesting our own seeds;
anything exotic (old-style RLE, XYZE) returns None and the caller falls back to Max I/O.

Values are LINEAR radiance floats, rows top-first, ``rows[y][x] = (r, g, b)``.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

_HEADER_MAGIC = (b"#?RADIANCE", b"#?RGBE")
_MIN_RLE_WIDTH, _MAX_RLE_WIDTH = 8, 32767


# --------------------------------------------------------------------------- pixel codec
def float_to_rgbe(r: float, g: float, b: float) -> Tuple[int, int, int, int]:
    # per-channel sanitize BEFORE max(): max(1.0, nan) keeps the finite value, so a NaN
    # in a non-max channel would sail past a max-only guard into int(nan) → ValueError
    r = r if math.isfinite(r) and r > 0.0 else 0.0
    g = g if math.isfinite(g) and g > 0.0 else 0.0
    b = b if math.isfinite(b) and b > 0.0 else 0.0
    m = max(r, g, b)
    if m < 1e-9:
        return 0, 0, 0, 0
    _mant, exp = math.frexp(m)              # m = mant * 2**exp, mant in [0.5, 1)
    if exp > 127:                           # finite but beyond RGBE range: saturate to the
        exp = 127                           # hottest exponent, channel ratios preserved
    scale = math.ldexp(1.0, 8 - exp)        # component * scale lands in [0, 256)
    return (
        min(255, max(0, int(r * scale))),
        min(255, max(0, int(g * scale))),
        min(255, max(0, int(b * scale))),
        exp + 128,
    )


def rgbe_to_float(r8: int, g8: int, b8: int, e8: int) -> Tuple[float, float, float]:
    if e8 == 0:
        return 0.0, 0.0, 0.0
    f = math.ldexp(1.0, e8 - 136)           # 2**(exp - 8)
    return r8 * f, g8 * f, b8 * f


# --------------------------------------------------------------------------- write
def _rle_encode_plane(plane: bytes) -> bytes:
    """One component plane → new-style RLE: runs (0x80+len, byte), literals (len, bytes),
    run lengths 1..127, literal lengths 1..128. Runs shorter than 4 ride in literals
    (matches Radiance's own encoder — a 2-byte run costs as much as it saves)."""
    out = bytearray()
    n = len(plane)
    pos = 0
    lit_start = pos
    while pos < n:
        run_len = 1
        while pos + run_len < n and run_len < 127 and plane[pos + run_len] == plane[pos]:
            run_len += 1
        if run_len >= 4:
            while lit_start < pos:                      # flush pending literal first
                take = min(128, pos - lit_start)
                out.append(take)
                out.extend(plane[lit_start:lit_start + take])
                lit_start += take
            out.append(0x80 + run_len)
            out.append(plane[pos])
            pos += run_len
            lit_start = pos
        else:
            pos += run_len
    while lit_start < pos:
        take = min(128, pos - lit_start)
        out.append(take)
        out.extend(plane[lit_start:lit_start + take])
        lit_start += take
    return bytes(out)


def write_hdr(path: str, rows: Sequence[Sequence[Tuple[float, float, float]]]) -> bool:
    """Rows of (r, g, b) linear floats → Radiance .hdr. False on empty/ragged input or
    OSError — mirrors the quiet-degrade convention of png_min."""
    if not rows or not rows[0]:
        return False
    height, width = len(rows), len(rows[0])
    if any(len(r) != width for r in rows):
        return False
    use_rle = _MIN_RLE_WIDTH <= width <= _MAX_RLE_WIDTH
    try:
        with open(path, "wb") as f:
            f.write(b"#?RADIANCE\n")
            f.write(b"# MaxGaffer dome seed\n")
            f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
            f.write(f"-Y {height} +X {width}\n".encode("ascii"))
            for row in rows:
                px = [float_to_rgbe(*p) for p in row]
                if use_rle:
                    f.write(bytes((2, 2, (width >> 8) & 0xFF, width & 0xFF)))
                    for c in range(4):
                        f.write(_rle_encode_plane(bytes(p[c] for p in px)))
                else:
                    f.write(bytes(b for p in px for b in p))
        return True
    except (OSError, TypeError, ValueError):   # documented contract: False, never raise
        try:
            os.remove(path)               # never leave a truncated .hdr behind
        except OSError:
            pass
        return False


# --------------------------------------------------------------------------- read
def _read_header(data: bytes) -> Optional[Tuple[int, int, int]]:
    """→ (width, height, offset-of-first-scanline) or None."""
    if not any(data.startswith(m) for m in _HEADER_MAGIC):
        return None
    pos = 0
    fmt_ok = False
    while pos < len(data):
        nl = data.find(b"\n", pos)
        if nl < 0:
            return None
        line = data[pos:nl]
        pos = nl + 1
        if line.startswith(b"FORMAT="):
            fmt_ok = line.strip() == b"FORMAT=32-bit_rle_rgbe"
        elif line == b"":                    # blank line ends the header
            break
    if not fmt_ok:
        return None
    nl = data.find(b"\n", pos)
    if nl < 0:
        return None
    parts = data[pos:nl].split()
    # only the standard orientation is supported (it's what we write)
    if len(parts) != 4 or parts[0] != b"-Y" or parts[2] != b"+X":
        return None
    try:
        height, width = int(parts[1]), int(parts[3])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height, nl + 1


def _decode_rle_scanline(data: bytes, pos: int, width: int
                         ) -> Optional[Tuple[List[Tuple[int, int, int, int]], int]]:
    planes: List[List[int]] = []
    for _c in range(4):
        plane: List[int] = []
        while len(plane) < width:
            if pos >= len(data):
                return None
            code = data[pos]
            pos += 1
            if code > 128:                   # run
                if pos >= len(data):
                    return None
                plane.extend([data[pos]] * (code - 128))
                pos += 1
            elif code > 0:                   # literal
                if pos + code > len(data):
                    return None
                plane.extend(data[pos:pos + code])
                pos += code
            else:
                return None
        if len(plane) != width:
            return None
        planes.append(plane)
    return [tuple(planes[c][x] for c in range(4)) for x in range(width)], pos


def read_hdr(path: str) -> Optional[List[List[Tuple[float, float, float]]]]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    head = _read_header(data)
    if head is None:
        return None
    width, height, pos = head
    # Old-style RLE honesty: scanlines in the 8..32767 band without the 2,2,w>>8,w marker
    # are only decodable as flat data when the payload is EXACTLY flat-sized AND free of
    # old-style repeat markers — (1,1,1,n) pixels mean the file is old-style RLE, which
    # flat-decodes into garbage; we refuse (None) rather than silently misdecode.
    in_band = _MIN_RLE_WIDTH <= width <= _MAX_RLE_WIDTH
    flat_exact = len(data) - pos == height * width * 4
    if flat_exact and in_band:
        flat_exact = not any(
            data[pos + i * 4] == 1 and data[pos + i * 4 + 1] == 1
            and data[pos + i * 4 + 2] == 1
            for i in range(height * width))
    if (
        in_band
        and not (pos + 4 <= len(data) and data[pos] == 2 and data[pos + 1] == 2
                 and ((data[pos + 2] << 8) | data[pos + 3]) == width)
        and not flat_exact
    ):
        return None
    rows: List[List[Tuple[float, float, float]]] = []
    for _y in range(height):
        is_rle = (
            in_band
            and pos + 4 <= len(data)
            and data[pos] == 2 and data[pos + 1] == 2
            and ((data[pos + 2] << 8) | data[pos + 3]) == width
        )
        if in_band and not is_rle and not flat_exact:
            return None     # in-band scanlines are new-style RLE by spec — anything else
                            # here is old-style RLE, which flat-decodes into garbage
        if is_rle:
            decoded = _decode_rle_scanline(data, pos + 4, width)
            if decoded is None:
                return None
            px, pos = decoded
        else:                                # flat: width × 4 raw bytes
            need = width * 4
            if pos + need > len(data):
                return None
            px = [tuple(data[pos + i * 4:pos + i * 4 + 4]) for i in range(width)]
            pos += need
        rows.append([rgbe_to_float(*p) for p in px])
    return rows
