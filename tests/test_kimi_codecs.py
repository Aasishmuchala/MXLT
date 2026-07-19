"""Round-2 audit regressions for the stdlib codecs + stats batch path (Cluster B).

Covers:
  - png_min: malformed IHDR → None (never struct.error), decompression-bomb guards
    (declared-dimension ceiling + bounded inflate via decompressobj max_length),
    max_dim=0 robustness
  - hdr_min: float_to_rgbe exponent saturation for huge finite values, write_hdr's
    "False, never raise" contract, honest rejection of old-style RLE scanlines
  - metrics/metrics_cli: one corrupt image degrades to None/null, never kills the batch

All hand-rolled bytes — no Pillow, no pymxs; runs anywhere.
"""

import json
import struct
import zlib

from maxgaffer.core import hdr_min, metrics, png_min
from maxgaffer.sidecar import metrics_cli

_SIG = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- helpers
def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def _png_bytes(width, height, pixels):
    """A real, valid 8-bit RGB PNG (filter-0 scanlines) around ``pixels``."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))
    return (_SIG + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
            + _png_chunk(b"IEND", b""))


def _png_with_ihdr(ihdr_payload: bytes, declared_len=None, idat: bytes = b""):
    """PNG bytes with full control over the IHDR chunk (for malformed variants)."""
    length = len(ihdr_payload) if declared_len is None else declared_len
    out = bytearray(_SIG)
    out += struct.pack(">I", length) + b"IHDR" + ihdr_payload
    out += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_payload) & 0xFFFFFFFF)
    if idat:
        out += _png_chunk(b"IDAT", idat)
    out += _png_chunk(b"IEND", b"")
    return bytes(out)


def _ihdr(width, height, color_type=2):
    return struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)


def _hdr_file(width, height, payload: bytes) -> bytes:
    return (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"
            + f"-Y {height} +X {width}\n".encode("ascii") + payload)


# --------------------------------------------------------------------------- png_min
def test_png_valid_handrolled_decodes(tmp_path):
    px = [[(10, 20, 30), (200, 100, 50), (0, 0, 0), (255, 255, 255)],
          [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)],
          [(90, 80, 70), (60, 50, 40), (30, 20, 10), (0, 128, 255)]]
    p = tmp_path / "ok.png"
    p.write_bytes(_png_bytes(4, 3, px))
    rows = png_min.read_png_rgb(str(p), max_dim=16)
    assert rows is not None and len(rows) == 3 and len(rows[0]) == 4
    assert rows[0][1] == (200, 100, 50) and rows[2][3] == (0, 128, 255)


def test_png_malformed_ihdr_length_returns_none(tmp_path):
    # valid signature, IHDR declared length 12 → struct.error before the fix
    p = tmp_path / "short_ihdr.png"
    p.write_bytes(_png_with_ihdr(b"\x00" * 12, declared_len=12))
    assert png_min.read_png_rgb(str(p)) is None


def test_png_truncated_ihdr_chunk_returns_none(tmp_path):
    # IHDR declares 13 but the file ends early — chunk slice is short
    p = tmp_path / "trunc.png"
    p.write_bytes(_SIG + struct.pack(">I", 13) + b"IHDR" + b"\x00" * 5)
    assert png_min.read_png_rgb(str(p)) is None


def test_png_bomb_declared_dimensions_rejected(tmp_path):
    # 100000×100000: dimensions legally declared, tiny real payload — the classic bomb.
    # Must fail fast (pre-allocation), not try to inflate.
    p = tmp_path / "bomb.png"
    p.write_bytes(_png_with_ihdr(_ihdr(100000, 100000), idat=zlib.compress(b"\x00" * 64)))
    assert png_min.read_png_rgb(str(p)) is None


def test_png_bomb_raw_byte_ceiling_rejected(tmp_path):
    # under the dimension ceiling but expected payload > 256 MB (16384² RGB ≈ 768 MB)
    p = tmp_path / "wide.png"
    p.write_bytes(_png_with_ihdr(_ihdr(16384, 16384), idat=zlib.compress(b"\x00" * 64)))
    assert png_min.read_png_rgb(str(p)) is None


def test_png_payload_larger_than_geometry_rejected(tmp_path):
    # declared 2×2 RGB (expected 14 decoded bytes) but the stream inflates far past it
    p = tmp_path / "overstay.png"
    p.write_bytes(_png_with_ihdr(_ihdr(2, 2), idat=zlib.compress(b"\x00" * 4096)))
    assert png_min.read_png_rgb(str(p)) is None


def test_png_max_dim_zero_no_zero_division(tmp_path):
    p = tmp_path / "ok.png"
    p.write_bytes(_png_bytes(2, 2, [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]]))
    rows = png_min.read_png_rgb(str(p), max_dim=0)
    assert rows is not None and rows[0][0] == (1, 2, 3)


def test_compute_stats_corrupt_png_returns_none_not_raise(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(_png_with_ihdr(b"\x00" * 12, declared_len=12))
    assert metrics.compute_stats(str(p)) is None     # SPEC: loop stats can never fail


# --------------------------------------------------------------------------- hdr_min
def test_float_to_rgbe_huge_finite_saturates():
    px = hdr_min.float_to_rgbe(2e38, 0.0, 0.0)        # exp byte would be 256 pre-fix
    # saturates to the hottest exponent with channel RATIOS preserved — a huge red
    # pixel stays red (the old fix flattened it to white)
    assert px[0] == 255 and px[3] == 255
    assert px[1] == 0 and px[2] == 0
    assert all(0 <= b <= 255 for b in px)
    # ordinary values untouched
    r, g, b, e = hdr_min.float_to_rgbe(1.0, 0.5, 0.25)
    assert r == 128 and 0 < e < 255


def test_write_hdr_huge_values_returns_true_and_roundtrips(tmp_path):
    p = str(tmp_path / "hot.hdr")
    assert hdr_min.write_hdr(p, [[(2e38, 1.0, 0.5)] * 8]) is True   # was: ValueError
    back = hdr_min.read_hdr(p)
    assert back is not None and len(back[0]) == 8
    assert back[0][0][0] > 1e38                    # saturated, finite, readable


def test_write_hdr_flat_path_huge_values_no_raise(tmp_path):
    p = str(tmp_path / "hot_flat.hdr")
    assert hdr_min.write_hdr(p, [[(1e40, 1.0, 1.0)]]) is True       # width 1 → flat


def test_read_hdr_old_style_rle_rejected(tmp_path):
    # width in the 8..32767 band, no 2,2,w>>8,w marker, payload not exactly flat-sized
    # → old-style RLE: must return None, not garbage floats
    bad = tmp_path / "old.hdr"
    bad.write_bytes(_hdr_file(16, 2, b"\x80\xc8" * 40))             # 80 bytes ≠ 128
    assert hdr_min.read_hdr(str(bad)) is None
    junk = tmp_path / "trailing.hdr"
    junk.write_bytes(_hdr_file(16, 2, bytes((128, 128, 128, 129)) * 32 + b"extra"))
    assert hdr_min.read_hdr(str(junk)) is None


def test_read_hdr_foreign_flat_in_band_still_decodes(tmp_path):
    # a legitimately FLAT in-band file (exactly height*width*4 bytes) keeps working
    p = tmp_path / "flat.hdr"
    p.write_bytes(_hdr_file(16, 2, bytes((128, 128, 128, 129)) * 32))
    rows = hdr_min.read_hdr(str(p))
    assert rows is not None and len(rows) == 2 and len(rows[0]) == 16
    assert abs(rows[0][0][0] - 1.0) < 1e-6         # 128 · 2^(129−136) = 1.0


# --------------------------------------------------------------------------- metrics_cli
def test_metrics_cli_corrupt_image_does_not_kill_batch(tmp_path, capsys):
    good = tmp_path / "good.png"
    good.write_bytes(_png_bytes(4, 4, [[(120, 130, 140)] * 4 for _ in range(4)]))
    bad = tmp_path / "bad.png"
    bad.write_bytes(_png_with_ihdr(b"\x00" * 12, declared_len=12))

    rc = metrics_cli.main([str(bad), str(good)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 2
    assert out[0]["path"] == str(bad) and out[0]["stats"] is None
    assert out[1]["path"] == str(good) and isinstance(out[1]["stats"], dict)
    assert out[1]["stats"]["count"] > 0
