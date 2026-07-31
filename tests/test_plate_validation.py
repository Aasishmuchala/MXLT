"""I1 — never score an unvalidated image. Pure, off-Max, PIL-free.

Each test encodes one way the 2026-07-30 TULA run could have known it was looking at
nothing, and the discriminator test (a dark-but-structured night render) is the one that
matters most: a check that cannot tell a moonlit exterior from a dead renderer is a check
that will be turned off.
"""

import zlib

import pytest

from maxgaffer.core import plate, png_min


def _stats(p95, p5, mean=0.1, hot=0.0, key=0.2, grid=None):
    """A stats dict shaped like metrics.compute_stats', with the keys plate.validate reads."""
    return {"mean_rgb": [mean, mean, mean],
            "p": {"5": p5, "50": (p5 + p95) / 2.0, "95": p95},
            "contrast": p95 - p5,
            "hot_frac": hot,
            "log_key": key,
            "grid_log": grid if grid is not None else [0.0] * 9}


BLACK = _stats(0.0, 0.0, mean=0.0, key=1e-6)
NEAR_BLACK = _stats(2.0 / 255.0, 1.5 / 255.0, mean=0.004, key=1e-5)


# ----------------------------------------------------------------- black vs near black
def test_hard_black_is_rejected():
    v = plate.validate(BLACK, want=(240, 135), got=(240, 135))
    assert v.ok is False and v.reason == "black"
    assert "100% BLACK" in v.detail


def test_near_black_flat_is_rejected_with_its_own_words():
    """A renderer that returned almost nothing must not be reported in the same sentence
    as one that returned nothing — the two have different causes and different fixes."""
    v = plate.validate(NEAR_BLACK, want=(240, 135), got=(240, 135))
    assert v.ok is False and v.reason == "near_black"
    assert "100% BLACK" not in v.detail
    assert "flat" in v.detail.lower() or "FLAT" in v.detail


@pytest.mark.parametrize("p95,p5", [
    (6.0 / 255.0, 1.0 / 255.0),      # practicals_dusk: dark, but carrying practicals
    (3.0 / 255.0, 1.0 / 255.0),      # a moonlit exterior with a sky gradient
    (2.0 / 255.0, 0.0),              # right on the level boundary, still 2 steps of span
])
def test_a_dark_but_structured_night_render_is_accepted(p95, p5):
    """THE discriminator. The requirement is that this can never false-positive on a
    legitimately dark render, and level alone cannot do it — both are dark. STRUCTURE
    can: a real render carries practicals, a sky gradient and sampler noise, so its
    5th-95th span clears a quantisation step even when p95 is 6/255. A frame a dead
    renderer handed back is dark AND flat AND has no hot pixel anywhere."""
    assert plate.validate(_stats(p95, p5), want=(240, 135), got=(240, 135)).ok is True


def test_a_dark_flat_frame_with_a_hot_pixel_is_accepted():
    """All three near-black terms are required. One sun glint is a picture."""
    s = _stats(2.0 / 255.0, 1.5 / 255.0, hot=0.001)
    assert plate.validate(s, want=(240, 135), got=(240, 135)).ok is True


def test_a_partial_stats_dict_is_not_evidence_either_way():
    """A hand-made or legacy stats dict must not be read as a black frame."""
    assert plate.validate({"log_key": 0.2}, want=(240, 135), got=(240, 135)).ok is True


def test_no_stats_is_reported_as_unmeasured_not_as_ok():
    v = plate.validate(None, want=(240, 135), got=(240, 135))
    assert v.ok is False and v.reason == "unmeasured"


# ----------------------------------------------------------------- wrong size
def test_wrong_size_is_rejected_and_names_both_sizes():
    v = plate.validate(_stats(0.8, 0.1), want=(240, 135), got=(1920, 1080))
    assert v.ok is False and v.reason == "wrong_size"
    assert "1920" in v.detail and "240" in v.detail


def test_an_unreadable_size_is_could_not_verify_never_a_rejection():
    """read_png_size returns None on anything that is not a PNG it claims to know.
    Fixtures write bytes and Max builds write other formats; neither is a wrong size."""
    assert plate.validate(_stats(0.8, 0.1), want=(240, 135), got=None).ok is True


# ----------------------------------------------------------------- read_png_size
def _png(width, height, extra_chunks=b""):
    import struct

    def chunk(ctype, payload):
        return (struct.pack(">I", len(payload)) + ctype + payload
                + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + extra_chunks
            + chunk(b"IEND", b""))


def test_read_png_size_reads_the_header(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png(240, 135))
    assert png_min.read_png_size(str(p)) == (240, 135)


def test_read_png_size_never_decodes(tmp_path, monkeypatch):
    """It must cost microseconds on every plate of a 190-render match, so it may not
    touch zlib at all. Breaking the decompressor proves it never reached one."""
    p = tmp_path / "a.png"
    p.write_bytes(_png(1920, 1080))

    def boom(*a, **k):
        raise AssertionError("read_png_size must not decode the image")

    monkeypatch.setattr(zlib, "decompressobj", boom)
    monkeypatch.setattr(zlib, "decompress", boom)
    assert png_min.read_png_size(str(p)) == (1920, 1080)


@pytest.mark.parametrize("payload", [
    b"",                                        # empty
    b"not a png at all, just bytes",            # wrong signature
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 4,         # truncated header
])
def test_read_png_size_degrades_to_none(tmp_path, payload):
    p = tmp_path / "b.png"
    p.write_bytes(payload)
    assert png_min.read_png_size(str(p)) is None


def test_read_png_size_on_a_missing_file_is_none(tmp_path):
    assert png_min.read_png_size(str(tmp_path / "nope.png")) is None


# ----------------------------------------------------------------- frozen
LIT_A = _stats(0.8, 0.1, key=0.20, grid=[0.1] * 9)
LIT_B = _stats(0.8, 0.1, key=0.31, grid=[0.2] * 9)


def test_identical_plates_with_an_unchanged_state_are_not_frozen():
    """V-Ray renders are DETERMINISTIC in the state — director.py records that the same
    state re-rendered scores 100.0 against itself. Two identical plates are only
    suspicious when the state moved between them, so this must never fire."""
    sig = plate.signature(LIT_A)
    v = plate.validate(LIT_A, want=(8, 8), got=(8, 8), prev_sig=sig,
                       state_changed=False)
    assert v.ok is True and v.reason == ""


def test_frozen_does_not_fire_on_two_in_a_row():
    """A dusk scene with the sun below the horizon legitimately renders identical frames
    across several coarse azimuths, and a wall of warnings is its own unreadable
    transcript."""
    sig = plate.signature(LIT_A)
    run = 0
    for _ in range(2):
        v = plate.validate(LIT_A, want=(8, 8), got=(8, 8), prev_sig=sig,
                           state_changed=True, frozen_run=run)
        run = v.frozen_run
        assert v.frozen_report is False
    assert run == 2


def test_three_identical_plates_with_a_changed_state_report_once():
    sig = plate.signature(LIT_A)
    run, reports = 0, 0
    for _ in range(5):
        v = plate.validate(LIT_A, want=(8, 8), got=(8, 8), prev_sig=sig,
                           state_changed=True, frozen_run=run,
                           changed_axes=["sun.azimuth_deg"])
        run = v.frozen_run
        reports += int(v.frozen_report)
        if v.frozen_report:
            assert "sun.azimuth_deg" in v.detail
            assert "not reaching this frame" in v.detail
    assert run == 5
    assert reports == 1, "three-in-a-row reports once, not once per probe"


def test_the_sixth_frozen_plate_escalates():
    sig = plate.signature(LIT_A)
    run, escalated = 0, False
    for _ in range(6):
        v = plate.validate(LIT_A, want=(8, 8), got=(8, 8), prev_sig=sig,
                           state_changed=True, frozen_run=run)
        run = v.frozen_run
        escalated = escalated or run == plate.FROZEN_ESCALATE_RUN
    assert escalated


def test_a_real_change_clears_the_frozen_run():
    v = plate.validate(LIT_B, want=(8, 8), got=(8, 8),
                       prev_sig=plate.signature(LIT_A), state_changed=True,
                       frozen_run=2)
    assert v.reason == "" and v.frozen_run == 0


def test_a_frozen_plate_is_reported_but_not_rejected():
    """The picture is a real measurement — it is the SEARCH that is broken. Rejecting it
    would abort a run a dusk scene can legitimately produce."""
    sig = plate.signature(LIT_A)
    for _ in range(3):
        v = plate.validate(LIT_A, want=(8, 8), got=(8, 8), prev_sig=sig,
                           state_changed=True, frozen_run=2)
    assert v.ok is True and v.reason == "frozen"


def test_the_signature_is_drawn_from_stats_the_critic_already_paid_for():
    """Not a hash of the file: two renders of one state can differ in PNG metadata while
    being the same picture, and hashing bytes would then miss every real freeze."""
    assert plate.signature(LIT_A) != plate.signature(LIT_B)
    assert plate.signature(dict(LIT_A)) == plate.signature(LIT_A)
    assert plate.signature(None) == ()
    assert plate.signature({"log_key": "not a number"}) == ()
