"""Vantage window grab — the ranking rule, the framing, and the four refusals.

Pure python, no Windows APIs and no Vantage: ctypes lives behind ``_pids_for_image`` /
``_enum_candidates`` / ``_client_rect`` / ``_occluded_fraction`` / ``_grab_rows``, and the
parts with actual judgement in them (which window, how it is framed, is it covered, is it
fresh, is it black, what gets written) take plain data. That split is the design, so this
is what it buys.

Three of the refusals below are the ones the 2026-07-30 spike hit live: a capture that came
back as 3ds Max's viewport through Vantage's rect with every API reporting success, and a
fully black grab that a whole-frame test scored "29% non-black" because a window's drop
shadow is not black. The fourth — a frame that is fine in every measurable way except that
it predates its own apply — is the one that has no symptom at all, which is why it is here.
"""

import os

import pytest

from maxgaffer.core import metrics, png_min
from maxgaffer.maxbridge import vgrab


# --------------------------------------------------------------------- ranking (pure)
def test_best_window_ranking():
    """(hwnd, pid, title, rect, minimized, cloaked) → the viewport, or nothing."""
    big_untitled = (11, 100, "", (0, 0, 1600, 900), False, False)
    titled = (12, 100, "TULA — Chaos Vantage", (0, 0, 800, 600), False, False)
    minimized = (13, 100, "Chaos Vantage", (0, 0, 2000, 1200), True, False)
    cloaked = (14, 100, "Chaos Vantage", (0, 0, 2000, 1200), False, True)
    tiny = (15, 100, "Chaos Vantage", (0, 0, 20, 12), False, False)

    # a minimised, cloaked or tooltip-sized window still HAS a rect and still returns
    # pixels — stale or empty ones
    assert vgrab._best_window([minimized, cloaked, tiny]) is None
    assert vgrab._best_window([]) is None
    # a title match beats a bigger untitled sibling (Vantage owns several helper windows)
    assert vgrab._best_window([big_untitled, titled])[0] == 12
    # among title matches the viewport is the big one
    bigger = (16, 100, "Chaos Vantage", (0, 0, 1920, 1080), False, False)
    assert vgrab._best_window([titled, bigger])[0] == 16
    # junk in the census is not evidence
    assert vgrab._best_window([("nonsense",), None, titled])[0] == 12


# --------------------------------------------------------------------- capture refusals
def _stub_window(monkeypatch, rows=None, occluded=0.0, client=(0, 0, 640, 360)):
    """Everything ctypes-shaped answers; the grab itself is supplied by the test.

    ``SETTLE_LIMIT_S`` is zeroed because the freshness poll is real wall-clock time: the
    tests that care about it set it back themselves.
    """
    grabbed = []
    vgrab.reset_settle()
    monkeypatch.setattr(vgrab, "SETTLE_LIMIT_S", 0.0)
    monkeypatch.setattr(vgrab, "SETTLE_STEP_S", 0.0)
    monkeypatch.setattr(vgrab, "find_window", lambda *a, **k: 4242)
    monkeypatch.setattr(vgrab, "_window_rect", lambda hwnd: (0, 0, 640, 360))
    monkeypatch.setattr(vgrab, "_client_rect", lambda hwnd: client)
    monkeypatch.setattr(vgrab, "_occluded_fraction", lambda hwnd, rect: occluded)
    monkeypatch.setattr(vgrab, "_grab_rows",
                        lambda rect, w, h: grabbed.append((w, h)) or rows)
    return grabbed


def test_black_grab_is_refused_and_writes_nothing(monkeypatch, tmp_path):
    """A black plate is the one thing a probe must never hand the critic — it ranks it."""
    _stub_window(monkeypatch, rows=[[(0, 0, 0)] * 32 for _ in range(18)])
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) is None
    assert not out.exists()                      # a stale file can never look like a probe
    assert "black" in vgrab.last_error()


def test_occluded_grab_is_refused(monkeypatch, tmp_path):
    """The screen DC reads what is ON SCREEN — anything on top of Vantage lands in the
    probe. Silent wrong pixels are worse than no pixels."""
    grabbed = _stub_window(monkeypatch, rows=[[(9, 9, 9)] * 32 for _ in range(18)],
                           occluded=1.0)
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) is None
    assert not out.exists()
    assert grabbed == []                         # refused BEFORE paying for the capture
    assert "occluded" in vgrab.last_error()


def test_unverifiable_occlusion_is_also_refused(monkeypatch, tmp_path):
    """Unverifiable is not the same as verified: the V-Ray fallback is always correct."""
    _stub_window(monkeypatch, rows=[[(9, 9, 9)] * 32 for _ in range(18)], occluded=None)
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) is None
    assert not out.exists()


def test_missing_window_is_refused(monkeypatch, tmp_path):
    """Driven through the REAL find_window: a running vantage.exe with no usable window."""
    monkeypatch.setattr(vgrab, "_pids_for_image", lambda name=vgrab.VANTAGE_IMAGE: (4242,))
    monkeypatch.setattr(vgrab, "_enum_candidates", lambda pids: [])
    monkeypatch.setattr(vgrab, "_grab_rows",
                        lambda *a: pytest.fail("nothing may be captured without a window"))
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) is None
    assert not out.exists()
    assert "not found" in vgrab.last_error()


def test_vantage_not_running_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(vgrab, "_pids_for_image", lambda name=vgrab.VANTAGE_IMAGE: ())
    assert vgrab.find_window() is None
    assert "not running" in vgrab.last_error()


# --------------------------------------------------------------------- the format contract
def test_good_grab_writes_a_decodable_probe_png(monkeypatch, tmp_path):
    """The probe must land as the exact PNG subset the stats floor decodes inside Max:
    8-bit, non-interlaced, truecolor. png_min writes and reads it; metrics scores it."""
    rows = [[((x * 7) % 256, (y * 13) % 256, 128) for x in range(32)] for y in range(18)]
    grabbed = _stub_window(monkeypatch, rows=rows)
    out = tmp_path / "probe.png"
    path = vgrab.capture_window_png("Vantage", str(out), 32, 18)
    assert path == str(out) and os.path.exists(path)
    assert grabbed == [(32, 18)]                 # GDI did the downscale, not python
    back = png_min.read_png_rgb(path, max_dim=64)
    assert back is not None and len(back) == 18 and len(back[0]) == 32
    stats = metrics.compute_stats(path)
    assert stats is not None and metrics.is_black(stats) is False


# --------------------------------------------------------------------------- framing
def test_the_grab_is_the_client_area_cropped_to_the_probe_aspect(monkeypatch, tmp_path):
    """metrics bins hot pixels into a 5x5 grid over the WHOLE image and hot_frac is a
    fraction of the WHOLE image, so title bar, borders and a squashed aspect do not
    average out: they displace every sun patch into a neighbouring cell and put bright UI
    over the ABSOLUTE HOT_THRESHOLD in fixed border cells of every probe."""
    taken = []
    rows = [[(9, 9, 9)] * 32 for _ in range(18)]
    _stub_window(monkeypatch, rows=rows, client=(100, 132, 3540, 1572))   # 3440x1440
    monkeypatch.setattr(vgrab, "_grab_rows",
                        lambda rect, w, h: taken.append(rect) or rows)
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "p.png"), 32, 18)
    left, top, right, bottom = taken[0]
    assert top == 132 and bottom == 1572                 # nothing trimmed vertically
    assert (right - left) == 2560 and (left + right) // 2 == 1820   # 16:9, centred
    # the window's OUTLINE is never what gets grabbed — the client area is
    assert (left, top) != (0, 0)


def test_client_area_that_cannot_be_read_is_refused(monkeypatch, tmp_path):
    """Approximating the framing with the window outline is how chrome gets scored. The
    V-Ray fallback is always correct."""
    _stub_window(monkeypatch, rows=[[(9, 9, 9)] * 32 for _ in range(18)], client=None)
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) is None
    assert not out.exists()
    assert "client area" in vgrab.last_error()


def test_aspect_crop_is_a_pure_centred_trim():
    assert vgrab._aspect_crop((0, 0, 3440, 1440), 16, 9) == (440, 0, 3000, 1440)
    assert vgrab._aspect_crop((0, 0, 1000, 1000), 16, 9) == (0, 219, 1000, 781)
    assert vgrab._aspect_crop((0, 0, 1920, 1080), 32, 18) == (0, 0, 1920, 1080)
    assert vgrab._aspect_crop((0, 0, 0, 0), 16, 9) == (0, 0, 0, 0)      # junk in, junk out
    assert vgrab._aspect_crop((0, 0, 640, 360), 16, 0) == (0, 0, 640, 360)


# --------------------------------------------------------------------------- freshness
def _plate(level):
    return [[(level, level, level)] * 32 for _ in range(18)]


def test_an_unchanged_window_is_refused_as_the_previous_probe(monkeypatch, tmp_path):
    """THE refusal with no symptom. Vantage is a separate process and the probe path
    suppresses Max's viewport redraw, so a grab taken in the same breath as the apply
    returns the frame for the PREVIOUS azimuth — not black, not occluded, not minimised.
    Every sample shifted one slot is worse than the slow run it replaces."""
    _stub_window(monkeypatch, rows=_plate(120))
    first = tmp_path / "a.png"
    assert vgrab.capture_window_png("Vantage", str(first), 32, 18) == str(first)

    second = tmp_path / "b.png"
    assert vgrab.capture_window_png("Vantage", str(second), 32, 18) is None
    assert not second.exists()
    assert "has not changed" in vgrab.last_error()


def test_accumulation_noise_is_not_a_new_frame(monkeypatch, tmp_path):
    """Vantage keeps refining while it sits idle, so "any difference at all" waves the
    stale frame straight through. The floor is a block mean, not a pixel."""
    _stub_window(monkeypatch, rows=_plate(120))
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18)

    monkeypatch.setattr(vgrab, "_grab_rows", lambda rect, w, h: _plate(121))  # +1 = noise
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "b.png"), 32, 18) is None
    assert "has not changed" in vgrab.last_error()

    monkeypatch.setattr(vgrab, "_grab_rows", lambda rect, w, h: _plate(160))  # sun moved
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "c.png"), 32, 18)


def test_the_poll_is_bounded_and_takes_the_frame_that_moved(monkeypatch, tmp_path):
    """Bounded: a probe may cost 0.6 s, never a stall on Max's main thread. The grab that
    is written is the one that MOVED, not the stale one the poll started from."""
    _stub_window(monkeypatch, rows=_plate(120))
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18)

    monkeypatch.setattr(vgrab, "SETTLE_LIMIT_S", 1.0)
    served = [_plate(120), _plate(120), _plate(200)]
    monkeypatch.setattr(vgrab, "_grab_rows",
                        lambda rect, w, h: served.pop(0) if served else _plate(200))
    out = tmp_path / "b.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) == str(out)
    back = png_min.read_png_rgb(str(out), max_dim=64)
    assert back[9][16] == (200, 200, 200)             # the frame after the change


def test_a_refused_plate_never_becomes_the_baseline(monkeypatch, tmp_path):
    """Proving the next frame differs from one nobody scored proves nothing."""
    _stub_window(monkeypatch, rows=_plate(0))         # black — refused
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18) is None
    assert vgrab._LAST_SIGNATURE is None

    monkeypatch.setattr(vgrab, "_grab_rows", lambda rect, w, h: _plate(120))
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "b.png"), 32, 18)


def test_reset_settle_forgets_the_last_run(monkeypatch, tmp_path):
    """A run that compares against a picture from an hour ago passes its freshness test on
    the strength of the artist having moved the camera in between."""
    _stub_window(monkeypatch, rows=_plate(120))
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18)
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "b.png"), 32, 18) is None
    vgrab.reset_settle()
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "c.png"), 32, 18)


def test_centre_crop_black_test_ignores_the_window_shadow():
    """A whole-frame test passes on the drop shadow — that is how the spike scored a
    fully black capture at 29% non-black."""
    dark = [[(0, 0, 0)] * 20 for _ in range(20)]
    dark[0][0] = (40, 40, 40)                    # shadow/border pixel, not content
    assert vgrab._all_black(dark) is True
    dark[10][10] = (1, 0, 0)                     # one lit pixel in the middle
    assert vgrab._all_black(dark) is False
    assert vgrab._all_black([]) is True
