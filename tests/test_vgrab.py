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
    # (the old assertion here was `grabbed == [(32, 18)]  # GDI did the downscale`, which
    # asserted that the test's own lambda had been called with the arguments the test
    # passed. It could not fail for any reason related to GDI or StretchBlt. Replaced
    # 2026-07-31 with a claim about the FILE, which can. The stub is now grabbed twice —
    # once for the plate, once for G-5's re-verified occlusion — so a count assertion
    # here would also have been asserting the mock.)
    back = png_min.read_png_rgb(path, max_dim=64)
    assert back is not None and len(back) == 18 and len(back[0]) == 32
    stats = metrics.compute_stats(path)
    assert stats is not None and metrics.is_black(stats) is False


@pytest.mark.parametrize("w,h", [(32, 18), (64, 36), (240, 135)])
def test_the_requested_probe_size_reaches_the_capture_layer(monkeypatch, tmp_path, w, h):
    """Removing the old `grabbed == [(32, 18)]` was right — it asserted the test's own
    lambda arguments — but it left nothing at all pinning that width/height survive
    _aspect_crop and reach _grab_rows. This asks the FILE instead: the stub honours
    whatever size it is handed, so a w/h-forwarding bug shows up as wrong PNG dimensions
    rather than as a mock call record. (2026-07-31)"""
    monkeypatch.setattr(vgrab, "SETTLE_LIMIT_S", 0.0)
    monkeypatch.setattr(vgrab, "SETTLE_STEP_S", 0.0)
    monkeypatch.setattr(vgrab, "find_window", lambda *a, **k: 4242)
    monkeypatch.setattr(vgrab, "_client_rect", lambda hwnd: (0, 0, 1600, 900))
    monkeypatch.setattr(vgrab, "_occluded_fraction", lambda hwnd, rect: 0.0)
    monkeypatch.setattr(vgrab, "_grab_rows", lambda rect, gw, gh: [
        [((x * 7) % 256, (y * 13) % 256, 128) for x in range(gw)] for y in range(gh)])
    vgrab.reset_settle()
    path = vgrab.capture_window_png("Vantage", str(tmp_path / "p.png"), w, h)
    assert path is not None
    assert png_min.read_png_size(path) == (w, h)


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


# ------------------------------------------------------- G-12: the Win32 half, off-box
#
# _stub_window replaces find_window, _window_rect, _client_rect, _occluded_fraction AND
# _grab_rows in one call, and 10 of the 13 tests above use it — so every line where the
# ctypes work lives had ZERO coverage in a 673-line module the commit message reported as
# covered by "3781 passed". These four tests reach the parts that were never touched.


def test_bgrx_is_swizzled_to_rgb_not_left_backwards():
    """The single most classic place in Win32 imaging to get red and blue backwards, and
    it was untested by all thirteen tests above. An R/B swap is not cosmetic here:
    luminance is 0.2126·R + 0.0722·B, so swapping shifts which pixels clear
    metrics.HOT_THRESHOLD and therefore the whole ranking the sun solve depends on."""
    # one 2x1 row: pure RED then pure BLUE, in GDI's BGRX byte order
    raw = bytes([0, 0, 255, 0,      # B=0   G=0   R=255  X → (255, 0, 0)
                 255, 0, 0, 0])     # B=255 G=0   R=0    X → (0, 0, 255)
    assert vgrab._rows_from_bgrx(raw, 2, 1) == [[(255, 0, 0), (0, 0, 255)]]


def test_bgrx_reads_rows_top_down_and_drops_the_alpha_byte():
    """biHeight is set NEGATIVE for a top-down DIB, so row 0 of the buffer is row 0 of the
    picture — no flip anywhere in this function."""
    raw = bytes([10, 20, 30, 99,        # row 0
                 40, 50, 60, 99])       # row 1
    assert vgrab._rows_from_bgrx(raw, 1, 2) == [[(30, 20, 10)], [(60, 50, 40)]]


def test_the_settle_poll_is_actually_bounded(monkeypatch, tmp_path):
    """The old test claimed boundedness in its docstring and MEASURED NOTHING — and
    _stub_window sets SETTLE_STEP_S = 0.0, so it was a busy-spin that would burn CPU
    rather than fail. A fake clock makes the claim checkable: at most
    SETTLE_LIMIT_S / SETTLE_STEP_S grabs.

    The grab stub RAISES past the bound rather than letting the fake clock spin forever:
    detection by pytest timeout is detection by hanging the suite, which is not a failing
    test. (2026-07-31)"""
    _stub_window(monkeypatch, rows=_plate(120))
    monkeypatch.setattr(vgrab, "SETTLE_LIMIT_S", 0.6)
    monkeypatch.setattr(vgrab, "SETTLE_STEP_S", 0.05)
    now = {"t": 0.0}
    monkeypatch.setattr(vgrab.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(vgrab.time, "sleep",
                        lambda s: now.__setitem__("t", now["t"] + s))
    grabs = []
    ceiling = int(0.6 / 0.05) + 2

    def _bounded(rect, w, h):
        grabs.append(1)
        if len(grabs) > ceiling:
            raise AssertionError(f"the settle poll is UNBOUNDED: {len(grabs)} grabs")
        return _plate(120)

    monkeypatch.setattr(vgrab, "_grab_rows", _bounded)
    vgrab.reset_settle()
    vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18)
    assert len(grabs) <= int(0.6 / 0.05) + 2, len(grabs)
    assert now["t"] <= 0.65


def test_render_probe_forwards_should_cancel_to_the_settle_poll(monkeypatch):
    """The predicate existed here from 2026-07-31 and NO production caller passed it, so
    the G-13 fix was inert on box: render_probe had no such parameter and _render_raw had
    nothing to give it. This is the wire, tested at the seam that was missing. The test
    below drives vgrab directly and could never have detected its absence."""
    from maxgaffer.maxbridge import render as rd

    seen = {}
    monkeypatch.setattr(rd.vgrab, "capture_window_png",
                        lambda t, o, w, h, should_cancel=None:
                        seen.update(should_cancel=should_cancel, size=(w, h)) or o)
    pred = lambda: True                                       # noqa: E731
    assert rd.render_probe(None, "g.png", 240, 135, backend="vantage",
                           should_cancel=pred) == ("g.png", "vantage")
    assert seen["should_cancel"] is pred
    assert seen["size"] == (240, 135), "the requested probe size must reach the grab"


def test_the_controller_hands_its_own_cancel_predicate_to_every_grab(tmp_path,
                                                                    monkeypatch):
    """…and the predicate it hands over has to be live: pressing ✕ must make it True."""
    import honesty_harness as H
    from maxgaffer.maxbridge import controller as ctl

    c = H.build(tmp_path, monkeypatch, probe_backend="vantage")
    c._probe_backend = "vantage"
    pressed = {"v": False}
    c._begin_operation(lambda: pressed["v"])
    seen = {}
    monkeypatch.setattr(ctl.rd, "render_probe",
                        lambda cam, out, w, h, backend="vray", log=None, fallback=True,
                        should_cancel=None:
                        seen.update(pred=should_cancel) or (out, "vantage"))
    monkeypatch.setattr(c, "stats_for", lambda p: dict(H.LIT))
    c._render_exposed(object(), str(tmp_path / "p.png"), 8, 8, probe=True)
    assert callable(seen["pred"])
    assert seen["pred"]() is False
    pressed["v"] = True
    assert seen["pred"]() is True


def test_the_settle_poll_consults_should_cancel_every_step(monkeypatch, tmp_path):
    """G-13: 0.6 s per probe × 44 probes is up to 26 seconds of unresponsive dock, added
    by a commit whose stated motivation was an artist unable to cancel."""
    _stub_window(monkeypatch, rows=_plate(120))
    monkeypatch.setattr(vgrab, "SETTLE_LIMIT_S", 0.6)   # _stub_window zeroes it
    monkeypatch.setattr(vgrab, "SETTLE_STEP_S", 0.0)
    monkeypatch.setattr(vgrab, "_grab_rows", lambda rect, w, h: _plate(120))
    vgrab.reset_settle()
    asked = []
    out = vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18,
                                   should_cancel=lambda: asked.append(1) or True)
    assert out is None
    assert asked, "the poll must look at the cancel flag"
    assert "cancelled" in vgrab.last_error()


def test_occlusion_is_re_verified_on_the_frame_that_is_returned(monkeypatch, tmp_path):
    """G-5: occlusion was measured, then _settled_rows re-grabbed up to twelve more times
    unchecked and returned one of THOSE. A toast during the 0.6 s poll moves the picture
    by far more than SETTLE_DELTA, so the freshness guard actively CERTIFIED the popup's
    pixels as 'the live link delivered'."""
    _stub_window(monkeypatch, rows=_plate(120))
    covered = iter([0.0, 0.9])          # clear when asked; a dialog by the time it matters
    monkeypatch.setattr(vgrab, "_occluded_fraction",
                        lambda hwnd, rect: next(covered, 0.9))
    vgrab.reset_settle()
    assert vgrab.capture_window_png("Vantage", str(tmp_path / "a.png"), 32, 18) is None
    assert "appeared over the Vantage window" in vgrab.last_error()


def test_the_module_never_configures_the_process_shared_windll():
    """G-11. ctypes.windll.user32 is ONE object for the whole Max process and ctypes
    caches argtypes on it, so installing this module's private _Point class as
    WindowFromPoint.argtypes made every OTHER caller in Max fail with
    'expected _Point instance instead of POINT' — permanently, until restart. The blast
    radius was other people's code."""
    import os as _os

    src = open(_os.path.join(_os.path.dirname(vgrab.__file__), "vgrab.py"),
               encoding="utf-8").read()
    code = src.split('"""', 2)[-1]           # drop the module docstring, which names it
    code = "\n".join(ln for ln in code.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "ctypes.windll" not in code
    assert 'ctypes.WinDLL("user32"' in code


# --------------------------------------------------------------------- tone correction
#
# The armed core.vtone transfer (2026-07-31). The contract under test: unarmed behaviour
# is byte-identical to before the feature existed; an armed correction lands on disk; a
# FAILING armed correction refuses the grab rather than writing uncorrected pixels —
# refuse over lying, because the consumer that armed it scores absolute values.
def _identity_plus_20():
    """A real fitted VTone that adds 20 to every channel — visible in written bytes."""
    from maxgaffer.core.vtone import VTone

    pairs = [(float(v), min(255.0, float(v) + 20.0)) for v in range(0, 256, 3)]
    tone = VTone.fit({"r": pairs, "g": pairs, "b": pairs})
    assert tone is not None
    return tone


def test_unarmed_grab_is_byte_identical_to_the_pre_feature_path(monkeypatch, tmp_path):
    """With no tone armed, the plate on disk is exactly png_min.write_png_rgb(rows) —
    the strictly-additive guarantee every existing consumer (the ordinal sun solve)
    relies on."""
    rows = _plate(120)
    _stub_window(monkeypatch, rows=rows)
    assert vgrab.armed_tone() is None
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) == str(out)
    ref = tmp_path / "ref.png"
    png_min.write_png_rgb(str(ref), rows)
    assert out.read_bytes() == ref.read_bytes()


def test_armed_tone_correction_lands_on_disk(monkeypatch, tmp_path):
    _stub_window(monkeypatch, rows=_plate(120))
    monkeypatch.setattr(vgrab, "_TONE", _identity_plus_20())
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) == str(out)
    back = png_min.read_png_rgb(str(out), max_dim=64)
    # every 120 in the source plate must have landed as 140
    assert back[2][2] == (140, 140, 140)


def test_failing_tone_correction_refuses_and_writes_nothing(monkeypatch, tmp_path):
    """An uncorrected plate the caller believes corrected is the silent-wrong-pixels
    failure — the refusal must fire BEFORE the write, leave no file, and name itself."""

    class _Broken:
        def to_vray_space(self, rows):
            raise RuntimeError("boom")

    _stub_window(monkeypatch, rows=_plate(120))
    monkeypatch.setattr(vgrab, "_TONE", _Broken())
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18) is None
    assert not out.exists()
    assert "tone correction failed" in vgrab.last_error()
    assert "uncorrected" in vgrab.last_error()


def test_arm_disarm_round_trip():
    tone = _identity_plus_20()
    try:
        vgrab.arm_tone(tone)
        assert vgrab.armed_tone() is tone
        vgrab.disarm_tone()
        assert vgrab.armed_tone() is None
        vgrab.arm_tone(tone)
        vgrab.arm_tone(None)                 # explicit disarm via arm_tone(None)
        assert vgrab.armed_tone() is None
    finally:
        vgrab.disarm_tone()                  # never leak state into another test


# --------------------------------------------------------------------- convergence mode
def test_converged_grab_accepts_a_stationary_picture(monkeypatch, tmp_path):
    """converged=True: after first motion the poll continues until two consecutive
    signatures agree. A static stub is stationary on its first comparison."""
    _stub_window(monkeypatch, rows=_plate(120))
    monkeypatch.setattr(vgrab, "CONVERGE_LIMIT_S", 1.0)
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18,
                                    converged=True) == str(out)


def test_converged_grab_refuses_a_picture_that_never_stops_moving(monkeypatch, tmp_path):
    """A frame still accumulating at the deadline has an uncontrolled timestamp in its
    value — not a measurement. The refusal must say so."""
    _stub_window(monkeypatch, rows=_plate(60))
    state = {"n": 0}

    def _restless(rect, w, h):
        state["n"] += 1
        return _plate((60 + state["n"] * 40) % 250)      # 40 levels per poll — never still

    monkeypatch.setattr(vgrab, "_grab_rows", _restless)
    monkeypatch.setattr(vgrab, "CONVERGE_LIMIT_S", 0.06)
    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18,
                                    converged=True) is None
    assert not out.exists()
    assert "still accumulating" in vgrab.last_error()


def test_converged_poll_honours_cancel(monkeypatch, tmp_path):
    _stub_window(monkeypatch, rows=_plate(60))
    state = {"n": 0}

    def _restless(rect, w, h):
        state["n"] += 1
        return _plate((60 + state["n"] * 40) % 250)

    monkeypatch.setattr(vgrab, "_grab_rows", _restless)
    monkeypatch.setattr(vgrab, "CONVERGE_LIMIT_S", 5.0)
    cancelled = {"after": 2}

    def _cancel():
        cancelled["after"] -= 1
        return cancelled["after"] <= 0

    out = tmp_path / "probe.png"
    assert vgrab.capture_window_png("Vantage", str(out), 32, 18,
                                    should_cancel=_cancel, converged=True) is None
    assert "cancelled" in vgrab.last_error()
