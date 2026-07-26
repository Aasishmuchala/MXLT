"""Shadow-edge sharpness and haze density, measured off synthetic frames with known answers.

Both quantities are currently GUESSED by a vision model — `light_quality` and `atmosphere` —
and on the measured cross-domain run the hardness reading collapsed to 0.0 agreement while
driving `sun.size` through transfer.HARDNESS_SIZE. These tests build images whose answer is
known by construction (a step edge IS hard, a 24-pixel ramp IS soft, a washed-out frame IS
hazier than a punchy one) and assert the ORDERING between them, never a magic number: the
thresholds are calibration and may move, but a step must always out-score a ramp or the
measurement is not a measurement.

Images are written with `png_min.write_png_rgb` — the same zero-dependency codec that has to
work inside Max — so the whole path from bytes to answer is exercised.
"""
import math
import os
import random

import pytest

from maxgaffer.core import png_min, shadowedge

#: Big enough that EDGE_MARGIN_PX (18) still leaves an interior to sample, and square so a
#: vertical boundary and a horizontal one are the same test.
SIZE = 160

#: The two levels every synthetic boundary runs between: a lit surface and the same surface
#: in shade, about 3:1 in display-referred sRGB. Comfortably clear of MIN_STEP_RATIO without
#: being a black-to-white cartoon.
SHADE, LIT = 70, 200


def write(tmp_path, name, painter, size=SIZE):
    """Paint an image with ``painter(x, y) -> (r, g, b)`` and write it as a real PNG."""
    rows = [[painter(x, y) for x in range(size)] for y in range(size)]
    path = str(tmp_path / name)
    assert png_min.write_png_rgb(path, rows) == path
    return path


def gray(value):
    v = max(0, min(255, int(round(value))))
    return (v, v, v)


def ramp_painter(width_px, centre=None, size=SIZE):
    """A lit/shadow boundary that transitions over exactly ``width_px`` pixels.

    ``width_px`` 0 is a step edge — a point-like sun. Larger is a broader source. This is the
    one knob the hardness estimator claims to read, so every hardness test is a comparison
    between two of these."""
    centre = size // 2 if centre is None else centre

    def paint(x, _y):
        if width_px <= 0:
            return gray(LIT if x >= centre else SHADE)
        t = (x - centre + width_px / 2.0) / float(width_px)
        return gray(SHADE + max(0.0, min(1.0, t)) * (LIT - SHADE))

    return paint


# ------------------------------------------------------------------ hardness: the ordering
def test_a_step_edge_reads_harder_than_a_gradient_edge(tmp_path):
    """THE load-bearing assertion. Same two levels, same everything, only the width of the
    transition differs — and that difference is the whole physical content of "hard light".
    A point-like sun puts the boundary inside a pixel; a broad source smears it over tens."""
    step = shadowedge.edge_hardness(write(tmp_path, "step.png", ramp_painter(0)))
    gradient = shadowedge.edge_hardness(write(tmp_path, "grad.png", ramp_painter(24)))
    assert step is not None and gradient is not None
    assert step["hardness"] > gradient["hardness"]
    assert step["label"] == "hard"
    assert gradient["label"] == "soft"
    # and the separation is not a rounding artefact — these are opposite ends of the scale
    assert step["hardness"] - gradient["hardness"] > 0.5


def test_hardness_falls_monotonically_as_the_source_widens(tmp_path):
    """Not just two points: the estimator has to be monotone across the range, or "mixed"
    is an arbitrary bucket rather than a source of intermediate size."""
    widths = (0, 3, 6, 10, 20)
    readings = [shadowedge.edge_hardness(
        write(tmp_path, "w%d.png" % w, ramp_painter(w))) for w in widths]
    assert all(r is not None for r in readings)
    values = [r["hardness"] for r in readings]
    assert values == sorted(values, reverse=True), values
    assert values[0] > values[-1]


def test_the_measured_width_is_the_width_that_was_painted(tmp_path):
    """The intermediate step everything else rests on: width = level difference / peak
    slope. If that arithmetic is right the reported width_px tracks the painted ramp, and
    the octave map on top of it is just labelling."""
    for painted in (6, 10):
        got = shadowedge.edge_hardness(
            write(tmp_path, "m%d.png" % painted, ramp_painter(painted)))
        assert got is not None
        assert abs(got["width_px"] - painted) <= 2.0, (painted, got)
    # a step cannot be reported narrower than the operator's own footprint
    step = shadowedge.edge_hardness(write(tmp_path, "s.png", ramp_painter(0)))
    assert step["width_px"] == pytest.approx(shadowedge.EDGE_OPERATOR_FLOOR_PX, abs=0.01)


def test_a_boundary_wider_than_the_probe_baseline_saturates_instead_of_lying(tmp_path):
    """The near probes span 16 px, so a wider transition cannot be fully spanned. Its
    measured level difference shrinks in exact proportion to its slope, so the estimate
    SATURATES at "maximally soft" rather than folding back and reporting a hard edge."""
    wide = shadowedge.edge_hardness(write(tmp_path, "wide.png", ramp_painter(24)))
    wider = shadowedge.edge_hardness(write(tmp_path, "wider.png", ramp_painter(40)))
    assert wide is not None and wider is not None
    assert wide["label"] == wider["label"] == "soft"
    assert wider["hardness"] <= wide["hardness"] + 0.05      # never climbs back up
    assert wider["width_px"] <= shadowedge.WIDTH_SOFT_PX


def test_orientation_does_not_change_the_answer(tmp_path):
    """A ramp's gradient magnitude is exact at any orientation, so a boundary running
    horizontally, vertically or diagonally is the same measurement. If it is not, the probe
    offsets are being taken along an axis instead of along the normal."""
    vertical = shadowedge.edge_hardness(write(tmp_path, "v.png", ramp_painter(8)))

    def horizontal(x, y):
        return ramp_painter(8)(y, x)

    def diagonal(x, y):
        t = ((x + y) / 2.0 - SIZE // 2 + 4) / 8.0
        return gray(SHADE + max(0.0, min(1.0, t)) * (LIT - SHADE))

    for name, painter in (("h.png", horizontal), ("d.png", diagonal)):
        got = shadowedge.edge_hardness(write(tmp_path, name, painter))
        assert got is not None, name
        assert abs(got["hardness"] - vertical["hardness"]) < 0.2, (name, got, vertical)


# ------------------------------------------------------ hardness: telling a boundary apart
def test_pure_texture_is_not_a_shadow_boundary(tmp_path):
    """Noise has gradients everywhere and level differences nowhere. The sustain test — the
    difference across +-8 px must survive across +-16 px — is what removes it, and removing
    it entirely (None, not a number) is the honest answer: this frame contains no shadow
    boundary, so it has no hardness."""
    rnd = random.Random(7)

    def noise(_x, _y):
        return gray(135 + rnd.randint(-60, 60))

    assert shadowedge.edge_hardness(write(tmp_path, "noise.png", noise)) is None


def test_texture_riding_ON_a_shadow_boundary_does_not_stop_it_being_measured(tmp_path):
    """The complement, and the case that matters in a real photograph: a boundary on a
    textured surface (carpet, render, grass) must still be found. A test that only rejects
    texture could pass by rejecting everything."""
    rnd = random.Random(11)
    base = ramp_painter(0)

    def gritty(x, y):
        r, g, b = base(x, y)
        n = rnd.randint(-18, 18)
        return gray(r + n)

    got = shadowedge.edge_hardness(write(tmp_path, "gritty.png", gritty))
    assert got is not None
    assert got["label"] == "hard"


def test_a_material_edge_is_not_a_shadow_edge(tmp_path):
    """A shadow changes the illumination over ONE albedo, so chromaticity survives it. A
    paint line changes the albedo itself. Both are sustained steps with a big luminance
    ratio, so only the colour tells them apart — and getting this wrong means every
    high-contrast object edge in the frame votes on how hard the sun is."""
    def two_materials(x, _y):
        return (40, 190, 60) if x >= SIZE // 2 else (190, 60, 50)   # green | red, same-ish L

    assert shadowedge.edge_hardness(write(tmp_path, "mat.png", two_materials)) is None


def test_a_shadow_that_shifts_BLUE_is_still_a_shadow(tmp_path):
    """The other side of the material test, and the reason its tolerance is loose. A real
    shadow is lit by sky and the sun-lit region by a warm sun, so the chromaticity DOES move
    across a genuine boundary — a tolerance tight enough to reject material edges cleanly
    would reject the golden-hour reference this module was built for."""
    def warm_lit_cool_shade(x, _y):
        return (215, 195, 150) if x >= SIZE // 2 else (58, 68, 92)

    got = shadowedge.edge_hardness(write(tmp_path, "blue.png", warm_lit_cool_shade))
    assert got is not None, "a sun/sky shadow edge must survive the material test"
    assert got["label"] == "hard"


def test_a_flat_frame_has_no_boundaries_to_report(tmp_path):
    """Absence is not softness. A frame with nothing in it must decline rather than return
    a middling number that transfer.score would then happily grade."""
    assert shadowedge.edge_hardness(write(tmp_path, "flat.png", lambda x, y: gray(128))) is None
    assert shadowedge.edge_hardness(write(tmp_path, "black.png", lambda x, y: gray(0))) is None


def test_a_frame_carrying_both_populations_reads_mixed(tmp_path):
    """A shaft of sun crossing a softly-filled room genuinely has hard edges AND soft ones,
    and its median is an average of two things neither of which is present in the middle.
    When both populations are really there the label says so."""
    half = SIZE // 2
    hard_side = ramp_painter(0, centre=SIZE // 4)
    soft_side = ramp_painter(20, centre=3 * SIZE // 4)

    def both(x, y):
        return hard_side(x, y) if y < half else soft_side(x, y)

    got = shadowedge.edge_hardness(write(tmp_path, "both.png", both))
    assert got is not None
    assert got["label"] == "mixed"
    assert got["hard_frac"] >= shadowedge.MIXED_MINORITY
    assert got["soft_frac"] >= shadowedge.MIXED_MINORITY


def test_hardness_is_not_fooled_by_exposure(tmp_path):
    """Width is a ratio of a level difference to a slope, so scaling the whole frame cancels.
    It has to: the reference and the render are never at the same exposure, and a hardness
    that drifted with brightness would be re-measuring what expose.py already handles."""
    bright = shadowedge.edge_hardness(write(tmp_path, "bright.png", ramp_painter(8)))

    def dim(x, y):
        r, g, b = ramp_painter(8)(x, y)
        return gray(r * 0.55)

    got = shadowedge.edge_hardness(write(tmp_path, "dim.png", dim))
    assert got is not None and bright is not None
    assert abs(got["hardness"] - bright["hardness"]) < 0.15


def test_confidence_reports_how_much_evidence_there_was(tmp_path):
    """A reading taken from a handful of samples, or one sitting on a label boundary, is a
    real result that must be held loosely — the posture sunsolve takes with a flat table."""
    clear = shadowedge.edge_hardness(write(tmp_path, "clear.png", ramp_painter(0)))
    assert 0.0 < clear["confidence"] <= 1.0
    assert clear["samples"] >= shadowedge.MIN_SAMPLES
    borderline = shadowedge.edge_hardness(write(tmp_path, "border.png", ramp_painter(5)))
    if borderline is not None and abs(borderline["hardness"] - shadowedge.HARD_LABEL) < 0.06:
        assert borderline["confidence"] < clear["confidence"]


# ------------------------------------------------------------------------------- haze
def checker(lift, gain, size=SIZE):
    """A high-frequency checkerboard scene, optionally veiled: ``gain`` compresses its
    contrast and ``lift`` raises its floor, which is exactly what airlight does to a frame
    (I = J*t + A*(1-t), so gain is the transmission and lift is A*(1-t))."""
    def paint(x, y):
        v = 10 if (x // 8 + y // 8) % 2 == 0 else 240
        return gray(lift + (v - 125) * gain)

    return paint


def test_a_washed_out_frame_reads_hazier_than_a_punchy_one(tmp_path):
    """The dark-channel prior, stated as an ordering: haze raises the darkest value findable
    in a local window, so the veiled frame must score higher than the clear one. Identical
    scene content in both — only the veil differs."""
    punchy = shadowedge.haze_estimate(write(tmp_path, "punchy.png", checker(125, 1.0)))
    washed = shadowedge.haze_estimate(write(tmp_path, "washed.png", checker(150, 0.18)))
    assert punchy is not None and washed is not None
    assert washed["dark_channel"] > punchy["dark_channel"]
    assert punchy["label"] == "none"
    assert washed["label"] in ("light_haze", "heavy_haze", "fog")


def test_the_dark_channel_rises_monotonically_with_the_veil(tmp_path):
    """Three densities of the same veil must come back in order, or the number is not
    measuring density — and it is being handed to atmosphere.distance_m, which is
    logarithmic in exactly this quantity."""
    veils = ((125, 1.0), (140, 0.45), (165, 0.22), (200, 0.06))
    values = []
    for i, (lift, gain) in enumerate(veils):
        got = shadowedge.haze_estimate(write(tmp_path, "veil%d.png" % i, checker(lift, gain)))
        assert got is not None
        values.append(got["dark_channel"])
    assert values == sorted(values), values
    assert values[-1] - values[0] > 0.4          # the range is used, not a flat line


def test_the_labels_walk_the_transfer_vocabulary_in_order(tmp_path):
    """The four labels have to be reachable and correctly ordered — they index straight into
    transfer.ATMOSPHERE_DISTANCE_M, where getting the order wrong swaps 4000 m for 20 m."""
    seen = []
    for i, (lift, gain) in enumerate(((125, 1.0), (150, 0.30), (185, 0.14), (215, 0.04))):
        got = shadowedge.haze_estimate(write(tmp_path, "lab%d.png" % i, checker(lift, gain)))
        seen.append(got["label"])
    order = ["none", "light_haze", "heavy_haze", "fog"]
    assert [order.index(s) for s in seen] == sorted([order.index(s) for s in seen]), seen
    assert seen[0] == "none" and seen[-1] == "fog"


def test_haze_also_costs_local_contrast(tmp_path):
    """The second half of what haze does, and the cross-check that keeps the first half
    honest. Reported so a caller can see both, because a lifted dark channel WITHOUT lost
    contrast is a white wall rather than thick air."""
    punchy = shadowedge.haze_estimate(write(tmp_path, "p2.png", checker(125, 1.0)))
    washed = shadowedge.haze_estimate(write(tmp_path, "w2.png", checker(150, 0.18)))
    assert washed["local_contrast"] < punchy["local_contrast"]


def test_a_bright_frame_with_its_contrast_INTACT_is_held_loosely(tmp_path):
    """The dark-channel prior's classic false positive: a white studio, a snowfield, a
    high-key product shot. It has no dark channel and no haze either. This cannot be fixed,
    only detected — so the reading still comes out, with its confidence cut, rather than
    being silently wrong at full confidence."""
    def bright_but_crisp(x, y):
        return gray(215 if (x // 6 + y // 6) % 2 == 0 else 165)   # high floor, full detail

    got = shadowedge.haze_estimate(write(tmp_path, "studio.png", bright_but_crisp))
    assert got is not None
    assert got["dark_channel"] > shadowedge.DARK_NONE       # the prior does fire
    assert got["local_contrast"] > shadowedge.LOW_CONTRAST  # but the detail is all there
    assert got["confidence"] <= shadowedge.BRIGHT_ALBEDO_PENALTY


def test_clear_floor_reports_that_somewhere_in_frame_the_air_is_clear(tmp_path):
    """Depth-varying haze looks different from a uniform veil: a dark foreground object puts
    the 10th percentile on the floor while the median stays lifted. Reported beside the
    headline so a caller can tell those two apart."""
    def far_haze_near_clear(x, y):
        if y > SIZE * 3 // 4:                       # near field: untouched
            return gray(10 if (x // 8) % 2 == 0 else 240)
        return gray(190 + ((10 if (x // 8) % 2 == 0 else 240) - 125) * 0.08)

    got = shadowedge.haze_estimate(write(tmp_path, "depth.png", far_haze_near_clear))
    assert got is not None
    assert got["clear_floor"] < got["dark_channel"]


def test_the_window_minimum_is_a_true_minimum(tmp_path):
    """The van Herk running minimum is the one piece of real algorithm in the haze path —
    it is what keeps a 9x9 min-filter O(n) instead of O(n*81). Checked against the naive
    definition, including the truncated windows at both borders where the block padding is
    easiest to get wrong."""
    rnd = random.Random(3)
    for length in (1, 2, 5, 9, 17, 40, 41):
        line = [rnd.random() for _ in range(length)]
        for radius in (0, 1, 4, 7):
            got = shadowedge._running_min(line, radius)
            want = [min(line[max(0, i - radius):i + radius + 1]) for i in range(length)]
            assert got == want, (length, radius)


# ---------------------------------------------------------------------------- contract
def test_pre_loaded_pixels_and_a_path_are_the_same_measurement(tmp_path):
    """compute_stats has already paid for the decode on every polish probe. Accepting its
    (pixels, w, h) triple is what keeps these two numbers cheap enough to take every time —
    so the two entry paths must not disagree."""
    from maxgaffer.core import metrics

    path = write(tmp_path, "dual.png", ramp_painter(6))
    loaded = metrics._load_pixels(path, max_dim=shadowedge.WORK_MAX_DIM)
    assert loaded is not None
    assert shadowedge.edge_hardness(loaded) == shadowedge.edge_hardness(path)
    assert shadowedge.haze_estimate(loaded) == shadowedge.haze_estimate(path)


def test_both_estimators_degrade_on_junk_rather_than_raising(tmp_path):
    """Same contract metrics, transfer and patchgeom hold to: these run beside the loop's
    own stats inside Max, and an exception there takes down a match that is otherwise fine."""
    for bad in (None, "", 42, [], {}, ("only", "two"), (None, 4, 4), ([], 0, 0),
                str(tmp_path / "missing.png"), (["not a pixel"], 8, 8)):
        assert shadowedge.edge_hardness(bad) is None, bad
        assert shadowedge.haze_estimate(bad) is None, bad
    not_an_image = tmp_path / "fake.png"
    not_an_image.write_bytes(b"certainly not a png")
    assert shadowedge.edge_hardness(str(not_an_image)) is None
    assert shadowedge.haze_estimate(str(not_an_image)) is None


def test_a_frame_too_small_to_hold_a_boundary_declines(tmp_path):
    """The far probe reaches EDGE_MARGIN_PX from the boundary, so a thumbnail has no
    interior left. Declining beats reporting a number measured on clamped windows."""
    tiny = write(tmp_path, "tiny.png", ramp_painter(0, centre=10, size=20), size=20)
    assert shadowedge.edge_hardness(tiny) is None
    assert shadowedge.haze_estimate(tiny) is not None     # haze needs no baseline


def test_labels_stay_inside_the_vocabularies_transfer_scores_against():
    """These strings are not free-form: transfer.score looks them up in HARDNESS_SIZE and
    ATMOSPHERE_DISTANCE_M, and a label outside those dicts is silently NOT SCORED — the
    criterion disappears from the denominator instead of failing loudly."""
    from maxgaffer.core import transfer

    assert set(transfer.HARDNESS_SIZE) == {"hard", "mixed", "soft"}
    assert set(transfer.ATMOSPHERE_DISTANCE_M) == {"none", "light_haze", "heavy_haze", "fog"}
    assert shadowedge.SOFT_LABEL < shadowedge.HARD_LABEL
    assert 0.0 < shadowedge.DARK_NONE < shadowedge.DARK_LIGHT < shadowedge.DARK_HEAVY < 1.0


def test_the_module_stays_pure_stdlib_and_sans_IO():
    """It runs inside Max's bundled Python beside the loop stats, where numpy and Pillow may
    not exist and a stray import would take the whole match down with it."""
    with open(shadowedge.__file__, encoding="utf-8") as fh:
        src = fh.read()
    for banned in ("import numpy", "from numpy", "import PIL", "from PIL",
                   "import pymxs", "from pymxs", "open(", "requests"):
        assert banned not in src, banned
    imports = sorted({line.split()[1].split(".")[0]
                      for line in src.splitlines() if line.startswith("import ")})
    assert imports == ["math"]
    assert math.isfinite(shadowedge.WIDTH_SOFT_PX)


def test_the_cost_is_small_enough_to_take_on_every_probe(tmp_path):
    """compute_stats runs on hundreds of renders per match, so these two ride along with it.
    An O(n*window^2) implementation would be correct and unusable; this pins the budget so a
    later 'small' change to the probe baseline cannot quietly reintroduce it."""
    import time

    path = write(tmp_path, "cost.png", ramp_painter(6), size=256)
    loaded = None
    from maxgaffer.core import metrics

    loaded = metrics._load_pixels(path, max_dim=shadowedge.WORK_MAX_DIM)
    start = time.perf_counter()
    shadowedge.edge_hardness(loaded)
    shadowedge.haze_estimate(loaded)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, "%.3f s for one 256px frame is not a per-probe cost" % elapsed
