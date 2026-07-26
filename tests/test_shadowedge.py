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


def test_confidence_describes_the_MEASUREMENT_not_the_label(tmp_path):
    """Landing near a label boundary must not collapse the confidence.

    It used to, and the ladder caught it: the sun.size 8 rung — a correct mid reading, dead
    centre of a strictly monotonic series — was reported at confidence 0.047 purely because
    its median sat 0.006 from a threshold. That term measured the ambiguity of the CATEGORY
    while the number it was attached to was perfectly well determined, and the number is what
    drives sun.size. Label ambiguity is now reported separately as `label_margin`."""
    clear = shadowedge.edge_hardness(write(tmp_path, "clear.png", sun_penumbra(0)))
    assert clear is not None
    assert 0.0 < clear["confidence"] <= 1.0
    assert clear["samples"] >= shadowedge.MIN_SAMPLES

    # a rung deliberately parked on a band edge: same scene, same sun, only the width differs
    on_edge = None
    for w in range(4, 14):
        got = shadowedge.edge_hardness(write(tmp_path, "edge%d.png" % w, sun_penumbra(w)))
        if got is not None and got["label_margin"] < 0.06:
            on_edge = got
            break
    if on_edge is not None:
        assert on_edge["confidence"] > 0.25, on_edge
        assert on_edge["confidence"] > 0.5 * clear["confidence"], (on_edge, clear)


# ------------------------------------------------------- the shared shadow ratio
#: A physical bi-illuminant scene, in LINEAR light: one warm sun, one blue sky, and albedos
#: to paint it onto. These are the quantities the invariant is written in, so the synthetics
#: are built here and encoded to sRGB on the way out rather than being picked as codes.
SUN = (0.55, 0.36, 0.16)            # warm key, low sun
SKY = (0.06, 0.075, 0.11)           # blue fill
GREY = (0.55, 0.50, 0.45)           # near-neutral albedo
BRICK = (0.42, 0.16, 0.11)          # a strongly different albedo
MOSS = (0.16, 0.34, 0.14)           # and another


def srgb8(linear):
    """Linear RGB → 8-bit sRGB, so a test can state its scene in light and paint in codes."""
    out = []
    for v in linear:
        v = max(0.0, min(1.0, v))
        s = v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1.0 / 2.4) - 0.055
        out.append(max(0, min(255, int(round(s * 255.0)))))
    return tuple(out)


def surface(albedo, ambient_scale=1.0, sunlit=False):
    """What one albedo looks like under this scene's light, in or out of the sun.

    ``ambient_scale`` is ambient occlusion: how much of the sky this point can see. It
    scales the SKY term in both the lit and the shadowed case, which is the whole reason
    the invariant has to be taken on ``ratio - 1`` rather than on the ratio."""
    return srgb8(tuple(albedo[i] * (SKY[i] * ambient_scale + (SUN[i] if sunlit else 0.0))
                       for i in range(3)))


def angle_deg(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (nu * nv)))))


def sun_penumbra(width_px, centre=SIZE // 2):
    """A physically-built lit/shadow boundary: one albedo, a warm sun, a blue sky, and a
    penumbra ``width_px`` wide. The ramp is linear in LIGHT, not in code values, because a
    penumbra is the fraction of the source the surface can see — that fraction varies
    linearly across it, and gamma-encoding is applied afterwards as a camera would."""
    def paint(x, _y):
        if width_px <= 0:
            visible = 1.0 if x >= centre else 0.0
        else:
            visible = max(0.0, min(1.0, (x - centre + width_px / 2.0) / float(width_px)))
        return srgb8(tuple(GREY[i] * (SKY[i] + visible * SUN[i]) for i in range(3)))

    return paint


def test_measured_hardness_DECREASES_as_the_source_widens(tmp_path):
    """THE regression test for the property that makes this drivable at all.

    A ladder of known penumbra widths, rendered here rather than read from disk so the test
    owns its own ground truth. What matters is not any absolute value — an absolute label is
    worth little — but that the response is correctly ORDERED, because that is what lets a
    measured hardness move sun.size in the right direction.

    This mirrors a real ladder rendered in 3ds Max at V-Ray sun sizes 1, 3, 8, 16 and 30,
    which measured 0.984 / 0.776 / 0.356 / 0.152 / 0.148 — strictly monotonic over a spread
    of 0.836 (2026-07-26). Losing monotonicity here would mean losing the only property this
    estimator has actually earned.

    The rungs stay inside the band the estimator can actually resolve: below
    EDGE_OPERATOR_FLOOR_PX everything reads 1.0 by construction, and beyond twice
    PLATEAU_NEAR_PX the near probes sit inside the ramp so the estimate saturates — measured
    on this synthetic, widths of 16, 18 and 22 px plateau together around 0.43 and do not
    separate. A single perfect ramp is also noisier than a rendered frame, because the 8-bit
    staircase is the only structure in it; the real ladder averages hundreds of boundaries
    over real content and came out cleanly monotonic."""
    widths = (2, 3, 5, 9, 15)                     # all within the resolvable band
    readings = []
    for w in widths:
        got = shadowedge.edge_hardness(
            write(tmp_path, "rung%02d.png" % w, sun_penumbra(w)))
        assert got is not None, "rung %d produced no reading" % w
        readings.append(got)
    values = [r["hardness"] for r in readings]
    for i in range(1, len(values)):
        assert values[i] < values[i - 1], (widths, values)
    assert values[0] - values[-1] > 0.6, values      # the full range is used, not a wobble
    # and the measured width tracks the painted one, which is WHY the ordering holds
    assert readings[-1]["width_px"] > readings[0]["width_px"] * 3
    # both ends of the vocabulary are reachable from a physically-built penumbra
    assert readings[0]["label"] == "hard"
    very_soft = shadowedge.edge_hardness(
        write(tmp_path, "rung_wide.png", sun_penumbra(30)))
    assert very_soft is not None and very_soft["label"] == "soft"
    assert very_soft["hardness"] < values[-1]


def test_the_label_bands_agree_with_the_rendered_ladder():
    """The thresholds are calibrated to generated ground truth: a single scene rendered at
    five known V-Ray sun sizes. Sizes 1 and 3 are hard, 8 is genuinely in between, 16 and 30
    are soft. Those measured hardnesses are pinned here against the bands, so a later change
    to either constant has to still classify the truth series correctly."""
    ladder = ((0.984, "hard"), (0.776, "hard"), (0.356, "mixed"),
              (0.152, "soft"), (0.148, "soft"))
    for value, expected in ladder:
        if value >= shadowedge.HARD_LABEL:
            got = "hard"
        elif value <= shadowedge.SOFT_LABEL:
            got = "soft"
        else:
            got = "mixed"
        assert got == expected, (value, got, expected)
    # each band boundary sits in the GAP between two rungs, not on top of one
    assert 0.356 < shadowedge.HARD_LABEL < 0.776
    assert 0.152 < shadowedge.SOFT_LABEL < 0.356


def test_a_material_edge_is_excluded_from_the_inlier_set(tmp_path):
    """THE property this consensus exists for. One frame has a hard shadow edge; the other
    has the identical shadow edge PLUS a strong material edge beside it. The material edge
    must contribute candidates and no inliers, leaving the measurement untouched — a shadow
    ratio has the albedo cancelled out of it, so a different albedo cannot join the cluster
    however contrasty it is."""
    shade, lit, other = surface(GREY), surface(GREY, sunlit=True), surface(BRICK, sunlit=True)

    def shadow_only(x, _y):
        return shade if x < 50 else lit

    def shadow_plus_material(x, _y):
        if x < 50:
            return shade
        return lit if x < 105 else other

    control = shadowedge.edge_hardness(write(tmp_path, "so.png", shadow_only))
    both = shadowedge.edge_hardness(write(tmp_path, "spm.png", shadow_plus_material))
    assert control is not None and both is not None
    # the material edge WAS seen as a candidate boundary...
    assert both["candidates"] > control["candidates"] * 1.5
    # ...and contributed nothing to the measurement
    assert both["samples"] == control["samples"]
    assert both["inlier_frac"] < 0.65
    assert both["hardness"] == control["hardness"]
    assert both["ratio_rgb"] == control["ratio_rgb"]     # the axis did not even shift


def test_a_soft_material_edge_cannot_drag_the_hardness_down(tmp_path):
    """The consequence of the above, stated as behaviour rather than bookkeeping. A hard
    shadow beside a SOFT material transition must still read hard; if the material samples
    leaked into the median they would pull it toward the middle."""
    shade, lit = surface(GREY), surface(GREY, sunlit=True)
    other = surface(MOSS, sunlit=True)

    def hard_shadow_soft_material(x, _y):
        if x < 50:
            return shade
        if x < 96:
            return lit
        t = max(0.0, min(1.0, (x - 96) / 22.0))          # a 22 px ramp between two albedos
        return tuple(int(round(lit[i] + t * (other[i] - lit[i]))) for i in range(3))

    got = shadowedge.edge_hardness(write(tmp_path, "hsm.png", hard_shadow_soft_material))
    assert got is not None
    assert got["label"] == "hard"
    assert got["soft_frac"] == 0.0


def test_the_shadow_ratio_cancels_the_albedo_EXACTLY():
    """The invariant, asserted as the identity it is rather than as an approximation. The
    same illumination change over three wildly different surfaces must give not merely a
    similar direction but the SAME one, because albedo divides out of (a+s)/a."""
    dirs = []
    for albedo in (GREY, BRICK, MOSS):
        shade = tuple(albedo[i] * SKY[i] for i in range(3))
        lit = tuple(albedo[i] * (SKY[i] + SUN[i]) for i in range(3))
        got = shadowedge._shadow_direction(lit, shade)
        assert got is not None
        dirs.append(got)
    for other in dirs[1:]:
        assert angle_deg(dirs[0], other) < 1e-9         # identical, not close
    expected = [SUN[i] / SKY[i] for i in range(3)]      # s/a, the sun-to-ambient ratio
    assert angle_deg(dirs[0], expected) < 1e-9


def test_a_shadow_crossing_two_surfaces_is_measured_as_ONE_boundary(tmp_path):
    """The same invariant at image level. A shadow band crosses two very different albedos,
    and one albedo boundary crosses the frame. The shadow edges must out-vote the material
    edge and the recovered axis must be the scene's real sun-to-ambient ratio.

    The vote is the point, and it is also the weakness — see
    ``test_a_repeated_material_pair_can_out_vote_the_shadows``."""
    def banded(x, y):
        albedo = GREY if y < SIZE // 2 else BRICK
        return surface(albedo, sunlit=not (45 <= x < 115))

    got = shadowedge.edge_hardness(write(tmp_path, "banded.png", banded))
    assert got is not None
    expected = [SUN[i] / SKY[i] for i in range(3)]
    assert angle_deg(got["ratio_rgb"], expected) < shadowedge.RATIO_INLIER_DEG, got
    assert got["label"] == "hard"
    assert got["ratio_is_warm"] is True


def test_the_sun_cluster_is_chosen_over_a_LARGER_repeated_material_pair(tmp_path):
    """The reason the biggest cluster is not the answer.

    The invariant is SYMMETRIC. A shadow edge's ratio is albedo-independent; a material
    edge's ratio is illumination-independent — the same two surfaces meeting in shadow and
    in sun give the same albedo ratio both times. So a material pair that REPEATS across a
    frame forms a cluster every bit as tight as the shadows do, and here it is deliberately
    built to be the larger one: two horizontal albedo boundaries against a single vertical
    shadow edge. Taking the majority picks the albedo pair — measured, that is exactly what
    happened, and it is the mechanism behind the dawn plate's canopy reading.

    Warmth breaks the tie on physics instead of on size, so the answer must be the sun's
    ratio even though the sun's population is outnumbered."""
    stripes = (GREY, BRICK, MOSS)

    def three_surfaces(x, y):
        return surface(stripes[min(2, y * 3 // SIZE)], sunlit=(x >= SIZE // 2))

    got = shadowedge.edge_hardness(write(tmp_path, "outvoted.png", three_surfaces))
    assert got is not None
    sun_axis = [SUN[i] / SKY[i] for i in range(3)]
    albedo_axis = [GREY[i] / BRICK[i] - 1.0 for i in range(3)]
    assert angle_deg(got["ratio_rgb"], sun_axis) < shadowedge.RATIO_INLIER_DEG, got
    assert angle_deg(got["ratio_rgb"], albedo_axis) > 30.0
    assert got["ratio_is_warm"] is True
    # the sun was NOT the biggest population here, and the reading says so
    assert got["cluster_rank"] > 0
    assert got["inlier_frac"] < 0.5
    assert got["label"] == "hard"


def test_the_ratio_direction_survives_a_change_in_how_deep_the_shade_is(tmp_path):
    """Why the clustered quantity is ``ratio - 1`` and not the ratio.

    Surfaces sit in different amounts of ambient occlusion, so one shadow edge's ambient is
    a fraction of another's. The ratio is then 1 + s/(k*a), a different vector for every k,
    and the ratios of genuine shadow edges do NOT share a direction. Subtracting the one
    leaves s/(k*a), whose direction is s/a whatever k is. This test builds the same shadow
    edge at two depths of shade and asserts the implemented quantity agrees while the raw
    ratio does not — if that second assertion ever fails, the deviation is unnecessary."""
    def edge(scale):
        def paint(x, _y):
            return surface(GREY, ambient_scale=scale, sunlit=(x >= SIZE // 2))
        return paint

    open_sky = shadowedge.edge_hardness(write(tmp_path, "open.png", edge(1.0)))
    occluded = shadowedge.edge_hardness(write(tmp_path, "occl.png", edge(0.35)))
    assert open_sky is not None and occluded is not None
    assert angle_deg(open_sky["ratio_rgb"], occluded["ratio_rgb"]) \
        < shadowedge.RATIO_INLIER_DEG

    # ...and the size of the effect, honestly. `ratio - 1` is EXACTLY invariant at any sun
    # strength; the raw ratio drifts, but only by as much as the "+1" is worth relative to
    # s/a. Under a sun that swamps the ambient the two are nearly the same question — which
    # is why this was worth measuring rather than assuming.
    def drift(sun, scales=(1.0, 0.35)):
        def raw(scale):
            return [(SKY[i] * scale + sun[i]) / (SKY[i] * scale) for i in range(3)]

        def minus_one(scale):
            return [sun[i] / (SKY[i] * scale) for i in range(3)]

        return (angle_deg(raw(scales[0]), raw(scales[1])),
                angle_deg(minus_one(scales[0]), minus_one(scales[1])))

    strong_raw, strong_m1 = drift(SUN)
    weak_raw, weak_m1 = drift((0.05, 0.04, 0.02), (1.0, 0.15))   # filled shadow, deep AO
    assert strong_m1 < 1e-6 and weak_m1 < 1e-6          # exact in BOTH regimes, by identity
    assert strong_raw > 1.0                             # the raw ratio always drifts...
    assert weak_raw > strong_raw * 2.0                  # ...and the weaker the sun, the more
    # measured 2026-07-26: 2.7° under a sun that swamps the ambient, 11.8° for a filled
    # shadow at deep occlusion — most of a cone width, i.e. enough to split one scene's
    # shadows into two clusters and hand the vote to whichever half is bigger
    assert weak_raw > 0.8 * shadowedge.RATIO_INLIER_DEG


def test_pure_albedo_noise_is_the_estimator_s_WEAKEST_case(tmp_path):
    """A CHARACTERISATION test, pinning a known weakness rather than asserting a guarantee.
    Read the numbers before trusting a confident hardness.

    A patchwork of unrelated albedos under one light has plenty of sustained, high-ratio,
    sharp boundaries and not one shadow among them. Measured over twelve seeds (2026-07-26):
    five declined outright, and the other seven found a warm cluster — the worst at
    confidence 0.841, which is HIGHER than the 0.358 the estimator gives the golden-hour
    plate's genuine sun cluster. Random albedo pairs skew warm often enough to form a
    coherent warm cluster by chance, and on the evidence in a single frame that is not
    distinguishable from a real sun population.

    This is the direct cost of removing cluster SHARE from the confidence, which had to go so
    that a correct-but-outnumbered sun cluster (10% of candidates on the golden-hour plate)
    would not be scored down for being small. The two requirements are in tension and this
    implementation cannot satisfy both: share-in-confidence suppresses noise but buries the
    right answer; share-out lets the right answer score but lets noise score too.

    The bound below is a regression pin on measured behaviour, not a claim of safety."""
    worst = 0.0
    declined = 0
    for seed in range(12):
        rnd = random.Random(seed)
        # every cell gets its OWN albedo, so no pair of surfaces ever meets twice. A
        # repeating palette would defeat the point: the same two colours meeting again and
        # again is a repeated material pair, which does form a real cluster (see above).
        cells = {}
        for cy in range(SIZE // 20 + 1):
            for cx in range(SIZE // 20 + 1):
                cells[(cx, cy)] = srgb8((rnd.uniform(0.05, 0.75), rnd.uniform(0.05, 0.75),
                                         rnd.uniform(0.05, 0.75)))

        def patchwork(x, y, _c=cells):
            return _c[(x // 20, y // 20)]

        got = shadowedge.edge_hardness(write(tmp_path, "patch%d.png" % seed, patchwork))
        if got is None:
            declined += 1
        else:
            worst = max(worst, got["confidence"])
    assert declined >= 3, "the warmth gate must still refuse some shadowless frames outright"
    assert worst <= 0.90, worst        # pinned at the measured 0.841 — see the docstring


def test_an_ambient_only_frame_reports_no_sun_at_all(tmp_path):
    """A dome-only scene has real illumination boundaries — ambient occlusion — and they
    share a ratio honestly, so a consensus accepts them. But that ratio is the SKY's colour,
    not a sun-to-sky ratio, and there is no sun whose size the width could be evidence
    about. The answer is absence, not a hardness with a caveat: a number that gets reported
    is a number that gets used."""
    # An occluded patch is not simply "less of the same light". It sees mostly warm bounce
    # off nearby surfaces, while an open patch also sees the blue sky — so the RATIO between
    # them is the sky against the bounce, which is blue-weighted. A uniform dimming would be
    # spectrally flat and would not exercise this at all.
    bounce = (0.09, 0.075, 0.055)
    occluded = srgb8(tuple(GREY[i] * bounce[i] for i in range(3)))
    open_to_sky = srgb8(tuple(GREY[i] * (bounce[i] + SKY[i]) for i in range(3)))

    def ambient_step(x, _y):
        return occluded if x < SIZE // 2 else open_to_sky

    # the boundary is real and would be measured — it just is not a sun's
    assert shadowedge._shadow_direction(
        tuple(GREY[i] * (bounce[i] + SKY[i]) for i in range(3)),
        tuple(GREY[i] * bounce[i] for i in range(3))) is not None
    assert shadowedge.edge_hardness(write(tmp_path, "ao.png", ambient_step)) is None


def test_the_direction_extractor_refuses_what_additive_light_cannot_do():
    """Sunlight only ever ADDS, so the lit side cannot be darker than the shadow side in any
    channel. A material edge routinely is — red brick against grey mortar is brighter in red
    and darker in blue — and that is free evidence, needing no consensus at all."""
    assert shadowedge._shadow_direction((0.4, 0.3, 0.2), (0.1, 0.1, 0.1)) is not None
    assert shadowedge._shadow_direction((0.4, 0.3, 0.05), (0.1, 0.1, 0.1)) is None  # blue down
    assert shadowedge._shadow_direction((0.1, 0.1, 0.1), (0.1, 0.1, 0.1)) is None   # no step
    # a denominator this small is quantisation, not a measurement
    assert shadowedge._shadow_direction((0.4, 0.3, 0.2), (0.001, 0.1, 0.1)) is None


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def test_the_cluster_finder_votes_rather_than_averaging():
    """A spherical mean would be dragged off the cluster by the outliers this exists to
    reject — on the dawn plate the outliers are the majority — so the estimator has to be a
    vote. Here two thirds of the directions are scattered and one third agree; the answer
    must be the agreeing third, not somewhere in between."""
    rnd = random.Random(23)
    cluster = [unit((0.8 + rnd.uniform(-0.01, 0.01), 0.5, 0.2)) for _ in range(60)]
    scatter = [unit((rnd.random() + 0.01, rnd.random() + 0.01, rnd.random() + 0.01))
               for _ in range(120)]
    dirs = cluster + scatter
    got = shadowedge._sun_cluster(dirs)
    assert got is not None
    axis, members, mean_cos, rank = got
    assert angle_deg(axis, (0.8, 0.5, 0.2)) < shadowedge.RATIO_INLIER_DEG
    assert set(range(60)) <= set(members), "the real cluster must be kept whole"
    assert mean_cos >= math.cos(math.radians(shadowedge.RATIO_INLIER_DEG))
    assert shadowedge._sun_cluster([]) is None


def test_the_sun_cluster_is_the_warm_one_even_when_a_cool_one_is_bigger():
    """The selection rule in isolation: a large cool cluster and a small warm one, and the
    warm one must win. Size is not evidence about which population is the sun's — physics
    is, because s/a is red-weighted in any daylight scene and an albedo ratio is not."""
    cool = [unit((0.2, 0.5, 0.84)) for _ in range(200)]      # a repeated material pair
    warm = [unit((0.84, 0.5, 0.2)) for _ in range(50)]       # the sun's, outnumbered 4:1
    got = shadowedge._sun_cluster(cool + warm)
    assert got is not None
    axis, members, _cos, rank = got
    assert shadowedge._is_warm(axis)
    assert angle_deg(axis, (0.84, 0.5, 0.2)) < 1.0
    assert set(members) == set(range(200, 250))
    assert rank == 1, "it was found only after the bigger cool cluster was peeled away"
    # ...and with no warm population anywhere, there is no sun to report
    assert shadowedge._sun_cluster(cool) is None


def test_the_warmth_line_sits_below_every_plausible_daylight_sun(tmp_path):
    """SUN_WARMTH_MIN is a physical dividing line, not a fitted one, and this pins it to the
    repo's own colour model rather than to any image. Every sun/sky pairing across the
    daylight range must land above it; measured material pairs land well below."""
    from maxgaffer.core.colortemp import kelvin_to_rgb

    for t_sun in (2000, 2500, 3200, 4500, 5500, 6500):
        for t_sky in (7000, 10000, 20000):
            if t_sky <= t_sun:
                continue
            s, a = kelvin_to_rgb(t_sun), kelvin_to_rgb(t_sky)
            assert shadowedge._is_warm(unit([s[i] / a[i] for i in range(3)])), (t_sun, t_sky)
    for bright, dark in (((0.55, 0.50, 0.45), (0.42, 0.16, 0.11)),      # grey  | brick
                         ((0.78, 0.77, 0.74), (0.55, 0.38, 0.23)),      # plaster | wood
                         ((0.59, 0.67, 0.75), (0.28, 0.38, 0.27))):     # sky   | foliage
        lift = [bright[i] / dark[i] - 1.0 for i in range(3)]
        assert not shadowedge._is_warm(unit(lift)), lift


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

    from maxgaffer.core import metrics

    path = write(tmp_path, "cost.png", ramp_painter(6), size=256)
    loaded = metrics._load_pixels(path, max_dim=shadowedge.WORK_MAX_DIM)
    start = time.perf_counter()
    shadowedge.edge_hardness(loaded)
    shadowedge.haze_estimate(loaded)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, "%.3f s for one 256px frame is not a per-probe cost" % elapsed
