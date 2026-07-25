"""Directional evidence from the sun-patch map — the free half of the direction problem.

ANALYZE's bearing is a single-image absolute judgement and measurably unreliable: four reads
of one golden-hour reference returned 45.0, -52.5, 77.6 and 64.9 degrees, and once the sign
came back inverted. `metrics.compute_stats` already reports WHERE the bright patches are on
every probe, and nothing was reading it. These tests pin what that map can honestly say —
the SIDE of frame the light is on, the sign of the rotation that would fix it — and, just as
important, every case where it must refuse rather than guess.
"""
import math

import pytest

from maxgaffer.core import patchgeom


def stats(cells, frac=0.05):
    """A stats dict carrying the patch map exactly as metrics.compute_stats emits it:
    `hot_grid` row-major 5x5 (index = row * 5 + col, row 0 at the TOP), normalised to sum
    to 1, beside the share of frame the patches cover."""
    grid = [0.0] * 25
    for cell in cells:
        grid[cell] = 1.0 / len(cells)
    return {"hot_frac": frac, "hot_grid": grid}


LEFT = stats([10, 15, 20])          # a shaft raking the left-hand floor  (col 0, rows 2-4)
RIGHT = stats([14, 19, 24])         # the same shaft mirrored             (col 4, rows 2-4)
MIDDLE = stats([12, 17, 22])        # centred                             (col 2, rows 2-4)


# ------------------------------------------------------------------------ description
def test_centroid_reads_x_rightward_and_y_DOWNWARD():
    """The y axis points down, because `hot_grid` is stored in image order (row 0 = top).
    Getting that backwards would silently mirror every floor patch into the ceiling, and
    nothing else in the module would complain."""
    assert patchgeom.patch_centroid(LEFT) == (0.1, 0.7)
    assert patchgeom.patch_centroid(RIGHT) == (0.9, 0.7)
    ceiling = patchgeom.patch_centroid(stats([0, 1, 2, 3, 4]))     # top row only
    floor = patchgeom.patch_centroid(stats([20, 21, 22, 23, 24]))  # bottom row only
    assert ceiling[1] == 0.1 and floor[1] == 0.9
    assert ceiling[0] == floor[0] == 0.5                # neither is biased left or right


def test_centroid_has_nothing_to_report_without_a_usable_map():
    """Absence is not a position. Every junk shape degrades to None instead of returning a
    plausible-looking (0.5, 0.5), which downstream would read as 'the light is centred'."""
    assert patchgeom.patch_centroid({"hot_grid": [0.0] * 25}) is None   # overcast: no patch
    assert patchgeom.patch_centroid({}) is None                        # legacy stats
    assert patchgeom.patch_centroid(None) is None
    assert patchgeom.patch_centroid({"hot_grid": [0.0] * 9}) is None    # 3x3, wrong map
    assert patchgeom.patch_centroid({"hot_grid": ["x"] * 25}) is None
    assert patchgeom.patch_centroid({"hot_grid": [float("nan")] * 25}) is None
    bad = [0.04] * 25
    bad[7] = -0.5                       # a normalised distribution cannot go negative
    assert patchgeom.patch_centroid({"hot_grid": bad}) is None


def test_spread_separates_one_shaft_from_a_scatter_of_speculars():
    """The number that decides whether a centroid means anything. Landmarks are exact for
    5x5 cell centres: one cell 0.0, two neighbours 0.10, evenly spread 0.40."""
    assert patchgeom.patch_spread(stats([12])) == 0.0
    assert patchgeom.patch_spread(stats([12, 13])) == pytest.approx(0.1, abs=1e-3)
    assert patchgeom.patch_spread({"hot_frac": 0.2, "hot_grid": [0.04] * 25}) \
        == pytest.approx(0.4, abs=1e-3)
    assert patchgeom.patch_spread(None) is None
    assert patchgeom.patch_spread(LEFT, axis="diagonal") is None      # unknown axis


def test_lateral_spread_forgives_the_window_AND_the_patch_it_throws():
    """A blown window and the floor patch it casts are two blobs in the SAME column: a
    perfectly localised direction with a large radial spread. Judging that 'scattered'
    would refuse the clearest evidence this module ever gets, so direction work reads the x
    axis alone — and the hint must still accept the pattern."""
    window_and_floor = stats([2, 12, 22])            # col 2, rows 0 / 2 / 4
    assert patchgeom.patch_spread(window_and_floor, axis="x") == 0.0
    assert patchgeom.patch_spread(window_and_floor) > 0.3        # radially very spread out
    assert patchgeom.bearing_hint(LEFT, window_and_floor, 30.0) is not None


# ------------------------------------------------------------------------ agreement
def test_bearing_agreement_reads_the_side_of_frame():
    """1.0 is the same column; 0.0 is outermost against outermost, the widest disagreement
    a 5x5 map can express — which is what a sign-flipped bearing looks like."""
    assert patchgeom.bearing_agreement(LEFT, LEFT) == 1.0
    assert patchgeom.bearing_agreement(LEFT, RIGHT) == 0.0
    assert patchgeom.bearing_agreement(RIGHT, LEFT) == 0.0       # symmetric
    near = patchgeom.bearing_agreement(LEFT, stats([11, 16, 21]))    # one column over
    assert 0.7 < near < 1.0
    assert near > patchgeom.bearing_agreement(LEFT, MIDDLE)      # and monotone with distance


def test_bearing_agreement_declines_instead_of_inventing_an_opposite():
    """metrics.highlight_similarity returns 0.0 when one side has no patches, and is right
    to — it is SCORED, and a candidate must not be able to delete a criterion by switching
    its sun off. This is not scored, so the same 0.0 would be a lie: it reads as 'they point
    opposite ways', which a missing map never said."""
    from maxgaffer.core.metrics import highlight_similarity

    dark = {"hot_frac": 0.0, "hot_grid": [0.0] * 25}
    assert highlight_similarity(LEFT, dark) == 0.0               # the scored metric
    assert patchgeom.bearing_agreement(LEFT, dark) is None       # the evidence extractor
    assert patchgeom.bearing_agreement(LEFT, {}) is None         # stats predating the map
    speculars = stats([12], frac=0.0004)                         # a few stray bright pixels
    assert patchgeom.bearing_agreement(LEFT, speculars) is None


# ------------------------------------------------------------------------ the hint
def test_hint_sign_points_the_sun_the_way_that_moves_the_patch_toward_the_reference():
    """The load-bearing case. The reference's shaft is hard LEFT of frame, the render's is
    hard RIGHT, and the render's sun is in front of the camera (bearing +30).

    Light travels away from the sun, so a shaft moves opposite to the sun's lateral
    direction: to drag the render's patch LEFT, the sun must rotate further camera-RIGHT.
    Positive delta_deg is camera-right, so this must come back positive — and the mirrored
    setup must come back as its exact negative, or the module is reading one of the two
    frames backwards."""
    left_ref = patchgeom.bearing_hint(LEFT, RIGHT, 30.0)
    right_ref = patchgeom.bearing_hint(RIGHT, LEFT, 30.0)
    assert left_ref["delta_deg"] > 0.0, left_ref
    assert right_ref["delta_deg"] < 0.0, right_ref
    assert left_ref["delta_deg"] == -right_ref["delta_deg"]
    assert left_ref["lateral_shift"] < 0.0        # the render's light must move left
    assert left_ref["bearing_deg"] == 30.0


def test_hint_uses_the_camera_yaw_to_reach_the_bearing():
    """The rig stores a world azimuth; the geometry lives in camera-relative bearings, the
    same conversion transfer.score and domeseed make (bearing = azimuth - camera yaw). A
    yaw of 95 with an azimuth of 125 is the identical shot as azimuth 30 with yaw 0."""
    turned = patchgeom.bearing_hint(LEFT, RIGHT, 125.0, camera_yaw_deg=95.0)
    assert turned["bearing_deg"] == 30.0
    assert turned["delta_deg"] == patchgeom.bearing_hint(LEFT, RIGHT, 30.0)["delta_deg"]


def test_hint_inverts_when_the_sun_is_BEHIND_the_camera():
    """The lateral motion per degree is -cos(bearing), so it changes sign as the sun passes
    the camera. A front-lit shot (bearing 150) needs the opposite rotation to a back-lit one
    (bearing 30) for the same patch disagreement — which is exactly why this function
    demands the current azimuth instead of returning a bare direction."""
    front = patchgeom.bearing_hint(LEFT, RIGHT, 30.0)["delta_deg"]
    behind = patchgeom.bearing_hint(LEFT, RIGHT, 150.0)["delta_deg"]
    assert front == -behind
    assert front > 0.0 > behind


def test_hint_refuses_the_side_on_bearing_where_the_map_cannot_know():
    """At bearing +-90 the patch's lateral offset is at its extremum: the derivative passes
    through zero and 80 degrees looks like 100. Guessing there is worse than silence,
    because a wrong sign sends the loop the long way round."""
    assert patchgeom.bearing_hint(LEFT, RIGHT, 90.0) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, -85.0) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, 100.0) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, 70.0) is not None     # outside the band
    assert patchgeom.bearing_hint(LEFT, RIGHT, 110.0) is not None


def test_hint_magnitude_grows_with_the_disagreement_and_stays_capped():
    """Monotone in the evidence, and never larger than MAX_HINT_DEG — the model knows the
    SIGN, not the room, so an over-shooting nudge would oscillate and could shove the sun
    through the +-90 band where the sign flips."""
    one_cell = patchgeom.bearing_hint(MIDDLE, stats([13, 18, 23]), 30.0)
    four_cells = patchgeom.bearing_hint(LEFT, RIGHT, 30.0)
    assert 0.0 < one_cell["delta_deg"] < four_cells["delta_deg"]
    assert four_cells["delta_deg"] <= patchgeom.MAX_HINT_DEG
    assert one_cell["confidence"] < four_cells["confidence"]     # weaker evidence, held loose


def test_hint_reports_a_zero_rotation_when_the_frames_already_agree():
    """Zero is a RESULT, not a refusal: the light is already on the right side, to the
    resolution a 5x5 map has. It also proves the vertical axis is out of scope — two frames
    whose patches differ only in HEIGHT get no azimuth advice, because that difference is
    altitude and room depth, not bearing."""
    same = patchgeom.bearing_hint(LEFT, LEFT, 30.0)
    assert same["delta_deg"] == 0.0 and same["confidence"] > 0.0
    high = stats([2, 7])                    # col 2, top of frame
    low = stats([17, 22])                   # col 2, bottom of frame
    vertical_only = patchgeom.bearing_hint(high, low, 30.0)
    assert vertical_only["delta_deg"] == 0.0


def test_hint_refuses_patches_too_scattered_to_localise():
    """A map smeared across the whole width has a centroid, but it is the middle of a smear
    rather than the position of anything — many small speculars, not one shaft."""
    scattered = {"hot_frac": 0.2, "hot_grid": [0.04] * 25}
    assert patchgeom.patch_spread(scattered, axis="x") > patchgeom.MAX_LATERAL_SPREAD
    assert patchgeom.bearing_hint(LEFT, scattered, 30.0) is None
    assert patchgeom.bearing_hint(scattered, RIGHT, 30.0) is None


def test_hint_declines_exteriors_because_the_transport_rule_inverts_there():
    """Indoors the bright patch is light CAST across the room, displaced away from the sun.
    Outdoors it is the sun-FACING facade, sitting on the SAME side as the sun — the sign
    flips, and mixed with sunlit glass and sky nothing here says which population won."""
    assert patchgeom.bearing_hint(LEFT, RIGHT, 30.0, scene_type="exterior") is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, 30.0, scene_type="") is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, 30.0, scene_type=None) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, 30.0,
                                  scene_type="interior_with_view") is not None


def test_hint_degrades_on_junk_rather_than_raising():
    """Loop stats can never fail — same contract metrics and transfer hold to."""
    assert patchgeom.bearing_hint(None, RIGHT, 30.0) is None
    assert patchgeom.bearing_hint(LEFT, None, 30.0) is None
    assert patchgeom.bearing_hint({}, RIGHT, 30.0) is None            # legacy stats
    assert patchgeom.bearing_hint(LEFT, RIGHT, float("nan")) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, float("inf")) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, "east") is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, None) is None
    assert patchgeom.bearing_hint(LEFT, RIGHT, 30.0, camera_yaw_deg="north") is None
    assert patchgeom.bearing_hint(LEFT, stats([14, 19, 24], frac=0.0), 30.0) is None


def test_a_grid_WITHOUT_hot_frac_describes_but_never_infers():
    """Deliberate split. The centroid describes whatever map it is handed; an inference has
    to believe the map means something first, so it also demands hot_frac — the same key
    sunsolve gates on before spending fifty renders."""
    legacy = {"hot_grid": stats([10, 15, 20])["hot_grid"]}     # map, but no frame fraction
    assert patchgeom.patch_centroid(legacy) == (0.1, 0.7)
    assert patchgeom.patch_spread(legacy) is not None
    assert patchgeom.bearing_agreement(legacy, RIGHT) is None
    assert patchgeom.bearing_hint(legacy, RIGHT, 30.0) is None


def test_a_free_hint_can_never_outrank_a_rendered_solve():
    """This is an uncalibrated model read across two different rooms. The controller accepts
    sunsolve's measured answer at confidence >= 0.5; every hint must land strictly below
    that, whatever the inputs, or a guess starts overruling a measurement."""
    assert patchgeom.MAX_CONFIDENCE < 0.5
    best_possible = patchgeom.bearing_hint(stats([10, 15, 20], frac=0.4),
                                           stats([14, 19, 24], frac=0.4), 0.0)
    assert best_possible["confidence"] <= patchgeom.MAX_CONFIDENCE
    assert best_possible["confidence"] < 0.5


def test_the_module_stays_pure_stdlib_and_sans_IO():
    """It runs inside Max's bundled Python on every probe, where numpy and Pillow may not
    exist and a stray import would take the whole loop down with it."""
    with open(patchgeom.__file__, encoding="utf-8") as fh:
        src = fh.read()
    for banned in ("import numpy", "from numpy", "import PIL", "from PIL",
                   "import pymxs", "from pymxs", "open(", "requests"):
        assert banned not in src, banned
    imports = sorted({line.split()[1].split(".")[0]
                      for line in src.splitlines() if line.startswith("import ")})
    assert imports == ["math"]
    assert math.isfinite(patchgeom.MAX_CENTROID_SEPARATION)
