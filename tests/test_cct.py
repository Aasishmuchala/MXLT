"""cct — is the measured illuminant temperature right, and can a caller invert it?

Two classes of failure are worth testing here and they are not the same size.

The first is ACCURACY: the module claims to replace ANALYZE's guessed wb_kelvin_estimate
with a measurement, so it has to land on the textbook temperatures of the CIE illuminants,
which are the only ground truth available without a spectrometer.

The second is DIRECTION, and it is the one that has cost this project real matches. A sign
or convention error here is silent — every accuracy test still passes, the number is still
plausible, and the rig ends up warm when the reference is cool. So the direction tests below
assert the ORDERING of temperature against light colour and the identity of the camera-WB
conversion, because those are the assertions an inversion actually breaks.

Plus the refusals: a chromaticity far off the Planckian locus has no correlated colour
temperature, and McCamy's polynomial does not know that — it answers anyway. Several tests
compute the ungated polynomial alongside the module to show exactly what the Duv gate is
suppressing.
"""

import math

import pytest

from maxgaffer.core import cct


# --------------------------------------------------------------------------- helpers
#: CIE XYZ → linear sRGB (D65), the inverse of cct.SRGB_D65_TO_XYZ. Lets a test start from
#: a textbook CHROMATICITY and manufacture the linear-RGB illuminant vector that metrics
#: would have produced for it, so the whole rgb → xy → kelvin chain is under test and not
#: just its second half.
XYZ_TO_SRGB_D65 = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def linear_rgb_for_xy(x, y):
    """The linear sRGB an illuminant of chromaticity (x, y) produces, normalised to Y=1."""
    xyz = (x / y, 1.0, (1.0 - x - y) / y)
    return [sum(row[i] * xyz[i] for i in range(3)) for row in XYZ_TO_SRGB_D65]


def unit3(vec):
    """Unit-normalise, exactly as metrics._unit3 does before storing an illum_* vector."""
    norm = math.sqrt(sum(c * c for c in vec))
    return [c / norm for c in vec]


def stats(sog=None, edge=None, illum=None, **extra):
    """A stats dict shaped like metrics.compute_stats output. When both estimators are
    given and no explicit blend is, ``illum`` is built the way metrics builds it — the
    unit-normalised mean — so the corroboration logic sees a realistic dict."""
    out = dict(extra)
    if sog is not None:
        out["illum_sog"] = unit3(sog)
    if edge is not None:
        out["illum_edge"] = unit3(edge)
    if illum is not None:
        out["illum"] = unit3(illum)
    elif sog is not None and edge is not None:
        out["illum"] = unit3([(out["illum_sog"][i] + out["illum_edge"][i]) / 2.0
                              for i in range(3)])
    return out


def mccamy_ungated(x, y):
    """McCamy's polynomial with NO validity gate — what the module would return if it did
    not refuse off-locus chromaticities. Used to show what the Duv gate suppresses."""
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33


#: Textbook chromaticities and nominal CCTs, with the tolerance each case earns. The
#: blackbody and daylight illuminants are held tight because they are the calibration
#: anchors. F11 gets more room on purpose: it is a triband fluorescent with a spiky, very
#: non-Planckian spectrum whose "4000 K" is a rounded catalogue figure, so demanding
#: single-kelvin agreement there would be testing the catalogue, not this module.
ILLUMINANTS = [
    ("A", 0.44757, 0.40745, 2856, 6.0),
    ("D50", 0.34567, 0.35850, 5003, 6.0),
    ("D55", 0.33243, 0.34744, 5503, 6.0),
    ("D65", 0.31272, 0.32903, 6504, 6.0),
    ("D75", 0.29902, 0.31485, 7504, 6.0),
    ("E", 1.0 / 3.0, 1.0 / 3.0, 5454, 6.0),
    ("F2", 0.37208, 0.37529, 4230, 6.0),
    ("F11", 0.38052, 0.37713, 4000, 12.0),
]


# --------------------------------------------------------------------------- accuracy
@pytest.mark.parametrize("name,x,y,nominal,tol",
                         ILLUMINANTS, ids=[c[0] for c in ILLUMINANTS])
def test_the_standard_illuminants_land_on_their_textbook_temperatures(
        name, x, y, nominal, tol):
    """The CIE illuminants are the only ground truth available without a spectrometer, so
    they are the calibration. Measured 2026-07-26: A +1.3 K, D50 -0.9, D55 -0.8, D65 -0.2,
    D75 -2.2, E +5.0, F2 +0.6, F11 +9.4. The tolerances leave headroom over those without
    leaving room for a regression that changes the matrix or the coefficients."""
    kelvin = cct.xy_to_cct(x, y)
    assert kelvin is not None, "%s must be answerable — it is near the locus" % name
    assert abs(kelvin - nominal) <= tol, "%s: %.1f K vs nominal %d" % (name, kelvin, nominal)


def test_neutral_linear_rgb_is_D65():
    """sRGB's white point IS D65, so equal linear channels must come back at 6504 K. This
    is a structural check on the matrix: a transposed or mis-transcribed row still produces
    plausible temperatures for coloured inputs but moves neutral off its own white point."""
    xy = cct.rgb_to_xy([1.0, 1.0, 1.0])
    assert xy is not None
    assert abs(cct.xy_to_cct(*xy) - 6504) <= 6.0


@pytest.mark.parametrize("name,x,y,nominal,tol",
                         ILLUMINANTS, ids=[c[0] for c in ILLUMINANTS])
def test_the_whole_rgb_chain_recovers_the_illuminant(name, x, y, nominal, tol):
    """Start from the textbook chromaticity, manufacture the linear RGB an illuminant of
    that colour would produce, and push it through the path production uses — rgb_to_xy
    then xy_to_cct. The previous test exercises only the second half; this one proves the
    matrix, the normalisation and the polynomial agree end to end."""
    rgb = linear_rgb_for_xy(x, y)
    kelvin = cct.xy_to_cct(*cct.rgb_to_xy(unit3(rgb)))
    assert kelvin is not None and abs(kelvin - nominal) <= tol, \
        "%s recovered as %s" % (name, kelvin)


def test_chromaticity_ignores_intensity():
    """Chromaticity divides out scale, so a brighter estimate of the same light is the same
    colour. metrics unit-normalises before storing, but nothing in this module's contract
    requires that, and a caller passing a raw accumulator must not get a different answer."""
    dim = cct.rgb_to_xy([0.02, 0.017, 0.011])
    bright = cct.rgb_to_xy([200.0, 170.0, 110.0])
    assert dim is not None and bright is not None
    assert abs(dim[0] - bright[0]) < 1e-9 and abs(dim[1] - bright[1]) < 1e-9


# --------------------------------------------------------------------------- direction
def test_warm_light_reports_a_LOW_kelvin_and_cool_light_a_HIGH_one():
    """The assertion an inversion breaks, and the only one that would catch it.

    Colour temperature runs backwards from the word "warm": orange light is LOW kelvin,
    blue light is HIGH. Every accuracy test above would still pass with the relationship
    reversed — the numbers would remain textbook-plausible and simply be attached to the
    wrong colours. This pins the ordering across the full practical range instead."""
    def kelvin_of(*rgb):
        return cct.xy_to_cct(*cct.rgb_to_xy(unit3(rgb)))

    tungsten = kelvin_of(1.0, 0.72, 0.42)      # orange lamp
    warm_sun = kelvin_of(1.0, 0.85, 0.65)
    neutral = kelvin_of(1.0, 1.0, 1.0)
    overcast = kelvin_of(0.92, 0.96, 1.0)
    shade = kelvin_of(0.78, 0.89, 1.0)         # blue open shade

    assert tungsten < warm_sun < neutral < overcast < shade, \
        (tungsten, warm_sun, neutral, overcast, shade)
    assert tungsten < 5000.0, "an orange lamp must not read as daylight or cooler"
    assert shade > 7000.0, "blue shade must not read as warm"


def test_the_camera_white_balance_takes_the_same_number_not_its_opposite():
    """Neutralising a light means naming it, not opposing it: a 2856 K tungsten illuminant
    is cancelled by a 2856 K camera WB — the LOW number. The tempting error is to
    "compensate" for warmth by raising the WB, which doubles the cast instead of removing
    it. colortemp.py documents the same direction from the other end: a WB of 3200 K makes
    the camera divide out orange, so the render comes back cooler.

    The identity is asserted, and so is the consequence, because a future refactor that
    quietly introduces an offset would still satisfy the identity for one value."""
    assert cct.camera_wb_kelvin_for_illuminant(2856.0) == 2856.0
    assert cct.camera_wb_kelvin_for_illuminant(6504.0) == 6504.0
    warm_wb = cct.camera_wb_kelvin_for_illuminant(2856.0)
    cool_wb = cct.camera_wb_kelvin_for_illuminant(9000.0)
    assert warm_wb < cool_wb, "a warm illuminant must map to a LOWER camera WB, not higher"


def test_illuminant_kelvin_reports_the_light_not_a_correction():
    """The top-level result must carry the illuminant's own temperature. If it ever started
    returning a correction — the mired-opposite, say — a tungsten scene would report above
    neutral instead of below, so the side of 6504 K the answer falls on is the check."""
    tungsten = cct.illuminant_kelvin(stats(sog=(1.0, 0.72, 0.42), edge=(1.0, 0.73, 0.43)))
    assert tungsten["kelvin"] < 6504.0, tungsten
    blue_sky = cct.illuminant_kelvin(stats(sog=(0.78, 0.89, 1.0), edge=(0.78, 0.89, 1.0)))
    assert blue_sky["kelvin"] > 6504.0, blue_sky


# --------------------------------------------------------------------------- duv
def test_illuminant_A_sits_on_the_planckian_locus():
    """CIE Illuminant A is DEFINED as a Planckian radiator at 2856 K, so its distance from
    the Planckian locus is zero by construction. That makes it a check on the physics
    rather than on a fit: measured -0.00008."""
    assert abs(cct.duv(0.44757, 0.40745)) < 0.0005


def test_the_daylight_series_sits_just_above_the_locus():
    """Daylight is not blackbody radiation — the CIE D series runs slightly ABOVE the
    Planckian locus, at Duv ≈ +0.003 across the whole series. Both the magnitude and the
    SIGN are physics, so this fails if the sign convention is ever flipped, which is the
    failure that would silently turn every green cast into a magenta one."""
    for name, x, y in (("D50", 0.34567, 0.35850), ("D55", 0.33243, 0.34744),
                       ("D65", 0.31272, 0.32903), ("D75", 0.29902, 0.31485)):
        d = cct.duv(x, y)
        assert 0.0028 < d < 0.0036, "%s: %s" % (name, d)


def test_duv_calls_green_positive_and_magenta_negative():
    """The sign convention, stated the way a colourist would: above the locus is green,
    below it is magenta. A green-tinted and a magenta-tinted light can sit at the same
    kelvin, so the sign is the only thing distinguishing them."""
    green = cct.duv(*cct.rgb_to_xy(unit3((0.6, 1.0, 0.6))))
    magenta = cct.duv(*cct.rgb_to_xy(unit3((1.0, 0.6, 1.0))))
    assert green > 0.0 and magenta < 0.0, (green, magenta)


# --------------------------------------------------------------------------- refusals
def test_a_deep_green_illuminant_is_refused_rather_than_named():
    """A CCT names the nearest blackbody, and a deep green light does not have one — no
    temperature of glowing metal is green. Past the CIE limit of 0.05 the answer is that
    there is no answer, so the module must say so instead of picking the closest point on a
    curve the colour is nowhere near."""
    x, y = cct.rgb_to_xy(unit3((0.2, 1.0, 0.2)))
    assert cct.duv(x, y) > cct.CIE_MAX_DUV
    assert cct.xy_to_cct(x, y) is None


def test_a_magenta_illuminant_is_refused_even_though_the_polynomial_answers():
    """The reason the gate is not defensive padding.

    McCamy's denominator is (0.1858 - y), and a magenta chromaticity passes close to that
    pole, so the polynomial returns a large, precise, entirely fictional number. Measured
    2026-07-26: xy (0.3191, 0.1935) — a magenta light — comes out at 28848 K. Anything
    downstream would read that as an extreme blue sky and white-balance hard the wrong way.
    The test asserts the ungated value really is that absurd, so it cannot rot into a
    tautology if the gate is later widened."""
    x, y = 0.3191, 0.1935
    assert mccamy_ungated(x, y) > 25000.0, "fixture no longer demonstrates the failure"
    assert abs(cct.duv(x, y)) > cct.CIE_MAX_DUV
    assert cct.xy_to_cct(x, y) is None


def test_a_pure_blue_light_is_not_reported_as_candlelight():
    """The inversion class this module exists to prevent, in its purest form. Ungated,
    McCamy reports the sRGB blue primary at 1667 K — candle temperature — which is as
    wrong as a colour-temperature answer can be: the bluest thing representable, labelled
    the warmest thing there is."""
    x, y = 0.15, 0.06
    assert mccamy_ungated(x, y) < 2000.0, "fixture no longer demonstrates the failure"
    assert cct.xy_to_cct(x, y) is None


def test_a_mild_tint_is_answered_but_discounted():
    """Refusal and doubt are different tools. A light can be visibly green and still have a
    perfectly meaningful nearest blackbody, so the module answers — and collapses the
    confidence rather than throwing the reading away. This is the boundary between the two
    behaviours, and it must not become an all-or-nothing cliff."""
    tinted = stats(sog=(0.6, 1.0, 0.6), edge=(0.6, 1.0, 0.6))
    got = cct.illuminant_kelvin(tinted)
    assert got is not None, "a mild tint is still nameable — it must not be refused"
    assert abs(got["duv"]) < cct.CIE_MAX_DUV
    assert got["confidence"] < 0.2, got


def test_the_mccamy_pole_itself_cannot_be_reached():
    """y = 0.1858 makes McCamy's denominator zero. The Duv gate already refuses everything
    near it, but a division by zero would be an exception rather than a wrong number, and
    this module promises never to raise. Belt and braces, asserted."""
    assert cct.xy_to_cct(0.3320, 0.1858) is None
    assert cct.xy_to_cct(0.5, 0.1858) is None


# --------------------------------------------------------------------------- confidence
def test_confidence_falls_as_the_tint_grows():
    """Confidence must track how far off-locus the chromaticity sits, because that is
    exactly how much the single number 'kelvin' is failing to describe the light. Asserted
    as a monotone sequence rather than against fixed values, so retuning the falloff scale
    does not break the test but reversing it does."""
    confidences = []
    for green in (1.0, 1.35, 1.7, 2.1):
        got = cct.illuminant_kelvin(stats(sog=(1.0, green, 1.0), edge=(1.0, green, 1.0)))
        assert got is not None
        confidences.append(got["confidence"])
    assert confidences == sorted(confidences, reverse=True), confidences
    assert confidences[0] > confidences[-1] * 3.0, confidences


def test_confidence_falls_when_the_two_estimators_disagree():
    """Shades-of-Gray and gray-edge fail differently — one on chromatic albedo, one on
    textureless frames — so when they agree the reading is corroborated and when they do
    not, one of them is being fooled and there is no way to tell which. Agreement is
    measured in mireds, transfer.py's scale, so a 4000-vs-6500 K split reads as the large
    disagreement it is."""
    agree = cct.illuminant_kelvin(stats(sog=(1.0, 0.72, 0.42), edge=(1.0, 0.73, 0.43)))
    clash = cct.illuminant_kelvin(stats(sog=(1.0, 0.72, 0.42), edge=(0.75, 0.85, 1.0)))
    assert agree["confidence"] > 0.8, agree
    assert clash["confidence"] < 0.2, clash
    assert clash["confidence"] < agree["confidence"]


def test_a_single_estimator_cannot_claim_full_confidence():
    """One estimator is an unreplicated measurement. consensus.py exists because exactly
    that has been wrong before on this input, so an answer nothing corroborated is capped
    however clean it looks — a lone reading is not evidence of agreement, only an absence
    of contradiction."""
    lone = cct.illuminant_kelvin({"illum_edge": unit3((1.0, 0.72, 0.42))})
    assert lone is not None
    assert lone["confidence"] <= cct.UNCORROBORATED_CONFIDENCE, lone


# --------------------------------------------------------------------------- source
def test_the_blend_is_only_called_a_blend_when_both_estimators_are_real():
    """metrics builds illum as the normalised mean of the two estimators, so when one is
    the degenerate [0,0,0] the 'blend' is silently just the other one — non-degenerate,
    indistinguishable, and carrying no corroboration at all. Reporting that as source
    'illum' would claim two measurements where there was one, and the confidence cap would
    be skipped along with it."""
    got = cct.illuminant_kelvin({
        "illum_sog": [0.0, 0.0, 0.0],
        "illum_edge": unit3((1.0, 0.72, 0.42)),
        "illum": unit3((1.0, 0.72, 0.42)),      # what metrics' averaging actually produces
    })
    assert got["source"] == "illum_edge", got
    assert got["confidence"] <= cct.UNCORROBORATED_CONFIDENCE, got


def test_a_refused_estimator_lets_the_other_one_answer():
    """Falling back on the first usable VECTOR would be wrong — a gray-edge estimate driven
    into deep green by a strongly coloured scene is a usable vector and an unusable
    temperature. The search is for the first candidate that produces a NUMBER, so the
    healthy estimator still gets its turn instead of the whole reading being lost."""
    got = cct.illuminant_kelvin({
        "illum_sog": unit3((1.0, 0.85, 0.65)),   # fine
        "illum_edge": unit3((0.2, 1.0, 0.2)),    # deep green — refused
        "illum": unit3((0.2, 1.0, 0.2)),
    })
    assert got is not None, "one poisoned estimator must not sink the reading"
    assert got["source"] == "illum_sog", got
    assert 4000.0 < got["kelvin"] < 6500.0, got


def test_the_result_carries_exactly_the_documented_keys():
    """Callers destructure this dict; the four keys are the contract."""
    got = cct.illuminant_kelvin(stats(sog=(1, 1, 1), edge=(1, 1, 1)))
    assert set(got) == {"kelvin", "duv", "source", "confidence"}
    assert 0.0 <= got["confidence"] <= 1.0


# --------------------------------------------------------------------------- degradation
@pytest.mark.parametrize("bad", [
    None, [], [1.0, 2.0], "not a vector", {}, 7,
    [0.0, 0.0, 0.0],                      # metrics' 'estimate unavailable' sentinel
    [float("nan")] * 3, [float("inf"), 1.0, 1.0],
    [1.0, 1.0, None], [-1.0, 1.0, 1.0],   # negative: impossible from either estimator
    [1e-15, 1e-15, 1e-15],                # non-zero but below the noise floor
])
def test_rgb_to_xy_degrades_to_None_and_never_raises(bad):
    """Every one of these has a plausible route into a stats dict — a black frame, a stats
    blob deserialised from an older session, a hand-built test fixture — and a white-balance
    cue is not worth taking a match run down for. Negatives are rejected on principle: both
    estimators are built from magnitudes, so a negative channel means corruption, and it
    would otherwise pass silently through the chromaticity maths as a believable colour."""
    assert cct.rgb_to_xy(bad) is None


@pytest.mark.parametrize("x,y", [
    (float("nan"), 0.33), (0.33, float("nan")), (float("inf"), 0.33),
    (None, None), ("a", "b"), (0.33, float("-inf")),
])
def test_the_chromaticity_entry_points_degrade_to_None(x, y):
    """Neither entry point may raise on input that is not a pair of numbers. NaN is the one
    that matters in practice: it propagates silently out of a degenerate frame, and every
    comparison against it is False, so an unguarded gate would wave it straight through."""
    assert cct.xy_to_cct(x, y) is None
    assert cct.duv(x, y) is None


@pytest.mark.parametrize("x,y", [(0.0, 0.0), (1e9, 1e9), (1.0, 0.0)])
def test_absurd_chromaticities_are_refused_by_measurement_not_by_special_case(x, y):
    """duv MEASURES and xy_to_cct JUDGES — the split is deliberate, so these points are
    refused for the same reason a magenta lamp is, rather than by a separate guard that
    could drift out of step with the real one.

    A distance function should answer for any finite point in the plane, so duv returns a
    number here (-0.33 for the origin) rather than None. What makes these unanswerable is
    that the number is enormous, and that is exactly what the CIE gate reads."""
    measured = cct.duv(x, y)
    assert measured is not None and math.isfinite(measured)
    assert abs(measured) > cct.CIE_MAX_DUV, measured
    assert cct.xy_to_cct(x, y) is None


@pytest.mark.parametrize("stats_dict,why", [
    (None, "not a dict at all"),
    ({}, "empty"),
    ({"count": 100, "mean_rgb": [0.5, 0.5, 0.5]}, "legacy stats predating illum_*"),
    ({"illum_sog": [0, 0, 0], "illum_edge": [0, 0, 0], "illum": [0, 0, 0]},
     "degenerate frame — every estimator unavailable"),
    ({"illum": "purple", "illum_sog": 5, "illum_edge": [1]}, "garbage of every shape"),
    ({"illum_sog": unit3((0.2, 1.0, 0.2)), "illum_edge": unit3((0.2, 1.0, 0.2)),
      "illum": unit3((0.2, 1.0, 0.2))}, "usable vectors, but no nameable temperature"),
])
def test_illuminant_kelvin_returns_None_when_it_has_nothing_to_say(stats_dict, why):
    """None is the honest answer for all of these, and it is a different answer from a low
    confidence: there is no reading here to discount. The legacy case matters most — stats
    dicts written before the illum_* keys existed must not make this raise."""
    assert cct.illuminant_kelvin(stats_dict) is None, why
