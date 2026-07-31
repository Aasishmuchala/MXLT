"""vtone — the Vantage→V-Ray display transfer: fit, enforcement, and the two honesty
instruments the 2026-07-31 stress-test made load-bearing.

The tests that matter most here are the ADVERSARIAL ones: the AE simulation (a transfer
that is NOT a function of input value alone must be caught by dispersion/held-out
residual, never absolved by a small in-sample number) and the monotonicity enforcement
(the ordinal sun solve — the one shipping consumer of grabs — survives any monotone
curve, so a fit that reordered brightness would break the only thing that already works).
"""

from __future__ import annotations

import math

from maxgaffer.core import vtone
from maxgaffer.core.vtone import VTone, curve_dispersion, merge_samples, \
    pixel_pairs, quantile_pairs


# --------------------------------------------------------------------------- fixtures
def _rows_from_fn(fn, w=48, h=27):
    """A synthetic plate whose pixel values sweep 0..255 deterministically, pushed
    through ``fn`` (the simulated tone curve) per channel."""
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            v = (x * 5 + y * 11) % 256
            row.append((fn(v), fn(v), fn(v)))
        rows.append(row)
    return rows


def _ident(v):
    return v


def _gamma22(v):
    """What Vantage would show if it applied a 1/2.2 display gamma V-Ray does not."""
    return max(0, min(255, int(round(((v / 255.0) ** (1.0 / 2.2)) * 255.0))))


def _scurve(v):
    """A filmic-ish S: toe below 0.3, shoulder above 0.7, monotone throughout."""
    x = v / 255.0
    y = x * x * (3.0 - 2.0 * x)          # smoothstep — classic S shape
    return max(0, min(255, int(round(y * 255.0))))


def _samples_from_curve(fn, levels=range(0, 256, 3)):
    """Per-channel (vantage_level, vray_level) samples where vantage shows fn(v) of the
    V-Ray value v — i.e. the fit must recover fn⁻¹."""
    pairs = [(float(fn(v)), float(v)) for v in levels]
    return {"r": list(pairs), "g": list(pairs), "b": list(pairs)}


# --------------------------------------------------------------------------- fitting
def test_identity_transfer_round_trips_exactly():
    tone = VTone.fit(_samples_from_curve(_ident))
    assert tone is not None
    rows = _rows_from_fn(_ident, 16, 9)
    assert tone.to_vray_space(rows) == rows
    assert tone.monotonic
    assert tone.monotonic_observed


def test_gamma_curve_is_recovered_and_inverted():
    """Vantage shows x^(1/2.2); the fit must map those display values back to V-Ray's.
    Tolerance is 2 levels: quantisation of the SIMULATED curve puts ±1 in the samples
    before the fit ever sees them."""
    tone = VTone.fit(_samples_from_curve(_gamma22))
    assert tone is not None
    for v in range(0, 256, 7):
        got = tone.luts["g"][_gamma22(v)]
        assert abs(got - v) <= 2.0, f"level {v}: {got}"


def test_scurve_is_recovered():
    tone = VTone.fit(_samples_from_curve(_scurve))
    assert tone is not None
    # the S-curve is flat at the extremes so invert only the well-conditioned middle
    for v in range(40, 216, 7):
        got = tone.luts["g"][_scurve(v)]
        assert abs(got - v) <= 3.0, f"level {v}: {got}"


def test_monotonicity_is_enforced_on_noisy_input():
    """Noisy-but-monotone truth in, monotone LUT out — and the observation flag stays
    True because the underlying relationship IS monotone within the slack."""
    import random

    rng = random.Random(7)
    pairs = []
    for v in range(0, 256, 2):
        for _ in range(5):
            pairs.append((float(v), v + rng.uniform(-6.0, 6.0)))
    tone = VTone.fit({"r": pairs, "g": pairs, "b": pairs})
    assert tone is not None
    assert tone.monotonic
    for ch in vtone.CHANNELS:
        lut = tone.luts[ch]
        assert all(b >= a - 1e-9 for a, b in zip(lut, lut[1:]))


def test_grossly_nonmonotone_observation_is_recorded_not_smoothed_away():
    """A relationship that genuinely DECREASES over a range is evidence the pairs never
    came from a 1D curve. The fit still produces a monotone LUT (its contract), but
    ``monotonic_observed`` must say what was actually seen."""
    pairs = [(float(v), float(v)) for v in range(0, 128, 2)]
    pairs += [(float(v), float(255 - v)) for v in range(128, 256, 2)]   # falls hard
    tone = VTone.fit({"r": pairs, "g": pairs, "b": pairs})
    assert tone is not None
    assert tone.monotonic                    # the LUT keeps the consumer contract
    assert not tone.monotonic_observed       # the observation is on the record


def test_sparse_bins_interpolate_across_gaps():
    """~26 populated levels of 256 — the realistic yield of a small calibration sweep.
    The LUT must interpolate the gaps and extrapolate FLAT outside the observed range
    (inventing slope where nothing was measured is how shadows get corrupted)."""
    pairs = [(float(v), float(v) * 0.8 + 20.0) for v in range(20, 236, 10)]
    tone = VTone.fit({"r": pairs, "g": pairs, "b": pairs})
    assert tone is not None
    # interior gap: level 105 sits between knots 100 and 110 — linear between them
    expect = 105 * 0.8 + 20.0
    assert abs(tone.luts["g"][105] - expect) <= 1.0
    # flat extrapolation outside the observed [20, 230]
    assert tone.luts["g"][0] == tone.luts["g"][20]
    assert tone.luts["g"][255] == tone.luts["g"][230]


def test_too_sparse_refuses():
    """Fewer than MIN_LEVELS distinct input levels is an assertion, not a measurement —
    fit() must refuse rather than hand back an object whose whole contract is 'safe to
    apply'."""
    pairs = [(float(v), float(v)) for v in (0, 64, 128, 255)]
    assert VTone.fit({"r": pairs, "g": pairs, "b": pairs}) is None


def test_degenerate_inputs_do_not_raise():
    assert VTone.fit(None) is None
    assert VTone.fit({}) is None
    assert VTone.fit({"r": [], "g": [], "b": []}) is None
    one = [(128.0, 128.0)]
    assert VTone.fit({"r": one, "g": one, "b": one}) is None
    same = [(128.0, float(v)) for v in range(0, 256, 8)]     # all one input level
    assert VTone.fit({"r": same, "g": same, "b": same}) is None
    junk = [(float("nan"), 1.0), (None, 2.0), ("x", 3.0)]
    assert VTone.fit({"r": junk, "g": junk, "b": junk}) is None


def test_hostile_containers_refuse_instead_of_raising():
    """Fuzz-gauntlet regression (2026-07-31): fit({"r": None}) raised TypeError out of
    list(None) BEFORE the refusal logic ran — a module whose contract is 'None is a
    refusal' must extend that manner to its inputs. Same for non-dict containers,
    non-iterable channels, and tuples of the wrong arity."""
    assert VTone.fit({"r": None, "g": None, "b": None}) is None
    assert VTone.fit({"r": 42, "g": "x", "b": object()}) is None
    assert VTone.fit(["not", "a", "dict"]) is None
    assert VTone.fit("rgb") is None
    wrong_arity = {ch: [(1.0,)] * 40 + [(1.0, 2.0, 3.0)] * 40 for ch in vtone.CHANNELS}
    assert VTone.fit(wrong_arity) is None                    # no well-formed pairs left
    # …and a channel that mixes junk with enough good pairs still fits from the good ones
    good = [(float(v), float(v)) for v in range(0, 256, 4)]
    mixed = {ch: [(1.0,), None, "x"] + good for ch in vtone.CHANNELS}
    assert VTone.fit(mixed) is not None

    tone = VTone.fit({ch: good for ch in vtone.CHANNELS})
    assert tone.residual_on({"r": None, "g": None, "b": None}) is None
    assert tone.residual_on(["not a dict"]) is None
    assert tone.residual_on({"r": 42, "g": 42, "b": 42}) is None


def test_out_of_range_pixels_do_not_index_error():
    """to_vray_space masks with & 0xFF — a hostile pixel value must never IndexError a
    256-entry LUT (a grab pipeline bug upstream should surface as wrong colour, not as
    a crash inside the correction)."""
    tone = VTone.fit({ch: [(float(v), float(v)) for v in range(0, 256, 2)]
                      for ch in vtone.CHANNELS})
    out = tone.to_vray_space([[(999, -5, 256), (12345, 0, 255)]])
    assert len(out) == 1 and len(out[0]) == 2


def test_per_channel_independence():
    """A red-only tint difference must correct red and leave green/blue untouched —
    folding channels together would launder a colour cast into a brightness error."""
    r_pairs = [(float(v), min(255.0, float(v) * 1.2)) for v in range(0, 256, 3)]
    gb_pairs = [(float(v), float(v)) for v in range(0, 256, 3)]
    tone = VTone.fit({"r": r_pairs, "g": gb_pairs, "b": gb_pairs})
    assert tone is not None
    rows = [[(100, 100, 100)] * 4]
    out = tone.to_vray_space(rows)
    assert out[0][0][0] == 120                   # red corrected
    assert out[0][0][1] == 100                   # green untouched
    assert out[0][0][2] == 100                   # blue untouched


# --------------------------------------------------------------------------- honesty
def test_ae_simulation_small_insample_large_heldout():
    """THE ESCAPE-HATCH TEST — the stress-test's central finding, encoded.

    Simulated auto-exposure: the same input level maps to two very different outputs
    depending on a hidden variable (scene brightness state). A fit pooled over one
    'state' looks clean IN-SAMPLE; the held-out residual on the OTHER state must be
    large, and the per-state curve dispersion must scream. If this test ever passes with
    a small held-out residual, the kill-switch is broken and the wider design is unsafe.
    """
    # state A: dim scene, AE lifts — display = v * 1.3 (clamped)
    state_a = {ch: [(float(v), min(255.0, v * 1.3)) for v in range(0, 256, 3)]
               for ch in vtone.CHANNELS}
    # state B: bright scene, AE pulls down — display = v * 0.7
    state_b = {ch: [(float(v), v * 0.7) for v in range(0, 256, 3)]
               for ch in vtone.CHANNELS}

    fit_a = VTone.fit(state_a)
    assert fit_a is not None
    in_sample = fit_a.residual_on(state_a)
    held_out = fit_a.residual_on(state_b)
    assert in_sample is not None and in_sample < 2.0        # looks clean in-sample…
    assert held_out is not None and held_out > vtone.RESIDUAL_LIMIT   # …and is not

    # …and the family of per-state curves detects it without any held-out data at all
    fit_b = VTone.fit(state_b)
    spread = curve_dispersion([fit_a, fit_b])
    assert spread is not None and spread > vtone.DISPERSION_LIMIT


def test_residual_on_clean_relationship_is_small():
    tone = VTone.fit(_samples_from_curve(_gamma22))
    assert tone is not None
    other_levels = _samples_from_curve(_gamma22, levels=range(1, 256, 7))
    resid = tone.residual_on(other_levels)
    assert resid is not None and resid < 2.0


def test_residual_on_degenerates():
    tone = VTone.fit(_samples_from_curve(_ident))
    assert tone is not None
    assert tone.residual_on(None) is None
    assert tone.residual_on({}) is None
    assert tone.residual_on({"r": [], "g": [], "b": []}) is None


def test_delivery_gap_identity_is_zero_and_offset_is_seen():
    ident = VTone.fit(_samples_from_curve(_ident))
    gap = ident.delivery_gap()
    assert gap["max"] <= 1.0
    shifted = VTone.fit({ch: [(float(v), min(255.0, float(v) + 20.0))
                              for v in range(0, 256, 3)] for ch in vtone.CHANNELS})
    gap2 = shifted.delivery_gap()
    assert gap2["p50"] >= 15.0


def test_curve_dispersion_needs_two_and_skips_none():
    tone = VTone.fit(_samples_from_curve(_ident))
    assert curve_dispersion([tone]) is None
    assert curve_dispersion([tone, None]) is None
    assert curve_dispersion([]) is None
    same = VTone.fit(_samples_from_curve(_ident))
    spread = curve_dispersion([tone, same, None])
    assert spread is not None and spread < 0.5


# --------------------------------------------------------------------------- pairing
def test_quantile_pairs_survive_resolution_mismatch():
    """The whole point of the estimator: same image content at DIFFERENT pixel counts
    must pair the same levels together, with no registration at all. The small plate is
    a true 2× subsample of the big one — same content, different sampling grid — because
    two independently-generated value patterns are different CONTENT, which quantile
    matching would rightly refuse to reconcile."""
    big = _rows_from_fn(_ident, 64, 36)
    small = [row[::2] for row in big[::2]]
    pairs = quantile_pairs(big, small)
    assert pairs is not None
    for v, y in pairs["g"]:
        assert abs(v - y) <= 6.0        # same distribution, different sampling grid


def test_quantile_pairs_recover_a_tone_difference():
    """Vantage-side rows pushed through the gamma curve, V-Ray-side identity: the pairs
    feed fit() and the recovered LUT must invert the curve — the full estimator path."""
    vant = _rows_from_fn(_gamma22, 64, 36)
    vray = _rows_from_fn(_ident, 64, 36)
    pairs = quantile_pairs(vant, vray)
    assert pairs is not None
    tone = VTone.fit(pairs)
    assert tone is not None
    for v in range(30, 226, 15):
        got = tone.luts["g"][_gamma22(v)]
        assert abs(got - v) <= 6.0, f"level {v}: {got}"


def test_quantile_pairs_empty_refuse():
    assert quantile_pairs([], _rows_from_fn(_ident)) is None
    assert quantile_pairs(_rows_from_fn(_ident), []) is None


def test_pixel_pairs_require_identical_dims():
    a = _rows_from_fn(_ident, 32, 18)
    b = _rows_from_fn(_ident, 31, 18)
    assert pixel_pairs(a, b) is None
    assert pixel_pairs(a, a) is not None


def test_pixel_pairs_detect_spatial_operator_quantile_cannot_see():
    """The two-role architecture: a spatially-varying difference (half the frame lifted,
    half darkened, distribution roughly preserved) hides from quantile matching but is
    exposed by the registered pairs' residual."""
    base = _rows_from_fn(_ident, 48, 26)
    spatial = []
    for y, row in enumerate(base):
        if y < 13:
            spatial.append([(min(255, r + 30), min(255, g + 30), min(255, b + 30))
                            for r, g, b in row])
        else:
            spatial.append([(max(0, r - 30), max(0, g - 30), max(0, b - 30))
                            for r, g, b in row])
    q = quantile_pairs(spatial, base)
    p = pixel_pairs(spatial, base)
    assert q is not None and p is not None
    tone = VTone.fit(q)
    assert tone is not None
    q_resid = tone.residual_on(q)
    p_resid = tone.residual_on(p)
    assert q_resid is not None and p_resid is not None
    # the detector must read clearly above the estimator on a spatial operator
    assert p_resid > q_resid * 2.0
    assert p_resid > vtone.RESIDUAL_LIMIT


def test_merge_samples_concats_and_skips_none():
    a = {ch: [(1.0, 1.0)] for ch in vtone.CHANNELS}
    b = {ch: [(2.0, 2.0)] for ch in vtone.CHANNELS}
    merged = merge_samples([a, None, b])
    assert merged["r"] == [(1.0, 1.0), (2.0, 2.0)]


# --------------------------------------------------------------------------- persistence
def test_round_trip_preserves_behaviour_exactly():
    tone = VTone.fit(_samples_from_curve(_scurve),
                     provenance={"vantage_exe_mtime": 123, "note": "test"})
    tone.residual = 1.25
    tone.dispersion = 0.5
    d = tone.to_dict()
    back = VTone.from_dict(d)
    assert back is not None
    rows = _rows_from_fn(_ident, 16, 9)
    assert back.to_vray_space(rows) == tone.to_vray_space(rows)
    assert back.provenance["vantage_exe_mtime"] == 123
    assert back.residual == 1.25
    assert back.dispersion == 0.5
    assert back.monotonic_observed == tone.monotonic_observed
    # and the round trip is JSON-safe end to end
    import json

    again = VTone.from_dict(json.loads(json.dumps(d)))
    assert again is not None
    assert again.to_vray_space(rows) == tone.to_vray_space(rows)


def test_from_dict_refuses_malformed():
    good = VTone.fit(_samples_from_curve(_ident)).to_dict()
    assert VTone.from_dict(None) is None
    assert VTone.from_dict([]) is None
    assert VTone.from_dict({}) is None
    short = dict(good)
    short["luts"] = {ch: good["luts"][ch][:255] for ch in vtone.CHANNELS}
    assert VTone.from_dict(short) is None
    junk = dict(good)
    junk["luts"] = {ch: ["x"] * 256 for ch in vtone.CHANNELS}
    assert VTone.from_dict(junk) is None
    nan = dict(good)
    nan["luts"] = {ch: [float("nan")] * 256 for ch in vtone.CHANNELS}
    assert VTone.from_dict(nan) is None
    badresid = dict(good)
    badresid["residual"] = "not a number"
    assert VTone.from_dict(badresid) is None


def test_apply_is_fast_enough_for_a_probe():
    """A probe is ≤480×270. The correction must stay far under the ~50 ms grab it
    follows — three list lookups per pixel, no maths in the loop."""
    import time

    tone = VTone.fit(_samples_from_curve(_gamma22))
    rows = _rows_from_fn(_ident, 480, 270)
    t0 = time.perf_counter()
    tone.to_vray_space(rows)
    assert time.perf_counter() - t0 < 0.5
