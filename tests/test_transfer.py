"""Lighting-TRANSFER score — the honest cross-domain metric.

The critic scores pixels, which is unanswerable when the reference is a photo of a
different building: MXLT scored a real dawn plate at 75 because a CG room cannot produce
that photograph's pixels, not because the lighting was wrong. transfer.score asks the
question an artist actually asks — is my sun where that photo's sun is, at that colour
temperature, hardness and haze — by comparing the recovered rig to the reference's own
ANALYZE reading.
"""
import pytest

from maxgaffer.core import transfer
from maxgaffer.core.genome import LightingState


def rig(**kw):
    st = LightingState()
    for k, v in kw.items():
        st.set(k.replace("__", "."), v)
    return st


GOLDEN = {"sun_active": True, "sun_bearing_deg": 40.0, "sun_altitude_band": "golden",
          "wb_kelvin_estimate": 4000.0, "light_quality": "hard", "atmosphere": "light_haze"}


def test_a_faithful_transfer_scores_high():
    """Camera yaw 65 + bearing 40 = azimuth 105; the golden band centres on 8 degrees."""
    st = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
             sun__size=1.0, exposure__wb_kelvin=4000.0, atmosphere__distance_m=250.0)
    out = transfer.score(GOLDEN, st, camera_yaw_deg=65.0)
    assert out["score"] > 95.0, out
    assert set(out["measurable"]) >= {"direction", "elevation", "warmth", "presence"}


def test_the_sunless_impersonation_is_caught():
    """The exact failure the pixel critic scored ~92: sun switched off against a sunlit
    reference. Transfer must call that what it is."""
    st = rig(sun__enabled=0.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
             exposure__wb_kelvin=4000.0)
    out = transfer.score(GOLDEN, st, camera_yaw_deg=65.0)
    assert out["parts"]["presence"] == 0.0
    assert any("sun is OFF" in n for n in out["notes"])


def test_a_sun_in_the_wrong_quadrant_scores_low():
    st = rig(sun__enabled=1.0, sun__azimuth_deg=285.0, sun__altitude_deg=9.0,
             exposure__wb_kelvin=4000.0)          # 180 degrees out
    out = transfer.score(GOLDEN, st, camera_yaw_deg=65.0)
    assert out["parts"]["direction"] == 0.0
    assert out["score"] < 70.0


def test_elevation_uses_the_BAND_not_a_point():
    """A band is a range: 'golden' centres on 8 degrees with 7 degrees of slack, so 13
    degrees is still a faithful transfer."""
    inside = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=13.0,
                 exposure__wb_kelvin=4000.0)
    assert transfer.score(GOLDEN, inside, 65.0)["parts"]["elevation"] == 1.0
    outside = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=55.0,
                  exposure__wb_kelvin=4000.0)
    assert transfer.score(GOLDEN, outside, 65.0)["parts"]["elevation"] < 0.6


def test_warmth_is_measured_in_mireds_not_kelvin():
    """Equal Kelvin gaps are NOT equally visible. 4000 to 5000 K is an obvious shift;
    9000 to 10000 K is barely perceptible — mireds capture that, Kelvin does not."""
    warm_miss = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
                    exposure__wb_kelvin=5000.0)
    cool_ref = dict(GOLDEN, wb_kelvin_estimate=9000.0)
    cool_miss = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
                    exposure__wb_kelvin=10000.0)
    warm_part = transfer.score(GOLDEN, warm_miss, 65.0)["parts"]["warmth"]
    cool_part = transfer.score(cool_ref, cool_miss, 65.0)["parts"]["warmth"]
    assert cool_part > warm_part      # same 1000 K gap, far less visible when cool


def test_absent_inputs_are_skipped_never_scored_as_agreement():
    bare = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0)
    out = transfer.score({"sun_active": True, "sun_bearing_deg": 40.0,
                          "sun_altitude_band": "golden"}, bare, 65.0)
    assert "warmth" not in out["parts"]        # no estimate and no wb on the rig
    assert "atmosphere" not in out["parts"]
    assert out["score"] > 0


def test_degrades_on_junk_rather_than_raising():
    assert transfer.score(None, None)["score"] == 0.0
    assert transfer.score({}, rig())["score"] == 0.0
    assert transfer.score({"sun_bearing_deg": float("nan")}, rig())["score"] == 0.0
