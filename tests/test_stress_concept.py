"""Concept-stress regressions — the physics and production findings, pinned.

Four findings from the whole-concept stress round:
  * EXIF orientation: a portrait phone reference read sideways poisons the direction
    grid AND what the vision model sees;
  * double-sun energy: a live VRaySun plus a baked dome disc doubles the direct light —
    the hybrid rule is ambient-only seeds when the parametric sun is active;
  * gateway-down: the sweep must still solve direction from the contrastive metric;
  * telemetry: every match leaves a JSON-safe run record (the critic-vs-human
    calibration trail).
"""

import json

import pytest

PIL = pytest.importorskip("PIL.Image")

from maxgaffer.core import domeseed, hdr_min, metrics
from maxgaffer.core.director import (IterationRecord, MatchResult, Hooks,
                                     run_sun_sweep)
from maxgaffer.core.genome import LightingState


def _state():
    st = LightingState()
    st.set("sun.enabled", 1)
    st.set("sun.azimuth_deg", 0.0)
    st.set("sun.altitude_deg", 30.0)
    return st


# --------------------------------------------------------------------- EXIF orientation
def test_exif_orientation_is_honored(tmp_path):
    from PIL import Image

    im = Image.new("RGB", (96, 64))
    px = im.load()
    for y in range(64):
        for x in range(96):
            px[x, y] = (240, 240, 240) if y < 20 else (20, 20, 20)  # bright TOP strip
    exif = Image.Exif()
    exif[274] = 6                     # orientation 6 = rotate 90° CW to display
    p = str(tmp_path / "portrait.jpg")
    im.save(p, "JPEG", exif=exif)

    loaded = metrics._load_pixels(p)
    assert loaded is not None
    _pixels, w, h = loaded
    assert h > w, f"EXIF rotation ignored — got {w}x{h}, expected portrait"

    # the bright strip must now sit on a SIDE (rotation applied), so the 3×3 grid's
    # brightest cell is in a side column, not the top row
    s = metrics.compute_stats(p)
    grid = s["grid"]
    brightest = max(range(9), key=lambda i: grid[i])
    assert brightest % 3 in (0, 2), \
        f"direction grid says brightest cell {brightest} — orientation not applied"


# --------------------------------------------------------------------- double-sun rule
def test_seed_skips_disc_when_parametric_sun_active(tmp_path):
    from PIL import Image

    Image.new("RGB", (64, 48), (200, 180, 150)).save(str(tmp_path / "r.png"))
    out = str(tmp_path / "hybrid.hdr")
    meta = domeseed.build_seed(out, ref_path=str(tmp_path / "r.png"),
                               semantics={"sun_active": True, "sky": "clear"},
                               cam_yaw_deg=0.0, sun_az_deg=90.0, sun_alt_deg=30.0,
                               out_w=64, out_h=32, parametric_sun_active=True)
    assert meta is not None
    assert meta["sun"] is None
    assert meta["disc_policy"] == "skipped_parametric_sun"
    rows = hdr_min.read_hdr(out)
    peak = max(domeseed._lum(p) for r in rows for p in r)
    assert peak < 5.0, "hybrid seed must be ambient-only — a disc doubles the sun energy"

    # same inputs, sun NOT parametric → the disc IS the direction
    out2 = str(tmp_path / "solo.hdr")
    meta2 = domeseed.build_seed(out2, ref_path=str(tmp_path / "r.png"),
                                semantics={"sun_active": True, "sky": "clear"},
                                cam_yaw_deg=0.0, sun_az_deg=90.0, sun_alt_deg=30.0,
                                out_w=64, out_h=32)
    assert meta2["disc_policy"] == "disc" and meta2["sun"] is not None


# --------------------------------------------------------------------- offline sweep
def test_sweep_falls_back_to_metric_when_llm_dead():
    # three probes whose contrastive grids differ; probe 1 matches the reference
    ref_grid = [0.3, 0.0, -0.3, 0.3, 0.0, -0.3, 0.3, 0.0, -0.3]   # lit from one side
    probe_grids = {
        "sweep000.png": [-0.3, 0.0, 0.3] * 3,     # opposite side
        "sweep120.png": [0.3, 0.0, -0.3] * 3,     # matches the reference
        "sweep240.png": [0.0, 0.3, -0.3] * 3,     # off-axis
    }
    applied = []
    hooks = Hooks(
        apply=lambda st: applied.append(st.get("sun.azimuth_deg")),
        render=lambda tag: f"{tag}.png",
        stats=lambda path: {"grid": probe_grids[path.split("/")[-1]]},
        llm_deltas=lambda ctx: "",
        log=lambda m: None,
    )
    az, hint, why = run_sun_sweep(
        _state(), [0.0, 120.0, 240.0], hooks,
        llm_pick=lambda paths, azs: "",           # gateway dead — unusable reply
        ref_stats={"grid": ref_grid})
    assert az == 120.0, f"metric-only fallback picked {az} — expected 120"
    assert "metric-only" in why


def test_sweep_still_fails_cleanly_without_metric():
    hooks = Hooks(apply=lambda st: None, render=lambda tag: f"{tag}.png",
                  stats=lambda path: None, llm_deltas=lambda ctx: "",
                  log=lambda m: None)
    az, _hint, why = run_sun_sweep(_state(), [0.0, 180.0], hooks,
                                   llm_pick=lambda p, a: "", ref_stats=None)
    assert az is None and "unusable" in why


# --------------------------------------------------------------------- run record
def test_match_result_summary_is_json_safe():
    r = MatchResult(
        best_state=_state(), best_score=91.2, best_render="/runs/iter03.png",
        stop_reason="target_reached",
        iterations=[IterationRecord(index=0, state=_state().to_dict(), score=41.0,
                                    analytic_changes={"exposure.ev": 11.5},
                                    llm_accepted={"sun.azimuth_deg": 208.0},
                                    llm_rejected=["x: locked by user"],
                                    assessment="warmer, sun left")],
        polish_gain=1.3, polish_probes=17, ceiling_converged=False)
    s = r.to_summary()
    text = json.dumps(s)                          # must serialize without custom encoders
    back = json.loads(text)
    assert back["best_score"] == 91.2
    assert back["iterations"][0]["llm"] == {"sun.azimuth_deg": 208.0}
    assert back["best_state"]["values"]["sun.altitude_deg"] == 30.0
