"""Regressions for the PATH-TO-10 stress-audit fixes: critic component gating,
LLM capability gate, keep-best crash guarantee, latest-render in LLM-visual mode,
stall/slump decoupling, plateau-vs-proven ceiling, clip-aware highlight WB read,
png_min subsampling, orphan/disabled-group warnings, session generation bust,
software exposure at every render site (via the shared helper), auto-detect."""
import json
import math

import pytest
from PIL import Image

from maxgaffer.core import critic, director, metrics, png_min
from maxgaffer.core.director import Hooks, MatchConfig, run_match, run_polish
from maxgaffer.core.genome import LightingState, apply_changes, rig_keys
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import controller as ctl


def full_state():
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.altitude_deg": 30.0,
                 "sun.intensity": 1.0, "sun.size": 3.0, "sun.turbidity": 3.0,
                 "exposure.ev": 12.0, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    st.groups["practicals"] = 1.0
    return st


NOOP_LLM = json.dumps({"assessment": "", "changes": [], "stop": False})


# ------------------------------------------------------------------ critic gating
def test_critic_empty_dicts_score_zero_not_perfect():
    v = critic.score({}, {})
    assert v.score == 0.0 and v.components == {}


def test_critic_partial_data_renormalizes_over_present_components():
    ref = {"log_key": 0.2, "lum_hist": [0.5, 0.5]}
    cur = {"log_key": 0.2, "lum_hist": [0.5, 0.5]}
    v = critic.score(ref, cur)
    assert set(v.components) == {"key", "histogram"}
    assert v.score == 100.0          # both PRESENT components genuinely match


def test_critic_malformed_lab_mean_is_excluded_not_a_crash():
    ref = {"log_key": 0.2, "lab_mean": [50.0, 2.0]}      # wrong length
    cur = {"log_key": 0.2, "lab_mean": [50.0, 2.0, 1.0]}
    v = critic.score(ref, cur)
    assert "color" not in v.components and v.score > 0


# ------------------------------------------------------------------ capability gate
def test_apply_changes_rejects_params_the_rig_lacks():
    st = LightingState()
    st.set("sun.azimuth_deg", 100.0)
    known = rig_keys(st)
    new, accepted, rejected = apply_changes(
        st, {"dome.rotation_deg": 90.0, "group.sconces": 0.5,
             "sun.azimuth_deg": 120.0}, known=known)
    assert "dome.rotation_deg" not in new.values          # phantom NOT fabricated
    assert "sconces" not in new.groups
    assert accepted == {"sun.azimuth_deg": 120.0}
    assert sum("rig has no such parameter" in r for r in rejected) == 2


def test_run_match_gates_llm_deltas_on_rig_capability():
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "dome.rotation_deg", "value": 45.0, "why": "spin"}], "stop": False})
    st = LightingState()
    st.set("sun.azimuth_deg", 100.0)
    st.set("sun.intensity", 1.0)
    hooks = Hooks(apply=lambda s: None, render=lambda t: f"/x/{t}.png",
                  stats=lambda p: None, llm_deltas=lambda ctx: reply,
                  log=lambda m: None)
    res = run_match(st, None, {}, hooks, MatchConfig(max_iterations=2))
    assert "dome.rotation_deg" not in res.best_state.values
    assert any("rig has no such parameter" in r
               for rec in res.iterations for r in rec.llm_rejected)


# ------------------------------------------------------------------ keep-best on crash
def test_crash_mid_match_still_lands_on_best_state():
    applied = []
    calls = {"n": 0}

    def stats(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"log_key": 0.2, "lum_hist": [1.0]}
        raise RuntimeError("stats engine exploded")

    hooks = Hooks(apply=lambda s: applied.append(s.copy()),
                  render=lambda t: f"/x/{t}.png",
                  stats=stats, llm_deltas=lambda ctx: NOOP_LLM, log=lambda m: None)
    ref = {"log_key": 0.2, "lum_hist": [1.0]}
    with pytest.raises(RuntimeError):
        run_match(full_state(), ref, {}, hooks,
                  MatchConfig(max_iterations=4, target_score=101, stall_patience=99))
    # the LAST apply must be the best measured state, not the exploratory one
    assert applied[-1].to_dict() is not None
    assert applied[-1].get("sun.azimuth_deg") == pytest.approx(
        full_state().get("sun.azimuth_deg"), abs=61)   # a real state, applied last


def test_crash_mid_polish_reapplies_best():
    applied = []

    def stats(path):
        raise RuntimeError("boom")

    hooks = Hooks(apply=lambda s: applied.append(s.copy()),
                  render=lambda t: f"/x/{t}.png", stats=stats,
                  llm_deltas=lambda ctx: NOOP_LLM, log=lambda m: None)
    with pytest.raises(RuntimeError):
        run_polish(full_state(), 50.0, {"log_key": 0.2}, hooks, MatchConfig())
    assert applied and applied[-1].get("sun.azimuth_deg") == 100.0   # best re-applied


# ------------------------------------------------------------------ best_render latest
def test_llm_visual_mode_best_render_is_latest_not_first():
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "sun.azimuth_deg", "value": 140.0, "why": "swing"}], "stop": False})
    replies = [reply, NOOP_LLM, NOOP_LLM]
    hooks = Hooks(apply=lambda s: None, render=lambda t: f"/x/{t}.png",
                  stats=lambda p: None,
                  llm_deltas=lambda ctx: replies.pop(0), log=lambda m: None)
    res = run_match(full_state(), None, {}, hooks, MatchConfig(max_iterations=3))
    assert res.best_render == "/x/iter02.png"            # latest, not iter00


# ------------------------------------------------------------------ stall/slump decouple
def test_single_slump_after_marginal_gain_does_not_stall_the_run(monkeypatch):
    scores = [50.0, 50.5, 40.0, 60.0]                    # marginal → slump → recovery

    def fake_score(ref, cur, w=None):
        return critic.Verdict(score=scores.pop(0), components={})

    monkeypatch.setattr(director.critic, "score", fake_score)
    hooks = Hooks(apply=lambda s: None, render=lambda t: f"/x/{t}.png",
                  stats=lambda p: {"log_key": 0.2},
                  llm_deltas=lambda ctx: NOOP_LLM, log=lambda m: None)
    res = run_match(full_state(), {"log_key": 0.2}, {}, hooks,
                    MatchConfig(max_iterations=4, target_score=101,
                                stall_patience=2, analytic=False))
    assert len(res.iterations) == 4                       # ran the full budget
    assert res.best_score == 60.0


# ------------------------------------------------------------------ plateau vs proven
class _FlatWorld:
    def hooks(self):
        return Hooks(apply=lambda s: None, render=lambda t: f"/x/{t}.png",
                     stats=lambda p: {"log_key": 0.2},
                     llm_deltas=lambda ctx: NOOP_LLM, log=lambda m: None)


def test_polish_flat_world_is_proven_ceiling(monkeypatch):
    monkeypatch.setattr(director.critic, "score",
                        lambda ref, cur, w=None: critic.Verdict(50.0, {}))
    st, sc_, probes, converged, proven = run_polish(
        full_state(), 50.0, {"log_key": 0.2}, _FlatWorld().hooks(),
        MatchConfig(polish_rounds=40, polish_stop_at=99.0, polish_max_probes=500,
                    polish_round_eps=0.0))     # eps 0 → exhaust steps, not plateau out
    assert converged is True and proven is True           # floor-exhausted = STRONG claim


def test_polish_refines_steps_before_declaring_a_plateau(monkeypatch):
    """A low-gain round at COARSE steps means the step size is wrong, not that the climb
    is over. This world's optimum sits +0.07 EV away — invisible to the 0.4 initial step
    and to one halving (0.2), reachable only at 0.1. The pre-fix code returned
    converged-plateau after two low-gain rounds WITHOUT refining, abandoning the peak
    (measured on-box: three archetypes quit at ~91 whose self-match optimum was 100)."""
    base_ev = full_state().get("exposure.ev")
    peak = base_ev + 0.07

    def narrow_peak(ref, cur, w=None):
        # cur carries no state, so score the LAST applied candidate the world recorded
        ev = applied.get("ev", base_ev)
        return critic.Verdict(50.0 + 10.0 * math.exp(-((ev - peak) / 0.06) ** 2), {})

    applied = {}

    class _World:
        def hooks(self):
            def _apply(s):
                applied["ev"] = s.get("exposure.ev")
            return Hooks(apply=_apply, render=lambda t: f"/x/{t}.png",
                         stats=lambda p: {"log_key": 0.2},
                         llm_deltas=lambda ctx: NOOP_LLM, log=lambda m: None)

    monkeypatch.setattr(director.critic, "score", narrow_peak)
    st, sc_, probes, converged, proven = run_polish(
        full_state(), 52.5, {"log_key": 0.2}, _World().hooks(),
        MatchConfig(polish_rounds=8, polish_stop_at=99.0, polish_max_probes=400))
    # the peak (60.0) is only reachable by refining past the initial step
    assert sc_ > 56.0, f"polish quit before refining steps (score {sc_})"
    assert st.get("exposure.ev") == pytest.approx(peak, abs=0.12)


# ------------------------------------------------------------------ clip-aware WB read
def _img(tmp_path, name, painter):
    im = Image.new("RGB", (64, 48))
    px = im.load()
    for y in range(48):
        for x in range(64):
            px[x, y] = painter(x, y)
    p = str(tmp_path / name)
    im.save(p)
    return p


def test_clipped_highlights_excluded_from_wb_read(tmp_path):
    # top half: pure clipped white; a band of WARM unclipped highlights; dark floor
    def paint(x, y):
        if y < 7:
            return (255, 255, 255)                        # clipped share of the quartile
        if y < 21:
            return (250, 200, 120)                        # warm, NOT clipped
        return (30, 30, 30)

    s = metrics.compute_stats(_img(tmp_path, "clip.png", paint))
    assert s["hi_clip_frac"] > 0.3
    assert s["lab_mean_hi"][2] > 10.0                     # warm b*, not dragged to ~0


def test_fully_clipped_quartile_falls_back_not_zero(tmp_path):
    s = metrics.compute_stats(_img(tmp_path, "blown.png",
                                   lambda x, y: (255, 255, 255)))
    assert s is not None and s["hi_clip_frac"] == 1.0


# ------------------------------------------------------------------ png_min subsample
def test_png_min_honors_max_dim_in_256_511_band(tmp_path):
    p = str(tmp_path / "wide.png")
    Image.new("RGB", (480, 270), (90, 90, 90)).save(p)
    rows = png_min.read_png_rgb(p, max_dim=256)
    assert rows is not None
    assert len(rows[0]) <= 256 and len(rows) <= 256       # 480//256 floored to 1 before


# ------------------------------------------------------------------ orphan warnings
def test_apply_warns_on_orphaned_dome_and_group(monkeypatch):
    from maxgaffer.maxbridge import apply as ap

    st = LightingState()
    st.set("dome.rotation_deg", 90.0)
    st.groups["ghost"] = 0.5
    warnings = []
    ap._apply_inner({"sun": None, "dome": None, "groups": {}}, {}, st, None, warnings)
    joined = "\n".join(warnings)
    assert "no dome light" in joined
    assert "no such light group" in joined


def test_apply_warns_when_group_lights_all_disabled(monkeypatch):
    from maxgaffer.maxbridge import apply as ap

    monkeypatch.setattr(ap.sc, "get_prop", lambda o, props, default=None: False)
    monkeypatch.setattr(ap.sc, "set_prop", lambda o, props, v: 1)
    st = LightingState()
    st.groups["lamps"] = 1.0
    warnings = []
    ap._apply_inner({"sun": None, "dome": None,
                     "groups": {"lamps": [object(), object()]}}, {}, st, None, warnings)
    assert any("DISABLED" in w for w in warnings)


# ------------------------------------------------------------------ helper + anchor
def test_render_exposed_applies_state_exposure(tmp_path, monkeypatch):
    cfg = cfgmod.Config(software_exposure=True)
    c = ctl.Controller(cfg)

    raw = str(tmp_path / "raw.png")

    def fake_render(cam, out, w, h):
        Image.new("RGB", (32, 32), (180, 180, 180)).save(out)
        return out

    monkeypatch.setattr(ctl.rd, "render_frame", fake_render)
    pre = LightingState()
    pre.set("exposure.ev", 10.0)
    pre.set("exposure.wb_kelvin", 6500.0)
    st = LightingState()
    st.set("exposure.ev", 12.0)                          # 2 stops darker than anchor
    st.set("exposure.wb_kelvin", 6500.0)

    class E:
        pre_match = pre

    out = c._render_exposed(object(), raw, 32, 32, state=st, entry=E())
    with Image.open(out) as opened:
        im = opened.convert("RGB")
    mean = sum(im.getpixel((x, y))[0] for y in range(32) for x in range(32)) / (32 * 32)
    assert mean < 120                                    # visibly darkened vs 180


def test_render_exposed_identity_without_flag(tmp_path, monkeypatch):
    c = ctl.Controller(cfgmod.Config(software_exposure=False))

    def fake_render(cam, out, w, h):
        Image.new("RGB", (8, 8), (100, 100, 100)).save(out)
        return out

    monkeypatch.setattr(ctl.rd, "render_frame", fake_render)
    out = c._render_exposed(object(), str(tmp_path / "o.png"), 8, 8,
                            state=None, entry=None)
    with Image.open(out) as opened:
        im = opened.convert("RGB")
    assert im.getpixel((0, 0)) == (100, 100, 100)


# ------------------------------------------------------------------ auto-detect
def test_verify_exposure_host_flips_flag_on_inert_host(tmp_path, monkeypatch):
    cfg = cfgmod.Config(software_exposure=False)
    c = ctl.Controller(cfg)

    class FakeHost:
        def __init__(self, cam):
            pass

        def read_ev(self):
            return 10.0

        def write_ev(self, v):
            return True

    import maxgaffer.maxbridge.exposure as expmod

    monkeypatch.setattr(expmod, "ExposureHost", FakeHost)

    def fake_render(cam, out, w, h):
        Image.new("RGB", (16, 16), (140, 140, 140)).save(out)   # identical both probes
        return out

    monkeypatch.setattr(ctl.rd, "render_frame", fake_render)
    logs = []
    c._verify_exposure_host(object(), str(tmp_path), logs.append)
    assert c.cfg.software_exposure is True
    assert any("display-stage" in ln for ln in logs)
    # and it never re-checks
    c.cfg.software_exposure = False
    c._verify_exposure_host(object(), str(tmp_path), logs.append)
    assert c.cfg.software_exposure is False


# ------------------------------------------------------------------ session generation
def test_session_cache_busts_on_scene_generation_bump(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl.sc, "scene_path", lambda: "")
    monkeypatch.setattr(ctl.cfgmod, "sessions_dir", lambda: str(tmp_path))
    gen = {"v": 0}
    monkeypatch.setattr(ctl.sc, "scene_generation", lambda: gen["v"])
    c = ctl.Controller(cfgmod.Config())
    s1 = c.session
    s1.entry("CamA").reference = "a.jpg"
    assert c.session is s1                                # stable within a scene
    gen["v"] += 1                                         # File > New happened
    s2 = c.session
    assert s2 is not s1
    assert not s2.cameras                                 # scene B starts clean
