"""Round-2 stress-test regressions — every test here encodes an attack that landed."""

import json

from maxgaffer.core.director import Hooks, MatchConfig, run_match
from maxgaffer.core.genome import LightingState
from maxgaffer.core.session import Session

REF = {"log_key": 0.20, "lab_mean": [55.0, 2.0, 12.0], "lab_std": [20, 4, 6],
       "p": {"5": 0.03, "25": 0.2, "50": 0.45, "75": 0.7, "95": 0.92},
       "contrast": 0.89,
       "lum_hist": [0.0] * 10 + [0.5, 0.5] + [0.0] * 20,
       "hue_hist": [0.6, 0.4] + [0.0] * 10}


def near(log_key):
    s = dict(REF)
    s["log_key"] = log_key
    return s


def start_state(ev=12.0):
    st = LightingState()
    # a real sun rig always carries turbidity/size — read_state includes every
    # rig-supported param, and apply_changes now capability-gates on exactly that
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.altitude_deg": 30.0,
                 "sun.intensity": 1.0, "sun.size": 3.0, "sun.turbidity": 3.0,
                 "exposure.ev": ev, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    return st


def hooks_with(stats_seq, llm_replies, logs):
    def llm(ctx):
        return llm_replies.pop(0) if llm_replies else json.dumps(
            {"assessment": "", "changes": [], "stop": False})

    return Hooks(apply=lambda st: None,
                 render=lambda tag: f"/tmp/{tag}.png",
                 stats=lambda p: stats_seq.pop(0) if stats_seq else dict(REF),
                 llm_deltas=llm, log=logs.append)


# --------------------------------------------------------------- baseline poisoning (C14)
def test_adopt_baselines_never_overwrites():
    s = Session()
    added = s.adopt_baselines({"Spot_A": 30.0, "Spot_B": 12.0})
    assert sorted(added) == ["Spot_A", "Spot_B"]
    # MaxGaffer dims the group to 0, a rescan reads 0 — adoption must refuse the poison
    added2 = s.adopt_baselines({"Spot_A": 0.0, "Spot_C": 5.0})
    assert added2 == ["Spot_C"]
    assert s.baselines["Spot_A"] == 30.0
    s.forget_baseline("Spot_A")            # explicit re-author path
    assert s.adopt_baselines({"Spot_A": 45.0}) == ["Spot_A"]
    assert s.baselines["Spot_A"] == 45.0


def test_baselines_and_pre_match_persist(tmp_path):
    p = str(tmp_path / "scene.maxgaffer.json")
    s = Session(p, now_fn=lambda: "t")
    s.adopt_baselines({"Spot_A": 30.0})
    pre = LightingState()
    pre.set("sun.altitude_deg", 45.0)
    s.entry("cam").pre_match = pre
    assert s.save()
    s2 = Session.load(p)
    assert s2.baselines == {"Spot_A": 30.0}
    assert s2.entry("cam").pre_match.get("sun.altitude_deg") == 45.0
    # junk baselines in a hand-edited file don't crash the load
    (tmp_path / "bad.json").write_text(json.dumps(
        {"baselines": {"X": "not a number", "Y": 2.0}}))
    s3 = Session.load(str(tmp_path / "bad.json"))
    assert s3.baselines == {"Y": 2.0}


# --------------------------------------------------------------- analytic leash (A1/A2)
def test_ev_leash_bounds_total_analytic_movement():
    # every render reads 3+ stops darker than the ref (albedo trap) — without the leash the
    # solver would walk EV down 2.5/iteration forever; with it, total movement caps at 4
    logs = []
    hooks = hooks_with([near(0.01) for _ in range(8)], [], logs)
    res = run_match(start_state(ev=12.0), REF, {}, hooks,
                    MatchConfig(max_iterations=8, target_score=101, stall_patience=99,
                                ev_leash=4.0))
    evs = [r.analytic_changes.get("exposure.ev") for r in res.iterations
           if r.analytic_changes]
    assert evs and min(evs) >= 12.0 - 4.0 - 1e-6
    assert any("leash" in m for m in logs)
    assert any("albedo" in m for m in logs)      # the diagnosis fires after 2+ hits


def test_wb_leash():
    warm_ref = dict(REF)
    warm_ref["lab_mean"] = [55.0, 2.0, 60.0]     # absurdly warm reference
    logs = []
    hooks = hooks_with([dict(REF) for _ in range(6)], [], logs)
    res = run_match(start_state(), warm_ref, {}, hooks,
                    MatchConfig(max_iterations=6, target_score=101, stall_patience=99,
                                wb_leash=3000.0))
    wbs = [r.analytic_changes.get("exposure.wb_kelvin") for r in res.iterations
           if "exposure.wb_kelvin" in r.analytic_changes]
    assert wbs and max(wbs) <= 6500.0 + 3000.0 + 1e-6


# --------------------------------------------------------------- contamination guard (B2)
def test_llm_intensity_changes_dropped_on_misexposed_frame():
    # render 2.5 stops dark → solver slams EV; the LLM (which saw the dark frame) tries to
    # crank sun intensity and practicals — those must be dropped; geometry must survive
    reply = json.dumps({"assessment": "too dark", "changes": [
        {"param": "sun.intensity", "value": 8.0, "why": "brighten"},
        {"param": "group.practicals", "value": 4.0, "why": "brighten"},
        {"param": "sun.azimuth_deg", "value": 140.0, "why": "shadow direction"}],
        "stop": False})
    st = start_state()
    st.groups["practicals"] = 1.0
    logs = []
    hooks = hooks_with([near(0.03), near(0.18)], [reply], logs)
    res = run_match(st, REF, {}, hooks,
                    MatchConfig(max_iterations=2, target_score=101, stall_patience=99))
    it0 = res.iterations[0]
    assert "sun.intensity" not in it0.llm_accepted
    assert "group.practicals" not in it0.llm_accepted
    assert "sun.azimuth_deg" in it0.llm_accepted
    assert any("contaminated" in r for r in it0.llm_rejected)


def test_small_ev_fix_does_not_trigger_guard():
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "sun.intensity", "value": 1.5, "why": "ratio"}], "stop": False})
    logs = []
    hooks = hooks_with([near(0.17), near(0.18)], [reply], logs)   # ~0.23 stops off
    res = run_match(start_state(), REF, {}, hooks,
                    MatchConfig(max_iterations=2, target_score=101, stall_patience=99))
    assert "sun.intensity" in res.iterations[0].llm_accepted


# --------------------------------------------------------------- center-weighted key (A1)
def test_center_weighted_key(tmp_path):
    import pytest

    PIL = pytest.importorskip("PIL.Image")
    from maxgaffer.core import metrics

    def painter(bright_center):
        def f(x, y):
            in_center = 24 <= x < 72 and 16 <= y < 48
            return (230, 230, 230) if (in_center == bright_center) else (25, 25, 25)
        return f

    def make(name, fn):
        im = PIL.new("RGB", (96, 64))
        px = im.load()
        for y in range(64):
            for x in range(96):
                px[x, y] = fn(x, y)
        p = str(tmp_path / name)
        im.save(p)
        return p

    bright_c = metrics.compute_stats(make("bc.png", painter(True)))
    dark_c = metrics.compute_stats(make("dc.png", painter(False)))
    # same pixel population, opposite placement — the center-weighted key must differ,
    # brighter-center strictly higher
    assert bright_c["log_key"] > dark_c["log_key"] * 1.5


# --------------------------------------------------------------- run-dir pruning (v0.2)
def test_prune_old_runs(tmp_path):
    from maxgaffer.maxbridge.controller import prune_old_runs

    for stamp in ("20260701-090000", "20260702-090000", "20260703-090000",
                  "20260704-090000", "20260705-090000"):
        (tmp_path / stamp).mkdir()
    (tmp_path / "not_a_dir.txt").write_text("x")
    assert prune_old_runs(str(tmp_path), keep=2) == 3
    left = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert left == ["20260704-090000", "20260705-090000"]     # newest survive
    assert prune_old_runs(str(tmp_path), keep=0) == 0          # 0 = keep everything
    assert prune_old_runs(str(tmp_path / "missing"), keep=2) == 0


# --------------------------------------------------------------- analytic ownership (live-fire find)
def test_llm_cannot_override_analytic_params_when_solver_active():
    """Live sim showed the model setting exposure.ev despite prompt guidance — with the
    solver active, analytic params must be structurally refused."""
    reply = json.dumps({"assessment": "too dark", "changes": [
        {"param": "exposure.ev", "value": 9.0, "why": "brighten"},
        {"param": "exposure.wb_kelvin", "value": 4000.0, "why": "warm"},
        {"param": "sun.turbidity", "value": 5.0, "why": "haze"}], "stop": False})
    logs = []
    hooks = hooks_with([near(0.18), near(0.18)], [reply], logs)
    res = run_match(start_state(), REF, {}, hooks,
                    MatchConfig(max_iterations=2, target_score=101, stall_patience=99))
    it0 = res.iterations[0]
    assert "exposure.ev" not in it0.llm_accepted
    assert "exposure.wb_kelvin" not in it0.llm_accepted
    assert "sun.turbidity" in it0.llm_accepted
    assert sum("solver owns it" in r for r in it0.llm_rejected) == 2


def test_llm_may_drive_exposure_in_llm_only_mode():
    """No metrics → no solver → the model is the only exposure driver; must be allowed."""
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "exposure.ev", "value": 11.0, "why": "brighten"}], "stop": False})
    logs = []
    hooks = hooks_with([], [reply], logs)
    res = run_match(start_state(), None, {}, hooks, MatchConfig(max_iterations=2))
    assert "exposure.ev" in res.iterations[0].llm_accepted
