"""Cluster H regressions — maxbridge/controller.py + api.py audit fixes.

Off-Max, pure python: pymxs is stubbed in sys.modules where the bridge needs it;
scene/apply/render collaborators are monkeypatched on the controller module (the
same pattern as test_recheck.py and test_kimi_render_vantage.py).

Covered findings (audit_briefs/H_controller_api.txt):
  P0 draft-sampler restore window (sweep / MatchConfig raises still restore the sampler)
  P1 seed-only flow restorable · refine() keeps the artist's pre-match · scenario board
     restores the found state on exception · reference-swap TOCTOU · seed fingerprint
     includes cam yaw + semantics
  P2 note length cap · config_overrides type check · loud save failure · EC leak past
     Restore · stale analyze-agreement · _safe reserved device names + makedirs guard ·
     finals/exports restore the found light · refs/ pruning
"""

import contextlib
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from maxgaffer.core.director import MatchResult
from maxgaffer.core.genome import LightingState


# --------------------------------------------------------------------- helpers
def _pymxs(monkeypatch, rt=None):
    """A REAL module object for sys.modules — under Max's real PySide6, shiboken
    replaces builtins.__import__ with a hook that reads ``module.__name__`` on every
    import result; a bare SimpleNamespace stub makes any lazy ``import pymxs`` in
    source raise AttributeError (order-dependent: only when PySide6 was imported)."""
    fake = ModuleType("pymxs")
    fake.runtime = rt or SimpleNamespace()
    fake.undo = lambda *a, **k: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "pymxs", fake)
    return fake


def _ctrl(monkeypatch, tmp_path, **cfg_over):
    """A Controller wired for off-Max tests: real in-memory Session, tmp sessions dir."""
    from maxgaffer.maxbridge import controller as cm
    from maxgaffer.maxbridge.config import Config

    cfg = Config()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    c = cm.Controller(cfg)
    monkeypatch.setattr(cm.cfgmod, "sessions_dir", lambda: str(tmp_path))
    return c, cm


def _stub_scene(monkeypatch, cm, rig=None, cam=None, yaw=0.0):
    rig = rig if rig is not None else {"sun": None, "dome": None,
                                       "groups": {}, "notes": []}
    monkeypatch.setattr(cm.sc, "classify_rig", lambda: rig)
    monkeypatch.setattr(cm.sc, "get_camera", lambda name=None: cam or object())
    monkeypatch.setattr(cm.sc, "set_active_camera", lambda name: True)
    monkeypatch.setattr(cm.sc, "camera_yaw_deg", lambda c: yaw)
    return rig


def _stub_llm(monkeypatch, cm, semantics=None, reply="{}"):
    """analyze_reference without a gateway: image block + valid analysis on tap."""
    monkeypatch.setattr(cm.omega, "call", lambda *a, **k: reply)
    monkeypatch.setattr(cm, "validate_analysis", lambda r: {"sample": True})
    monkeypatch.setattr(cm.consensus, "consolidate_analyses",
                        lambda samples: dict(semantics or {"sem": 1}))


def _semantics():
    return {"time_of_day": "dusk", "sky": "clear", "sun_altitude_band": "low",
            "sun_bearing_deg": 15.0, "wb_kelvin_estimate": 4300.0, "key_notes": "warm"}


def test_analyze_detects_reference_overwritten_without_rebinding(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path, analyze_samples=1)
    ref = tmp_path / "reference.png"
    ref.write_bytes(b"old")
    c.session.set_reference("Cam", str(ref))
    e = c.session.entry("Cam")
    e.semantics = {"sem": "old"}
    e.score = 91.0
    ref.write_bytes(b"new image bytes")
    monkeypatch.setattr(c, "_image_block", lambda path: {"type": "image"})
    _stub_llm(monkeypatch, cm, semantics={"sem": "new"})

    assert c.analyze_reference("Cam") == {"sem": "new"}
    assert e.score is None


# --------------------------------------------------------------------- _safe / dirs
def test_safe_strips_windows_reserved_device_names():
    from maxgaffer.maxbridge.controller import _safe

    assert _safe("CON") == "_CON"
    assert _safe("con") == "_con"
    assert _safe("PRN") == "_PRN"
    assert _safe("AUX") == "_AUX"
    assert _safe("NUL") == "_NUL"
    assert _safe("COM1") == "_COM1"
    assert _safe("com7") == "_com7"
    assert _safe("LPT9") == "_LPT9"
    assert _safe("CON.txt") == "_CON.txt"      # reserved by stem, not extension
    assert _safe("console") == "console"       # a mere prefix is fine
    assert _safe("COM10") == "COM10"           # only COM1-9 are reserved
    assert _safe("trail.") == "trail"          # Win32 strips trailing dots
    assert _safe("...") == "unnamed"
    assert _safe("") == "unnamed"
    # char-rewritten names carry a stable 6-hex hash of the RAW name, so raw names
    # that sanitize to the same string ("Cam 01/hero" vs "Cam-01\hero") can no longer
    # share a run dir — the union keeps the box-day collision guard
    s = _safe("Cam 01/hero")
    assert s.startswith("Cam_01_hero_") and len(s) == len("Cam_01_hero_") + 6
    assert _safe("Cam 01/hero") == s                      # deterministic
    assert _safe("Cam-01\\hero") != s                     # collision separated


def test_prune_old_files_keeps_newest(tmp_path):
    from maxgaffer.maxbridge.controller import prune_old_files

    made = []
    for i in range(5):
        p = tmp_path / f"ref_{i}.png"
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
        made.append(p)
    (tmp_path / "sub").mkdir()                       # dirs are never touched
    assert prune_old_files(str(tmp_path), keep=2) == 3
    assert [p.name for p in sorted(tmp_path.glob("*.png"))] == ["ref_3.png", "ref_4.png"]
    assert prune_old_files(str(tmp_path), keep=0) == 0     # 0 = keep everything
    assert prune_old_files(str(tmp_path / "missing"), keep=2) == 0


def test_ensure_run_dir_falls_back_when_sessions_dir_unwritable(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.setattr(cm.cfgmod, "sessions_dir", lambda: str(blocker))
    d = c._ensure_run_dir("refs")
    assert "MaxGaffer_sessions" in d and os.path.isdir(d)


def test_transcode_ref_prunes_the_refs_cache(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path, keep_runs=2)
    refs = tmp_path / "unsaved" / "refs"
    refs.mkdir(parents=True)
    for i in range(3):                               # stale transcodes from earlier runs
        old = refs / f"ref_old{i}.png"
        old.write_bytes(b"x")
        os.utime(old, (100 + i, 100 + i))

    def fake_transcode(src, dst, max_dim=None):
        with open(dst, "wb") as f:
            f.write(b"png")
        return dst

    monkeypatch.setattr(cm.rd, "transcode_to_png", fake_transcode)
    out = c._transcode_ref("shot.exr")
    assert out and os.path.exists(out)
    assert len(list(refs.glob("*.png"))) == 2        # keep_runs policy now covers refs/


# --------------------------------------------------------------------- analyze
def test_analyze_raises_when_reference_swapped_mid_wait(monkeypatch, tmp_path):
    """The io wait is a click-window: a swap during it must not cache the OLD image's
    semantics under the NEW reference path."""
    c, cm = _ctrl(monkeypatch, tmp_path)
    c.session.set_reference("Cam", "A.png")
    c._image_block = lambda p: {"type": "image"}

    def swap_during_wait(*a, **k):
        c.session.cameras["Cam"].reference = "B.png"     # user clicked Load mid-wait
        return "{}"

    monkeypatch.setattr(cm.omega, "call", swap_during_wait)
    monkeypatch.setattr(cm, "validate_analysis", lambda r: {"sample": True})
    monkeypatch.setattr(cm.consensus, "consolidate_analyses", lambda s: {"sem": 1})
    with pytest.raises(RuntimeError, match="changed while analyzing"):
        c.analyze_reference("Cam")
    assert c.session.cameras["Cam"].semantics == {}      # stale read NOT cached


def test_analyze_agreement_flag_lifecycle(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    c.session.set_reference("CamA", "A.png")
    c._image_block = lambda p: {"type": "image"}
    _stub_llm(monkeypatch, cm, semantics={"a": 1, "consensus_agreement": 0.4})
    c.analyze_reference("CamA")
    assert c._last_analyze_agreement == 0.4              # contested read is surfaced

    c.session.cameras["CamB"] = type(c.session.cameras["CamA"])()  # fresh entry
    c.session.set_reference("CamB", "B.png")
    c.session.cameras["CamB"].semantics = {"cached": True}
    c.analyze_reference("CamB")                          # cached read
    assert c._last_analyze_agreement is None             # stale flag cleared

    c.session.set_reference("CamC", "C.png")
    c._last_analyze_agreement = 0.4                      # leftover from another camera
    _stub_llm(monkeypatch, cm, semantics={"c": 1})       # agreement defaults to 1.0
    c.analyze_reference("CamC")
    # fresh 100% read RESETS the flag — unanimous reads store None (the flag only
    # ever carries a real contest), so run_match can never log a ghost warning
    assert c._last_analyze_agreement is None


# --------------------------------------------------------------------- refine
def _refine_harness(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _stub_scene(monkeypatch, cm)
    c.session.set_reference("Cam", "ref.png")
    e = c.session.cameras["Cam"]
    e.semantics = _semantics()                           # analyze short-circuits (cached)
    c.ref_stats = lambda p: None
    c._image_block = lambda p: {"b": 1}
    c._apply_logged = lambda *a, **k: None
    monkeypatch.setattr(cm.rd, "render_frame", lambda *a, **k: None)
    monkeypatch.setattr(cm.omega, "call",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no gateway")))
    captured = {}

    def fake_run_match(cam_name, log, should_cancel=lambda: False, **kw):
        captured.update(kw)
        captured["plan_snapped"] = c._plan_snapped
        return MatchResult(best_state=LightingState(), best_score=1.0,
                           best_render=None, stop_reason="stubbed")

    c.run_match = fake_run_match
    return c, cm, e, captured


def test_refine_caps_directors_note(monkeypatch, tmp_path):
    c, cm, e, captured = _refine_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: LightingState())
    logs = []
    c.refine("Cam", "x" * 5000, logs.append)
    assert len(e.notes[-1]) == 500                       # persisted form is capped
    assert captured["director_note"] == "x" * 500        # prompt-pinned form is capped
    assert any("truncated" in m for m in logs)           # …and the truncation is told


def test_refine_snapshots_true_pre_match_before_nudges(monkeypatch, tmp_path):
    c, cm, e, captured = _refine_harness(monkeypatch, tmp_path)
    found = LightingState()
    found.set("sun.azimuth_deg", 42.0)
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: found)
    c.refine("Cam", "brighter", lambda m: None)
    assert e.pre_match is found                          # artist's light, captured up front
    assert captured["plan_snapped"] == "Cam"             # run_match told NOT to re-snapshot


def test_refine_never_overwrites_an_existing_pre_match(monkeypatch, tmp_path):
    c, cm, e, captured = _refine_harness(monkeypatch, tmp_path)
    artist = LightingState()
    artist.set("exposure.ev", 9.0)
    e.pre_match = artist                                 # from the original MATCH
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: LightingState())
    c.refine("Cam", "warmer", lambda m: None)
    assert e.pre_match is artist                         # Restore still returns A's light


# --------------------------------------------------------------------- scenario board
def test_scenarios_restore_found_state_on_exception(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _stub_scene(monkeypatch, cm)
    found = LightingState()
    found.set("sun.azimuth_deg", 10.0)
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: found)
    applied = []
    monkeypatch.setattr(cm.ap, "apply_state",
                        lambda rig, base, st, cam=None, **k: applied.append(st) or [])
    board = [{"key": k, "label": k, "why": "w", "state": LightingState()}
             for k in ("a", "b")]
    monkeypatch.setattr(cm.scen, "build_scenarios", lambda *a, **k: board)
    renders = {"n": 0}

    def die_on_second(*a, **k):
        renders["n"] += 1
        if renders["n"] == 2:
            raise RuntimeError("pymxs died mid-board")
        return None

    monkeypatch.setattr(cm.rd, "render_frame", die_on_second)
    with pytest.raises(RuntimeError, match="pymxs died"):
        c.run_scenarios("Cam", lambda m: None)
    assert applied[-1] is found          # 'leave the scene exactly as it was found'


# --------------------------------------------------------------------- restore
def test_restore_reaches_pre_seed_without_pre_match(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _pymxs(monkeypatch)
    dome = object()
    _stub_scene(monkeypatch, cm, rig={"sun": None, "dome": dome,
                                      "groups": {}, "notes": []})
    calls = []
    monkeypatch.setattr(cm.sc, "set_dome_texture",
                        lambda d, p: calls.append(("tex", p)) or "texmap")
    monkeypatch.setattr(cm.sc, "write_dome_rotation",
                        lambda d, v: calls.append(("rot", v)) or "spinner")
    e = c.session.entry("Cam")
    e.pre_match = None                                   # seed-only session, never matched
    e.pre_seed = {"file": "old.hdr", "rotation": 25.0}
    e.seed_hdri = "seed_cam_x.hdr"
    c.apply_state = lambda *a, **k: pytest.fail("no pre_match — nothing to apply")
    assert c.restore_pre_match("Cam") is True            # NOT the old silent False
    assert ("tex", "old.hdr") in calls and ("rot", 25.0) in calls
    assert e.pre_seed == {} and e.seed_hdri == ""


def test_restore_returns_false_when_nothing_to_restore(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _pymxs(monkeypatch)
    c.session.entry("Cam")
    assert c.restore_pre_match("Cam") is False


def test_restore_removes_auto_created_exposure_control(monkeypatch, tmp_path):
    rt = SimpleNamespace(SceneExposureControl=SimpleNamespace(exposureControl="EC_OBJ"),
                         undefined="UNDEFINED")
    _pymxs(monkeypatch, rt)
    c, cm = _ctrl(monkeypatch, tmp_path)
    _stub_scene(monkeypatch, cm)
    e = c.session.entry("Cam")
    e.pre_match = LightingState()
    e.ec_created = True                                  # run_match auto-created the EC
    c.apply_state = lambda *a, **k: []
    logs = []
    assert c.restore_pre_match("Cam", log=logs.append) is True
    assert rt.SceneExposureControl.exposureControl == "UNDEFINED"
    assert e.ec_created is False
    assert any("exposure control" in m for m in logs)


# --------------------------------------------------------------------- draft P0
def _match_harness(monkeypatch, tmp_path, **cfg_over):
    cfg_over.setdefault("draft_sampler", True)
    cfg_over.setdefault("auto_exposure_control", False)
    c, cm = _ctrl(monkeypatch, tmp_path, **cfg_over)
    _stub_scene(monkeypatch, cm, rig={"sun": object(), "dome": None,
                                      "groups": {}, "notes": []})
    e = c.session.entry("Cam")
    e.reference = "ref.png"
    e.semantics = _semantics()
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: LightingState())
    monkeypatch.setattr(cm.ap, "apply_state", lambda *a, **k: [])
    monkeypatch.setattr(cm.rules, "initial_state", lambda *a, **k: (LightingState(), []))
    c.ref_stats = lambda p: None
    c._image_block = lambda p: {"b": 1}
    restored = []
    monkeypatch.setattr(cm.df, "apply_draft", lambda seconds=0.0: ["draft on"])
    monkeypatch.setattr(cm.df, "pending_snapshot", lambda: True)
    monkeypatch.setattr(cm.df, "restore_draft", lambda: restored.append(1) or ["off"])
    return c, cm, restored


def test_draft_restored_when_sweep_raises(monkeypatch, tmp_path):
    c, cm, restored = _match_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(cm, "run_sun_sweep",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gateway down")))
    with pytest.raises(RuntimeError, match="gateway down"):
        c.run_match("Cam", lambda m: None, do_sweep=True)
    assert restored == [1]       # the artist's sampler survived the gateway error


def test_draft_restored_when_config_value_is_bad(monkeypatch, tmp_path):
    c, cm, restored = _match_harness(monkeypatch, tmp_path, max_iterations="five")
    with pytest.raises(ValueError):
        c.run_match("Cam", lambda m: None, do_sweep=False)   # MatchConfig build raises
    assert restored == [1]       # int() now lives INSIDE the restoring try/finally


# --------------------------------------------------------------------- finals / exports
def test_render_finals_restores_found_light(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _stub_scene(monkeypatch, cm)
    found = LightingState()
    found.set("sun.azimuth_deg", 7.0)
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: found)
    applied = []
    monkeypatch.setattr(cm.ap, "apply_state",
                        lambda rig, base, st, cam=None, **k: applied.append(st) or [])
    monkeypatch.setattr(cm.rd, "render_frame", lambda *a, **k: "frame.png")
    for name in ("CamA", "CamB"):
        c.session.entry(name).state = LightingState()
    progress = []
    res = c.render_finals_vray(["CamA", "CamB"], str(tmp_path / "out"),
                               lambda cam, s: progress.append((cam, s)))
    assert res == {"CamA": "ok", "CamB": "ok"}
    assert applied[-1] is found  # NOT left holding the last camera's light


def test_render_finals_restores_even_when_a_render_raises(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _stub_scene(monkeypatch, cm)
    found = LightingState()
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: found)
    applied = []
    monkeypatch.setattr(cm.ap, "apply_state",
                        lambda rig, base, st, cam=None, **k: applied.append(st) or [])
    monkeypatch.setattr(cm.rd, "render_frame",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vray blew up")))
    c.session.entry("CamA").state = LightingState()
    # union contract: one camera's raise is fault-isolated (recorded, batch continues)
    # AND the found light is still restored by the finally
    res = c.render_finals_vray(["CamA"], str(tmp_path / "out"), lambda c_, s: None)
    assert res["CamA"].startswith("error:") and "vray blew up" in res["CamA"]
    assert applied[-1] is found


def test_prepare_vantage_jobs_restores_found_light(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)
    _stub_scene(monkeypatch, cm)
    found = LightingState()
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: found)
    applied = []
    monkeypatch.setattr(cm.ap, "apply_state",
                        lambda rig, base, st, cam=None, **k: applied.append(st) or [])
    monkeypatch.setattr(cm.vt, "export_vrscene", lambda path, name: path)
    c.session.entry("CamA").state = LightingState()
    jobs = c.prepare_vantage_jobs(["CamA"], str(tmp_path), lambda c_, s: None)
    assert jobs and jobs[0]["camera"] == "CamA"
    assert applied[-1] is found


# --------------------------------------------------------------------- dome seed
def test_seed_token_tracks_yaw_and_semantics(monkeypatch, tmp_path):
    """Max caches bitmaps by path: a moved camera or a changed analysis must produce a
    changed seed filename, or the dome renders the previous orientation's pano."""
    c, cm = _ctrl(monkeypatch, tmp_path)
    _pymxs(monkeypatch)
    dome = object()
    yaw = {"v": 10.0}
    _stub_scene(monkeypatch, cm, rig={"sun": None, "dome": dome,
                                      "groups": {}, "notes": []})
    monkeypatch.setattr(cm.sc, "camera_yaw_deg", lambda cam: yaw["v"])
    monkeypatch.setattr(cm.sc, "get_dome_texture", lambda d: "")
    monkeypatch.setattr(cm.sc, "read_dome_rotation", lambda d: 0.0)
    monkeypatch.setattr(cm.sc, "write_dome_rotation", lambda d, v: "spinner")
    monkeypatch.setattr(cm.ap, "read_state", lambda *a, **k: LightingState())
    outs = []
    monkeypatch.setattr(cm.domeseed, "build_seed",
                        lambda out, **k: outs.append(out)
                        or {"source": "reference", "sun": None})
    c.set_dome_hdri = lambda p: "texmap.HDRIMapName"
    e = c.session.entry("Cam")
    e.reference = "ref.png"                              # missing file → stable OSError sig
    e.semantics = {"sun_bearing_deg": 30.0}
    log = lambda m: None
    c.seed_dome("Cam", log)
    first = outs[-1]
    yaw["v"] = 25.0                                      # camera moved, same reference
    c.seed_dome("Cam", log)
    assert outs[-1] != first
    e.semantics = {"sun_bearing_deg": 90.0}              # re-analyzed, same sun numbers
    c.seed_dome("Cam", log)
    assert outs[-1] != first and outs[-1] != outs[-2]


# --------------------------------------------------------------------- save warning
def test_save_or_warn_is_loud_when_persistence_is_off(monkeypatch, tmp_path):
    c, cm = _ctrl(monkeypatch, tmp_path)                 # unsaved scene → save() False
    logs = []
    c._save_or_warn(logs.append)
    assert any("NOT saved" in m for m in logs)


# --------------------------------------------------------------------- api overrides
def test_api_config_overrides_are_type_checked(monkeypatch):
    import maxgaffer.api as api
    from maxgaffer.maxbridge.config import Config

    monkeypatch.setattr(api, "_shared", None)
    monkeypatch.setattr(api._config, "load", lambda: Config())
    ctrl = api.get_controller({"loop_width": "480"})     # coercible string → int
    assert ctrl.cfg.loop_width == 480 and isinstance(ctrl.cfg.loop_width, int)
    api.get_controller({"target_score": "85.5"})
    assert ctrl.cfg.target_score == 85.5
    api.get_controller({"draft_sampler": True})
    assert ctrl.cfg.draft_sampler is True
    with pytest.raises(TypeError, match="loop_width"):
        api.get_controller({"loop_width": "wide"})
    with pytest.raises(TypeError, match="draft_sampler"):
        api.get_controller({"draft_sampler": 1})         # bool fields refuse ints
    with pytest.raises(TypeError, match="critic_weights"):
        api.get_controller({"critic_weights": "heavy"})
    api.get_controller({"no_such_key": 1})               # unknown keys stay ignored
