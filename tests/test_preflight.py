"""I4 — preflight the scene before spending anything.

Every test here is a thing that was true of the TULA scene on 2026-07-30 and that nothing
in the plugin looked at. The headline one is DR_DEAD_PORT: two property reads and one
localhost socket connect, and it is the whole of that day.
"""

import os
import socket
import types

import pytest

from maxgaffer.core.errors import PreflightBlocked
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import preflight as pf


# ------------------------------------------------------------------ scaffolding
class _Node:
    def __init__(self, name, props=None, target=None):
        self.name = name
        self._props = dict(props or {})
        self.target = target


class _Renderer:
    def __init__(self, **props):
        self._props = dict(props)


def _rt_for(props_by_obj, renderer, valid_nodes=()):
    """A pymxs stand-in that answers only what preflight actually asks it."""
    class _RT:
        class renderers:
            current = renderer

        @staticmethod
        def isProperty(obj, name):
            return str(name) in props_by_obj.get(id(obj), {})

        @staticmethod
        def Name(n):
            return n

        @staticmethod
        def classOf(obj):
            return "V_Ray_7__update_2"

        @staticmethod
        def getRendType():
            return "#view"

        @staticmethod
        def isValidNode(node):
            return node in valid_nodes

    return _RT


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A clean scene: one enabled sun, DR off, a readable reference, a writable run dir."""
    sun = _Node("SUN SCENE 04 shot02")
    renderer = _Renderer()
    props = {id(sun): {"enabled": True}, id(renderer): {}}
    monkeypatch.setattr(pf.sc, "_rt", lambda: _rt_for(props, renderer))
    monkeypatch.setattr(pf.sc, "get_prop",
                        lambda obj, names, default=None: next(
                            (props.get(id(obj), {})[n] for n in names
                             if n in props.get(id(obj), {})), default))
    monkeypatch.setattr(pf.sc, "matched_prop",
                        lambda obj, names: next((n for n in names
                                                 if n in props.get(id(obj), {})), None))
    monkeypatch.setattr(pf.sc, "list_cameras",
                        lambda: [{"name": "SCENE 04 shot02", "duplicate": False}])
    monkeypatch.setattr(pf.sc, "active_camera_identity", lambda: "cam-1")
    monkeypatch.setattr(pf.sc, "active_camera_name", lambda: "SCENE 04 shot02")
    monkeypatch.setattr(pf.sc, "metres_from_world_units", lambda v: v / 100.0)
    # isolate from whatever Vantage is or is not doing on the machine running the suite
    monkeypatch.setattr(pf.vt, "link_running", lambda *a, **k: None)
    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"\xff\xd8\xff\xe0reference")
    ctrl = types.SimpleNamespace(
        _set_active_camera=lambda name: True,
        _black_probe_message=lambda head="": head + ". DIAGNOSIS SENTENCE.")
    return types.SimpleNamespace(
        ctrl=ctrl, sun=sun, renderer=renderer, props=props, tmp=tmp_path,
        rig={"sun": sun, "suns": [sun], "notes": []},
        cfg=cfgmod.Config(auto_exposure_control=False),
        ref=str(ref),
        ref_stats={"hot_frac": 0.04, "contrast": 0.6, "mean_rgb": [0.3] * 3,
                   "p": {"95": 0.9, "5": 0.1}, "count": 65536})


def _run(ctx, log=None, **over):
    monkey = dict(camera_name="SCENE 04 shot02", cam=object(), rig=ctx.rig,
                  entry=None, cfg=ctx.cfg, ref_path=ctx.ref, ref_stats=ctx.ref_stats,
                  run_dir=str(ctx.tmp))
    monkey.update(over)
    return pf.run(ctx.ctrl, monkey.pop("camera_name"), monkey.pop("cam"),
                  monkey.pop("rig"), monkey.pop("entry"), monkey.pop("cfg"),
                  log if log is not None else (lambda _m: None), **monkey)


# ------------------------------------------------------------------ DR_DEAD_PORT
def test_dr_on_with_a_dead_port_blocks(ctx, monkeypatch):
    """2026-07-30, entire. A dead Chaos Vantage live link left distributed_rendering ON,
    pointed at 127.0.0.1:20701 with the local machine excluded, and V-Ray rendered 100%
    black for hours. Two property reads and one refused connect."""
    ctx.props[id(ctx.renderer)]["distributed_rendering"] = True
    ctx.props[id(ctx.renderer)]["distributed_rendering_hosts"] = "127.0.0.1:20701"
    ctx.props[id(ctx.renderer)]["distributed_rendering_useLocalMachine"] = False
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
    with pytest.raises(PreflightBlocked):
        _run(ctx)
    report = _run(ctx, **{"cfg": cfgmod.Config(preflight_level="warn",
                                               auto_exposure_control=False)})
    dr = report.find("DR_DEAD_PORT")
    assert dr.severity == "block"
    assert "127.0.0.1:20701" in dr.detail
    assert "local machine is excluded" in dr.detail
    # ONE wording of this diagnosis, not two: it comes from _black_probe_message
    assert "DIAGNOSIS SENTENCE." in dr.detail


def test_dr_on_with_a_live_port_is_info(ctx):
    """A real listener on a real ephemeral port — no mocking of the socket layer."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        ctx.props[id(ctx.renderer)]["distributed_rendering"] = True
        ctx.props[id(ctx.renderer)]["distributed_rendering_hosts"] = \
            "127.0.0.1:%d" % srv.getsockname()[1]
        dr = _run(ctx).find("DR_DEAD_PORT")
        assert dr.severity == "info"
        assert "answered" in dr.detail
    finally:
        srv.close()


def test_dr_off_costs_one_property_read_and_no_socket(ctx, monkeypatch):
    """The check must be free on the 99% of scenes where DR is simply off."""
    # the live-link probe is a DIFFERENT check (GPU_VANTAGE_CONFLICT) and is cached once
    # per run, so silence it here to isolate the claim being made about DR
    monkeypatch.setattr(pf.vt, "link_running", lambda *a, **k: None)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail("no socket may be opened"))
    dr = _run(ctx).find("DR_DEAD_PORT")
    assert dr.severity == "info"


def test_the_dr_host_parser_survives_every_spelling():
    assert pf._parse_dr_hosts("127.0.0.1:20701") == [("127.0.0.1", 20701)]
    assert pf._parse_dr_hosts(["10.0.0.4:1234", "10.0.0.5"]) == [
        ("10.0.0.4", 1234), ("10.0.0.5", pf.VRAY_DR_PORT)]
    assert pf._parse_dr_hosts("a:1, b:2; c:3") == [("a", 1), ("b", 2), ("c", 3)]
    assert pf._parse_dr_hosts(None) == []


def test_a_portless_host_is_probed_on_the_dr_spawner_port_not_the_live_link_one():
    """The 2026-07-31 false positive. A studio spells its farm as bare addresses; those
    nodes listen on V-Ray's DR spawner port (20204), never on Chaos Vantage's live-link
    port. Probing 20701 found nothing, and DR_DEAD_PORT is a BLOCK — a healthy farm was
    locked out of the tool by the fix for a dead one."""
    assert pf.VRAY_DR_PORT not in pf.vt.LIVE_LINK_PORTS
    hosts = pf._parse_dr_hosts("192.168.1.50 192.168.1.51")
    assert hosts == [("192.168.1.50", pf.VRAY_DR_PORT),
                     ("192.168.1.51", pf.VRAY_DR_PORT)]


def test_a_healthy_portless_dr_farm_is_info_not_block(ctx, monkeypatch):
    """End to end through the check: a real listener on the DR spawner port, reached by a
    host string that names no port at all, must not block the run."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    monkeypatch.setattr(pf, "VRAY_DR_PORT", port)
    try:
        ctx.props[id(ctx.renderer)]["distributed_rendering"] = True
        ctx.props[id(ctx.renderer)]["distributed_rendering_hosts"] = "127.0.0.1"
        report = _run(ctx)
        dr = report.find("DR_DEAD_PORT")
        assert dr.severity == "info" and "answered" in dr.detail
        assert report.blocked() == []
    finally:
        srv.close()


def test_a_remote_dr_host_is_given_longer_than_a_loopback_one():
    """A loaded LAN node that completes its handshake in 300 ms is not a dead node, and
    0.25 s of patience would have blocked the run on it."""
    assert pf._dr_timeout("127.0.0.1") == pf.DR_CONNECT_TIMEOUT_S
    assert pf._dr_timeout("localhost") == pf.DR_CONNECT_TIMEOUT_S
    assert pf._dr_timeout("192.168.1.50") == pf.DR_REMOTE_TIMEOUT_S
    assert pf.DR_REMOTE_TIMEOUT_S > pf.DR_CONNECT_TIMEOUT_S


def test_the_dr_connect_timeout_used_is_the_one_for_that_host(ctx, monkeypatch):
    seen = []

    def _conn(addr, timeout=None):
        seen.append((addr[0], timeout))
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", _conn)
    ctx.props[id(ctx.renderer)]["distributed_rendering"] = True
    ctx.props[id(ctx.renderer)]["distributed_rendering_hosts"] = "127.0.0.1 10.0.0.9"
    pf.check_dr_dead_port(ctx)
    assert seen == [("127.0.0.1", pf.DR_CONNECT_TIMEOUT_S),
                    ("10.0.0.9", pf.DR_REMOTE_TIMEOUT_S)]


# ------------------------------------------------------------------ the reference
def test_a_missing_reference_blocks(ctx):
    with pytest.raises(PreflightBlocked, match="reference file is gone"):
        _run(ctx, ref_path=str(ctx.tmp / "gone.jpg"))


def test_a_black_reference_blocks(ctx):
    with pytest.raises(PreflightBlocked, match="100% BLACK"):
        _run(ctx, ref_stats={"mean_rgb": [0.0, 0.0, 0.0], "p": {"95": 0.0},
                             "contrast": 0.0, "hot_frac": 0.0})


def test_an_undecodable_reference_warns_rather_than_blocks(ctx):
    """DEVIATION from the plan, and a deliberate one: run_match has a designed degraded
    mode for exactly this (ref_stats None turns the analytic solver off and judges the
    render visually through the model). Blocking would delete a supported feature rather
    than prevent a non-measurement."""
    report = _run(ctx, ref_stats=None)
    assert report.find("REFERENCE_READABLE").severity == "warn"
    assert report.blocked() == []


def test_a_sunless_reference_predicts_the_skipped_solve(ctx):
    """sunsolve returns None on hot_frac == 0. Say so now, not 44 probes later."""
    f = _run(ctx, ref_stats=dict(ctx.ref_stats, hot_frac=0.0)).find("REFERENCE_HAS_SUN")
    assert f.severity == "warn" and "SKIPPED" in f.detail


# ------------------------------------------------------------------ the sun family
def test_the_driven_sun_being_off_is_named_along_with_the_one_that_is_on(ctx):
    """classify_rig never reads LIGHT_ON. On TULA the ONLY enabled sun of twelve was
    authored for a different shot at the other end of the site, and the rig steered it
    for hours."""
    driven = _Node("SUN SCENE 04 shot02")
    other = _Node("SUN SHOT 01 ENTRANCE")
    ctx.props[id(driven)] = {"enabled": False}
    ctx.props[id(other)] = {"enabled": True}
    rig = {"sun": driven, "suns": [driven, other], "notes": []}
    f = _run(ctx, rig=rig).find("SUN_ENABLED")
    assert f.severity == "warn"
    assert "SUN SCENE 04 shot02" in f.detail and "SUN SHOT 01 ENTRANCE" in f.detail


def test_sun_name_match_declines_to_guess_on_the_real_tula_names(ctx):
    """The four real names. Camera 'SCENE 04 shot02' → {scene,4,shot,2}; the enabled sun
    'SUN SHOT 01 ENTRANCE' shares only {shot} → 0.25, which is under the 0.5 bar, and NO
    other sun scores higher. So nothing is proposed and the bare fact is reported —
    exactly the sentence the artist needed, with zero false-positive risk."""
    driven = _Node("SUN SHOT 01 ENTRANCE")
    others = [_Node(n) for n in ("SUN SHOT 07 PLAZA", "SUN SHOT 12 REGIONAL PARK")]
    for n in [driven] + others:
        ctx.props[id(n)] = {"enabled": True}
    rig = {"sun": driven, "suns": [driven] + others, "notes": []}
    f = _run(ctx, rig=rig).find("SUN_NAME_MATCH")
    assert f.severity == "warn"
    assert "25%" in f.detail
    assert "nothing is proposed" in f.detail
    for other in others:
        assert other.name not in f.detail


def test_sun_name_match_proposes_only_a_strict_winner_over_half(ctx):
    driven = _Node("SUN SHOT 01 ENTRANCE")
    right = _Node("SUN SCENE 04 shot02")
    for n in (driven, right):
        ctx.props[id(n)] = {"enabled": True}
    rig = {"sun": driven, "suns": [driven, right], "notes": []}
    f = _run(ctx, rig=rig).find("SUN_NAME_MATCH")
    assert f.severity == "warn"
    assert "SUN SCENE 04 shot02" in f.detail and "100%" in f.detail


def test_a_project_that_names_its_suns_by_mood_is_never_locked_out(ctx):
    """An unrelated naming scheme produces all-zero scores and no strict winner. This is
    why the check is relative and why it declines to fuse with the other three."""
    driven = _Node("MoodyGolden")
    other = _Node("ColdMorningFeel")
    for n in (driven, other):
        ctx.props[id(n)] = {"enabled": True}
    rig = {"sun": driven, "suns": [driven, other], "notes": []}
    f = _run(ctx, rig=rig).find("SUN_NAME_MATCH")
    assert other.name not in f.detail
    assert f.severity == "warn" and "0%" in f.detail


def test_untargeted_suns_are_excluded_from_geometry(ctx, monkeypatch):
    """_sun_pivot returns Point3(0,0,0) for a Sun Positioner or an untargeted VRaySun, and
    comparing a world-origin placeholder against a camera 400 m away is a guaranteed
    false positive."""
    from maxgaffer.maxbridge import digest as dg

    monkeypatch.setattr(dg, "camera_basis",
                        lambda cam: {"pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0,
                                     "look": [40000.0, 0.0, 0.0]})
    driven = _Node("SunA", target=None)
    other = _Node("SunB", target=None)
    rig = {"sun": driven, "suns": [driven, other], "notes": []}
    f = _run(ctx, rig=rig).find("SUN_GEOMETRY")
    assert f.severity == "info"
    assert "untargeted" in f.detail


def test_rig_notes_reach_the_artist(ctx):
    """classify_rig writes them; they went ONLY into two LLM prompts. 'this sun aims by
    node rotation so azimuth writes will not re-aim it' is not a prompt detail."""
    rig = dict(ctx.rig, notes=["sun 'S' has no target — an untargeted VRaySun aims by "
                               "node rotation"])
    f = _run(ctx, rig=rig).find("RIG_NOTES")
    assert f.severity == "warn" and "node rotation" in f.detail


# ------------------------------------------------------------------ VANTAGE_ARMED
def _vantage_ctx(ctx, monkeypatch, frames, camera="SCENE 04 shot02", cls="V_Ray_7"):
    """Arm everything the check needs and drive it off a list of grab signatures.

    ``frames`` is one signature per accepted grab; ``None`` means the grab is refused.
    """
    from maxgaffer.maxbridge import vgrab

    ctx.cfg = cfgmod.Config(probe_backend="vantage", auto_exposure_control=False)
    monkeypatch.setattr(pf.vt, "link_running", lambda *a, **k: 20701)
    monkeypatch.setattr(pf, "_renderer_class", lambda _c: cls)
    monkeypatch.setattr(pf.sc, "active_camera_name", lambda: camera)
    monkeypatch.setattr(vgrab, "find_window", lambda *a, **k: 4242)
    monkeypatch.setattr(vgrab, "_client_rect", lambda h: (0, 0, 1600, 900))
    monkeypatch.setattr(vgrab, "last_error", lambda: "stubbed")
    seq = list(frames)
    state = {"sig": None}

    def _grab(title, out, w, h, should_cancel=None):
        sig = seq.pop(0) if seq else None
        state["sig"] = sig
        return None if sig is None else out

    monkeypatch.setattr(vgrab, "capture_window_png", _grab)
    monkeypatch.setattr(vgrab, "last_signature", lambda: state["sig"])
    monkeypatch.setattr(vgrab, "reset_settle", lambda: None)
    return ctx


def test_a_settled_vantage_window_arms_the_backend(ctx, monkeypatch):
    _vantage_ctx(ctx, monkeypatch, [[10.0] * 4, [10.0] * 4])
    report = _run(ctx)
    armed = report.find("VANTAGE_ARMED")
    assert armed.severity == "info" and "armed" in armed.detail
    assert report.demotions["probe_backend"] == "vantage"


def test_a_window_still_ingesting_is_refused_by_the_second_grab(ctx, monkeypatch):
    """The docstring promised two grabs half a second apart with NO apply between them
    and the body took ONE and checked only that it returned a path — which is exactly what
    a scene mid-ingest also does. 44 angles then get ranked across 44 different pictures:
    not black, not occluded, not stale, no symptom at all. (2026-07-31)"""
    _vantage_ctx(ctx, monkeypatch, [[10.0] * 4, [90.0] * 4])
    report = _run(ctx)
    armed = report.find("VANTAGE_ARMED")
    assert armed.severity == "warn"
    assert "still ingesting" in armed.detail
    assert report.demotions["probe_backend"] == "vray"


def test_a_wrong_camera_in_the_viewport_disarms_the_grab(ctx, monkeypatch):
    """The guard was INVERTED: `not active_camera_identity() and name != active_name()`.
    active_camera_identity() is non-empty whenever ANY camera is the viewport camera, so
    the guard could only fire when the viewport was on no camera at all — never in the
    case its own sentence describes."""
    _vantage_ctx(ctx, monkeypatch, [[10.0] * 4, [10.0] * 4], camera="SCENE 01 shot01")
    report = _run(ctx)
    armed = report.find("VANTAGE_ARMED")
    assert armed.severity == "warn"
    assert "different shot" in armed.detail
    assert report.demotions["probe_backend"] == "vray"


def test_a_gpu_renderer_never_arms_the_grab(ctx, monkeypatch):
    _vantage_ctx(ctx, monkeypatch, [[10.0] * 4, [10.0] * 4], cls="V_Ray_GPU")
    assert _run(ctx).demotions["probe_backend"] == "vray"


def test_the_backend_decision_is_recorded_in_BOTH_directions(ctx, monkeypatch):
    """The controller reads ``demotions``, so an ABSENT key has to mean "nobody checked"
    and never "go ahead". preflight_level='off' produces exactly that absence."""
    _vantage_ctx(ctx, monkeypatch, [[10.0] * 4, [10.0] * 4])
    assert "probe_backend" in _run(ctx).demotions
    off = _run(ctx, cfg=cfgmod.Config(probe_backend="vantage", preflight_level="off",
                                      auto_exposure_control=False))
    assert off.demotions == {} and off.findings == []


def test_a_raising_vantage_check_is_not_an_arming(ctx, monkeypatch):
    """`_check` turns a raise into an "info" finding, and "info" is what an armed backend
    also looks like. ``verified`` is what tells them apart."""
    _vantage_ctx(ctx, monkeypatch, [[10.0] * 4, [10.0] * 4])
    monkeypatch.setattr(pf, "check_vantage_armed",
                        lambda _c: (_ for _ in ()).throw(RuntimeError("interface moved")))
    monkeypatch.setattr(pf, "CHECKS", tuple(
        (k, pf.check_vantage_armed if k == "VANTAGE_ARMED" else f) for k, f in pf.CHECKS))
    report = _run(ctx)
    armed = report.find("VANTAGE_ARMED")
    assert armed.severity == "info" and armed.verified is False
    assert report.demotions["probe_backend"] == "vray"


def test_a_check_that_ran_is_marked_verified(ctx):
    assert all(f.verified for f in _run(ctx).findings)
    assert pf._check("X", lambda _c: None, None).verified is False


# ------------------------------------------------------------------ structure
def test_every_check_reports_on_both_outcomes(ctx):
    """The structural guarantee, enforced by a test rather than by discipline: no check
    may return None, and every one must produce a Finding on a clean scene."""
    report = _run(ctx)
    assert len(report.findings) == len(pf.CHECKS)
    keys = [f.key for f in report.findings]
    for key, _fn in pf.CHECKS:
        assert key in keys, key
    for f in report.findings:
        assert f.detail.strip(), f.key
        assert f.severity in ("info", "warn", "block")


def test_a_raising_check_reports_could_not_verify_never_ok(ctx):
    """vgrab._occluded_fraction's own rule. A check that cannot run has not passed."""
    def boom(_ctx):
        raise RuntimeError("the interface moved on this build")

    f = pf._check("SOME_CHECK", boom, None)
    assert f.severity == "info"
    assert "could not verify" in f.detail
    assert "the interface moved on this build" in f.detail


def test_a_check_that_returns_nothing_is_treated_as_a_failure_to_verify():
    f = pf._check("SILENT", lambda _c: None, None)
    assert "could not verify" in f.detail


def test_preflight_performs_no_renders_and_no_mutations(ctx, monkeypatch):
    """The contract. One balanced EXPOSURE_ROUNDTRIP aside, nothing is written and no
    frame is rendered — which is what makes it safe to run on every match."""
    from maxgaffer.maxbridge import render as rd

    monkeypatch.setattr(rd, "render_frame",
                        lambda *a, **k: pytest.fail("preflight may not render"))
    monkeypatch.setattr(pf.sc, "set_prop",
                        lambda *a, **k: pytest.fail("preflight may not mutate"))
    evs = []

    class _Host:
        kind = "vray_physical"

        def __init__(self, cam):
            pass

        def read_ev(self):
            return evs[-1] if evs else 10.0

        def write_ev(self, v):
            evs.append(v)
            return True

    import maxgaffer.maxbridge.exposure as exmod

    monkeypatch.setattr(exmod, "ExposureHost", _Host)
    report = _run(ctx)
    assert report.find("EXPOSURE_ROUNDTRIP").severity == "info"
    assert evs == [12.0, 10.0], "the EV round-trip must be balanced"


def test_a_refusing_exposure_host_is_a_warning_not_a_silence(ctx, monkeypatch):
    class _Host:
        kind = "vray_physical"

        def __init__(self, cam):
            pass

        def read_ev(self):
            return 10.0

        def write_ev(self, v):
            return False

    import maxgaffer.maxbridge.exposure as exmod

    monkeypatch.setattr(exmod, "ExposureHost", _Host)
    f = _run(ctx).find("EXPOSURE_ROUNDTRIP")
    assert f.severity == "warn" and "REFUSED" in f.detail


def test_an_unwritable_run_dir_blocks(ctx):
    with pytest.raises(PreflightBlocked, match="not writable"):
        _run(ctx, run_dir=str(ctx.tmp / "does" / "not" / "exist"))


def test_preflight_level_warn_downgrades_and_says_so(ctx, monkeypatch):
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
    ctx.props[id(ctx.renderer)]["distributed_rendering"] = True
    logs = []
    report = _run(ctx, log=logs.append,
                  cfg=cfgmod.Config(preflight_level="warn", auto_exposure_control=False))
    assert report.blocked()          # the finding is still a block…
    assert any("downgraded to warnings" in ln for ln in logs)   # …and it says so


def test_preflight_level_off_skips_and_says_so(ctx):
    logs = []
    report = _run(ctx, log=logs.append,
                  cfg=cfgmod.Config(preflight_level="off", auto_exposure_control=False))
    assert report.findings == []
    assert any("OFF" in ln and "nothing about this scene was verified" in ln
               for ln in logs)


def test_a_clean_run_collapses_to_one_summary_line(ctx):
    """Every check reports, but eighteen clean ones must not bury the two that matter."""
    logs = []
    _run(ctx, log=logs.append)
    assert any(ln.startswith("preflight: ") and "clean in" in ln for ln in logs)


def test_the_draft_cap_truncation_is_predicted_before_the_run(ctx):
    """G-10. probe_max_seconds 110 → int(1.833) → 1 → a 60 s cap: a 45% shortfall the
    2026-07-31 sub-minute fix does not catch because it only refuses truncation to ZERO."""
    ctx.props[id(ctx.renderer)]["progressive_max_render_time"] = 0    # an INT field
    cfg = cfgmod.Config(draft_sampler=True, probe_max_seconds=110.0,
                        auto_exposure_control=False)
    f = _run(ctx, cfg=cfg).find("DRAFT_CAP_BINDS")
    assert f.severity == "warn"
    assert "60" in f.detail and "110" in f.detail


def test_a_configured_cap_with_draft_off_says_it_will_not_bind(ctx):
    cfg = cfgmod.Config(draft_sampler=False, probe_max_seconds=20.0,
                        auto_exposure_control=False)
    f = _run(ctx, cfg=cfg).find("DRAFT_CAP_BINDS")
    assert f.severity == "warn" and "NOT be in effect" in f.detail


# ------------------------------------------------------------------ name tokens (pure)
@pytest.mark.parametrize("name,expected", [
    ("SCENE 04 shot02", {"scene", "4", "shot", "2"}),
    ("SUN SHOT 01 ENTRANCE", {"shot", "1", "entrance"}),
    # "vraysun" is a stopword and the letter/digit boundary strips the zeros
    ("VRaySun001", {"1"}),
    ("cam_A-07", {"a", "7"}),
])
def test_name_tokens(name, expected):
    assert pf.name_tokens(name) == expected


def test_name_overlap_is_relative_to_the_camera():
    """Absolute counts would rank a sun with a long name above a correctly-matched short
    one. The question is 'does this sun name this camera', so normalise on the camera."""
    assert pf.name_overlap("SUN SCENE 04 shot02", "SCENE 04 shot02") == 1.0
    assert pf.name_overlap("SUN SHOT 01 ENTRANCE", "SCENE 04 shot02") == 0.25
    assert pf.name_overlap("anything", "") == 0.0
