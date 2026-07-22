"""Off-box host-probe tests — the NON-DESTRUCTIVE census probes must never mutate
Max, never fire a render, and above all never TOGGLE the Chaos Vantage live link.

Three read-only probes run against the hostile fake ``pymxs`` runtime in
``tests/mock_pymxs.py`` — its ``mutation_log`` records every ``rt.execute`` /
``("set", …)`` the bridge causes, so "changed nothing" is an assertion, not a hope:

  * ``vantage.probe_entrypoints`` — the live-link entry-point census. The V-Ray
    "Initiate a Live-Link" action is a TOGGLE (executing it flips an active link
    OFF), so the toggle globals are EXISTENCE-tested only and the located action is
    never executed. The load-bearing guard: after every call, no live-link toggle
    global and no ``action.execute()`` appears in the execute log.
  * ``vantage.vrscene_export_available`` — reports whether the exporter global and
    the Vantage executables exist; exports nothing.
  * ``render.probe_colorspace`` — reports the colour-management mode and degrades to
    ``legacy``/``unknown`` when ColorPipelineMgr (pre-2024 Max) or pymxs is absent.

``globalVars`` / ``ColorPipelineMgr`` are composed from the mock's exported
primitives on the test side (attached to the runtime instance), so the shared mock
needs no edit. Stdlib-only, deterministic; ``link_running`` (a real socket) is always
monkeypatched so a live Vantage on the dev box can never flip a result.
"""

import sys
from types import SimpleNamespace

import pytest

from tests.mock_pymxs import FakeMaxRuntime, MockObject, install

from maxgaffer.maxbridge import render as rd
from maxgaffer.maxbridge import vantage as vt


# --------------------------------------------------------------------- probe key contracts
ENTRYPOINTS_KEYS = {"link_running_port", "link_attached", "probe_state",
                    "action_present", "action_label", "globals_present",
                    "ports", "in_max"}
EXPORT_KEYS = {"available", "method", "console_exe_present", "vantage_exe_present",
               "notes"}
COLORSPACE_KEYS = {"available", "color_management", "mode", "ocio_active",
                   "view_transform", "display_gamma", "vray_display_correction",
                   "risk"}


# --------------------------------------------------------------------- test-side pymxs surfaces
class _GlobalVars:
    """Minimal ``rt.globalVars`` stand-in: read-only ``isGlobal(name)`` over a fixed
    set of BARE global identifiers (no trailing ``()``). Lets a test prove the toggle
    globals are existence-tested — never executed — and that a state query is gated on
    its own existence before its (safe) read."""

    def __init__(self, names=()):
        self._names = {str(n) for n in names}

    def isGlobal(self, name):
        return str(name) in self._names


class _FakeAction:
    """One actionMan action. probe_entrypoints must LOCATE it (label match) but MUST
    NOT call ``execute`` — ``executed`` stays False through a clean census."""

    def __init__(self, label):
        self._label = label
        self.executed = False

    def getDescriptionText(self):
        return self._label

    def execute(self):
        self.executed = True
        return True


class _FakeActionTable:
    def __init__(self, actions):
        self._actions = list(actions)
        self.numActions = len(self._actions)

    def getAction(self, index):
        return self._actions[index - 1]


def _install_live_link_action(rt, label="Initiate a Live-Link to Chaos Vantage"):
    """Seed one Vantage-live-link action into the mock's actionMan via direct
    prop/field writes (nothing lands in ``mutation_log``). → the action, for an
    executed-flag assertion."""
    action = _FakeAction(label)
    rt.actionMan._tables = [_FakeActionTable([action])]
    rt.actionMan._mg["props"]["numActionTables"] = 1
    return action


def _color_mgr(rt, mode, **extra):
    """Attach a ColorPipelineMgr with a ``mode`` (and any best-effort view/gamma
    props) — construction and instance-attach never log a mutation."""
    props = {"mode": mode}
    props.update(extra)
    rt.ColorPipelineMgr = MockObject(rt, "ColorPipelineMgr", props)
    return rt.ColorPipelineMgr


def _executes(rt):
    """Every expression the probe actually ``rt.execute``'d — the live-link TOGGLE
    globals must never appear here (only a gated, safe state read may)."""
    return [entry[1] for entry in rt.mutation_log if entry and entry[0] == "execute"]


def _assert_toggle_never_fired(rt):
    executed = _executes(rt)
    assert not any(expr in vt.LIVE_LINK_GLOBALS for expr in executed), executed


# --------------------------------------------------------------------- fixtures
@pytest.fixture()
def max_rt(monkeypatch):
    """A quiet (chaos-off) fake runtime installed as ``pymxs``; process stays clean."""
    rt = FakeMaxRuntime()
    monkeypatch.setitem(sys.modules, "pymxs", install(rt))
    return rt


@pytest.fixture(autouse=True)
def _deterministic_link(monkeypatch):
    """A real socket probe against 20701/20703 could attach to a live Vantage on the
    dev box — pin ``link_running`` to 'nothing attached' unless a test overrides it."""
    monkeypatch.setattr(vt, "link_running", lambda *a, **k: None)


def _offbox(monkeypatch):
    """Remove any leaked pymxs stub so ``_rt()`` raises → the genuine off-Max path."""
    monkeypatch.delitem(sys.modules, "pymxs", raising=False)


# --------------------------------------------------------------------- probe_entrypoints
def test_probe_entrypoints_offbox_is_inert_and_shaped(monkeypatch):
    _offbox(monkeypatch)
    res = vt.probe_entrypoints()
    assert set(res) == ENTRYPOINTS_KEYS
    assert res == {
        "link_running_port": None, "link_attached": False, "probe_state": None,
        "action_present": False, "action_label": None, "globals_present": [],
        "ports": list(vt.LIVE_LINK_PORTS), "in_max": False,
    }


def test_probe_entrypoints_in_max_without_globals_executes_nothing(max_rt):
    res = vt.probe_entrypoints()
    assert set(res) == ENTRYPOINTS_KEYS
    assert res["in_max"] is True
    assert res["probe_state"] is None            # globalVars absent → undetectable
    assert res["globals_present"] == []
    assert res["action_present"] is False
    assert res["action_label"] is None
    assert max_rt.mutation_log == []             # nothing executed, nothing mutated
    _assert_toggle_never_fired(max_rt)


def test_probe_entrypoints_globals_are_existence_tested_never_executed(max_rt):
    # toggle globals AND one state-query global all EXIST; the state read is armed True
    max_rt.globalVars = _GlobalVars(["vantageStartLiveLink", "startVantageLiveLink",
                                     "vantageLiveLinkActive"])
    max_rt.arm_execute("vantageLiveLinkActive()", True)
    res = vt.probe_entrypoints()
    # every toggle global that exists is REPORTED (order follows LIVE_LINK_GLOBALS)…
    assert res["globals_present"] == ["vantageStartLiveLink", "startVantageLiveLink"]
    # …the gated state query is the ONLY thing executed, and it answered True…
    assert res["probe_state"] is True
    assert _executes(max_rt) == ["vantageLiveLinkActive()"]
    # …and NO toggle global was ever rt.execute'd (the load-bearing guarantee)…
    _assert_toggle_never_fired(max_rt)
    # …and a census writes nothing (no ("set", …)/("set_global", …)/create/render).
    assert not [e for e in max_rt.mutation_log if e[0] != "execute"]


def test_probe_entrypoints_globals_present_but_state_query_absent(max_rt):
    # the toggle exists but NO state-query global does → reported, yet nothing executed
    max_rt.globalVars = _GlobalVars(["vantageStartLiveLink"])
    res = vt.probe_entrypoints()
    assert res["globals_present"] == ["vantageStartLiveLink"]
    assert res["probe_state"] is None
    assert _executes(max_rt) == []               # the ungated query is never evaluated
    _assert_toggle_never_fired(max_rt)


def test_probe_entrypoints_action_located_but_not_executed(max_rt):
    action = _install_live_link_action(max_rt)
    res = vt.probe_entrypoints()
    assert res["action_present"] is True
    assert res["action_label"] == "Initiate a Live-Link to Chaos Vantage"
    assert action.executed is False              # LOCATED, never fired
    assert _executes(max_rt) == []               # scanning action tables executes nothing
    _assert_toggle_never_fired(max_rt)


def test_probe_entrypoints_attach_signal_from_socket(monkeypatch, max_rt):
    monkeypatch.setattr(vt, "link_running", lambda *a, **k: 20701)
    res = vt.probe_entrypoints()
    assert res["link_running_port"] == 20701
    assert res["link_attached"] is True
    _assert_toggle_never_fired(max_rt)


# --------------------------------------------------------------------- vrscene_export_available
def test_vrscene_export_available_offbox(monkeypatch):
    _offbox(monkeypatch)
    res = vt.vrscene_export_available()
    assert set(res) == EXPORT_KEYS
    assert res["available"] is False
    assert res["method"] == "unavailable"
    assert res["console_exe_present"] is None    # no cfg supplied
    assert res["vantage_exe_present"] is None
    assert "vrayExportVRScene" in res["notes"]


def test_vrscene_export_available_in_max_reports_exporter_and_exports_nothing(max_rt):
    max_rt.add_maker("vrayExportVRScene", lambda *a, **k: None)   # exporter global present
    res = vt.vrscene_export_available()
    assert res["available"] is True
    assert res["method"] == "vrayExportVRScene"
    assert res["console_exe_present"] is None
    assert res["vantage_exe_present"] is None
    assert max_rt.mutation_log == []             # a report exports nothing


def test_vrscene_export_available_cfg_exe_presence(monkeypatch, tmp_path):
    _offbox(monkeypatch)
    present = tmp_path / "vantage_console.exe"
    present.write_bytes(b"x")
    cfg = SimpleNamespace(vantage_console=str(present),
                          vantage_exe=str(tmp_path / "missing_vantage.exe"))
    res = vt.vrscene_export_available(cfg)
    assert res["console_exe_present"] is True     # os.path.exists on the real file
    assert res["vantage_exe_present"] is False    # missing sibling
    assert res["available"] is False              # off-box: exporter global unreadable


# --------------------------------------------------------------------- probe_colorspace
def test_probe_colorspace_offbox(monkeypatch):
    _offbox(monkeypatch)
    res = rd.probe_colorspace()
    assert set(res) == COLORSPACE_KEYS
    assert res["available"] is False
    assert res["color_management"] == "unknown"
    assert res["mode"] == ""
    assert res["ocio_active"] is False
    assert res["view_transform"] is None
    assert res["display_gamma"] is None
    assert res["vray_display_correction"] is None
    assert "framebuffer" in res["risk"]           # the raw-plate caveat always rides along


def test_probe_colorspace_degrades_to_legacy_without_mgr(max_rt):
    res = rd.probe_colorspace()                    # FakeMaxRuntime has no ColorPipelineMgr
    assert res["available"] is False
    assert res["color_management"] == "legacy"
    assert res["mode"] == ""
    assert max_rt.mutation_log == []


def test_probe_colorspace_reports_gamma_mode(max_rt):
    _color_mgr(max_rt, "#gamma")
    res = rd.probe_colorspace()
    assert res["available"] is True
    assert res["mode"] == "#gamma"
    assert res["color_management"] == "gamma"
    assert res["ocio_active"] is False
    assert max_rt.mutation_log == []               # pure read — nothing mutated


@pytest.mark.parametrize("mode", ["OCIO", "#OCIO", "ACEScg"])
def test_probe_colorspace_reports_ocio_mode(max_rt, mode):
    _color_mgr(max_rt, mode)
    res = rd.probe_colorspace()
    assert res["color_management"] == "ocio"
    assert res["ocio_active"] is True
    assert res["mode"] == mode                     # raw string preserved verbatim


def test_probe_colorspace_unknown_mode_is_not_guessed(max_rt):
    _color_mgr(max_rt, "customPipeline")
    res = rd.probe_colorspace()
    assert res["available"] is True
    assert res["color_management"] == "unknown"    # unrecognized → not guessed
    assert res["ocio_active"] is False
    assert res["mode"] == "customPipeline"


def test_probe_colorspace_reads_best_effort_view_and_gamma(max_rt):
    _color_mgr(max_rt, "#gamma", displayViewTransform="sRGB 2.2", displayGamma=2.2)
    max_rt.renderers = MockObject(
        max_rt, "renderers",
        {"current": MockObject(max_rt, "V_Ray", {"output_srgb": "sRGB"})})
    res = rd.probe_colorspace()
    assert res["view_transform"] == "sRGB 2.2"
    assert res["display_gamma"] == 2.2
    assert res["vray_display_correction"] == "sRGB"
    assert max_rt.mutation_log == []
