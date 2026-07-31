"""I3 — always interruptible, with a bounded worst case.

On 2026-07-30 the artist pressed ✕ repeatedly over many minutes and NOTHING HAPPENED,
correctly, per the code: the renders owned Max's main thread. The honest invariant is not
"cancel is instant" — a V-Ray frame in flight cannot be aborted from Python and no design
here pretends otherwise. It is:

    after ✕, the artist waits at most ONE probe, and that number was measured and stated
    before the run committed.

The bound is enforced in ONE place — the latch at the top of Controller._render_exposed,
the single function every render in the plugin passes through. That is the whole argument
for doing it there instead of adding twenty scattered should_cancel checks: forgetting one
then costs a probe, not a run.
"""

import os
import types

import pytest

import honesty_harness as H
from maxgaffer.core.errors import MatchCancelled
from maxgaffer.maxbridge import controller as ctl


# ------------------------------------------------------------------ the latch
def test_the_latch_raises_before_the_render_not_after(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    c._cancel_latch = "cancelled by the artist"
    with pytest.raises(MatchCancelled):
        c._render_exposed(object(), str(tmp_path / "x.png"), 8, 8)
    assert c.h.renders == [], "the latch must be checked BEFORE the frame starts"


@pytest.mark.parametrize("stage", ["basin", "tone", "sunsolve", "sweep", "iter",
                                   "polish", "final", "plan", "evcheck"])
def test_the_latch_bounds_every_stage(tmp_path, monkeypatch, stage):
    """The invariant expressed as a test rather than as twenty scattered checks: whatever
    loop is running, at most ONE further render happens after the latch is set."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    c._cost_stages = []
    before = len(c.h.renders)
    c._cancel_latch = ""
    for i in range(3):
        if i == 1:
            c._cancel_latch = "cancelled by the artist"
        try:
            c._render_exposed(object(), str(tmp_path / f"{stage}{i}.png"), 8, 8)
        except MatchCancelled:
            break
    assert len(c.h.renders) - before == 1, stage


def test_the_latch_bounds_the_polish_diagonal_ride(tmp_path, monkeypatch):
    """director.py's diagonal valley ride had no should_cancel check at all — its sibling
    single-axis climb does. The latch bounds it without touching it, because the ride
    cannot render except through _render_exposed."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    fired = []

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        # stand in for the ride: an unguarded while-loop asking for renders
        for i in range(50):
            fired.append(hooks.render(f"polish0_ev{i}"))
        return H.done()

    H.stub_director(monkeypatch, fake)
    cancel = {"v": False}

    def sc():
        cancel["v"] = cancel["v"] or len(c.h.renders) >= 2
        return cancel["v"]

    with pytest.raises(MatchCancelled):
        c.run_match("CamA", lambda _m: None, should_cancel=sc, multi_start=False,
                    do_sweep=False)
    assert len(c.h.renders) <= 3, c.h.renders


def test_the_latch_survives_sunsolves_deliberate_swallow(tmp_path, monkeypatch):
    """sunsolve.probe catches every exception per probe BY DESIGN, so before the explicit
    re-raise a cancel was swallowed 44 times and each swallow fired another 60-second
    render."""
    from maxgaffer.core import sunsolve

    calls = {"n": 0}

    def render(tag):
        calls["n"] += 1
        raise MatchCancelled("cancelled by the artist")

    with pytest.raises(MatchCancelled):
        sunsolve.solve_sun_angles(
            H.make_state(), {"hot_frac": 0.04, "hot_grid": [0.04] * 25},
            lambda st: None, render, lambda p: None)
    assert calls["n"] == 1, "the swallow must not eat a validated stop"


def test_the_sticky_abort_also_bounds_every_stage(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    c._probe_abort = "black"
    with pytest.raises(RuntimeError):
        c._render_exposed(object(), str(tmp_path / "x.png"), 8, 8)
    assert c.h.renders == []


# ------------------------------------------------------------------ swallow sites
def test_cancel_during_analyze_is_not_reported_as_gateway_unavailable(tmp_path,
                                                                     monkeypatch):
    """_analyze_or_fallback reported a cancel as "⚠ gateway unavailable (cancelled) —
    ANALYTIC-ONLY run" and then PROCEEDED on fabricated semantics: the artist pressed ✕
    and the tool answered by starting a worse version of the same run."""
    c = H.build(tmp_path, monkeypatch)
    monkeypatch.setattr(c, "analyze_reference",
                        lambda name: (_ for _ in ()).throw(
                            MatchCancelled("cancelled by the artist")))
    logs = []
    with pytest.raises(MatchCancelled):
        c._analyze_or_fallback("CamA", logs.append)
    assert not any("gateway unavailable" in ln for ln in logs)


def test_cancel_during_a_multi_reference_analyze_does_not_fire_more_gateway_calls(
        tmp_path, monkeypatch):
    """A bare `continue` let reference #2 fire another three gateway calls after ✕."""
    c = H.build(tmp_path, monkeypatch)
    e = c.session.entry("CamA")
    c.session.add_reference("CamA", str(c.h.ref))
    second = c.h.tmp / "ref2.jpg"
    second.write_bytes(b"\xff\xd8\xff\xe0two")
    c.session.add_reference("CamA", str(second))
    calls = {"n": 0}

    def sample(path):
        calls["n"] += 1
        raise MatchCancelled("cancelled by the artist")

    monkeypatch.setattr(c, "_analyze_samples", sample)
    with pytest.raises(MatchCancelled):
        c._analyze_multi_reference(e)
    assert calls["n"] == 1


def test_a_cancelled_run_does_not_fire_the_final_re_render(tmp_path, monkeypatch):
    """The final full re-render fires whenever best_state and best_render exist —
    including when the run was cancelled. It is one more 60-second frame after ✕."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)

    seen = {}

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        c._cancel_latch = "cancelled by the artist"
        seen["at_director"] = len(c.h.renders)
        # a state AND a render path: exactly the shape that used to fire one more frame
        r = H.done(score=71.0, state=H.make_state())
        r.best_render = str(tmp_path / "best.png")
        return r

    H.stub_director(monkeypatch, fake)
    c.run_match("CamA", lambda _m: None, multi_start=False, do_sweep=False)
    assert len(c.h.renders) == seen["at_director"], "✕ must not buy one more full frame"


# -------------------------------------------------- the latch must also be RELEASABLE
def test_a_cancelled_match_does_not_poison_refine_and_the_board(tmp_path, monkeypatch):
    """The blast radius of promoting the latch to gate _render_exposed (2026-07-31).

    Both latches are set from a dozen places and were cleared in exactly one — run_match.
    So after ✕, REFINE, the scenario board, execute_plan's effect probes and the delivered
    finals all raised the stale message before a single pixel, for the rest of the Max
    session, and only a fresh MATCH resurrected them."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    H.quiet_sun(monkeypatch)
    monkeypatch.setattr(ctl.ex, "execute_plan",
                        lambda ops, cam: {"changes": [], "created": [], "warnings": []})

    entries = {
        "execute_plan": lambda: c.execute_plan([], "CamA", lambda _m: None),
        "run_scenarios": lambda: c.run_scenarios("CamA", lambda _m: None),
        "refine": lambda: c.refine("CamA", "warmer", lambda _m: None),
        "render_finals_vray": lambda: c.render_finals_vray(
            ["CamA"], str(tmp_path / "out"), lambda *a: None),
    }
    for name, entry in entries.items():
        c._cancel_latch = "cancelled by the artist"
        c._probe_abort = "first probe rendered 100% BLACK"
        try:
            entry()
        except Exception:  # noqa: BLE001 — the point is only that the LATCH was cleared
            pass
        assert c._cancel_latch == "", name
        assert c._probe_abort == "", name


def test_plan_first_does_not_carry_a_stale_latch_into_the_next_match(tmp_path,
                                                                    monkeypatch):
    """With plan_first the dock runs execute_plan BEFORE run_match, so a stale latch
    killed the plan probe and the match never reached run_match's own reset."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    monkeypatch.setattr(ctl.ex, "execute_plan",
                        lambda ops, cam: {"changes": [], "created": [], "warnings": []})
    c._cancel_latch = "cancelled by the artist"
    c.execute_plan([], "CamA", lambda _m: None, measure=True)
    assert c._cancel_latch == ""
    assert c.h.renders, "the plan's effect probes must actually have rendered"


def test_beginning_an_operation_records_its_cancel_predicate(tmp_path, monkeypatch):
    """_cancel_poll is what the Vantage settle poll consults — it has to reflect BOTH the
    latch and the dock's live ✕, and it has to survive a predicate that raises."""
    c = H.build(tmp_path, monkeypatch)
    pressed = {"v": False}
    c._begin_operation(lambda: pressed["v"])
    assert c._cancel_poll() is False
    pressed["v"] = True
    assert c._cancel_poll() is True
    pressed["v"] = False
    c._cancel_latch = "cancelled by the artist"
    assert c._cancel_poll() is True
    c._begin_operation(lambda: (_ for _ in ()).throw(RuntimeError("broken")))
    assert c._cancel_poll() is False, "a broken predicate must not wedge the poll"


# ------------------------------------------------------------------ the dock, D4
@pytest.fixture
def dock(monkeypatch):
    pytest.importorskip("PySide6")
    from maxgaffer.ui import dock as dk

    app = dk.QtWidgets.QApplication.instance() or dk.QtWidgets.QApplication([])
    assert app is not None
    d = dk.MaxGafferDock.__new__(dk.MaxGafferDock)
    d._busy = True
    d._cancel = False
    d._detached = False
    d._beat = None
    d._log_lines = []
    d._log = d._log_lines.append
    d.ctrl = types.SimpleNamespace(_generation=0, cost_estimate=lambda: None)
    for name in ("btn_match", "btn_match_all", "btn_refine", "btn_board", "btn_cancel"):
        setattr(d, name, dk.QtWidgets.QPushButton())
    d.lbl_stage = dk.QtWidgets.QLabel()
    d.lbl_pct = dk.QtWidgets.QLabel()
    return d


def test_the_second_cancel_press_does_not_unset_the_flag(dock):
    """THE regression. _force_release set self._cancel = False, so the zombie's next
    checkpoint read False and the run RESUMED as if ✕ had never been pressed — with the
    cancel button now disabled, so the artist could not re-arm it. That is the reported
    symptom, in code."""
    dock._cancel = True
    dock._force_release("forced")
    assert dock._cancel is True, "detaching is not the artist changing their mind"
    assert dock._detached is True


def test_a_detached_run_voids_the_controllers_write_backs(dock):
    """The dialog promises the result "will be discarded". Before this only the DOCK's
    return value was, while the controller went on to apply state, record the match and
    save the session minutes later."""
    dock._force_release("forced")
    assert dock.ctrl._generation == 1


def test_start_match_refuses_while_a_previous_generation_is_live(dock):
    dock._busy = False
    dock._detached = True
    assert dock._run_blocked() is True
    assert any("RUN STILL FINISHING" in ln for ln in dock._log_lines)


def test_a_returned_run_frees_the_dock_again(dock):
    dock._busy = False
    dock._detached = False
    assert dock._run_blocked() is False


def test_a_detached_run_discards_its_result_and_says_so(tmp_path, monkeypatch):
    """The controller half of D4: a generation bump mid-run means nothing is written back."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    c.session.record_match("CamA", H.make_state(ev=9.0), 88.0)
    applied = []
    monkeypatch.setattr(c, "_record_match",
                        lambda *a, **k: applied.append("record"))

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        c._generation += 1                     # the dock detached us mid-run
        return H.done(score=95.0)

    H.stub_director(monkeypatch, fake)
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert applied == [], "a detached run must not record a match"
    assert any("detached" in ln and "DISCARDED" in ln for ln in logs)
    assert c.session.cameras["CamA"].score == 88.0     # the accepted score is intact


#: the start of the next method — a named constant so the newline escape does not have to
#: survive being read inside a triple-quoted docstring above it
_NEXT_DEF = "\n    def "


def _dock_source() -> str:
    """Read the dock as TEXT. PySide6 is not installed off-box, and these three claims are
    about lines of code rather than about Qt — the same trick test_recheck.py uses."""
    import maxgaffer.ui

    return open(os.path.join(os.path.dirname(maxgaffer.ui.__file__), "dock.py"),
                encoding="utf-8").read()


def _method_source(fn_name: str) -> str:
    src = _dock_source()
    body = src[src.index("def %s(" % fn_name):]
    return body[:body.index(_NEXT_DEF, 1)]


# ---------------------------------------------------- D4, provable without a Qt install
def test_force_release_never_clears_the_cancel_request_in_source():
    """The Qt tests above skip wherever PySide6 is absent — which is most CI. This one
    cannot skip, and it pins the exact line that caused the reported symptom."""
    body = _method_source("_force_release")
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "self._cancel = False" not in code, \
        "detaching the UI is not the artist changing their mind"
    assert "self._detached = True" in body
    assert "self.ctrl._generation += 1" in body


def test_every_run_entry_point_refuses_a_detached_dock_in_source():
    for fn in ("_start_match", "_start_match_all", "_start_refine", "_open_scenarios"):
        assert "self._run_blocked()" in _method_source(fn), fn


def test_every_run_entry_point_also_clears_detached_when_it_returns():
    """The other half of the same gate, and the half that was missing. _open_scenarios
    checked _run_blocked but its finally never cleared _detached, so ONE force-release
    taken during the scenario board left the flag set with nothing able to clear it —
    MATCH, MATCH ALL, REFINE and BOARD blocked for the life of the dock. (2026-07-31)"""
    for fn in ("_start_match", "_start_match_all", "_start_refine", "_open_scenarios"):
        body = _method_source(fn)
        assert "self._detached = False" in body, fn
        head, _, tail = body.partition("finally:")
        assert "self._detached = False" in tail, f"{fn}: must clear it in the finally"


def test_the_board_frees_the_dock_after_a_force_release(dock, monkeypatch):
    """Behaviourally, through the real method: force-release during the board, then let
    the board return, and a new run must be startable."""
    from maxgaffer.ui import dock as dk

    dock._busy = False
    dock._ab_on_pre = False
    dock._current_camera = lambda: "CamA"
    dock._progress_begin = lambda _s: None
    dock._show_log = lambda: None
    dock.refresh_cameras = lambda: None
    dock.ctrl.run_scenarios = lambda *a, **k: dock._force_release("forced") or []
    monkeypatch.setattr(dk.QtWidgets.QApplication, "processEvents",
                        staticmethod(lambda *a, **k: None), raising=False)
    dock._open_scenarios()
    assert dock._detached is False
    assert dock._run_blocked() is False


def test_the_cancel_tooltip_does_not_promise_what_cannot_be_delivered():
    """A V-Ray frame in flight cannot be aborted from Python. ESC is offered as "try",
    not promised: every probe renders vfb:false quiet:true precisely so it does not steal
    focus, and whether ESC still reaches V-Ray under that is an on-box calibration item."""
    src = _dock_source()
    assert "cannot be interrupted once it has started" in src
    assert "try pressing ESC" in src
