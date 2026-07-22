import contextlib
import sys
import types

import pytest

from maxgaffer.core.animation import interpolate, sample_keyframes
from maxgaffer.core.genome import LightingState
from maxgaffer.maxbridge import apply as ap


def _state(az, intensity, group=1.0):
    st = LightingState()
    st.set("sun.azimuth_deg", az)
    st.set("sun.intensity", intensity)
    st.set("group.practicals", group)
    return st


def test_interpolation_uses_short_angle_arc_and_log_intensity():
    mid = interpolate(_state(350, 1), _state(10, 4), 0.5, "linear")
    assert mid.get("sun.azimuth_deg") == pytest.approx(0.0)
    assert mid.get("sun.intensity") == pytest.approx(2.0)


def test_sampling_is_inclusive_and_deduplicates_declared_frames():
    frames = sample_keyframes([(1, _state(0, 1)), (5, _state(90, 2)),
                               (5, _state(180, 4)), (9, _state(270, 8))], step=2,
                              easing="linear")
    assert [f for f, _ in frames] == [1, 3, 5, 7, 9]
    assert frames[2][1].get("sun.azimuth_deg") == 180


def test_bridge_bakes_every_sample_under_animate_and_attime(monkeypatch):
    events = []

    @contextlib.contextmanager
    def mark(kind, value):
        events.append((kind, "enter", value))
        try:
            yield
        finally:
            events.append((kind, "exit", value))

    pymxs = types.ModuleType("pymxs")
    pymxs.runtime = types.SimpleNamespace(redrawViews=lambda: events.append(("redraw",)))
    pymxs.undo = lambda enabled, name=None: mark("undo", name)
    pymxs.animate = lambda enabled: mark("animate", enabled)
    pymxs.attime = lambda frame: mark("time", frame)
    monkeypatch.setitem(sys.modules, "pymxs", pymxs)
    applied = []
    monkeypatch.setattr(ap, "_apply_inner",
                        lambda rig, bases, state, cam, warnings: applied.append(state))
    samples = [(1, _state(0, 1)), (3, _state(20, 2)), (5, _state(40, 4))]
    assert ap.bake_animation({}, {}, samples) == []
    assert len(applied) == 3
    assert [(e[0], e[2]) for e in events if e[0] == "time" and e[1] == "enter"] == [
        ("time", 1), ("time", 3), ("time", 5)]
