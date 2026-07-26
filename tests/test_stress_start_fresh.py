"""start_fresh, driven for real. The inspection test in test_recheck locks the ORDER of
operations in source; these run the thing. The property that matters is asymmetric loss:
a reset throws away everything EXCEPT the one thing that cannot be re-derived — the
pre-match snapshot of the artist's own light — and it must keep that exactly when the
restore that would consume it fails."""

import pytest

from maxgaffer.core.genome import LightingState
from maxgaffer.maxbridge.controller import Controller


def _state(**kw):
    st = LightingState()
    for k, v in kw.items():
        st.set(k, v)
    return st


@pytest.fixture()
def ctrl():
    """An off-box controller with three cameras in known states: one restorable, one with
    nothing to restore, one whose restore will be made to fail."""
    c = Controller()
    for name, snapped in (("CamA", True), ("CamB", False), ("CamC", True)):
        e = c.session.entry(name, camera_id=name.lower())
        e.reference = f"{name}.png"
        e.semantics = {"time_of_day": "golden_hour"}
        e.notes = ["too warm"]
        e.locks = {"exposure.ev"}
        if snapped:
            e.pre_match = _state(**{"exposure.ev": 10.0, "sun.azimuth_deg": 100.0})
        e.state = _state(**{"exposure.ev": 12.0, "sun.azimuth_deg": 140.0})
    return c


def test_a_clean_reset_restores_then_forgets_everything(ctrl, monkeypatch):
    """Both halves must happen and in this order: the scene goes back, THEN the records go.
    Forgetting first would discard the only copy of the light being restored."""
    restored = []
    monkeypatch.setattr(ctrl, "restore_pre_match",
                        lambda cam, log=lambda m: None: restored.append(cam) or True)
    monkeypatch.setattr(ctrl, "save_session", lambda: True)

    out = ctrl.start_fresh(log=lambda m: None)

    assert sorted(restored) == ["CamA", "CamB", "CamC"], "every camera gets a restore try"
    assert sorted(out["restored"]) == ["CamA", "CamB", "CamC"]
    assert out["cleared"] == 3
    assert ctrl.session.cameras == {}, "records survived a reset that reported clearing them"


def test_a_failing_restore_keeps_that_camera_and_only_that_camera(ctrl, monkeypatch):
    """The asymmetric-loss property. pre_match is the only record of the artist's original
    light; a camera whose restore RAISES must keep its entry for a second attempt while the
    healthy cameras still clear. One bad camera must never strand the rest — and must never
    be silently dropped either, which would destroy the snapshot along with the failure."""
    def restore(cam, log=lambda m: None):
        if cam == "CamC":
            raise RuntimeError("scene node vanished")
        return True

    monkeypatch.setattr(ctrl, "restore_pre_match", restore)
    monkeypatch.setattr(ctrl, "save_session", lambda: True)

    out = ctrl.start_fresh(log=lambda m: None)

    assert sorted(out["restored"]) == ["CamA", "CamB"]
    assert [c for c, _why in out["failed"]] == ["CamC"]
    kept = list(ctrl.session.cameras.values())
    assert len(kept) == 1 and kept[0].camera_name == "CamC"
    assert kept[0].pre_match is not None, "the failed camera's snapshot was thrown away"


def test_reset_reports_honestly_when_there_is_nothing_to_do(monkeypatch):
    """An empty session is a fine thing to reset — the summary must say nothing happened
    rather than invent work, and the sidecar must still be written so the emptied state is
    durable."""
    c = Controller()
    saves = []
    monkeypatch.setattr(c, "save_session", lambda: saves.append(True) or True)

    out = c.start_fresh(log=lambda m: None)

    assert out["restored"] == [] and out["failed"] == [] and out["cleared"] == 0
    assert saves, "the emptied session was never persisted"


def test_baselines_survive_because_a_reset_cannot_rederive_them(ctrl, monkeypatch):
    """Baselines are read from the artist's own authored light multipliers; MaxGaffer did
    not write them and cannot recompute them from inside a reset. Forgetting them would
    make the NEXT match wrong, which is a strange gift for a button called reset."""
    ctrl.session.baselines = {"Key_Light": 2.5, "Fill": 0.8}
    monkeypatch.setattr(ctrl, "restore_pre_match", lambda cam, log=lambda m: None: True)
    monkeypatch.setattr(ctrl, "save_session", lambda: True)

    ctrl.start_fresh(log=lambda m: None)

    assert ctrl.session.baselines == {"Key_Light": 2.5, "Fill": 0.8}
    assert ctrl._baselines == {"Key_Light": 2.5, "Fill": 0.8}


def test_the_log_states_the_outcome_in_numbers(ctrl, monkeypatch):
    """After a destructive act the artist is owed a receipt: how many restored, how many
    cleared, how many kept back. A silent reset is indistinguishable from a broken one."""
    def restore(cam, log=lambda m: None):
        if cam == "CamB":
            return False                      # nothing snapped — legitimate, not a failure
        if cam == "CamC":
            raise RuntimeError("boom")
        return True

    monkeypatch.setattr(ctrl, "restore_pre_match", restore)
    monkeypatch.setattr(ctrl, "save_session", lambda: True)
    lines = []

    out = ctrl.start_fresh(log=lines.append)

    assert out["nothing_to_restore"] == ["CamB"]
    summary = [ln for ln in lines if ln.startswith("reset:")]
    assert summary, "no receipt line"
    assert "1 camera(s) restored" in summary[0]
    assert "2 cleared" in summary[0]
    assert "1 KEPT" in summary[0]
