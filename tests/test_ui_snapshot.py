"""Renders the dock to docs/ui/*.png so a layout change can be SEEN before it ships.

Not an assertion suite — one smoke test that builds the dock in a realistic mid-session
state (verdict populated, deltas on the rig, both plate sides loaded, meter mid-run) and
saves the frame. It exists because scripts/ui_preview.py needs an environment this box no
longer has, while pytest under Max's own python with QT_QPA_PLATFORM=offscreen is
exercised daily by the dock suite and known to work. Skips cleanly without Qt.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtGui = pytest.importorskip("PySide6.QtGui")

from tests.test_ui_dock import Config, FakeController  # noqa: E402 — same fixture world

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "ui")


def test_snapshot_dock_mid_session(monkeypatch, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: Config(api_key="oc_test"))
    d = dockmod.MaxGafferDock()
    d.refresh_cameras()

    # a realistic mid-session state: a finished match with one weak component, two rig
    # deltas, plates on both sides of the wipe, and the meter mid-run
    d._set_verdict(87.7, {"key": 0.96, "envelope": 0.93, "histogram": 0.84,
                          "color": 0.90, "hue": 0.99, "direction": 0.98,
                          "highlight": 0.81},
                   ["sun-patch agreement 81% — the score cannot see this, your eyes can"])
    d._rig_deltas = {"sun.azimuth_deg": ("135.94", "30.00"),
                     "exposure.ev": ("13.00", "9.96")}
    d.rebuild_rig_controls()

    def plate(color):
        pix = QtGui.QPixmap(480, 270)
        pix.fill(QtGui.QColor(color))
        return pix

    d.ref_thumb.setPixmap(plate("#7a5228"))       # warm stand-in for the reference
    d.match_thumb.setPixmap(plate("#41506a"))     # cool stand-in for the match
    d.plate._split = 0.55
    d._progress_begin("sun solve")
    d._on_match_progress("sun solve", 27, 44, 34.8)

    d.resize(1060, 1240)
    d.show()
    app.processEvents()
    os.makedirs(OUT, exist_ok=True)
    ok = d.grab().save(os.path.join(OUT, "dock_v2.png"))
    d._progress_end()
    d.close()
    app.processEvents()
    assert ok, "the snapshot did not save"
