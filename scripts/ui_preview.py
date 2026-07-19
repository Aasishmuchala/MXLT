"""Offscreen UI preview — renders the dock and every dialog to docs/ui/*.png.

Design changes are judged on these images, not on adjectives. Runs anywhere with the dev
venv (PySide6 + Pillow), no Max needed:

    .venv/bin/python scripts/ui_preview.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "docs", "ui")


def _fake_render(path: str, w: int = 480, h: int = 270, warm: bool = True) -> str:
    from PIL import Image

    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            t = y / h
            if warm:   # dusk interior vibe
                px[x, y] = (int(235 - 90 * t), int(175 - 80 * t), int(120 - 40 * t))
            else:      # cool north light
                px[x, y] = (int(150 - 40 * t), int(175 - 30 * t), int(205 - 20 * t))
    im.save(path)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    from PySide6 import QtWidgets

    import tests.test_ui_dock as T
    from maxgaffer.maxbridge.config import Config
    from maxgaffer.ui import dock as dockmod

    dockmod.Controller = T.FakeController
    dockmod.cfgmod.load = lambda: Config(api_key="oc_preview")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    ref_warm = _fake_render(os.path.join(OUT, "_ref.png"), warm=True)
    ref_cool = _fake_render(os.path.join(OUT, "_render.png"), warm=False)

    d = dockmod.MaxGafferDock()
    for cam in ("PhysCam_Hero", "PhysCam_Lounge", "PhysCam_Stair"):
        d.ctrl.session.set_reference(cam, ref_warm)
    d.ctrl.session.record_match("PhysCam_Hero", T.demo_state(), 98.2)
    d.ctrl.best_render = ref_cool
    d.refresh_cameras()
    d._show_reference("PhysCam_Hero")
    d._set_match_thumb(ref_cool)

    plan_report = {
        "changes": [
            {"target": "node:VRaySun001", "prop": "turbidity", "before": 3.0,
             "after": 5.5, "why": "hazier warm sky per reference"},
            {"target": "exposure", "prop": "ev", "before": 13.2, "after": 11.5,
             "why": "reference key is brighter"},
        ],
        "created": [{"type": "VRayLight_plane", "name": "MG_window_fill",
                     "at": "-70° / 250u / h+120", "why": "soft camera-left fill"}],
        "warnings": [],
        "effect": {"before": 41.3, "after": 82.9},
    }
    rows = [
        {"target": "PhysCam_Hero", "prop": "sun.altitude_deg", "before": 55.0,
         "after": 8.0, "why": ""},
        {"target": "PhysCam_Hero", "prop": "sun.azimuth_deg", "before": 120.0,
         "after": 208.0, "why": ""},
        {"target": "PhysCam_Hero", "prop": "exposure.wb_kelvin", "before": 6500.0,
         "after": 4870.0, "why": ""},
    ]
    d._fill_changes(plan_report, rows, "PhysCam_Hero — target_reached, score 98.2")
    for line in ("— match: PhysCam_Hero —",
                 "reference: golden_hour, hazy sky, sun golden @ bearing -60°, "
                 "wb ~4900K — long warm rake through the glazing",
                 "iter 0: score 41.3/100 (key=0.44, color=0.51, direction=0.38)",
                 "iter 2: score 82.9/100 (key=0.91, color=0.84, direction=0.79)",
                 "iter 4: score 98.2/100 (key=0.99, color=0.97, direction=0.96)",
                 "✓ done (target_reached) — best 98.2"):
        d._log(line)
    d.resize(1040, 1100)
    d.show()
    app.processEvents()
    d.grab().save(os.path.join(OUT, "dock.png"))

    cands = [
        {"key": "as_analyzed", "label": "As analyzed", "why": "the reference's own read",
         "state": T.demo_state(), "render": ref_warm, "score": 84.1},
        {"key": "golden_low", "label": "Golden low sun", "why": "late warm key",
         "state": T.demo_state(), "render": ref_warm, "score": 77.7},
        {"key": "overcast_soft", "label": "Overcast soft", "why": "sky as the key",
         "state": T.demo_state(), "render": ref_cool, "score": 61.0},
        {"key": "cool_north", "label": "Cool north light", "why": "atelier look",
         "state": T.demo_state(), "render": ref_cool, "score": 58.4},
        {"key": "practicals_dusk", "label": "Practicals at dusk", "why": "lamps carry",
         "state": T.demo_state(), "render": ref_warm, "score": None},
    ]
    board = dockmod.ScenarioBoardDialog(cands)
    board.show()
    app.processEvents()
    board.grab().save(os.path.join(OUT, "board.png"))

    report = dockmod.ChangeReportDialog(plan_report, rows,
                                        "PhysCam_Hero — target_reached, score 98.2")
    report.show()
    app.processEvents()
    report.grab().save(os.path.join(OUT, "report.png"))

    plan = dockmod.PlanPreviewDialog(
        ["set  node:VRaySun001 · turbidity → 5.5   (hazier warm sky per reference)",
         "set  exposure · ev → 11.5   (reference key is brighter)",
         "new  VRayLight_plane 'MG_window_fill' bearing -70° · dist 250 · h +120   "
         "(soft camera-left fill)"],
        {"read": "Scene is lit midday-neutral; reference is a hazy golden hour with a "
                 "low warm key raking camera-left and practicals off."},
    )
    plan.show()
    app.processEvents()
    plan.grab().save(os.path.join(OUT, "plan.png"))

    settings = dockmod.SettingsDialog(Config(api_key="oc_preview"))
    settings.show()
    app.processEvents()
    settings.grab().save(os.path.join(OUT, "settings.png"))

    for f in ("_ref.png", "_render.png"):
        try:
            os.remove(os.path.join(OUT, f))
        except OSError:
            pass
    print(f"previews → {OUT}: dock.png, board.png, report.png, plan.png, settings.png")


if __name__ == "__main__":
    main()
