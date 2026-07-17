"""MaxGaffer dock — PySide6, instrument-grade dark (the LightMatch/MaxDirector house style).

Layout: camera board on the left (pick a shot, see its reference + score), work column on
the right (reference, match loop, rig sliders, Vantage). Threading contract:
  * every pymxs touch happens on Max's MAIN thread — always;
  * slow pure-I/O (gateway calls, sidecar stats) runs on a QThread while the main thread
    spins a local QEventLoop, so Max stays responsive mid-match and Cancel always works;
  * renders block Max by nature — the log narrates so it never feels dead.

Loaded inside 3ds Max only (bootstrap checks deps first).
"""

from __future__ import annotations

import html as _html
import os
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

from ..core.genome import GROUP_PREFIX, LightingState, spec_for
from ..core.omega import OmegaError, ping
from ..maxbridge import config as cfgmod
from ..maxbridge.controller import Controller

# Apple-dark design language (macOS system palette): flat surfaces, ONE hairline border
# everywhere, ONE accent color, no shadows, no gradients — the layout does the talking.
ACCENT = "#0a84ff"          # systemBlue (dark) — primary actions, selection, links. The
                            # only color in the app; everything else is a gray.
BG = "#1c1c1e"              # window
PANEL = "#2c2c2e"           # cards
INSET = "#1c1c1e"           # wells: inputs, thumbs, trees, transcript
FILL = "#3a3a3c"            # neutral controls (buttons, popups)
HAIR = "1px solid rgba(255,255,255,0.10)"
TEXT = "#f2f2f7"
DIM = "#98989d"             # secondary text / captions
FAINT = "#636366"           # disabled / placeholders
ERR = "#ff453a"             # systemRed (dark) — error emphasis only
OK = "#f2f2f7"

STYLE = (
    f"QWidget{{background:{BG};color:{TEXT};font-family:'SF Pro Text','Segoe UI Variable',"
    f"'Segoe UI',Inter;font-size:13px;}}"
    f"QFrame#card{{background:{PANEL};border:{HAIR};border-radius:12px;}}"
    f"QLabel{{background:transparent;}}"
    f"QPushButton{{background:{FILL};border:none;border-radius:8px;padding:7px 14px;"
    f"color:{TEXT};}}"
    f"QPushButton::menu-indicator{{image:none;width:0;}}"
    f"QPushButton:hover{{background:#48484a;}}"
    f"QPushButton:pressed{{background:#2c2c2e;}}"
    f"QPushButton:disabled{{background:#2c2c2e;color:{FAINT};}}"
    f"QPushButton#primary{{background:{ACCENT};color:#ffffff;font-weight:600;}}"
    f"QPushButton#primary:hover{{background:#2f95ff;}}"
    f"QPushButton#primary:pressed{{background:#0774e8;}}"
    f"QPushButton#primary:disabled{{background:#2c2c2e;color:{FAINT};}}"
    f"QPushButton#ghost{{background:transparent;border:none;color:{ACCENT};"
    f"padding:5px 8px;}}"
    f"QPushButton#ghost:hover{{color:#66b3ff;}}"
    f"QPushButton#ghost:pressed{{color:#0774e8;}}"
    f"QLineEdit,QTextEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox{{background:{INSET};"
    f"border:{HAIR};border-radius:8px;padding:5px 8px;"
    f"selection-background-color:{ACCENT};selection-color:#ffffff;}}"
    f"QComboBox{{background:{FILL};border:none;border-radius:8px;padding:6px 10px;}}"
    f"QComboBox:hover{{background:#48484a;}}"
    f"QComboBox::drop-down{{border:none;width:20px;}}"
    f"QComboBox QLineEdit{{background:transparent;border:none;padding:0;}}"
    f"QComboBox QAbstractItemView{{background:{PANEL};border:{HAIR};padding:4px;"
    f"selection-background-color:{ACCENT};selection-color:#ffffff;}}"
    f"QMenu{{background:{PANEL};border:{HAIR};border-radius:10px;padding:6px;}}"
    f"QMenu::item{{padding:6px 24px;border-radius:6px;}}"
    f"QMenu::item:selected{{background:{ACCENT};color:#ffffff;}}"
    f"QTreeWidget,QListWidget{{background:{INSET};border:{HAIR};border-radius:10px;"
    f"padding:4px;selection-background-color:{ACCENT};selection-color:#ffffff;}}"
    f"QLabel#dim{{color:{DIM};}}"
    f"QLabel#h{{color:{TEXT};font-weight:600;letter-spacing:3px;}}"
    f"QLabel#cap{{color:{DIM};font-size:11px;font-weight:600;letter-spacing:1px;}}"
    f"QHeaderView::section{{background:transparent;color:{DIM};border:none;"
    f"border-bottom:{HAIR};padding:5px 8px;font-size:11px;}}"
    f"QScrollBar:vertical{{background:transparent;width:8px;margin:2px;}}"
    f"QScrollBar::handle:vertical{{background:#48484a;border-radius:4px;min-height:30px;}}"
    f"QScrollBar::add-line,QScrollBar::sub-line{{height:0;}}"
    f"QToolTip{{background:{PANEL};color:{TEXT};border:{HAIR};padding:5px;}}"
)


def _cap(text: str) -> QtWidgets.QLabel:
    """Section caption — the one place small-caps styling lives."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("cap")
    return lbl


class _Worker(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _ProgressRelay(QtCore.QObject):
    """Marshals worker-thread progress callbacks onto the main thread — Qt widgets must
    never be touched from a vantage_console watcher thread."""

    progress = QtCore.Signal(str, str)


class MaxGafferDock(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaxGaffer")
        self.setStyleSheet(STYLE)
        self.cfg = cfgmod.load()
        self.ctrl = Controller(self.cfg)
        self.ctrl.io = self._run_blocking_io   # gateway waits run off-thread, Max stays alive
        self._workers: List[_Worker] = []
        self._cancel = False
        self._busy = False
        self._sliders: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._build()
        self.refresh_cameras()
        self._recover_draft_snapshot()

    def _recover_draft_snapshot(self):
        """A leftover snapshot means Max died mid-match with draft settings applied —
        put the artist's render settings back before anything else happens."""
        try:
            from ..maxbridge import draft as df

            if df.pending_snapshot():
                self._log("⚠ recovering render settings from a previous crashed session:")
                for line in df.restore_draft():
                    self._log("  " + line)
        except Exception as e:  # noqa: BLE001
            self._log(f"draft recovery check failed: {e}")

    # ================================================================= layout
    def _card(self, parent_layout):
        f = QtWidgets.QFrame()
        f.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        parent_layout.addWidget(f)
        return lay

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(0)
        host = QtWidgets.QScrollArea()
        host.setWidgetResizable(True)
        host.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(inner)
        col.setContentsMargins(2, 2, 10, 2)
        col.setSpacing(16)
        host.setWidget(inner)
        outer.addWidget(host)

        # ---- header: wordmark · camera dropdown · score · settings
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)
        title = QtWidgets.QLabel("MAXGAFFER")
        title.setObjectName("h")
        head.addWidget(title)
        head.addStretch(1)
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.setMinimumWidth(280)
        self.cam_combo.setToolTip("Camera — each keeps its own reference, notes, locks "
                                  "and matched lighting state.")
        self.cam_combo.currentIndexChanged.connect(self._on_camera_combo)
        head.addWidget(self.cam_combo)
        self.lbl_score = QtWidgets.QLabel("—")
        self.lbl_score.setObjectName("dim")
        self.lbl_score.setToolTip("Last match score for this camera.")
        head.addWidget(self.lbl_score)
        btn_settings = QtWidgets.QPushButton("Settings")
        btn_settings.clicked.connect(self._open_settings)
        head.addWidget(btn_settings)
        col.addLayout(head)

        # ---- card: reference vs latest match
        lr = self._card(col)
        thumbs = QtWidgets.QHBoxLayout()
        thumbs.setSpacing(14)

        def _thumb(placeholder, cap):
            wrap = QtWidgets.QVBoxLayout()
            wrap.setSpacing(6)
            t = QtWidgets.QLabel(placeholder)
            t.setFixedSize(272, 153)
            t.setAlignment(QtCore.Qt.AlignCenter)
            t.setStyleSheet(f"background:{INSET};border:{HAIR};border-radius:10px;"
                            f"color:{FAINT};")
            wrap.addWidget(t)
            wrap.addWidget(_cap(cap), 0, QtCore.Qt.AlignHCenter)
            thumbs.addLayout(wrap)
            return t

        self.ref_thumb = _thumb("no reference", "REFERENCE")
        self.match_thumb = _thumb("no match yet", "LATEST MATCH")
        side = QtWidgets.QVBoxLayout()
        side.setSpacing(10)
        btn_ref = QtWidgets.QPushButton("Load / swap reference…")
        btn_ref.clicked.connect(self._pick_reference)
        side.addWidget(btn_ref)
        self.lbl_ref_info = QtWidgets.QLabel("")
        self.lbl_ref_info.setObjectName("dim")
        self.lbl_ref_info.setWordWrap(True)
        side.addWidget(self.lbl_ref_info, 1)
        thumbs.addLayout(side, 1)
        lr.addLayout(thumbs)

        # ---- card: action bar (dropdowns, not checkbox walls)
        la = self._card(col)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(10)
        self.btn_match = QtWidgets.QPushButton("MATCH")
        self.btn_match.setObjectName("primary")
        self.btn_match.setToolTip("Run the match against this camera's reference.")
        self.btn_match.clicked.connect(self._start_match)
        bar.addWidget(self.btn_match, 1)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["Standard", "Deep → 99", "Loop only"])
        self.cmb_mode.setToolTip(
            "Standard — scene-wide plan + match loop.\n"
            "Deep → 99 — plan + loop + coordinate-descent polish to the ceiling.\n"
            "Loop only — skip the scene-wide plan.")
        bar.addWidget(self.cmb_mode)

        self.btn_locks = QtWidgets.QPushButton("Locks ▾")
        self.btn_locks.setToolTip("Locked parameters are never touched — not by the "
                                  "solver, not by the model.")
        self.lock_menu = QtWidgets.QMenu(self)
        self.btn_locks.setMenu(self.lock_menu)
        bar.addWidget(self.btn_locks)

        btn_opts = QtWidgets.QPushButton("Options ▾")
        m = QtWidgets.QMenu(self)

        def _act(label, checked, tip):
            a = m.addAction(label)
            a.setCheckable(True)
            a.setChecked(checked)
            a.setToolTip(tip)
            return a

        self.act_sweep = _act("Sun sweep first", True,
                              "Grid-solve the sun direction before iterating.")
        self.act_autoexec = _act("Auto-execute plan", bool(self.cfg.auto_execute_plan),
                                 "Skip the plan preview dialog.")
        self.act_draft = _act("Draft sampler", bool(self.cfg.draft_sampler),
                              "Draft render settings during matches (crash-safe restore).")
        self.act_popup = _act("Report popup", bool(self.cfg.show_report_popup),
                              "Show the scene-changed popup after runs.")
        self.act_live = _act("Live-apply sliders", True,
                             "Rig sliders write to the scene as you drag (Vantage mirrors).")
        self.act_apply_select = _act("Apply saved light on camera switch", True, "")
        self.act_apply_select.toggled.connect(self._on_apply_on_select)
        btn_opts.setMenu(m)
        bar.addWidget(btn_opts)

        self.btn_match_all = QtWidgets.QPushButton("ALL")
        self.btn_match_all.setToolTip("Match every camera that has a reference bound.")
        self.btn_match_all.clicked.connect(self._start_match_all)
        bar.addWidget(self.btn_match_all)
        self.btn_board = QtWidgets.QPushButton("BOARD")
        self.btn_board.setToolTip(
            "Scenario board — render candidate rigs (golden, overcast, backlit, north "
            "light, dusk practicals…), critic-scored against the reference when one is "
            "bound. Adopt one, then MATCH/REFINE from it.")
        self.btn_board.clicked.connect(self._open_scenarios)
        bar.addWidget(self.btn_board)
        self.btn_cancel = QtWidgets.QPushButton("✕")
        self.btn_cancel.setToolTip("Cancel after the current step.")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_match)
        bar.addWidget(self.btn_cancel)
        la.addLayout(bar)

        # ---- card: CHANGES (the record) + collapsed transcript
        lc = self._card(col)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(_cap("CHANGES"))
        crow.addStretch(1)
        for label, slot, tip in (("A/B", self._ab_flip, "Flip pre-match ↔ matched."),
                                 ("Restore", self._restore_pre_match,
                                  "Return to the pre-match light."),
                                 ("Runs", self._open_run_dir, "Open the run folder.")):
            b = QtWidgets.QPushButton(label)
            b.setObjectName("ghost")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            crow.addWidget(b)
        self.btn_transcript = QtWidgets.QPushButton("Transcript ▾")
        self.btn_transcript.setObjectName("ghost")
        self.btn_transcript.clicked.connect(self._toggle_transcript)
        crow.addWidget(self.btn_transcript)
        lc.addLayout(crow)
        self.changes_tree = QtWidgets.QTreeWidget()
        self.changes_tree.setHeaderLabels(["what", "before", "after"])
        self.changes_tree.setRootIsDecorated(True)
        self.changes_tree.setColumnWidth(0, 320)
        self.changes_tree.setColumnWidth(1, 140)
        self.changes_tree.setMinimumHeight(180)
        lc.addWidget(self.changes_tree)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        self.log.setVisible(False)
        lc.addWidget(self.log)

        # ---- card: refine (editable dropdown = notes + presets in one)
        lf = self._card(col)
        frow = QtWidgets.QHBoxLayout()
        frow.setSpacing(10)
        self.cmb_note = QtWidgets.QComboBox()
        self.cmb_note.setEditable(True)
        self.cmb_note.lineEdit().setPlaceholderText("tell the gaffer — or pick a note ▾")
        self.cmb_note.addItems(["", "too bright", "too dark", "too warm", "too cool",
                                "harder shadows", "softer shadows",
                                "sun more left", "sun more right"])
        self.cmb_note.lineEdit().returnPressed.connect(self._start_refine)
        frow.addWidget(self.cmb_note, 1)
        self.btn_refine = QtWidgets.QPushButton("REFINE")
        self.btn_refine.setObjectName("primary")
        self.btn_refine.setToolTip("Instant craft-table nudges, then a 3-lens ensemble; "
                                   "the measured winner continues into a deep match with "
                                   "your note pinned into every prompt.")
        self.btn_refine.clicked.connect(self._start_refine)
        frow.addWidget(self.btn_refine)
        lf.addLayout(frow)

        # ---- card: rig
        lg = self._card(col)
        grow = QtWidgets.QHBoxLayout()
        grow.addWidget(_cap("RIG"))
        grow.addStretch(1)
        for label, slot in (("Read scene", self.rebuild_rig_controls),
                            ("HDRI…", self._pick_hdri),
                            ("Seed dome", self._seed_dome),
                            ("Save preset…", self._save_preset),
                            ("Load preset…", self._load_preset)):
            b = QtWidgets.QPushButton(label)
            b.setObjectName("ghost")
            b.clicked.connect(slot)
            grow.addWidget(b)
        lg.addLayout(grow)
        self.rig_form = QtWidgets.QFormLayout()
        self.rig_form.setHorizontalSpacing(18)
        self.rig_form.setVerticalSpacing(8)
        lg.addLayout(self.rig_form)

        # ---- card: output
        lo = self._card(col)
        orow = QtWidgets.QHBoxLayout()
        orow.setSpacing(10)
        orow.addWidget(_cap("OUTPUT"))
        btn_link = QtWidgets.QPushButton("Live link")
        btn_link.setToolTip("V-Ray's 'Initiate a Live-Link to Chaos Vantage' — a toggle; "
                            "starts Vantage if needed (port 20701).")
        btn_link.clicked.connect(self._start_live_link)
        orow.addWidget(btn_link)
        b1 = QtWidgets.QPushButton("Final (selected)")
        b1.setToolTip("V-Ray final render of this camera under its matched light.")
        b1.clicked.connect(lambda: self._render_finals(selected_only=True))
        orow.addWidget(b1)
        b2 = QtWidgets.QPushButton("Final ALL")
        b2.clicked.connect(lambda: self._render_finals(selected_only=False))
        orow.addWidget(b2)
        b3 = QtWidgets.QPushButton("→ Vantage queue")
        b3.setToolTip("Export per-camera vrscenes and open Vantage's batch queue.")
        b3.clicked.connect(self._export_for_vantage)
        orow.addWidget(b3)
        self.lbl_link = QtWidgets.QLabel("")
        self.lbl_link.setObjectName("dim")
        orow.addWidget(self.lbl_link, 1)
        lo.addLayout(orow)
        col.addStretch(1)

    def _toggle_transcript(self):
        vis = not self.log.isVisible()
        self.log.setVisible(vis)
        self.btn_transcript.setText("Transcript ▴" if vis else "Transcript ▾")

    def _fill_changes(self, plan_report, state_rows, headline):
        """The CHANGES panel — the requested always-visible record of what was done."""
        t = self.changes_tree
        t.clear()
        top = QtWidgets.QTreeWidgetItem([headline, "", ""])
        top.setForeground(0, QtGui.QBrush(QtGui.QColor("#f2f2f2")))
        t.addTopLevelItem(top)
        pr = plan_report or {"changes": [], "created": [], "warnings": []}
        if pr.get("effect"):
            eff = pr["effect"]
            top.addChild(QtWidgets.QTreeWidgetItem(
                ["plan effect (measured)", f"{eff['before']:.1f}", f"{eff['after']:.1f}"]))
        for c in pr["changes"]:
            top.addChild(QtWidgets.QTreeWidgetItem(
                [f"{c['target']} · {c['prop']}", str(c["before"]), str(c["after"])]))
        for c in pr["created"]:
            top.addChild(QtWidgets.QTreeWidgetItem(
                [f"+ {c['type']} '{c['name']}'", "", c["at"]]))
        for r in state_rows:
            top.addChild(QtWidgets.QTreeWidgetItem(
                [r["prop"], str(r["before"]), str(r["after"])]))
        for w in pr["warnings"]:
            top.addChild(QtWidgets.QTreeWidgetItem(["! " + w, "", ""]))
        top.setExpanded(True)

    # ================================================================= helpers
    def _log(self, msg: str):
        if msg.startswith("THUMB::"):
            url = QtCore.QUrl.fromLocalFile(msg[len("THUMB::"):]).toString()
            self.log.append(f'<img src="{url}" width="240">')
        else:
            self.log.append(_html.escape(msg))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        QtWidgets.QApplication.processEvents()

    def _run_blocking_io(self, fn):
        """Run pure-I/O ``fn`` on a worker while the MAIN thread spins a local event loop —
        Max stays alive, pymxs is never touched off-thread, exceptions re-raise here."""
        loop = QtCore.QEventLoop()
        box = {}
        w = _Worker(fn)
        self._workers.append(w)
        w.done.connect(lambda r: (box.__setitem__("r", r), loop.quit()))
        w.failed.connect(lambda e: (box.__setitem__("e", e), loop.quit()))
        w.start()
        loop.exec()
        w.wait()
        self._workers.remove(w)
        if "e" in box:
            raise RuntimeError(box["e"])
        return box.get("r")

    def _current_camera(self) -> str:
        data = self.cam_combo.currentData() if hasattr(self, "cam_combo") else None
        return data or ""
    # ================================================================= cameras
    def refresh_cameras(self):
        current = self._current_camera()
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        try:
            cams = self.ctrl.cameras()
        except Exception as e:  # noqa: BLE001
            self._log(f"camera scan failed: {e}")
            cams = []
        for c in cams:
            mark = "●  " if c.get("reference") else "○  "
            score = f"   ·  {c['score']:.0f}" if c.get("score") is not None else ""
            self.cam_combo.addItem(mark + c["name"] + score, c["name"])
        idx = self.cam_combo.findData(current)
        self.cam_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.cam_combo.blockSignals(False)
        try:
            self.act_apply_select.setChecked(
                bool(self.ctrl.session.settings.get("apply_on_select", True)))
        except Exception:
            pass
        self._sync_score_badge()
        self.rebuild_rig_controls()
        self._rebuild_locks(self._current_camera())

    def _sync_score_badge(self):
        e = self.ctrl.session.cameras.get(self._current_camera())
        self.lbl_score.setText(f"{e.score:.1f}" if (e and e.score is not None) else "—")
    def _on_camera_combo(self, _idx: int):
        if self._busy:
            self._log("busy — camera switch ignored until the current run finishes")
            return
        name = self._current_camera()
        if not name:
            return
        self._ab_on_pre = False
        try:
            for w in self.ctrl.select_camera(name):
                self._log("⚠ " + w)
        except Exception as e:  # noqa: BLE001
            self._log(f"select failed: {e}")
        self._show_reference(name)
        self._rebuild_locks(name)
        self._sync_score_badge()
        self.rebuild_rig_controls()
    def _on_apply_on_select(self, checked: bool):
        self.ctrl.session.settings["apply_on_select"] = bool(checked)
        self.ctrl.save_session()

    # ================================================================= reference
    def _pick_reference(self):
        cam = self._current_camera()
        if not cam:
            self._log("select a camera first")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Reference for {cam}", "", "Images (*.jpg *.jpeg *.png *.webp)")
        if not path:
            return
        self.ctrl.session.set_reference(cam, path)
        if not self.ctrl.save_session():
            self._log("⚠ scene not saved yet — bindings live in memory only until you "
                      "save the .max file")
        self._show_reference(cam)
        self.refresh_cameras()

    def _show_reference(self, cam: str):
        e = self.ctrl.session.cameras.get(cam)
        ref = e.reference if e else ""
        if ref and os.path.exists(ref):
            pix = QtGui.QPixmap(ref)
            if not pix.isNull():
                self.ref_thumb.setPixmap(pix.scaled(
                    self.ref_thumb.size(), QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))
                info = os.path.basename(ref)
                if e and e.semantics:
                    s = e.semantics
                    info += (f"\n{s.get('time_of_day')}, {s.get('sky')} sky, "
                             f"wb ~{s.get('wb_kelvin_estimate', 0):.0f}K")
                if e and e.score is not None:
                    info += f"\nlast match: {e.score:.1f}/100 at {e.matched_at}"
                self.lbl_ref_info.setText(info)
                return
        self.ref_thumb.setPixmap(QtGui.QPixmap())
        self.ref_thumb.setText("no reference")
        self.lbl_ref_info.setText("Bind a lighting reference image to this camera.")

    def _rebuild_locks(self, cam: str):
        self.lock_menu.clear()
        e = self.ctrl.session.cameras.get(cam)
        locked = set(e.locks) if e else set()
        try:
            state = self.ctrl.read_state(cam)
        except Exception:
            state = LightingState()
        keys = sorted(state.keys())
        if not keys:
            a = self.lock_menu.addAction("(no rig parameters)")
            a.setEnabled(False)
        for key in keys:
            a = self.lock_menu.addAction(key)
            a.setCheckable(True)
            a.setChecked(key in locked)
    def _locks(self) -> set:
        return {a.text() for a in self.lock_menu.actions()
                if a.isCheckable() and a.isChecked()}
    # ================================================================= rig sliders
    def rebuild_rig_controls(self):
        while self.rig_form.rowCount() > 0:
            self.rig_form.removeRow(0)
        self._sliders.clear()
        try:
            state = self.ctrl.read_state(self._current_camera())
        except Exception as e:  # noqa: BLE001
            lbl = QtWidgets.QLabel(f"rig unavailable: {e}")
            lbl.setObjectName("dim")
            self.rig_form.addRow(lbl)
            return
        for key in sorted(state.keys()):
            spec = spec_for(key)
            if spec is None:
                continue
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(spec.lo, spec.hi)
            spin.setDecimals(2)
            spin.setSingleStep(1.0 if spec.hi - spec.lo > 20 else 0.1)
            spin.setValue(state.get(key))
            spin.setToolTip(spec.doc)
            spin.valueChanged.connect(lambda v, k=key: self._on_slider(k, v))
            self._sliders[key] = spin
            self.rig_form.addRow(key, spin)
    def _pick_hdri(self):
        if self._busy:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Dome HDRI", "", "HDR images (*.hdr *.exr *.jpg *.png *.tif)")
        if not path:
            return
        how = self.ctrl.set_dome_hdri(path)
        self._log(f"dome HDRI → {os.path.basename(path)} ({how})" if how != "failed"
                  else "✗ could not set the dome texture (no dome, or unknown file prop — "
                       "checklist #16)")
        if how != "failed":
            # a manual pick outranks the seed — otherwise the next camera switch would
            # silently re-bind the seed over the artist's explicit choice
            cam = self._current_camera()
            e = self.ctrl.session.cameras.get(cam) if cam else None
            if e is not None and e.seed_hdri:
                e.seed_hdri = ""
                self.ctrl.save_session()
                self._log(f"seed released for {cam} — the manual HDRI takes over "
                          "(Restore still returns to the pre-seed dome)")

    def _seed_dome(self):
        """Reference → HDR pano → dome texture (controller snapshots for Restore)."""
        if self._busy:
            return
        cam = self._current_camera()
        if not cam:
            self._log("select a camera first")
            return
        e = self.ctrl.session.cameras.get(cam)
        if not (e and e.reference):
            self._log("bind a reference image first — the seed is built FROM it")
            return
        self._busy = True
        try:
            meta = self.ctrl.seed_dome(cam, log=self._log)
            sun = (meta or {}).get("sun")
            self._log("✓ dome seeded"
                      + (f" — sun disc at az {sun['azimuth_deg']:.0f}° / "
                         f"alt {sun['altitude_deg']:.0f}°" if sun
                         else " (no disc — overcast/night reference)"))
            self.rebuild_rig_controls()
        except Exception as err:  # noqa: BLE001
            self._log(f"✗ seed: {err}")
        finally:
            self._busy = False

    # ================================================================= scenario board
    def _open_scenarios(self):
        if self._busy:
            return
        cam = self._current_camera()
        if not cam:
            self._log("select a camera first")
            return
        self._busy = True
        self._cancel = False
        for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._log(f"— scenario board: {cam} —")
        results = []
        try:
            results = self.ctrl.run_scenarios(cam, log=self._log,
                                              should_cancel=lambda: self._cancel)
        except Exception as err:  # noqa: BLE001
            self._log(f"✗ board: {err}")
        finally:
            self._busy = False
            for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
                b.setEnabled(True)
            self.btn_cancel.setEnabled(False)
        if not results:
            return
        dlg = ScenarioBoardDialog(results, self)
        if dlg.exec() and dlg.chosen is not None:
            c = results[dlg.chosen]
            try:
                for w in self.ctrl.adopt_scenario(cam, c["state"], c.get("score")):
                    self._log("⚠ " + w)
                self._log(f"✓ adopted scenario: {c['label']}"
                          + (f" ({c['score']:.1f})" if c.get("score") is not None else ""))
                self._set_match_thumb(c.get("render"))
                self.rebuild_rig_controls()
                self.refresh_cameras()
            except Exception as err:  # noqa: BLE001
                self._log(f"✗ adopt: {err}")
        else:
            self._log("board closed — current light kept (it was re-applied already)")

    def _save_preset(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save lighting preset", "", "MaxGaffer preset (*.json)")
        if not path:
            return
        ok = self.ctrl.save_preset(path, self._current_camera())
        self._log(f"preset saved → {path}" if ok else f"✗ could not write {path}")

    def _load_preset(self):
        if self._busy:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load lighting preset", "", "MaxGaffer preset (*.json)")
        if not path:
            return
        try:
            for w in self.ctrl.load_preset(path, self._current_camera()):
                self._log("⚠ " + w)
            self._log(f"preset applied: {os.path.basename(path)}")
            self.rebuild_rig_controls()
            self.refresh_cameras()
        except Exception as err:  # noqa: BLE001
            self._log(f"✗ {err}")

    def _on_slider(self, key: str, value: float):
        if not self.act_live.isChecked() or self._busy:
            return
        st = LightingState()
        if key.startswith(GROUP_PREFIX):
            st.groups[key[len(GROUP_PREFIX):]] = value
        else:
            st.set(key, value)
        try:
            self.ctrl.apply_state(st, self._current_camera())
        except Exception as e:  # noqa: BLE001
            self._log(f"apply failed: {e}")

    # ================================================================= match
    def _start_match(self):
        if self._busy:
            return
        cam = self._current_camera()
        if not cam:
            self._log("select a camera first")
            return
        e = self.ctrl.session.cameras.get(cam)
        if not (e and e.reference):
            self._log("bind a reference image first")
            return
        if not self.cfg.api_key:
            self._log("no API key — open Settings and paste your oc_ key")
            return
        self._busy = True
        self._cancel = False
        for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        mode = self.cmb_mode.currentIndex()          # 0 standard · 1 deep · 2 loop-only
        self.cfg.auto_execute_plan = self.act_autoexec.isChecked()
        self.cfg.draft_sampler = self.act_draft.isChecked()
        self.log.clear()
        self._log(f"— match: {cam} —")
        plan_report = None
        try:
            if mode != 2:
                try:
                    ops, lines, meta, _raw = self.ctrl.make_plan(cam, log=self._log)
                except (OmegaError, RuntimeError) as err:
                    self._log(f"⚠ plan skipped ({err}) — continuing with the match loop")
                    ops, lines, meta = [], [], {}
                if not ops:
                    self._log("plan: no operations proposed — continuing to the match loop")
                elif self.act_autoexec.isChecked() or PlanPreviewDialog(
                        lines, meta, self).exec():
                    self._log(f"— executing plan ({len(ops)} ops) —")
                    plan_report = self.ctrl.execute_plan(ops, cam, log=self._log)
                else:
                    self._log("plan declined — continuing with the match loop only")
            result = self.ctrl.run_match(
                cam, log=self._log,
                should_cancel=lambda: self._cancel,
                locks=self._locks(),
                do_sweep=self.act_sweep.isChecked(),
                deep=(mode == 1))
            score = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
            ceiling = (" · ceiling proven — the gap left is content, not lighting"
                       if result.ceiling_converged and (result.best_score or 0) < 99 else "")
            self._log(f"✓ done ({result.stop_reason}) — best {score}{ceiling}")
            self._set_match_thumb(result.best_render)
            headline = f"{cam} — {result.stop_reason}, score {score}"
            self._fill_changes(plan_report, self.ctrl.state_change_rows(cam), headline)
            if self.act_popup.isChecked():
                ChangeReportDialog(plan_report, self.ctrl.state_change_rows(cam),
                                   headline, self).exec()
        except (OmegaError, RuntimeError) as err:
            self._log(f"✗ {err}")
        except Exception as err:  # noqa: BLE001
            self._log(f"✗ unexpected: {err}")
        finally:
            self._busy = False
            self._ab_on_pre = False
            for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
                b.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.refresh_cameras()
            self._show_reference(cam)
    def _start_match_all(self):
        if self._busy:
            return
        queue = [n for n, e in self.ctrl.session.cameras.items() if e.reference]
        if not queue:
            self._log("no cameras have references bound — bind references first")
            return
        est = len(queue) * (int(self.cfg.max_iterations)
                            + (self.cfg.sweep_count if self.act_sweep.isChecked() else 0))
        if QtWidgets.QMessageBox.question(
                self, "Match ALL",
                f"Match {len(queue)} camera(s) sequentially (~{est} loop renders total)?\n"
                f"{', '.join(queue)}",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        self._busy = True
        self._cancel = False
        for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.cfg.draft_sampler = self.act_draft.isChecked()
        self.log.clear()
        self._log(f"— batch match: {len(queue)} cameras —")
        try:
            results = self.ctrl.match_all(log=self._log,
                                          should_cancel=lambda: self._cancel,
                                          do_sweep=self.act_sweep.isChecked())
            self._log("— batch summary —")
            for cam, status in results.items():
                self._log(f"  {cam}: {status}")
        except Exception as err:  # noqa: BLE001
            self._log(f"✗ batch: {err}")
        finally:
            self._busy = False
            self._ab_on_pre = False
            for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
                b.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.refresh_cameras()
    def _cancel_match(self):
        self._cancel = True
        self._log("cancelling after the current step…")

    def _set_match_thumb(self, path):
        if path and os.path.exists(path):
            pix = QtGui.QPixmap(path)
            if not pix.isNull():
                self.match_thumb.setPixmap(pix.scaled(
                    self.match_thumb.size(), QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))
                return
        self.match_thumb.setPixmap(QtGui.QPixmap())
        self.match_thumb.setText("no match yet")

    def _start_refine(self):
        if self._busy:
            return
        cam = self._current_camera()
        note = self.cmb_note.currentText().strip()
        if not cam or not note:
            self._log("select a camera and type a note first")
            return
        e = self.ctrl.session.cameras.get(cam)
        if not (e and e.reference):
            self._log("bind a reference image first")
            return
        self._busy = True
        self._cancel = False
        for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._log(f"— refine: {cam} — “{note}”")
        try:
            result = self.ctrl.refine(cam, note, log=self._log,
                                      should_cancel=lambda: self._cancel)
            score = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
            ceiling = (" · ceiling proven — the gap left is content, not lighting"
                       if result.ceiling_converged and (result.best_score or 0) < 99 else "")
            self._log(f"✓ refine done ({result.stop_reason}) — best {score}{ceiling}")
            self._set_match_thumb(result.best_render)
            headline = f"{cam} — refined to {score}"
            self._fill_changes(None, self.ctrl.state_change_rows(cam), headline)
            self.cmb_note.setCurrentText("")
            if self.act_popup.isChecked():
                ChangeReportDialog(None, self.ctrl.state_change_rows(cam),
                                   headline, self).exec()
        except (OmegaError, RuntimeError) as err:
            self._log(f"✗ {err}")
        except Exception as err:  # noqa: BLE001
            self._log(f"✗ unexpected: {err}")
        finally:
            self._busy = False
            self._ab_on_pre = False
            for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
                b.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.refresh_cameras()
            self._show_reference(cam)
    def _open_run_dir(self):
        d = self.ctrl._run_dir or cfgmod.sessions_dir()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))

    def _restore_pre_match(self):
        if self._busy:
            return
        cam = self._current_camera()
        if cam and self.ctrl.restore_pre_match(cam):
            self._log(f"restored pre-match lighting for {cam}")
            self._ab_on_pre = True
            self.rebuild_rig_controls()
        else:
            self._log("no pre-match snapshot for this camera yet")

    def _ab_flip(self):
        if self._busy:
            return
        cam = self._current_camera()
        e = self.ctrl.session.cameras.get(cam) if cam else None
        if not (e and e.pre_match is not None and e.state is not None):
            self._log("A/B needs both a pre-match snapshot and a matched state — run a "
                      "match first")
            return
        self._ab_on_pre = not getattr(self, "_ab_on_pre", False)
        try:
            self.ctrl.apply_state(e.pre_match if self._ab_on_pre else e.state, cam)
            self._log(f"A/B → showing {'A (pre-match)' if self._ab_on_pre else 'B (matched)'}")
            self.rebuild_rig_controls()
        except Exception as err:  # noqa: BLE001
            self._log(f"A/B failed: {err}")

    # ================================================================= vantage
    def _start_live_link(self):
        ok, how = self.ctrl.start_live_link()
        self.lbl_link.setText(("link: started — " if ok else "link: ") + how)
        self._log(("vantage live link: " if ok else "⚠ vantage live link: ") + how)

    def _final_targets(self, selected_only: bool):
        cams = ([self._current_camera()] if selected_only
                else self.ctrl.session.cameras_with_states())
        return [c for c in cams if c]

    def _render_finals(self, selected_only: bool):
        if self._busy:
            return
        cams = self._final_targets(selected_only)
        if not cams:
            self._log("no cameras to render (match or save states first)")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if not out_dir:
            return
        self._busy = True
        try:
            if self.cfg.final_render_backend == "vantage_cli":
                # Developer-Edition CLI only — exports main-thread, renders on a worker
                jobs = self.ctrl.prepare_vantage_jobs(
                    cams, out_dir, on_progress=lambda c, s: self._log(f"vantage {c}: {s}"))
                relay = _ProgressRelay()
                relay.progress.connect(lambda c, s: self._log(f"vantage {c}: {s}"))
                results = self._run_blocking_io(
                    lambda: self.ctrl.run_vantage_jobs(
                        jobs, on_progress=lambda c, s: relay.progress.emit(c, s)))
            else:
                results = self.ctrl.render_finals_vray(
                    cams, out_dir, on_progress=lambda c, s: self._log(f"final {c}: {s}"))
            for cam, status in results.items():
                self._log(f"{'✓' if status == 'ok' else '✗'} {cam}: {status}")
        except Exception as e:  # noqa: BLE001
            self._log(f"✗ final renders: {e}")
        finally:
            self._busy = False

    def _export_for_vantage(self):
        if self._busy:
            return
        cams = self._final_targets(selected_only=False)
        if not cams:
            self._log("no matched cameras to export")
            return
        self._busy = True
        try:
            jobs, launched, export_dir = self.ctrl.export_and_open_vantage(
                cams, on_progress=lambda c, s: self._log(f"export {c}: {s}"))
            self._log(f"✓ {len(jobs)} vrscene(s) → {export_dir}")
            self._log("Vantage opened — add the files to its Batch Render queue"
                      if launched else
                      f"⚠ could not launch Vantage ({self.cfg.vantage_exe}) — open the "
                      "folder manually")
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(export_dir))
        except Exception as e:  # noqa: BLE001
            self._log(f"✗ vantage export: {e}")
        finally:
            self._busy = False

    # ================================================================= settings
    def _open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self.cfg.save()
            self.ctrl.cfg = self.cfg
            self._log("settings saved")


class ScenarioBoardDialog(QtWidgets.QDialog):
    """Light Gen, measured: candidate rigs as clickable cards — probe thumbnail, name,
    and the critic's score when a reference is bound. The best-scoring card comes
    preselected; ADOPT applies it and saves it as the camera's state."""

    def __init__(self, candidates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaxGaffer — scenario board")
        self.setStyleSheet(STYLE)
        self.chosen: Optional[int] = None
        lay = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)
        self._cards: List[QtWidgets.QToolButton] = []
        for i, c in enumerate(candidates):
            btn = QtWidgets.QToolButton()
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            score = f"   ·   {c['score']:.1f}" if c.get("score") is not None else ""
            btn.setText(c["label"] + score)
            btn.setToolTip(c.get("why", ""))
            render = c.get("render")
            if render and os.path.exists(render):
                pix = QtGui.QPixmap(render)
                if not pix.isNull():
                    btn.setIcon(QtGui.QIcon(pix))
                    btn.setIconSize(QtCore.QSize(240, 135))
            btn.setStyleSheet(
                f"QToolButton{{background:{INSET};border:{HAIR};border-radius:12px;"
                f"padding:10px;color:{TEXT};}}"
                f"QToolButton:checked{{border:2px solid {ACCENT};"
                f"background:rgba(10,132,255,0.12);}}")
            btn.clicked.connect(lambda _=False, idx=i: setattr(self, "chosen", idx))
            grid.addWidget(btn, i // 3, i % 3)
            self._cards.append(btn)
        scored = [i for i, c in enumerate(candidates) if c.get("score") is not None]
        if scored:                             # preselect the measured winner
            best = max(scored, key=lambda i: candidates[i]["score"])
            self._cards[best].setChecked(True)
            self.chosen = best
        lay.addLayout(grid)
        note = QtWidgets.QLabel(
            "Adopting applies the rig and saves it as this camera's state — MATCH / "
            "REFINE continue from it. 'Restore' returns to the light before the board.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        lay.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("ADOPT")
        ok.setObjectName("primary")
        ok.clicked.connect(self._adopt)
        row.addWidget(ok, 1)
        keep = QtWidgets.QPushButton("Keep current light")
        keep.clicked.connect(self.reject)
        row.addWidget(keep)
        lay.addLayout(row)

    def _adopt(self):
        if self.chosen is None:
            for i, b in enumerate(self._cards):
                if b.isChecked():
                    self.chosen = i
                    break
        if self.chosen is None:
            self.reject()
        else:
            self.accept()


class PlanPreviewDialog(QtWidgets.QDialog):
    """The approval gate: the model's scene read + every operation it wants to run."""

    def __init__(self, lines, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaxGaffer — change plan")
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(560)
        lay = QtWidgets.QVBoxLayout(self)
        read = QtWidgets.QLabel(meta.get("read") or "")
        read.setWordWrap(True)
        read.setObjectName("dim")
        lay.addWidget(read)
        box = QtWidgets.QPlainTextEdit("\n".join(lines))
        box.setReadOnly(True)
        box.setMinimumHeight(220)
        lay.addWidget(box)
        note = QtWidgets.QLabel("Executes as ONE undo step · new lights land on the "
                                "MG_lights layer.")
        note.setObjectName("dim")
        lay.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton(f"EXECUTE {len(lines)} OPS")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        row.addWidget(ok, 1)
        skip = QtWidgets.QPushButton("Skip plan")
        skip.clicked.connect(self.reject)
        row.addWidget(skip)
        lay.addLayout(row)


class ChangeReportDialog(QtWidgets.QDialog):
    """The 'scene changed' popup: values changed (before → after), lights placed, warnings."""

    def __init__(self, plan_report, state_rows, headline, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaxGaffer — scene changed")
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(600)
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel(headline)
        head.setStyleSheet(f"color:{ACCENT};font-weight:600;letter-spacing:1px;")
        lay.addWidget(head)
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(["what", "before", "after", "why"])
        tree.setRootIsDecorated(True)
        tree.setColumnWidth(0, 240)

        def add_group(title, rows, fmt):
            if not rows:
                return
            top = QtWidgets.QTreeWidgetItem([f"{title} ({len(rows)})", "", "", ""])
            top.setForeground(0, QtGui.QBrush(QtGui.QColor(ACCENT)))
            tree.addTopLevelItem(top)
            for r in rows:
                top.addChild(QtWidgets.QTreeWidgetItem(fmt(r)))
            top.setExpanded(True)

        pr = plan_report or {"changes": [], "created": [], "warnings": []}
        if pr.get("effect"):
            eff = pr["effect"]
            worse = eff["after"] < eff["before"] - 5.0
            eff_lbl = QtWidgets.QLabel(
                f"plan effect (measured): critic {eff['before']:.1f} → {eff['after']:.1f}"
                + ("   ⚠ worse — one Ctrl+Z reverts the plan" if worse else ""))
            eff_lbl.setStyleSheet(f"color:{ERR};" if worse else f"color:{OK};")
            lay.addWidget(eff_lbl)
        add_group("Plan — values changed", pr["changes"], lambda c: [
            f"{c['target']} · {c['prop']}", str(c["before"]), str(c["after"]),
            c.get("why", "")])
        add_group("Plan — lights placed", pr["created"], lambda c: [
            f"{c['type']}  '{c['name']}'", "", c["at"], c.get("why", "")])
        add_group("Match loop — lighting values", state_rows, lambda c: [
            c["prop"], str(c["before"]), str(c["after"]), ""])
        add_group("Warnings", [{"w": w} for w in pr["warnings"]],
                  lambda c: [c["w"], "", "", ""])
        if tree.topLevelItemCount() == 0:
            tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(
                ["no changes were applied", "", "", ""]))
        lay.addWidget(tree)
        note = QtWidgets.QLabel("Plan = one Ctrl+Z · match loop states restorable via "
                                "'Restore pre-match light'.")
        note.setObjectName("dim")
        lay.addWidget(note)
        ok = QtWidgets.QPushButton("OK")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        lay.addWidget(ok)


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, cfg: cfgmod.Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaxGaffer — settings")
        self.setStyleSheet(STYLE)
        self.cfg = cfg
        form = QtWidgets.QFormLayout(self)
        self.ed_key = QtWidgets.QLineEdit(cfg.api_key)
        self.ed_key.setEchoMode(QtWidgets.QLineEdit.Password)
        form.addRow("oc_ API key", self.ed_key)
        self.ed_model = QtWidgets.QLineEdit(cfg.model)
        form.addRow("model", self.ed_model)
        self.ed_vantage = QtWidgets.QLineEdit(cfg.vantage_console)
        form.addRow("vantage_console.exe", self.ed_vantage)
        self.ed_syspy = QtWidgets.QLineEdit(cfg.system_python)
        self.ed_syspy.setPlaceholderText("optional: python.exe with Pillow (sidecar)")
        form.addRow("system python", self.ed_syspy)
        tune = QtWidgets.QHBoxLayout()
        self.sp_iters = QtWidgets.QSpinBox()
        self.sp_iters.setRange(1, 12)
        self.sp_iters.setValue(int(cfg.max_iterations))
        self.sp_target = QtWidgets.QDoubleSpinBox()
        self.sp_target.setRange(50.0, 100.0)
        self.sp_target.setValue(float(cfg.target_score))
        tune.addWidget(QtWidgets.QLabel("iterations"))
        tune.addWidget(self.sp_iters)
        tune.addWidget(QtWidgets.QLabel("target"))
        tune.addWidget(self.sp_target)
        form.addRow("match tuning", tune)
        res = QtWidgets.QHBoxLayout()
        self.sp_w = QtWidgets.QSpinBox()
        self.sp_w.setRange(160, 1920)
        self.sp_w.setValue(cfg.loop_width)
        self.sp_h = QtWidgets.QSpinBox()
        self.sp_h.setRange(90, 1080)
        self.sp_h.setValue(cfg.loop_height)
        res.addWidget(self.sp_w)
        res.addWidget(QtWidgets.QLabel("×"))
        res.addWidget(self.sp_h)
        form.addRow("loop render size", res)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setObjectName("dim")
        form.addRow(self.lbl_status)
        btns = QtWidgets.QHBoxLayout()
        b_test = QtWidgets.QPushButton("Test gateway")
        b_test.clicked.connect(self._test)
        btns.addWidget(b_test)
        b_ok = QtWidgets.QPushButton("Save")
        b_ok.setObjectName("primary")
        b_ok.clicked.connect(self._save)
        btns.addWidget(b_ok)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_cancel)
        form.addRow(btns)

    def _test(self):
        self.lbl_status.setText("pinging…")
        QtWidgets.QApplication.processEvents()
        try:
            self.lbl_status.setText(ping(self.ed_key.text().strip(),
                                         self.ed_model.text().strip()))
            self.lbl_status.setStyleSheet(f"color:{OK};")
        except OmegaError as e:
            self.lbl_status.setText(str(e))
            self.lbl_status.setStyleSheet(f"color:{ERR};")

    def _save(self):
        self.cfg.api_key = self.ed_key.text().strip()
        self.cfg.model = self.ed_model.text().strip() or "claude-opus-4-8"
        self.cfg.vantage_console = self.ed_vantage.text().strip()
        self.cfg.system_python = self.ed_syspy.text().strip()
        self.cfg.loop_width = int(self.sp_w.value())
        self.cfg.loop_height = int(self.sp_h.value())
        self.cfg.max_iterations = int(self.sp_iters.value())
        self.cfg.target_score = float(self.sp_target.value())
        self.accept()


_dock_instance: Optional[MaxGafferDock] = None


def show_dock():
    """Create (or raise) the dock inside 3ds Max's main window."""
    global _dock_instance
    parent = None
    try:
        import qtmax  # Max 2021+

        parent = qtmax.GetQMaxMainWindow()
    except Exception:
        parent = None
    if _dock_instance is not None:
        try:
            _dock_instance.show()
            _dock_instance.raise_()
            return _dock_instance
        except RuntimeError:
            _dock_instance = None
    if parent is not None:
        dock = QtWidgets.QDockWidget("MaxGaffer", parent)
        dock.setObjectName("MaxGafferDock")
        widget = MaxGafferDock(dock)
        dock.setWidget(widget)
        parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setFloating(True)
        dock.resize(1040, 1100)
        dock.show()
        _dock_instance = widget
    else:  # dev fallback: plain window
        _dock_instance = MaxGafferDock()
        _dock_instance.resize(1040, 1100)
        _dock_instance.show()
    return _dock_instance
