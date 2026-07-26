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
from ..core import providers
from ..core.omega import OmegaError, ping
from ..maxbridge import config as cfgmod
from ..maxbridge.controller import Controller

# SthyraDesign2 — the REFGRADE grading-bay language, ported to Qt. The desk stays NEUTRAL
# (every gray R=G=B, so the UI never biases color judgment next to Resolve); ONE signal
# color (mint green = live/active/go); depth from surface steps (bay→panel→raise), not
# shadows; hairline borders; 2px corners; mono instrument labels, sans prose.
SIGNAL = "#57e39a"          # the ONLY accent — live / active / go / selection / links
SIGNAL_HI = "#7aecb2"       # signal hover (brighter)
SIGNAL_LO = "#3fbe80"       # signal pressed (deeper)
SIGNAL_SEL = "rgba(87,227,160,0.22)"   # selection halo — readable under light text
SIGNAL_DIM = "rgba(87,227,160,0.16)"   # signal fills / armed tint
ALERT = "#f0a54a"           # amber — offline / attention (sparingly)
ACCENT = SIGNAL             # the one accent is signal green (name kept for call sites)
INK = "#0f1712"             # near-black text ON a signal fill
BG = "#171717"              # bay — surround / deepest surface / window
PANEL = "#1e1e1e"           # panels: instrument bar, command dock, cards, sheets
RAISE = "#262626"           # raised controls: buttons, chips, combos
RAISE_HI = "#2f2f2f"        # control hover
RAISE_LO = "#1a1a1a"        # control pressed / disabled
FILL = RAISE                # neutral controls (name kept for call sites)
INSET = "#121212"           # input / tree wells (a step under the bay)
WELLIMG = "#0a0a0a"         # image / plate wells (near black, so imagery pops)
LINE = "rgba(255,255,255,0.09)"    # hairline dividers
LINE2 = "rgba(255,255,255,0.16)"   # emphasized / focused borders
HAIR = f"1px solid {LINE}"
WELLINE = f"1px solid {LINE}"
TEXT = "#e7e7e7"            # primary text
DIM = "#9b9b9b"             # secondary values
FAINT = "#6a6a6a"           # labels, captions, idle glyph
ERR = "#ff7b72"             # coral — failure / notice
OK = "#e7e7e7"
RAD = "2px"                 # everything — tight, machined, instrument-like
UI = "'Segoe UI','SF Pro Text',system-ui,sans-serif"           # prose
MONO = "'JetBrains Mono','Cascadia Mono',Consolas,'SF Mono',monospace"  # instrument labels

STYLE = (
    # base is prose sans; labels/readouts/controls opt into mono below (the voice is the
    # contrast between clipped mono instrument labels and humane sans prose).
    f"QWidget{{background:{BG};color:{TEXT};font-family:{UI};font-size:12px;}}"
    f"QFrame#card{{background:{PANEL};border:{HAIR};border-radius:{RAD};}}"
    f"QLabel{{background:transparent;}}"
    f"QPushButton{{background:{RAISE};border:{HAIR};border-radius:{RAD};padding:6px 13px;"
    f"color:{TEXT};font-family:{MONO};letter-spacing:1px;}}"
    f"QPushButton::menu-indicator{{image:none;width:0;}}"
    f"QPushButton:hover{{background:{RAISE_HI};border-color:{LINE2};}}"
    f"QPushButton:pressed{{background:{RAISE_LO};}}"
    f"QPushButton:disabled{{background:{RAISE_LO};color:{FAINT};border-color:{LINE};}}"
    f"QPushButton#primary{{background:{SIGNAL};color:{INK};font-weight:700;border:none;}}"
    f"QPushButton#primary:hover{{background:{SIGNAL_HI};}}"
    f"QPushButton#primary:pressed{{background:{SIGNAL_LO};}}"
    f"QPushButton#primary:disabled{{background:{RAISE_LO};color:{FAINT};}}"
    f"QPushButton#ghost{{background:transparent;border:none;color:{SIGNAL};"
    f"padding:5px 8px;font-family:{MONO};letter-spacing:1px;}}"
    f"QPushButton#ghost:hover{{color:{SIGNAL_HI};}}"
    f"QPushButton#ghost:pressed{{color:{SIGNAL_LO};}}"
    f"QLineEdit,QTextEdit,QPlainTextEdit{{background:{INSET};"
    f"border:{WELLINE};border-radius:{RAD};padding:5px 8px;"
    f"selection-background-color:{SIGNAL_SEL};selection-color:{TEXT};}}"
    f"QSpinBox,QDoubleSpinBox{{background:{INSET};border:{WELLINE};border-radius:{RAD};"
    f"padding:5px 8px;font-family:{MONO};"
    f"selection-background-color:{SIGNAL_SEL};selection-color:{TEXT};}}"
    f"QLineEdit:focus,QTextEdit:focus,QPlainTextEdit:focus,QSpinBox:focus,"
    f"QDoubleSpinBox:focus{{border:1px solid {SIGNAL};}}"
    f"QComboBox{{background:{RAISE};border:{HAIR};border-radius:{RAD};padding:5px 10px;"
    f"font-family:{MONO};letter-spacing:1px;}}"
    f"QComboBox:hover{{background:{RAISE_HI};border-color:{LINE2};}}"
    f"QComboBox::drop-down{{border:none;width:18px;}}"
    f"QComboBox QLineEdit{{background:transparent;border:none;padding:0;}}"
    f"QComboBox QAbstractItemView{{background:{PANEL};border:{HAIR};padding:3px;"
    f"font-family:{MONO};selection-background-color:{SIGNAL_SEL};selection-color:{TEXT};}}"
    f"QMenu{{background:{PANEL};border:{HAIR};border-radius:{RAD};padding:5px;"
    f"font-family:{MONO};}}"
    f"QMenu::item{{padding:6px 24px;border-radius:{RAD};}}"
    f"QMenu::item:selected{{background:{SIGNAL_SEL};color:{TEXT};}}"
    f"QTreeWidget,QListWidget{{background:{INSET};border:{WELLINE};border-radius:{RAD};"
    f"padding:3px;font-family:{MONO};selection-background-color:{SIGNAL_SEL};"
    f"selection-color:{TEXT};outline:none;}}"
    f"QLabel#dim{{color:{DIM};}}"
    f"QLabel#h{{color:{TEXT};font-family:{MONO};font-weight:700;letter-spacing:3px;}}"
    f"QLabel#cap{{color:{FAINT};font-family:{MONO};font-size:10px;font-weight:600;"
    f"letter-spacing:2px;}}"
    f"QHeaderView::section{{background:{PANEL};color:{FAINT};border:none;"
    f"border-bottom:{HAIR};padding:5px 8px;font-family:{MONO};font-size:10px;"
    f"letter-spacing:1px;}}"
    f"QScrollBar:vertical{{background:transparent;width:10px;margin:2px;}}"
    f"QScrollBar::handle:vertical{{background:{RAISE};border-radius:{RAD};min-height:28px;}}"
    f"QScrollBar::handle:vertical:hover{{background:{RAISE_HI};}}"
    f"QScrollBar::add-line,QScrollBar::sub-line{{height:0;}}"
    f"QToolTip{{background:{PANEL};color:{TEXT};border:{HAIR};padding:5px;font-family:{MONO};}}"
)


def _cap(text: str) -> QtWidgets.QLabel:
    """Section caption — the one place small-caps styling lives."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("cap")
    return lbl


# ---------------------------------------------------------------- crash forensics + safe decode
_LOG_MIRROR = os.path.join(os.path.dirname(cfgmod.CONFIG_PATH), "last_session.log")


def _reset_log_mirror() -> None:
    """Truncate the crash-forensics log at dock open. A native Max crash bypasses every
    Python try/except — but every log line is mirrored+flushed here, so after a crash the
    LAST line of %LOCALAPPDATA%/MaxGaffer/last_session.log names the step that died."""
    try:
        with open(_LOG_MIRROR, "w", encoding="utf-8") as f:
            f.write("MaxGaffer session log (crash forensics — last line = last step)\n")
    except OSError:
        pass


def _mirror_log(msg: str) -> None:
    try:
        with open(_LOG_MIRROR, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _bounded_pixmap(path: str, target: QtCore.QSize) -> QtGui.QPixmap:
    """Decode an image for a thumbnail WITHOUT the full-resolution transient — a
    QPixmap(path) on a 50-100 MP reference spikes 0.5-1+ GB, which is an OOM-crash risk
    at match end on a box already loaded with V-Ray + Vantage. QImageReader scales JPEGs
    DURING decode (never materializes full res); anything still huge is rejected."""
    reader = QtGui.QImageReader(path)
    reader.setDecideFormatFromContent(True)
    reader.setAutoTransform(True)       # phone references must preview with EXIF orientation
    size = reader.size()
    if size.isValid():
        # >120 MP even defeats a scaled decode's scratch buffers on some formats — skip
        if size.width() * size.height() > 120_000_000:
            return QtGui.QPixmap()
        reader.setScaledSize(size.scaled(target, QtCore.Qt.KeepAspectRatio))
    img = reader.read()
    if img.isNull():
        return QtGui.QPixmap()
    return QtGui.QPixmap.fromImage(img)


class _Worker(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except BaseException as e:  # noqa: BLE001 — SystemExit must still free every waiter
            self.failed.emit(str(e))


class _ProgressRelay(QtCore.QObject):
    """Marshals worker-thread progress callbacks onto the main thread — Qt widgets must
    never be touched from a vantage_console watcher thread."""

    progress = QtCore.Signal(str, str)
    #: stage label, done, total, overall percent — emitted from the match worker thread
    match_progress = QtCore.Signal(str, int, int, float)


class MaxGafferDock(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaxGaffer")
        self.setStyleSheet(STYLE)
        self.cfg = cfgmod.load()
        self.ctrl = Controller(self.cfg)
        self.ctrl.io = self._run_blocking_io   # gateway waits run off-thread, Max stays alive
        self._relay = _ProgressRelay(self)
        self._relay.match_progress.connect(self._on_match_progress)
        self._workers: List[_Worker] = []
        self._cancel = False
        self._busy = False
        self._beat = None
        self._last_tick = ""
        self._active_camera = ""
        self._active_camera_id = ""
        self._camera_fingerprint = ()
        self._sliders: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        _reset_log_mirror()   # crash forensics: last_session.log starts fresh per dock
        self._build()
        self.refresh_cameras()
        self._camera_timer = QtCore.QTimer(self)
        self._camera_timer.setInterval(1000)
        self._camera_timer.timeout.connect(self._poll_cameras)
        self._camera_timer.start()
        self._recover_draft_snapshot()
        app = QtWidgets.QApplication.instance()
        if app is not None:                      # drain workers before Qt tears down
            app.aboutToQuit.connect(self._drain_workers)

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
        btn_refresh_cams = QtWidgets.QPushButton("↻")
        btn_refresh_cams.setFixedWidth(34)
        btn_refresh_cams.setToolTip("Refresh cameras after adding, deleting or renaming shots.")
        btn_refresh_cams.clicked.connect(self.refresh_cameras)
        head.addWidget(btn_refresh_cams)
        # RESET is not the camera re-scan beside it and must not be mistaken for it: that
        # one is cheap and idempotent, this one throws work away. Neutral, never signal —
        # signal means go, and this is the opposite of go.
        self.btn_reset = QtWidgets.QPushButton("RESET")
        self.btn_reset.setFixedWidth(62)
        self.btn_reset.setToolTip(
            "Start fresh — put every camera's light back the way it was, remove the "
            "exposure control MaxGaffer created, and forget all references, readings, "
            "notes and locks. Asks first.")
        self.btn_reset.clicked.connect(self._on_reset)
        head.addWidget(self.btn_reset)
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
            t.setStyleSheet(f"background:{WELLIMG};border:{WELLINE};border-radius:{RAD};"
                            f"color:{FAINT};font-family:{MONO};letter-spacing:1px;")
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
        btn_add_ref = QtWidgets.QPushButton("Add reference…")
        btn_add_ref.setToolTip(
            "Bind an EXTRA observed angle (the first reference stays the primary — swap "
            "it above). Extra views denoise the reference read; they do NOT reconstruct "
            "unseen geometry, so the single-view caveats still hold.")
        btn_add_ref.clicked.connect(self._add_reference)
        side.addWidget(btn_add_ref)
        self.lbl_ref_info = QtWidgets.QLabel("")
        self.lbl_ref_info.setObjectName("dim")
        self.lbl_ref_info.setWordWrap(True)
        side.addWidget(self.lbl_ref_info, 1)
        thumbs.addLayout(side, 1)
        lr.addLayout(thumbs)

        # ---- multi-reference roster + honest FAIRNESS readout (additive; the primary
        # still binds/swaps above via _pick_reference). Both reveal themselves only when
        # there is something to show — a single reference stays clutter-free.
        refs_row = QtWidgets.QHBoxLayout()
        self.ref_list_cap = _cap("REFERENCES")
        refs_row.addWidget(self.ref_list_cap)
        refs_row.addStretch(1)
        self.btn_ref_remove = QtWidgets.QPushButton("Remove")
        self.btn_ref_remove.setObjectName("ghost")
        self.btn_ref_remove.setToolTip("Remove the selected reference. Removing the "
                                       "primary promotes the next one.")
        self.btn_ref_remove.clicked.connect(self._remove_reference)
        refs_row.addWidget(self.btn_ref_remove)
        lr.addLayout(refs_row)
        self.ref_list = QtWidgets.QListWidget()
        self.ref_list.setMaximumHeight(78)
        self.ref_list.setToolTip("Every reference bound to this camera (role · file). The "
                                 "first is the primary the solve uses; the rest denoise "
                                 "the reference read (Route A).")
        lr.addWidget(self.ref_list)
        self.lbl_fairness = QtWidgets.QLabel("")
        self.lbl_fairness.setWordWrap(True)
        self.lbl_fairness.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_fairness.setToolTip("Can the reference actually constrain this scene? A "
                                     "read-only honesty check from the last match — it "
                                     "advises LOCK, never proposes values.")
        lr.addWidget(self.lbl_fairness)
        self.ref_list_cap.setVisible(False)
        self.btn_ref_remove.setVisible(False)
        self.ref_list.setVisible(False)
        self.lbl_fairness.setVisible(False)

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
        self.cmb_mode.addItems(["Standard", "Hero → 99", "Loop only", "Fast"])
        self.cmb_mode.setToolTip(
            "Standard — scene-wide plan + balanced match loop.\n"
            "Hero → 99 — bounded plan + loop + coordinate polish (under 100 renders).\n"
            "Loop only — skip the scene-wide plan.\n"
            "Fast — lower-resolution preview with a small render budget.")
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
                                 ("Accept", lambda: self._artist_feedback(True),
                                  "Record that the artist accepts this result."),
                                 ("Needs work", lambda: self._artist_feedback(False),
                                  "Record that the score did not satisfy the artist."),
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
        # ---- PROGRESS readout. A match runs for minutes and the log alone cannot tell a
        # working run from a hung one. Instrument, not toy: mono stage label on the left,
        # tabular probe count and percent on the right, and a hairline meter that only
        # moves because work was actually done (SthyraDesign2 loading/processing).
        self.progress_row = QtWidgets.QWidget()
        pr = QtWidgets.QVBoxLayout(self.progress_row)
        pr.setContentsMargins(0, 2, 0, 0)
        pr.setSpacing(5)
        ptop = QtWidgets.QHBoxLayout()
        ptop.setSpacing(8)
        self.lbl_stage = QtWidgets.QLabel("IDLE")
        self.lbl_stage.setStyleSheet(
            f"font-family:{MONO};font-size:9px;letter-spacing:.14em;color:{DIM};")
        self.lbl_pct = QtWidgets.QLabel("")
        self.lbl_pct.setStyleSheet(
            f"font-family:{MONO};font-size:9px;letter-spacing:.10em;color:{SIGNAL};")
        self.lbl_pct.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ptop.addWidget(self.lbl_stage)
        ptop.addStretch(1)
        ptop.addWidget(self.lbl_pct)
        pr.addLayout(ptop)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)          # tenths of a percent — the meter never jumps
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(3)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:{INSET};border:none;border-radius:1px;}}"
            f"QProgressBar::chunk{{background:{SIGNAL};border-radius:1px;}}")
        pr.addWidget(self.bar)
        self.progress_row.setVisible(False)
        lc.addWidget(self.progress_row)

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
        top.setForeground(0, QtGui.QBrush(QtGui.QColor(TEXT)))
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
        cam = self._current_camera()
        entry = self._camera_entry(cam) if cam else None
        card = getattr(entry, "scorecard", {}) if entry is not None else {}
        if card:
            group = QtWidgets.QTreeWidgetItem([
                f"Scorecard · {card.get('confidence', 'unknown')} confidence",
                f"coverage {float(card.get('coverage', 0)):.0%}",
                "content gap" if card.get("content_gap") else "lighting gap"])
            for key, value in sorted((card.get("components") or {}).items()):
                group.addChild(QtWidgets.QTreeWidgetItem(
                    [key, "", f"{float(value) * 100:.0f}%"]))
            group.addChild(QtWidgets.QTreeWidgetItem(
                [str(card.get("disclaimer") or "proxy score—not an artist verdict"), "", ""]))
            top.addChild(group)
        top.setExpanded(True)

    # ================================================================= helpers
    def _on_reset(self):
        """Undo everything MaxGaffer did and forget everything it learned."""
        if self._busy:
            self._log("busy — reset ignored until the current run finishes")
            return
        cams = list(getattr(self.ctrl.session, "cameras", {}) or {})
        refs = sum(1 for e in (getattr(self.ctrl.session, "cameras", {}) or {}).values()
                   if getattr(e, "reference", ""))
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Start fresh")
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setText("Put the scene back and forget everything?")
        # Spell out both halves. "Reset" means nothing on its own, and the scene half is
        # the one an artist would be upset to discover afterwards.
        box.setInformativeText("\n".join([
            "This will:",
            "  · restore each camera's light to the snapshot taken before its first match",
            "  · remove the exposure control MaxGaffer created, and undo any dome seed",
            f"  · forget {len(cams)} camera record(s), {refs} bound reference(s), and every",
            "    cached reading, note, lock and score",
            "",
            "Your scene geometry, materials and cameras are untouched, and the light "
            "multipliers MaxGaffer measured from your own rig are kept.",
            "",
            "This cannot be undone from inside MaxGaffer.",
        ]))
        box.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Reset)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if box.exec() != QtWidgets.QMessageBox.Reset:
            self._log("reset cancelled — nothing changed")
            return
        self.log.setVisible(True)
        self._log("— reset —")
        try:
            out = self.ctrl.start_fresh(log=self._log)
        except Exception as err:  # noqa: BLE001 — report, never take the dock down
            self._log(f"⚠ reset failed: {err}")
            return
        for cam in out.get("restored", []):
            self._log(f"restored {cam}")
        for cam, why in out.get("failed", []):
            self._log(f"⚠ {cam} NOT restored ({why}) — its snapshot was kept, try again")
        self._clear_after_reset()
        self._log("✓ fresh — bind a reference and match when ready")

    def _clear_after_reset(self):
        """Empty the panels too. Leaving a stale thumbnail or change list beside an emptied
        session is how an artist ends up trusting a reading that no longer exists."""
        self.changes_tree.clear()
        self.progress_row.setVisible(False)
        self.bar.setValue(0)
        self.lbl_stage.setText("IDLE")
        self.lbl_pct.setText("")
        for attr in ("thumb_ref", "thumb_match"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.clear()
                except Exception:  # noqa: BLE001
                    pass
        self.refresh_cameras()
        self._refresh_reference_panel(self._current_camera())

    def _on_match_progress(self, stage: str, done: int, total: int, pct: float):
        """Main-thread slot. Qt widgets are never touched from the worker."""
        self.lbl_stage.setText(str(stage).upper())
        self._last_tick = "%d/%d   %3d%%   " % (done, total, int(pct))
        self.lbl_pct.setText(self._last_tick + self._clock())
        self.bar.setValue(int(round(pct * 10)))

    def _clock(self) -> str:
        if not hasattr(self, "_elapsed"):
            return ""
        secs = int(self._elapsed.elapsed() / 1000)
        return "%d:%02d" % (secs // 60, secs % 60)

    def _progress_begin(self, what: str):
        self.lbl_stage.setText(str(what).upper())
        self.lbl_pct.setText("0:00")
        self.bar.setValue(0)
        self.progress_row.setVisible(True)
        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()
        self._last_tick = ""
        if getattr(self, "_beat", None) is None:
            self._beat = QtCore.QTimer(self)
            self._beat.setInterval(1000)
            self._beat.timeout.connect(self._on_heartbeat)
        self._beat.start()

    def _on_heartbeat(self):
        """Proves the run is alive when nothing is rendering.

        The pre-render phases are network waits — three ANALYZE image calls and a plan call
        — during which no probe completes, the CPU sits near zero and a render-driven meter
        would show a frozen bar. A clock that keeps counting is the difference between
        'working' and 'hung', and it costs one timer."""
        if not self._busy or not hasattr(self, "_elapsed"):
            return
        secs = int(self._elapsed.elapsed() / 1000)
        clock = "%d:%02d" % (secs // 60, secs % 60)
        self.lbl_pct.setText(f"{self._last_tick}{clock}" if self._last_tick else clock)

    def _progress_stage(self, what: str):
        """Name a phase that renders nothing, so the label is never stale during a wait."""
        self.lbl_stage.setText(str(what).upper())
        self.progress_row.setVisible(True)

    def _progress_end(self, note: str = ""):
        if getattr(self, "_beat", None) is not None:
            self._beat.stop()
        # Left ON at 100 rather than hidden: after a long run the artist wants to see that
        # it finished, not an empty space where the meter was.
        self.bar.setValue(1000)
        self.lbl_stage.setText("DONE")
        self.lbl_pct.setText(note or "100%")

    def _log(self, msg: str):
        _mirror_log(msg)
        if msg.startswith("THUMB::"):
            url = QtCore.QUrl.fromLocalFile(msg[len("THUMB::"):]).toString()
            self.log.append(f'<img src="{url}" width="240">')
        else:
            self.log.append(_html.escape(msg))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        QtWidgets.QApplication.processEvents()

    def _run_blocking_io(self, fn):
        """Run pure-I/O ``fn`` on a worker while the MAIN thread pumps events in a
        wait-poll — Max stays alive, pymxs is never touched off-thread, exceptions
        re-raise here, and Cancel (or an io fn that never returns) can never wedge
        the nested event loop forever."""
        box = {}
        w = _Worker(fn, self)
        self._workers.append(w)
        w.done.connect(lambda r: box.__setitem__("r", r))
        w.failed.connect(lambda e: box.__setitem__("e", e))
        w.start()
        while not w.wait(100):
            QtWidgets.QApplication.processEvents()
            if self._cancel:
                break                        # stop blocking; the worker finishes off-thread
        QtWidgets.QApplication.processEvents()   # drain the queued done/failed signal
        if w.isRunning():                    # cancelled mid-io — keep tracking it so the
            w.finished.connect(lambda: self._discard_worker(w))  # dock drains it on close
        else:
            self._workers.remove(w)
        if self._cancel and "r" not in box and "e" not in box:
            raise RuntimeError("cancelled")
        if "e" in box:
            raise RuntimeError(box["e"])
        return box.get("r")

    def _discard_worker(self, w):
        if w in self._workers:
            self._workers.remove(w)

    def _drain_workers(self):
        """Max is exiting (or the dock is closing) — a QThread finalized while running is
        a hard 'QThread: Destroyed while thread is still running' abort, so quit+wait."""
        self._cancel = True
        for w in list(self._workers):
            try:
                w.quit()
                w.wait(3000)
            except RuntimeError:
                pass                         # C++ object already gone

    def closeEvent(self, event):
        self._drain_workers()
        super().closeEvent(event)

    def _current_camera(self) -> str:
        data = self.cam_combo.currentData() if hasattr(self, "cam_combo") else None
        return data or ""
    def _current_camera_id(self) -> str:
        if not hasattr(self, "cam_combo") or self.cam_combo.currentIndex() < 0:
            return ""
        return str(self.cam_combo.currentData(QtCore.Qt.UserRole + 1) or "")
    def _camera_entry(self, cam: str):
        if hasattr(self.ctrl, "camera_entry"):
            return self.ctrl.camera_entry(cam)
        return self.ctrl.session.cameras.get(cam)
    def _find_camera_index(self, name: str, camera_id: str = "") -> int:
        for i in range(self.cam_combo.count()):
            if self.cam_combo.itemData(i) == name:
                item_id = str(self.cam_combo.itemData(i, QtCore.Qt.UserRole + 1) or "")
                if not camera_id or item_id == str(camera_id):
                    return i
        return -1
    def _poll_cameras(self):
        if self._busy or not hasattr(self.ctrl, "camera_fingerprint"):
            return
        try:
            fingerprint = self.ctrl.camera_fingerprint()
        except Exception:
            return
        if fingerprint != self._camera_fingerprint:
            self.refresh_cameras()
    # ================================================================= cameras
    def refresh_cameras(self):
        current = self._current_camera()
        current_id = self._current_camera_id()
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        try:
            cams = self.ctrl.cameras()
        except Exception as e:  # noqa: BLE001
            self._log(f"camera scan failed: {e}")
            cams = []
        active_cam = next((c for c in cams if c.get("active")), {})
        active = active_cam.get("name", "")
        active_id = str(active_cam.get("id", ""))
        for c in cams:
            mark = "●  " if c.get("reference") else "○  "
            score = f"   ·  {c['score']:.0f}" if c.get("score") is not None else ""
            duplicate = (f"   [id {str(c.get('id') or '?')[-6:]}]"
                         if c.get("duplicate") else "")
            self.cam_combo.addItem(mark + c["name"] + score + duplicate, c["name"])
            self.cam_combo.setItemData(self.cam_combo.count() - 1, str(c.get("id") or ""),
                                       QtCore.Qt.UserRole + 1)
        idx = self._find_camera_index(current, current_id)
        if idx < 0:
            idx = self._find_camera_index(active, active_id)
        self.cam_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.cam_combo.blockSignals(False)
        self._active_camera = self._current_camera()
        self._active_camera_id = self._current_camera_id()
        self._camera_fingerprint = tuple((str(c.get("id", "")), c.get("name", ""),
                                          c.get("class", "")) for c in cams)
        try:
            self.act_apply_select.setChecked(
                bool(self.ctrl.session.settings.get("apply_on_select", True)))
        except Exception:
            pass
        self._sync_score_badge()
        self.rebuild_rig_controls()
        self._rebuild_locks(self._current_camera())
        self._show_reference(self._current_camera())
        self._refresh_reference_panel(self._current_camera())

    def _sync_score_badge(self):
        e = self._camera_entry(self._current_camera())
        self.lbl_score.setText(f"{e.score:.1f}" if (e and e.score is not None) else "—")
    def _on_camera_combo(self, _idx: int):
        if self._busy:
            self._log("busy — camera switch ignored until the current run finishes")
            idx = self._find_camera_index(self._active_camera, self._active_camera_id)
            if idx >= 0:                     # point the combo back at the camera actually running
                self.cam_combo.blockSignals(True)
                self.cam_combo.setCurrentIndex(idx)
                self.cam_combo.blockSignals(False)
            return
        name = self._current_camera()
        camera_id = self._current_camera_id()
        if not name:
            return
        previous = self._active_camera
        self._ab_on_pre = False
        try:
            for w in self.ctrl.select_camera(name, camera_id=camera_id):
                self._log("⚠ " + w)
        except Exception as e:  # noqa: BLE001
            self._log(f"select failed: {e}")
            idx = self._find_camera_index(previous, self._active_camera_id)
            if idx >= 0:
                self.cam_combo.blockSignals(True)
                self.cam_combo.setCurrentIndex(idx)
                self.cam_combo.blockSignals(False)
            return
        self._active_camera = name
        self._active_camera_id = camera_id
        self.match_thumb.setPixmap(QtGui.QPixmap())
        self.match_thumb.setText("no match preview")
        self._show_reference(name)
        self._refresh_reference_panel(name)
        self._rebuild_locks(name)
        self._sync_score_badge()
        self.rebuild_rig_controls()
    def _on_apply_on_select(self, checked: bool):
        self.ctrl.session.settings["apply_on_select"] = bool(checked)
        self.ctrl.save_session()

    # ================================================================= reference
    def _pick_reference(self):
        if self._busy:
            self._log("busy — reference swap ignored until the current run finishes")
            return
        cam = self._current_camera()
        if not cam:
            self._log("select a camera first")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Reference for {cam}", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.exr *.hdr);;"
            "All files (*.*)")
        if not path:
            return
        if not os.path.isfile(path):
            self._log(f"reference file not found: {path}")
            return
        if hasattr(self.ctrl, "bind_reference"):
            self.ctrl.bind_reference(cam, path)
        else:
            self.ctrl.session.set_reference(cam, path)
        if not self.ctrl.save_session():
            self._log("⚠ scene not saved yet — bindings live in memory only until you "
                      "save the .max file")
        self._show_reference(cam)
        self.refresh_cameras()

    def _show_reference(self, cam: str):
        e = self._camera_entry(cam)
        ref = e.reference if e else ""
        if ref and os.path.isfile(ref):
            pix = _bounded_pixmap(ref, self.ref_thumb.size())
            if not pix.isNull():
                self.ref_thumb.setPixmap(pix)
                info = os.path.basename(ref)
                if e and e.semantics:
                    s = e.semantics
                    info += f"\n{s.get('time_of_day')}, {s.get('sky')} sky"
                    wb = s.get("wb_kelvin_estimate")
                    if isinstance(wb, (int, float)):   # sidecar is human-editable —
                        info += f", wb ~{wb:.0f}K"     # only format real numerics
                if e and e.score is not None:
                    info += f"\nlast match: {e.score:.1f}/100 at {e.matched_at}"
                self.lbl_ref_info.setText(info)
                return
            self.ref_thumb.setPixmap(QtGui.QPixmap())
            self.ref_thumb.setText("preview unavailable")
            self.lbl_ref_info.setText(
                os.path.basename(ref) + "\nQt cannot preview this format; MaxGaffer will "
                "still try Pillow / 3ds Max bitmap ingestion when matching.")
            return
        self.ref_thumb.setPixmap(QtGui.QPixmap())
        if ref:
            self.ref_thumb.setText("reference missing")
            self.lbl_ref_info.setText(os.path.basename(ref) +
                                      "\nFile moved or was deleted — load it again.")
        else:
            self.ref_thumb.setText("no reference")
            self.lbl_ref_info.setText("Bind a lighting reference image to this camera.")

    # ------------------------------------------------------------- multi-reference + fairness
    def _add_reference(self):
        """Bind an EXTRA observed angle to this camera (Route A: denoises the reference
        read; does NOT reconstruct unseen geometry). The primary is unchanged — swap it
        with 'Load / swap reference…' above."""
        if self._busy:
            self._log("busy — reference add ignored until the current run finishes")
            return
        cam = self._current_camera()
        if not cam:
            self._log("select a camera first")
            return
        if not hasattr(self.ctrl, "add_reference"):
            self._log("this build has no multi-reference support — use Load / swap reference")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Add reference for {cam}", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.exr *.hdr);;"
            "All files (*.*)")
        if not path:
            return
        if not os.path.isfile(path):
            self._log(f"reference file not found: {path}")
            return
        try:
            self.ctrl.add_reference(cam, path)
        except Exception as e:  # noqa: BLE001 — a stale rig must not escape the slot
            self._log(f"✗ add reference: {e}")
            return
        if not self.ctrl.save_session():
            self._log("⚠ scene not saved yet — bindings live in memory only until you "
                      "save the .max file")
        self._log(f"added reference: {os.path.basename(path)}")
        self._refresh_reference_panel(cam)

    def _remove_reference(self):
        cam = self._current_camera()
        if self._busy:
            self._log("busy — reference remove ignored until the current run finishes")
            return
        if not cam:
            return
        if not hasattr(self.ctrl, "remove_reference"):
            self._log("this build has no multi-reference support")
            return
        item = self.ref_list.currentItem()
        if item is None:
            self._log("select a reference in the list to remove")
            return
        ref = item.data(QtCore.Qt.UserRole)      # signature (stable) or path fallback
        try:
            removed = bool(self.ctrl.remove_reference(cam, ref))
        except Exception as e:  # noqa: BLE001
            self._log(f"✗ remove reference: {e}")
            return
        if not removed:
            self._log("reference not found")
            return
        self.ctrl.save_session()
        self._log("reference removed")
        self.refresh_cameras()    # the primary may have changed → re-mirror thumb + panel

    def _refresh_reference_panel(self, cam: str):
        """Additive companion to _show_reference: repaint the reference roster and the
        fairness readout. Never raises — a controller without the multi-reference surface,
        a stale entry, or an older run with no fairness must all degrade silently."""
        try:
            self._refresh_reference_list(cam)
        except Exception:                        # noqa: BLE001 — UI-only, never fatal
            pass
        try:
            self._show_fairness(cam)
        except Exception:                        # noqa: BLE001
            pass

    def _refresh_reference_list(self, cam: str):
        self.ref_list.clear()
        refs: List[Dict] = []
        if cam and hasattr(self.ctrl, "references"):
            refs = list(self.ctrl.references(cam) or [])
        for r in refs:
            role = str(r.get("role") or "ref")
            base = os.path.basename(str(r.get("path") or "")) or "—"
            item = QtWidgets.QListWidgetItem(f"{role} · {base}")
            item.setData(QtCore.Qt.UserRole,
                         str(r.get("signature") or r.get("path") or ""))
            if not bool(r.get("has_semantics")):
                item.setForeground(QtGui.QBrush(QtGui.QColor(DIM)))
            self.ref_list.addItem(item)
        # single reference is the DEFAULT — keep the roster hidden until an extra angle
        # is added (the primary always shows in the plate + info above)
        show = len(refs) > 1
        self.ref_list_cap.setVisible(show)
        self.ref_list.setVisible(show)
        self.btn_ref_remove.setVisible(show)

    def _show_fairness(self, cam: str):
        """Read the last match's honest fairness verdict off the scorecard (C5-shaped, so
        the full assess() dict and the fallback both render), per-sub-field null-guarded.
        No 'fairness' key (older runs) → the badge stays hidden. Read-only: it names the
        gap and advises LOCK, it never proposes values."""
        entry = self._camera_entry(cam) if cam else None
        card = getattr(entry, "scorecard", {}) if entry is not None else {}
        fair = (card or {}).get("fairness") or {}
        if not fair:
            self.lbl_fairness.clear()
            self.lbl_fairness.setVisible(False)
            return
        verdict = str(fair.get("verdict") or "unknown").lower()
        # mint = go/fair · amber = attention/marginal · coral = poor · faint = unknown
        color = {"fair": SIGNAL, "marginal": ALERT, "unfair": ERR}.get(verdict, FAINT)
        reasons: List[str] = []
        c = fair.get("constrainable")
        if isinstance(c, (int, float)):
            reasons.append(f"constrainable {max(0.0, min(1.0, float(c))):.0%}")
        ev = fair.get("predicted_ev_gap")
        wb = fair.get("predicted_wb_gap")
        if (isinstance(ev, (int, float)) and isinstance(wb, (int, float))
                and (float(ev) > 0.05 or float(wb) > 1.0)):
            reasons.append(f"predicted gap ~{float(ev):.1f} stops · ~{float(wb):.0f}K")
        if fair.get("same_scene"):
            reasons.append("same scene — exposure / WB only")
        parts = [f'<span style="font-family:{MONO};color:{color};font-weight:700;">'
                 f'FAIRNESS · {_html.escape(verdict.upper())}</span>']
        if reasons:
            parts.append(f'<span style="color:{DIM};">'
                         f'{_html.escape(" · ".join(reasons))}</span>')
        remedy = fair.get("remedy")
        if remedy:
            parts.append(f'<span style="color:{color};">{_html.escape(str(remedy))}</span>')
        for caveat in list(fair.get("unreconstructable", []) or [])[:3]:
            parts.append(f'<span style="color:{FAINT};">— {_html.escape(str(caveat))}</span>')
        self.lbl_fairness.setText("<br>".join(parts))
        self.lbl_fairness.setVisible(True)

    def _rebuild_locks(self, cam: str):
        self.lock_menu.clear()
        e = self._camera_entry(cam)
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
        try:
            how = self.ctrl.set_dome_hdri(path)
        except Exception as e:  # noqa: BLE001 — a stale rig (deleted dome) must not
            self._log(f"✗ HDRI: {e}")        # escape the slot
            self.rebuild_rig_controls()
            return
        self._log(f"dome HDRI → {os.path.basename(path)} ({how})" if how != "failed"
                  else "✗ could not set the dome texture (no dome, or unknown file prop — "
                       "checklist #16)")
        if how != "failed":
            # a manual pick outranks the seed — otherwise the next camera switch would
            # silently re-bind the seed over the artist's explicit choice
            cam = self._current_camera()
            e = self._camera_entry(cam) if cam else None
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
        e = self._camera_entry(cam)
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
        if self._busy:
            self._log("busy — preset save ignored until the current run finishes")
            return
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
        e = self._camera_entry(cam)
        if not (e and e.reference):
            self._log("bind a reference image first")
            return
        if not self.cfg.api_key and self.cfg.semantic_provider not in (
                "offline", "openai_compatible", "local"):
            self._log("no semantic API key — continuing with the reduced-intelligence "
                      "offline analytic solver")
        self._busy = True
        self._cancel = False
        for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        mode = self.cmb_mode.currentIndex()       # 0 standard · 1 hero · 2 loop-only · 3 fast
        self.cfg.auto_execute_plan = self.act_autoexec.isChecked()
        self.cfg.draft_sampler = self.act_draft.isChecked()
        self.log.clear()
        self._progress_begin("reading the reference")
        self._log(f"— match: {cam} —")
        self._log("· the reference is read by the gateway first — this is a network wait, "
                  "not a render, so the clock moves and the meter does not")
        plan_report = None
        try:
            if mode != 2:
                try:
                    self._progress_stage("planning")
                    plan = self.ctrl.make_plan(cam, log=self._log)
                except (OmegaError, RuntimeError) as err:
                    self._log(f"⚠ plan skipped ({err}) — continuing with the match loop")
                    plan = None
                # None = junk plan reply twice (controller already logged it) —
                # the match proceeds plan-less
                ops, lines, meta = plan[:3] if plan is not None else ([], [], {})
                if not ops:
                    self._log("plan: no operations proposed — continuing to the match loop")
                elif self.act_autoexec.isChecked() or PlanPreviewDialog(
                        lines, meta, self).exec():
                    self._log(f"— executing plan ({len(ops)} ops) —")
                    plan_report = self.ctrl.execute_plan(ops, cam, log=self._log)
                else:
                    self._log("plan declined — continuing with the match loop only")
            self._progress_stage("matching")
            result = self.ctrl.run_match(
                cam, log=self._log,
                should_cancel=lambda: self._cancel,
                on_progress=lambda st, d, t, p: self._relay.match_progress.emit(
                    st, d, t, p),
                locks=self._locks(),
                do_sweep=self.act_sweep.isChecked(),
                deep=(mode == 1),
                quality_profile=("fast" if mode == 3 else
                                 "hero" if mode == 1 else "standard"))
            score = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
            ceiling = ""
            if (result.best_score or 0) < 99:
                if getattr(result, "ceiling_proven", False):
                    ceiling = " · ceiling proven — the gap left is content, not lighting"
                elif result.ceiling_converged:
                    ceiling = " · plateau (finer steps untested — not a proven ceiling)"
            self._progress_end(f"score {score}")
            self._log(f"✓ done ({result.stop_reason}) — best {score}{ceiling}")
            self._set_match_thumb(result.best_render)
            headline = f"{cam} — {result.stop_reason}, score {score}"
            self._fill_changes(plan_report, self.ctrl.state_change_rows(cam), headline)
            if self.act_popup.isChecked():
                self._log("· showing the change report…")   # crash breadcrumb
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
            self._log("· match UI settled")   # crash breadcrumb: last healthy step
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
        """First press asks the run to stop; a second press releases the controls.

        Cancel can only ever be a REQUEST — it sets a flag the running code has to notice,
        and between checks there are long uninterruptible stretches (a gateway round trip,
        a V-Ray frame). That is fine when the run is healthy. It is useless when the run is
        wedged somewhere that never looks at the flag again, and then the dock stays locked
        with no way out but reloading it, which is what an artist experiences as "stuck".

        So the second press stops pretending. It hands the controls back. It does NOT
        pretend to have killed anything — the worker may still be alive, and the log says
        so plainly, because a released UI that silently leaves work running is its own
        kind of lie."""
        if not self._cancel:
            self._cancel = True
            self._log("cancelling after the current step… (press ✕ again to force the "
                      "controls back if it does not respond)")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Release the controls?")
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setText("This run is not responding to cancel.")
        box.setInformativeText("\n".join([
            "Releasing hands the buttons back so you can carry on.",
            "",
            "It does NOT stop whatever is still running — a gateway call or a render may "
            "finish in the background, and if it does its result is discarded.",
            "",
            "If this keeps happening, reload the dock with scripts/reload_dock.py.",
        ]))
        box.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        box.setDefaultButton(QtWidgets.QMessageBox.Ok)
        if box.exec() != QtWidgets.QMessageBox.Ok:
            return
        self._force_release("forced — the controls are yours again; anything still running "
                            "in the background will be discarded when it lands")

    def _force_release(self, why: str):
        self._busy = False
        self._cancel = False
        if getattr(self, "_beat", None) is not None:
            self._beat.stop()
        for b in (self.btn_match, self.btn_match_all, self.btn_refine, self.btn_board):
            b.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_stage.setText("RELEASED")
        self.lbl_pct.setText("")
        self._log("⚠ " + why)

    def _set_match_thumb(self, path):
        if path and os.path.exists(path):
            pix = _bounded_pixmap(path, self.match_thumb.size())
            if not pix.isNull():
                self.match_thumb.setPixmap(pix)
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
        e = self._camera_entry(cam)
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
                                      should_cancel=lambda: self._cancel,
                                      locks=self._locks())
            score = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
            ceiling = ""
            if (result.best_score or 0) < 99:
                if getattr(result, "ceiling_proven", False):
                    ceiling = " · ceiling proven — the gap left is content, not lighting"
                elif result.ceiling_converged:
                    ceiling = " · plateau (finer steps untested — not a proven ceiling)"
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
        if self._busy:
            self._log("busy — run folder opens when the current run finishes")
            return
        d = self.ctrl._run_dir or cfgmod.sessions_dir()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))

    def _restore_pre_match(self):
        if self._busy:
            return
        cam = self._current_camera()
        try:
            if cam and self.ctrl.restore_pre_match(cam):
                self._log(f"restored pre-match lighting for {cam}")
                self._ab_on_pre = True
            else:
                self._log("no pre-match snapshot for this camera yet")
        except Exception as e:  # noqa: BLE001 — a light deleted since the snapshot must
            self._log(f"✗ restore: {e}")     # not escape the slot half-way
        finally:
            self.rebuild_rig_controls()

    def _ab_flip(self):
        if self._busy:
            return
        cam = self._current_camera()
        e = self._camera_entry(cam) if cam else None
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

    def _artist_feedback(self, accepted: bool):
        cam = self._current_camera()
        if not cam:
            self._log("select a camera before recording feedback")
            return
        try:
            self.ctrl.record_artist_feedback(cam, accepted)
            self._log("artist verdict recorded: " + ("accepted" if accepted else "needs work"))
        except Exception as err:  # noqa: BLE001
            self._log(f"could not record artist feedback: {err}")

    # ================================================================= vantage
    def _start_live_link(self):
        if self._busy:
            self._log("busy — live-link toggle ignored until the current run finishes")
            return
        ok, how = self.ctrl.start_live_link()
        self.lbl_link.setText(("link: started — " if ok else "link: ") + how)
        self._log(("vantage live link: " if ok else "⚠ vantage live link: ") + how)

    def _final_targets(self, selected_only: bool):
        cams = ([self._current_camera()] if selected_only
                else self.ctrl.session.cameras_with_states())
        return [c for c in cams if c]

    def _on_vantage_progress(self, cam: str, status: str):
        self._log(f"vantage {cam}: {status}")

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
        self._cancel = False
        self.btn_cancel.setEnabled(True)     # finals can wedge for an hour per CLI job —
        try:                                 # Cancel must reach the io wait-poll
            if self.cfg.final_render_backend == "vantage_cli":
                # Developer-Edition CLI only — exports main-thread, renders on a worker
                jobs = self.ctrl.prepare_vantage_jobs(
                    cams, out_dir, on_progress=lambda c, s: self._log(f"vantage {c}: {s}"))
                relay = _ProgressRelay()
                # bound slot, NOT a lambda: Qt can only marshal the signal onto the
                # main thread when the receiver is a QObject with thread affinity —
                # a lambda resolves to DirectConnection and _log would run on the
                # vantage watcher thread (exactly what _ProgressRelay exists to stop)
                relay.progress.connect(self._on_vantage_progress)
                results = self._run_blocking_io(
                    lambda: self.ctrl.run_vantage_jobs(
                        jobs, on_progress=lambda c, s: relay.progress.emit(c, s),
                        should_cancel=lambda: self._cancel))
            else:
                results = self.ctrl.render_finals_vray(
                    cams, out_dir, on_progress=lambda c, s: self._log(f"final {c}: {s}"),
                    should_cancel=lambda: self._cancel)
            for cam, status in results.items():
                self._log(f"{'✓' if status == 'ok' else '✗'} {cam}: {status}")
        except Exception as e:  # noqa: BLE001
            self._log(f"✗ final renders: {e}")
        finally:
            self._busy = False
            self.btn_cancel.setEnabled(False)

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
            QtWidgets.QApplication.clipboard().setText(
                "\n".join(job["scene_file"] for job in jobs))
            self._log("ordered vrscene list copied to the clipboard; queue manifest and "
                      "Batch Render instructions are beside the exports")
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
        if self._busy:
            self._log("busy — settings ignored until the current run finishes")
            return
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self.ctrl.cfg = self.cfg
            try:
                self.cfg.save()
                self._log("settings saved")
            except OSError as e:             # read-only profile / locked config.json
                self._log(f"✗ settings not persisted: {e} — changes live this session only")


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
                pix = _bounded_pixmap(render, QtCore.QSize(240, 135))
                if not pix.isNull():
                    btn.setIcon(QtGui.QIcon(pix))
                    btn.setIconSize(QtCore.QSize(240, 135))
            btn.setStyleSheet(
                f"QToolButton{{background:{WELLIMG};border:{WELLINE};border-radius:{RAD};"
                f"padding:10px;color:{TEXT};font-family:{MONO};letter-spacing:1px;}}"
                f"QToolButton:checked{{border:2px solid {SIGNAL};"
                f"background:{SIGNAL_DIM};}}")
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
        head.setStyleSheet(f"color:{SIGNAL};font-family:{MONO};font-weight:700;"
                           f"letter-spacing:1px;")
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
            eff_lbl.setStyleSheet(f"color:{ERR};" if worse else f"color:{SIGNAL};")
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
        form.addRow("semantic API key", self.ed_key)
        self.cmb_provider = QtWidgets.QComboBox()
        self.cmb_provider.addItems(["omega", "anthropic", "openai",
                                    "openai_compatible", "offline"])
        self.cmb_provider.setCurrentText(getattr(cfg, "semantic_provider", "omega"))
        form.addRow("semantic provider", self.cmb_provider)
        self.ed_base_url = QtWidgets.QLineEdit(getattr(cfg, "semantic_base_url", ""))
        self.ed_base_url.setPlaceholderText("optional/custom: https://host/v1/chat/completions")
        form.addRow("provider URL", self.ed_base_url)
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
        self.cb_norender = QtWidgets.QCheckBox(
            "apply settings only — never render (loop, sweep, board probes, finals off)")
        self.cb_norender.setChecked(bool(getattr(cfg, "no_renders", False)))
        form.addRow("no-render mode", self.cb_norender)
        self.cb_swexpose = QtWidgets.QCheckBox(
            "apply EV/WB to frames in software (auto-detected; needed on V-Ray GPU)")
        self.cb_swexpose.setChecked(bool(getattr(cfg, "software_exposure", False)))
        form.addRow("software exposure", self.cb_swexpose)
        self.cmb_backend = QtWidgets.QComboBox()
        self.cmb_backend.addItems(["vray", "vantage_cli"])
        self.cmb_backend.setCurrentText(
            getattr(cfg, "final_render_backend", "vray") or "vray")
        form.addRow("finals backend", self.cmb_backend)
        self.cmb_preference = QtWidgets.QComboBox()
        self.cmb_preference.addItems(["balanced", "direction", "color_mood", "tonal"])
        self.cmb_preference.setCurrentText(getattr(cfg, "artist_preference", "balanced"))
        form.addRow("score preference", self.cmb_preference)
        self.ed_vantage_exe = QtWidgets.QLineEdit(getattr(cfg, "vantage_exe", ""))
        form.addRow("vantage.exe", self.ed_vantage_exe)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setObjectName("dim")
        form.addRow(self.lbl_status)
        btns = QtWidgets.QHBoxLayout()
        self.btn_test = QtWidgets.QPushButton("Test semantic provider")
        self.btn_test.clicked.connect(self._test)
        btns.addWidget(self.btn_test)
        b_ok = QtWidgets.QPushButton("Save")
        b_ok.setObjectName("primary")
        b_ok.clicked.connect(self._save)
        btns.addWidget(b_ok)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_cancel)
        form.addRow(btns)

    def _test(self):
        """Ping the gateway OFF Max's main thread (SPEC §2 threading is law): the dock's
        injectable io worker runs the blocking call, the dialog stays responsive, and
        the result lands in the status label when the worker finishes."""
        key = self.ed_key.text().strip()
        model = self.ed_model.text().strip()
        provider = self.cmb_provider.currentText()
        base_url = self.ed_base_url.text().strip()
        self.lbl_status.setText("pinging…")
        self.btn_test.setEnabled(False)
        try:
            io_runner = getattr(self.parent(), "_run_blocking_io", None)
            run = io_runner or (lambda fn: fn())   # standalone use: no pump available
            if provider == "omega":
                # honour the Settings base-URL for omega too — Test gateway must probe
                # the endpoint the match will actually use, not the shipped default
                self.lbl_status.setText(run(lambda: ping(key, model, base_url=base_url)))
            else:
                self.lbl_status.setText(run(
                    lambda: providers.ping(provider, key, model, base_url)))
            self.lbl_status.setStyleSheet(f"color:{SIGNAL};")
        except OmegaError as e:
            self.lbl_status.setText(str(e))
            self.lbl_status.setStyleSheet(f"color:{ERR};")
        except RuntimeError as e:                  # the io relay re-raises worker failures
            self.lbl_status.setText(str(e))        # as RuntimeError — same typed message
            self.lbl_status.setStyleSheet(f"color:{ERR};")
        finally:
            self.btn_test.setEnabled(True)

    def _save(self):
        self.cfg.api_key = self.ed_key.text().strip()
        self.cfg.semantic_provider = self.cmb_provider.currentText() or "omega"
        self.cfg.semantic_base_url = self.ed_base_url.text().strip()
        self.cfg.model = self.ed_model.text().strip() or "claude-opus-4-8"
        self.cfg.vantage_console = self.ed_vantage.text().strip()
        self.cfg.system_python = self.ed_syspy.text().strip()
        self.cfg.loop_width = int(self.sp_w.value())
        self.cfg.loop_height = int(self.sp_h.value())
        self.cfg.max_iterations = int(self.sp_iters.value())
        self.cfg.target_score = float(self.sp_target.value())
        self.cfg.no_renders = bool(self.cb_norender.isChecked())
        self.cfg.software_exposure = bool(self.cb_swexpose.isChecked())
        self.cfg.final_render_backend = self.cmb_backend.currentText() or "vray"
        self.cfg.artist_preference = self.cmb_preference.currentText() or "balanced"
        self.cfg.vantage_exe = self.ed_vantage_exe.text().strip()
        self.accept()


_dock_instance: Optional[MaxGafferDock] = None
_dock_wrapper: Optional[QtWidgets.QWidget] = None  # the QDockWidget host (or the window)


def show_dock():
    """Create (or raise) the dock inside 3ds Max's main window."""
    global _dock_instance, _dock_wrapper
    parent = None
    try:
        import qtmax  # Max 2021+

        parent = qtmax.GetQMaxMainWindow()
    except Exception:
        parent = None
    if _dock_wrapper is not None:
        try:
            # closing the dock's X only HIDES the QDockWidget — re-show the WRAPPER
            # (showing the inner widget alone can never un-hide its host)
            _dock_wrapper.show()
            _dock_wrapper.raise_()
            if _dock_instance is not None:
                # The panel may have been hidden while another .max scene was opened or
                # cameras were edited.  Never re-show a stale shot list/reference card.
                _dock_instance.refresh_cameras()
                return _dock_instance
        except RuntimeError:                 # C++ object deleted (Max shutdown, teardown)
            _dock_wrapper = None             # — fall through and rebuild cleanly
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
        _dock_wrapper = dock
        _dock_instance = widget
    else:  # dev fallback: plain window
        _dock_instance = MaxGafferDock()
        _dock_instance.resize(1040, 1100)
        _dock_instance.show()
        _dock_wrapper = _dock_instance
    return _dock_instance
