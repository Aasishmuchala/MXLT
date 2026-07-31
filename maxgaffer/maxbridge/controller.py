"""Controller — the one object the UI talks to. Wires core (brains) to bridge (hands).

Owns: the per-scene Session, the classified rig + light baselines, the stats-provider chain,
the three LLM calls, and the director hooks. Everything scene-touching stays main-thread
(the UI guarantees callers); LLM/stats calls are pure I/O the UI may run on workers.

Stats provider chain (first that yields wins):
  1. core.metrics in-process — Pillow if installed; for our own PNG renders the stdlib
     reader ALWAYS works, so loop stats never fail;
  2. sidecar: config.system_python -m maxgaffer.sidecar.metrics_cli (Pillow there);
  3. references only: Max transcode → PNG → stdlib reader.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..core import (animation, ask as askmod, consensus, critic, domeseed, expose,
                    fairness, feedback, plate, png_min, transfer,
                    metrics, omega, planner, profiles, prompts, providers, rules,
                    scenedigest, solver, sunsolve, refread, scenarios as scen)
from ..core.director import (Hooks, MatchConfig, MatchResult, TRANSFER_WEIGHT,
                             blend_transfer, run_match, run_sun_sweep)
from ..core.errors import MatchCancelled, PreflightBlocked
from ..core.genome import LightingState
from ..core.parse import ParseError, validate_analysis
from ..core.session import (Session, preset_dumps, preset_loads, reference_signature,
                            sidecar_path)
from . import apply as ap
from . import config as cfgmod
from . import digest as dg
from . import draft as df
from . import execute as ex
from . import render as rd
from . import preflight as pf
from . import scene as sc
from . import vantage as vt

# formats Max reads natively but Pillow/stdlib can't — always ingest via Max transcode
MAX_FIRST_EXTS = (".exr", ".hdr", ".tif", ".tiff")

# director's notes are persisted AND pinned into every DELTAS prompt of the ensemble and
# the whole deep match — cap their SIZE (the [-6:] list cap only bounds their count)
MAX_NOTE_CHARS = 500

#: Seconds of cancel latency above which the run STOPS AND ASKS rather than committing.
#:
#: A module constant, deliberately NOT a config field. I3 says cancel responsiveness is an
#: invariant with a bounded worst case; ``draft_sampler`` and ``probe_max_seconds`` being
#: config fields is exactly how 2026-07-30 ended up with no bound at all — they were on
#: disk, the Options menu overwrote them, and nothing said so.
#:
#: A V-Ray frame in flight CANNOT be aborted from Python: rt.render() is a synchronous C++
#: call and Python does not execute again until it returns, so polling cannot run during
#: the window it needs to act in. Cancel latency therefore cannot be reduced below one
#: probe, and the honest invariant is not "cancel is instant" but "after ✕ the artist
#: waits at most one probe, and that number was measured and stated before the run
#: committed". 15 s is roughly where waiting stops feeling like waiting.
CANCEL_LATENCY_BUDGET_S = 15.0

#: The reality check's thresholds (I6). Weakest-link: ANY of the three fires.
#:
#: Warm/cool inversion. critic's colour component is 1 - d_lab/30, so 0.35 is a LAB
#: distance of ~19.5 — about a full golden-hour-to-dusk swing, far outside white-balance
#: residual. On 2026-07-30 the delivered frame was a cool sunless dusk courtyard against a
#: warm golden-hour reference and no stage of the pipeline ever said so.
UNLIKE_REFERENCE_COLOR = 0.35
#: The reference has directional light and the render essentially does not. The existing
#: 0.75 line says "not landing the same way"; this says "not there".
UNLIKE_REFERENCE_HIGHLIGHT = 0.35
#: Below this the reference has no sun patch to be missing (sunsolve skips the solve
#: entirely at 0.0), so the highlight term must not fire.
REF_HAS_SUN_HOT_FRAC = 0.02
#: The absolute floor, set at the BASIN gate's number rather than higher. This repo's own
#: recorded scores include legitimate basins at 77.6 and 80.35 and structurally-wrong-but-
#: plausible finished matches at 63.2 and 77.21; 45 is below every legitimate number ever
#: measured on this box and above every black-frame number (2.7-12.0), so it can only fire
#: on garbage. The colour and highlight terms carry the discrimination.
UNLIKE_REFERENCE_SCORE = 45.0
#: The multi-start basin floor (gate 3). Same number, same argument: the black frames of
#: 2026-07-30 scored 12.0, 10.9, 8.7 and 2.7 and the picker announced a "best basin" from
#: them within ~7 renders. Nothing this tool can adjust gets from 12 to a match.
BASIN_FLOOR_SCORE = 45.0
#: sun_bearing_agreement below which the direction gate asks the artist. 0.34 is a
#: circular spread above 40° — the samples cannot agree on a QUADRANT, which is the
#: coarsest thing a human answers in five seconds. consensus.py already calls 60° "no
#: confidence"; 40° is where the reading stops constraining anything the grid could not
#: find faster. It sits below refread's established 0.75 for the opposite reason: this
#: gate spends an hour of an artist's day when it fires, so it should be shy about firing.
BEARING_ASK_AGREEMENT = 0.34
#: A quadrant is a quadrant, not an angle. An artist answering "over my left shoulder" has
#: given ±45°, and locking the axis outright would forfeit refinement the render can still
#: do — so the answer becomes a START VALUE plus this much slack, never a lock.
QUADRANT_SLACK_DEG = 45.0

#: How much confidence a cost basis string carries, worst last. ``cost_estimate`` quotes
#: the WEAKEST basis across the priced stages, because a headline built partly out of an
#: extrapolation is an extrapolation. (2026-07-31)
_BASIS_ORDER = ("measured at this size", "affine fit", "scaled from", "EXTRAPOLATED")


def _BASIS_RANK(basis: str) -> int:
    for i, prefix in enumerate(_BASIS_ORDER):
        if str(basis).startswith(prefix):
            return i
    return -1                 # "" — nothing priced yet


def _needs_max_ingest(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MAX_FIRST_EXTS


class Controller:
    def __init__(self, cfg: Optional[cfgmod.Config] = None):
        self.cfg = cfg or cfgmod.load()
        self._session: Optional[Session] = None
        self._session_scene = None
        self._rig = None
        self._baselines: Dict[str, float] = {}   # light NAME → authored multiplier
        self._ref_cache: Dict[str, Dict] = {}     # ref path+mtime → stats
        self._run_dir: Optional[str] = None
        self._last_analyze_agreement: Optional[float] = None
        self._selected_camera_name = ""
        self._selected_camera_id = ""
        # pure-I/O runner — the UI swaps in a worker-thread pump so gateway waits never
        # freeze Max; pymxs is NEVER called through this (network/subprocess only)
        self.io: Callable = lambda fn: fn()
        # ESCALATION seam, wired exactly like ``io``: the dock swaps in a QMessageBox, a
        # headless caller leaves it alone and cfg.uncertainty_policy decides. Core stays
        # dialog-free. Added 2026-07-31 — on 2026-07-30 the analyzer knew it could not
        # agree with itself and spent 90 minutes guessing instead of asking.
        self.ask: Callable[[askmod.Question], str] = lambda _q: ""
        self._session_gen = sc.scene_generation()
        # I3: one sticky latch, checked at the top of _render_exposed, which is the ONE
        # function every render in the plugin passes through. Setting it bounds EVERY
        # loop — polish's diagonal ride, tone-align, execute_plan, the basin picker, the
        # board, the finals — without touching any of them, because none of them can
        # render without coming through here. Forgetting a loop then costs one probe, not
        # a run.
        self._cancel_latch: str = ""
        #: sticky: a validated black/near-black plate stops the run everywhere, once
        self._probe_abort: str = ""
        #: the CURRENT operation's cancel predicate, for the one place inside a render
        #: that can still poll (see _begin_operation / _cancel_poll)
        self._op_should_cancel: Callable[[], bool] = lambda: False
        # I1: plate validation state. _plate_memo means stats_for() returns the dict
        # _render_exposed already computed, so validating every frame costs LESS than the
        # status quo, which computed it twice for the one guarded probe.
        self._plate_memo: Tuple[str, Optional[Dict]] = ("", None)
        self._plate_prev_sig: Tuple = ()
        self._plate_prev_state: Optional[Dict] = None
        self._frozen_run = 0
        self._frozen_total = 0
        self._frozen_escalated = False
        self._skip_sun_stages = False
        # I2: measured render timings, keyed by pixel count. Vantage samples are excluded
        # at the source — a 50 ms grab pricing 181 V-Ray renders reads as "9 seconds".
        self._render_times: Dict[int, List[float]] = {}
        self._cold_seconds: Optional[float] = None
        self._cost_stages: List = []
        self._cost_reported = False
        # I6: every fallback, degradation and skipped stage, replayed in the final report.
        # They already log at the moment they happen and then scroll away under 190 THUMB
        # lines; the ledger is what makes "announced" survive to the verdict.
        self._degradations: List[str] = []
        self._decisions: List[Dict] = []
        #: stamped into every write-back path. A run the dock has DETACHED must not apply
        #: state, record a match or save the session minutes later (see dock._force_release).
        self._generation = 0
        cfgmod.add_warn_sink(self._config_warn)
        sc.register_scene_callbacks()

    # ------------------------------------------------------------------ I3 bookkeeping
    def _begin_operation(self, should_cancel: Optional[Callable[[], bool]] = None
                         ) -> None:
        """A NEW artist-initiated operation starts uncancelled and un-aborted.

        The two latches are one-way WITHIN an operation and are checked at the top of
        ``_render_exposed`` — the one function every render passes through — so whatever
        sets one bounds every loop. That is the point. But until 2026-07-31 only
        ``run_match`` cleared them, and they were promoted to gate _render_exposed in the
        same commit: after a ✕ or a black-frame abort, REFINE, the scenario board,
        ``execute_plan``'s effect probes and the delivered finals all raised the stale
        message before rendering a single pixel, for the rest of the Max session, and only
        a fresh MATCH resurrected them. With ``plan_first`` it was worse — the dock runs
        ``execute_plan`` BEFORE ``run_match``, so the next match died before reaching the
        reset. Every entry point an artist can press now says so here.

        It also records the operation's cancel predicate. A V-Ray frame is uninterruptible
        and that is physics; the Vantage settle poll is a ``time.sleep`` loop on the main
        thread and that is a choice — so it gets the predicate and the artist gets ✕ back
        inside ``vgrab.SETTLE_STEP_S`` instead of ``SETTLE_LIMIT_S``.
        """
        self._cancel_latch = ""
        self._probe_abort = ""
        self._op_should_cancel: Callable[[], bool] = should_cancel or (lambda: False)

    def _cancel_poll(self) -> bool:
        """The predicate handed to anything that SLEEPS inside a render (I3)."""
        if self._cancel_latch:
            return True
        try:
            return bool(self._op_should_cancel())
        except Exception:  # noqa: BLE001 — a broken predicate must not wedge a poll
            return False

    # ------------------------------------------------------------------ plumbing sinks
    def _config_warn(self, msg: str) -> None:
        """Config warnings land in the artist's transcript, not only Max's Listener.

        29bbae6 added "config could not be read — using DEFAULTS" precisely because that
        sentence was the difference between three lost hours and a two-second diagnosis on
        2026-07-30, and then print()ed it into a window nobody reads. Buffered until a log
        is available so a warning raised at construction is not lost."""
        self._pending_warnings = getattr(self, "_pending_warnings", [])
        self._pending_warnings.append(str(msg))

    def _drain_warnings(self, log: Callable[[str], None]) -> None:
        for msg in getattr(self, "_pending_warnings", []):
            log(msg)
        self._pending_warnings = []

    def _degrade(self, log: Callable[[str], None], msg: str) -> None:
        """Degrade LOUDLY, and remember it. One bad component must never kill a match —
        but degrading SILENTLY is exactly what cost 2026-07-30, so every fallback lands
        both in the transcript now and in the final report later."""
        if msg not in self._degradations:
            self._degradations.append(msg)
        log("⚠ " + msg)

    # ------------------------------------------------------------------ scene / session
    def _bust_stale_caches(self) -> None:
        """File new/reset/open invalidates every per-scene cache — the path-keyed check
        alone can't see a change between two UNSAVED scenes (both path '')."""
        gen = sc.scene_generation()
        if gen != self._session_gen:
            self._session_gen = gen
            self._session = None
            self._session_scene = None
            self._rig = None
            self._baselines = {}
            # A cached MEASUREMENT must not outlive the thing it measured. The
            # exposure-host verdict is a property of THIS renderer on THIS scene: a
            # V-Ray GPU→CPU switch or a different .max left last scene's verdict
            # governing every later run, silently double-applying EV on a host that bakes
            # it in. Same for the plate-linear flag and the Pillow warning, which is only
            # reset inside run_match and so was suppressed entirely for refine's branch
            # probes. (2026-07-31)
            self._exposure_host_checked = False
            self._plate_linear = False
            self._sw_warned = False
            self._render_times = {}
            self._cold_seconds = None
            if getattr(self, "_sw_auto", False):
                # ONLY the flag WE set off a measurement is cleared. cfg.software_exposure
                # is also an artist Settings choice, and wiping that on every scene open
                # would be this module deciding it knows better than the person who
                # ticked it.
                self.cfg.software_exposure = False
                self._sw_auto = False

    @property
    def session(self) -> Session:
        self._bust_stale_caches()
        scene = sc.scene_path()
        if self._session is None or scene != self._session_scene:
            self._session = Session.load(sidecar_path(scene))
            self._session_scene = scene
        return self._session

    def save_session(self) -> bool:
        self.session.path = sidecar_path(sc.scene_path())   # scene may have been saved-as
        return self.session.save()

    def _save_or_warn(self, log: Callable[[str], None]) -> None:
        """save_session + the SPEC-mandated LOUD warning when persistence is off or
        failed (unsaved / read-only scene): everything then lives in memory ONLY."""
        if not self.save_session():
            log("⚠ session NOT saved (unsaved or read-only scene) — states, snapshots "
                "and notes live in memory only and vanish when the scene closes")

    def rig(self, refresh: bool = False):
        self._bust_stale_caches()
        if self._rig is None or refresh:
            self._rig = sc.classify_rig()
            # adopt-only-new into the session: re-scans NEVER overwrite a known baseline,
            # so a group MaxGaffer previously dimmed to 0 keeps its authored value
            fresh = ap.capture_baselines(self._rig)
            if self.session.adopt_baselines(fresh):
                self.save_session()
            self._baselines = dict(self.session.baselines)
        return self._rig

    def cameras(self) -> List[Dict]:
        cams = sc.list_cameras()
        active = sc.active_camera_name()
        active_id = sc.active_camera_identity()
        for c in cams:
            e = self.session.find(c["name"], c.get("id", ""))
            if e is not None:
                if c.get("id") and not e.camera_id:
                    e.camera_id = str(c["id"])
                e.camera_name = c["name"]
            c["reference"] = e.reference if e else ""
            c["score"] = e.score if e else None
            c["has_state"] = bool(e and e.state)
            c["active"] = (bool(active_id) and c.get("id") == active_id) or (
                not active_id and c["name"] == active)
        return cams

    def _camera_id(self, camera_name: str) -> str:
        if camera_name == self._selected_camera_name and self._selected_camera_id:
            return self._selected_camera_id
        matches = [c.get("id", "") for c in sc.list_cameras() if c.get("name") == camera_name]
        return str(matches[0]) if len(matches) == 1 else ""

    def camera_entry(self, camera_name: str, create: bool = False):
        camera_id = self._camera_id(camera_name)
        return (self.session.entry(camera_name, camera_id) if create
                else self.session.find(camera_name, camera_id))

    def camera_node(self, camera_name: str):
        camera_id = self._camera_id(camera_name)
        return (sc.get_camera(camera_name, camera_id) if camera_id
                else sc.get_camera(camera_name))

    def _set_active_camera(self, camera_name: str) -> bool:
        camera_id = self._camera_id(camera_name)
        return (sc.set_active_camera(camera_name, camera_id) if camera_id
                else sc.set_active_camera(camera_name))

    def bind_reference(self, camera_name: str, path: str) -> None:
        self.session.set_reference(camera_name, path, self._camera_id(camera_name))

    def add_reference(self, camera_name: str, path: str, role: str = "") -> None:
        """Append a supporting reference view (Route A: the primary stays authoritative for
        the solve; extra views only DENOISE the semantic ANALYZE). No-op on an empty path or
        a duplicate signature; persists the session."""
        self.session.add_reference(camera_name, path, role, self._camera_id(camera_name))
        self.save_session()

    def set_references(self, camera_name: str, items: List) -> None:
        """Replace the whole reference list — ``items`` are path strings and/or
        ``{"path", "role"}`` dicts, the FIRST becoming the primary. Empty → clears them."""
        self.session.set_references(camera_name, items, self._camera_id(camera_name))
        self.save_session()

    def remove_reference(self, camera_name: str, ref) -> bool:
        """Drop a reference by int index OR path/signature string; True when one was removed.
        Removing the primary promotes the next view (the legacy mirror re-syncs)."""
        removed = self.session.remove_reference(camera_name, ref,
                                                self._camera_id(camera_name))
        self.save_session()
        return removed

    def references(self, camera_name: str) -> List[Dict]:
        """The camera's references as plain dicts (primary first). A legacy single-reference
        camera reports its lone synthesized primary — ``{path, relative, signature, role,
        score, has_semantics}`` each. Read-only."""
        return [{"path": r.path, "relative": r.relative, "signature": r.signature,
                 "role": r.role, "score": r.score, "has_semantics": bool(r.semantics)}
                for r in self.session.references(camera_name, self._camera_id(camera_name))]

    def _record_match(self, camera_name: str, state: LightingState,
                      score: Optional[float]) -> None:
        self.session.record_match(camera_name, state, score, self._camera_id(camera_name))
        # Also leave a NATIVE Max Scene State. The rig is global — one sun, one dome, one
        # exposure control — so only one camera's look can be live at a time, and the
        # sidecar that remembers the rest is the PLUGIN's memory. A Scene State is Max's:
        # it saves inside the .max file, restores from Tools > Manage Scene States without
        # MaxGaffer installed, and travels to whoever opens the scene. Captures Light
        # Properties + Light Transforms + Environment only, so restoring a lighting state
        # can never revert the artist's geometry or shaders. Best-effort by design: an
        # older Max without the interface must not fail a finished match.
        if getattr(self.cfg, "capture_scene_states", True):
            try:
                sc.capture_scene_state(camera_name)
            except Exception as err:  # noqa: BLE001 — bookkeeping must never sink a match
                # …but a bare `pass` here meant the artist believed their match was saved
                # as a native Max Scene State INSIDE the .max file, travelling to whoever
                # opens the scene, and it was not. (2026-07-31)
                self._config_warn(f"the native Max Scene State could not be captured "
                                  f"({err}) — this camera's look lives ONLY in "
                                  f"MaxGaffer's sidecar, not in the .max file")

    def camera_fingerprint(self) -> Tuple:
        return tuple((c.get("id", ""), c.get("name", ""), c.get("class", ""))
                     for c in sc.list_cameras())

    def relink_missing_references(self, roots: List[str]) -> Dict[str, str]:
        changed = self.session.relink_references(roots)
        if changed:
            self.save_session()
        return changed

    def read_state(self, camera_name: str = "") -> LightingState:
        cam = self.camera_node(camera_name) if camera_name else None
        return ap.read_state(self.rig(), self._baselines, cam)

    def apply_state(self, state: LightingState, camera_name: str = "") -> List[str]:
        cam = self.camera_node(camera_name) if camera_name else None
        return ap.apply_state(self.rig(), self._baselines, state, cam)

    def select_camera(self, camera_name: str, apply_saved: bool = True,
                      camera_id: str = "") -> List[str]:
        camera_id = str(camera_id or self._camera_id(camera_name))
        activated = (sc.set_active_camera(camera_name, camera_id) if camera_id
                     else sc.set_active_camera(camera_name))
        if activated is False:
            raise RuntimeError(f"camera '{camera_name}' is no longer available")
        self._selected_camera_name = camera_name
        self._selected_camera_id = camera_id
        warnings: List[str] = []
        if apply_saved and self.session.settings.get("apply_on_select", True):
            e = self.camera_entry(camera_name)
            if e and e.state is not None:
                warnings = self.apply_state(e.state, camera_name)
            self._rebind_seed(camera_name)     # the dome is scene-global; seeds are not
        return warnings

    def _rebind_seed(self, camera_name: str) -> None:
        """Per-camera dome seeds live on ONE scene-global dome — whenever a camera's
        light is applied for viewing or rendering, its own seed must come back, or shot A
        silently renders under shot B's sky. Never touches rotation: the applied state
        owns dome.rotation_deg (only the initial seed bind zeroes it)."""
        e = self.camera_entry(camera_name)
        if not (e and e.seed_hdri) or not os.path.exists(e.seed_hdri):
            return
        dome = self.rig().get("dome")
        if dome is None:
            return
        if sc.get_dome_texture(dome) != e.seed_hdri:
            with self._dome_undo():
                sc.set_dome_texture(dome, e.seed_hdri)

    # ------------------------------------------------------------------ stats providers
    def stats_for(self, path: str) -> Optional[Dict]:
        # MEMO, not a second engine: _render_exposed validates every plate and has to
        # compute these anyway, and every caller then asks for them again one line later.
        # Without this, validating every frame would double the stats cost of a match;
        # with it, validating every frame is CHEAPER than the old one-shot guard, which
        # computed the first probe's stats twice. Keyed by path, so it can never answer
        # for a different image.
        if path and path == self._plate_memo[0]:
            return self._plate_memo[1]
        s = metrics.compute_stats(path)
        if s is not None:
            return s
        return self._sidecar_stats(path)

    def _sidecar_stats(self, path: str) -> Optional[Dict]:
        py = self.cfg.system_python
        if not py or not os.path.exists(py):
            return None
        try:
            repo = self.cfg.repo_path or os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # through the io runner like the gateway calls — a wedged sidecar must
            # never freeze Max's main thread (and Cancel) for the full timeout
            proc = self.io(lambda: subprocess.run(
                [py, "-m", "maxgaffer.sidecar.metrics_cli", path],
                capture_output=True, text=True, timeout=60, cwd=repo))
            data = json.loads(proc.stdout or "null")
            if isinstance(data, list) and data and isinstance(data[0].get("stats"), dict):
                return data[0]["stats"]
        except Exception:
            pass
        return None

    def ref_stats(self, ref_path: str) -> Optional[Dict]:
        key = reference_signature(ref_path)
        if key.endswith("|missing"):
            return None
        if key in self._ref_cache:
            return self._ref_cache[key]
        s = None
        if _needs_max_ingest(ref_path):   # EXR/HDR/TIFF: Max's bitmap I/O is the reader
            png = self._transcode_ref(ref_path)
            if png:
                s = metrics.compute_stats(png)
        if s is None:
            s = self.stats_for(ref_path)
        if s is None:  # last resort: Max transcodes anything it can read to a small PNG
            png = self._transcode_ref(ref_path)
            if png:
                s = metrics.compute_stats(png)
        if s is not None:
            self._ref_cache[key] = s
        return s

    def _transcode_ref(self, ref_path: str) -> Optional[str]:
        token = _reference_token(ref_path)
        png = os.path.join(self._ensure_run_dir("refs"),
                           "ref_" + _safe(os.path.basename(ref_path)) +
                           f"_{token}.png")
        out = rd.transcode_to_png(ref_path, png)
        if out:
            prune_old_files(os.path.dirname(png), keep=int(self.cfg.keep_runs))
        return out

    # ------------------------------------------------------------------ LLM plumbing
    def _semantic_call(self, system: str, messages: list, max_tokens: int) -> str:
        return providers.call(
            self.cfg.semantic_provider, self.cfg.api_key, system, messages,
            model=self.cfg.model, max_tokens=max_tokens,
            base_url=self.cfg.semantic_base_url,
        )

    def _critic_weights(self) -> Dict[str, float]:
        return critic.weights_for(self.cfg.artist_preference, self.cfg.critic_weights)

    def _image_block(self, path: str) -> Optional[dict]:
        """Payload-slim image block: Pillow in-process → sidecar --b64 → raw file (small
        renders) → Max transcode to PNG. EXR/HDR/TIFF skip straight to Max transcode."""
        if _needs_max_ingest(path):
            token = _reference_token(path)
            png = os.path.join(self._ensure_run_dir("refs"),
                               "llm_" + _safe(os.path.basename(path)) +
                               f"_{token}.png")
            if rd.transcode_to_png(path, png, max_dim=768):
                prune_old_files(os.path.dirname(png), keep=int(self.cfg.keep_runs))
                return omega.image_block_from_file(png)
            return None
        try:
            from PIL import Image, ImageOps  # type: ignore
            import io

            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)   # the model must see the photo upright
                im = im.convert("RGB")
                im.thumbnail((768, 768))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
            return omega.image_block(
                base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg")
        except Exception:
            pass
        py = self.cfg.system_python
        if py and os.path.exists(py):
            try:
                repo = self.cfg.repo_path or os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                # io runner, same as _sidecar_stats — sidecar waits never block the UI
                proc = self.io(lambda: subprocess.run(
                    [py, "-m", "maxgaffer.sidecar.metrics_cli", path, "--b64"],
                    capture_output=True, text=True, timeout=60, cwd=repo))
                data = json.loads(proc.stdout or "null")
                if isinstance(data, list) and data and data[0].get("b64"):
                    return omega.image_block(data[0]["b64"],
                                             data[0].get("media_type", "image/jpeg"))
            except Exception:
                pass
        try:
            if os.path.getsize(path) <= 3_500_000:
                block = omega.image_block_from_file(path)
                if block is not None:
                    return block
        except OSError:
            return None
        png = os.path.join(self._ensure_run_dir("refs"),
                           "llm_" + _safe(os.path.basename(path)) + "_" +
                           _reference_token(path) + ".png")
        if rd.transcode_to_png(path, png, max_dim=768):
            prune_old_files(os.path.dirname(png), keep=int(self.cfg.keep_runs))
            return omega.image_block_from_file(png)
        return None

    def analyze_reference(self, camera_name: str) -> Dict:
        """ANALYZE call (cached in the session until the reference changes).

        A single reference (the default) takes the byte-for-byte original path. When the
        camera carries MORE than one reference, each view is analyzed under its own signature
        gate and the reads are FUSED with the same consensus consolidator — a denoised primary
        ANALYZE (Route A: extra views sharpen the semantic consensus; they never reach the
        solve/critic/LLM and never constrain unseen geometry). The fused read is mirrored onto
        the primary (``e.semantics``), so run_match / scenarios / seed_dome are untouched."""
        e = self.camera_entry(camera_name, create=True)
        if not e.reference:
            raise RuntimeError("no reference image bound to this camera")
        if len(e.references) > 1:
            return self._analyze_multi_reference(e)
        current_signature = reference_signature(e.reference)
        if e.reference_signature != current_signature:
            if not e.reference_signature and e.semantics:
                # Backward-compatible adoption for v1 sidecars written before signatures
                # existed.  An explicit Load/swap still invalidates via set_reference().
                e.reference_signature = current_signature
            else:
                # Same path, new pixels: never reuse the previous image's analysis/score.
                e.reference_signature = current_signature
                e.semantics = {}
                e.score = None
        if e.semantics:
            # cached read — clear any stale contest flag so a LATER camera's run never
            # logs this camera's (or an older run's) disagreement
            self._last_analyze_agreement = None
            return e.semantics
        # a fresh analysis supersedes any previous read — a stale contested-consensus
        # flag must never attach its warning to a run it doesn't describe
        self._last_analyze_agreement = None
        ref_path = e.reference   # pinned: the gateway waits below are click-windows
        ref_signature = e.reference_signature
        block = self._image_block(ref_path)
        if block is None:
            raise RuntimeError(f"could not read reference image: {ref_path}")
        messages = [{"role": "user",
                     "content": [block, omega.text_block(prompts.analyze_user_text())]}]
        # self-consistency: N independent reads, consolidated (majority/median/circular
        # mean) — live evidence showed single samples reading the same ref as golden hour
        # OR midday, and a wrong read poisons the whole run
        samples = []
        n = max(1, int(self.cfg.analyze_samples))
        last_reply = ""
        for _ in range(n):
            last_reply = self.io(lambda: self._semantic_call(
                prompts.ANALYZE_SYSTEM, messages, 2048))
            try:
                samples.append(validate_analysis(last_reply))
            except ParseError:
                continue
        if not samples:   # every sample was junk — one strict retry, then give up loudly
            retry = messages + [
                {"role": "assistant", "content": last_reply[:1500]},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the "
                                            "JSON object, nothing else."}]
            samples.append(validate_analysis(self.io(lambda: self._semantic_call(
                prompts.ANALYZE_SYSTEM, retry, 2048))))
        if len(samples) < n:
            # 2 of 3 samples unusable means the "consensus" is one read wearing a
            # consensus's clothes, and nothing downstream could tell. (2026-07-31)
            self._config_warn(f"analyze: {n - len(samples)} of {n} samples were unusable "
                              f"— the consensus rests on {len(samples)}")
        semantics = consensus.consolidate_analyses(samples)
        agreement = semantics.pop("consensus_agreement", 1.0)   # kept out of the cache
        if (e.reference != ref_path or e.reference_signature != ref_signature
                or reference_signature(ref_path) != ref_signature):
            # the reference was swapped during a gateway wait — caching the OLD image's
            # read against the NEW path would poison every later run of this camera
            raise RuntimeError("reference image changed while analyzing — run again to "
                               "analyze the new reference")
        # a fresh 100% agreement must CLEAR a contested flag left by an earlier
        # analysis (same or another camera), or run_match logs a ghost warning —
        # unanimous reads store None, only a real contest carries a value
        self._last_analyze_agreement = agreement if agreement < 1.0 else None
        e.semantics = semantics
        self.save_session()
        # MEASURE what the model was guessing at. ANALYZE is the least reliable component
        # in the plugin — four reads of one reference gave sun bearings 130 degrees apart —
        # and a good deal of what it reports is sitting in the pixels: whether there is any
        # directional light at all, what colour the illuminant is, how sharply shadows end.
        # The cached e.semantics stays exactly as the model said it; the measurements ride
        # on top of the copy the match actually uses, and only where they are confident.
        try:
            return refread.fuse(
                semantics,
                refread.measure(self.ref_stats(ref_path), reading=semantics,
                                path=ref_path))
        except Exception as err:  # noqa: BLE001 — a measurement must never sink an analysis
            # …and it must not disappear either. This bare except deleted the entire
            # pixel-measured refinement of ANALYZE — the part that checks whether there is
            # any directional light at all, what colour the illuminant is, how sharply
            # shadows end — and left the raw model reading standing in its place with no
            # sign that the measurement had been lost. (2026-07-31)
            self._config_warn(f"the reference MEASUREMENT could not be fused ({err}) — "
                              "this run uses the model's raw reading, unrefined by pixels")
            return semantics

    def _analyze_samples(self, path: str) -> Dict:
        """N-sample self-consistency ANALYZE of ONE image → a consolidated semantics dict
        (the single-image ``consensus_agreement`` is stripped — it is not a cross-reference
        signal). The reused engine behind the multi-reference fusion; the single-reference
        path keeps its own inline body with the reference-swap guard."""
        block = self._image_block(path)
        if block is None:
            raise RuntimeError(f"could not read reference image: {path}")
        messages = [{"role": "user",
                     "content": [block, omega.text_block(prompts.analyze_user_text())]}]
        samples = []
        n = max(1, int(self.cfg.analyze_samples))
        last_reply = ""
        for _ in range(n):
            last_reply = self.io(lambda: self._semantic_call(
                prompts.ANALYZE_SYSTEM, messages, 2048))
            try:
                samples.append(validate_analysis(last_reply))
            except ParseError:
                continue
        if not samples:   # every sample was junk — one strict retry, then give up loudly
            retry = messages + [
                {"role": "assistant", "content": last_reply[:1500]},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the "
                                            "JSON object, nothing else."}]
            samples.append(validate_analysis(self.io(lambda: self._semantic_call(
                prompts.ANALYZE_SYSTEM, retry, 2048))))
        out = consensus.consolidate_analyses(samples)
        out.pop("consensus_agreement", None)
        return out

    def _analyze_multi_reference(self, e) -> Dict:
        """Fuse every bound reference's ANALYZE into a denoised primary read (see
        analyze_reference). Cross-reference scatter is intentional, so the single-image
        contested-consensus flag is suppressed (``self._last_analyze_agreement = None``).

        The extra views are preserved as durable per-RefEntry data; the FUSED read is written
        to the primary mirror so the solve/critic/LLM — which only ever see the primary — get
        a sharper, cross-angle consensus without any radiance ever reaching the solve."""
        self._last_analyze_agreement = None
        primary_sig = reference_signature(e.reference)
        angles = e.references[1:]
        angle_stale = [(not ref.semantics)
                       or (ref.signature != reference_signature(ref.path))
                       for ref in angles]
        primary_stale = (e.reference_signature != primary_sig)
        if e.semantics and not primary_stale and not any(angle_stale):
            return e.semantics          # the fused read is cached on the primary mirror
        # A recompute re-analyzes the primary fresh — its individual read is not recoverable
        # from the fused mirror — reuses fresh angle caches, and (re)reads only stale angles.
        per_ref: List[Dict] = [self._analyze_samples(e.reference)]
        for ref, stale in zip(angles, angle_stale):
            if stale:
                try:
                    read = self._analyze_samples(ref.path)
                except (MatchCancelled, PreflightBlocked):
                    raise               # a bare `continue` let reference #2 fire another
                                        # three gateway calls after the artist pressed ✕
                except (omega.OmegaError, RuntimeError):
                    continue            # a dead / unreadable angle must not kill the fusion
                ref.signature = reference_signature(ref.path)
                ref.semantics = read
            if ref.semantics:
                per_ref.append(ref.semantics)
        if reference_signature(e.reference) != primary_sig:
            # the primary was swapped during a gateway wait — caching the OLD read under the
            # NEW path would poison every later run (mirrors the single-reference guard)
            raise RuntimeError("reference image changed while analyzing — run again to "
                               "analyze the new reference")
        fused = consensus.consolidate_analyses(per_ref)
        fused.pop("consensus_agreement", None)
        e.reference_signature = primary_sig
        e.semantics = fused             # Route A mirror: the solve/critic/LLM see this
        self.save_session()
        return fused

    def _llm_deltas_hook(self, ref_block: dict) -> Callable[[Dict], str]:
        def call_llm(ctx: Dict) -> str:
            if getattr(self, "_llm_down", False):
                # gateway already failed this run — don't burn 3 backoff retries per
                # iteration; the analytic solver + metric sweep carry the match alone
                return '{"assessment": "", "changes": [], "stop": false}'
            render_block = self._image_block(ctx["render_path"])
            content = [ref_block]
            if render_block is not None:
                content.append(render_block)
            else:  # the prompt promises two images — correct the record so the model
                content.append(omega.text_block(   # doesn't judge the reference against itself
                    "NOTE: the current render could not be attached — Image 1 is the "
                    "REFERENCE and there is NO second image. Base changes on the state "
                    "table and score history only; do not describe the render."))
            content.append(omega.text_block(prompts.deltas_user_text(
                ctx["state_table"], ctx["semantics"], ctx["score_history"],
                ctx["analytic_applied"], ctx["iteration"], ctx["max_iterations"],
                ctx.get("rig_notes", ""), ctx.get("param_history", ""),
                ctx.get("director_note", ""))))
            try:
                return self.io(lambda: self._semantic_call(
                    prompts.DELTAS_SYSTEM,
                    [{"role": "user", "content": content}], 2048))
            except (omega.OmegaError, RuntimeError) as err:
                self._llm_down = True
                return ('{"assessment": "LLM offline (%s) — analytic-only from here", '
                        '"changes": [], "stop": false}' % str(err)[:80].replace('"', "'"))
        return call_llm

    def _analyze_or_fallback(self, camera_name: str, log: Callable[[str], None]) -> Dict:
        """ANALYZE, or — gateway down — the neutral base semantics with the LLM marked
        down for this run. The analytic solver, metric-only sweep and critic still run:
        a dead gateway degrades the match, it must never abort it."""
        try:
            out = self.analyze_reference(camera_name)
            self._drain_warnings(log)   # e.g. "2 of 3 samples were unusable"
            return out
        except (MatchCancelled, PreflightBlocked):
            # FIRST, and before the RuntimeError arm below would swallow it. A cancel
            # reported as "⚠ gateway unavailable (cancelled) — ANALYTIC-ONLY run" then
            # PROCEEDED on fabricated semantics: the artist pressed ✕ and the tool
            # answered by starting a worse version of the same run. (2026-07-31)
            raise
        except (omega.OmegaError, RuntimeError) as err:
            self._llm_down = True
            e = self.camera_entry(camera_name)
            if e is not None and e.semantics:
                log(f"⚠ gateway unavailable ({err}) — using this camera's cached analysis")
                return e.semantics
            log(f"⚠ gateway unavailable ({err}) — ANALYTIC-ONLY run on neutral base "
                "semantics; steer with BOARD, locks and notes")
            return dict(scen.DEFAULT_SEMANTICS)

    # ------------------------------------------------------------------ scene-wide plan
    def make_plan(self, camera_name: str, log: Callable[[str], None]):
        """READ (full digest) → UNDERSTAND (LLM sees ref + every current setting) →
        PLAN (validated, digest-grounded ops). Returns (ops, lines, meta, raw_digest),
        or None when the model replied junk twice — a PLAN failure degrades to
        "no plan" so the match proceeds plan-less (unlike ANALYZE, which fails loud)."""
        e = self.camera_entry(camera_name, create=True)
        if not e.reference:
            raise RuntimeError("bind a reference image to this camera first")
        # snapshot the genome part BEFORE the plan touches anything — Restore pre-match
        # must return to the true starting light (the plan itself is one Ctrl+Z)
        e.pre_match = ap.read_state(self.rig(refresh=True), self._baselines,
                                    self.camera_node(camera_name))
        self._plan_snapped = camera_name
        self.save_session()
        log("reading the scene — every renderer/environment/exposure/light/camera setting…")
        raw = dg.build_digest()
        cat = scenedigest.catalog(raw)
        n_props = sum(len(v) for v in cat.values())
        log(f"digest: {len(raw.get('lights') or [])} lights · "
            f"{len(raw.get('cameras') or [])} cameras · {n_props} settable properties")
        semantics = self.analyze_reference(camera_name)
        ref_block = self._image_block(e.reference)
        if ref_block is None:
            raise RuntimeError("reference image could not be prepared for the LLM")
        text = planner.plan_user_text(scenedigest.to_text(raw), semantics, camera_name)
        messages = [{"role": "user", "content": [ref_block, omega.text_block(text)]}]
        reply = self.io(lambda: self._semantic_call(planner.PLAN_SYSTEM, messages, 4096))
        try:
            ops, rejected, meta = planner.validate_plan(reply, cat)
        except ParseError:
            retry = messages + [
                {"role": "assistant", "content": reply[:1500]},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the "
                                            "JSON object, nothing else."}]
            try:
                ops, rejected, meta = planner.validate_plan(self.io(
                    lambda: self._semantic_call(planner.PLAN_SYSTEM, retry, 4096)), cat)
            except ParseError:
                log("⚠ plan reply was invalid JSON twice — plan skipped, continuing "
                    "with the match loop")
                return None
        if meta.get("read"):
            log("scene read: " + meta["read"])
        for r in rejected:
            log("plan rejected: " + r)
        return ops, planner.describe_plan(ops), meta, raw

    # ------------------------------------------------------------- exposed renders
    def _exposure_anchor(self, entry) -> Tuple[float, float]:
        """The (ev, wb) the RAW render buffer corresponds to under software exposure —
        anchored at the camera's ``pre_match`` snapshot, the one state every
        exploration records before touching anything. Loop frames, board probes, plan
        probes, refine branches and FINALS all expose relative to this same basis, so
        their pixels stay mutually consistent AND the finals reproduce the exact
        exposure the accepted match iteration showed."""
        base_ev, base_wb = 10.0, 6500.0
        pre = getattr(entry, "pre_match", None) if entry is not None else None
        if pre is not None:
            base_ev = pre.get("exposure.ev", base_ev)
            base_wb = pre.get("exposure.wb_kelvin", base_wb)
        return base_ev, base_wb

    def _render_exposed(self, cam, out_path: str, w: int, h: int, state=None,
                        entry=None, log: Optional[Callable[[str], None]] = None,
                        probe: bool = False, validate: bool = True, tag: str = ""):
        """The ONE render path — and therefore the one place I1, I2 and I3 are enforced.

        Everything below happens on EVERY render in the plugin, because everything renders
        through here: the loop, the board, the basin picker, the sun solve, refine's
        ensemble branches, the plan probes, the fairness probe and the delivered finals.

          * I3 — a sticky cancel latch is checked BEFORE the render starts, so ✕ is
            bounded by one probe no matter which loop is running;
          * I2 — the wall clock is measured and fed to the cost model;
          * I1 — the returned plate is VALIDATED (black / near-black / wrong-size /
            frozen / stale) before any caller is allowed to score it.

        Written 2026-07-31. The 29bbae6 guard checked the FIRST probe only, and its own
        comment names the worst case it was accepting: if the link dies at probe 50, the
        remaining 83 probes and the delivered final are all ranked unvalidated. Doing it
        here closes that, covers both backends, and covers the five scored paths the old
        guard never saw at all — run_scenarios (where 2026-07-30's "12.0, 10.9, 8.7, 2.7"
        appeared), probe_score, refine's branches, assess_fairness, and the final full
        re-render that OVERWRITES result.best_score and hands it to the artist.

        ``validate=False`` exists for exactly one caller — a reference transcode, which is
        not a render and has no requested size to check against. No scored render uses it.
        """
        if self._cancel_latch:
            raise MatchCancelled(self._cancel_latch)
        if self._probe_abort:
            # STICKY, and now bounding EVERY render rather than only render_hook's — the
            # board, refine's branches and the finals all come through here too.
            raise RuntimeError(self._probe_abort)
        _t0 = time.time()
        path = self._render_raw(cam, out_path, w, h, state=state, entry=entry, log=log,
                                probe=probe)
        used = getattr(self, "_last_render_backend", "vray")
        self._observe_render(w, h, time.time() - _t0, used, log, started=_t0)
        if not validate or not path:
            return path
        return self._validate_plate(path, w, h, state, log, tag or out_path, _t0)

    def _render_raw(self, cam, out_path: str, w: int, h: int, state=None,
                    entry=None, log: Optional[Callable[[str], None]] = None,
                    probe: bool = False):
        """``render_frame`` + software exposure when enabled — the ONE render path
        every probe/loop/board/final goes through, so the EV/WB the plugin sets always
        reach the scored (and delivered) pixels, even on renderers whose exposure host
        is display-stage only (V-Ray GPU).

        ``probe=True`` marks a THROWAWAY direction probe and is the only thing that lets
        the Vantage window-grab backend answer instead of V-Ray. Everything that does not
        pass it — the plan-effect probes, the loop, the board, refine, the finals — stays
        on V-Ray by DEFAULT, which is the safe direction for a default to fail in."""
        if probe and getattr(self, "_probe_backend", "vray") == "vantage":
            # REFUSE, do not substitute. render_probe falls through to render_frame on a
            # Vantage refusal, so probe a210 came back a V-Ray plate appended to the same
            # sunsolve table and the same argmax as the Vantage grabs before it. That
            # table is ranked on highlight_similarity, whose presence half is gated on the
            # ABSOLUTE metrics.HOT_THRESHOLD, so a tonemap difference is a systematic
            # offset between the two halves of one comparison — and CROSS_DOMAIN_AGREEMENT
            # and DECISIVE_MARGIN are absolute too, so the mixture also changes which
            # branch of the solver runs and what confidence it reports. sunsolve.probe
            # already treats path=None as a skip, so the grid loses a few samples of ONE
            # domain instead of gaining samples of a second. (2026-07-31)
            path, used = rd.render_probe(cam, out_path, w, h, backend="vantage", log=log,
                                         fallback=False,
                                         should_cancel=self._cancel_poll)
            if path is None:
                # Disarm for the NEXT stage, not this probe, and only after three
                # consecutive refusals: _settled_rows reads a legitimately static dusk
                # viewport as "stale", so a single settle timeout used to disarm the fast
                # path on exactly the scene class where 60 s → 50 ms matters most.
                self._vantage_refusals = getattr(self, "_vantage_refusals", 0) + 1
                if self._vantage_refusals >= 3:
                    self._probe_backend = "vray"
                    if log:
                        self._degrade(log, "vantage probe backend demoted to V-Ray after "
                                            "3 consecutive refused grabs — the remaining "
                                            "direction probes render at full cost")
            else:
                self._vantage_refusals = 0
        else:
            path, used = rd.render_frame(cam, out_path, w, h), "vray"
        # PROVENANCE, recorded rather than inferred: a caller that asked for a probe may
        # still have been handed a V-Ray render (the grab refuses more often than it
        # succeeds), and the black-frame guard is only meaningful on a plate V-Ray made.
        self._last_render_backend = used
        if used == "vantage":
            # A Vantage plate ALREADY carries its exposure and its own tonemap. The
            # software-exposure transform below assumes the renderer handed back the
            # PRE-exposure buffer (core/expose.py:1-14); applied here it would multiply
            # the same EV delta in a second time, the solver would measure the residual
            # and ask for another step, and the loop would overshoot ~2× — bit for bit
            # the failure the _plate_linear branch documents. Same for display_encode.
            return path
        if path and getattr(self, "_plate_linear", False):
            # OCIO/ACES raw save measured (see _verify_exposure_host): the buffer is
            # LINEAR, but metrics decodes sRGB→linear and the LLM is shown the file —
            # encode to display space first so both see what they assume. Failure keeps
            # the raw frame: degraded scores beat a dead loop.
            expose.display_encode_png(path, path)
        if not path or not getattr(self.cfg, "software_exposure", False):
            return path
        st = state
        if st is None:
            try:
                st = ap.read_state(self.rig(), self._baselines, cam)
            except Exception:
                st = None
        if st is None or "exposure.ev" not in st.values:
            return path
        base_ev, base_wb = self._exposure_anchor(entry)
        if expose.expose_image_file(
                path, path, st.get("exposure.ev", base_ev), base_ev,
                st.get("exposure.wb_kelvin", base_wb), base_wb) is None \
                and not getattr(self, "_sw_warned", False):
            self._sw_warned = True
            if log:
                self._degrade(log, "software exposure needs Pillow and Pillow is not "
                                   "installed — every frame from here is scored "
                                   "UN-EXPOSED while the solver keeps prescribing "
                                   "exposure.ev changes that cannot reach the pixels")
        return path

    # ------------------------------------------------------- I2: what will this cost?
    def _observe_render(self, w: int, h: int, seconds: float, backend: str,
                        log: Optional[Callable[[str], None]], started: float = 0.0
                        ) -> None:
        """One timed sample per render, and the FIRST honest cost estimate it enables.

        Three rules, each of which was a way to get the number badly wrong:

          * VANTAGE SAMPLES NEVER PRICE A V-RAY PLAN. A ~50 ms window grab multiplied by
            181 V-Ray renders reads as "9 seconds" for a three-hour run. Same rule the
            black guard already lives by: gate on the backend that actually answered.
          * COLD IS NOT STEADY. The first plate of a session carries one-time scene
            translation, BVH build and the light-cache prepass — on 18M triangles with 460
            lights that is minutes, not seconds. Multiplying it by 181 is how you get a
            wildly wrong estimate, so it is added ONCE and never multiplied.
          * ONE SAMPLE DOES NOT EXTRAPOLATE SILENTLY. With two distinct sizes this fits
            affine t(px) ≈ fixed + slope·px (render time on a heavy V-Ray scene is affine,
            not proportional — the fixed term dominates at small sizes). With one, it
            prices that size and says which basis it used.
        """
        if str(backend).lower() != "vray" or seconds <= 0:
            return
        px = max(1, int(w) * int(h))
        if self._cold_seconds is None and not any(self._render_times.values()):
            # …and it is NOT also filed as a steady sample at its size (2026-07-31). It
            # was, which made the docstring above false and the fit wrong in the worst
            # direction: the canary is the ONLY 160×90 sample there is, so _price's affine
            # branch fitted a line through (14400 px, cold) and (32400 px, warm). On the
            # heavy scenes this model exists for, cold > warm, the slope comes out
            # NEGATIVE, and every larger size clamps to max(0.0, …) — the 480×270 loop
            # and the full-size final, the most expensive frames in the run, priced at
            # 0.0 s and quoted with a confident "affine fit" basis. Measured: canary 180 s
            # @160×90 + first basin probe 60 s @240×135 → "182 minutes, measured".
            self._cold_seconds = float(seconds)
            return
        samples = self._render_times.setdefault(px, [])
        samples.append(float(seconds))
        del samples[:-4]                     # two are enough; four is cheap insurance
        if log is not None:
            self._maybe_report_cost(px, log)

    def _steady_seconds(self, px: int) -> Optional[float]:
        """The fastest sample seen at this size — the steady-state per-frame cost."""
        samples = self._render_times.get(px)
        return min(samples) if samples else None

    def _price(self, px: int) -> Optional[Tuple[float, str]]:
        """→ (seconds for one render at ``px`` pixels, how it was derived)."""
        known = {p: min(v) for p, v in self._render_times.items() if v}
        if not known:
            return None
        if px in known:
            return known[px], "measured at this size"
        if len(known) >= 2:
            lo, hi = min(known), max(known)
            slope = (known[hi] - known[lo]) / float(hi - lo) if hi != lo else -1.0
            # A NON-POSITIVE SLOPE IS NOT A FIT. More pixels cannot cost less time, so a
            # slope ≤ 0 says the two samples are not comparable (different warmth,
            # different backend load, a paused viewport) and the line through them prices
            # everything larger at max(0.0, …) — zero seconds, quoted as a measurement.
            # Fall through to the honest single-sample path instead. (2026-07-31)
            if slope > 0.0:
                fixed = known[lo] - slope * lo
                fitted = fixed + slope * px
                if fitted > 0.0:
                    return fitted, "affine fit over two measured sizes"
        base_px, base_s = min(known.items(), key=lambda kv: abs(kv[0] - px))
        ratio = px / float(base_px)
        if ratio > 4.0 or ratio < 0.25:
            # REFUSE to extrapolate silently. Beyond 4× the fixed/variable split is
            # unknown and a confident wrong number is worse than an honest range.
            return base_s * min(4.0, max(0.25, ratio)), (
                f"EXTRAPOLATED from one sample at {base_px} px — this size is "
                f"{ratio:.1f}× that, so treat it as a lower bound")
        return base_s * ratio, f"scaled from one sample at {base_px} px"

    def cost_estimate(self) -> Optional[Dict]:
        """Price the planned stages from what has actually been measured. → a dict or None.

        Pure arithmetic over ``profiles.planned_renders`` and ``_render_times`` — no
        renders, no guessing, and every "cheaper" figure it quotes is re-priced from the
        same model against a re-planned budget rather than made up.
        """
        if not self._cost_stages or not any(self._render_times.values()):
            return None
        total = 0.0
        rows: List[Dict] = []
        basis = ""
        for stage in self._cost_stages:
            priced = self._price(max(1, stage.width * stage.height))
            if priced is None:
                continue
            seconds, how = priced
            # the WEAKEST basis of any stage, not the first stage's (2026-07-31). The
            # first stage of a standard profile is two 160×90 exposure plates; quoting its
            # basis let a plan whose 7 full-size loop renders were EXTRAPOLATED be
            # announced as "measured at this size". The headline number is only as honest
            # as its least honest term.
            if _BASIS_RANK(how) > _BASIS_RANK(basis):
                basis = how
            total += seconds * stage.count
            rows.append({"key": stage.key, "label": stage.label, "count": stage.count,
                         "seconds_each": round(seconds, 2)})
        if not rows:
            return None
        cold = float(self._cold_seconds or 0.0)
        steady = min(min(v) for v in self._render_times.values() if v)
        warmup = max(0.0, cold - steady)
        # the headline "one probe took N s" quotes the resolution that DOMINATES the plan
        # — 180 of a standard profile's 188 renders are the sweep/polish size, so quoting
        # the first stage (two 160×90 exposure plates) would understate it fourfold
        dominant = max(rows, key=lambda r: r["count"])
        return {"minutes": (total + warmup) / 60.0,
                "renders": sum(r["count"] for r in rows),
                "seconds_each": dominant["seconds_each"],
                "seconds_basis_stage": dominant["label"],
                "warmup_seconds": round(warmup, 1),
                "basis": basis, "stages": rows}

    def _maybe_report_cost(self, px: int, log: Callable[[str], None]) -> None:
        """Fire ONCE, on the first V-Ray plate that can price the plan.

        In run_match order that is the first multi-start basin probe — after roughly 1 of
        190 renders, and BEFORE the 44-probe sun solve and the 120-probe polish. That is
        where "before committing" has to mean. The existing estimate in director.py fires
        at iter00, which on TULA is reached about an hour and sixty renders into a run it
        is supposed to be pricing.
        """
        if self._cost_reported or not self._cost_stages:
            return
        est = self.cost_estimate()
        if est is None:
            return
        self._cost_reported = True
        seconds = est["seconds_each"]
        log(f"cost: one {int(px ** 0.5 * 1.33)}px-class probe took {seconds:.1f} s"
            + (f" (+{est['warmup_seconds']:.0f} s one-time scene translation)"
               if est["warmup_seconds"] >= 1.0 else "")
            + f" — {est['renders']} renders planned → about "
            + _human_minutes(est["minutes"])
            + f" ({est['basis']}).")
        log(f"cost: cancel will take up to {seconds:.0f} s per press — a V-Ray frame "
            "cannot be interrupted once it has started, so ✕ is bounded by one probe.")
        self._gate_cost(est, log)

    def _gate_cost(self, est: Dict, log: Callable[[str], None]) -> None:
        """GATE 5 — above ``cfg.cost_ask_minutes``, that is a decision for the human.

        This is the enforcement config.py's own comment already describes and did not
        implement ("180 probes at 60s a frame is three hours"). The remedies offered are
        re-priced from the SAME model against a re-planned budget — never guessed — and
        the cap line is honest about the fact that a 60 s cap saves nothing when the probe
        already takes 61 s, which is itself the finding.
        """
        limit = float(getattr(self.cfg, "cost_ask_minutes", 0.0) or 0.0)
        if limit <= 0 or est["minutes"] <= limit:
            return
        seconds = est["seconds_each"]
        lines = [f"{est['renders']} renders planned at {seconds:.0f} s each → "
                 f"{_human_minutes(est['minutes'])}."]
        options = [("proceed", f"Proceed — I'll wait {_human_minutes(est['minutes'])}")]
        if seconds > CANCEL_LATENCY_BUDGET_S:
            lines.append(f"Cancel will take up to {seconds:.0f} s per press, and there is "
                         "no way to interrupt a V-Ray frame once it has started.")
            cap = 60.0
            capped = est["minutes"] * min(1.0, cap / seconds)
            lines.append(f"A {cap:.0f} s probe cap would bring that to about "
                         + _human_minutes(capped)
                         + (" — i.e. it would save nothing, because your probes are "
                            "already faster than the cap" if capped >= est["minutes"]
                            else ""))
            if capped < est["minutes"]:
                options.append(("cap", f"Cap probes at {cap:.0f} s "
                                       f"(~{_human_minutes(capped)})"))
        options.append(("stop", "Stop — I'll change something first"))
        answer = self._escalate(askmod.cost_question(
            est["minutes"], est["renders"], seconds, tuple(options),
            "\n".join(lines)), log)
        if answer == "stop":
            self._cancel_latch = "stopped: the cost was declined before it was spent"
            raise MatchCancelled(self._cancel_latch)
        if answer == "cap":
            # Applied through the existing snapshot-and-restore-in-a-finally path, and
            # DECOUPLED from draft_sampler: a time cap the artist just agreed to is a
            # SAFETY control, not a quality control. apply_draft's int-minutes refusal
            # hands control back rather than silently uncapping.
            for line in df.apply_draft(60.0):
                log(line)
            self._cap_applied = df.pending_snapshot()

    #: The four quadrant answers, as a bearing RELATIVE TO THE CAMERA (0 = into the lens,
    #: i.e. the sun is in front of the camera; 180 = behind the camera, lighting the shot).
    QUADRANTS = (("behind", "Behind the camera (lighting the shot)", 180.0),
                 ("left", "Over my left shoulder", 135.0),
                 ("right", "Over my right shoulder", -135.0),
                 ("lens", "Into the lens (backlit)", 0.0))

    def _gate_bearing(self, start, locks: set, sem_live, cfg_kw: Dict,
                      cam_yaw: float, semantics: Dict,
                      log: Callable[[str], None]):
        """GATE 1 — the analyzer does not know where the sun is. ASK, do not spend an hour.

        An artist answering "over my left shoulder" has given a QUADRANT, not an angle, so
        the answer becomes a START VALUE plus 45° of transfer slack and the loop and polish
        may still refine inside it. Locking the axis outright would forfeit refinement the
        render can still do, so only the explicit "I know the angle" branch locks — and
        ``sunsolve.solve_sun_angles`` then returns None on that lock with no new machinery
        in the solver at all.
        """
        est = self.cost_estimate()
        sunsolve_min = 0.0
        for stage in self._cost_stages:
            if stage.key == "sunsolve":
                priced = self._price(max(1, stage.width * stage.height))
                if priced:
                    sunsolve_min = priced[0] * stage.count / 60.0
        spread = semantics.get("sun_bearing_spread_deg")
        cost_txt = (f" (~{_human_minutes(sunsolve_min)} at this scene's measured probe "
                    "time)" if sunsolve_min > 0 else "")
        options = tuple((k, label) for k, label, _b in self.QUADRANTS)
        answer = self._escalate(askmod.Question(
            key="sun_bearing",
            headline=("ANALYZE read the sun's direction "
                      + (f"{int(self.cfg.analyze_samples)} times and got ±{spread:.0f}° "
                         "of scatter" if spread is not None
                         else "and produced no usable direction evidence")
                      + " — it does not know where the light is"),
            detail=(f"Solving it on the render grid costs "
                    f"{sum(s.count for s in self._cost_stages if s.key == 'sunsolve')} "
                    f"probes{cost_txt}. You can answer in five seconds by looking at the "
                    "photograph. A quadrant is enough — it becomes a starting direction "
                    "with 45° of slack, not a lock, so the render can still refine it."),
            options=options + (("solve", f"Solve it on the grid anyway{cost_txt}"),
                               ("stop", "Stop — I'll lock the azimuth myself")),
            default="solve",
            facts={"spread_deg": spread,
                   "agreement": semantics.get("sun_bearing_agreement"),
                   "sunsolve_minutes": round(sunsolve_min, 1)}), log)
        if answer == "stop":
            self._cancel_latch = "stopped: the sun's direction is yours to set"
            raise MatchCancelled(self._cancel_latch)
        if answer == "solve":
            return start, locks
        bearing = dict((k, b) for k, _l, b in self.QUADRANTS).get(answer)
        if bearing is None:
            return start, locks
        azimuth = (cam_yaw + bearing) % 360.0
        if "sun.azimuth_deg" in start.values:
            start.set("sun.azimuth_deg", azimuth)
        if sem_live is not None:
            sem_live["sun_bearing_deg"] = round(bearing, 1)
            sem_live["sun_bearing_slack_deg"] = QUADRANT_SLACK_DEG
            # the artist's eye is better evidence about direction than a decisive grid —
            # full transfer weight, and the slack is what keeps it honest
            sem_live["sun_bearing_agreement"] = 1.0
            cfg_kw["transfer_weight"] = TRANSFER_WEIGHT
        log(f"sun direction: taken from you — {bearing:+.0f}° from the camera "
            f"(azimuth {azimuth:.0f}°), held to ±{QUADRANT_SLACK_DEG:.0f}°. The "
            "44-probe grid solve is SKIPPED; the loop and polish may still refine inside "
            "that quadrant.")
        self._skip_sun_stages = True
        return start, locks

    def _gate_frozen_plates(self, log: Callable[[str], None]) -> str:
        """GATE 4 — six pixel-identical probes in a row while the state kept changing.

        Headless default is "skip", not "continue": spending 38 more renders that are
        provably the same picture is not a defensible default for anyone.
        """
        answer = self._escalate(askmod.Question(
            key="frozen_plates",
            headline=(f"{self._frozen_run} probes in a row rendered PIXEL-IDENTICAL "
                      "while the light being steered changed"),
            detail=("The sun MaxGaffer is driving is not reaching this frame. Preflight's "
                    "SUN_ENABLED and RIG_NOTES lines above name the two likeliest "
                    "causes: the sun is switched off, or it is an untargeted VRaySun, "
                    "which aims by node rotation so azimuth writes do not re-aim it. "
                    "Every remaining direction probe will rank the same picture."),
            options=(("enable", "Stop — I'll enable the right sun and re-run"),
                     ("skip", "Skip the sun stages and match tone only"),
                     ("continue", "Continue anyway")),
            default="skip",
            facts={"frozen_run": self._frozen_run,
                   "frozen_total": self._frozen_total}), log)
        if answer == "enable":
            self._cancel_latch = ("stopped: the light being steered is not reaching the "
                                  "frame")
            raise MatchCancelled(self._cancel_latch)
        if answer == "skip":
            self._skip_sun_stages = True
            self._degrade(log, "sun stages SKIPPED for the rest of this run — their "
                               "probes were provably ranking one unchanging picture")
        return answer

    # ------------------------------------------------------- I5: ask, do not guess
    def _escalate(self, q: "askmod.Question", log: Callable[[str], None]) -> str:
        """Put a decision to the artist — or, headless, to ``cfg.uncertainty_policy``.

        The question and its measured facts are LOGGED BEFORE anyone is asked, every time.
        A question the transcript does not record is the silent-degradation failure again,
        just with better manners. The answer is logged too, WITH who gave it, and both go
        into run.json.
        """
        log(f"? {q.headline}")
        if q.detail:
            for line in str(q.detail).splitlines():
                if line.strip():
                    log("  " + line.strip())
        policy = str(getattr(self.cfg, "uncertainty_policy", "ask") or "ask").lower()
        if policy not in ("ask", "assume", "abort"):
            log(f"  (uncertainty_policy '{policy}' is not a known value — reading it as "
                "'ask', the same defensive default probe_backend uses)")
            policy = "ask"
        answer, who = q.default, "policy"
        if policy == "abort":
            self._decisions.append({"key": q.key, "answer": "abort", "who": "policy",
                                    "headline": q.headline, **dict(q.facts)})
            raise PreflightBlocked(q.headline)
        if policy == "assume":
            log(f"  policy: assume — answering '{q.label_for(q.default)}' without asking")
        else:
            try:
                answer = str(self.ask(q) or "")
            except Exception as err:  # noqa: BLE001 — a broken dialog must not sink a run
                log(f"  ⚠ the question could not be put to you ({err}) — falling back to "
                    f"'{q.label_for(q.default)}'")
                answer = ""
            if answer and answer in q.values():
                who = "artist"
            else:
                if answer:
                    log(f"  ⚠ '{answer}' is not one of this question's answers — using "
                        f"'{q.label_for(q.default)}'")
                answer = q.default
        log(f"  → {q.label_for(answer)} ({who})")
        self._decisions.append({"key": q.key, "answer": answer, "who": who,
                                "headline": q.headline, **dict(q.facts)})
        return answer

    # ------------------------------------------------------- I1: is this plate real?
    @staticmethod
    def _state_fingerprint(state) -> Optional[Dict]:
        if state is None:
            return None
        try:
            out = {str(k): round(float(v), 6) for k, v in state.values.items()}
            out.update({f"group:{k}": round(float(v), 6)
                        for k, v in (state.groups or {}).items()})
            return out
        except Exception:  # noqa: BLE001 — a fingerprint must never fail a render
            return None

    def _validate_plate(self, path: str, w: int, h: int, state,
                        log: Optional[Callable[[str], None]], tag: str,
                        t0: float):
        """I1 — never score an unvalidated image. → the path, or None when rejected."""
        stats = self.stats_for(path)
        self._plate_memo = (path, stats)
        if stats is None:
            return path            # unmeasurable: callers already treat None stats as skip
        fingerprint = self._state_fingerprint(state)
        changed_axes: List[str] = []
        state_changed = False
        if fingerprint is not None and self._plate_prev_state is not None:
            changed_axes = sorted(k for k in set(fingerprint) | set(self._plate_prev_state)
                                  if fingerprint.get(k) != self._plate_prev_state.get(k))
            state_changed = bool(changed_axes)
        try:
            got = png_min.read_png_size(path)
        except Exception:  # noqa: BLE001
            got = None
        verdict = plate.validate(stats, want=(int(w), int(h)), got=got,
                                 prev_sig=self._plate_prev_sig,
                                 state_changed=state_changed,
                                 frozen_run=self._frozen_run,
                                 changed_axes=changed_axes[:4])
        self._plate_prev_sig = verdict.signature
        if fingerprint is not None:
            self._plate_prev_state = fingerprint
        self._frozen_run = verdict.frozen_run
        # STALE — handled at the source rather than as a fifth test: render_frame deletes a
        # pre-existing target before rendering and capture_window_png holds the same
        # contract, so what is left uncovered is a path whose mtime PREDATES this call.
        # One comparison, and t0 is already in hand for the timing.
        try:
            if os.path.getmtime(path) < t0 - 2.0:
                if log:
                    self._degrade(log, f"the renderer returned a plate older than the "
                                       f"call that asked for it ({tag}) — this is a stale "
                                       "file from an earlier run, not a measurement")
                return None
        except OSError:
            pass
        if verdict.reason == "frozen":
            self._frozen_total += 1
            if verdict.frozen_report and log:
                self._degrade(log, verdict.detail)
            return path
        if verdict.ok:
            return path
        if verdict.reason in ("black", "near_black"):
            # STICKY, and not merely an exception: sunsolve.probe catches every exception
            # per probe BY DESIGN (sunsolve.py:117-124), as do the tone-align and
            # sun-solve blocks, so a bare raise would be swallowed 44 times and each
            # swallow would fire another 60-second render. With the flag, every later
            # render raises before rendering (microseconds), the surviving stages degrade
            # the way they already know how, and the error escapes through the
            # draft-restoring finally into the dock's "✗ …".
            head = verdict.detail + f" ({tag})"
            self._probe_abort = (self._black_probe_message(head)
                                 if verdict.reason == "black" else head)
            if log:
                log("✗ " + self._probe_abort)
            raise RuntimeError(self._probe_abort)
        if log:
            self._degrade(log, verdict.detail + f" ({tag}) — this plate was REJECTED, not "
                                                "scored")
        return None

    def _verify_exposure_host(self, cam, run_dir: str,
                              log: Callable[[str], None],
                              should_cancel: Callable[[], bool] = lambda: False) -> None:
        """One-time (per Controller) MEASUREMENT: does the renderer bake EV into the
        saved buffer? Two tiny probes 2 EV apart must move the key ~2 stops; if it
        barely moves, the host is display-stage only (measured on-box for V-Ray GPU)
        and software exposure is switched on for the session, loudly.

        PREFLIGHT B rides on the FIRST of those two plates, at zero additional cost. It
        was already firing two unannounced 160×90 renders here — on TULA that is about two
        minutes of silence immediately after "run dir: …" — so the canary is free: it only
        has to ASSERT things about a plate that was going to be rendered anyway.

        Three assertions and one measurement:
          1. it produced a file at all → else BLOCK;
          2. it is not black → BLOCK only when preflight named a CAUSE. The existing
             reasoning below stays and is right: nothing has been applied yet, so a scene
             whose lights the artist left off is legitimately black here. But that does not
             apply to the DR case — DR-on-with-a-dead-port PLUS a black plate is not an
             unlit scene, it is a dead renderer, and the socket test is exactly what tells
             them apart. Pairing the two is what licenses the promotion;
          3. it is the size that was asked for → else BLOCK. render_frame has three layered
             size spellings, the third of which mutates rt.renderWidth/Height globally; a
             build where all three fall through saves a valid frame at SCENE resolution and
             compute_stats downsamples it to 256 px so the numbers look normal;
          4. wall-clock → the I2 seed. Quoted as a LOWER BOUND, not a scaled-down sample:
             on a heavy scene the ~60 s probe cost is dominated by scene translation and
             BVH build across 18M triangles, not by pixel count.
        """
        if getattr(self, "_exposure_host_checked", False) \
                or getattr(self.cfg, "software_exposure", False) \
                or getattr(self.cfg, "no_renders", False):
            return
        if should_cancel():
            self._cancel_latch = self._cancel_latch or "cancelled before the host check"
            raise MatchCancelled(self._cancel_latch)
        self._exposure_host_checked = True
        from .exposure import ExposureHost

        host = ExposureHost(cam)
        ev0 = host.read_ev()
        if ev0 is None:
            # EVERY early return here logs its reason now. A silent one meant the artist
            # could not tell "measured and fine" from "never measured".
            kind = getattr(host, "kind", "unknown")
            self._degrade(log, f"the exposure host ({kind}) would not report an EV — "
                               "the renderer's exposure was NOT measured this run, so "
                               "software exposure stays as configured")
            return
        wrote = False
        try:
            # DELIBERATELY rd.render_frame and NOT the probe dispatcher: this measures the
            # HOST — whether THIS renderer bakes EV into the saved buffer — and measuring
            # it through a tonemapped Vantage grab is precisely how software_exposure gets
            # switched on by mistake, which then double-applies EV to every later frame.
            log("checking the renderer's exposure host with two 160×90 plates — the "
                "first is also this run's CANARY (it proves the renderer can produce "
                "pixels at the size it was asked for, and it times one frame)")
            _t0 = time.time()
            p1 = rd.render_frame(cam, os.path.join(run_dir, "evcheck_a.png"), 160, 90)
            s1 = metrics.compute_stats(p1) if p1 else None
            self._canary(p1, s1, time.time() - _t0, log)
            wrote = bool(host.write_ev(ev0 + 2.0))
            p2 = rd.render_frame(cam, os.path.join(run_dir, "evcheck_b.png"), 160, 90)
            s2 = metrics.compute_stats(p2) if p2 else None
        finally:
            host.write_ev(ev0)
        if not wrote:
            # write_ev returns False whenever set_prop finds no matching spelling, the
            # host kind is 'none', or the legacy ISO path fails. A failed write means p2
            # rendered at ev0, moved ≈ 0.0, and the branch below switches software
            # exposure ON for the whole session with the confident, WRONG sentence
            # "+2 EV moved the render only 0.00 stops — display-stage only".
            kind = getattr(host, "kind", "unknown")
            self._degrade(log, f"the exposure host ({kind}) REFUSED the +2 EV test "
                               "write — the host was NOT measured, and software exposure "
                               "is left exactly as configured rather than mis-flagged off "
                               "a write that never happened")
            return
        # Two BLACK plates give moved == 0.0, i.e. "this host is display-stage only", and
        # would switch software_exposure ON for the whole session off a dead render. This
        # check runs BEFORE the first render_hook probe, so without the black gate the
        # session is mis-flagged even though the black-frame guard aborts a moment later.
        if metrics.is_black(s1) and metrics.is_black(s2):
            # SAY SO. These are the two earliest and cheapest V-Ray plates of the run and
            # they are the exact evidence the 2026-07-30 post-mortem wanted; a gate that
            # says nothing when it closes is the same failure as the silent draft branch.
            # It reports rather than aborts on purpose: nothing has been APPLIED yet, so a
            # scene whose lights the artist has left off is legitimately black here and is
            # about to be lit by the first guess. The probe guard below sees a plate that
            # was rendered AFTER an apply, which is the one that can tell those apart.
            log("⚠ " + self._black_probe_message(
                "the exposure-host check rendered two 100% BLACK plates before the match "
                "began — software exposure left alone rather than mis-flagged off a dead "
                "render"))
            return
        if not (s1 and s2) or metrics.is_black(s1) or metrics.is_black(s2):
            self._degrade(log, "the exposure-host check could not measure both of its "
                               "plates — the renderer's exposure behaviour is UNKNOWN "
                               "this run and software exposure is left as configured")
            return
        import math

        moved = abs(math.log2(max(1e-5, s1["log_key"]) / max(1e-5, s2["log_key"])))
        if moved < 1.0:      # expected ~2 stops; under half = EV not reaching pixels
            self.cfg.software_exposure = True
            self._sw_auto = True       # WE set it — _bust_stale_caches may clear it
            if not _pillow_available():
                # Pillow is what expose_image_file needs, and without it every frame is
                # scored un-exposed while the analytic solver keeps prescribing
                # exposure.ev changes that cannot reach the pixels: the loop burns its
                # budget, hits its EV leash repeatedly, and director.py blames scene
                # albedo. Drop the axes instead of pretending they are solvable.
                self._unsolvable_axes = {"exposure.ev", "exposure.wb_kelvin"}
                self._degrade(log, "software exposure is needed on this renderer but "
                                   "Pillow is not installed — exposure.ev and "
                                   "exposure.wb_kelvin are DROPPED from the solvable "
                                   "axes for this run (install Pillow, or set "
                                   "system_python, to get them back)")
            log(f"⚠ measured: +2 EV moved the render only {moved:.2f} stops — this "
                "renderer's exposure is display-stage only. Software exposure ON for "
                "this session (turn it on in Settings to persist).")
        elif moved > 3.0:    # ~2× over-response = the saved plate is LINEAR (OCIO/ACES
            # raw save): metrics' sRGB→linear decode then linearizes it TWICE, inflating
            # every EV/WB correction ~2× → the loop oscillates instead of converging
            # (measured on-box 2026-07-24: +2 EV read as 3.96 stops under OCIO_Default).
            self._plate_linear = True
            cs = ""
            try:
                cs = rd.probe_colorspace().get("mode") or ""
            except Exception:
                pass
            log(f"⚠ measured: +2 EV moved the render {moved:.2f} stops (expected ~2) — "
                f"the saved plate is linear (color management: {cs or 'unknown'}). "
                "Loop plates will be sRGB-encoded in software before scoring.")

    def _canary(self, path: Optional[str], stats: Optional[Dict], seconds: float,
                log: Callable[[str], None]) -> None:
        """PREFLIGHT B, riding free on the exposure-host check's first plate.

        BLOCKs on the three conditions where continuing produces a number that is not a
        measurement, and seeds the cost model with the one thing it cannot get any other
        way: a real timed frame from THIS scene."""
        if path is None:
            raise PreflightBlocked(
                "the canary render produced no file at all — this renderer cannot save a "
                "frame at 160×90, so nothing in this run could be measured. Check the "
                "output path, the renderer, and the render type.")
        size = png_min.read_png_size(path)
        if size is not None and size != (160, 90):
            raise PreflightBlocked(
                f"the canary render was asked for 160×90 and came back {size[0]}×"
                f"{size[1]} — none of render_frame's three size spellings bound on this "
                "build, so every 'probe-resolution' frame in this run would actually be a "
                "full-resolution one. That is not a cheap run, and the stats would look "
                "entirely normal.")
        if metrics.is_black(stats):
            cause = getattr(self, "_preflight_black_cause", "")
            if cause:
                raise PreflightBlocked(
                    "the canary render came back 100% BLACK and preflight already named "
                    "the cause: " + cause)
            log("⚠ " + self._black_probe_message(
                "the canary render came back 100% BLACK before anything was applied — "
                "reported, not aborted, because a scene whose lights the artist left off "
                "is legitimately black at this point and is about to be lit by the first "
                "guess"))
        self._observe_render(160, 90, seconds, "vray", None)
        log(f"canary: one 160×90 frame in {seconds:.1f} s — a LOWER BOUND on this "
            "scene's per-frame cost (pixel count is not the driver on a heavy scene; "
            "scene translation and BVH build are), re-priced from the first real probe")

    def _arm_probe_backend(self, log, report=None) -> None:
        """Decide ONCE per run where direction probes come from, and say so.

        Preflight has already done the work — the live link, the window, the stability
        gate, the GPU conflict and the active-camera identity — so this READS a decision
        rather than re-deriving one (and rather than paying vt.link_running's two 1-second
        socket waits on the main thread a second time; controller.py used to call it
        twice per run). Without a report — or with an EMPTY one — it falls back to the old
        direct probe.

        "Empty" is the case that mattered (2026-07-31). run_match always passes a report
        object, never None, and preflight_level='off' (or a preflight that raised and was
        degraded) hands back a report with no findings at all. Reading only ``demotions``
        then armed the Vantage grab having checked NOTHING: not the live link, not the
        window, not the GPU conflict (checklist #14's documented Max-crash configuration),
        not the camera. A report that never ran VANTAGE_ARMED has not armed anything.
        """
        self._probe_backend = "vray"
        self._vantage_refusals = 0
        if str(getattr(self.cfg, "probe_backend", "vray")).lower() != "vantage":
            return
        from . import vgrab

        decided = report.demotions.get("probe_backend") if report is not None else None
        if decided is not None:
            if decided != "vantage":
                # preflight said why, in its own line; do not say it twice
                self._degrade(log, "vantage probe backend NOT armed (see the preflight "
                                   "line above) — every direction probe renders in V-Ray "
                                   "at full cost")
                return
            # the freshness chain is inductive (vgrab.SETTLE_LIMIT_S) — a new run must not
            # inherit the last run's final frame as its baseline
            vgrab.reset_settle()
            self._probe_backend = "vantage"
            log("probe backend: VANTAGE window grab for the sun solve's direction probes "
                "— the sweep, the tone stages, the basin, polish, the plan probes and "
                "the final all stay on V-Ray")
            return
        vgrab.reset_settle()
        port = vt.link_running()
        hwnd = vgrab.find_window(vgrab.VANTAGE_TITLE) if port else None
        if port and hwnd:
            self._probe_backend = "vantage"
            log(f"probe backend: VANTAGE window grab for the sun solve's 44 probes "
                f"(live link on {port}) — the sweep, the tone stages, the basin, polish, "
                f"the plan probes and the final all stay on V-Ray")
            self._degrade(log, "this arming was NOT preflighted (no VANTAGE_ARMED "
                               "finding in this run's report): the live link and the "
                               "window were found, but the GPU-conflict check "
                               "(checklist #14) and the two-grab stability gate did not "
                               "run, so the grabs are of whatever Vantage is showing")
        else:
            self._degrade(log, "vantage probe backend requested but " +
                          ("no live link is streaming" if not port
                           else f"the window was not found ({vgrab.last_error()})") +
                          " — rendering every probe in V-Ray at full cost")

    def _black_probe_message(self, head: str = "") -> str:
        """Name the cause instead of guessing at it. On 2026-07-30 a whole run came back
        black because a dead Chaos Vantage live link left V-Ray's distributed_rendering
        ON, pointed at a dead port with the local machine excluded — and the flag is
        saved WITH the scene, so it recurs on every open (plugcfg/vrayrt_dr.cfg's
        'restore' path writes it back true). Candidates-based read (checklist #15).

        ``head`` names WHICH render came back black; the diagnosis after it is the same
        one wherever the black plate was measured.

        Reporting is where this stops: clearing a render setting behind the artist's back
        is the same house rule that keeps draft_sampler opt-in."""
        dr = None
        try:
            dr = sc.get_prop(sc._rt().renderers.current,
                             ("distributed_rendering", "system_distributedRendering",
                              "distributed_rendering_on"))
        except Exception:  # noqa: BLE001 — a diagnostic must never outrank the diagnosis
            pass
        head = head or ("first probe rendered 100% BLACK — stopping now instead of "
                        "ranking black frames for hours (that is exactly what happened "
                        "on 2026-07-30)")
        if dr:
            return (head + ". MEASURED: this renderer's distributed rendering is ON — a "
                    "dead Vantage live link leaves it pointed at 127.0.0.1:20701 with no "
                    "local server. Set renderers.production.distributed_rendering = false "
                    "and re-run.")
        return (head + ". Likeliest causes: distributed rendering left ON by a closed "
                "Vantage live link (check renderers.production.distributed_rendering), "
                "or the renderer is producing no pixels at all — wrong/hidden camera, an "
                "empty render region, or a failed GPU device.")

    def probe_score(self, camera_name: str, tag: str) -> Optional[float]:
        """One loop-res render of the current scene scored against the camera's reference
        — the cheap 'did that help?' measurement.

        ``self._probe_why`` records WHICH of the six ways this can return None actually
        happened. It used to return a bare None six different ways and ``execute_plan``
        had no ``else`` on either branch, so a scene-wide plan that mutated dozens of
        properties could carry no effect measurement and no explanation of why —
        unmeasured, on the most destructive operation in the plugin. (2026-07-31)"""
        self._probe_why = ""
        if getattr(self.cfg, "no_renders", False):
            self._probe_why = "no-render mode is ON (Settings)"
            return None                    # no-render mode: plans apply unmeasured
        e = self.camera_entry(camera_name)
        if not (e and e.reference):
            self._probe_why = "this camera has no reference bound"
            return None
        ref = self.ref_stats(e.reference)
        cam = self.camera_node(camera_name)
        if ref is None:
            self._probe_why = "the reference could not be decoded by any reader"
            return None
        if cam is None:
            self._probe_why = f"camera '{camera_name}' is not in the scene"
            return None
        path = self._render_exposed(
            cam, os.path.join(self._ensure_run_dir(_safe(camera_name)), f"probe_{tag}.png"),
            self.cfg.loop_width, self.cfg.loop_height, entry=e, tag=f"probe_{tag}")
        if not path:
            self._probe_why = "the probe render produced no usable plate"
            return None
        cur = self.stats_for(path)
        if cur is None:
            self._probe_why = "the probe plate could not be measured"
            return None
        return critic.score(ref, cur, self._critic_weights()).score

    def execute_plan(self, ops, camera_name: str,
                     log: Callable[[str], None], measure: bool = True,
                     should_cancel: Callable[[], bool] = lambda: False) -> Dict:
        """Execute a validated plan (one undo record; MG_ layer for created lights) and
        return the before/after change report — including the plan's MEASURED effect
        (critic score before vs after, one probe render each side). Rig re-classified
        after so new lights join the dimmer boards immediately."""
        self._begin_operation(should_cancel)
        # D2 (2026-07-31): a plan's two probe renders are two full frames, and on a heavy
        # scene that is two minutes the artist may already have asked to skip. Defaulted
        # so api.py and every existing caller are unaffected until they pass one.
        if should_cancel():
            self._cancel_latch = self._cancel_latch or "cancelled before the plan ran"
            raise MatchCancelled(self._cancel_latch)
        before_score = self.probe_score(camera_name, "preplan") if measure else None
        before_why = getattr(self, "_probe_why", "")
        cam = self.camera_node(camera_name)
        report = ex.execute_plan(ops, cam)
        for c in report["changes"]:
            log(f"  {c['target']} · {c['prop']}: {c['before']} → {c['after']}")
        for c in report["created"]:
            log(f"  + {c['type']} '{c['name']}' at {c['at']}")
        for w in report["warnings"]:
            log("  ⚠ " + w)
        self.rig(refresh=True)
        if not measure:
            log("plan effect: NOT measured (the caller asked for no measurement) — the "
                "plan was applied unverified")
        elif before_score is None:
            # I6: no `else` on either branch meant the most destructive operation in the
            # plugin could complete with no effect key and no explanation at all
            log(f"plan effect: NOT measured ({before_why or 'no before-probe'}) — the "
                "plan was applied unverified")
        else:
            after_score = self.probe_score(camera_name, "postplan")
            if after_score is None:
                after_why = getattr(self, "_probe_why", "") or "no after-probe"
                log(f"plan effect: NOT measured ({after_why}) — the plan was applied "
                    "unverified")
            else:
                report["effect"] = {"before": before_score, "after": after_score}
                log(f"plan effect: critic {before_score:.1f} → {after_score:.1f}"
                    + ("  ⚠ the plan made the match WORSE — one Ctrl+Z reverts it"
                       if after_score < before_score - 5.0 else ""))
        return report

    def state_change_rows(self, camera_name: str) -> List[Dict]:
        """pre-match → current saved state, as popup rows (the classic loop's report)."""
        e = self.camera_entry(camera_name)
        if not (e and e.pre_match is not None and e.state is not None):
            return []
        rows = []
        for key, (before, after) in sorted(e.pre_match.diff(e.state).items()):
            rows.append({"target": camera_name, "prop": key,
                         "before": round(before, 2), "after": round(after, 2), "why": ""})
        return rows

    def record_artist_feedback(self, camera_name: str, accepted: bool,
                               rating: Optional[int] = None, note: str = "") -> Dict:
        item = self.session.record_artist_feedback(
            camera_name, accepted, rating, note, self._camera_id(camera_name))
        self.save_session()
        return item

    def bake_lighting_animation(self, camera_name: str, keyframes,
                                step: int = 1, easing: str = "smooth") -> Dict:
        """Interpolate sparse ``(frame, LightingState)`` keys and bake them into Max."""
        cam = self.camera_node(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found in the scene")
        sampled = animation.sample_keyframes(keyframes, step=step, easing=easing)
        if not sampled:
            raise RuntimeError("animation needs at least one lighting keyframe")
        warnings = ap.bake_animation(self.rig(refresh=True), self._baselines,
                                     sampled, cam)
        return {"camera": camera_name, "keys": len(sampled),
                "start": sampled[0][0], "end": sampled[-1][0],
                "step": max(1, int(step)), "easing": easing,
                "warnings": warnings}

    # ------------------------------------------------------------------ fairness / probes
    def assess_fairness(self, camera_name: str,
                        log: Callable[[str], None] = lambda _m: None) -> Dict:
        """Read-only PREDICTIVE fairness estimate (D9): the camera's PRIMARY reference vs the
        current scene render, WITHOUT the director leash / critic content-gap signals (they do
        not exist before a match), so the guarantee narrows to "consistent with the numbers
        fairness can see" (stated in the result's notes). Renders one probe of the current
        scene for cur_stats (skipped in no-render mode) but NEVER applies a state, records a
        match, or saves — the rig is untouched. → a fully-shaped fairness.assess(...) dict."""
        e = self.camera_entry(camera_name)
        if e is None or not e.reference:
            return fairness.assess(None, None)
        refs = self.references(camera_name)
        ref_stats = self.ref_stats(e.reference)
        cur_stats = None
        if not getattr(self.cfg, "no_renders", False):
            cam = self.camera_node(camera_name)
            if cam is not None:
                path = self._render_exposed(
                    cam, os.path.join(self._ensure_run_dir(_safe(camera_name)),
                                      "fairness_probe.png"),
                    self.cfg.loop_width, self.cfg.loop_height, entry=e, log=log)
                cur_stats = self.stats_for(path) if path else None
        return fairness.assess(ref_stats, cur_stats,
                               components=None, coverage=None,
                               n_references=len(refs),
                               roles=[r["role"] for r in refs])

    def rig_report(self, record: bool = False) -> Dict:
        """Read-only census of the LIVE rig: the real property aliases the bridge would touch
        (scene.report_aliases), the scene's colour-management mode (render.probe_colorspace),
        and every Vantage live-link entry point (vantage.probe_entrypoints). Every probe is
        strictly non-destructive — fires no render, flips no toggle, mutates nothing — and
        record-don't-raise, degrading to its empty/legacy shape off-Max. Surfaced under
        run.json's ``probes`` and available to the UI; ``record=True`` also snapshots the
        aliases into scene.LAST_ALIASES for the on-box checklist. → {aliases, colorspace,
        vantage}."""
        report: Dict = {}
        try:
            report["aliases"] = sc.report_aliases(self.rig(), record=record)
        except Exception:  # noqa: BLE001 a diagnostic probe must never fail the caller
            report["aliases"] = {}
        try:
            report["colorspace"] = rd.probe_colorspace()
        except Exception:  # noqa: BLE001
            report["colorspace"] = {}
        try:
            report["vantage"] = vt.probe_entrypoints()
        except Exception:  # noqa: BLE001
            report["vantage"] = {}
        return report

    # ------------------------------------------------------------------ the headline act
    def run_match(
        self,
        camera_name: str,
        log: Callable[[str], None],
        should_cancel: Callable[[], bool] = lambda: False,
        locks: Optional[set] = None,
        on_progress: Optional[Callable[[str, int, int, float], None]] = None,
        do_sweep: bool = False,
        deep: bool = False,
        quality_profile: str = "standard",
        start_override: Optional[LightingState] = None,
        director_note: str = "",
        multi_start: bool = True,
    ) -> MatchResult:
        e = self.camera_entry(camera_name, create=True)
        if not e.reference:
            raise RuntimeError("bind a reference image to this camera first")
        rig = self.rig(refresh=True)
        cam = self.camera_node(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found in the scene")
        # I3 — a NEW run starts uncancelled. The latch is one-way WITHIN a run; only
        # starting the next one clears it, and the dock refuses to start one while a
        # previous generation is still live.
        self._begin_operation(should_cancel)
        self._generation += 1
        self._degradations = []
        self._decisions = []
        self._frozen_run = self._frozen_total = 0
        self._frozen_escalated = False
        self._skip_sun_stages = False
        self._skip_sun_said = False
        self._plate_prev_sig = ()
        self._plate_prev_state = None
        self._cost_reported = False
        self._cost_stages = []
        self._unsolvable_axes = set()
        self._preflight_black_cause = ""
        self._cap_applied = False
        generation = self._generation
        self._drain_warnings(log)
        # the GPU + live-link crash vector (checklist #14) used to be checked here; it is
        # now GPU_VANTAGE_CONFLICT inside preflight, where it sits beside the eighteen
        # other things that were never checked at all — and where its CONSEQUENCE for the
        # Vantage probe backend can be named in the same breath
        if self.cfg.auto_exposure_control:
            from .exposure import ExposureHost, ensure_exposure_control

            if ExposureHost(cam).kind == "none":
                created = ensure_exposure_control(camera=cam)
                if created:
                    log("⚠ " + created)
                    # remember WE created it — Restore must exit the experiment entirely
                    e.ec_created = True
        locks = set(locks if locks is not None else e.locks)
        # UNCONDITIONAL, and back here rather than only inside preflight (2026-07-31).
        # check_camera_active is the REPORT on this call's return value; it is not the
        # call itself, and preflight_level='off' returns before any check runs. V-Ray
        # renders the camera node it is handed either way, but the Vantage live link
        # mirrors the ACTIVE VIEWPORT camera, so a run with preflight off was grabbing
        # 44 direction probes of whatever shot the artist last looked at.
        self._set_active_camera(camera_name)
        run_dir = self._new_run_dir(camera_name)
        log(f"run dir: {run_dir}")

        # ---- I4: PREFLIGHT THE SCENE BEFORE SPENDING ANYTHING.
        # Here, and not later, on purpose: BEFORE pre_match is snapshotted, before the
        # 10-60 s ANALYZE network round trip, and before anything is mutated. It performs
        # no renders and no mutations (one balanced EV round-trip aside), so on a clean
        # scene it costs about forty milliseconds of property reads and one localhost
        # socket refusal — and on 2026-07-30's scene it would have stopped the run in two.
        # ref_stats is hoisted here to feed the reference checks and is _ref_cache-backed,
        # so the pinned read further down is free.
        preflight_report = pf.PreflightReport()
        try:
            preflight_report = pf.run(self, camera_name, cam, rig, e, self.cfg, log,
                                      ref_path=e.reference,
                                      ref_stats=self.ref_stats(e.reference),
                                      run_dir=run_dir)
        except PreflightBlocked:
            raise
        except Exception as err:  # noqa: BLE001 — preflight must never invent a failure
            self._degrade(log, f"preflight could not run ({err}) — this scene was NOT "
                               "verified before the run began")
        dr = preflight_report.find("DR_DEAD_PORT")
        if dr is not None and dr.severity == "block" \
                and str(getattr(self.cfg, "preflight_level", "block")
                        or "block").lower() == "block":
            # remembered so the canary can promote a black plate from "report" to "block":
            # DR-on-with-a-dead-port PLUS a black frame is not an unlit scene, it is a
            # dead renderer, and the socket test is what tells them apart.
            #
            # GATED ON THE LEVEL ACTUALLY IN FORCE (2026-07-31). A Finding keeps its
            # authored severity in the report even when pf.run downgraded it, so reading
            # .severity alone made preflight_level='warn' — whose own tooltip promises
            # "report everything, stop for nothing" — STRICTER than 'off', which produces
            # an empty report and no cause at all. The canary then raised PreflightBlocked
            # on a scene that is legitimately black before the first apply.
            self._preflight_black_cause = dr.detail

        # snapshot the light as it stands — matches are explorations, not commitments
        # (unless the plan stage of this same run already snapshotted the true start).
        # The flag is cleared UNCONDITIONALLY: a stale flag from an aborted plan on
        # another camera must never suppress a later snapshot.
        snapped_for = getattr(self, "_plan_snapped", None)
        self._plan_snapped = None
        if snapped_for != camera_name:
            e.pre_match = ap.read_state(rig, self._baselines, cam)
        self._save_or_warn(log)

        log("analyzing reference…")
        self._llm_down = False
        semantics = self._analyze_or_fallback(camera_name, log)
        log(f"reference: {semantics['time_of_day']}, {semantics['sky']} sky, "
            f"sun {semantics['sun_altitude_band']} @ bearing "
            f"{semantics['sun_bearing_deg']:+.0f}°, wb ~{semantics['wb_kelvin_estimate']:.0f}K"
            f" — {semantics['key_notes']}")

        # pinned AFTER analyze (its io waits are click-windows): all remaining run
        # inputs derive from ONE reference, never a mid-run swap
        ref_path = e.reference
        ref_stats = self.ref_stats(ref_path)
        if ref_stats is None:
            # distinguish the real cause: a MISSING/unreadable file (ref_stats getmtime
            # raised OSError) is not a stats-engine problem — telling the user to install
            # Pillow when their reference path is simply wrong sends them down a dead end
            if not os.path.exists(ref_path):
                log(f"⚠ reference file not found: {ref_path} — the analytic EV/WB "
                    "solver is OFF; re-bind via Load reference…. Running LLM-visual mode")
            else:
                log("⚠ reference unreadable by the stats engine (install Pillow or set "
                    "system_python) — analytic EV/WB solver OFF, LLM-visual mode")
        ref_block = self._image_block(ref_path)
        if ref_block is None and not getattr(self.cfg, "no_renders", False):
            if not os.path.exists(ref_path):
                raise RuntimeError(f"reference file not found: {ref_path} — "
                                   "re-bind it via Load reference…")
            raise RuntimeError("reference image could not be prepared for the LLM")

        if start_override is not None:
            start = start_override.copy()
            log("refine: starting from the ensemble winner (first-guess skipped)")
        else:
            current = ap.read_state(rig, self._baselines, cam)
            start, why = rules.initial_state(semantics, current, sc.camera_yaw_deg(cam),
                                             locks,
                                             overcast_sun_mode=self.cfg.overcast_sun_mode)
            for line in why:
                log("first guess: " + line)

        if getattr(self.cfg, "no_renders", False):
            e.locks = locks
            return self._apply_only(camera_name, e, rig, cam, start, run_dir,
                                    semantics, log)

        self._arm_probe_backend(log, preflight_report)

        draft_applied = False
        if self.cfg.draft_sampler:
            for line in df.apply_draft(getattr(self.cfg, "probe_max_seconds", 0.0)):
                log(line)
            draft_applied = df.pending_snapshot()
        else:
            # A gate that says nothing when it is CLOSED cost three hours on TULA
            # (2026-07-30): draft_sampler:true and a 20 s cap sat in config.json from
            # 15:27, the Options menu wrote false over them at match time, and not one
            # line in the transcript said the branch was False. apply_draft always
            # returns at least one line on every path, so silence here is the only
            # state the artist could not tell apart from "it ran and did nothing".
            msg = ("draft: OFF — probes render at your own settings "
                   "(Options ▾ → Draft sampler, or Settings)")
            secs = float(getattr(self.cfg, "probe_max_seconds", 0.0) or 0.0)
            if secs > 0:
                msg += (f"; the {secs:g}s probe cap is gated by this flag and is "
                        "NOT in effect")
            log(msg)

        # the restoring finally opens IMMEDIATELY after the draft is applied: a gateway
        # error in the exposure check, the sweep or the MatchConfig build must never
        # strand the artist's sampler settings for the rest of the Max session
        try:
            self._verify_exposure_host(cam, run_dir, log, should_cancel)

            profile = profiles.resolve_profile(
                "hero" if deep else quality_profile,
                loop_width=self.cfg.loop_width,
                loop_height=self.cfg.loop_height,
                max_iterations=self.cfg.max_iterations,
                sweep_count=self.cfg.sweep_count,
                target_score=self.cfg.target_score,
            )
            # ONE budget. There used to be two hand-maintained copies of this number and
            # they disagreed by 50 renders — the one printed to the artist said 133 and
            # the progress bar's said 183, and neither counted the tone passes, the
            # exposure-host plates or the final re-render. That is the same
            # two-sources-of-truth shape the draft cap was just fixed for. (2026-07-31)
            self._cost_stages = profiles.planned_renders(
                profile, do_sweep=do_sweep,
                has_sun=rig.get("sun") is not None and start_override is None,
                multi_start=bool(multi_start and start_override is None
                                 and ref_stats is not None),
                exposure_free="exposure.ev" not in locks,
                plan_ran=False, verify_exposure=False, final_render=True,
                ref_has_patches=bool(ref_stats
                                     and float(ref_stats.get("hot_frac") or 0.0) > 0.0))
            hard_cap = sum(s.count for s in self._cost_stages)
            log(f"{profile.label.upper()} PROFILE: {profile.loop_width}×"
                f"{profile.loop_height} loop · "
                + " · ".join(f"≤{s.count} {s.label}" for s in self._cost_stages)
                + f" · hard worst-case ≤{hard_cap} renders")

            # software exposure: apply the just-applied state's EV/WB to each loop frame
            # before it's scored, anchored at the camera's pre_match snapshot (the same
            # anchor probes/board/finals use, so all exposed pixels stay consistent)
            self._sw_state = start
            self._sw_warned = False
            # The FIRST-probe-only black guard is GONE (2026-07-31). Its own comment named
            # the worst case it accepted — a link that dies at probe 50 leaves the
            # remaining 83 probes and the delivered final unvalidated — and every plate is
            # now validated inside _render_exposed, which is strictly cheaper because
            # stats_for reads the memo instead of recomputing.
            if getattr(self.cfg, "software_exposure", False) \
                    and "exposure.ev" in start.values:
                log("software exposure ON — EV/WB applied to loop frames before scoring "
                    "(renderer-independent; V-Ray GPU's exposure is display-stage only)")
            if self._unsolvable_axes:
                locks = set(locks) | set(self._unsolvable_axes)
                log("locked (unsolvable this run): " + ", ".join(sorted(
                    self._unsolvable_axes)))

            # ---- PROGRESS. Stages are counted by their render tags and weighted by the
            # budget each one is allowed, so the bar tracks work actually done rather than
            # a guess. The budget comes from planned_renders — the SAME function that
            # priced the run — so the bar and the estimate can never disagree again.
            _budget_of = {s.key: s.count for s in self._cost_stages}
            _stage_budget = [
                ("basin", "multi-start", max(1, _budget_of.get("basin", 6))),
                ("sunsolve", "sun solve", max(1, _budget_of.get("sunsolve", 44))),
                ("sweep", "sun sweep", max(1, _budget_of.get("sweep",
                                                             profile.sweep_count))),
                ("iter", "match loop", max(1, _budget_of.get("iter",
                                                             profile.max_iterations))),
                ("polish", "polish", max(1, _budget_of.get(
                    "polish", profile.polish_max_probes))),
            ]
            _total_units = float(sum(b for _k, _l, b in _stage_budget)) or 1.0
            _seen = {k: 0 for k, _l, _b in _stage_budget}

            def _stage_of(tag: str) -> str:
                if tag.startswith("sweep_basin"):
                    return "basin"
                if tag.startswith("sunsolve"):
                    return "sunsolve"
                if tag.startswith("sweep"):
                    return "sweep"
                if tag.startswith("polish"):
                    return "polish"
                return "iter"

            def _tick(tag: str) -> None:
                if on_progress is None:
                    return
                st = _stage_of(tag)
                _seen[st] = _seen[st] + 1
                done_units = 0.0
                label, cur, cap = st, _seen[st], 1
                for key, lab, budget in _stage_budget:
                    # a stage that overran its budget still counts as finished, never >100
                    done_units += min(_seen[key], budget)
                    if key == st:
                        label, cap = lab, budget
                try:
                    on_progress(label, min(cur, cap), cap,
                                max(0.0, min(100.0, 100.0 * done_units / _total_units)))
                except Exception:  # noqa: BLE001 — a readout must never sink a match
                    pass

            def render_hook(tag: str):
                # Every unit of work in a match passes through here, and the tag says which
                # STAGE it belongs to — so this is the one place progress can be counted
                # without threading a callback through the solver, the sweep, the loop and
                # polish separately. A match can run for many minutes; without this the
                # artist has no way to tell a working run from a hung one.
                if self._probe_abort:      # STICKY — see _validate_plate
                    raise RuntimeError(self._probe_abort)
                _tick(tag)
                # I3 — LATCH the artist's ✕ here and let _render_exposed enforce it. Once
                # set it stays set for the run: a stage that swallows the exception
                # (sunsolve does, by design) simply raises again on its next probe at
                # microsecond cost instead of firing another 60-second render each time.
                if not self._cancel_latch and should_cancel():
                    self._cancel_latch = "cancelled by the artist"
                # DIRECTION-solving stages render small. They are comparing where the light
                # falls, not judging tone, and there are dozens of them: the global sun
                # solve alone probes up to 56. Measured — it was rendering at full loop
                # size because its tags start with "sunsolve" rather than "sweep", which
                # turned a two-minute stage into most of a fifty-minute match.
                is_sweep = tag.startswith(("sweep", "sunsolve"))
                if is_sweep:
                    width, height = profile.sweep_width, profile.sweep_height
                elif tag.startswith("polish"):
                    # POLISH renders half-size. It is 82% of a match and asks only "did that
                    # nudge help" — a comparison, not a verdict. Measured on-box 2026-07-26:
                    # half-resolution ranked eight varied states identically to full and
                    # scored within 0.63 of it, at half the render time. The state it lands
                    # on is re-rendered and re-scored at FULL size afterwards, so nothing
                    # the artist is shown or told comes from a cheap frame.
                    width, height = profile.polish_size
                else:
                    width, height = profile.loop_width, profile.loop_height
                # Geometry-only, and named by PREFIX rather than by exception. The global
                # sun solve is the one stage whose probes exist purely to be RANKED
                # against each other on the hot-patch map and whose output is an angle —
                # the one thing that transfers between renderers — and it is also 44 of
                # the 133 probes in a standard profile, so it is where the saving is.
                # Three neighbours that look like they qualify do not:
                #   * sunsolve_tonealign exists to set the artist's EXPOSURE via solve_ev,
                #     so it must be measured on the renderer that owns that exposure;
                #   * sweep_basin_* is the multi-start picker (_stage_of already calls it
                #     its own stage). It scores on the FULL weighted critic, which takes
                #     min(comps.values()) across every measured component (critic.py:299-
                #     314) — a cross-renderer key/colour collapse costs up to 33 points
                #     there and would drag the pixel term down against the state-derived
                #     transfer term, choosing the basin on the wrong evidence. The basin
                #     caps the whole run (polish is a basin-finisher);
                #   * sweep### (8 probes) is scored on cosine(ref_grid, probe_grid) —
                #     the reference PHOTO's 3x3 luminance grid against the probe's — and
                #     its contrastive table subtracts the probes' mean grid FROM the
                #     reference's (director.py:1290-1315), which only means anything while
                #     both sides come off the same tone curve. Its plates are also the
                #     ones uploaded to the model by llm_pick, and a screenshot of another
                #     renderer is not what that prompt describes. Eight probes is not
                #     worth a second cross-source path.
                # exactly the DIRECTION probes: sunsolve's coarse/fine grid and the sun
                # sweep's sweep### frames. NOT sunsolve_tonealign (an exposure stage that
                # happens to share the prefix) and NOT sweep_basin_* (the multi-start
                # picker, which is not a sun stage at all).
                sun_probe = (tag.startswith(("sunsolve_a", "sunsolve_fine"))
                             or (tag.startswith("sweep")
                                 and not tag.startswith("sweep_basin")))
                if self._skip_sun_stages and sun_probe:
                    # THE STAGE THAT IS RUNNING, not just the next one (2026-07-31).
                    # _skip_sun_stages is set from inside this hook — GATE 4 fires at the
                    # 6th frozen plate, i.e. coarse probe 7 of 44 — and it was only read
                    # at the two stage entrances further down. An artist who answered
                    # "skip the sun stages" still paid for the remaining ~37 probes of
                    # the solve already in flight (~37 minutes on TULA), and the report
                    # then told them the sun stages had been SKIPPED. sunsolve.probe reads
                    # None as a skip and run_sun_sweep handles a None plate, so returning
                    # here ends the stage in progress at its next probe.
                    if not getattr(self, "_skip_sun_said", False):
                        self._skip_sun_said = True
                        self._degrade(log, "sun direction probes are being SKIPPED from "
                                           "here — the stage already in flight is ending "
                                           "at this probe rather than finishing its grid")
                    return None
                use_vantage = tag.startswith(("sunsolve_a", "sunsolve_fine"))
                path = self._render_exposed(
                    cam, os.path.join(run_dir, f"{tag}.png"),
                    width, height,
                    state=getattr(self, "_sw_state", None) or start, entry=e, log=log,
                    probe=use_vantage, tag=tag)
                # I1 lives in _render_exposed now — EVERY plate from EVERY caller, both
                # backends, black / near-black / wrong-size / frozen / stale. The
                # 29bbae6 one-shot guard that used to sit here is deleted: its own comment
                # named the case it accepted (the link dies at probe 50; 83 probes and the
                # delivered final go unvalidated), and it could only ever see render_hook.
                if not self._frozen_escalated \
                        and self._frozen_run >= plate.FROZEN_ESCALATE_RUN:
                    self._frozen_escalated = True
                    self._gate_frozen_plates(log)
                if path:
                    log(f"THUMB::{path}")   # UI renders these markers as inline thumbnails
                return path

            def apply_hook(st):
                self._sw_state = st        # the frame render_hook is about to expose
                self._apply_logged(rig, st, cam, log)

            # SEMANTIC transfer hook: ANALYZE already read the reference's sun bearing,
            # altitude band, colour temperature, hardness and haze. Pixel statistics
            # cannot pin sun direction on an interior (measured: 64 degrees out still
            # scored 90.92), so the search is also judged on agreeing with that reading.
            cam_yaw = sc.camera_yaw_deg(cam)
            # How far to trust that reading, from how well it agreed with ITSELF. Measured
            # on-box 2026-07-25, four runs against one golden-hour reference read the sun
            # bearing as 45.0, -52.5, 77.6 and 64.9 degrees. Weighting a coin flip at a
            # flat 0.25 makes the match faithfully chase a wrong sun; the run that scored
            # 94.75 was partly luck in the draw. Trust now scales with the samples'
            # agreement.
            #
            # The floor is deliberate and is NOT zero: pixels are not merely noisy about
            # sun direction, they are blind to it (a 64-degree error still scored 0.957 on
            # the critic's direction component), so even a contested reading is the best
            # direction evidence available. Full trust when the samples agree, a quarter of
            # it when they do not.
            bearing_trust = 1.0
            cfg_kw = {"transfer_weight": 0.0}
            if semantics:
                # ABSENT EVIDENCE IS NOT EVIDENCE. This read `.get(..., 1.0)`, so a
                # gateway-down run — whose DEFAULT_SEMANTICS carry no agreement key at all
                # — steered 10% of every loop and polish score toward an INVENTED -60°
                # sun, and printed "sun mid @ bearing -60°" in the same sentence shape it
                # uses for a real reading. A missing key now reads as zero trust.
                # (2026-07-31)
                bearing_trust = max(0.25, min(1.0, float(
                    semantics.get("sun_bearing_agreement", 0.0) or 0.0)))
                spread = semantics.get("sun_bearing_spread_deg")
                if bearing_trust < 1.0:
                    log("⚠ ANALYZE "
                        + (f"disagreed with itself on the sun's direction (±{spread:.0f}° "
                           f"across samples)" if spread is not None
                           else "left no evidence about the sun's direction at all")
                        + f" — leaning harder on the render and less on the reading "
                          f"(trust {bearing_trust:.0%})")
                cfg_kw["transfer_weight"] = TRANSFER_WEIGHT * bearing_trust

            # A COPY: the sun sweep updates the bearing below, and e.semantics is the
            # cached record of what ANALYZE actually read. Writing a sweep-derived bearing
            # back into the cache would launder a measurement into the reading and poison
            # every later run of this camera.
            sem_live = dict(semantics) if semantics else None

            # ---- GATE 1: THE DIRECTION GATE (I5).
            # Fires only when the 44-probe solve is actually going to run and the analyzer
            # has admitted it does not know. On 2026-07-30 that was ±76° of scatter and
            # ~90 minutes of probing a coin toss, with the artist sitting right there.
            if (start_override is None and rig.get("sun") is not None
                    and "sun.azimuth_deg" not in locks and ref_stats is not None
                    and float(ref_stats.get("hot_frac") or 0.0) > 0.0
                    and bearing_trust <= max(0.25, BEARING_ASK_AGREEMENT)
                    and float(semantics.get("sun_bearing_agreement", 0.0) or 0.0)
                    < BEARING_ASK_AGREEMENT):
                start, locks = self._gate_bearing(
                    start, locks, sem_live, cfg_kw, cam_yaw, semantics, log)

            def transfer_hook(st):
                try:
                    return transfer.score(sem_live, st, cam_yaw)["score"] / 100.0
                except Exception as err:  # noqa: BLE001
                    # Swallowing to None DELETES the semantic quarter of the objective for
                    # the whole match — after the log line has already announced its
                    # weight. Say it once, then keep degrading quietly so 190 probes do
                    # not each print it. (2026-07-31)
                    if not getattr(self, "_transfer_warned", False):
                        self._transfer_warned = True
                        self._degrade(log, f"the semantic transfer objective FAILED "
                                           f"({err}) — the announced "
                                           f"{cfg_kw['transfer_weight']:.0%} of the "
                                           "objective that judges agreement with the "
                                           "reference's lighting reading is not in effect")
                    return None

            hooks = Hooks(
                apply=apply_hook,
                render=render_hook,
                stats=self.stats_for,
                llm_deltas=self._llm_deltas_hook(ref_block),
                log=log,
                should_cancel=should_cancel,
                transfer=transfer_hook if semantics else None,
            )

            agreement = getattr(self, "_last_analyze_agreement", None)
            if agreement is not None and agreement < 1.0:
                log(f"⚠ analyze samples disagreed (agreement {agreement:.0%}) — consensus "
                    "used; consider a cleaner reference if the match fights you")
                self._last_analyze_agreement = None

            # MULTI-START over the shipped preset rigs. The loop used to commit to ONE
            # first guess, and polish can only refine WITHIN the basin it is handed
            # (it is a basin-finisher by design) — measured on-box 2026-07-25: the loop
            # delivered a 77.6 basin and polish could only reach ~91 from it, while the
            # same polish reaches 96.6 from a good basin. We already ship six hand-tuned
            # scenario rigs for the BOARD; probing them at SWEEP resolution costs a
            # handful of small renders and starts the match from whichever basin
            # actually measures closest to the reference.
            if multi_start and start_override is None and ref_stats is not None \
                    and not should_cancel():
                start = self._pick_start_basin(start, semantics, rig, cam, e, locks,
                                               hooks, ref_stats, log)
                self._sw_state = start

            # The start state decides the lighting ARCHETYPE; the loop refines inside it and
            # does not get to abolish the key light. Without this the search kept finding a
            # SUNLESS metamer of a sunlit reference — measured 2026-07-25: sun switched off,
            # then dome 1.75 and WB pinned at 15000 compensating for the missing key,
            # azimuth left anywhere (no sun = no direction gradient), scoring ~92 with
            # structurally wrong light. Single-axis escapes cannot undo it because the
            # compensators have co-adapted, so the sun is pinned ON for the rest of the
            # match. It can still be DIMMED via sun.intensity, and a genuinely sunless start
            # (overcast/cool-north) is left free.
            #
            # This guard belongs to the STATE, not to how the state was chosen. It used to
            # sit inside the multi-start branch above, so `refine` — which supplies a
            # start_override and therefore skips that branch — was the one path that could
            # hand the loop a sun-off start with nothing pinning the sun back on.
            if start.get("sun.enabled", 0.0) >= 0.5 and "sun.enabled" not in locks:
                locks = set(locks) | {"sun.enabled"}
                log("sun locked ON for this match (the starting rig is sunlit — it may "
                    "be dimmed, not switched off)")

            # TONE BEFORE GEOMETRY. The sun solve ranks directions on the hot-patch
            # map, and that map is thresholded on absolute luminance — so the probes'
            # EXPOSURE decides which geometry looks right. Measured on the street scene,
            # same photo, three runs: exposure free -> the backlit 277-degree sun (the
            # photograph's character, 76.6); exposure locked a stop brighter -> a bland
            # front-lit 53-degree sun (63.2), because under a hot tone a front-lit canopy
            # out-glows the true glare streak. So exposure is aligned to the reference
            # FIRST, with up to two small probes; a locked exposure is respected and the
            # consequence said out loud rather than silently deciding the basin.
            if start_override is None and ref_stats is not None                     and rig.get("sun") is not None and not should_cancel():
                if "exposure.ev" in locks:
                    log("exposure.ev is locked — the sun solve will judge geometry "
                        "under your chosen tone, which can favour a different basin")
                elif "exposure.ev" in start.values:
                    # an ASSIST, like the solve it serves: losing it costs the assist,
                    # never the match — off-box and degraded rigs skip it cleanly
                    try:
                        for _tone_pass in range(2):
                            hooks.apply(start)
                            tone_path = hooks.render("sunsolve_tonealign")
                            tone_stats = (self.stats_for(tone_path)
                                          if tone_path else None)
                            if tone_stats is None:
                                # a break with no line looked exactly like a stage that
                                # ran and agreed with itself (2026-07-31)
                                self._degrade(log, f"tone-align stopped after "
                                                   f"{_tone_pass} pass(es): the probe "
                                                   "could not be measured, so the sun "
                                                   "solve judges geometry at the "
                                                   "starting exposure")
                                break
                            ev_new = solver.solve_ev(ref_stats, tone_stats,
                                                     start.get("exposure.ev"))
                            if ev_new is None:
                                log(f"tone-align: settled after {_tone_pass + 1} pass(es) "
                                    "— the exposure solver asked for no further change")
                                break
                            log(f"tone-align: exposure.ev "
                                f"{start.get('exposure.ev'):.2f} → {ev_new:.2f} so the "
                                f"sun solve judges geometry at the reference's exposure")
                            start.set("exposure.ev", ev_new)
                    except (MatchCancelled, PreflightBlocked):
                        raise      # a validated stop is not an "assist that failed"
                    except Exception as err:  # noqa: BLE001
                        self._degrade(log, f"tone-align SKIPPED ({err}) — the sun solve "
                                           "will judge geometry at whatever exposure the "
                                           "starting rig happens to have")

            # GLOBAL SUN SOLVE, before the sweep and usually instead of it. Sun direction
            # was this plugin's worst failure and it was being attacked with local search:
            # measured across four runs of ONE golden-hour interior, azimuth came back 2.8,
            # 171, 168 and 78 degrees out. The strategy was never the problem — nothing
            # could TELL. The critic's only spatial descriptor was an averaged luminance
            # grid, which scored a 171-degree error 0.922 and a 13.5-degree error 0.917.
            # Patch agreement separates those same states 0.912 against 0.546, and with a
            # measure that discriminates, two bounded angles are cheaper to SOLVE on a grid
            # than to hill-climb. A grid cannot land in a local optimum.
            solved: Optional[Dict] = None
            if self._skip_sun_stages:
                log("sun solve: SKIPPED — the direction is already settled (you answered "
                    "it, or the probes were provably ranking one unchanging picture)")
            elif start_override is None and rig.get("sun") is not None \
                    and ref_stats is not None and not should_cancel():
                try:
                    solved = sunsolve.solve_sun_angles(
                        start, ref_stats, hooks.apply, hooks.render, self.stats_for,
                        log=log, should_cancel=should_cancel, locks=locks,
                        max_probes=sum(s.count for s in self._cost_stages
                                       if s.key == "sunsolve") or 56)
                except (MatchCancelled, PreflightBlocked):
                    raise      # a validated stop is not an "assist that failed"
                except Exception as err:  # noqa: BLE001 — the solve is an ASSIST; losing
                    # it must cost the assist, never the match (the sweep still runs)
                    self._degrade(log, f"sun solve FAILED ({err}) — falling back to the "
                                       "sweep, which is a coarser instrument")
                    solved = None
            if solved is not None and solved.get("confidence", 0.0) >= 0.5:
                start.set("sun.azimuth_deg", solved["azimuth_deg"])
                if solved.get("altitude_deg") is not None \
                        and "sun.altitude_deg" not in locks \
                        and "sun.altitude_deg" in start.values:
                    start.set("sun.altitude_deg", solved["altitude_deg"])
                hooks.apply(start)
                self._sw_state = start
                # the objective is aimed at what was MEASURED here, at this solve's own
                # resolution, so it neither fights the answer nor defends it more precisely
                # than the grid can justify
                if sem_live is not None:
                    sem_live["sun_bearing_deg"] = round(
                        (solved["azimuth_deg"] - cam_yaw + 180.0) % 360.0 - 180.0, 1)
                    sem_live["sun_bearing_slack_deg"] = 15.0
                    sem_live["sun_bearing_agreement"] = solved["confidence"]
                    cfg_kw["transfer_weight"] = TRANSFER_WEIGHT * max(
                        0.25, min(1.0, solved["confidence"]))
                do_sweep = False          # the better instrument already answered
            elif solved is not None:
                log("sun solve was not decisive — falling back to the sweep, and the "
                    "answer will be held loosely either way")

            if do_sweep and self._skip_sun_stages:
                log("sun sweep: SKIPPED — the direction is already settled")
                do_sweep = False
            if do_sweep and start_override is None and rig.get("sun") is not None \
                    and "sun.azimuth_deg" not in locks:
                log(f"sun sweep: {profile.sweep_count} directions at "
                    f"{profile.sweep_width}×{profile.sweep_height}…")
                sweep_out: Dict = {}
                az, alt_hint, _why = run_sun_sweep(
                    start, rules.sweep_azimuths(profile.sweep_count), hooks,
                    llm_pick=lambda paths, azs: self._sweep_call(ref_block, paths, azs),
                    ref_stats=ref_stats, out=sweep_out)
                if _why:
                    # run_sun_sweep produces four distinct failure strings and every one
                    # of them was DISCARDED at this line — the sweep's basis ("metric-
                    # only" when the gateway is down) among them. (2026-07-31)
                    log("sun sweep: " + str(_why))
                if az is None:
                    self._degrade(log, "the sun sweep returned NO direction" +
                                  (f" ({_why})" if _why else "") +
                                  " — the sun stays where the starting rig put it")
                if az is not None:
                    start.set("sun.azimuth_deg", az)
                    # ANALYZE estimates the bearing from ONE image, which is a hard
                    # absolute judgement and measurably unreliable: four reads of one
                    # golden-hour reference gave 45.0, -52.5, 77.6 and 64.9 degrees, and on
                    # A2 the sign came out backwards, putting the sun 19.4 degrees off —
                    # exactly twice the 9.7 it reported. The sweep is a different and better
                    # kind of evidence: it renders candidate directions in THIS scene and
                    # picks comparatively, which is a far easier judgement than naming an
                    # angle. So the reading is the PRIOR, the sweep is the measurement that
                    # updates it, and the transfer objective then defends that answer
                    # against metamer drift for the rest of the loop. Without this the
                    # objective spent the whole match pulling the sun back off the swept
                    # direction and onto the noisy estimate.
                    if sem_live is not None:
                        swept_bearing = (az - cam_yaw + 180.0) % 360.0 - 180.0
                        prior = sem_live.get("sun_bearing_deg")
                        # Point the objective at where the sweep put the sun, so it neither
                        # defends nor fights the measurement — the WEIGHT decides how hard
                        # it resists polish moving away from it.
                        sem_live["sun_bearing_deg"] = round(swept_bearing, 1)
                        # ...to the resolution the sweep actually has. Inside half a step
                        # the sweep cannot tell directions apart, so the objective must not
                        # pretend it can — polish and the pixel term choose within the
                        # bracket.
                        sem_live["sun_bearing_slack_deg"] = max(
                            15.0, 180.0 / max(2, profile.sweep_count))
                        # How hard, from how DECISIVE the sweep actually was. Asserting
                        # certainty here was a mistake and it cost a match: on an interior
                        # where every direction lit the room about equally the sweep landed
                        # ~195 degrees out, full trust made the objective defend a sun on
                        # the far side of the building, and the run finished at 84.95 with
                        # the azimuth 180 degrees wrong. The sweep already knew it was
                        # guessing — the margin between its winner and the runner-up was
                        # right there and nobody looked at it.
                        sweep_conf = float(sweep_out.get("confidence", 0.5) or 0.0)
                        # the best direction evidence available, never worse than the
                        # reading it replaces
                        sem_live["sun_bearing_agreement"] = max(
                            sweep_conf, float(semantics.get("sun_bearing_agreement", 0.0)
                                              or 0.0))
                        bearing_trust = max(0.25, min(
                            1.0, sem_live["sun_bearing_agreement"]))
                        cfg_kw["transfer_weight"] = TRANSFER_WEIGHT * bearing_trust
                        if sweep_conf < 0.5:
                            log(f"⚠ the sweep could not separate the sun's direction "
                                f"(winner led by {sweep_out.get('margin')}) — every "
                                f"direction lights this scene about equally. Holding the "
                                f"answer loosely (trust {bearing_trust:.0%}); lock "
                                f"sun.azimuth_deg if you know where the light is.")
                        elif prior is not None and abs(
                                (float(prior) - swept_bearing + 180.0) % 360.0 - 180.0) > 20.0:
                            log(f"sweep measured the sun at {swept_bearing:+.0f}° from the "
                                f"camera; ANALYZE had estimated {float(prior):+.0f}° — "
                                f"going with the render (confidence {sweep_conf:.0%})")
                    # the hint was judged against real renders of THIS scene — trust it over
                    # the ANALYZE band when the altitude isn't locked
                    if alt_hint != "na" and "sun.altitude_deg" not in locks \
                            and "sun.altitude_deg" in start.values:
                        start.set("sun.altitude_deg", rules.ALTITUDE_DEG.get(
                            alt_hint, start.get("sun.altitude_deg")))
                        log(f"sweep: altitude refined to "
                            f"{start.get('sun.altitude_deg'):.0f}° ('{alt_hint}')")

            cfg = MatchConfig(
                max_iterations=profile.max_iterations,
                target_score=profile.target_score,
                analytic=ref_stats is not None,
                weights=self._critic_weights(),
                polish=profile.polish,
                polish_rounds=profile.polish_rounds,
                polish_max_probes=profile.polish_max_probes,
                # the loop solves GEOMETRY; it was quitting after 3 of 10 hero iterations
                # on one dip because this never came through from the profile
                stall_patience=profile.stall_patience,
                # a quarter of the objective is "does this rig agree with what ANALYZE
                # read off the reference" — enough to pin direction, not enough to
                # override the pixels that carry tone and detail
                transfer_weight=cfg_kw["transfer_weight"],
                cost_reported=self._cost_reported,
            )
            if profile.polish:
                log("HERO MATCH: target 99 · bounded coordinate-descent polish "
                    "to the measured scene ceiling")
            result = run_match(start, ref_stats, semantics, hooks, cfg, locks,
                               rig_notes="; ".join(rig.get("notes", [])),
                               director_note=director_note)
        finally:
            if draft_applied or getattr(self, "_cap_applied", False):
                # crash-safe: even a raise puts the artist's sampler back. The gate-5 cap
                # rides the same snapshot-and-restore path, so agreeing to a cap can no
                # more strand a render setting than draft mode can.
                self._cap_applied = False
                for line in df.restore_draft():
                    log(line)
        e.locks = locks
        result.degradations = list(self._degradations)
        result.decisions = list(self._decisions)
        # I3/D4 — a run the dock DETACHED must not write anything back. The dialog at
        # _force_release promises the result "will be discarded"; before this, only the
        # dock's return value was discarded while the controller went on to apply state,
        # record the match and save the session minutes later. (2026-07-31)
        if self._generation != generation:
            self._degrade(log, f"run #{generation} was detached while it was still "
                               f"running — its result is DISCARDED and the scene is left "
                               "exactly as this later run leaves it")
            return result
        if result.best_score is None and result.stop_reason in ("cancelled",
                                                                "render_failed",
                                                                "max_iterations",
                                                                "stalled"):
            # nothing was ever measured — recording would overwrite the camera's
            # accepted state+score with an unmeasured first guess. Put the scene back
            # to its previous light instead (the director skipped its final apply too).
            #
            # "max_iterations"/"stalled" joined the set 2026-07-31: repeated stats
            # failures produce a FULL match at best_score None under those reasons, which
            # used to fall through to the else-branch — so the controller recorded the
            # match, applied the final state, and handed the artist "best score n/a" with
            # nothing saying no frame in the run had ever been measured.
            prev = e.state if e.state is not None else e.pre_match
            if prev is not None:
                self._apply_logged(rig, prev, cam, log)
            log(f"✗ match {result.stop_reason} and NO FRAME IN THIS RUN WAS EVER "
                "MEASURED — nothing was recorded, kept previous lighting")
        else:
            # LIGHTING TRANSFER, reported alongside the pixel score. The critic answers
            # "do these frames match", which is unanswerable when the reference is a photo
            # of a different building; this answers "is the sun where that photo's sun is,
            # at that colour temperature, hardness and haze" — the question the artist
            # actually asked, and the only one with a fair answer cross-domain.
            try:
                tr = transfer.score(semantics, result.best_state,
                                    sc.camera_yaw_deg(cam))
                result.transfer = tr
                log("lighting transfer: %.1f/100 (%s)"
                    % (tr["score"], ", ".join(tr.get("notes") or ["faithful"])[:160]))
            except Exception:  # noqa: BLE001 — a diagnostic must never sink a match
                pass
            # REPORT the pixel similarity, not the search objective. Blending the
            # reference's lighting reading into the objective is what breaks the ties
            # pixels cannot see, and it stays — but it must not be the headline number,
            # because after the sweep the objective aims at the sweep's OWN answer. The
            # search then scores itself as agreeing with a target it chose, and the bonus
            # is self-congratulation. Measured on-box 2026-07-25: a match whose sun was
            # 171 degrees from the reference reported 86.27 while its best plate scored
            # 77.21 against that reference. The artist is owed the 77.
            try:
                # Polish probes render half-size, so its winning plate is a CHEAP frame.
                # Re-render the state it landed on at full loop size before scoring or
                # showing anything: the reported number and the plate the artist sees must
                # both come from a full render, or the speed-up would be paid for in
                # honesty. One render, at the end, against 120 saved during polish.
                if self._cancel_latch and result.best_state is not None:
                    # ✕ must not buy one more full-size frame. This block fires whenever
                    # best_state and best_render exist — including on a cancelled run —
                    # and on TULA that is another 60 seconds after the artist asked it to
                    # stop. The polish plate stands as the best render; the score below is
                    # the one already measured for it. (2026-07-31)
                    self._degrade(log, "cancelled — the final full-size re-render was "
                                       "SKIPPED, so the reported score comes from the "
                                       "half-size polish plate the search landed on")
                elif result.best_state is not None and result.best_render:
                    full = os.path.join(run_dir, "final_full.png")
                    ap.apply_state(rig, self._baselines, result.best_state, cam,
                                   undo=False)
                    shot = self._render_exposed(cam, full, profile.loop_width,
                                                profile.loop_height,
                                                state=result.best_state, entry=e, log=log)
                    if shot:
                        result.best_render = shot
                honest = self.stats_for(result.best_render) if result.best_render else None
                if honest is not None and ref_stats is not None:
                    verdict = critic.score(ref_stats, honest, self._critic_weights())
                    if result.best_score is not None and abs(
                            verdict.score - result.best_score) > 0.05:
                        log(f"score {verdict.score:.1f} (the search steered on "
                            f"{result.best_score:.1f}, which counts agreement with the "
                            f"reference's lighting reading)")
                    result.objective_score = result.best_score
                    result.best_score = verdict.score
                    result.best_components = verdict.components
                    # SUN-PATCH agreement, reported beside the score because the score
                    # cannot see it. The weighted critic averages its grid cells, which is
                    # what erases a sun patch: measured on-box 2026-07-25, a match with no
                    # directional light anywhere scored 0.92 on the direction component
                    # against a reference covered in golden floor patches. This reads 0.56
                    # on that same pair. Diagnostic ONLY for now — folding it into the
                    # weighted score changes every number in the archetype matrix and has
                    # to be validated across all of them first.
                    hi = metrics.highlight_similarity(ref_stats, honest)
                    if hi is not None:
                        result.highlight = hi
                        if hi < 0.75:
                            log(f"⚠ sun-patch agreement {hi:.0%} — the reference's bright "
                                f"directional light is not landing the same way in this "
                                f"match. The score above cannot see this; your eyes can.")
                    self._reality_check(result, ref_stats, log)
            except (MatchCancelled, PreflightBlocked):
                raise
            except Exception:  # noqa: BLE001 — never sink a match over the readout
                pass
            self._record_match(camera_name, result.best_state, result.best_score)
        if result.best_score is not None:
            # A preliminary card yields the authoritative critic content_gap + metric
            # coverage; those, with the director's leash_hits when the result surfaces them,
            # feed the READ-ONLY fairness estimate (D8) so it can only ever read
            # as-bad-or-worse than the director/critic, never softer. The reference is judged
            # against the accepted iteration's OWN render (best_render, already on file — no
            # new render fired); fairness.assess never raises and degrades to "unknown".
            prelim = critic.scorecard(
                result.best_score, result.best_components,
                ceiling_proven=result.ceiling_proven,
                ceiling_converged=result.ceiling_converged)
            refs = self.references(camera_name)
            best_cur_stats = (self.stats_for(result.best_render)
                              if result.best_render else None)
            fairness_card = fairness.assess(
                ref_stats, best_cur_stats,
                components=result.best_components,
                coverage=prelim.get("coverage"),
                n_references=len(refs),
                roles=[r["role"] for r in refs],
                ceiling_proven=result.ceiling_proven,
                content_gap=prelim.get("content_gap"),
                leash_hits=getattr(result, "leash_hits", None))
            result.scorecard = critic.scorecard(
                result.best_score, result.best_components,
                ceiling_proven=result.ceiling_proven,
                ceiling_converged=result.ceiling_converged,
                fairness=fairness_card)
            e.scorecard = dict(result.scorecard)
            card = result.scorecard
            weak = ", ".join(card["weakest"]) or "none measured"
            log(f"scorecard: {card['confidence']} confidence · weakest: {weak} · "
                f"metric coverage {card['coverage']:.0%}")
            if card["content_gap"]:
                log("diagnosis: remaining gap is likely scene content/material/albedo, "
                    "not a lighting control the optimizer can solve")
            # fairness LOGGED ALONGSIDE the content_gap diagnosis, never replacing it
            fair = card.get("fairness") or {}
            if fair.get("verdict") in ("marginal", "unfair"):
                remedy = fair.get("remedy") or ""
                log("fairness: reference constrains this scene "
                    + ("poorly" if fair.get("verdict") == "unfair" else "only partially")
                    + f" ({fair.get('verdict')})"
                    + (f" — {remedy}" if remedy else ""))
            log("scorecard warning: score measures tone/color/direction statistics; "
                "artist acceptance is separate")
        self._save_or_warn(log)
        try:                          # read-only rig census — diagnostic, never fails the run
            probes = self.rig_report()
        except Exception:  # noqa: BLE001 a probe must never abort a completed match
            probes = {}
        try:   # the calibration trail: every run leaves a machine-readable record
            with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
                json.dump({"camera": camera_name, "reference": e.reference,
                           "quality_profile": profile.name,
                           "semantics": semantics, "llm_down": bool(self._llm_down),
                           "probes": probes,
                           # 44 probes of evidence used to be dropped on the floor at the
                           # end of the sun solve; preflight's findings had nowhere to go
                           # at all. Both survive the run now. (2026-07-31)
                           "preflight": preflight_report.to_json(),
                           "sun_solve": solved,
                           "cost": self.cost_estimate(),
                           **result.to_summary()}, f, indent=1)
        except (OSError, TypeError, ValueError):
            pass
        self._final_report(result, log)
        return result

    # ------------------------------------------------------------------ I6: prove it
    def _reality_check(self, result, ref_stats: Optional[Dict],
                       log: Callable[[str], None]) -> None:
        """Say "this does not look like your reference" when three instruments agree.

        Every number here is ALREADY COMPUTED a few lines above and none of them was ever
        compared to a threshold — that, and not the metric, was the 2026-07-30 gap. Worked
        through by hand on the delivered frame (a cool sunless dusk courtyard against a
        warm golden-hour reference, EV solved) the critic returns roughly 35/100 with
        colour ~0.10 and highlight ~0.05. It was already capable of screaming.

        WEAKEST-LINK, matching the critic's own aggregation philosophy: any one of the
        three suffices. The absolute score term is deliberately the WEAKEST of the three
        at 45.0 — this repo's recorded legitimate scores go down to 63.2, and a single
        confident absolute number on the headline verdict is exactly the kind of false
        positive this whole exercise exists to kill. Colour and highlight carry the
        discrimination; the score floor only catches garbage.
        """
        reasons: List[str] = []
        comps = result.best_components or {}
        score = result.best_score
        if score is not None and score < UNLIKE_REFERENCE_SCORE:
            reasons.append(f"score {score:.1f}/100 — below every legitimate match ever "
                           f"measured on this box")
        colour = comps.get("color", comps.get("colour"))
        if colour is not None and float(colour) < UNLIKE_REFERENCE_COLOR:
            reasons.append(f"colour {float(colour):.2f} — the reference and this render "
                           "are not the same colour of light")
        hi = result.highlight
        ref_hot = float((ref_stats or {}).get("hot_frac") or 0.0)
        if hi is not None and float(hi) < UNLIKE_REFERENCE_HIGHLIGHT \
                and ref_hot > REF_HAS_SUN_HOT_FRAC:
            reasons.append(f"sun-patch agreement {float(hi):.0%} — the reference has "
                           "directional light and this render essentially has none")
        if not reasons:
            return
        result.unlike_reference = True
        result.unlike_reasons = reasons
        log("✗ THIS DOES NOT LOOK LIKE YOUR REFERENCE.")
        for r in reasons:
            log("  · " + r)
        transfer_score = (result.transfer or {}).get("score")
        if transfer_score is not None:
            log(f"  · lighting transfer {float(transfer_score):.0f}/100")
        log(f"  {len(reasons)} independent measure(s) agree. Do not deliver this frame.")

    def _final_report(self, result, log: Callable[[str], None]) -> None:
        """What actually IMPROVED, measured — not what was applied (I6).

        "✓ done (stalled) — best 71.8" says nothing about whether the run was worth its
        three hours. This states the start basin, the measured gain, what polish bought
        for its probes, and — the part that kept getting lost — every degradation and
        skipped stage, replayed from the ledger after 190 THUMB lines have scrolled the
        originals away.
        """
        score_txt = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
        log(f"match finished: {result.stop_reason}, best score {score_txt} "
            f"({len(result.iterations)} iterations)")
        scored = [r.score for r in result.iterations if r.score is not None]
        if scored and result.best_score is not None:
            log(f"  start {scored[0]:.1f} → final {result.best_score:.1f} "
                f"({result.best_score - scored[0]:+.1f} measured, same reference, "
                "same size)")
        if result.polish_probes:
            tail = ""
            if getattr(result, "ceiling_proven", False):
                tail = " · CEILING PROVEN (no fine move improves)"
            elif result.ceiling_converged:
                tail = " · plateau (finer steps untested)"
            per_probe = result.polish_gain / max(1, result.polish_probes)
            log(f"  polish: +{result.polish_gain:.2f} over {result.polish_probes} probes "
                f"({per_probe:.3f} points per probe){tail}")
            # polish_gain was computed and reported WITHOUT the 120 probes it bought. It is
            # the only measured-improvement number in the pipeline; quoting it without its
            # cost is how "the tool worked for three hours" stays unfalsifiable.
            est = self.cost_estimate()
            if result.polish_gain < 1.0 and est is not None:
                spent = result.polish_probes * est["seconds_each"] / 60.0
                log(f"  polish spent {_human_minutes(spent)} for less than a point — this "
                    "scene was already at its basin's ceiling when the loop handed it over")
        for d in result.degradations:
            log("  DEGRADED  " + d)
        if not result.degradations:
            log("  no degradations: every stage ran on its intended instrument")

    def _apply_only(self, camera_name: str, e, rig, cam, start: LightingState,
                    run_dir: str, semantics: Dict,
                    log: Callable[[str], None]) -> MatchResult:
        """no_renders mode: apply the state as ONE undoable change, verify by read-back,
        report every changed value — never render. 'Applied' is measured, not assumed."""
        for w in ap.apply_state(rig, self._baselines, start, cam):
            log("⚠ " + w)
        back = ap.read_state(rig, self._baselines, cam)
        drift = {k: v for k, v in back.diff(start).items() if abs(v[0] - v[1]) > 0.51}
        changes = (sorted(e.pre_match.diff(start).items())
                   if e.pre_match is not None else [])
        for key, (before, after) in changes:
            log(f"applied: {key} {before:.2f} → {after:.2f}")
        if not changes:
            log("applied: state matches the pre-match light — nothing to change")
        if drift:
            log("⚠ read-back drift on " + ", ".join(sorted(drift))
                + " — those parameters may not have stuck")
        else:
            log(f"verified by read-back: {len(changes)} setting(s) applied, "
                "0 renders fired")
        self._record_match(camera_name, start, None)
        self.save_session()
        result = MatchResult(best_state=start, best_score=None, best_render=None,
                             stop_reason="applied (no-render mode)")
        try:
            with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
                json.dump({"camera": camera_name, "reference": e.reference,
                           "semantics": semantics, "no_renders": True,
                           **result.to_summary()}, f, indent=1)
        except (OSError, TypeError, ValueError):
            pass
        log("match finished: applied (no-render mode)")
        return result

    def refine(
        self,
        camera_name: str,
        note: str,
        log: Callable[[str], None],
        should_cancel: Callable[[], bool] = lambda: False,
        locks: Optional[Set[str]] = None,
    ) -> MatchResult:
        """The conversation turn: a director's note → instant deterministic nudges →
        3-lens ENSEMBLE (competing corrections, each branch rendered and scored) → the
        winner continues into a deep match with the note pinned into every prompt.
        ``locks`` is the UI's CURRENT lock selection (run_match's convention); None
        falls back to the camera's persisted locks."""
        self._begin_operation(should_cancel)
        e = self.camera_entry(camera_name, create=True)
        if not e.reference:
            raise RuntimeError("bind a reference image to this camera first")
        if len(note) > MAX_NOTE_CHARS:
            log(f"note truncated to {MAX_NOTE_CHARS} chars ({len(note)} given) — notes "
                "are persisted and pinned into every prompt of the run")
            note = note[:MAX_NOTE_CHARS]
        e.notes = (getattr(e, "notes", []) + [note])[-6:]
        combined = " · ".join(e.notes[-3:])
        locks = set(locks if locks is not None else e.locks)
        rig = self.rig(refresh=True)
        cam = self.camera_node(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found")
        self._set_active_camera(camera_name)
        run_dir = self._new_run_dir(camera_name)
        self._llm_down = False
        semantics = self._analyze_or_fallback(camera_name, log)  # re-analyzes on ref swap
        ref_stats = self.ref_stats(e.reference)
        ref_block = self._image_block(e.reference)
        if ref_block is None and not getattr(self.cfg, "no_renders", False):
            if not os.path.exists(e.reference):
                raise RuntimeError(f"reference file not found: {e.reference} — "
                                   "re-bind it via Load reference…")
            raise RuntimeError("reference image could not be prepared for the LLM")

        # snapshot-first: capture the artist's true light BEFORE the note's nudges and
        # lens probes touch the scene, and tell run_match (via _plan_snapped) not to
        # re-snapshot a MaxGaffer-mutated state over it — Restore must return the
        # artist's light, never our own intermediate
        if e.pre_match is None:
            e.pre_match = ap.read_state(rig, self._baselines, cam)
        self._plan_snapped = camera_name
        self._save_or_warn(log)

        base = (e.state.copy() if e.state is not None
                else ap.read_state(rig, self._baselines, cam))
        deltas = feedback.nudges_from_note(note, base.keys(), list(base.groups))
        state0, applied = feedback.apply_note_deltas(base, deltas, locks)
        for k, v in applied.items():
            log(f"note → {k} = {v:.2f} (instant)")

        if getattr(self.cfg, "no_renders", False):
            if not applied:
                log("note matched no craft-table nudge — nothing to apply (the LLM "
                    "ensemble that would interpret it needs probe renders)")
            # fresh snapshot EVERY refine (run_match's convention): Restore must return
            # to the light before THIS note, and the change report must show only it —
            # a kept older snapshot would report the whole match+refine cumulatively
            e.pre_match = ap.read_state(rig, self._baselines, cam)
            return self._apply_only(camera_name, e, rig, cam, state0, run_dir,
                                    semantics, log)

        # every branch probe exposes against the same pre-refine anchor, so a
        # note-only EV/WB nudge is VISIBLE to the ensemble scoring (it wasn't when
        # probes rendered raw — the user's "darker" branch could silently lose)
        if e.pre_match is None:
            e.pre_match = ap.read_state(rig, self._baselines, cam)

        # the reference's own lighting reading, on the same terms the loop will use it
        probe_yaw = sc.camera_yaw_deg(cam)

        def _probe_transfer(st):
            try:
                return transfer.score(semantics, st, probe_yaw)["score"] / 100.0
            except Exception:  # noqa: BLE001
                return None

        probe_hooks = Hooks(apply=lambda s: None, render=lambda t: "",
                            stats=lambda p: None, llm_deltas=lambda c: "",
                            transfer=_probe_transfer if semantics else None)
        probe_cfg = MatchConfig(
            transfer_weight=TRANSFER_WEIGHT if semantics else 0.0)

        def probe(st: LightingState, tag: str):
            self._apply_logged(rig, st, cam, log)
            path = self._render_exposed(cam, os.path.join(run_dir, f"{tag}.png"),
                                        self.cfg.loop_width, self.cfg.loop_height,
                                        state=st, entry=e, log=log)
            stats = self.stats_for(path) if path else None
            if stats is None or ref_stats is None:
                return None, path
            value = critic.score(ref_stats, stats, self._critic_weights()).score
            # The winning branch becomes run_match's start_override, so this comparison
            # decides the whole refine round — and it must apply the same rule the loop and
            # polish will. On pixels alone it could crown a structurally wrong branch that
            # the loop then scores WORSE than the branch it rejected.
            return blend_transfer(value, st, probe_hooks, probe_cfg), path

        score0, path0 = probe(state0, "refine_note")
        log(f"branch note-only: {score0:.1f}" if score0 is not None
            else "branch note-only: unscored")
        branches = [("note-only", state0, score0)]
        render0_block = self._image_block(path0) if path0 else None
        from ..core.genome import apply_changes, rig_keys, state_table
        from ..core.parse import validate_deltas

        for lens_name, lens_line in feedback.LENSES:
            if should_cancel():
                break
            content = [ref_block]
            if render0_block is not None:
                content.append(render0_block)
            content.append(omega.text_block(prompts.deltas_user_text(
                state_table(state0, locks), semantics, [], {}, 0, 1,
                "; ".join(rig.get("notes", [])), "", combined)))
            try:
                reply = self.io(lambda: self._semantic_call(
                    feedback.lens_system(prompts.DELTAS_SYSTEM, lens_line),
                    [{"role": "user", "content": content}], 2048))
                proposal = validate_deltas(reply)
            except Exception as err:  # noqa: BLE001 a dead lens must not kill the round
                log(f"lens {lens_name}: unusable ({err})")
                continue
            cand, accepted, _rej = apply_changes(state0, proposal["changes"], locks,
                                                 limit=True, known=rig_keys(state0))
            if not accepted:
                log(f"lens {lens_name}: no valid changes")
                continue
            sc_val, _p = probe(cand, f"refine_{lens_name}")
            log(f"branch {lens_name}: "
                + (f"{sc_val:.1f}" if sc_val is not None else "unscored")
                + " · " + ", ".join(f"{k}→{v:.2f}" for k, v in accepted.items()))
            branches.append((lens_name, cand, sc_val))

        scored = [b for b in branches if b[2] is not None]
        winner = (max(scored, key=lambda b: b[2]) if scored else branches[0])
        log(f"ensemble winner: {winner[0]}"
            + (f" at {winner[2]:.1f}" if winner[2] is not None else ""))
        return self.run_match(camera_name, log, should_cancel, locks=locks,
                              do_sweep=False, deep=True, start_override=winner[1],
                              director_note=combined)

    def match_all(
        self,
        log: Callable[[str], None],
        should_cancel: Callable[[], bool] = lambda: False,
        do_sweep: bool = True,
    ) -> Dict[str, str]:
        """Unattended queue: match every camera that has a reference bound, sequentially.
        Per-camera failures are recorded and the queue continues; cancel stops between
        cameras (and mid-match via the shared flag)."""
        results: Dict[str, str] = {}
        queue = [name for name, e in self.session.cameras.items() if e.reference]
        if not queue:
            return {"": "no cameras have references bound"}
        # A 45-camera overnight queue must not stop on the first dialog and block until
        # morning, so the batch answers its own questions from each question's default and
        # LOGS every one of them. Restored in a finally: the artist's interactive policy is
        # theirs, not the batch's. (2026-07-31)
        prev_policy = getattr(self.cfg, "uncertainty_policy", "ask")
        self.cfg.uncertainty_policy = "assume"
        log("batch: questions will be answered from their defaults and logged — an "
            "unattended queue cannot wait for you")
        try:
            return self._match_all_queue(queue, log, should_cancel, do_sweep, results)
        finally:
            self.cfg.uncertainty_policy = prev_policy

    def _match_all_queue(self, queue, log, should_cancel, do_sweep,
                         results: Dict[str, str]) -> Dict[str, str]:
        if self.cfg.plan_first:
            log("note: scene-wide plans run on single MATCH only — the batch queue uses "
                "the match loop (plans need your preview)")
        for i, name in enumerate(queue):
            if should_cancel():
                results[name] = "cancelled"
                break
            log(f"— batch {i + 1}/{len(queue)}: {name} —")
            try:
                r = self.run_match(name, log, should_cancel,
                                   locks=None, do_sweep=do_sweep)
                if r.unlike_reference:
                    # a bare score here read as success. It was not. (2026-07-31)
                    results[name] = (f"FAILED — does not look like the reference "
                                     f"({r.best_score:.1f})"
                                     if r.best_score is not None else
                                     "FAILED — does not look like the reference")
                else:
                    results[name] = (f"{r.best_score:.1f}" if r.best_score is not None
                                     else r.stop_reason)
            except MatchCancelled:
                results[name] = "cancelled"
                log(f"— batch stopped at {name} —")
                break
            except Exception as err:  # noqa: BLE001 one bad camera must not kill the night
                results[name] = f"error: {err}"
                log(f"✗ {name}: {err}")
        return results

    # ------------------------------------------------------------------ scenario board
    def _pick_start_basin(self, first_guess, semantics, rig, cam, e, locks, hooks,
                          ref_stats, log):
        """Score the shipped scenario rigs against the first guess at SWEEP resolution and
        return whichever measures closest — the match's starting basin.

        Why: polish is a basin-finisher, so the loop's opening state caps the whole run.
        A single first guess is one sample of a multi-modal landscape; the board's rigs
        (golden-hour rake, overcast sky-key, backlit rim, cool north, practicals-at-dusk)
        are the modes worth sampling, and they already exist. Failures degrade to the
        first guess — a basin probe must never be able to end a match."""
        try:
            current = ap.read_state(rig, self._baselines, cam)
            board = scen.build_scenarios(semantics, current, sc.camera_yaw_deg(cam),
                                         set(locks or ()),
                                         overcast_sun_mode=self.cfg.overcast_sun_mode)
        except Exception as err:  # noqa: BLE001 — never let basin selection kill a match
            log(f"multi-start: candidates unavailable ({err}) — keeping the first guess")
            return first_guess
        # "as_analyzed" IS the first guess; probe it once under its own name
        cands = [("first_guess", first_guess)]
        cands += [(c["key"], c["state"]) for c in board
                  if c.get("key") != "as_analyzed" and c.get("state") is not None]
        if len(cands) < 2:
            return first_guess
        cfg_probe = MatchConfig(
            transfer_weight=TRANSFER_WEIGHT if hooks.transfer is not None else 0.0)
        best_key, best_state, best_score = "first_guess", first_guess, None
        scores: List[float] = []
        for key, st in cands:
            if hooks.should_cancel():
                break
            try:
                hooks.apply(st)
                path = hooks.render(f"sweep_basin_{key}")
                cur = self.stats_for(path) if path else None
                if cur is None:
                    # a silent `continue` turned a 6-candidate board into a 1-candidate
                    # board with no line saying so (2026-07-31)
                    log(f"multi-start: {key} produced no measurable plate — that basin "
                        "is NOT in the comparison")
                    continue
                value = critic.score(ref_stats, cur, self._critic_weights()).score
                # Judged on the SAME objective the loop will use. On pixels alone this
                # probe picked a SUNLESS basin for a sunlit golden-hour reference
                # (measured on-box 2026-07-25) — which then suppressed the sun lock
                # below, because that lock deliberately leaves a genuinely sunless basin
                # free. One weak comparison at the top of the run cost the whole match:
                # the loop finished at 80.35 with the sun switched off.
                value = blend_transfer(value, st, hooks, cfg_probe)
            except (MatchCancelled, PreflightBlocked):
                raise      # a validated stop is not "one bad candidate"
            except Exception as err:  # noqa: BLE001 one bad candidate ≠ a dead match
                log(f"multi-start: {key} probe failed ({err})")
                continue
            log(f"multi-start: {key} → {value:.1f}")
            scores.append(value)
            if best_score is None or value > best_score:
                best_key, best_state, best_score = key, st.copy(), value
        if best_score is not None:
            log(f"multi-start: best basin '{best_key}' at {best_score:.1f} "
                f"(of {len(cands)} probed) — the match starts here")
            if len(scores) >= 2:
                lead = best_score - max(s for s in scores if s != best_score) \
                    if len(set(scores)) > 1 else 0.0
                if lead < 1.0:
                    log(f"multi-start: the winner leads by only {lead:.2f} points — a "
                        "near-tie across the board is not a choice, it is the board "
                        "saying these rigs are indistinguishable here")
            # ---- GATE 3: THE BASIN FLOOR (I5). This is the gate that would have stopped
            # 2026-07-30 within about seven renders. The black frames scored 12.0, 10.9,
            # 8.7 and 2.7 and this function announced a "best basin" from them.
            if best_score < BASIN_FLOOR_SCORE:
                answer = self._escalate(askmod.Question(
                    key="basin_floor",
                    headline=(f"the best of {len(scores)} starting rigs scored "
                              f"{best_score:.1f}/100 against your reference"),
                    detail=("Nothing this tool can adjust gets from "
                            f"{best_score:.1f} to a match — every legitimate basin ever "
                            f"measured on this box scored above {BASIN_FLOOR_SCORE:.0f}, "
                            "and the 2026-07-30 run that ranked 100% black frames for "
                            "hours scored 2.7 to 12.0 exactly here. Something upstream is "
                            "wrong: the renderer, the reference, or which camera this is. "
                            "Continuing spends the whole budget."),
                    options=(("stop", "Stop — something upstream is wrong"),
                             ("continue", "Continue anyway")),
                    default="stop",
                    facts={"best_score": round(best_score, 2), "best_basin": best_key,
                           "probed": len(scores),
                           "floor": BASIN_FLOOR_SCORE}), log)
                if answer == "stop":
                    self._cancel_latch = (
                        f"stopped: the best starting rig scored {best_score:.1f}/100 — "
                        "that is not a basin, it is a broken input")
                    raise PreflightBlocked(self._cancel_latch)
                self._degrade(log, f"continuing from a {best_score:.1f} basin on your "
                                   "say-so — the final score will be measured, not "
                                   "assumed, and the verdict will say what it found")
        try:
            hooks.apply(best_state)     # leave the scene wearing the chosen basin
        except Exception:  # noqa: BLE001
            pass
        return best_state

    def run_scenarios(
        self,
        camera_name: str,
        log: Callable[[str], None],
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> List[Dict]:
        """Render the candidate rigs from core.scenarios at loop res, score each against
        the reference when one is bound, and leave the scene exactly as it was found.
        → [{key, label, why, state, render, score}] in board order."""
        self._begin_operation(should_cancel)
        rig = self.rig(refresh=True)
        cam = self.camera_node(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found in the scene")
        self._set_active_camera(camera_name)
        e = self.camera_entry(camera_name, create=True)
        semantics: Optional[Dict] = None
        if e.reference:
            try:
                semantics = self.analyze_reference(camera_name)
            except Exception as err:  # noqa: BLE001 the board must run reference-less too
                log(f"⚠ analyze failed ({err}) — board runs on the neutral base")
        current = ap.read_state(rig, self._baselines, cam)
        e.pre_match = current.copy()   # explorations must be restorable, same as matches
        self.save_session()
        board = scen.build_scenarios(semantics, current, sc.camera_yaw_deg(cam),
                                     set(e.locks),
                                     overcast_sun_mode=self.cfg.overcast_sun_mode)
        if not board:
            log("no scenario candidates for this rig")
            return []
        if getattr(self.cfg, "no_renders", False):
            log("no-render mode: candidates listed without probes or scores — pick by "
                "name; ADOPT applies it (the scene is untouched until you adopt)")
            return [{**cand, "render": None, "score": None} for cand in board]
        ref = self.ref_stats(e.reference) if e.reference else None
        if e.reference and ref is None:
            log("⚠ reference stats unavailable — board renders without scores")
        run_dir = self._new_run_dir(camera_name)
        results: List[Dict] = []
        try:
            for cand in board:
                if should_cancel():
                    log("scenario board cancelled")
                    break
                self._apply_logged(rig, cand["state"], cam, log)
                path = self._render_exposed(
                    cam, os.path.join(run_dir, f"scen_{cand['key']}.png"),
                    self.cfg.loop_width, self.cfg.loop_height,
                    state=cand["state"], entry=e, log=log)
                score = None
                if path:
                    log(f"THUMB::{path}")
                    if ref is not None:
                        cur = self.stats_for(path)
                        if cur is not None:
                            score = critic.score(ref, cur, self._critic_weights()).score
                log(f"scenario {cand['label']}: "
                    + (f"{score:.1f}" if score is not None else "unscored")
                    + f" — {cand['why']}")
                results.append({**cand, "render": path, "score": score})
        finally:
            # SPEC law: leave the scene exactly as it was found — even when an apply or
            # a render raises mid-board (the dock's "✗ board" must not hide a light swap)
            try:
                self._apply_logged(rig, current, cam, log)
            except Exception as err:  # noqa: BLE001 the restore must not mask the cause
                log(f"⚠ could not restore the found light after the board ({err}) — "
                    "use Restore to get it back")
        return results

    def adopt_scenario(self, camera_name: str, state: LightingState,
                       score: Optional[float] = None) -> List[str]:
        """Apply a board candidate and save it as the camera's state — MATCH/REFINE
        continue from it exactly like any matched state. Returns apply warnings."""
        warnings = self.apply_state(state, camera_name)
        self._record_match(camera_name, state, score)
        self.save_session()
        return warnings

    # ------------------------------------------------------------------ dome seed
    def seed_dome(self, camera_name: str, log: Callable[[str], None]) -> Dict:
        """Reference → world-oriented HDR pano → dome texture, rotation zeroed (the pano
        is world-aligned; dome.rotation_deg stays live for the loop to spin). The dome's
        previous texture/rotation is snapshotted once for Restore. → build_seed meta."""
        e = self.camera_entry(camera_name, create=True)
        if not e.reference:
            raise RuntimeError("bind a reference image first — the seed is built from it")
        rig = self.rig(refresh=True)
        dome = rig.get("dome")
        if dome is None:
            raise RuntimeError("no VRayLight dome in the rig — add one (or let a plan "
                               "create it), Read scene, then seed")
        cam = self.camera_node(camera_name)
        semantics = self.analyze_reference(camera_name)
        # sun placement: the camera's matched/current rig wins over the semantics guess
        st = e.state or ap.read_state(rig, self._baselines, cam)
        sun_az = sun_alt = None
        if "sun.azimuth_deg" in st.values and st.get("sun.enabled", 1.0) >= 0.5:
            sun_az = st.get("sun.azimuth_deg")
            sun_alt = st.get("sun.altitude_deg", 35.0)
        if not e.pre_seed:                     # snapshot once; Restore clears it
            e.pre_seed = {"file": sc.get_dome_texture(dome),
                          "rotation": sc.read_dome_rotation(dome)}
        # fingerprint the inputs into the FILENAME: Max caches bitmaps by path, so a
        # re-seed into the same filename renders the STALE pano (seed_filename's own
        # invariant). The pano is oriented around cam_yaw and carries a semantics-derived
        # sun disc when sun_az/sun_alt are None — so yaw AND the semantics signature are
        # fingerprint inputs, not just the reference and the sun numbers
        import hashlib

        yaw = sc.camera_yaw_deg(cam) if cam is not None else 0.0
        try:
            ref_sig = f"{e.reference}:{os.path.getmtime(e.reference):.0f}"
        except OSError:
            ref_sig = e.reference
        sem_sig = hashlib.md5(json.dumps(semantics, sort_keys=True, default=str)
                              .encode("utf-8", "replace")).hexdigest()[:8]
        token = hashlib.md5(
            f"{ref_sig}|{sun_az}|{sun_alt}|{yaw:.3f}|{sem_sig}"
            .encode("utf-8", "replace")).hexdigest()[:8]
        out = os.path.join(self._ensure_run_dir(_safe(camera_name)),
                           domeseed.seed_filename(camera_name, token))
        prev_seed = e.seed_hdri
        src = e.reference
        if _needs_max_ingest(src):             # EXR/HDR/TIFF ref: Max transcodes first
            src = self._transcode_ref(e.reference) or src
        parametric = sun_az is not None        # live VRaySun owns direction → no disc
        meta = domeseed.build_seed(out, ref_path=src, semantics=semantics,
                                   cam_yaw_deg=yaw, sun_az_deg=sun_az,
                                   sun_alt_deg=sun_alt,
                                   parametric_sun_active=parametric,
                                   blur_passes=self.cfg.seed_blur_passes)
        if meta is None and src == e.reference:   # plain loader failed (JPEG, no Pillow)
            png = self._transcode_ref(e.reference)
            if png:
                meta = domeseed.build_seed(out, ref_path=png, semantics=semantics,
                                           cam_yaw_deg=yaw, sun_az_deg=sun_az,
                                           sun_alt_deg=sun_alt,
                                           parametric_sun_active=parametric,
                                           blur_passes=self.cfg.seed_blur_passes)
        if meta is None:
            raise RuntimeError("could not read the reference for seeding")
        how = self.set_dome_hdri(out)
        if how == "failed":
            raise RuntimeError("dome texture could not be set (no writable file property "
                               "— checklist #16)")
        with self._dome_undo():
            rot_how = sc.write_dome_rotation(dome, 0.0)
        e.seed_hdri = out
        # a matched state's dome.rotation_deg was tuned against the REPLACED texture —
        # re-applying it on camera switch would spin the world-oriented seed off-axis
        # (dome sun disc lands rotation° away from the parametric sun)
        if e.state is not None and "dome.rotation_deg" in e.state.values \
                and abs(e.state.get("dome.rotation_deg")) > 1e-6:
            e.state.set("dome.rotation_deg", 0.0)
            log("dome seed: saved state's dome rotation reset to 0 (it was tuned "
                "against the replaced HDRI)")
        if prev_seed and prev_seed != out \
                and os.path.basename(prev_seed).startswith("seed_") \
                and not _seed_still_referenced(self.session, prev_seed, exclude=e):
            try:
                os.remove(prev_seed)           # superseded seed — don't litter the run dir
            except OSError:
                pass
        self.save_session()
        sun = meta.get("sun")
        log(f"dome seed: {os.path.basename(out)} ({meta['source']}, {how}, "
            f"{meta.get('reflection_quality', 'sharp')}, rotation zeroed via {rot_how})")
        if sun:
            log(f"dome seed: sun disc at az {sun['azimuth_deg']:.0f}° / "
                f"alt {sun['altitude_deg']:.0f}° ({sun['kelvin']:.0f}K)")
        elif meta.get("disc_policy") == "skipped_parametric_sun":
            log("dome seed: ambient-only — the live VRaySun owns the direct light "
                "(a baked disc would double the sun energy)")
        elif meta.get("overcast_lift"):
            log("dome seed: overcast — sky lifted, no disc")
        return meta

    # ------------------------------------------------------------------ presets / HDRI
    def save_preset(self, path: str, camera_name: str = "") -> bool:
        state = self.read_state(camera_name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(preset_dumps(state, name=os.path.basename(path), now=_stamp()))
            return True
        except OSError:
            return False

    def load_preset(self, path: str, camera_name: str = "") -> List[str]:
        """Apply a preset to the scene; if a camera is given, it becomes that camera's
        saved state too. Raises on an unreadable/invalid file."""
        with open(path, "r", encoding="utf-8") as f:
            state = preset_loads(f.read())
        if state is None:
            raise RuntimeError(f"not a MaxGaffer preset: {path}")
        warnings = self.apply_state(state, camera_name)
        if camera_name:
            self._record_match(camera_name, state, None)
            self.save_session()
        return warnings

    def set_dome_hdri(self, hdri_path: str, role_index: int = 0) -> str:
        """Bind an HDRI to a role-sorted dome (0=primary, 1=reflection/fill, …)."""
        rig = self.rig()
        domes = list(rig.get("domes") or [])
        if not domes and rig.get("dome") is not None:
            domes = [rig["dome"]]
        try:
            dome = domes[int(role_index)]
        except (IndexError, TypeError, ValueError):
            dome = None
        if dome is None:
            return "failed"
        with self._dome_undo():
            how = sc.set_dome_texture(dome, hdri_path)
        try:
            import pymxs

            pymxs.runtime.redrawViews()
        except Exception:
            pass
        return how

    def _sweep_call(self, ref_block: dict, paths: List[str], azimuths: List[float]) -> str:
        content: List[dict] = [ref_block]
        for p in paths:
            block = self._image_block(p)
            if block is not None:
                content.append(block)
        content.append(omega.text_block(prompts.sweep_user_text(azimuths)))
        try:
            return self.io(lambda: self._semantic_call(
                prompts.SWEEP_SYSTEM,
                [{"role": "user", "content": content}], 1024))
        except (omega.OmegaError, RuntimeError):
            self._llm_down = True
            return ""      # run_sun_sweep falls back to the contrastive-metric winner

    def _apply_logged(self, rig, state: LightingState, cam, log) -> None:
        """Loop/probe/board applies — UNDO-FREE by design: a deep match plus polish is
        130+ applies, and recording each would flood the artist's undo stack out of
        existence (their own work becomes unreachable). The official revert for
        explorations is the pre-match snapshot + Restore, not Ctrl+Z. Manual paths
        (sliders, presets, adopt, camera switch) keep normal undo via apply_state."""
        for w in ap.apply_state(rig, self._baselines, state, cam, undo=False):
            log("⚠ " + w)

    @staticmethod
    def _dome_undo():
        """Dome texture/rotation writes get their own undo record, mirroring
        apply_state's 'one undo record per apply' (apply.py) — seed binds, manual HDRI
        picks and Restore's pre-seed branch must all be Ctrl+Z-able. Degrades to a
        no-op context off-Max (restore paths are unit-tested against mocked scene IO)."""
        try:
            import pymxs

            return pymxs.undo(True, "MaxGaffer dome")
        except ImportError:
            import contextlib

            return contextlib.nullcontext()

    # ------------------------------------------------------------------ restore
    def start_fresh(self, log: Callable[[str], None] = lambda _m: None) -> Dict:
        """Undo everything MaxGaffer did and forget everything it learned. → a summary.

        Two halves, and both are needed or "fresh" is a lie. The SCENE half puts each
        camera's light back to the snapshot taken before its first match, removes the
        exposure control we auto-created, and undoes a dome seed — that is
        restore_pre_match, per camera. The SESSION half then drops the bound references,
        the cached ANALYZE readings, the notes, the locks, the scorecards and the match
        history, and writes the emptied sidecar.

        Restoring is attempted for EVERY camera before anything is cleared: a camera whose
        restore fails must not have its snapshot thrown away, because that snapshot is the
        only record of the artist's original light. Failures are reported and the entry is
        kept, so a second attempt is possible. Everything else is cleared regardless.

        Baselines are deliberately KEPT. They are a reading of the artist's own authored
        light multipliers, not something MaxGaffer authored, and re-deriving them is the
        one thing this cannot do from inside a reset."""
        summary = {"restored": [], "nothing_to_restore": [], "failed": [], "cleared": 0}
        for name in list(self.session.cameras.keys()):
            entry = self.session.cameras.get(name)
            label = getattr(entry, "camera_name", "") or name
            try:
                if self.restore_pre_match(label, log=log):
                    summary["restored"].append(label)
                else:
                    summary["nothing_to_restore"].append(label)
            except Exception as err:  # noqa: BLE001 — one bad camera must not strand the rest
                summary["failed"].append((label, str(err)))
                log(f"⚠ could not restore {label}: {err} — its snapshot is being KEPT")
        keep = {n: e for n, e in self.session.cameras.items()
                if any(lbl == (getattr(e, "camera_name", "") or n)
                       for lbl, _why in summary["failed"])}
        summary["cleared"] = len(self.session.cameras) - len(keep)
        self.session.cameras = keep
        self._baselines = dict(self.session.baselines)   # baselines survive, see docstring
        self.save_session()
        log("reset: %d camera(s) restored, %d cleared%s"
            % (len(summary["restored"]), summary["cleared"],
               ", %d KEPT after a failed restore" % len(summary["failed"])
               if summary["failed"] else ""))
        return summary

    def restore_pre_match(self, camera_name: str,
                          log: Callable[[str], None] = lambda _m: None) -> bool:
        """Restore what MaxGaffer changed: the pre-match light (when one was snapped),
        the pre-seed dome texture/rotation (when a seed replaced them — reachable even
        with NO pre_match, so a seed-only session is restorable too), and the exposure
        control WE auto-created. Returns False when there is nothing to restore."""
        e = self.camera_entry(camera_name)
        if e is None or (e.pre_match is None and not e.pre_seed):
            return False
        if e.pre_match is not None:
            self.apply_state(e.pre_match, camera_name)
        if getattr(e, "ec_created", False):
            # WE auto-created the scene's exposure control for the match — Restore must
            # exit the experiment entirely, not leave our EC in the artist's scene
            try:
                import pymxs

                with pymxs.undo(True, "MaxGaffer exposure control"):
                    pymxs.runtime.SceneExposureControl.exposureControl = \
                        pymxs.runtime.undefined
                log("removed the auto-created V-Ray exposure control")
            except Exception as err:  # noqa: BLE001 cleanup must not fail the restore
                log(f"⚠ could not remove the auto-created exposure control ({err})")
            e.ec_created = False
        if e.pre_seed:                         # a seed replaced the dome texture — undo it
            dome = self.rig().get("dome")
            if dome is not None:
                prev = str(e.pre_seed.get("file") or "")
                with self._dome_undo():
                    if prev:
                        sc.set_dome_texture(dome, prev)
                    else:                      # dome had no HDRI: disable texture use
                        sc.set_prop(dome, sc.DOME_TEX_ON, False)
                    try:
                        sc.write_dome_rotation(dome,
                                               float(e.pre_seed.get("rotation") or 0.0))
                    except (TypeError, ValueError):
                        pass
            e.pre_seed = {}
            e.seed_hdri = ""
        self.save_session()
        return True

    # ------------------------------------------------------------------ vantage
    def start_live_link(self) -> Tuple[bool, str]:
        return vt.start_live_link()

    def prepare_vantage_jobs(
        self,
        camera_names: List[str],
        out_dir: str,
        on_progress: Callable[[str, str], None],
        use_saved_states: bool = True,
    ) -> List[Dict]:
        """MAIN-THREAD half: per camera, apply its saved lighting state and export the
        .vrscene. Raises on export failure (nothing has rendered yet — cheap to abort).
        The scene's found light (and dome texture) is restored in a finally — a
        snapshot-first tool never strands the LAST camera's light in the artist's scene.

        vrscene exports are the heavyweight artifacts (100s of MB on real interiors), so
        old export batches are pruned by the same keep_runs policy as loop renders."""
        vantage_parent = self._ensure_run_dir("vantage")
        export_dir = os.path.join(vantage_parent, _stamp())
        jobs: List[Dict] = []
        rig = self.rig()
        dome = rig.get("dome")
        found = found_tex = None
        if use_saved_states:
            found = ap.read_state(rig, self._baselines, None)
            found_tex = sc.get_dome_texture(dome) if dome is not None else None
        try:
            for name in camera_names:
                on_progress(name, "applying + exporting")
                if use_saved_states:
                    e = self.camera_entry(name)
                    if e and e.state is not None:
                        self.apply_state(e.state, name)
                    self._rebind_seed(name)    # each vrscene must carry ITS camera's sky
                scene_file = vt.export_vrscene(
                    os.path.join(export_dir, f"{_safe(name)}.vrscene"), name)
                if scene_file is None:
                    raise RuntimeError(f"{name}: vrscene export failed "
                                       "(vrayExportVRScene missing or camera not set)")
                jobs.append({"camera": name, "scene_file": scene_file,
                             "output": os.path.join(out_dir, f"{_safe(name)}.png")})
        finally:
            if found is not None:
                try:
                    self._apply_logged(rig, found, None, lambda _m: None)
                    if dome is not None and found_tex \
                            and sc.get_dome_texture(dome) != found_tex:
                        with self._dome_undo():
                            sc.set_dome_texture(dome, found_tex)
                except Exception:  # noqa: BLE001 the export result stands regardless
                    on_progress("", "⚠ scene light could not be restored after export")
        prune_old_runs(vantage_parent, keep=int(self.cfg.keep_runs))
        return jobs

    def run_vantage_jobs(self, jobs: List[Dict],
                         on_progress: Callable[[str, str], None],
                         should_cancel: Optional[Callable[[], bool]] = None
                         ) -> Dict[str, str]:
        """LEGACY (Developer-Edition CLI only) — pure subprocess, worker-thread safe."""
        if getattr(self.cfg, "no_renders", False):
            for job in jobs:
                on_progress(job.get("camera", ""),
                            "skipped — no-render mode is ON (Settings)")
            return {job.get("camera", ""): "skipped (no-render mode)" for job in jobs}
        return vt.render_stills(jobs, self.cfg.vantage_console,
                                self.cfg.final_width, self.cfg.final_height, on_progress,
                                should_cancel=should_cancel)

    def render_finals_vray(
        self,
        camera_names: List[str],
        out_dir: str,
        on_progress: Callable[[str, str], None],
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, str]:
        """DEFAULT final-render backend (stock Vantage 3.x has no headless CLI): per
        camera, apply its saved state and production-render through V-Ray at final res.
        MAIN THREAD — renders block Max by nature; progress narrates between shots.
        ``should_cancel`` is checked between cameras (remaining ones record
        "cancelled"), and one camera's failure never aborts the batch. The scene's
        found light (and dome texture) is restored in a finally — finals must not
        strand the LAST camera's light in the artist's scene."""
        self._begin_operation(should_cancel)
        if getattr(self.cfg, "no_renders", False):
            for name in camera_names:
                on_progress(name, "skipped — no-render mode is ON (Settings)")
            return {name: "skipped (no-render mode)" for name in camera_names}
        results: Dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)
        rig = self.rig()
        dome = rig.get("dome")
        found = ap.read_state(rig, self._baselines, None)
        found_tex = sc.get_dome_texture(dome) if dome is not None else None
        try:
            for i, name in enumerate(camera_names):
                if should_cancel is not None and should_cancel():
                    for rest in camera_names[i:]:
                        results[rest] = "cancelled"
                        on_progress(rest, "cancelled")
                    break
                try:
                    on_progress(name, "applying state")
                    e = self.camera_entry(name)
                    if e and e.state is not None:
                        self.apply_state(e.state, name)
                    self._rebind_seed(name)    # finals render under their own seed too
                    cam = sc.get_camera(name)
                    if cam is None:
                        results[name] = "camera not found"
                        on_progress(name, results[name])
                        continue
                    on_progress(name, f"rendering {self.cfg.final_width}×"
                                      f"{self.cfg.final_height} (V-Ray)")
                    # finals go through the SAME exposed path as the loop, anchored at
                    # this camera's pre_match — the delivered PNG carries the exposure
                    # the accepted match iteration showed, even on display-stage-exposure
                    # renderers
                    out = self._render_exposed(
                        cam, os.path.join(out_dir, f"{_safe(name)}.png"),
                        self.cfg.final_width, self.cfg.final_height,
                        state=(e.state if e else None), entry=e)
                    results[name] = "ok" if out else "render failed"
                except Exception as err:  # noqa: BLE001 one bad camera must not kill the night
                    results[name] = f"error: {err}"
                on_progress(name, results[name])
        finally:
            try:
                self._apply_logged(rig, found, None, lambda _m: None)
                if dome is not None and found_tex \
                        and sc.get_dome_texture(dome) != found_tex:
                    with self._dome_undo():
                        sc.set_dome_texture(dome, found_tex)
            except Exception:  # noqa: BLE001 the renders stand regardless
                on_progress("", "⚠ scene light could not be restored after finals")
        return results

    def export_and_open_vantage(
        self,
        camera_names: List[str],
        on_progress: Callable[[str, str], None],
    ) -> Tuple[List[Dict], bool, str]:
        """Vantage-quality path on stock 3.3: export per-camera vrscenes (each with its
        matched light applied) and open Vantage — drop them into its in-app Batch Render
        queue. Returns (jobs, vantage_launched, export_dir)."""
        jobs = self.prepare_vantage_jobs(camera_names, cfgmod.sessions_dir(), on_progress,
                                         use_saved_states=True)
        export_dir = os.path.dirname(jobs[0]["scene_file"]) if jobs else ""
        manifest = vt.write_queue_manifest(export_dir, jobs,
                                           self.cfg.final_width, self.cfg.final_height)
        if manifest:
            on_progress("", f"queue manifest ready: {manifest}")
        launched = vt.launch_vantage(self.cfg.vantage_exe,
                                     jobs[0]["scene_file"] if jobs else None)
        return jobs, launched, export_dir

    # ------------------------------------------------------------------ dirs
    def _ensure_run_dir(self, sub: str) -> str:
        stem = _safe(os.path.splitext(os.path.basename(sc.scene_path() or "unsaved"))[0])
        try:
            d = os.path.join(cfgmod.sessions_dir(), stem, sub)
            os.makedirs(d, exist_ok=True)
        except OSError as err:
            # disk full / unwritable %LOCALAPPDATA% — fall back to the temp dir and say
            # WHY (a bare "✗ unexpected" from the dock's catch-all tells nothing)
            import tempfile

            d = os.path.join(tempfile.gettempdir(), "MaxGaffer_sessions", stem, sub)
            os.makedirs(d, exist_ok=True)
            # print() goes to Max's Listener, which is not where the artist reads. This
            # fallback silently moves every run artefact to a different disk. (2026-07-31)
            self._config_warn(f"the sessions directory is unavailable ({err}) — every run "
                              f"file for this session is going to {d} instead")
        if sub == "refs":
            # the transcode cache is FILES (ref_*.png / llm_*.png), not run dirs —
            # prune it under the same keep policy or it grows without bound
            prune_old_files(d, keep=int(self.cfg.keep_runs))
        return d

    def _new_run_dir(self, camera_name: str) -> str:
        parent = self._ensure_run_dir(_safe(camera_name))
        d = os.path.join(parent, _stamp())
        os.makedirs(d, exist_ok=True)
        self._run_dir = d
        prune_old_runs(parent, keep=int(self.cfg.keep_runs))
        return d


def _pillow_available() -> bool:
    """Is the thing software exposure actually needs installed? Probed at ARM TIME.

    ``expose.expose_image_file`` returns None without Pillow, and the caller only warned
    ONCE — so on a V-Ray GPU scene without Pillow every frame after the first was scored
    un-exposed while the solver kept prescribing exposure changes that could not reach the
    pixels. Better to know before the loop starts."""
    try:
        import PIL  # noqa: F401 — presence probe only
        return True
    except Exception:  # noqa: BLE001
        return False


def _human_minutes(minutes: float) -> str:
    """"3 h 12 min" rather than "192.4 minutes" — the artist is deciding whether to wait."""
    minutes = max(0.0, float(minutes))
    if minutes < 1.0:
        return f"{minutes * 60.0:.0f} s"
    if minutes < 60.0:
        return f"{minutes:.0f} min"
    return f"{int(minutes // 60)} h {int(round(minutes % 60)):02d} min"


def _seed_still_referenced(session, path: str, exclude=None) -> bool:
    """True if any OTHER camera's seed or pre-seed snapshot points at ``path``. Two
    cameras share one scene-global dome, so B's pre_seed can legitimately be A's seed
    file — deleting it on A's re-seed would turn B's Restore into a black dome."""
    for entry in session.cameras.values():
        if entry is exclude:
            continue
        if entry.seed_hdri == path or (entry.pre_seed or {}).get("file") == path:
            return True
    return False


def prune_old_runs(parent_dir: str, keep: int) -> int:
    """Delete the oldest run folders beyond ``keep`` (timestamp names sort chronologically).
    keep <= 0 disables pruning. Returns how many were removed."""
    if keep <= 0:
        return 0
    try:
        dirs = sorted(d for d in os.listdir(parent_dir)
                      if os.path.isdir(os.path.join(parent_dir, d)))
    except OSError:
        return 0
    removed = 0
    for d in dirs[:-keep]:
        import shutil

        shutil.rmtree(os.path.join(parent_dir, d), ignore_errors=True)
        removed += 1
    return removed


def prune_old_files(parent_dir: str, keep: int) -> int:
    """Delete the oldest FILES beyond ``keep`` (mtime order) — the refs/ transcode cache
    grows unbounded otherwise. keep <= 0 disables pruning. Mirrors prune_old_runs."""
    if keep <= 0:
        return 0
    try:
        files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)
                 if os.path.isfile(os.path.join(parent_dir, f))]
        files.sort(key=lambda p: os.path.getmtime(p))
    except OSError:
        return 0
    removed = 0
    for p in files[:-keep]:
        try:
            os.remove(p)
            removed += 1
        except OSError:
            pass
    return removed


# Win32 reserved device names: a camera/scene named "CON", "NUL", "COM1"… makes
# makedirs fail with a confusing OSError, and trailing dots/spaces are stripped by the
# filesystem (path then collides or misbehaves) — both are normalized away here
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)})


def _safe(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if s != name:
        # char sanitization rewrote the name — "Cam/A" and "Cam\A" both became "Cam_A"
        # (colliding run dirs/files); a short stable hash of the RAW name disambiguates.
        # The dot-strip and reserved-name prefix below are NOT hashed — the filesystem
        # itself collapses "trail." onto "trail", and "_CON" is already unique.
        import hashlib

        s += "_" + hashlib.md5(name.encode("utf-8", "replace")).hexdigest()[:6]
    s = s.rstrip(". ")
    if not s:
        return "unnamed"
    if s.split(".", 1)[0].lower() in _RESERVED_NAMES:
        s = "_" + s
    return s


def _reference_token(path: str) -> str:
    """Short cache-buster for transcoded references.

    Max and Qt both cache decoded bitmaps by filename.  Including the source identity in
    the output path prevents a replaced image (or two folders containing ``ref.exr``)
    from displaying or uploading an older transcode.
    """
    import hashlib

    return hashlib.md5(reference_signature(path).encode("utf-8", "replace")).hexdigest()[:10]


def _stamp() -> str:
    """Run-dir timestamp: sorts chronologically and has SUB-SECOND resolution — two runs
    on the same camera inside one second used to share a dir and overwrite run.json."""
    now = time.time()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + \
        f"-{int(now * 1000) % 1000:03d}"
