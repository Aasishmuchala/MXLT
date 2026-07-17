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
from typing import Callable, Dict, List, Optional, Tuple

from ..core import (consensus, critic, domeseed, feedback, metrics, omega, planner,
                    prompts, rules, scenedigest, scenarios as scen)
from ..core.director import Hooks, MatchConfig, MatchResult, run_match, run_sun_sweep
from ..core.genome import LightingState
from ..core.parse import ParseError, validate_analysis
from ..core.session import Session, preset_dumps, preset_loads, sidecar_path
from . import apply as ap
from . import config as cfgmod
from . import digest as dg
from . import draft as df
from . import execute as ex
from . import render as rd
from . import scene as sc
from . import vantage as vt

# formats Max reads natively but Pillow/stdlib can't — always ingest via Max transcode
MAX_FIRST_EXTS = (".exr", ".hdr", ".tif", ".tiff")


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
        # pure-I/O runner — the UI swaps in a worker-thread pump so gateway waits never
        # freeze Max; pymxs is NEVER called through this (network/subprocess only)
        self.io: Callable = lambda fn: fn()

    # ------------------------------------------------------------------ scene / session
    @property
    def session(self) -> Session:
        scene = sc.scene_path()
        if self._session is None or scene != self._session_scene:
            self._session = Session.load(sidecar_path(scene))
            self._session_scene = scene
        return self._session

    def save_session(self) -> bool:
        self.session.path = sidecar_path(sc.scene_path())   # scene may have been saved-as
        return self.session.save()

    def rig(self, refresh: bool = False):
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
        for c in cams:
            e = self.session.cameras.get(c["name"])
            c["reference"] = e.reference if e else ""
            c["score"] = e.score if e else None
            c["has_state"] = bool(e and e.state)
        return cams

    def read_state(self, camera_name: str = "") -> LightingState:
        cam = sc.get_camera(camera_name) if camera_name else None
        return ap.read_state(self.rig(), self._baselines, cam)

    def apply_state(self, state: LightingState, camera_name: str = "") -> List[str]:
        cam = sc.get_camera(camera_name) if camera_name else None
        return ap.apply_state(self.rig(), self._baselines, state, cam)

    def select_camera(self, camera_name: str, apply_saved: bool = True) -> List[str]:
        sc.set_active_camera(camera_name)
        warnings: List[str] = []
        if apply_saved and self.session.settings.get("apply_on_select", True):
            e = self.session.cameras.get(camera_name)
            if e and e.state is not None:
                warnings = self.apply_state(e.state, camera_name)
            self._rebind_seed(camera_name)     # the dome is scene-global; seeds are not
        return warnings

    def _rebind_seed(self, camera_name: str) -> None:
        """Per-camera dome seeds live on ONE scene-global dome — whenever a camera's
        light is applied for viewing or rendering, its own seed must come back, or shot A
        silently renders under shot B's sky. Never touches rotation: the applied state
        owns dome.rotation_deg (only the initial seed bind zeroes it)."""
        e = self.session.cameras.get(camera_name)
        if not (e and e.seed_hdri) or not os.path.exists(e.seed_hdri):
            return
        dome = self.rig().get("dome")
        if dome is None:
            return
        if sc.get_dome_texture(dome) != e.seed_hdri:
            sc.set_dome_texture(dome, e.seed_hdri)

    # ------------------------------------------------------------------ stats providers
    def stats_for(self, path: str) -> Optional[Dict]:
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
            proc = subprocess.run(
                [py, "-m", "maxgaffer.sidecar.metrics_cli", path],
                capture_output=True, text=True, timeout=60, cwd=repo)
            data = json.loads(proc.stdout or "null")
            if isinstance(data, list) and data and isinstance(data[0].get("stats"), dict):
                return data[0]["stats"]
        except Exception:
            pass
        return None

    def ref_stats(self, ref_path: str) -> Optional[Dict]:
        try:
            key = f"{ref_path}:{os.path.getmtime(ref_path)}"
        except OSError:
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
        png = os.path.join(self._ensure_run_dir("refs"),
                           "ref_" + _safe(os.path.basename(ref_path)) + ".png")
        return rd.transcode_to_png(ref_path, png)

    # ------------------------------------------------------------------ LLM plumbing
    def _image_block(self, path: str) -> Optional[dict]:
        """Payload-slim image block: Pillow in-process → sidecar --b64 → raw file (small
        renders) → Max transcode to PNG. EXR/HDR/TIFF skip straight to Max transcode."""
        if _needs_max_ingest(path):
            png = os.path.join(self._ensure_run_dir("refs"),
                               "llm_" + _safe(os.path.basename(path)) + ".png")
            if rd.transcode_to_png(path, png, max_dim=768):
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
                proc = subprocess.run(
                    [py, "-m", "maxgaffer.sidecar.metrics_cli", path, "--b64"],
                    capture_output=True, text=True, timeout=60, cwd=repo)
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
                           "llm_" + os.path.basename(path) + ".png")
        if rd.transcode_to_png(path, png, max_dim=768):
            return omega.image_block_from_file(png)
        return None

    def analyze_reference(self, camera_name: str) -> Dict:
        """ANALYZE call (cached in the session until the reference changes)."""
        e = self.session.entry(camera_name)
        if not e.reference:
            raise RuntimeError("no reference image bound to this camera")
        if e.semantics:
            return e.semantics
        block = self._image_block(e.reference)
        if block is None:
            raise RuntimeError(f"could not read reference image: {e.reference}")
        messages = [{"role": "user",
                     "content": [block, omega.text_block(prompts.analyze_user_text())]}]
        # self-consistency: N independent reads, consolidated (majority/median/circular
        # mean) — live evidence showed single samples reading the same ref as golden hour
        # OR midday, and a wrong read poisons the whole run
        samples = []
        n = max(1, int(self.cfg.analyze_samples))
        last_reply = ""
        for _ in range(n):
            last_reply = self.io(lambda: omega.call(
                self.cfg.api_key, prompts.ANALYZE_SYSTEM, messages,
                model=self.cfg.model, max_tokens=2048))
            try:
                samples.append(validate_analysis(last_reply))
            except ParseError:
                continue
        if not samples:   # every sample was junk — one strict retry, then give up loudly
            retry = messages + [
                {"role": "assistant", "content": last_reply[:1500]},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the "
                                            "JSON object, nothing else."}]
            samples.append(validate_analysis(self.io(lambda: omega.call(
                self.cfg.api_key, prompts.ANALYZE_SYSTEM, retry,
                model=self.cfg.model, max_tokens=2048))))
        semantics = consensus.consolidate_analyses(samples)
        agreement = semantics.pop("consensus_agreement", 1.0)
        if len(samples) > 1 and agreement < 1.0:
            # keep it out of the cached dict but tell the human the read was contested
            self._last_analyze_agreement = agreement
        e.semantics = semantics
        self.save_session()
        return semantics

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
                return self.io(lambda: omega.call(
                    self.cfg.api_key, prompts.DELTAS_SYSTEM,
                    [{"role": "user", "content": content}],
                    model=self.cfg.model, max_tokens=2048))
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
            return self.analyze_reference(camera_name)
        except (omega.OmegaError, RuntimeError) as err:
            self._llm_down = True
            e = self.session.cameras.get(camera_name)
            if e is not None and e.semantics:
                log(f"⚠ gateway unavailable ({err}) — using this camera's cached analysis")
                return e.semantics
            log(f"⚠ gateway unavailable ({err}) — ANALYTIC-ONLY run on neutral base "
                "semantics; steer with BOARD, locks and notes")
            return dict(scen.DEFAULT_SEMANTICS)

    # ------------------------------------------------------------------ scene-wide plan
    def make_plan(self, camera_name: str, log: Callable[[str], None]):
        """READ (full digest) → UNDERSTAND (LLM sees ref + every current setting) →
        PLAN (validated, digest-grounded ops). Returns (ops, lines, meta, raw_digest)."""
        e = self.session.entry(camera_name)
        if not e.reference:
            raise RuntimeError("bind a reference image to this camera first")
        # snapshot the genome part BEFORE the plan touches anything — Restore pre-match
        # must return to the true starting light (the plan itself is one Ctrl+Z)
        e.pre_match = ap.read_state(self.rig(refresh=True), self._baselines,
                                    sc.get_camera(camera_name))
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
        reply = self.io(lambda: omega.call(self.cfg.api_key, planner.PLAN_SYSTEM, messages,
                                           model=self.cfg.model, max_tokens=4096))
        try:
            ops, rejected, meta = planner.validate_plan(reply, cat)
        except ParseError:
            retry = messages + [
                {"role": "assistant", "content": reply[:1500]},
                {"role": "user", "content": "That was not valid JSON. Reply with ONLY the "
                                            "JSON object, nothing else."}]
            ops, rejected, meta = planner.validate_plan(self.io(lambda: omega.call(
                self.cfg.api_key, planner.PLAN_SYSTEM, retry,
                model=self.cfg.model, max_tokens=4096)), cat)
        if meta.get("read"):
            log("scene read: " + meta["read"])
        for r in rejected:
            log("plan rejected: " + r)
        return ops, planner.describe_plan(ops), meta, raw

    def probe_score(self, camera_name: str, tag: str) -> Optional[float]:
        """One loop-res render of the current scene scored against the camera's reference
        — the cheap 'did that help?' measurement."""
        if getattr(self.cfg, "no_renders", False):
            return None                    # no-render mode: plans apply unmeasured
        e = self.session.cameras.get(camera_name)
        if not (e and e.reference):
            return None
        ref = self.ref_stats(e.reference)
        cam = sc.get_camera(camera_name)
        if ref is None or cam is None:
            return None
        path = rd.render_frame(
            cam, os.path.join(self._ensure_run_dir(_safe(camera_name)), f"probe_{tag}.png"),
            self.cfg.loop_width, self.cfg.loop_height)
        cur = self.stats_for(path) if path else None
        if cur is None:
            return None
        return critic.score(ref, cur, self.cfg.critic_weights or None).score

    def execute_plan(self, ops, camera_name: str,
                     log: Callable[[str], None], measure: bool = True) -> Dict:
        """Execute a validated plan (one undo record; MG_ layer for created lights) and
        return the before/after change report — including the plan's MEASURED effect
        (critic score before vs after, one probe render each side). Rig re-classified
        after so new lights join the dimmer boards immediately."""
        before_score = self.probe_score(camera_name, "preplan") if measure else None
        cam = sc.get_camera(camera_name)
        report = ex.execute_plan(ops, cam)
        for c in report["changes"]:
            log(f"  {c['target']} · {c['prop']}: {c['before']} → {c['after']}")
        for c in report["created"]:
            log(f"  + {c['type']} '{c['name']}' at {c['at']}")
        for w in report["warnings"]:
            log("  ⚠ " + w)
        self.rig(refresh=True)
        if measure and before_score is not None:
            after_score = self.probe_score(camera_name, "postplan")
            if after_score is not None:
                report["effect"] = {"before": before_score, "after": after_score}
                log(f"plan effect: critic {before_score:.1f} → {after_score:.1f}"
                    + ("  ⚠ the plan made the match WORSE — one Ctrl+Z reverts it"
                       if after_score < before_score - 5.0 else ""))
        return report

    def state_change_rows(self, camera_name: str) -> List[Dict]:
        """pre-match → current saved state, as popup rows (the classic loop's report)."""
        e = self.session.cameras.get(camera_name)
        if not (e and e.pre_match is not None and e.state is not None):
            return []
        rows = []
        for key, (before, after) in sorted(e.pre_match.diff(e.state).items()):
            rows.append({"target": camera_name, "prop": key,
                         "before": round(before, 2), "after": round(after, 2), "why": ""})
        return rows

    # ------------------------------------------------------------------ the headline act
    def run_match(
        self,
        camera_name: str,
        log: Callable[[str], None],
        should_cancel: Callable[[], bool] = lambda: False,
        locks: Optional[set] = None,
        do_sweep: bool = False,
        deep: bool = False,
        start_override: Optional[LightingState] = None,
        director_note: str = "",
    ) -> MatchResult:
        e = self.session.entry(camera_name)
        if not e.reference:
            raise RuntimeError("bind a reference image to this camera first")
        rig = self.rig(refresh=True)
        cam = sc.get_camera(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found in the scene")
        sc.set_active_camera(camera_name)
        if self.cfg.auto_exposure_control:
            from .exposure import ExposureHost, ensure_exposure_control

            if ExposureHost(cam).kind == "none":
                created = ensure_exposure_control()
                if created:
                    log("⚠ " + created)
        locks = set(locks if locks is not None else e.locks)
        run_dir = self._new_run_dir(camera_name)
        log(f"run dir: {run_dir}")

        # snapshot the light as it stands — matches are explorations, not commitments
        # (unless the plan stage of this same run already snapshotted the true start).
        # The flag is cleared UNCONDITIONALLY: a stale flag from an aborted plan on
        # another camera must never suppress a later snapshot.
        snapped_for = getattr(self, "_plan_snapped", None)
        self._plan_snapped = None
        if snapped_for != camera_name:
            e.pre_match = ap.read_state(rig, self._baselines, cam)
        self.save_session()

        log("analyzing reference…")
        self._llm_down = False
        semantics = self._analyze_or_fallback(camera_name, log)
        log(f"reference: {semantics['time_of_day']}, {semantics['sky']} sky, "
            f"sun {semantics['sun_altitude_band']} @ bearing "
            f"{semantics['sun_bearing_deg']:+.0f}°, wb ~{semantics['wb_kelvin_estimate']:.0f}K"
            f" — {semantics['key_notes']}")

        ref_stats = self.ref_stats(e.reference)
        if ref_stats is None:
            log("⚠ reference stats unavailable (install Pillow or set system_python) — "
                "running LLM-visual mode")
        ref_block = self._image_block(e.reference)
        if ref_block is None and not getattr(self.cfg, "no_renders", False):
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

        draft_applied = False
        if self.cfg.draft_sampler:
            for line in df.apply_draft():
                log(line)
            draft_applied = df.pending_snapshot()

        def render_hook(tag: str):
            path = rd.render_frame(cam, os.path.join(run_dir, f"{tag}.png"),
                                   self.cfg.loop_width, self.cfg.loop_height)
            if path:
                log(f"THUMB::{path}")   # UI renders these markers as inline thumbnails
            return path

        hooks = Hooks(
            apply=lambda st: self._apply_logged(rig, st, cam, log),
            render=render_hook,
            stats=self.stats_for,
            llm_deltas=self._llm_deltas_hook(ref_block),
            log=log,
            should_cancel=should_cancel,
        )

        agreement = getattr(self, "_last_analyze_agreement", None)
        if agreement is not None and agreement < 1.0:
            log(f"⚠ analyze samples disagreed (agreement {agreement:.0%}) — consensus "
                "used; consider a cleaner reference if the match fights you")
            self._last_analyze_agreement = None

        if do_sweep and start_override is None and rig.get("sun") is not None \
                and "sun.azimuth_deg" not in locks:
            log(f"sun sweep: {self.cfg.sweep_count} directions…")
            az, alt_hint, _why = run_sun_sweep(
                start, rules.sweep_azimuths(self.cfg.sweep_count), hooks,
                llm_pick=lambda paths, azs: self._sweep_call(ref_block, paths, azs),
                ref_stats=ref_stats)
            if az is not None:
                start.set("sun.azimuth_deg", az)
                # the hint was judged against real renders of THIS scene — trust it over
                # the ANALYZE band when the altitude isn't locked
                if alt_hint != "na" and "sun.altitude_deg" not in locks \
                        and "sun.altitude_deg" in start.values:
                    start.set("sun.altitude_deg", rules.ALTITUDE_DEG.get(
                        alt_hint, start.get("sun.altitude_deg")))
                    log(f"sweep: altitude refined to "
                        f"{start.get('sun.altitude_deg'):.0f}° ('{alt_hint}')")

        cfg = MatchConfig(
            max_iterations=(max(int(self.cfg.max_iterations), 10) if deep
                            else int(self.cfg.max_iterations)),
            target_score=99.0 if deep else float(self.cfg.target_score),
            analytic=ref_stats is not None,
            weights=self.cfg.critic_weights or None,
            polish=deep,
        )
        if deep:
            log(f"DEEP MATCH: target 99 · up to {cfg.max_iterations} iterations · "
                "coordinate-descent polish to the scene's ceiling afterwards")
        try:
            result = run_match(start, ref_stats, semantics, hooks, cfg, locks,
                               rig_notes="; ".join(rig.get("notes", [])),
                               director_note=director_note)
        finally:
            if draft_applied:   # crash-safe: even a raise puts the artist's sampler back
                for line in df.restore_draft():
                    log(line)
        e.locks = locks
        self.session.record_match(camera_name, result.best_state, result.best_score)
        self.save_session()
        try:   # the calibration trail: every run leaves a machine-readable record
            with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
                json.dump({"camera": camera_name, "reference": e.reference,
                           "semantics": semantics, "llm_down": bool(self._llm_down),
                           **result.to_summary()}, f, indent=1)
        except (OSError, TypeError, ValueError):
            pass
        score_txt = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
        log(f"match finished: {result.stop_reason}, best score {score_txt} "
            f"({len(result.iterations)} iterations)")
        if result.polish_probes:
            log(f"polish: +{result.polish_gain:.2f} over {result.polish_probes} probes"
                + (" · CEILING PROVEN (no fine move improves)"
                   if result.ceiling_converged else ""))
        return result

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
        self.session.record_match(camera_name, start, None)
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
    ) -> MatchResult:
        """The conversation turn: a director's note → instant deterministic nudges →
        3-lens ENSEMBLE (competing corrections, each branch rendered and scored) → the
        winner continues into a deep match with the note pinned into every prompt."""
        e = self.session.entry(camera_name)
        if not e.reference:
            raise RuntimeError("bind a reference image to this camera first")
        e.notes = (getattr(e, "notes", []) + [note])[-6:]
        combined = " · ".join(e.notes[-3:])
        rig = self.rig(refresh=True)
        cam = sc.get_camera(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found")
        sc.set_active_camera(camera_name)
        run_dir = self._new_run_dir(camera_name)
        self._llm_down = False
        semantics = self._analyze_or_fallback(camera_name, log)  # re-analyzes on ref swap
        ref_stats = self.ref_stats(e.reference)
        ref_block = self._image_block(e.reference)
        if ref_block is None and not getattr(self.cfg, "no_renders", False):
            raise RuntimeError("reference image could not be prepared for the LLM")

        base = (e.state.copy() if e.state is not None
                else ap.read_state(rig, self._baselines, cam))
        deltas = feedback.nudges_from_note(note, base.keys(), list(base.groups))
        state0, applied = feedback.apply_note_deltas(base, deltas)
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

        def probe(st: LightingState, tag: str):
            self._apply_logged(rig, st, cam, log)
            path = rd.render_frame(cam, os.path.join(run_dir, f"{tag}.png"),
                                   self.cfg.loop_width, self.cfg.loop_height)
            stats = self.stats_for(path) if path else None
            if stats is None or ref_stats is None:
                return None, path
            return critic.score(ref_stats, stats,
                                self.cfg.critic_weights or None).score, path

        score0, path0 = probe(state0, "refine_note")
        log(f"branch note-only: {score0:.1f}" if score0 is not None
            else "branch note-only: unscored")
        branches = [("note-only", state0, score0)]
        render0_block = self._image_block(path0) if path0 else None
        from ..core.genome import apply_changes, state_table
        from ..core.parse import validate_deltas

        for lens_name, lens_line in feedback.LENSES:
            if should_cancel():
                break
            content = [ref_block]
            if render0_block is not None:
                content.append(render0_block)
            content.append(omega.text_block(prompts.deltas_user_text(
                state_table(state0, e.locks), semantics, [], {}, 0, 1,
                "; ".join(rig.get("notes", [])), "", combined)))
            try:
                reply = self.io(lambda: omega.call(
                    self.cfg.api_key,
                    feedback.lens_system(prompts.DELTAS_SYSTEM, lens_line),
                    [{"role": "user", "content": content}],
                    model=self.cfg.model, max_tokens=2048))
                proposal = validate_deltas(reply)
            except Exception as err:  # noqa: BLE001 a dead lens must not kill the round
                log(f"lens {lens_name}: unusable ({err})")
                continue
            cand, accepted, _rej = apply_changes(state0, proposal["changes"], e.locks,
                                                 limit=True)
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
        return self.run_match(camera_name, log, should_cancel, locks=set(e.locks),
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
                results[name] = (f"{r.best_score:.1f}" if r.best_score is not None
                                 else r.stop_reason)
            except Exception as err:  # noqa: BLE001 one bad camera must not kill the night
                results[name] = f"error: {err}"
                log(f"✗ {name}: {err}")
        return results

    # ------------------------------------------------------------------ scenario board
    def run_scenarios(
        self,
        camera_name: str,
        log: Callable[[str], None],
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> List[Dict]:
        """Render the candidate rigs from core.scenarios at loop res, score each against
        the reference when one is bound, and leave the scene exactly as it was found.
        → [{key, label, why, state, render, score}] in board order."""
        rig = self.rig(refresh=True)
        cam = sc.get_camera(camera_name)
        if cam is None:
            raise RuntimeError(f"camera '{camera_name}' not found in the scene")
        sc.set_active_camera(camera_name)
        e = self.session.entry(camera_name)
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
        for cand in board:
            if should_cancel():
                log("scenario board cancelled")
                break
            self._apply_logged(rig, cand["state"], cam, log)
            path = rd.render_frame(
                cam, os.path.join(run_dir, f"scen_{cand['key']}.png"),
                self.cfg.loop_width, self.cfg.loop_height)
            score = None
            if path:
                log(f"THUMB::{path}")
                if ref is not None:
                    cur = self.stats_for(path)
                    if cur is not None:
                        score = critic.score(ref, cur,
                                             self.cfg.critic_weights or None).score
            log(f"scenario {cand['label']}: "
                + (f"{score:.1f}" if score is not None else "unscored")
                + f" — {cand['why']}")
            results.append({**cand, "render": path, "score": score})
        self._apply_logged(rig, current, cam, log)   # leave the scene as it was found
        return results

    def adopt_scenario(self, camera_name: str, state: LightingState,
                       score: Optional[float] = None) -> List[str]:
        """Apply a board candidate and save it as the camera's state — MATCH/REFINE
        continue from it exactly like any matched state. Returns apply warnings."""
        warnings = self.apply_state(state, camera_name)
        self.session.record_match(camera_name, state, score)
        self.save_session()
        return warnings

    # ------------------------------------------------------------------ dome seed
    def seed_dome(self, camera_name: str, log: Callable[[str], None]) -> Dict:
        """Reference → world-oriented HDR pano → dome texture, rotation zeroed (the pano
        is world-aligned; dome.rotation_deg stays live for the loop to spin). The dome's
        previous texture/rotation is snapshotted once for Restore. → build_seed meta."""
        e = self.session.entry(camera_name)
        if not e.reference:
            raise RuntimeError("bind a reference image first — the seed is built from it")
        rig = self.rig(refresh=True)
        dome = rig.get("dome")
        if dome is None:
            raise RuntimeError("no VRayLight dome in the rig — add one (or let a plan "
                               "create it), Read scene, then seed")
        cam = sc.get_camera(camera_name)
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
        # re-seed (new reference, moved sun) into the same file can render stale
        import hashlib

        try:
            ref_sig = f"{e.reference}:{os.path.getmtime(e.reference):.0f}"
        except OSError:
            ref_sig = e.reference
        token = hashlib.md5(
            f"{ref_sig}|{sun_az}|{sun_alt}".encode("utf-8", "replace")).hexdigest()[:8]
        out = os.path.join(self._ensure_run_dir(_safe(camera_name)),
                           domeseed.seed_filename(camera_name, token))
        prev_seed = e.seed_hdri
        yaw = sc.camera_yaw_deg(cam) if cam is not None else 0.0
        src = e.reference
        if _needs_max_ingest(src):             # EXR/HDR/TIFF ref: Max transcodes first
            src = self._transcode_ref(e.reference) or src
        parametric = sun_az is not None        # live VRaySun owns direction → no disc
        meta = domeseed.build_seed(out, ref_path=src, semantics=semantics,
                                   cam_yaw_deg=yaw, sun_az_deg=sun_az,
                                   sun_alt_deg=sun_alt,
                                   parametric_sun_active=parametric)
        if meta is None and src == e.reference:   # plain loader failed (JPEG, no Pillow)
            png = self._transcode_ref(e.reference)
            if png:
                meta = domeseed.build_seed(out, ref_path=png, semantics=semantics,
                                           cam_yaw_deg=yaw, sun_az_deg=sun_az,
                                           sun_alt_deg=sun_alt,
                                           parametric_sun_active=parametric)
        if meta is None:
            raise RuntimeError("could not read the reference for seeding")
        how = self.set_dome_hdri(out)
        if how == "failed":
            raise RuntimeError("dome texture could not be set (no writable file property "
                               "— checklist #16)")
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
            f"rotation zeroed via {rot_how})")
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
            self.session.record_match(camera_name, state, None)
            self.save_session()
        return warnings

    def set_dome_hdri(self, hdri_path: str) -> str:
        dome = self.rig().get("dome")
        if dome is None:
            return "failed"
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
            return self.io(lambda: omega.call(
                self.cfg.api_key, prompts.SWEEP_SYSTEM,
                [{"role": "user", "content": content}],
                model=self.cfg.model, max_tokens=1024))
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

    # ------------------------------------------------------------------ restore
    def restore_pre_match(self, camera_name: str) -> bool:
        e = self.session.cameras.get(camera_name)
        if not (e and e.pre_match is not None):
            return False
        self.apply_state(e.pre_match, camera_name)
        if e.pre_seed:                         # a seed replaced the dome texture — undo it
            dome = self.rig().get("dome")
            if dome is not None:
                prev = str(e.pre_seed.get("file") or "")
                if prev:
                    sc.set_dome_texture(dome, prev)
                else:                          # dome had no HDRI: disable texture use
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
        Note: the scene is left holding the LAST camera's lighting state.

        vrscene exports are the heavyweight artifacts (100s of MB on real interiors), so
        old export batches are pruned by the same keep_runs policy as loop renders."""
        vantage_parent = self._ensure_run_dir("vantage")
        export_dir = os.path.join(vantage_parent, _stamp())
        jobs: List[Dict] = []
        for name in camera_names:
            on_progress(name, "applying + exporting")
            if use_saved_states:
                e = self.session.cameras.get(name)
                if e and e.state is not None:
                    self.apply_state(e.state, name)
                self._rebind_seed(name)        # each vrscene must carry ITS camera's sky
            scene_file = vt.export_vrscene(
                os.path.join(export_dir, f"{_safe(name)}.vrscene"), name)
            if scene_file is None:
                raise RuntimeError(f"{name}: vrscene export failed "
                                   "(vrayExportVRScene missing or camera not set)")
            jobs.append({"camera": name, "scene_file": scene_file,
                         "output": os.path.join(out_dir, f"{_safe(name)}.png")})
        prune_old_runs(vantage_parent, keep=int(self.cfg.keep_runs))
        return jobs

    def run_vantage_jobs(self, jobs: List[Dict],
                         on_progress: Callable[[str, str], None]) -> Dict[str, str]:
        """LEGACY (Developer-Edition CLI only) — pure subprocess, worker-thread safe."""
        if getattr(self.cfg, "no_renders", False):
            for job in jobs:
                on_progress(job.get("camera", ""),
                            "skipped — no-render mode is ON (Settings)")
            return {job.get("camera", ""): "skipped (no-render mode)" for job in jobs}
        return vt.render_stills(jobs, self.cfg.vantage_console,
                                self.cfg.final_width, self.cfg.final_height, on_progress)

    def render_finals_vray(
        self,
        camera_names: List[str],
        out_dir: str,
        on_progress: Callable[[str, str], None],
    ) -> Dict[str, str]:
        """DEFAULT final-render backend (stock Vantage 3.x has no headless CLI): per
        camera, apply its saved state and production-render through V-Ray at final res.
        MAIN THREAD — renders block Max by nature; progress narrates between shots."""
        if getattr(self.cfg, "no_renders", False):
            for name in camera_names:
                on_progress(name, "skipped — no-render mode is ON (Settings)")
            return {name: "skipped (no-render mode)" for name in camera_names}
        results: Dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)
        for name in camera_names:
            on_progress(name, "applying state")
            e = self.session.cameras.get(name)
            if e and e.state is not None:
                self.apply_state(e.state, name)
            self._rebind_seed(name)            # finals render under their own seed too
            cam = sc.get_camera(name)
            if cam is None:
                results[name] = "camera not found"
                on_progress(name, results[name])
                continue
            on_progress(name, f"rendering {self.cfg.final_width}×{self.cfg.final_height} (V-Ray)")
            out = rd.render_frame(cam, os.path.join(out_dir, f"{_safe(name)}.png"),
                                  self.cfg.final_width, self.cfg.final_height)
            results[name] = "ok" if out else "render failed"
            on_progress(name, results[name])
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
        launched = vt.launch_vantage(self.cfg.vantage_exe,
                                     jobs[0]["scene_file"] if jobs else None)
        return jobs, launched, export_dir

    # ------------------------------------------------------------------ dirs
    def _ensure_run_dir(self, sub: str) -> str:
        stem = _safe(os.path.splitext(os.path.basename(sc.scene_path() or "unsaved"))[0])
        d = os.path.join(cfgmod.sessions_dir(), stem, sub)
        os.makedirs(d, exist_ok=True)
        return d

    def _new_run_dir(self, camera_name: str) -> str:
        parent = self._ensure_run_dir(_safe(camera_name))
        d = os.path.join(parent, _stamp())
        os.makedirs(d, exist_ok=True)
        self._run_dir = d
        prune_old_runs(parent, keep=int(self.cfg.keep_runs))
        return d


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


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "unnamed"


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
