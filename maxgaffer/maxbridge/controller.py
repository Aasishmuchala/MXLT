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

from ..core import (animation, consensus, critic, domeseed, expose, fairness, feedback,
                    transfer,
                    metrics, omega, planner, profiles, prompts, providers, rules,
                    scenedigest, scenarios as scen)
from ..core.director import (Hooks, MatchConfig, MatchResult, TRANSFER_WEIGHT,
                             blend_transfer, run_match, run_sun_sweep)
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
from . import scene as sc
from . import vantage as vt

# formats Max reads natively but Pillow/stdlib can't — always ingest via Max transcode
MAX_FIRST_EXTS = (".exr", ".hdr", ".tif", ".tiff")

# director's notes are persisted AND pinned into every DELTAS prompt of the ensemble and
# the whole deep match — cap their SIZE (the [-6:] list cap only bounds their count)
MAX_NOTE_CHARS = 500


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
        self._session_gen = sc.scene_generation()
        sc.register_scene_callbacks()

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
            except Exception:  # noqa: BLE001 — bookkeeping must never sink a good match
                pass

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
            return self.analyze_reference(camera_name)
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
                        entry=None, log: Optional[Callable[[str], None]] = None):
        """``render_frame`` + software exposure when enabled — the ONE render path
        every probe/loop/board/final goes through, so the EV/WB the plugin sets always
        reach the scored (and delivered) pixels, even on renderers whose exposure host
        is display-stage only (V-Ray GPU)."""
        path = rd.render_frame(cam, out_path, w, h)
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
            if log:
                log("⚠ software exposure needs Pillow — frame left un-exposed")
            self._sw_warned = True
        return path

    def _verify_exposure_host(self, cam, run_dir: str,
                              log: Callable[[str], None]) -> None:
        """One-time (per Controller) MEASUREMENT: does the renderer bake EV into the
        saved buffer? Two tiny probes 2 EV apart must move the key ~2 stops; if it
        barely moves, the host is display-stage only (measured on-box for V-Ray GPU)
        and software exposure is switched on for the session, loudly."""
        if getattr(self, "_exposure_host_checked", False) \
                or getattr(self.cfg, "software_exposure", False) \
                or getattr(self.cfg, "no_renders", False):
            return
        self._exposure_host_checked = True
        from .exposure import ExposureHost

        host = ExposureHost(cam)
        ev0 = host.read_ev()
        if ev0 is None:
            return
        try:
            p1 = rd.render_frame(cam, os.path.join(run_dir, "evcheck_a.png"), 160, 90)
            s1 = metrics.compute_stats(p1) if p1 else None
            host.write_ev(ev0 + 2.0)
            p2 = rd.render_frame(cam, os.path.join(run_dir, "evcheck_b.png"), 160, 90)
            s2 = metrics.compute_stats(p2) if p2 else None
        finally:
            host.write_ev(ev0)
        if not (s1 and s2):
            return
        import math

        moved = abs(math.log2(max(1e-5, s1["log_key"]) / max(1e-5, s2["log_key"])))
        if moved < 1.0:      # expected ~2 stops; under half = EV not reaching pixels
            self.cfg.software_exposure = True
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

    def probe_score(self, camera_name: str, tag: str) -> Optional[float]:
        """One loop-res render of the current scene scored against the camera's reference
        — the cheap 'did that help?' measurement."""
        if getattr(self.cfg, "no_renders", False):
            return None                    # no-render mode: plans apply unmeasured
        e = self.camera_entry(camera_name)
        if not (e and e.reference):
            return None
        ref = self.ref_stats(e.reference)
        cam = self.camera_node(camera_name)
        if ref is None or cam is None:
            return None
        path = self._render_exposed(
            cam, os.path.join(self._ensure_run_dir(_safe(camera_name)), f"probe_{tag}.png"),
            self.cfg.loop_width, self.cfg.loop_height, entry=e)
        cur = self.stats_for(path) if path else None
        if cur is None:
            return None
        return critic.score(ref, cur, self._critic_weights()).score

    def execute_plan(self, ops, camera_name: str,
                     log: Callable[[str], None], measure: bool = True) -> Dict:
        """Execute a validated plan (one undo record; MG_ layer for created lights) and
        return the before/after change report — including the plan's MEASURED effect
        (critic score before vs after, one probe render each side). Rig re-classified
        after so new lights join the dimmer boards immediately."""
        before_score = self.probe_score(camera_name, "preplan") if measure else None
        cam = self.camera_node(camera_name)
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
        self._set_active_camera(camera_name)
        # checklist #14 is a CRASH vector, not a perf note: V-Ray GPU loop renders while
        # a Vantage live link streams the same scene can VRAM-starve the one card and
        # take Max down. Detect the combination and say so BEFORE the first render.
        try:
            from . import vantage as _vt

            port = _vt.link_running()
            rcls = str(sc._rt().classOf(sc._rt().renderers.current)).lower()
            if port is not None and "gpu" in rcls:
                log(f"⚠ Vantage live link is UP (port {port}) and the renderer is "
                    "V-Ray GPU — one GPU doing both can crash Max (checklist #14). "
                    "Close the link for the match or switch V-Ray to CPU.")
        except Exception:
            pass
        if self.cfg.auto_exposure_control:
            from .exposure import ExposureHost, ensure_exposure_control

            if ExposureHost(cam).kind == "none":
                created = ensure_exposure_control(camera=cam)
                if created:
                    log("⚠ " + created)
                    # remember WE created it — Restore must exit the experiment entirely
                    e.ec_created = True
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

        draft_applied = False
        if self.cfg.draft_sampler:
            for line in df.apply_draft():
                log(line)
            draft_applied = df.pending_snapshot()

        # the restoring finally opens IMMEDIATELY after the draft is applied: a gateway
        # error in the exposure check, the sweep or the MatchConfig build must never
        # strand the artist's sampler settings for the rest of the Max session
        try:
            self._verify_exposure_host(cam, run_dir, log)

            profile = profiles.resolve_profile(
                "hero" if deep else quality_profile,
                loop_width=self.cfg.loop_width,
                loop_height=self.cfg.loop_height,
                max_iterations=self.cfg.max_iterations,
                sweep_count=self.cfg.sweep_count,
                target_score=self.cfg.target_score,
            )
            sweep_part = profile.sweep_count if do_sweep else 0
            polish_part = profile.polish_max_probes if profile.polish else 0
            hard_cap = profile.max_iterations + sweep_part + polish_part
            log(f"{profile.label.upper()} PROFILE: {profile.loop_width}×"
                f"{profile.loop_height} loop · ≤{profile.max_iterations} iterations"
                + (f" · ≤{sweep_part} low-res sweep renders" if sweep_part else "")
                + (f" · ≤{polish_part} polish probes" if polish_part else "")
                + f" · hard worst-case ≤{hard_cap} renders")

            # software exposure: apply the just-applied state's EV/WB to each loop frame
            # before it's scored, anchored at the camera's pre_match snapshot (the same
            # anchor probes/board/finals use, so all exposed pixels stay consistent)
            self._sw_state = start
            self._sw_warned = False
            if getattr(self.cfg, "software_exposure", False) \
                    and "exposure.ev" in start.values:
                log("software exposure ON — EV/WB applied to loop frames before scoring "
                    "(renderer-independent; V-Ray GPU's exposure is display-stage only)")

            def render_hook(tag: str):
                is_sweep = tag.startswith("sweep")
                width = profile.sweep_width if is_sweep else profile.loop_width
                height = profile.sweep_height if is_sweep else profile.loop_height
                path = self._render_exposed(
                    cam, os.path.join(run_dir, f"{tag}.png"),
                    width, height,
                    state=getattr(self, "_sw_state", None) or start, entry=e, log=log)
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
                bearing_trust = max(0.25, min(1.0, float(
                    semantics.get("sun_bearing_agreement", 1.0) or 0.0)))
                spread = semantics.get("sun_bearing_spread_deg")
                if spread is not None and bearing_trust < 1.0:
                    log(f"⚠ ANALYZE disagreed with itself on the sun's direction "
                        f"(±{spread:.0f}° across samples) — leaning harder on the render "
                        f"and less on the reading (trust {bearing_trust:.0%})")
                cfg_kw["transfer_weight"] = TRANSFER_WEIGHT * bearing_trust

            # A COPY: the sun sweep updates the bearing below, and e.semantics is the
            # cached record of what ANALYZE actually read. Writing a sweep-derived bearing
            # back into the cache would launder a measurement into the reading and poison
            # every later run of this camera.
            sem_live = dict(semantics) if semantics else None

            def transfer_hook(st):
                try:
                    return transfer.score(sem_live, st, cam_yaw)["score"] / 100.0
                except Exception:  # noqa: BLE001
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

            if do_sweep and start_override is None and rig.get("sun") is not None \
                    and "sun.azimuth_deg" not in locks:
                log(f"sun sweep: {profile.sweep_count} directions at "
                    f"{profile.sweep_width}×{profile.sweep_height}…")
                sweep_out: Dict = {}
                az, alt_hint, _why = run_sun_sweep(
                    start, rules.sweep_azimuths(profile.sweep_count), hooks,
                    llm_pick=lambda paths, azs: self._sweep_call(ref_block, paths, azs),
                    ref_stats=ref_stats, out=sweep_out)
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
            )
            if profile.polish:
                log("HERO MATCH: target 99 · bounded coordinate-descent polish "
                    "to the measured scene ceiling")
            result = run_match(start, ref_stats, semantics, hooks, cfg, locks,
                               rig_notes="; ".join(rig.get("notes", [])),
                               director_note=director_note)
        finally:
            if draft_applied:   # crash-safe: even a raise puts the artist's sampler back
                for line in df.restore_draft():
                    log(line)
        e.locks = locks
        if result.best_score is None and result.stop_reason in ("cancelled",
                                                                "render_failed"):
            # nothing was ever measured — recording would overwrite the camera's
            # accepted state+score with an unmeasured first guess. Put the scene back
            # to its previous light instead (the director skipped its final apply too).
            prev = e.state if e.state is not None else e.pre_match
            if prev is not None:
                self._apply_logged(rig, prev, cam, log)
            log(f"match {result.stop_reason} before any successful render — "
                "no measurement, kept previous lighting")
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
                           **result.to_summary()}, f, indent=1)
        except (OSError, TypeError, ValueError):
            pass
        score_txt = f"{result.best_score:.1f}" if result.best_score is not None else "n/a"
        log(f"match finished: {result.stop_reason}, best score {score_txt} "
            f"({len(result.iterations)} iterations)")
        if result.polish_probes:
            tail = ""
            if getattr(result, "ceiling_proven", False):
                tail = " · CEILING PROVEN (no fine move improves)"
            elif result.ceiling_converged:
                tail = " · plateau (finer steps untested)"
            log(f"polish: +{result.polish_gain:.2f} over {result.polish_probes} probes"
                + tail)
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
        for key, st in cands:
            if hooks.should_cancel():
                break
            try:
                hooks.apply(st)
                path = hooks.render(f"sweep_basin_{key}")
                cur = self.stats_for(path) if path else None
                if cur is None:
                    continue
                value = critic.score(ref_stats, cur, self._critic_weights()).score
                # Judged on the SAME objective the loop will use. On pixels alone this
                # probe picked a SUNLESS basin for a sunlit golden-hour reference
                # (measured on-box 2026-07-25) — which then suppressed the sun lock
                # below, because that lock deliberately leaves a genuinely sunless basin
                # free. One weak comparison at the top of the run cost the whole match:
                # the loop finished at 80.35 with the sun switched off.
                value = blend_transfer(value, st, hooks, cfg_probe)
            except Exception as err:  # noqa: BLE001 one bad candidate ≠ a dead match
                log(f"multi-start: {key} probe failed ({err})")
                continue
            log(f"multi-start: {key} → {value:.1f}")
            if best_score is None or value > best_score:
                best_key, best_state, best_score = key, st.copy(), value
        if best_score is not None:
            log(f"multi-start: best basin '{best_key}' at {best_score:.1f} "
                f"(of {len(cands)} probed) — the match starts here")
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
            print(f"MaxGaffer: sessions dir unavailable ({err}) — run files go to {d}")
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
