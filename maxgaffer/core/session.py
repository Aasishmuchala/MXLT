"""Per-camera lighting sessions — every camera can carry its own reference + matched rig.

Archviz reality (the TULA shot-book workflow): each shot wants its own sun. So MaxGaffer's
unit of work is (camera, reference, LightingState, score), persisted in a sidecar JSON next
to the .max file — human-readable, diff-able, survives Max crashes, never bloats the scene.

The bridge owns *when* to apply a camera's state (on select / on render); this owns the data.
Timestamps are injected so tests stay deterministic.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .genome import LightingState

log = logging.getLogger(__name__)

FORMAT_VERSION = 2


def _str_list(value) -> List:
    """Only real JSON arrays pass — a hand-edited ``"locks": "sun.intensity"`` string
    would otherwise load as a set of single characters."""
    return list(value) if isinstance(value, (list, tuple, set)) else []


def sidecar_path(scene_path: str) -> Optional[str]:
    """foo.max → foo.maxgaffer.json (None for an unsaved scene)."""
    if not scene_path:
        return None
    root, _ = os.path.splitext(scene_path)
    return root + ".maxgaffer.json" if root else None


def reference_signature(path: str) -> str:
    """Stable identity for a reference, including in-place file replacements.

    Artists commonly overwrite ``reference.jpg`` and bind the same path again.  Path-only
    invalidation kept the old semantic analysis and score in that case.  Nanosecond mtime
    plus size catches the replacement without reading or hashing a multi-gigabyte EXR.
    """
    normalized = os.path.normcase(os.path.abspath(path)) if path else ""
    try:
        st = os.stat(path)
        return f"{normalized}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return f"{normalized}|missing"


@dataclass
class CameraEntry:
    camera_id: str = ""                       # persistent Max anim handle when available
    camera_name: str = ""                     # display label; may change without losing state
    reference: str = ""                       # reference image path ("" = none bound)
    reference_relative: str = ""              # portable path relative to the scene folder
    reference_signature: str = ""             # path + stat identity for in-place changes
    state: Optional[LightingState] = None     # last accepted rig for this camera
    score: Optional[float] = None
    matched_at: str = ""
    locks: Set[str] = field(default_factory=set)
    semantics: Dict = field(default_factory=dict)   # cached ANALYZE result for the reference
    pre_match: Optional[LightingState] = None       # the light BEFORE the last match run
    notes: List[str] = field(default_factory=list)  # director's notes, newest last
    seed_hdri: str = ""                             # generated dome-seed .hdr ("" = none)
    pre_seed: Dict = field(default_factory=dict)    # dome texture/rotation before seeding
    scorecard: Dict = field(default_factory=dict)   # component confidence + honest gap read
    artist_feedback: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "reference": self.reference,
            "reference_relative": self.reference_relative,
            "reference_signature": self.reference_signature,
            "state": self.state.to_dict() if self.state else None,
            "score": self.score,
            "matched_at": self.matched_at,
            "locks": sorted(self.locks),
            "semantics": self.semantics,
            "pre_match": self.pre_match.to_dict() if self.pre_match else None,
            "notes": list(self.notes),
            "seed_hdri": self.seed_hdri,
            "pre_seed": dict(self.pre_seed),
            "scorecard": dict(self.scorecard),
            "artist_feedback": list(self.artist_feedback),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CameraEntry":
        state = d.get("state")
        pre = d.get("pre_match")
        score_raw = d.get("score")
        try:
            score = float(score_raw) if isinstance(score_raw, (int, float)) else None
            if score is not None and not math.isfinite(score):
                score = None
        except (TypeError, ValueError):
            score = None
        return cls(
            camera_id=str(d.get("camera_id") or ""),
            camera_name=str(d.get("camera_name") or ""),
            reference=str(d.get("reference") or ""),
            reference_relative=str(d.get("reference_relative") or ""),
            reference_signature=str(d.get("reference_signature") or ""),
            state=LightingState.from_dict(state) if isinstance(state, dict) else None,
            score=score,
            matched_at=str(d.get("matched_at") or ""),
            locks=set(x for x in _str_list(d.get("locks")) if isinstance(x, str)),
            semantics=d.get("semantics") if isinstance(d.get("semantics"), dict) else {},
            pre_match=LightingState.from_dict(pre) if isinstance(pre, dict) else None,
            notes=[str(x) for x in _str_list(d.get("notes")) if isinstance(x, str)],
            seed_hdri=str(d.get("seed_hdri") or ""),
            pre_seed=d.get("pre_seed") if isinstance(d.get("pre_seed"), dict) else {},
            scorecard=d.get("scorecard") if isinstance(d.get("scorecard"), dict) else {},
            artist_feedback=[dict(x) for x in _str_list(d.get("artist_feedback"))
                             if isinstance(x, dict)][-50:],
        )


class Session:
    def __init__(self, path: Optional[str] = None, now_fn: Callable[[], str] = None):
        self.path = path
        self.cameras: Dict[str, CameraEntry] = {}
        self.settings: Dict = {"apply_on_select": True}
        # AUTHORED light multipliers, keyed by light NAME, adopted once and never
        # overwritten — group factors multiply these. Persisting them is what prevents the
        # baseline-poisoning bug: re-capturing after MaxGaffer set a group to 0 would record
        # base=0 and kill the group forever (0 × factor). Names survive Max restarts;
        # anim handles do not.
        self.baselines: Dict[str, float] = {}
        self._now = now_fn or _iso_now
        # Set when the on-disk sidecar failed to load (or is a newer format): auto-save
        # must NOT overwrite the file until the user saves explicitly (save(force=True)).
        self._protect_existing = False

    def adopt_baselines(self, fresh: Dict[str, float]) -> List[str]:
        """Adopt baselines for lights we have never seen; NEVER overwrite known ones.
        A 0.0 (or non-finite) multiplier is declined: it is almost always a dimmed light,
        not an authored value, and adopting it kills the group forever (0 × any factor).
        Returns the names actually adopted."""
        added: List[str] = []
        for name, value in (fresh or {}).items():
            if name in self.baselines:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                log.warning("MaxGaffer: refusing non-finite baseline for light %r: %r",
                            name, value)
                continue
            if v == 0.0:
                log.warning("MaxGaffer: declining to adopt 0.0 baseline for light %r — "
                            "likely dimmed, not authored (0.0 × any factor stays 0 "
                            "forever); re-author the light, or forget_baseline + re-scan "
                            "to force re-capture", name)
                continue
            self.baselines[str(name)] = v
            added.append(str(name))
        return added

    def forget_baseline(self, name: str) -> None:
        """Explicit re-adopt hook (user re-authored a light and wants the new value)."""
        self.baselines.pop(name, None)

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Optional[str], now_fn: Callable[[], str] = None) -> "Session":
        s = cls(path, now_fn)
        if not path or not os.path.exists(path):
            return s
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError, RecursionError) as e:
            s._quarantine_corrupt(f"unreadable sidecar ({e})")
            return s
        if not isinstance(d, dict):
            # valid JSON, wrong shape (hand-edit / another tool) — same data-loss guard
            s._quarantine_corrupt(f"sidecar top level is {type(d).__name__}, not an object")
            return s
        version = d.get("version")
        if isinstance(version, (int, float)) and version > FORMAT_VERSION:
            log.warning("MaxGaffer: sidecar %s is format v%s, NEWER than this build's "
                        "v%d — loading best-effort and blocking auto-save so the newer "
                        "file is not silently downgraded; save explicitly to force",
                        path, version, FORMAT_VERSION)
            s._protect_existing = True
        cameras = d.get("cameras")
        if isinstance(cameras, dict):
            for name, entry in cameras.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    loaded = CameraEntry.from_dict(entry)
                    if not loaded.camera_name:
                        loaded.camera_name = str(name)
                    if loaded.reference_relative and path:
                        candidate = os.path.normpath(os.path.join(
                            os.path.dirname(path), loaded.reference_relative))
                        # Prefer the project-relative copy after a project folder move.
                        if os.path.isfile(candidate):
                            loaded.reference = candidate
                    s.cameras[str(name)] = loaded
                except Exception as e:  # one corrupt camera must not kill the rest
                    log.warning("MaxGaffer: skipping corrupt camera entry %r in %s: %s",
                                name, path, e)
        if isinstance(d.get("settings"), dict):
            s.settings.update(d["settings"])
        if isinstance(d.get("baselines"), dict):
            s.adopt_baselines(d["baselines"])
        return s

    def _quarantine_corrupt(self, reason: str) -> None:
        """Move the unreadable sidecar to a timestamped .corrupt backup, log loudly, and
        block auto-save — the file may still be human-recoverable, and the old behavior
        (empty session silently saved over it) destroyed it on the next rig scan."""
        self._protect_existing = True
        stamp = "".join(c if c.isalnum() else "" for c in self._now())
        backup = f"{self.path}.{stamp}.{os.getpid()}.corrupt"
        try:
            os.replace(self.path, backup)
            log.warning("MaxGaffer: %s — moved %s → %s and started an EMPTY session; "
                        "auto-save is blocked until you save explicitly",
                        reason, self.path, backup)
        except OSError as e:
            log.warning("MaxGaffer: %s — could not move %s aside (%s); started an EMPTY "
                        "session, auto-save is blocked until you save explicitly",
                        reason, self.path, e)

    def save(self, force: bool = False) -> bool:
        if not self.path:
            return False
        if self._protect_existing and not force:
            log.warning("MaxGaffer: auto-save BLOCKED for %s — the previous sidecar "
                        "failed to load (or is a newer format); call save(force=True) "
                        "to overwrite deliberately", self.path)
            return False
        payload = {
            "version": FORMAT_VERSION,
            "cameras": {n: e.to_dict() for n, e in self.cameras.items()},
            "settings": self.settings,
            "baselines": self.baselines,
        }
        # per-writer tmp name: two Max instances on the same scene can't tear each
        # other's write or collide on os.replace
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            os.replace(tmp, self.path)
            self._protect_existing = False   # a successful explicit save re-arms saving
            return True
        except (OSError, TypeError, ValueError) as e:
            # a failed atomic write must not strand the .tmp (json.dump raises TypeError
            # mid-stream on a non-serializable settings value, OSError on full disks)
            log.warning("MaxGaffer: session save failed for %s: %s", self.path, e)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False

    # ------------------------------------------------------------------ camera API
    def find(self, camera: str, camera_id: str = "") -> Optional[CameraEntry]:
        """Find by persistent identity first, then by legacy/name key."""
        if camera_id:
            for entry in self.cameras.values():
                if entry.camera_id == str(camera_id):
                    if camera:
                        entry.camera_name = camera
                    return entry
        direct = self.cameras.get(camera)
        if direct is not None and (not camera_id or not direct.camera_id
                                   or direct.camera_id == str(camera_id)):
            return direct
        for entry in self.cameras.values():
            if entry.camera_name == camera and (not camera_id or not entry.camera_id
                                                or entry.camera_id == str(camera_id)):
                return entry
        return None

    def entry(self, camera: str, camera_id: str = "") -> CameraEntry:
        found = self.find(camera, camera_id)
        if found is not None:
            if camera_id and not found.camera_id:
                found.camera_id = str(camera_id)
            if camera:
                found.camera_name = camera
            return found
        key = camera
        if key in self.cameras:
            key = f"{camera}@@{camera_id or len(self.cameras) + 1}"
        entry = CameraEntry(camera_id=str(camera_id or ""), camera_name=camera)
        self.cameras[key] = entry
        return entry

    def set_reference(self, camera: str, ref_path: str, camera_id: str = "") -> None:
        e = self.entry(camera, camera_id)
        rel = ""
        if ref_path and self.path:
            try:
                rel_candidate = os.path.relpath(ref_path, os.path.dirname(self.path))
                if rel_candidate != os.pardir and not rel_candidate.startswith(os.pardir + os.sep):
                    rel = rel_candidate
            except (OSError, ValueError):
                pass
        signature = reference_signature(ref_path)
        if e.reference != ref_path or e.reference_signature != signature:
            e.reference = ref_path
            e.reference_relative = rel
            e.reference_signature = signature
            e.semantics = {}          # a new reference invalidates the cached analysis
            e.score = None

    def record_match(self, camera: str, state: LightingState,
                     score: Optional[float], camera_id: str = "") -> None:
        e = self.entry(camera, camera_id)
        e.state = state.copy()
        e.score = score
        e.matched_at = self._now()

    def record_artist_feedback(self, camera: str, accepted: bool,
                               rating: Optional[int] = None, note: str = "",
                               camera_id: str = "") -> Dict:
        """Persist the human verdict that the numeric score can never infer."""
        e = self.entry(camera, camera_id)
        try:
            stars = min(5, max(1, int(rating))) if rating is not None else None
        except (TypeError, ValueError):
            stars = None
        item = {"at": self._now(), "accepted": bool(accepted),
                "rating": stars, "note": str(note or "")[:500],
                "score": e.score, "scorecard": dict(e.scorecard)}
        e.artist_feedback = (e.artist_feedback + [item])[-50:]
        return item

    def cameras_with_states(self) -> List[str]:
        return [(e.camera_name or n) for n, e in self.cameras.items() if e.state is not None]

    def relink_references(self, roots: List[str]) -> Dict[str, str]:
        """Relink missing references by their portable relative path or unique basename."""
        changed: Dict[str, str] = {}
        valid_roots = [os.path.abspath(r) for r in roots if r and os.path.isdir(r)]
        for key, entry in self.cameras.items():
            if not entry.reference or os.path.isfile(entry.reference):
                continue
            candidates: List[str] = []
            for root in valid_roots:
                if entry.reference_relative:
                    candidates.append(os.path.join(root, entry.reference_relative))
                basename = os.path.basename(entry.reference)
                candidates.extend(os.path.join(dp, basename) for dp, _dirs, files in os.walk(root)
                                  if basename in files)
            hits = [os.path.normpath(p) for p in candidates if os.path.isfile(p)]
            hits = list(dict.fromkeys(hits))
            if len(hits) == 1:
                entry.reference = hits[0]
                entry.reference_signature = reference_signature(hits[0])
                changed[entry.camera_name or key] = hits[0]
        return changed


PRESET_VERSION = 1


def preset_dumps(state: LightingState, name: str = "", now: str = "") -> str:
    """Serialize a lighting state as a portable preset (share across scenes/machines)."""
    return json.dumps({"maxgaffer_preset": PRESET_VERSION, "name": name, "saved_at": now,
                       "state": state.to_dict()}, indent=1)


def preset_loads(text: str) -> Optional[LightingState]:
    """Parse a preset; None if it isn't one. Values re-clamped by the genome on load."""
    try:
        d = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(d, dict) or "maxgaffer_preset" not in d:
        return None
    state = d.get("state")
    if not isinstance(state, dict):
        return None
    try:
        return LightingState.from_dict(state)
    except (TypeError, ValueError):
        return None


def _iso_now() -> str:
    import datetime

    return datetime.datetime.now().replace(microsecond=0).isoformat()
