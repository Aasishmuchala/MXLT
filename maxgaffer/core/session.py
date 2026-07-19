"""Per-camera lighting sessions — every camera can carry its own reference + matched rig.

Archviz reality (the TULA shot-book workflow): each shot wants its own sun. So MaxGaffer's
unit of work is (camera, reference, LightingState, score), persisted in a sidecar JSON next
to the .max file — human-readable, diff-able, survives Max crashes, never bloats the scene.

The bridge owns *when* to apply a camera's state (on select / on render); this owns the data.
Timestamps are injected so tests stay deterministic.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .genome import LightingState

FORMAT_VERSION = 1


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


@dataclass
class CameraEntry:
    reference: str = ""                       # reference image path ("" = none bound)
    state: Optional[LightingState] = None     # last accepted rig for this camera
    score: Optional[float] = None
    matched_at: str = ""
    locks: Set[str] = field(default_factory=set)
    semantics: Dict = field(default_factory=dict)   # cached ANALYZE result for the reference
    pre_match: Optional[LightingState] = None       # the light BEFORE the last match run
    notes: List[str] = field(default_factory=list)  # director's notes, newest last
    seed_hdri: str = ""                             # generated dome-seed .hdr ("" = none)
    pre_seed: Dict = field(default_factory=dict)    # dome texture/rotation before seeding

    def to_dict(self) -> Dict:
        return {
            "reference": self.reference,
            "state": self.state.to_dict() if self.state else None,
            "score": self.score,
            "matched_at": self.matched_at,
            "locks": sorted(self.locks),
            "semantics": self.semantics,
            "pre_match": self.pre_match.to_dict() if self.pre_match else None,
            "notes": list(self.notes),
            "seed_hdri": self.seed_hdri,
            "pre_seed": dict(self.pre_seed),
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
            reference=str(d.get("reference") or ""),
            state=LightingState.from_dict(state) if isinstance(state, dict) else None,
            score=score,
            matched_at=str(d.get("matched_at") or ""),
            locks=set(x for x in _str_list(d.get("locks")) if isinstance(x, str)),
            semantics=d.get("semantics") if isinstance(d.get("semantics"), dict) else {},
            pre_match=LightingState.from_dict(pre) if isinstance(pre, dict) else None,
            notes=[str(x) for x in _str_list(d.get("notes")) if isinstance(x, str)],
            seed_hdri=str(d.get("seed_hdri") or ""),
            pre_seed=d.get("pre_seed") if isinstance(d.get("pre_seed"), dict) else {},
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

    def adopt_baselines(self, fresh: Dict[str, float]) -> List[str]:
        """Adopt baselines for lights we have never seen; NEVER overwrite known ones.
        Returns the names actually adopted."""
        added: List[str] = []
        for name, value in (fresh or {}).items():
            if name not in self.baselines:
                try:
                    self.baselines[str(name)] = float(value)
                    added.append(str(name))
                except (TypeError, ValueError):
                    continue
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
        except (OSError, ValueError):
            return s
        if not isinstance(d, dict):
            return s      # valid JSON but not a session (hand-edit) — start clean
        cameras = d.get("cameras")
        for name, entry in (cameras.items() if isinstance(cameras, dict) else ()):
            if isinstance(entry, dict):
                try:
                    s.cameras[str(name)] = CameraEntry.from_dict(entry)
                except (TypeError, ValueError, AttributeError):
                    continue          # one junk entry must not kill the whole session
        if isinstance(d.get("settings"), dict):
            s.settings.update(d["settings"])
        if isinstance(d.get("baselines"), dict):
            s.adopt_baselines(d["baselines"])
        return s

    def save(self) -> bool:
        if not self.path:
            return False
        payload = {
            "version": FORMAT_VERSION,
            "cameras": {n: e.to_dict() for n, e in self.cameras.items()},
            "settings": self.settings,
            "baselines": self.baselines,
        }
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            os.replace(tmp, self.path)
            return True
        except (OSError, TypeError, ValueError):
            # a failed atomic write must not strand the .tmp (json.dump raises TypeError
            # mid-stream on a non-serializable settings value, OSError on full disks)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False

    # ------------------------------------------------------------------ camera API
    def entry(self, camera: str) -> CameraEntry:
        if camera not in self.cameras:
            self.cameras[camera] = CameraEntry()
        return self.cameras[camera]

    def set_reference(self, camera: str, ref_path: str) -> None:
        e = self.entry(camera)
        if e.reference != ref_path:
            e.reference = ref_path
            e.semantics = {}          # a new reference invalidates the cached analysis
            e.score = None

    def record_match(self, camera: str, state: LightingState,
                     score: Optional[float]) -> None:
        e = self.entry(camera)
        e.state = state.copy()
        e.score = score
        e.matched_at = self._now()

    def cameras_with_states(self) -> List[str]:
        return [n for n, e in self.cameras.items() if e.state is not None]


PRESET_VERSION = 1


def preset_dumps(state: LightingState, name: str = "", now: str = "") -> str:
    """Serialize a lighting state as a portable preset (share across scenes/machines)."""
    return json.dumps({"maxgaffer_preset": PRESET_VERSION, "name": name, "saved_at": now,
                       "state": state.to_dict()}, indent=1)


def preset_loads(text: str) -> Optional[LightingState]:
    """Parse a preset; None if it isn't one. Values re-clamped by the genome on load."""
    try:
        d = json.loads(text)
    except ValueError:
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
