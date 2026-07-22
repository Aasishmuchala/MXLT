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

FORMAT_VERSION = 3


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
class RefEntry:
    """One reference image bound to a camera.

    A camera can carry several (a hero angle plus supporting views); the FIRST entry is
    always the PRIMARY and its fields are mirrored onto the legacy single-reference columns
    of :class:`CameraEntry` for backward compatibility.  ``semantics`` / ``score`` are only
    meaningful for the primary today — Route A never feeds the extra views to the solve.
    """

    path: str = ""                            # reference image path
    relative: str = ""                        # portable path relative to the scene folder
    signature: str = ""                       # path + stat identity for in-place changes
    role: str = "primary"                     # "primary" | "angle_N" | artist-supplied label
    semantics: Dict = field(default_factory=dict)   # cached ANALYZE result (primary only)
    score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "relative": self.relative,
            "signature": self.signature,
            "role": self.role,
            "semantics": self.semantics,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RefEntry":
        score_raw = d.get("score")
        try:
            score = float(score_raw) if isinstance(score_raw, (int, float)) else None
            if score is not None and not math.isfinite(score):
                score = None
        except (TypeError, ValueError):
            score = None
        return cls(
            path=str(d.get("path") or ""),
            relative=str(d.get("relative") or ""),
            signature=str(d.get("signature") or ""),
            role=str(d.get("role") or "primary"),
            semantics=d.get("semantics") if isinstance(d.get("semantics"), dict) else {},
            score=score,
        )


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
    # MULTI-REFERENCE (appended last, all existing fields unchanged): references[0] is the
    # PRIMARY. The legacy reference / reference_relative / reference_signature / semantics /
    # score columns ABOVE always mirror references[0]; when references == [] and a legacy
    # reference is set the primary is synthesized on demand (to_dict/_effective_refs) so an
    # in-memory legacy-only entry serializes IDENTICALLY to its post-load form.
    references: List["RefEntry"] = field(default_factory=list)

    # -- primary mirror helpers -----------------------------------------------------------
    def _synth_primary(self) -> "RefEntry":
        """The single primary RefEntry implied by the legacy columns."""
        return RefEntry(path=self.reference, relative=self.reference_relative,
                        signature=self.reference_signature, role="primary",
                        semantics=self.semantics, score=self.score)

    def _effective_refs(self) -> List["RefEntry"]:
        """The reference list this entry serializes as.

        When ``references`` is populated the PRIMARY is re-synced from the authoritative
        legacy columns first — record_match / analyze update the legacy trio+semantics+score,
        not the stored RefEntry — so to_dict stays symmetric with from_dict.  When empty but a
        legacy reference is set, the primary is synthesized on demand; when both empty, [].
        """
        if self.references:
            primary = self.references[0]
            primary.path = self.reference
            primary.relative = self.reference_relative
            primary.signature = self.reference_signature
            primary.role = "primary"
            primary.semantics = self.semantics
            primary.score = self.score
            return self.references
        if self.reference != "":
            return [self._synth_primary()]
        return []

    def _mirror_primary(self) -> None:
        """Re-mirror the legacy trio + semantics + score onto references[0] (or clear when
        the list is empty, reproducing set_reference('')'s '|missing' cleared signature)."""
        if self.references:
            p = self.references[0]
            p.role = "primary"
            self.reference = p.path
            self.reference_relative = p.relative
            self.reference_signature = p.signature
            self.semantics = p.semantics
            self.score = p.score
        else:
            self.reference = ""
            self.reference_relative = ""
            self.reference_signature = reference_signature("")
            self.semantics = {}
            self.score = None

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
            "references": [r.to_dict() for r in self._effective_refs()],
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
        entry = cls(
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
        raw_refs = d.get("references")
        parsed = ([RefEntry.from_dict(r) for r in raw_refs if isinstance(r, dict)]
                  if isinstance(raw_refs, (list, tuple)) else [])
        if parsed:
            entry.references = parsed
            entry._mirror_primary()          # self-heal the legacy trio to references[0]
        elif entry.reference != "":
            entry.references = [entry._synth_primary()]   # legacy / v2 sidecar → 1 primary
        else:
            entry.references = []
        return entry


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
                    if path and loaded.references:
                        for ref in loaded.references:
                            if not ref.relative:
                                continue
                            candidate = os.path.normpath(os.path.join(
                                os.path.dirname(path), ref.relative))
                            # Prefer the project-relative copy after a project folder move.
                            if os.path.isfile(candidate):
                                ref.path = candidate
                                try:
                                    ref.relative = os.path.relpath(
                                        candidate, os.path.dirname(path))
                                except (OSError, ValueError):
                                    pass
                        loaded._mirror_primary()   # re-mirror the FULL trio to references[0]
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

    def _relative_to_sidecar(self, ref_path: str) -> str:
        """Portable path of ``ref_path`` relative to the sidecar folder ("" when not
        expressible as a forward-only relative, or when there is no bound scene)."""
        if ref_path and self.path:
            try:
                rel_candidate = os.path.relpath(ref_path, os.path.dirname(self.path))
                if rel_candidate != os.pardir and not rel_candidate.startswith(os.pardir + os.sep):
                    return rel_candidate
            except (OSError, ValueError):
                pass
        return ""

    def set_reference(self, camera: str, ref_path: str, camera_id: str = "") -> None:
        """Replace ALL references with a single primary (unchanged single-reference API).

        The change-gate compares against the OLD primary path/signature: binding the same
        reference does NOT reset the cached analysis or score; binding a new one (or the same
        path whose file changed in place) does — byte-identical to the previous behavior."""
        e = self.entry(camera, camera_id)
        rel = self._relative_to_sidecar(ref_path)
        signature = reference_signature(ref_path)
        changed = (e.reference != ref_path or e.reference_signature != signature)
        if not changed:
            # Same primary: preserve the legacy trio / semantics / score EXACTLY (a
            # byte-identical no-op for the single-reference case), only collapsing any extra
            # angle references down to the one primary per the replace-all contract.
            if ref_path == "":
                e.references = []
            else:
                e.references = [RefEntry(path=e.reference, relative=e.reference_relative,
                                         signature=e.reference_signature, role="primary",
                                         semantics=e.semantics, score=e.score)]
            return
        if ref_path == "":
            e.references = []
            e.reference = ""
            e.reference_relative = ""
            e.reference_signature = signature      # reference_signature("") == "|missing"
            e.semantics = {}          # a new reference invalidates the cached analysis
            e.score = None
        else:
            e.references = [RefEntry(path=ref_path, relative=rel, signature=signature,
                                     role="primary", semantics={}, score=None)]
            e._mirror_primary()

    def add_reference(self, camera: str, ref_path: str, role: str = "",
                      camera_id: str = "") -> None:
        """Append a supporting reference without disturbing the primary.

        No-op for an empty path or a duplicate signature.  When the camera has no reference
        yet the added one becomes the primary (identical to :meth:`set_reference`)."""
        if ref_path == "":
            return
        e = self.entry(camera, camera_id)
        if not e.references:
            if e.reference != "":
                e.references = [e._synth_primary()]   # materialize the legacy-only primary
            else:
                self.set_reference(camera, ref_path, camera_id)
                return
        signature = reference_signature(ref_path)
        for r in e.references:
            if r.signature == signature:
                return
        e.references.append(RefEntry(path=ref_path,
                                     relative=self._relative_to_sidecar(ref_path),
                                     signature=signature,
                                     role=(role or f"angle_{len(e.references)}")))

    def set_references(self, camera: str, items: List, camera_id: str = "") -> None:
        """Replace the whole reference list; ``items`` are path strings or {"path","role"}.

        The first surviving item is the primary (role forced "primary"); the rest keep their
        given role or default to ``angle_<i>``.  Duplicates (by signature) are dropped, first
        wins.  The primary's cached analysis / score is only reset when the new primary
        path/signature differs from the old primary (same gate as set_reference).  Empty
        ``items`` is equivalent to ``set_reference(camera, "")``."""
        normalized: List = []
        for it in items or []:
            if isinstance(it, dict):
                p = str(it.get("path") or "")
                role = str(it.get("role") or "")
            else:
                p = str(it or "")
                role = ""
            if p != "":
                normalized.append((p, role))
        if not normalized:
            self.set_reference(camera, "", camera_id)
            return
        e = self.entry(camera, camera_id)
        old_primary_path = e.reference
        old_primary_sig = e.reference_signature
        new_refs: List["RefEntry"] = []
        seen: Set[str] = set()
        for i, (p, role) in enumerate(normalized):
            sig = reference_signature(p)
            if sig in seen:
                continue
            seen.add(sig)
            forced_role = "primary" if i == 0 else (role or f"angle_{i}")
            new_refs.append(RefEntry(path=p, relative=self._relative_to_sidecar(p),
                                     signature=sig, role=forced_role))
        primary = new_refs[0]
        if old_primary_path == primary.path and old_primary_sig == primary.signature:
            primary.semantics = e.semantics   # unchanged primary → preserve cached analysis
            primary.score = e.score
        e.references = new_refs
        e._mirror_primary()

    def references(self, camera: str, camera_id: str = "") -> List["RefEntry"]:
        """The resolved camera's references (camera_id-first lookup); [] when unknown.
        Returns the stored list, or the synthesized 1-item primary for a legacy-only entry.
        Read-only — does not create an entry."""
        e = self.find(camera, camera_id)
        if e is None:
            return []
        return list(e._effective_refs())

    def remove_reference(self, camera: str, ref, camera_id: str = "") -> bool:
        """Remove a reference by int index or str path/signature; True if one was removed.
        When the primary (index 0) is removed the next reference becomes primary (or the
        legacy columns clear when the list empties) and the legacy trio re-mirrors."""
        e = self.find(camera, camera_id)
        if e is None:
            return False
        if not e.references and e.reference != "":
            e.references = [e._synth_primary()]   # materialize the legacy-only primary
        if not e.references:
            return False
        idx: Optional[int] = None
        if isinstance(ref, bool):                 # bool is an int subclass — never an index
            return False
        if isinstance(ref, int):
            if 0 <= ref < len(e.references):
                idx = ref
        else:
            target = str(ref)
            for i, r in enumerate(e.references):
                if r.path == target or r.signature == target:
                    idx = i
                    break
        if idx is None:
            return False
        e.references.pop(idx)
        e._mirror_primary()
        return True

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
        """Relink missing references by their portable relative path or unique basename.

        EVERY RefEntry a camera carries is repaired — a hit rewrites its path, signature AND
        portable relative (the old single-reference path did not refresh the relative; the
        per-RefEntry rewrite fixes that) — then the legacy trio is re-mirrored to the primary.
        The returned dict is still keyed camera name -> newly-resolved PRIMARY path (only
        cameras whose primary was relinked appear)."""
        changed: Dict[str, str] = {}
        valid_roots = [os.path.abspath(r) for r in roots if r and os.path.isdir(r)]
        for key, entry in self.cameras.items():
            if not entry.references and entry.reference != "":
                entry.references = [entry._synth_primary()]   # materialize legacy-only primary
            touched = False
            primary_relinked = False
            for idx, ref in enumerate(entry.references):
                if not ref.path or os.path.isfile(ref.path):
                    continue
                candidates: List[str] = []
                for root in valid_roots:
                    if ref.relative:
                        candidates.append(os.path.join(root, ref.relative))
                    basename = os.path.basename(ref.path)
                    candidates.extend(os.path.join(dp, basename)
                                      for dp, _dirs, files in os.walk(root)
                                      if basename in files)
                hits = [os.path.normpath(p) for p in candidates if os.path.isfile(p)]
                hits = list(dict.fromkeys(hits))
                if len(hits) == 1:
                    ref.path = hits[0]
                    ref.signature = reference_signature(hits[0])
                    ref.relative = self._relative_to_sidecar(hits[0]) or ref.relative
                    touched = True
                    if idx == 0:
                        primary_relinked = True
            if touched:
                entry._mirror_primary()
            if primary_relinked:
                changed[entry.camera_name or key] = entry.references[0].path
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
