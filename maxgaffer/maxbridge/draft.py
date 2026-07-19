"""Opt-in draft sampler for match sessions — loop cost is render cost on heavy scenes.

House rule: render setups belong to the artist, so this is engineered around NOT losing
them. Before touching anything, the original values are written to a crash-safe snapshot
file; restore happens in the match's ``finally``; and if Max died mid-match, the snapshot
survives and is auto-restored on the next MaxGaffer launch. Only sampler/noise/time-cap
properties are candidates — never GI or lights, which would change the very lighting
character we're matching.

Property names are candidates-based like everything else (checklist #15): the first
existing property per row is used, missing rows are skipped and reported.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from . import config as cfgmod
from .scene import get_prop, set_prop

# (candidates, draft_value) — conservative: quicker convergence, identical lighting
DRAFT_PROPS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("options_progressiveNoiseThreshold", "progressive_noiseThreshold",
      "noise_threshold", "options_dmc_threshold"), 0.05),
    (("options_progressiveMaxSubdivs", "progressive_maxSubdivs"), 12),
    (("options_progressiveTimeLimit", "progressive_max_render_time",
      "progressive_timeLimit"), 1.0),          # minutes per frame cap
    (("options_maxSubdivs", "twoLevel_maxSubdivs"), 8),
)

SNAPSHOT_PATH = os.path.join(os.path.dirname(cfgmod.CONFIG_PATH), "draft_snapshot.json")


def _rt():
    import pymxs

    return pymxs.runtime


def _renderer():
    try:
        return _rt().renderers.current
    except Exception:
        return None


def pending_snapshot() -> bool:
    """True if a previous session died with draft settings applied (restore needed)."""
    return os.path.exists(SNAPSHOT_PATH)


def apply_draft() -> List[str]:
    """Snapshot originals to disk FIRST, then apply draft values. Returns log lines.
    A pre-existing snapshot means a crashed session — it is NOT overwritten (the oldest
    snapshot is the true original); we restore it first, then re-apply.

    Two passes, strictly ordered: collect every original → write the crash-safe file →
    only THEN mutate the renderer. A crash before the write leaves the scene untouched;
    a crash after it leaves a complete recovery file (the match's ``finally`` or the
    next launch's recovery restores it). If the file can't be written, nothing has been
    touched and draft mode aborts."""
    lines: List[str] = []
    if pending_snapshot():
        lines += restore_draft()
    r = _renderer()
    if r is None:
        return lines + ["draft: no current renderer — skipped"]
    # pass 1 — collect originals, no mutation
    originals: Dict[str, float] = {}
    planned: List[Tuple[str, float]] = []     # (name, coerced draft value)
    for candidates, draft_value in DRAFT_PROPS:
        for name in candidates:
            original = get_prop(r, (name,))
            if original is None:
                continue
            try:
                originals[name] = float(original)
            except (TypeError, ValueError):
                break
            planned.append((name, type(original)(draft_value)
                            if isinstance(original, int) else float(draft_value)))
            break
    if not originals:
        return lines + ["draft: no known sampler properties on this renderer — "
                        "nothing changed (checklist #15)"]
    # crash-safe snapshot BEFORE any set_prop — the module's whole contract
    try:
        with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
            json.dump(originals, f, indent=1)
    except OSError:
        # no safety net → refuse BEFORE touching anything (nothing was mutated yet)
        return lines + ["draft: could not write the safety snapshot — draft mode ABORTED, "
                        "settings untouched"]
    # pass 2 — apply, per-prop fault-isolated; only log a change that actually happened
    for name, draft_value in planned:
        try:
            applied = set_prop(r, (name,), draft_value)
        except Exception:  # noqa: BLE001
            applied = None
        if applied:
            lines.append(f"draft: {name} {originals[name]:g} → {draft_value:g}")
        else:
            # leave the snapshot: it still covers every prop that DID change
            lines.append(f"draft: {name} would not take the draft value — left at "
                         f"{originals[name]:g}")
    return lines


def restore_draft() -> List[str]:
    """Put the snapshotted originals back and delete the snapshot. Safe to call anytime.
    Never raises: one bad value or prop must not strand the rest, and the snapshot file
    is cleared no matter what (a leftover file re-fails recovery at every launch)."""
    if not pending_snapshot():
        return []
    lines: List[str] = []
    try:
        try:
            with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except (OSError, ValueError):
            return ["draft: snapshot unreadable — render settings may need a manual check"]
        r = _renderer()
        if r is not None and isinstance(snapshot, dict):
            for name, value in snapshot.items():
                # per-prop isolation: if the RENDERER changed between crash and relaunch,
                # a prop can be gone (current=None) — type(None)(value) used to raise here,
                # abandoning the remaining restores AND the snapshot file. The value may
                # also be a string (hand-edited file) — format the COERCED float, never
                # the raw entry.
                try:
                    current = get_prop(r, (name,))
                    coerced = (type(current)(float(value)) if isinstance(current, int)
                               else float(value))
                    restored = set_prop(r, (name,), coerced)
                except Exception:  # noqa: BLE001 one lost prop must not strand the rest
                    restored = None
                lines.append(f"draft: restored {name} → {coerced:g}"
                             if restored else f"draft: could not restore {name}")
        return lines
    finally:
        try:
            os.remove(SNAPSHOT_PATH)
        except OSError:
            pass
