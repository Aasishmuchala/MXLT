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
from typing import Dict, List, Optional, Tuple

from . import config as cfgmod
from .scene import get_prop, set_prop

# (candidates, draft_value) — conservative: quicker convergence, identical lighting.
#
# The V-Ray 7.30 names were MEASURED on-box 2026-07-25 (rt.getPropNames over the live
# renderer, 401 properties) after a benchmark showed only ONE of the original four rows
# matching anything: a 1-minute-per-frame cap on frames that take ~6s, so it never bound
# and the draft sampler was very nearly inert. The real levers are the progressive AND
# bucket noise thresholds (V-Ray uses one or the other depending on imageSampler_type),
# the fine-subdiv count, and the shading rate.
DRAFT_PROPS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    # progressive image sampler noise threshold (default 0.01)
    (("progressive_noise_threshold", "options_progressiveNoiseThreshold",
      "progressive_noiseThreshold", "noise_threshold", "options_dmc_threshold"), 0.05),
    # bucket ("two level") image sampler noise threshold (default 0.01) — the sibling
    # setting for scenes not using the progressive sampler
    (("twoLevel_threshold", "options_dmc_threshold"), 0.05),
    # bucket fine subdivs (default 25)
    (("twoLevel_fineSubdivs", "options_maxSubdivs", "twoLevel_maxSubdivs"), 8),
    # samples per pixel budget (default 6) — the single biggest cost knob
    (("imageSampler_shadingRate",), 2.0),
    (("options_progressiveMaxSubdivs", "progressive_maxSubdivs"), 12),
)

#: Per-frame time cap, applied separately because its value comes from config rather than
#: being fixed. At the old hardcoded 1.0 minutes it never bound on frames taking ~6s, which
#: is why the draft sampler measured nearly inert. It is the one lever that does not scale
#: with scene size, so it is what makes a heavy scene affordable at all.
TIME_CAP_PROPS: Tuple[str, ...] = ("progressive_max_render_time",
                                   "options_progressiveTimeLimit",
                                   "progressive_timeLimit")

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


def apply_draft(seconds: Optional[float] = None) -> List[str]:
    """Snapshot originals to disk FIRST, then apply draft values. Returns log lines.
    A pre-existing snapshot means a crashed session — it is NOT overwritten (the oldest
    snapshot is the true original); we restore it first, then re-apply.

    THE CALLER OWNS THE PROBE CAP. ``seconds`` is the per-probe time budget; None means
    "no caller value" (crash recovery, a bare listener call) and only then is the disk
    config consulted. The gate that decides whether draft mode runs at all lives in the
    controller's in-memory ``cfg``, so re-reading the cap from disk here made two sources
    of truth out of one setting: ``api.get_controller({"probe_max_seconds": …})`` was
    silently ignored, and a Settings edit that failed to persist silently reverted the
    cap while the transcript still said draft was on.

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
    if seconds is None:      # no caller value (crash-recovery, a bare listener call) —
        seconds = getattr(cfgmod.load(), "probe_max_seconds", 0.0)   # fall back to disk
    seconds = float(seconds or 0.0)
    props = list(DRAFT_PROPS)
    cap_names: Tuple[str, ...] = ()
    if seconds > 0:
        props.append((TIME_CAP_PROPS, round(seconds / 60.0, 5)))   # V-Ray states it in MINUTES
        cap_names = TIME_CAP_PROPS
    for candidates, draft_value in props:
        for name in candidates:
            original = get_prop(r, (name,))
            if original is None:
                continue
            value = (type(original)(draft_value) if isinstance(original, int)
                     else float(draft_value))
            if candidates is cap_names and draft_value > 0 and value <= 0:
                # An INT minutes field truncates a 20 s cap to 0 — and V-Ray reads 0 as
                # NO limit, i.e. the exact opposite of what was asked, logged as a
                # cheerful "0 → 0". Refuse loudly rather than uncap the probe. (On the
                # TULA build 2026-07-30 progressive_max_render_time read back as 0.0, a
                # FLOAT, so 20 s binds there — but this varies across V-Ray builds.)
                lines.append(f"draft: {name} is whole MINUTES on this build — a "
                             f"{seconds:g}s cap truncates to 0, which V-Ray reads as no "
                             f"limit; left alone (use probe_max_seconds ≥ 60, or the "
                             f"progressive noise threshold does the work)")
                # CONTINUE, not break. A refused int-typed candidate used to abandon the
                # whole row, so if progressive_max_render_time is int-typed here then
                # options_progressiveTimeLimit was never tried — even on a build where it
                # is a float and would have taken the exact value asked for. (2026-07-31)
                continue
            if candidates is cap_names and draft_value > 0:
                # ROUNDING, not just truncation-to-zero. probe_max_seconds 110 becomes
                # int(1.833) = 1 = a SIXTY-second cap: a 45% shortfall, logged as
                # "0 → 1" with no unit and no mention that the request was rounded — and
                # the "0 →" side is equally opaque, because 0 means "no limit", not "zero
                # minutes". Same class of bug the sub-minute fix was written to kill,
                # moved above the 60 s boundary. (2026-07-31)
                effective = float(value) * 60.0
                if abs(effective - seconds) > 0.5:
                    lines.append(
                        f"draft: {name} stores whole MINUTES on this build — you asked "
                        f"for a {seconds:g}s probe cap and it will bind at "
                        f"{effective:g}s")
            try:
                originals[name] = float(original)
            except (TypeError, ValueError):
                break
            planned.append((name, value))
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
            unit = ""
            if name in TIME_CAP_PROPS:
                # "progressive_max_render_time 0 → 1" says nothing an artist can act on:
                # the unit is missing and 0 means NO LIMIT, not zero minutes.
                unit = (f"  (minutes — was {'no limit' if originals[name] <= 0 else ''}"
                        f"{'' if originals[name] <= 0 else f'{originals[name] * 60:g}s'}"
                        f", now {float(draft_value) * 60.0:g}s per probe)")
            lines.append(f"draft: {name} {originals[name]:g} → {draft_value:g}{unit}")
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
        except (OSError, ValueError, RecursionError):
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
