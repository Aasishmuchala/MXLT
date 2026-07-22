"""Chaos Vantage — live link (the user's real-time monitor) + scene handoff.

VERIFIED against Chaos docs/forums 2026-07-16, for Vantage 3.x:
  * LIVE LINK — installed with V-Ray (5.1+ incl. 7) as the 3ds Max toolbar action
    "Initiate a Live-Link to Chaos Vantage": it STARTS Vantage if needed, streams on port
    20701 (20703 for V-Ray 7.3 DR2), and the SAME action TOGGLES the link off. We probe
    legacy maxscript globals, then scan actionMan for that action text.
  * COMMAND-LINE RENDERING WAS REMOVED in Vantage 2.0+ (Chaos support: "The command line
    control has been removed"; it returned only in the paid Developer Edition). On stock
    Vantage 3.3 there is NO -sceneFile/-outputFile headless render. Therefore:
      - the DEFAULT final-render backend is V-Ray inside Max (fully scriptable);
      - per-camera .vrscene exports remain first-class — drop them into Vantage's in-app
        Batch Render queue (or double-click one) for Vantage-quality finals;
      - the CLI runner below is kept ONLY for Developer-Edition/legacy consoles and is
        gated behind config.final_render_backend == "vantage_cli".
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple

from . import scene as sc

LIVE_LINK_GLOBALS = (
    "vantageStartLiveLink()",
    "startVantageLiveLink()",
    "vrayStartVantageLiveLink()",
    "vray_startVantageLiveLink()",
)

# Candidate state queries — older V-Ray exposes none of these, in which case the link
# state is undetectable and we must report a TOGGLE honestly instead of "started".
LIVE_LINK_PROBES = (
    "vantageLiveLinkActive()",
    "vrayVantageLiveLinkActive()",
    "isVantageLiveLinkActive()",
)

#: Per-job wall-clock limit for one vantage_console still. 20 minutes is generous for a
#: single frame; the old 60-minute budget left a hung console (dead license server, modal
#: on a hidden desktop) un-killable for a full hour per camera.
DEFAULT_JOB_TIMEOUT_S = 20 * 60

#: Files stamped within this slack of the spawn time still count as this run's output
#: (filesystem mtime granularity / clock skew).
_MTIME_SLACK_S = 2.0


def _rt():
    import pymxs

    return pymxs.runtime


def _probe_live_link(rt):
    """Best-effort live-link state read. → True/False when a probe answered, None when
    the state is not detectable (stock V-Ray exposes no documented query)."""
    for expr in LIVE_LINK_PROBES:
        try:
            val = rt.execute(expr)
        except Exception:
            continue
        if isinstance(val, bool):
            return val
    return None


LIVE_LINK_PORTS = (20701, 20703)   # 20701 stock · 20703 on V-Ray 7.3 DR2


def link_running(ports: Tuple[int, ...] = LIVE_LINK_PORTS) -> Optional[int]:
    """The port a live link is streaming on, or None. A listener there means Vantage is
    attached — the only non-destructive way to tell (the V-Ray action is a toggle, so
    asking by executing it would FLIP the state). Pure stdlib socket, safe anywhere."""
    import socket

    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return port
        except OSError:
            continue
    return None


def start_live_link() -> Tuple[bool, str]:
    """Execute V-Ray's 'Initiate a Live-Link to Chaos Vantage' action. That action is a
    TOGGLE (it also stops an active link), so we probe the link state first and never
    claim "started" without a state read. → (executed?, how/diagnostic). Degrades off-Max."""
    try:
        rt = _rt()
    except Exception:
        return False, "pymxs unavailable (not running inside 3ds Max)"
    state = _probe_live_link(rt)
    if state is True:
        return True, "live link already active — left untouched"
    fire = None
    for expr in LIVE_LINK_GLOBALS:
        try:
            rt.execute(expr)
            fire = f"maxscript global {expr}"
            break
        except Exception:
            continue
    if fire is None:
        hit = _find_live_link_action()
        if hit is not None:
            action, label = hit
            try:
                action.execute()
                fire = f"actionMan: {label}"
            except Exception as e:  # noqa: BLE001
                return False, f"found action '{label}' but execute failed: {e}"
    if fire is None:
        return False, ("no live-link entry point found — start it once via the V-Ray menu "
                       "(Chaos Vantage live link); the link then mirrors everything MaxGaffer does")
    after = _probe_live_link(rt)
    if state is False or after is True:
        return True, f"started via {fire}"
    if after is False:
        return True, f"toggled via {fire} — a link was already active and is now OFF"
    return True, (f"toggled via {fire} (link state not detectable — the V-Ray action is "
                  "a toggle: if a link was already running, it is now off)")


def _find_live_link_action():
    """Scan actionMan for an action whose text mentions Vantage. Returns (action, label)."""
    rt = _rt()
    try:
        num_tables = int(rt.actionMan.numActionTables)
    except Exception:
        return None
    for t in range(1, num_tables + 1):
        try:
            table = rt.actionMan.getActionTable(t)
            for a in range(1, int(table.numActions) + 1):
                action = table.getAction(a)
                label = ""
                for getter in ("getDescriptionText", "getButtonText", "getMenuText"):
                    try:
                        label = str(getattr(action, getter)())
                        if label:
                            break
                    except Exception:
                        continue
                low = label.lower()
                if "vantage" in low and ("live" in low or "link" in low):
                    return action, label
        except Exception:
            continue
    return None


def export_vrscene(path: str, camera_name: Optional[str] = None) -> Optional[str]:
    """Export the current scene (single frame, active camera) as .vrscene. The viewport's
    previously active camera is restored afterwards — a batch export must not leave the
    user's viewport parked on the last exported camera."""
    rt = _rt()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prev_cam = None
    try:
        prev_cam = rt.viewport.getCamera()
    except Exception:
        prev_cam = None
    try:
        if camera_name and not sc.set_active_camera(camera_name):
            return None
        if not hasattr(rt, "vrayExportVRScene"):
            return None
        try:
            frame = int(rt.currentTime.frame) if hasattr(rt.currentTime, "frame") else 0
        except Exception:
            frame = 0
        try:  # rich signature first (compressed keeps multi-GB interiors manageable)
            rt.vrayExportVRScene(path, exportCompressed=True, startFrame=frame, endFrame=frame)
        except Exception:
            try:
                rt.vrayExportVRScene(path)
            except Exception:
                return None
        return path if os.path.exists(path) else None
    finally:
        if prev_cam is not None:
            try:
                rt.viewport.setCamera(prev_cam)
            except Exception:
                pass


def _output_written(output: str, min_mtime: Optional[float] = None) -> bool:
    """Vantage may write frame-suffixed files (out.0000.png) instead of the exact name —
    accept the exact file or a ``stem.<digits><ext>`` sibling (a bare startswith would
    also match "Shot10.png" for stem "Shot1"). With ``min_mtime``, only files modified at
    or after that timestamp count, so a stale file from a previous batch can never pass."""
    def fresh(p: str) -> bool:
        try:
            return (os.path.getsize(p) > 0
                    and (min_mtime is None or os.path.getmtime(p) >= min_mtime))
        except OSError:
            return False

    if os.path.exists(output) and fresh(output):
        return True
    folder = os.path.dirname(output) or "."
    base = os.path.basename(output)
    stem, ext = os.path.splitext(base)
    try:
        for f in os.listdir(folder):
            if f == base or not f.startswith(stem + "."):
                continue
            num, tail = os.path.splitext(f[len(stem):])   # ".0000.png" → ".0000", ".png"
            if num[1:].isdigit() and tail.lower() == ext.lower() \
                    and fresh(os.path.join(folder, f)):
                return True
        return False
    except OSError:
        return False


def vantage_command(console_exe: str, scene_file: str, output: str,
                    width: int, height: int, frame: int = 0) -> List[str]:
    return [
        console_exe,
        f"-scenefile={scene_file}",
        f"-outputFile={output}",
        f"-outputWidth={int(width)}",
        f"-outputHeight={int(height)}",
        f"-frames={int(frame)}-{int(frame)}",
    ]


def launch_vantage(vantage_exe: str, scene_file: Optional[str] = None) -> bool:
    """Open Vantage (optionally on a vrscene) — the handoff for the in-app batch queue."""
    if not os.path.exists(vantage_exe):
        return False
    try:
        args = [vantage_exe] + ([scene_file] if scene_file else [])
        subprocess.Popen(args, close_fds=True)
        return True
    except Exception:
        return False


def write_queue_manifest(export_dir: str, jobs: List[Dict],
                         width: int, height: int) -> Optional[str]:
    """Write an ordered, machine-readable handoff for stock Vantage's in-app queue."""
    if not export_dir:
        return None
    try:
        os.makedirs(export_dir, exist_ok=True)
    except OSError:
        return None
    manifest = os.path.join(export_dir, "vantage_queue.json")
    payload = {
        "format": 1,
        "renderer": "Chaos Vantage in-app Batch Render",
        "requires_in_app_queue": True,
        "resolution": {"width": int(width), "height": int(height)},
        "jobs": [{"order": index + 1,
                  "camera": str(job.get("camera") or ""),
                  "scene_file": os.path.abspath(str(job.get("scene_file") or "")),
                  "output": os.path.abspath(str(job.get("output") or "")),
                  "status": "ready_for_vantage_queue"}
                 for index, job in enumerate(jobs)],
    }
    try:
        with open(manifest, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        instructions = os.path.join(export_dir, "VANTAGE_BATCH_INSTRUCTIONS.txt")
        with open(instructions, "w", encoding="utf-8") as stream:
            stream.write(
                "MaxGaffer prepared these scenes in render order.\n"
                "In Chaos Vantage, open Batch Render, add the .vrscene files in the "
                "manifest order, set the listed resolution/output paths, and start.\n"
                "Stock Vantage 3.x does not expose this queue as a normal headless API.\n")
        return manifest
    except OSError:
        return None


def _run_console(cmd: List[str], timeout_s: float,
                 should_cancel: Optional[Callable[[], bool]]) -> Tuple[str, Optional[int]]:
    """Run one vantage_console job, polling so a cancel or timeout actually kills the
    child (output is unused and goes to DEVNULL — unread pipes could deadlock a chatty
    console). → (status, returncode): status ∈ "done" | "timeout" | "cancelled"."""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + float(timeout_s)
    status = "done"
    while proc.poll() is None:
        if should_cancel is not None and should_cancel():
            status = "cancelled"
            break
        if time.monotonic() > deadline:
            status = "timeout"
            break
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            pass
    return status, proc.returncode


def render_stills(
    jobs: List[Dict],                      # {camera, scene_file, output}
    console_exe: str,
    width: int,
    height: int,
    on_progress: Optional[Callable[[str, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
) -> Dict[str, str]:
    """LEGACY/Developer-Edition ONLY: sequential vantage_console CLI batch. Stock Vantage
    2.0+ removed these flags — use the V-Ray backend or the in-app batch queue instead.

    Every job gets a result entry — a failed OR MALFORMED job never abandons the rest
    of the batch (one bad shot must not cost the rest of the night). ``should_cancel``
    (a pollable callable, e.g. ``threading.Event.is_set``) is checked between jobs and
    while a console runs, and kills the running child. Each job is capped at
    ``timeout_s`` seconds (default DEFAULT_JOB_TIMEOUT_S = 20 min — a hung console is
    killed, not waited on for an hour). A pre-existing exact output is deleted before
    launch and the post-run check requires a fresh mtime, so a stale render from a
    previous batch is never accepted as this run's."""
    def _cam(job, idx) -> str:
        if isinstance(job, dict) and job.get("camera"):
            return str(job["camera"])
        return f"job{idx}"

    results: Dict[str, str] = {}
    if not os.path.exists(console_exe):
        return {_cam(j, i): f"vantage_console not found: {console_exe}"
                for i, j in enumerate(jobs)}
    for i, job in enumerate(jobs):
        cam = _cam(job, i)
        if should_cancel is not None and should_cancel():
            results[cam] = "cancelled"
            if on_progress:
                on_progress(cam, results[cam])
            continue
        if on_progress:
            on_progress(cam, "rendering (vantage)")
        spawned = time.time()
        try:
            out = job["output"]             # inside the try — a malformed job dict is
            try:                            # an error entry, not a dead batch; never
                if os.path.exists(out):     # accept a stale file as this run's
                    os.remove(out)
            except OSError:
                pass
            cmd = vantage_command(console_exe, job["scene_file"], out, width, height)
            status, rc = _run_console(cmd, timeout_s, should_cancel)
            if status == "cancelled":
                results[cam] = "cancelled"
            elif status == "timeout":
                results[cam] = f"timeout after {int(timeout_s)}s"
            elif rc == 0 and _output_written(out, min_mtime=spawned - _MTIME_SLACK_S):
                results[cam] = "ok"
            elif rc == 0:
                results[cam] = "vantage exit 0 but no output written"
            else:
                results[cam] = f"vantage exit {rc}"
        except Exception as e:  # noqa: BLE001 one bad job must not kill the batch
            results[cam] = f"error: {e}"
        if on_progress:
            on_progress(cam, "done" if results[cam] == "ok" else results[cam])
    return results
