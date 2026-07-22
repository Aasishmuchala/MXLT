"""MaxGaffer's public API — the LightMatch engine MaxDirector's SPEC deferred to P2.

Import inside 3ds Max (any tool, MaxDirector's pipeline, a listener one-liner):

    from maxgaffer.api import match_camera, apply_camera_state, render_cameras

    result = match_camera("PhysCam_Hero", r"D:/refs/dusk.jpg", log=print)
    # → {"score": 84.2, "stop_reason": "target_reached", "iterations": 4,
    #    "state": {...genome dict...}, "run_dir": "...", "renders": [...]}

Contract notes:
  * main-thread only (it drives pymxs) — callers own threading;
  * per-camera state/reference/locks persist in the scene's session sidecar exactly as if
    driven from the dock, so the UI and API stay interchangeable mid-project;
  * ``config_overrides`` mutate the shared controller's config for the REST OF THE SESSION
    (nothing is written to disk) — pass them once up front. Values are coerced to each
    field's existing type; anything uncoercible raises TypeError immediately.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .core.genome import LightingState
from .maxbridge import config as _config
from .maxbridge.controller import Controller

__all__ = ["match_camera", "match_all_cameras", "apply_camera_state",
           "render_cameras", "export_vrscenes_for_vantage", "get_controller",
           "seed_dome", "scenario_board", "adopt_scenario", "bake_lighting_animation"]

_shared: Optional[Controller] = None


def _coerce_overrides(cfg, overrides: Dict) -> None:
    """Type-sane config overrides: each value is coerced to the config field's existing
    type, or rejected with a clear TypeError. A raw setattr would let e.g. a STRING
    ``loop_width`` into the shared controller, where it only explodes deep inside a
    later render — far from the call that caused it."""
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            continue
        current = getattr(cfg, k)
        try:
            if isinstance(current, bool):          # before int: bool IS an int
                if not isinstance(v, bool):
                    raise TypeError("expected a bool")
                coerced = v
            elif isinstance(current, int):
                coerced = int(v)
            elif isinstance(current, float):
                coerced = float(v)
            elif isinstance(current, str):
                if not isinstance(v, str):
                    raise TypeError("expected a string")
                coerced = v
            elif isinstance(current, dict):
                if not isinstance(v, dict):
                    raise TypeError("expected a dict")
                coerced = v
            else:
                coerced = v
        except (TypeError, ValueError):
            raise TypeError(
                f"config_overrides[{k!r}] must fit the config field's type "
                f"({type(current).__name__}); got {v!r}") from None
        setattr(cfg, k, coerced)


def get_controller(config_overrides: Optional[Dict] = None) -> Controller:
    """One shared Controller (session/rig caches stay warm across API calls)."""
    global _shared
    if _shared is None:
        _shared = Controller(_config.load())
    if config_overrides:
        _coerce_overrides(_shared.cfg, config_overrides)
    return _shared


def match_camera(
    camera_name: str,
    reference_path: str = "",
    log: Callable[[str], None] = lambda m: None,
    should_cancel: Callable[[], bool] = lambda: False,
    locks: Optional[set] = None,
    sweep: bool = True,
    deep: bool = False,
    quality_profile: str = "standard",
    config_overrides: Optional[Dict] = None,
) -> Dict:
    """Bind ``reference_path`` (if given) to ``camera_name`` and run the full match loop.
    ``quality_profile`` accepts ``fast``, ``standard``, or ``hero``. ``deep=True`` remains
    a compatibility alias for Hero mode: target 99 with bounded coordinate-descent polish;
    ``ceiling_converged`` in the result means the remaining gap is content, not lighting."""
    ctrl = get_controller(config_overrides)
    if reference_path:
        ctrl.bind_reference(camera_name, reference_path)
        ctrl._save_or_warn(log)
    result = ctrl.run_match(camera_name, log=log, should_cancel=should_cancel,
                            locks=locks, do_sweep=sweep, deep=deep,
                            quality_profile=quality_profile)
    return {
        "score": result.best_score,
        "stop_reason": result.stop_reason,
        "iterations": len(result.iterations),
        "state": result.best_state.to_dict(),
        "run_dir": ctrl._run_dir,
        "renders": [r.render_path for r in result.iterations if r.render_path],
        "polish_gain": result.polish_gain,
        "ceiling_converged": result.ceiling_converged,
        "ceiling_proven": getattr(result, "ceiling_proven", False),
    }


def match_all_cameras(
    log: Callable[[str], None] = lambda m: None,
    should_cancel: Callable[[], bool] = lambda: False,
    sweep: bool = True,
    config_overrides: Optional[Dict] = None,
) -> Dict[str, str]:
    """Match every camera that has a reference bound (unattended queue)."""
    return get_controller(config_overrides).match_all(log, should_cancel, do_sweep=sweep)


def bake_lighting_animation(camera_name: str, keyframes: Dict[int, Dict],
                            step: int = 1, easing: str = "smooth") -> Dict:
    """Bake sparse ``{frame: LightingState.to_dict()}`` data into native Max keys."""
    sparse = [(int(frame), LightingState.from_dict(state))
              for frame, state in keyframes.items()]
    return get_controller().bake_lighting_animation(camera_name, sparse, step, easing)


def apply_camera_state(camera_name: str) -> List[str]:
    """Re-apply a camera's saved lighting state. Returns warnings."""
    ctrl = get_controller()
    e = ctrl.camera_entry(camera_name)
    if not (e and e.state is not None):
        raise RuntimeError(f"no saved lighting state for camera '{camera_name}'")
    return ctrl.apply_state(e.state, camera_name)


def seed_dome(
    camera_name: str,
    reference_path: str = "",
    log: Callable[[str], None] = lambda m: None,
    config_overrides: Optional[Dict] = None,
) -> Dict:
    """Build a reference-derived HDR pano and bind it to the dome (rotation zeroed;
    previous dome texture snapshotted — restore_pre_match puts it back). → seed meta."""
    ctrl = get_controller(config_overrides)
    if reference_path:
        ctrl.bind_reference(camera_name, reference_path)
        ctrl._save_or_warn(log)
    return ctrl.seed_dome(camera_name, log=log)


def scenario_board(
    camera_name: str,
    log: Callable[[str], None] = lambda m: None,
    should_cancel: Callable[[], bool] = lambda: False,
    config_overrides: Optional[Dict] = None,
) -> List[Dict]:
    """Render + score the candidate rigs (reference optional). The scene is left as it
    was found. → [{key, label, why, state, render, score}] with ``state`` as a plain
    genome DICT (JSON-safe pipeline seam) — feed it straight back to adopt_scenario."""
    results = get_controller(config_overrides).run_scenarios(camera_name, log=log,
                                                             should_cancel=should_cancel)
    return [{**r, "state": r["state"].to_dict()} for r in results]


def _as_state(state) -> LightingState:
    """Accept a LightingState OR its to_dict() form. Anything else raises — a wrong type
    fed to from_dict would 'succeed' as an EMPTY state and silently wipe the camera."""
    if isinstance(state, LightingState):
        return state
    if isinstance(state, dict):
        return LightingState.from_dict(state)
    raise TypeError(f"state must be a LightingState or its to_dict() form, "
                    f"got {type(state).__name__}")


def adopt_scenario(camera_name: str, state,
                   score: Optional[float] = None) -> List[str]:
    """Apply a board candidate (dict from scenario_board, or a LightingState) and save
    it as the camera's state."""
    return get_controller().adopt_scenario(camera_name, _as_state(state), score)


def render_cameras(
    camera_names: List[str],
    out_dir: str,
    on_progress: Callable[[str, str], None] = lambda c, s: None,
    backend: str = "",
) -> Dict[str, str]:
    """Final renders, each camera under its saved matched light. Default backend renders
    through V-Ray in Max (stock Vantage 3.x removed its render CLI); pass
    backend="vantage_cli" only with a Developer-Edition console. BLOCKING, main thread."""
    ctrl = get_controller()
    backend = backend or ctrl.cfg.final_render_backend
    if backend == "vantage_cli":
        jobs = ctrl.prepare_vantage_jobs(camera_names, out_dir, on_progress)
        return ctrl.run_vantage_jobs(jobs, on_progress)
    return ctrl.render_finals_vray(camera_names, out_dir, on_progress)


def export_vrscenes_for_vantage(
    camera_names: List[str],
    on_progress: Callable[[str, str], None] = lambda c, s: None,
) -> List[Dict]:
    """Per-camera .vrscene exports (matched light applied) + open Vantage for its in-app
    Batch Render queue — the Vantage-quality finals path on stock 3.x."""
    jobs, _launched, _dir = get_controller().export_and_open_vantage(
        camera_names, on_progress)
    return jobs
