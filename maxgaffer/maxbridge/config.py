"""Settings + the oc_ key, stored at %LOCALAPPDATA%/MaxGaffer/config.json.

Stdlib-only (importable off-Max, same as MaxDirector's). If MaxGaffer has no key yet but a
MaxDirector install does, we borrow it silently — same gateway, same owner, one less paste.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


def _warn(msg: str) -> None:
    """Config problems must be LOUD (Max listener / console) but never fatal."""
    try:
        print("[MaxGaffer] config: " + msg)
    except Exception:
        pass


def _ensure_dir(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        _warn(f"could not create {d} ({e})")


def _appdata_dir(name: str, create: bool = False) -> str:
    """Path only by default — directory creation is deferred to first use. Importing this
    module must never touch the disk (an unwritable profile, or a FILE named 'MaxGaffer'
    in %LOCALAPPDATA%, would otherwise kill the whole plugin load at import time)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, name)
    if create:
        _ensure_dir(d)
    return d


CONFIG_PATH = os.path.join(_appdata_dir("MaxGaffer"), "config.json")


@dataclass
class Config:
    api_key: str = ""                        # oc_ gateway key
    model: str = "claude-opus-4-8"           # vision-capable — the loop shows it images
    semantic_provider: str = "omega"          # omega | anthropic | openai | openai_compatible | offline
    # blank = each provider's shipped default. For "omega" this overrides the Omega Plus
    # endpoint and accepts either docs form (".../v1" or ".../v1/messages"), so a gateway
    # move is a Settings edit, not a code change.
    semantic_base_url: str = ""
    vantage_exe: str = r"C:\Program Files\Chaos\Vantage\vantage.exe"
    # Vantage 3.x REMOVED stock command-line rendering (Chaos support-confirmed; it now
    # needs the Developer Edition). Default backend renders finals through V-Ray in Max —
    # fully scriptable tonight. "vantage_cli" only works with a Dev Edition console exe.
    final_render_backend: str = "vray"       # "vray" | "vantage_cli"
    vantage_console: str = r"C:\Program Files\Chaos\Vantage\vantage_console.exe"
    # after a match, also capture the lighting as a native Max Scene State
    # (Light Properties + Light Transforms + Environment) so each camera's look
    # lives in the .max file and restores from Tools > Manage Scene States
    capture_scene_states: bool = True
    auto_exposure_control: bool = True       # create a V-Ray exposure control if none exists
    system_python: str = ""                  # optional Pillow-equipped python for the sidecar
    loop_width: int = 480                    # iteration-render size (speed over beauty)
    loop_height: int = 270
    final_width: int = 1920
    final_height: int = 1080
    max_iterations: int = 5
    # 82 meant a Standard match declared victory well below a usable match and
    # stopped; the bar people actually want is 95. Fast still caps itself lower.
    target_score: float = 95.0
    analyze_samples: int = 3                 # ANALYZE self-consistency (1 = single-shot)
    sweep_count: int = 8
    seed_blur_passes: int = 0                # 0 sharp reflections; 1-2 diffuse-light blur
    keep_runs: int = 10                      # run folders kept per camera (0 = keep all)
    # STAYS OPT-IN — render setups belong to the artist (draft.py's house rule), so this
    # is never switched on behind their back even though it is snapshot-protected.
    # It is, however, the single biggest speed lever: a match's cost IS its render cost
    # (measured on-box 2026-07-25 at ~10 s/frame with up to 500 polish probes). Turn it on
    # in Settings for heavy scenes; V-Ray GPU was the fastest renderer measured.
    draft_sampler: bool = False
    #: Seconds a single PROBE render may take. 0 leaves V-Ray's own settings alone.
    #:
    #: This is the only knob whose cost does not scale with the scene. Resolution and
    #: sample counts all scale WITH it — a ten-times-heavier scene is still ten times
    #: slower after halving resolution — so on a big scene a match is priced by V-Ray's
    #: per-frame time and nothing else: measured, plugin overhead is about zero, and 180
    #: probes at 60s a frame is three hours. A time budget inverts that. "Render for four
    #: seconds and give me what you have" costs four seconds whether the scene is a teapot
    #: or eighteen million triangles; what changes is how noisy the frame is, and noise is
    #: what a probe can afford, because it needs a RANKING and not an accurate picture.
    #:
    #: Binds only under the PROGRESSIVE image sampler — the bucket sampler has no
    #: equivalent, and apply_draft says so rather than failing quietly.
    probe_max_seconds: float = 0.0
    # apply-only mode: MaxGaffer never fires a render. MATCH = analyze → first guess →
    # apply as ONE undoable change → read-back verification → change report. The loop,
    # sun sweep, board probes, plan effect measurement and V-Ray finals are all off.
    no_renders: bool = False
    # software exposure: V-Ray GPU applies exposure only at the VFB display stage, so
    # loop renders don't reflect the EV/WB the solver sets (the host is inert in the
    # saved buffer). When on, EV/WB are applied to each loop frame in software before
    # scoring, so the analytic solver converges on any renderer. Recommended ON for
    # V-Ray GPU; harmless where exposure already bakes in (near-identity early frames).
    software_exposure: bool = False
    plan_first: bool = True                  # scene-wide plan (any setting, create lights)
    auto_execute_plan: bool = False          # skip the preview dialog (still one undo)
    show_report_popup: bool = True           # "scene changed" popup after execution
    # "dim" is the DOC-BACKED default: VRaySky auto-binds to "the first enabled VRaySun"
    # (Chaos docs/forums), so disabling the sun can gut a VRaySky environment. "disable"
    # remains available for dome-only rigs.
    overcast_sun_mode: str = "dim"
    critic_weights: Dict[str, float] = field(default_factory=dict)   # override critic defaults
    artist_preference: str = "balanced"       # balanced | direction | color_mood | tonal
    repo_path: str = ""                      # clone folder, written by install.bat

    def save(self) -> None:
        _ensure_dir(os.path.dirname(CONFIG_PATH))
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=1)


def _type_ok(value: Any, default: Any) -> Tuple[bool, Any]:
    """Check a loaded JSON value against the dataclass default's type. Returns
    (accepted, coerced-value). bool is NOT an int here; an int IS accepted for a float
    field (JSON has one number type) and widened."""
    t = type(default)
    if t is bool:
        return type(value) is bool, value
    if t is int:
        return type(value) is int, value
    if t is float:
        ok = type(value) in (int, float)
        return ok, float(value) if ok else default
    if t is str:
        return isinstance(value, str), value
    if t is dict:
        return isinstance(value, dict), value
    return isinstance(value, t), value


def load() -> Config:
    cfg = Config()
    defaults = Config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            # valid JSON but not an object (null, [], "str", 42 — hand-edit, crashed sync
            # tool, version downgrade): d.items() would raise AttributeError, which the
            # except below deliberately does NOT cover. Treat as empty, loudly.
            _warn(f"{CONFIG_PATH} holds {type(d).__name__}, not an object — "
                  "ignoring it, using defaults")
        else:
            for k, v in d.items():
                if hasattr(cfg, k):
                    ok, vv = _type_ok(v, getattr(defaults, k))
                    if ok:
                        setattr(cfg, k, vv)
                    else:
                        # wrong-typed values surface as TypeErrors three modules away
                        # (dock slots, critic weights) — reject them AT THE SOURCE
                        _warn(f"'{k}' is {type(v).__name__}, expected "
                              f"{type(getattr(defaults, k)).__name__} — keeping default "
                              f"{getattr(defaults, k)!r}")
    except (OSError, ValueError, RecursionError):
        pass
    if not cfg.api_key:
        cfg.api_key = _borrow_maxdirector_key()
    return cfg


def _borrow_maxdirector_key() -> str:
    try:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        with open(os.path.join(base, "MaxDirector", "config.json"), encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return str(d.get("api_key") or "")
        _warn("MaxDirector config.json is not an object — no key borrowed")
    except (OSError, ValueError, RecursionError):
        pass
    return ""


def sessions_dir() -> str:
    d = os.path.join(_appdata_dir("MaxGaffer"), "sessions")
    _ensure_dir(d)
    return d
