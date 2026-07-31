"""Preflight — catch in milliseconds what used to cost a day, BEFORE spending anything.

Written 2026-07-31, straight off the TULA post-mortem of 2026-07-30. That run lost an
artist an entire day to a sequence of individually small silent failures, and the two
worst of them were both visible in the scene before the first render:

  * a dead Chaos Vantage live link had left V-Ray's ``distributed_rendering`` ON, pointed
    at 127.0.0.1:20701 with the local machine excluded, so V-Ray rendered 100% BLACK.
    That is a property read and one localhost socket connect — about two milliseconds;
  * of 12 per-shot VRaySuns the only ENABLED one was authored for a different shot at the
    other end of the site, and the rig's first-enabled-wins pick steered it for hours.
    ``classify_rig`` never reads ``LIGHT_ON`` at all, so nothing knew.

Contract, and it is the whole reason this module is safe to run on every match:

  * NO RENDERS and NO MUTATIONS. ``EXPOSURE_ROUNDTRIP`` is the single exception — it
    writes one EV and puts it back in a ``finally`` — and ``VANTAGE_ARMED`` grabs a window
    (a read of somebody else's pixels, not a change to this scene).
  * every check is fault-isolated through ``_check``: one that RAISES reports
    "could not verify", never "ok". That rule is ``vgrab._occluded_fraction``'s own and it
    is the difference between a gate and a decoration.
  * every check reports on BOTH outcomes. A gate that says nothing when it passes is what
    29bbae6 was written to kill; a gate that says nothing when it CLOSES is the same
    failure wearing a different hat. ``info`` findings collapse into one summary line so
    the transcript stays readable; every ``warn`` and ``block`` gets its own line, and all
    of them go into run.json.

BLOCK vs WARN, stated once: BLOCK only where continuing would produce a number that is
not a measurement. Six conditions — reference missing / undecodable / 100% black, DR on
with no listener, run dir unwritable in both locations, and (in the canary) no file /
black-with-a-named-cause / wrong size. EVERYTHING else warns, including every sun↔camera
finding and a non-V-Ray renderer: naming conventions are conventions, and a project that
names its suns by mood must not be locked out of its own tool.
"""

from __future__ import annotations

import os
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core import metrics
from ..core.errors import PreflightBlocked
from . import scene as sc
from . import vantage as vt

#: Renderer property rows, checklist #15 shape (first hit wins, a missing row is reported
#: rather than raised). The distributed-rendering spellings were taken from
#: controller._black_probe_message, which already reads the first of them.
DR_ON_PROPS = ("distributed_rendering", "system_distributedRendering",
               "distributed_rendering_on")
DR_HOSTS_PROPS = ("distributed_rendering_hosts", "system_distributedRenderingHosts",
                  "dr_hosts", "hosts")
DR_LOCAL_PROPS = ("distributed_rendering_useLocalMachine", "use_local_machine",
                  "dr_useLocalMachine")
NO_FINAL_IMAGE_PROPS = ("options_dontRenderImage", "options_dontRenderFinalImage",
                        "system_dontRenderFinalImage")

#: V-Ray's DR spawner port. A host written WITHOUT one ("192.168.1.50") means "the DR
#: spawner's default", which is this — NOT vt.LIVE_LINK_PORTS[0]. That was the 2026-07-31
#: bug: DR_DEAD_PORT is a BLOCK, so probing a healthy farm on Vantage's live-link port
#: found nothing listening and locked every DR studio out of its own tool. 20701 belongs
#: only to the case where the host list is UNREADABLE, because that is where a dead
#: Vantage link leaves it pointed.
VRAY_DR_PORT = 20204

#: A refused TCP connection on localhost returns ECONNREFUSED in well under a
#: millisecond; only a FILTERED host pays this. Worst case 0.25 s × n_hosts, and a dead DR
#: host is exactly the thing worth a quarter of a second.
DR_CONNECT_TIMEOUT_S = 0.25

#: A REMOTE node has to complete a real TCP handshake across the LAN, and a loaded render
#: node answering in 300 ms is not a dead one. 0.25 s would have read it as dead and
#: blocked the run — the same false positive, one layer down. (2026-07-31)
DR_REMOTE_TIMEOUT_S = 1.0
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def _dr_timeout(host: str) -> float:
    return (DR_CONNECT_TIMEOUT_S if str(host).strip().lower() in LOOPBACK_HOSTS
            else DR_REMOTE_TIMEOUT_S)

#: Tokens that carry no identity — they appear in every sun and every camera name in an
#: archviz scene and would inflate every overlap score toward 1.0.
#: "shot" is deliberately NOT here: on TULA it is the token that carries the shot NUMBER's
#: context, and dropping it would make "SUN SHOT 01 ENTRANCE" score 0.00 against
#: "SCENE 04 shot02" instead of the honest 0.25.
NAME_STOPWORDS = frozenset({"sun", "vraysun", "camera", "cam", "vray", "light",
                            "the", "of"})

#: Propose a different sun ONLY when one scores strictly higher than the driven sun AND
#: clears this. Against the real 2026-07-30 names — camera "SCENE 04 shot02" — the
#: enabled sun "SUN SHOT 01 ENTRANCE" scores 0.25 and "SUN SCENE 04 shot02" would score
#: 1.00. On TULA nothing clears 0.5, so nothing is proposed and the check reports the bare
#: fact instead. An unrelated naming scheme produces all-zero scores and no strict winner,
#: which is why this cannot false-positive on a project that names suns by mood.
NAME_MATCH_PROPOSE = 0.5

#: A reference whose 5th-95th percentile span is under this is flat — a scan of a grey
#: card, a blown-out phone snap, a screenshot of a solid colour. Not a block: an artist
#: may legitimately be matching a fog plate.
REF_FLAT_CONTRAST = 0.02


@dataclass(frozen=True)
class Finding:
    key: str
    severity: str                    # "block" | "warn" | "info"
    detail: str                      # the artist-facing sentence, house voice
    spelling: str = ""               # WHICH candidates row answered, or "" for none
    elapsed_ms: float = 0.0
    #: False when the check RAISED (or returned nothing) and ``_check`` substituted a
    #: "could not verify" line. It is severity "info" so it does not shout, but "info" and
    #: "verified ok" are not the same claim and something that ARMS a fast path off a
    #: finding has to be able to tell them apart. (2026-07-31)
    verified: bool = True

    def to_json(self) -> Dict:
        return {"key": self.key, "severity": self.severity, "detail": self.detail,
                "spelling": self.spelling, "elapsed_ms": round(self.elapsed_ms, 2),
                "verified": self.verified}


@dataclass
class PreflightReport:
    findings: List[Finding] = field(default_factory=list)
    elapsed_ms: float = 0.0
    #: checks that asked the run to change course without stopping it (e.g. the Vantage
    #: backend demoting itself to V-Ray) — the controller reads these, never guesses
    demotions: Dict[str, Any] = field(default_factory=dict)

    def blocked(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "block"]

    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    def clean(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "info"]

    def find(self, key: str) -> Optional[Finding]:
        for f in self.findings:
            if f.key == key:
                return f
        return None

    def to_json(self) -> Dict:
        return {"elapsed_ms": round(self.elapsed_ms, 2),
                "findings": [f.to_json() for f in self.findings],
                "demotions": dict(self.demotions)}


@dataclass
class Context:
    """Everything the checks may read. Assembled once; the checks never reach past it."""
    ctrl: Any
    camera_name: str
    cam: Any = None
    rig: Dict = field(default_factory=dict)
    entry: Any = None
    cfg: Any = None
    log: Callable[[str], None] = lambda _m: None
    ref_path: str = ""
    ref_stats: Optional[Dict] = None
    renderer: Any = None
    run_dir: str = ""
    #: cached once per run so vt.link_running's two 1-second socket waits are not paid
    #: twice on the main thread (controller.py used to call it at :991 AND :786)
    link_port: Optional[int] = None
    link_checked: bool = False

    def live_link_port(self) -> Optional[int]:
        if not self.link_checked:
            self.link_checked = True
            try:
                self.link_port = vt.link_running()
            except Exception:  # noqa: BLE001
                self.link_port = None
        return self.link_port


# --------------------------------------------------------------------------- name tokens
_SPLIT = re.compile(r"[^0-9a-z]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")


def name_tokens(name: str) -> set:
    """Lowercase identity tokens of a node name.

    Splits on non-alphanumerics AND on letter↔digit boundaries, so ``shot02`` becomes
    ``{shot, 2}`` and matches ``SHOT 2``; leading zeros are stripped so ``04`` == ``4``.
    Pure — the whole sun↔camera name comparison is unit-testable off-Max.
    """
    out = set()
    for tok in _SPLIT.split(str(name or "").lower()):
        if not tok:
            continue
        if tok.isdigit():
            tok = str(int(tok))
        if tok in NAME_STOPWORDS:
            continue
        out.add(tok)
    return out


def name_overlap(sun_name: str, camera_name: str) -> float:
    """How much of the CAMERA's name this sun's name accounts for. 0..1, relative.

    Deliberately relative and never absolute: an absolute token count would rank a sun
    with a long name above a correctly-matched short one. Normalising on the camera means
    "does this sun name this camera", which is the actual question.
    """
    cam_toks = name_tokens(camera_name)
    if not cam_toks:
        return 0.0
    return len(name_tokens(sun_name) & cam_toks) / float(len(cam_toks))


# --------------------------------------------------------------------------- the checks
def _node_name(node) -> str:
    try:
        return str(node.name)
    except Exception:  # noqa: BLE001
        return "<unnamed>"


def check_reference_present(ctx: Context) -> Finding:
    from ..core.session import reference_signature

    path = ctx.ref_path
    if not path:
        return Finding("REFERENCE_PRESENT", "block",
                       "no reference image is bound to this camera — there is nothing to "
                       "match against")
    sig = reference_signature(path)
    if sig.endswith("|missing") or not os.path.exists(path):
        return Finding("REFERENCE_PRESENT", "block",
                       f"the reference file is gone: {path} — re-bind it via "
                       "Load reference… (nothing has been spent yet)")
    return Finding("REFERENCE_PRESENT", "info", f"reference on disk: {path}")


def check_reference_readable(ctx: Context) -> Finding:
    if not ctx.ref_path or not os.path.exists(ctx.ref_path):
        return Finding("REFERENCE_READABLE", "info",
                       "not checked — the reference file is missing (see "
                       "REFERENCE_PRESENT)")
    if ctx.ref_stats is None:
        # WARN, not BLOCK — a deviation from the plan, and a deliberate one. run_match
        # has a DESIGNED degraded mode for exactly this: ref_stats None turns the
        # analytic EV/WB solver off and runs LLM-visual, judging the render against the
        # reference IMAGE through the model rather than through pixel statistics. Blocking
        # here would delete a supported feature rather than prevent a non-measurement. The
        # truly fatal case — the reference cannot be prepared for the model either — is
        # already a hard raise a few lines later in run_match. (2026-07-31)
        return Finding("REFERENCE_READABLE", "warn",
                       f"the reference could not be decoded by any of the three stats "
                       f"readers (in-process, sidecar python, Max transcode): "
                       f"{ctx.ref_path}. The analytic EV/WB solver and every critic score "
                       "are OFF for this run — it will be judged visually by the model "
                       "only. Install Pillow or set system_python to get them back.")
    return Finding("REFERENCE_READABLE", "info",
                   f"reference decoded ({int(ctx.ref_stats.get('count') or 0)} sampled "
                   "pixels)")


def check_reference_degenerate(ctx: Context) -> Finding:
    if ctx.ref_stats is None:
        return Finding("REFERENCE_DEGENERATE", "info",
                       "not checked — the reference could not be decoded")
    if metrics.is_black(ctx.ref_stats):
        return Finding("REFERENCE_DEGENERATE", "block",
                       "the reference image is 100% BLACK — matching to it would rank "
                       "whichever render is darkest, which is not a lighting match")
    try:
        contrast = float(ctx.ref_stats.get("contrast", 1.0))
    except (TypeError, ValueError):
        contrast = 1.0
    if contrast < REF_FLAT_CONTRAST:
        return Finding("REFERENCE_DEGENERATE", "warn",
                       f"the reference is almost FLAT (5th-95th percentile span "
                       f"{contrast:.3f}) — it carries very little for the critic to steer "
                       "on, so expect a confident score that means little")
    return Finding("REFERENCE_DEGENERATE", "info",
                   f"reference has usable structure (span {contrast:.3f})")


def check_reference_has_sun(ctx: Context) -> Finding:
    if ctx.ref_stats is None:
        return Finding("REFERENCE_HAS_SUN", "info",
                       "not checked — the reference could not be decoded")
    try:
        hot = float(ctx.ref_stats.get("hot_frac") or 0.0)
    except (TypeError, ValueError):
        hot = 0.0
    if hot <= 0.0:
        # PREDICT the skipped solve instead of discovering it 44 probes later
        # (sunsolve.py:90-96 returns None on exactly this condition).
        return Finding("REFERENCE_HAS_SUN", "warn",
                       "the reference carries no bright directional patch at all — the "
                       "sun solve will be SKIPPED (there is nothing to place a sun "
                       "toward), and sun-patch agreement will not be a usable signal")
    return Finding("REFERENCE_HAS_SUN", "info",
                   f"reference carries directional light ({hot:.1%} of frame in patches)")


def _parse_dr_hosts(raw) -> List[Tuple[str, int]]:
    """``ip:port`` pairs out of whatever the DR host property hands back.

    V-Ray spells this as a string, a string array, or a MAXScript array wrapper depending
    on build, so this parses defensively and falls back to the caller's default ports.
    """
    out: List[Tuple[str, int]] = []
    items: List[str] = []
    if raw is None:
        return out
    if isinstance(raw, str):
        items = re.split(r"[\s,;]+", raw)
    else:
        try:
            items = [str(v) for v in raw]
        except TypeError:
            items = [str(raw)]
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        m = re.match(r"^\[?([0-9a-zA-Z_.\-]+)\]?(?::(\d+))?$", item)
        if not m:
            continue
        host = m.group(1)
        port = int(m.group(2)) if m.group(2) else VRAY_DR_PORT
        out.append((host, port))
    return out


def check_dr_dead_port(ctx: Context) -> Finding:
    """The check that is the whole point. Two milliseconds; it is 2026-07-30 entire."""
    r = ctx.renderer
    if r is None:
        return Finding("DR_DEAD_PORT", "info",
                       "no current renderer to interrogate — distributed rendering "
                       "could not be verified")
    spelling = sc.matched_prop(r, DR_ON_PROPS) or ""
    dr = sc.get_prop(r, DR_ON_PROPS)
    if not dr:
        return Finding("DR_DEAD_PORT", "info",
                       "distributed rendering is off" if spelling else
                       "this renderer exposes no distributed-rendering flag",
                       spelling=spelling)
    hosts_raw = sc.get_prop(r, DR_HOSTS_PROPS)
    local = sc.get_prop(r, DR_LOCAL_PROPS)
    hosts = _parse_dr_hosts(hosts_raw)
    if not hosts:
        # unreadable host list: probe the live-link ports, which is where a dead Vantage
        # link leaves it pointed (plugcfg/vrayrt_dr.cfg's 'restore' path writes it back),
        # AND the DR spawner's own port, so a healthy local spawner answers rather than
        # being blocked by a check that only knew about Vantage's ports (2026-07-31)
        hosts = [("127.0.0.1", p) for p in (VRAY_DR_PORT,) + tuple(vt.LIVE_LINK_PORTS)]
    alive: List[str] = []
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=_dr_timeout(host)):
                alive.append(f"{host}:{port}")
        except OSError:
            continue
    if alive:
        return Finding("DR_DEAD_PORT", "info",
                       f"distributed rendering is ON and a server answered on "
                       f"{', '.join(alive)}", spelling=spelling)
    listed = ", ".join(f"{h}:{p}" for h, p in hosts) or "no hosts listed"
    head = (f"distributed rendering is ON and NOTHING is listening on {listed}"
            + (" (and the local machine is excluded)" if local is False else "")
            + " — this renderer will hand back black frames")
    try:
        detail = ctx.ctrl._black_probe_message(head)
    except Exception:  # noqa: BLE001 — a diagnostic must never outrank the diagnosis
        detail = head
    return Finding("DR_DEAD_PORT", "block", detail, spelling=spelling)


def check_render_type(ctx: Context) -> Finding:
    try:
        rt = sc._rt()
        kind = str(rt.getRendType()).lower().lstrip("#")
    except Exception:  # noqa: BLE001
        return Finding("RENDER_TYPE", "info",
                       "render type could not be read on this build")
    if kind and kind != "view":
        return Finding("RENDER_TYPE", "warn",
                       f"the render type is '{kind}', not 'view' — a region/crop render "
                       "leaves everything outside it black, and the critic scores the "
                       "whole frame")
    return Finding("RENDER_TYPE", "info", "render type is 'view' (full frame)")


def check_no_final_image(ctx: Context) -> Finding:
    r = ctx.renderer
    if r is None:
        return Finding("NO_FINAL_IMAGE", "info", "no current renderer to interrogate")
    spelling = sc.matched_prop(r, NO_FINAL_IMAGE_PROPS) or ""
    if not spelling:
        return Finding("NO_FINAL_IMAGE", "info",
                       "this renderer exposes no 'don't render final image' flag")
    if sc.get_prop(r, NO_FINAL_IMAGE_PROPS):
        return Finding("NO_FINAL_IMAGE", "warn",
                       f"'{spelling}' is ON — this renderer is set to compute GI and skip "
                       "the beauty pass, so every probe comes back with no image to score",
                       spelling=spelling)
    return Finding("NO_FINAL_IMAGE", "info", f"'{spelling}' is off", spelling=spelling)


def _renderer_class(ctx: Context) -> str:
    try:
        return str(sc._rt().classOf(ctx.renderer))
    except Exception:  # noqa: BLE001
        return ""


def check_renderer_is_vray(ctx: Context) -> Finding:
    cls = _renderer_class(ctx)
    if not cls:
        return Finding("RENDERER_IS_VRAY", "info",
                       "the current renderer's class could not be read")
    # same normalisation scripts/preflight.py uses: the box reports
    # V_Ray_GPU_7__update_2_hotfix_2, so an exact match would never fire
    norm = cls.lower().replace("_", "")
    if "vray" not in norm:
        return Finding("RENDERER_IS_VRAY", "warn",
                       f"the current renderer is {cls}, not V-Ray — every measured claim "
                       "in this plugin (draft sampler names, the exposure host, the "
                       "distributed-rendering diagnosis) was calibrated on V-Ray and may "
                       "not apply")
    return Finding("RENDERER_IS_VRAY", "info", f"renderer: {cls}")


def check_gpu_vantage_conflict(ctx: Context) -> Finding:
    """checklist #14 is a CRASH vector, not a perf note — moved here from run_match:985.

    V-Ray GPU loop renders while a Vantage live link streams the same scene can VRAM-starve
    the one card and take Max down. Now it also names its CONSEQUENCE: the Vantage probe
    backend REQUIRES that link to stay up for the whole match (see G-9), so arming it on
    a GPU renderer is opting into the documented crash configuration.
    """
    port = ctx.live_link_port()
    cls = _renderer_class(ctx)
    is_gpu = "gpu" in cls.lower()
    if port is not None and is_gpu:
        wants_vantage = str(getattr(ctx.cfg, "probe_backend", "vray")).lower() == "vantage"
        return Finding("GPU_VANTAGE_CONFLICT", "warn",
                       f"a Vantage live link is UP (port {port}) and the renderer is "
                       f"V-Ray GPU ({cls}) — one GPU doing both can crash Max "
                       "(checklist #14). Close the link for the match or switch V-Ray to "
                       "CPU."
                       + (" The Vantage probe backend will NOT be armed, because arming "
                          "it would require that link to stay up for the whole match."
                          if wants_vantage else ""))
    if port is not None:
        return Finding("GPU_VANTAGE_CONFLICT", "info",
                       f"a Vantage live link is up (port {port}) and the renderer is not "
                       "GPU — no VRAM conflict")
    return Finding("GPU_VANTAGE_CONFLICT", "info", "no Vantage live link is streaming")


def check_vantage_armed(ctx: Context) -> Finding:
    """Only when the artist opted in. Two settle budgets (~1.2 s), once per run.

    Two grabs half a second apart with NO apply between them must be STABLE. This is the
    exact inverse of ``_settled_rows``, and it is deliberate: during Vantage's scene
    ingest the viewport changes continuously as geometry arrives, so "did the picture
    move" is satisfied by GEOMETRY APPEARING rather than by the sun moving, and 44 angles
    get ranked against each other on plates of a different scene each time — not black,
    not occluded, not stale, no symptom at all. A settled viewport passes this; a scene
    still ingesting does not.
    """
    if str(getattr(ctx.cfg, "probe_backend", "vray")).lower() != "vantage":
        return Finding("VANTAGE_ARMED", "info",
                       "probe backend is V-Ray — the Vantage grab path is not in use")
    from . import vgrab

    cls = _renderer_class(ctx)
    if "gpu" in cls.lower():
        return Finding("VANTAGE_ARMED", "warn",
                       f"vantage probe backend NOT armed: the renderer is {cls} and the "
                       "grab path needs the live link up for the whole match, which is "
                       "the documented Max-crash configuration (checklist #14). Probes "
                       "render in V-Ray.")
    port = ctx.live_link_port()
    if port is None:
        return Finding("VANTAGE_ARMED", "warn",
                       "vantage probe backend NOT armed: no live link is streaming — "
                       "every probe renders in V-Ray")
    try:
        # INVERTED until 2026-07-31. The guard read `not active_camera_identity() and
        # name != active_camera_name()`, and active_camera_identity() is non-empty
        # whenever ANY camera is the viewport camera — so it could only fire when the
        # viewport was on NO camera, i.e. never in the case the sentence below describes.
        # A viewport left on "SCENE 01 shot01" while "SCENE 04 shot02" is matched sailed
        # through as info, which is precisely the wrong-shot grab this check exists for.
        active = sc.active_camera_name()
        if active != ctx.camera_name:
            return Finding("VANTAGE_ARMED", "warn",
                           "vantage probe backend NOT armed: the live link mirrors the "
                           f"ACTIVE viewport camera and that is "
                           f"{('no camera at all' if not active else repr(active))}, not "
                           f"'{ctx.camera_name}' — the grabs would be of a different shot")
    except Exception:  # noqa: BLE001
        pass
    hwnd = vgrab.find_window(vgrab.VANTAGE_TITLE)
    if not hwnd:
        return Finding("VANTAGE_ARMED", "warn",
                       f"vantage probe backend NOT armed: the window was not found "
                       f"({vgrab.last_error()}) — every probe renders in V-Ray")
    scratch = os.path.join(ctx.run_dir or ".", "preflight_vantage.png")
    vgrab.reset_settle()
    first = vgrab.capture_window_png(vgrab.VANTAGE_TITLE, scratch, 64, 36)
    if not first:
        return Finding("VANTAGE_ARMED", "warn",
                       f"vantage probe backend NOT armed: a test grab was refused "
                       f"({vgrab.last_error()}) — every probe renders in V-Ray")
    # THE SECOND GRAB, which the docstring above promised and the body did not take
    # (2026-07-31). reset_settle again so this grab is not gated by the freshness test —
    # here the question is the opposite one, and the answer must be "nothing changed".
    sig1 = vgrab.last_signature()
    vgrab.reset_settle()
    second = vgrab.capture_window_png(vgrab.VANTAGE_TITLE, scratch, 64, 36)
    sig2 = vgrab.last_signature()
    vgrab.reset_settle()          # neither of these is a probe's baseline
    if not second:
        return Finding("VANTAGE_ARMED", "warn",
                       f"vantage probe backend NOT armed: the second test grab was "
                       f"refused ({vgrab.last_error()}) — every probe renders in V-Ray")
    if sig1 and sig2 and vgrab._moved(sig2, sig1):
        return Finding("VANTAGE_ARMED", "warn",
                       f"vantage probe backend NOT armed: the window CHANGED between two "
                       f"grabs {vgrab.SETTLE_LIMIT_S:g}s apart with nothing applied "
                       "between them — Vantage is still ingesting this scene, and 44 sun "
                       "angles ranked against each other on 44 different pictures is the "
                       "failure with no symptom. Let it settle and re-run; probes render "
                       "in V-Ray meanwhile.")
    # G-6, reported rather than refused (2026-07-31). Vantage draws its menu bar, toolbar,
    # right-hand properties panel and status bar INSIDE the client area, so _aspect_crop
    # centres on the client rather than on the viewport: on a 3440-wide client with a
    # 380 px panel the content sits ~190 px left of where the crop thinks it is, and
    # light-on-dark UI text clears metrics' ABSOLUTE HOT_THRESHOLD in fixed cells of every
    # probe. The aspect ratio is the only cheap tell, and it is NOT decisive — a
    # legitimately maximised window on a 21:9 monitor has the same distortion with no
    # panel at all — so refusing on it would disable a working backend on ordinary
    # hardware. The proper fix is EnumChildWindows for the D3D surface, which cannot be
    # verified off-box. This states the measurement and leaves the decision to the artist.
    client = vgrab._client_rect(int(hwnd))
    if client:
        cw, ch = client[2] - client[0], client[3] - client[1]
        want = 16.0 / 9.0
        got = (cw / float(ch)) if ch else want
        if ch and abs(got - want) / want > 0.05:
            return Finding("VANTAGE_ARMED", "info",
                           f"vantage probe backend armed (live link on {port}) — but its "
                           f"client area is {cw}×{ch} ({got:.2f}:1 against the probe's "
                           f"{want:.2f}:1), so the centre crop may be off-centre if "
                           "Vantage is showing docked panels. Eyeball one "
                           "sunsolve_a000.png before trusting an angle from this backend.")
    return Finding("VANTAGE_ARMED", "info",
                   f"vantage probe backend armed (live link on {port}, test grab ok)")


def check_sun_identity(ctx: Context) -> Finding:
    suns = list(ctx.rig.get("suns") or [])
    driven = ctx.rig.get("sun")
    if driven is None:
        return Finding("SUN_IDENTITY", "warn",
                       "no sun in this rig — every sun.* parameter is disabled and the "
                       "sun solve cannot run")
    names = [_node_name(s) for s in suns]
    driven_name = _node_name(driven)
    lowered = driven_name.lower()
    # WHICH keyword won — this is what surfaces the substring landmine _role_rank really
    # has: "gi" matches reGIon/oriGIn/loGIc, "key" matches KEYstone, "main" matches
    # doMAIN. On a landscape project "SUN SHOT 12 REGIONAL PARK" outranks every correctly
    # named sun, and nothing said so until now.
    won = [w for w in ("primary", "main", "hero", "key", "lighting", "gi")
           if w in lowered]
    how = (f"its name contains {', '.join(repr(w) for w in won)}" if won
           else "it won the alphabetical tiebreak among equally-ranked suns")
    detail = (f"steering sun '{driven_name}' of {len(suns)} in this scene — {how}")
    if len(suns) > 4:
        detail += (f". Only the first four role-sorted suns are matchable; "
                   f"{', '.join(names[4:])} cannot be reached at all")
    return Finding("SUN_IDENTITY", "warn" if len(suns) > 1 else "info", detail)


def check_sun_enabled(ctx: Context) -> Finding:
    """``classify_rig`` never reads LIGHT_ON. Twelve suns, one enabled, wrong shot."""
    suns = list(ctx.rig.get("suns") or [])
    driven = ctx.rig.get("sun")
    if driven is None or not suns:
        return Finding("SUN_ENABLED", "info", "no sun in this rig to check")
    on: List[str] = []
    off: List[str] = []
    for s in suns:
        (on if sc.get_prop(s, sc.LIGHT_ON, True) else off).append(_node_name(s))
    driven_name = _node_name(driven)
    spelling = sc.matched_prop(driven, sc.LIGHT_ON) or ""
    if driven_name in off:
        extra = (f"; the only enabled sun(s) here are {', '.join(on)}" if on
                 else "; no sun in this scene is enabled at all")
        return Finding("SUN_ENABLED", "warn",
                       f"the sun being steered, '{driven_name}', is switched OFF{extra}. "
                       "Every sun probe will render the same picture and the solve will "
                       "rank noise.", spelling=spelling)
    if off:
        return Finding("SUN_ENABLED", "info",
                       f"'{driven_name}' is enabled ({len(off)} of {len(suns)} suns in "
                       "this scene are off)", spelling=spelling)
    return Finding("SUN_ENABLED", "info",
                   f"'{driven_name}' is enabled (all {len(suns)} suns are on)",
                   spelling=spelling)


def check_sun_name_match(ctx: Context) -> Finding:
    """Relative token overlap, and it DECLINES TO GUESS unless one sun strictly wins.

    No combined verdict with the other three sun checks, ever. A fused score is precisely
    what produces a confident false positive, and this one has to survive a project that
    names its suns by mood.
    """
    suns = list(ctx.rig.get("suns") or [])
    driven = ctx.rig.get("sun")
    if driven is None or len(suns) < 2:
        return Finding("SUN_NAME_MATCH", "info",
                       "one sun (or none) — there is no alternative to propose")
    driven_name = _node_name(driven)
    scored = [(name_overlap(_node_name(s), ctx.camera_name), _node_name(s)) for s in suns]
    driven_score = name_overlap(driven_name, ctx.camera_name)
    rivals = [(v, n) for v, n in scored if n != driven_name]
    best = max(rivals) if rivals else (0.0, "")
    if best[0] > driven_score and best[0] >= NAME_MATCH_PROPOSE:
        return Finding("SUN_NAME_MATCH", "warn",
                       f"'{best[1]}' names this camera far better ({best[0]:.0%} of "
                       f"'{ctx.camera_name}'s name) than the sun being steered, "
                       f"'{driven_name}' ({driven_score:.0%}). Check which sun this shot "
                       "was authored for before spending the run.")
    return Finding("SUN_NAME_MATCH", "warn" if driven_score < NAME_MATCH_PROPOSE else "info",
                   f"the sun being steered, '{driven_name}', matches {driven_score:.0%} "
                   f"of this camera's name, and no sun in this scene names "
                   f"'{ctx.camera_name}' any better — so nothing is proposed. If this "
                   "shot has its own sun, it is not named after it.")


def _dist(a, b) -> Optional[float]:
    try:
        return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2
                + (float(a[2]) - float(b[2])) ** 2) ** 0.5
    except (TypeError, ValueError, IndexError):
        return None


def check_sun_geometry(ctx: Context) -> Finding:
    """TARGETED suns only, ORDINAL only, and it prints both world units and metres.

    Sun Positioners and untargeted VRaySuns are skipped outright: ``scene._sun_pivot``
    returns Point3(0,0,0) for them, and comparing a world-origin placeholder against a
    camera 400 m away is a guaranteed false positive. No absolute distance threshold
    either — the test is purely "is another sun's target more than twice as close to what
    this camera is looking at", which self-calibrates from a teapot to an 18M-triangle
    site and stays silent when the scene has one sun.
    """
    from . import digest as dg

    suns = list(ctx.rig.get("suns") or [])
    driven = ctx.rig.get("sun")
    if driven is None or len(suns) < 2 or ctx.cam is None:
        return Finding("SUN_GEOMETRY", "info",
                       "not applicable — fewer than two suns, or no camera node")
    basis = dg.camera_basis(ctx.cam)
    if basis is None:
        return Finding("SUN_GEOMETRY", "info",
                       "the camera's look-at could not be read — geometry not verified")
    look = basis["look"]
    rt = sc._rt()

    def target_pos(node):
        try:
            tgt = getattr(node, "target", None)
            if tgt is None or not rt.isValidNode(tgt):
                return None                 # gate: scene.py:403's own test
            p = tgt.pos
            return (float(p.x), float(p.y), float(p.z))
        except Exception:  # noqa: BLE001
            return None

    driven_t = target_pos(driven)
    if driven_t is None:
        return Finding("SUN_GEOMETRY", "info",
                       f"'{_node_name(driven)}' is untargeted (or a Sun Positioner) — it "
                       "has no world pivot to compare, so geometry is not evidence here")
    d_driven = _dist(driven_t, look)
    cam_target = _dist(basis["pos"], look)
    if d_driven is None or cam_target is None:
        return Finding("SUN_GEOMETRY", "info", "distances could not be computed")
    closer = []
    for s in suns:
        if s is driven:
            continue
        t = target_pos(s)
        if t is None:
            continue
        d = _dist(t, look)
        if d is not None and d * 2.0 < d_driven:
            closer.append((d, _node_name(s)))
    if closer and d_driven > cam_target:
        d, name = min(closer)
        return Finding("SUN_GEOMETRY", "warn",
                       f"the sun being steered aims {d_driven:.0f} world units "
                       f"({sc.metres_from_world_units(d_driven):.0f} m) from what this "
                       f"camera looks at, further than the camera's own target distance; "
                       f"'{name}' aims {d:.0f} units "
                       f"({sc.metres_from_world_units(d):.0f} m) away — more than twice "
                       "as close. This shot may have been authored around that one.")
    return Finding("SUN_GEOMETRY", "info",
                   f"the steered sun aims {d_driven:.0f} world units "
                   f"({sc.metres_from_world_units(d_driven):.0f} m) from this camera's "
                   "look-at; no other targeted sun is materially closer")


def check_rig_notes(ctx: Context) -> Finding:
    """``classify_rig`` writes these and they go ONLY into two LLM prompts. Nothing in
    ui/dock.py or api.py reads them. They are the classifier telling the artist that the
    sun it is about to steer aims by node rotation, or that a Daylight assembly will fight
    every move — exactly the facts a run is about to be spent ignoring."""
    notes = [str(n) for n in (ctx.rig.get("notes") or []) if str(n).strip()]
    if not notes:
        return Finding("RIG_NOTES", "info", "the rig classifier raised no notes")
    return Finding("RIG_NOTES", "warn", "rig: " + " · ".join(notes))


def check_exposure_roundtrip(ctx: Context) -> Finding:
    """Write one EV, read it back, put it back in a ``finally``. <5 ms, zero renders.

    The one mutating check in this module, and it is the only way to catch "an exposure
    host that will not respond" before the loop spends its whole budget asking for EV
    changes that never reach a pixel.
    """
    from .exposure import ExposureHost

    if ctx.cam is None:
        return Finding("EXPOSURE_ROUNDTRIP", "info", "no camera node to check")
    host = ExposureHost(ctx.cam)
    if host.kind == "none":
        return Finding("EXPOSURE_ROUNDTRIP", "warn",
                       "this scene has no exposure control MaxGaffer can drive — "
                       "exposure.ev will not be solvable")
    ev0 = host.read_ev()
    if ev0 is None:
        return Finding("EXPOSURE_ROUNDTRIP", "warn",
                       f"the exposure host ({host.kind}) would not report an EV — the "
                       "analytic exposure solver has nothing to anchor on")
    try:
        wrote = host.write_ev(ev0 + 2.0)
        back = host.read_ev()
    finally:
        host.write_ev(ev0)
    if not wrote:
        return Finding("EXPOSURE_ROUNDTRIP", "warn",
                       f"the exposure host ({host.kind}) REFUSED a write — every EV the "
                       "solver prescribes this run would be silently discarded")
    if back is None or abs(float(back) - (float(ev0) + 2.0)) > 0.05:
        return Finding("EXPOSURE_ROUNDTRIP", "warn",
                       f"the exposure host ({host.kind}) accepted EV {ev0 + 2.0:.2f} but "
                       f"read back {back} — writes are not sticking")
    return Finding("EXPOSURE_ROUNDTRIP", "info",
                   f"exposure host '{host.kind}' round-trips (+2 EV written and read back)")


def check_camera_active(ctx: Context) -> Finding:
    """The DISCARDED return value of ``_set_active_camera``. It matters for the Vantage
    backend (the live link mirrors the active viewport camera, not the node V-Ray is
    handed) and for anything that reads the viewport."""
    try:
        ok = ctx.ctrl._set_active_camera(ctx.camera_name)
    except Exception as err:  # noqa: BLE001
        return Finding("CAMERA_ACTIVE", "warn",
                       f"the viewport could not be pointed at '{ctx.camera_name}' "
                       f"({err})")
    if ok is False:
        return Finding("CAMERA_ACTIVE", "warn",
                       f"the viewport could NOT be switched to '{ctx.camera_name}' — "
                       "V-Ray will still render the camera node, but anything that reads "
                       "the viewport (a Vantage live-link grab) would be of another shot")
    return Finding("CAMERA_ACTIVE", "info",
                   f"viewport is on '{ctx.camera_name}'")


def check_camera_duplicate(ctx: Context) -> Finding:
    """``list_cameras`` computes this at scene.py:214-222 and no match has ever shown it."""
    try:
        cams = sc.list_cameras()
    except Exception as err:  # noqa: BLE001
        return Finding("CAMERA_DUPLICATE", "info",
                       f"the camera list could not be read ({err})")
    dupes = [c for c in cams if c.get("name") == ctx.camera_name and c.get("duplicate")]
    if dupes:
        return Finding("CAMERA_DUPLICATE", "warn",
                       f"more than one node in this scene is named "
                       f"'{ctx.camera_name}' (merged scenes do this) — the one being "
                       "matched is whichever Max resolves first, which may not be the "
                       "one you selected")
    return Finding("CAMERA_DUPLICATE", "info",
                   f"'{ctx.camera_name}' is a unique node name")


def check_run_dir_writable(ctx: Context) -> Finding:
    """``_ensure_run_dir`` already falls back to %TEMP% — but it only ``print()``s it."""
    if not ctx.run_dir:
        return Finding("RUN_DIR_WRITABLE", "info", "no run directory to check")
    probe = os.path.join(ctx.run_dir, ".preflight")
    try:
        with open(probe, "wb") as f:
            f.write(b"1")
        with open(probe, "rb") as f:
            data = f.read()
        os.unlink(probe)
    except OSError as err:
        return Finding("RUN_DIR_WRITABLE", "block",
                       f"the run directory is not writable ({err}): {ctx.run_dir}. Every "
                       "probe render would fail and the run would be scored on nothing.")
    if data != b"1":
        return Finding("RUN_DIR_WRITABLE", "block",
                       f"the run directory did not read back what was written to it: "
                       f"{ctx.run_dir}")
    return Finding("RUN_DIR_WRITABLE", "info", f"run dir writable: {ctx.run_dir}")


def check_draft_cap_binds(ctx: Context) -> Finding:
    """Predict G-10's truncation BEFORE the run, in whole seconds.

    V-Ray states the per-frame cap in MINUTES. 29bbae6 caught the case where that
    truncates to 0 (which V-Ray reads as NO limit); it does not catch 110 s → int(1.833)
    → 1 → a 60 s cap, a 45% shortfall logged as a cheerful "0 → 1" with no unit.
    """
    from . import draft as df

    secs = float(getattr(ctx.cfg, "probe_max_seconds", 0.0) or 0.0)
    if not getattr(ctx.cfg, "draft_sampler", False):
        if secs > 0:
            return Finding("DRAFT_CAP_BINDS", "warn",
                           f"a {secs:g}s probe cap is configured but draft sampler is "
                           "OFF, and the cap is gated by that flag — it will NOT be in "
                           "effect")
        return Finding("DRAFT_CAP_BINDS", "info", "draft sampler is off; no probe cap")
    if secs <= 0:
        return Finding("DRAFT_CAP_BINDS", "info",
                       "draft sampler is on with no probe time cap (probe_max_seconds 0)")
    r = ctx.renderer
    name = sc.matched_prop(r, df.TIME_CAP_PROPS) if r is not None else None
    if not name:
        return Finding("DRAFT_CAP_BINDS", "warn",
                       f"the {secs:g}s probe cap has nowhere to land — none of "
                       f"{', '.join(df.TIME_CAP_PROPS)} exists on this renderer "
                       "(checklist #15)")
    original = sc.get_prop(r, (name,))
    if isinstance(original, int) and not isinstance(original, bool):
        effective = int(secs / 60.0) * 60.0
        if effective <= 0:
            return Finding("DRAFT_CAP_BINDS", "warn",
                           f"'{name}' is whole MINUTES on this build, so a {secs:g}s cap "
                           "truncates to 0 — which V-Ray reads as NO LIMIT. The cap will "
                           "be refused rather than uncap your probes; use ≥ 60 s.",
                           spelling=name)
        if abs(effective - secs) > 0.5:
            return Finding("DRAFT_CAP_BINDS", "warn",
                           f"'{name}' is whole MINUTES on this build, so your {secs:g}s "
                           f"cap will actually bind at {effective:g}s", spelling=name)
    return Finding("DRAFT_CAP_BINDS", "info",
                   f"the {secs:g}s probe cap will land on '{name}'", spelling=name)


#: ORDER MATTERS: the cheap property reads run first, the reference decode and the
#: half-second Vantage stability gate last, so a blocking finding is usually known in
#: single-digit milliseconds.
CHECKS: Tuple[Tuple[str, Callable[[Context], Finding]], ...] = (
    ("REFERENCE_PRESENT", check_reference_present),
    ("DR_DEAD_PORT", check_dr_dead_port),
    ("RENDERER_IS_VRAY", check_renderer_is_vray),
    ("RENDER_TYPE", check_render_type),
    ("NO_FINAL_IMAGE", check_no_final_image),
    ("RUN_DIR_WRITABLE", check_run_dir_writable),
    ("DRAFT_CAP_BINDS", check_draft_cap_binds),
    ("CAMERA_ACTIVE", check_camera_active),
    ("CAMERA_DUPLICATE", check_camera_duplicate),
    ("RIG_NOTES", check_rig_notes),
    ("SUN_IDENTITY", check_sun_identity),
    ("SUN_ENABLED", check_sun_enabled),
    ("SUN_NAME_MATCH", check_sun_name_match),
    ("SUN_GEOMETRY", check_sun_geometry),
    ("EXPOSURE_ROUNDTRIP", check_exposure_roundtrip),
    ("REFERENCE_READABLE", check_reference_readable),
    ("REFERENCE_DEGENERATE", check_reference_degenerate),
    ("REFERENCE_HAS_SUN", check_reference_has_sun),
    ("GPU_VANTAGE_CONFLICT", check_gpu_vantage_conflict),
    ("VANTAGE_ARMED", check_vantage_armed),
)


def _check(key: str, fn: Callable[[Context], Finding], ctx: Context) -> Finding:
    """Time one check and convert a RAISE into "could not verify" — never into "ok"."""
    t0 = time.time()
    try:
        finding = fn(ctx)
    except Exception as err:  # noqa: BLE001 — one broken check must not stop a match
        finding = Finding(key, "info", f"could not verify: {type(err).__name__}: {err}",
                          verified=False)
    if not isinstance(finding, Finding):     # a check that returned None is a bug, and a
        finding = Finding(key, "info",       # silent "ok" is exactly the bug class here
                          "could not verify: the check returned no finding",
                          verified=False)
    return Finding(finding.key or key, finding.severity, finding.detail,
                   finding.spelling, (time.time() - t0) * 1000.0, finding.verified)


def run(ctrl, camera_name: str, cam, rig: Dict, entry, cfg,
        log: Callable[[str], None], ref_path: str = "",
        ref_stats: Optional[Dict] = None, run_dir: str = "") -> PreflightReport:
    """Run every check, report on both outcomes, and raise ``PreflightBlocked`` if asked.

    ``cfg.preflight_level``: "block" (default) stops on a block; "warn" downgrades every
    block and LOGS that it did; "off" skips the whole thing and logs that too.
    """
    level = str(getattr(cfg, "preflight_level", "block") or "block").lower()
    report = PreflightReport()
    if getattr(cfg, "no_renders", False) and level != "off":
        # Apply-only mode fires no render, so every BLOCK here has nothing to protect:
        # they exist to stop a run SPENDING on a non-measurement, and this one spends
        # nothing. Blocking would be this module deciding an artist may not push a first
        # guess into a scene whose reference happens to be offline. The checks still RUN
        # and still report — a scene whose only enabled sun belongs to another shot is
        # worth knowing about whether or not a frame is rendered. (2026-07-31)
        level = "warn"
    if level == "off":
        log("⚠ preflight: OFF (preflight_level='off') — nothing about this scene was "
            "verified before spending renders on it")
        return report
    try:
        renderer = sc._rt().renderers.current
    except Exception:  # noqa: BLE001
        renderer = None
    ctx = Context(ctrl=ctrl, camera_name=camera_name, cam=cam, rig=rig or {},
                  entry=entry, cfg=cfg, log=log, ref_path=ref_path,
                  ref_stats=ref_stats, renderer=renderer, run_dir=run_dir)
    t0 = time.time()
    for key, fn in CHECKS:
        report.findings.append(_check(key, fn, ctx))
    report.elapsed_ms = (time.time() - t0) * 1000.0

    if str(getattr(cfg, "probe_backend", "vray")).lower() == "vantage":
        # RECORD THE DECISION IN BOTH DIRECTIONS so the controller reads a decision rather
        # than re-deriving one — and, more importantly, so the ABSENCE of a decision is
        # unambiguous. A report that carries no "probe_backend" key is a report in which
        # VANTAGE_ARMED never ran (preflight_level='off', or pf.run itself raised), and
        # arming a window grab off that is arming it having checked nothing. A check that
        # RAISED is not an arming either — hence ``verified``. (2026-07-31)
        armed = report.find("VANTAGE_ARMED")
        report.demotions["probe_backend"] = (
            "vantage" if (armed is not None and armed.severity == "info"
                          and armed.verified) else "vray")

    blocked = report.blocked()
    if blocked and level == "warn":
        log(f"⚠ preflight: {len(blocked)} BLOCKING finding(s) downgraded to warnings"
            + (" because this is a no-render run and there is nothing to spend"
               if getattr(cfg, "no_renders", False) else
               " by preflight_level='warn' — this run will produce numbers that are not "
               "measurements"))
        for f in blocked:
            log(f"⚠ preflight {f.key}: {f.detail}")
    for f in report.warnings():
        log(f"⚠ preflight {f.key}: {f.detail}")
    clean = report.clean()
    log(f"preflight: {len(report.findings)} checks, {len(clean)} clean in "
        f"{report.elapsed_ms:.0f} ms")
    if blocked and level != "warn":
        for f in blocked:
            log(f"✗ preflight {f.key}: {f.detail}")
        raise PreflightBlocked(blocked[0].detail)
    return report
