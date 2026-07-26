"""Omega gateway client — MaxGaffer's single LLM network surface.

Ported contract-verbatim from MaxDirector (itself ported from LightMatch), because the wire
contract is hard-won and verified live against the real gateway:
  - NO tools / tool_choice (the gateway 500s on them) — the JSON schema is embedded in the
    system prompt and the reply is parsed out of the TEXT blocks;
  - non-streaming; Bearer key; anthropic-version header;
  - retries with jittered backoff on 429/5xx; the 120s timeout is per socket I/O op
    (urllib semantics), NOT a total wall-clock bound — a slow-drip server can hold
    one attempt far longer;
  - multimodal via base64 image blocks (reference + render comparisons are the whole point).

The only module in core allowed to touch the network; ``post`` is injectable so the entire
suite runs offline.
"""

from __future__ import annotations

import json
import random
import time
from typing import Optional

#: Default Omega Plus messages endpoint (Anthropic-compatible wire, Bearer auth). The
#: docs print the base as ".../v1"; the messages path is appended by resolve_url, which
#: also honours a user-configured override so a future endpoint move needs no code change.
#: (Was omega.kesarcloud.in until 2026-07-25 — same vendor, new host.)
GATEWAY_BASE_URL = "https://api.omegaplusapi.com/v1"
GATEWAY_URL = GATEWAY_BASE_URL + "/messages"
TIMEOUT_S = 120
BACKOFF_S = (2.0, 6.0, 15.0)
DEFAULT_MODEL = "claude-opus-4-8"
MAX_JSON_SCAN_CHARS = 16_000_000   # fuzz contract accepts a valid 10 MB object


class OmegaError(RuntimeError):
    def __init__(self, message: str, kind: str = "other", raw: str = ""):
        super().__init__(message)
        self.kind = kind
        self.raw = raw


def extract_text(payload: dict) -> str:
    if not isinstance(payload, dict):       # a 200 proxy/error page can parse as a list
        return ""
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return ""     # a 200 carrying `null`/[] degrades into the retry path, not a crash
    return "\n".join(
        str(b.get("text") or "")            # text:null / non-string text coerces, never raises
        for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def parse_json_from_text(text: str) -> Optional[dict]:
    """First balanced top-level ``{...}`` object, found in one linear scan.

    Restarting at every opening brace made a hostile brace flood quadratic and capable of
    freezing Max.  A character cap also bounds work if a provider ignores token limits.
    """
    if not isinstance(text, str) or len(text) > MAX_JSON_SCAN_CHARS:
        return None
    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                esc = False
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, RecursionError):
                    pass
                start = None
    return None


#: Grace added to the caller's timeout before the watchdog abandons the request thread.
_WATCHDOG_GRACE_S = 15


def _default_post(url: str, headers: dict, body: bytes, timeout: int) -> tuple:
    """Stdlib HTTP POST with a HARD overall deadline, imported lazily so tests never touch
    the network by accident.

    The urlopen timeout only bounds socket operations — connect and read. Everything BEFORE
    the socket is unbounded: getaddrinfo has no timeout parameter at all, and Windows proxy
    auto-discovery can block indefinitely. That gap is not theoretical here. Measured
    2026-07-26, twice in one day: the identical request completed in 3.4-5.7s from a plain
    python process while hanging FOREVER (70 minutes at ~0% CPU, and again for 15 minutes)
    inside 3ds Max's embedded interpreter — wedged before the first byte moved, where the
    120s "timeout" never sees it. Inside the dock that presented as the plugin freezing
    with the MATCH button dead, which cost most of a day to corner.

    So the request runs on a daemon worker and this thread waits ``timeout`` plus grace.
    On expiry the worker is ABANDONED — there is no portable way to kill it, and a leaked
    daemon thread parked in getaddrinfo costs nothing — and a typed network error is
    raised, which the retry loop and the dock's offline fallback already know how to
    handle. A hang becomes an error; an error has a message; a message can be acted on."""
    import queue
    import threading
    import urllib.error
    import urllib.request

    def _worker(out: "queue.Queue") -> None:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 fixed https base
                out.put((resp.getcode(), resp.read().decode("utf-8", errors="replace")))
        except urllib.error.HTTPError as e:   # 4xx/5xx still carry a body worth reading
            out.put((e.code, e.read().decode("utf-8", errors="replace")))
        except BaseException as e:  # noqa: BLE001 — marshalled, re-raised on the caller
            out.put(e)

    out: "queue.Queue" = queue.Queue(maxsize=1)
    t = threading.Thread(target=_worker, args=(out,), daemon=True,
                         name="maxgaffer-gateway-post")
    t.start()
    try:
        got = out.get(timeout=float(timeout) + _WATCHDOG_GRACE_S)
    except queue.Empty:
        raise OmegaError(
            f"gateway gave no response within {int(timeout) + _WATCHDOG_GRACE_S}s — the "
            f"request never reached the socket stage (DNS or proxy discovery wedged, a "
            f"known failure inside 3ds Max's python). The request thread was abandoned.",
            kind="network")
    if isinstance(got, BaseException):
        raise got
    return got


def resolve_url(base_url: str = "") -> str:
    """The messages endpoint for an optional configured base. Accepts EITHER the
    Anthropic-compatible base as the Omega Plus docs print it (``https://api.
    omegaplusapi.com/v1``) or a full ``.../v1/messages`` endpoint, so a user pasting
    either form from the docs into Settings gets a working gateway. Blank → the shipped
    default."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return GATEWAY_URL
    return url if url.endswith("/messages") else url + "/messages"


def call(
    key: str,
    system: str,
    messages: list,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    post=_default_post,
    base_url: str = "",
) -> str:
    """One resilient gateway round; returns the reply TEXT. Raises OmegaError with a typed
    kind (auth | network | other) on failure. ``post`` is injectable for tests.
    ``base_url`` overrides the shipped endpoint (config.semantic_base_url)."""
    if not key:
        raise OmegaError("No API key set — paste your oc_ key in MaxGaffer's settings.", "auth")
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "stream": False,
            "system": system,
            "messages": messages,
        }
    ).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {key}",
        "anthropic-version": "2023-06-01",
        # Cloudflare in front of the gateway rejects UA-less urllib (error 1010) —
        # found by live fire 2026-07-16; identify ourselves like any honest API client
        "user-agent": "MaxGaffer (+python-urllib)",
        "accept": "application/json",
    }
    last = "gateway request failed"
    for attempt in range(len(BACKOFF_S) + 1):
        status = None
        text_body = ""
        try:
            status, text_body = post(resolve_url(base_url), headers, body, TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 network layer surfaces as a retryable miss
            last = f"network error: {e}"
        if status is not None:
            if status == 401:
                raise OmegaError("Gateway returned 401 — the API key is missing or invalid.", "auth")
            if 200 <= status < 300:
                try:
                    payload = json.loads(text_body)
                except (ValueError, RecursionError):
                    payload = {}
                text = extract_text(payload)
                if text:
                    return text
                last = "the model returned no text"
            elif status == 429 or 500 <= status <= 599:
                last = f"gateway HTTP {status}"
            else:
                raise OmegaError(
                    f"Gateway request failed: HTTP {status} — {text_body[:200]}",
                    "other",
                    text_body[:2000],
                )
        if attempt < len(BACKOFF_S):
            base = BACKOFF_S[attempt]
            time.sleep(base + random.uniform(0.0, base * 0.5))  # jitter vs retry herds
    raise OmegaError(last, "network")


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(b64: str, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}


def image_block_from_file(path: str, max_bytes: int = 3_500_000) -> Optional[dict]:
    """Base64 image block straight from disk (media type from the extension).

    Files over ``max_bytes`` are refused (None): the gateway caps images near 5 MB and
    base64 inflates ~x4/3, so 3.5 MB mirrors the controller's own guard for callers
    that don't pre-check size."""
    import base64
    import os

    ext = os.path.splitext(path)[1].lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "webp": "image/webp"}.get(ext.lstrip("."), None)
    if media is None:
        return None
    try:
        if os.path.getsize(path) > max_bytes:
            return None
        with open(path, "rb") as f:
            return image_block(base64.b64encode(f.read()).decode("ascii"), media)
    except OSError:
        return None


def ping(key: str, model: str = DEFAULT_MODEL, post=_default_post,
         base_url: str = "") -> str:
    text = call(
        key, "Reply with exactly the two characters: OK",
        [{"role": "user", "content": "ping"}], model=model, max_tokens=16, post=post,
        base_url=base_url,
    )
    return f"gateway reachable ({model}): {text.strip()[:24]!r}"
