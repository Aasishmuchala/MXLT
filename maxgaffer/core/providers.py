"""Provider-neutral semantic-model transport.

The rest of MaxGaffer speaks one Anthropic-style block format. Adapters keep that stable
while supporting the bundled Kesar/Omega gateway, Anthropic directly, any
OpenAI-compatible chat endpoint (including local servers), or an explicit offline mode.
"""

from __future__ import annotations

import json
import random
import time
from typing import Callable, Optional

from . import omega

DEFAULT_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
}


class ProviderError(omega.OmegaError):
    pass


def _post(url: str, headers: dict, body: bytes, timeout: int) -> tuple:
    return omega._default_post(url, headers, body, timeout)


def _openai_content(content):
    if isinstance(content, str):
        return content
    out = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            out.append({"type": "text", "text": str(block.get("text") or "")})
        elif block.get("type") == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64" and src.get("data"):
                media = src.get("media_type") or "image/png"
                out.append({"type": "image_url", "image_url": {
                    "url": f"data:{media};base64,{src['data']}"}})
    return out


def _extract_openai(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text").strip()
    return ""


def _http_call(url: str, headers: dict, payload: dict,
               extract: Callable[[dict], str], post: Callable, timeout: int) -> str:
    body = json.dumps(payload).encode("utf-8")
    last = "provider request failed"
    for attempt, pause in enumerate((0.0, 2.0, 6.0, 15.0)):
        if pause:
            time.sleep(pause + random.uniform(0.0, pause * 0.25))
        status = None
        raw = ""
        try:
            status, raw = post(url, headers, body, timeout)
        except Exception as err:  # noqa: BLE001
            last = f"network error: {err}"
        if status in (401, 403):
            raise ProviderError(f"semantic provider returned HTTP {status}", "auth", raw[:2000])
        if status is not None and 200 <= status < 300:
            try:
                data = json.loads(raw)
            except (ValueError, RecursionError):
                data = {}
            text = extract(data)
            if text:
                return text
            last = "the semantic model returned no text"
        elif status == 429 or (status is not None and 500 <= status <= 599):
            last = f"semantic provider HTTP {status}"
        elif status is not None:
            raise ProviderError(f"semantic provider HTTP {status}: {raw[:200]}",
                                "other", raw[:2000])
        if attempt == 3:
            break
    raise ProviderError(last, "network")


def call(provider: str, key: str, system: str, messages: list, *, model: str,
         max_tokens: int = 8192, base_url: str = "", post: Optional[Callable] = None,
         timeout: int = omega.TIMEOUT_S) -> str:
    kind = str(provider or "omega").strip().lower().replace("-", "_")
    if kind in ("omega", "kesar", "kesarcloud"):
        kwargs = {"model": model, "max_tokens": max_tokens}
        if post is not None:
            kwargs["post"] = post
        return omega.call(key, system, messages, **kwargs)
    if kind == "offline":
        raise ProviderError("semantic provider is set to offline", "offline")
    if kind == "anthropic":
        if not key:
            raise ProviderError("Anthropic API key is missing", "auth")
        url = base_url.strip() or DEFAULT_URLS["anthropic"]
        headers = {"content-type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01", "user-agent": "MaxGaffer"}
        payload = {"model": model, "max_tokens": max_tokens, "system": system,
                   "messages": messages}
        return _http_call(url, headers, payload, omega.extract_text,
                          post or _post, timeout)
    if kind in ("openai", "openai_compatible", "local"):
        if kind == "openai" and not key:
            raise ProviderError("OpenAI API key is missing", "auth")
        url = base_url.strip() or DEFAULT_URLS.get(kind, "")
        if not url:
            raise ProviderError("an OpenAI-compatible base URL is required", "other")
        headers = {"content-type": "application/json", "user-agent": "MaxGaffer"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        translated = [{"role": "system", "content": system}]
        translated.extend({"role": m.get("role", "user"),
                           "content": _openai_content(m.get("content", ""))}
                          for m in messages if isinstance(m, dict))
        payload = {"model": model, "max_tokens": max_tokens,
                   "stream": False, "messages": translated}
        return _http_call(url, headers, payload, _extract_openai,
                          post or _post, timeout)
    raise ProviderError(f"unknown semantic provider {provider!r}", "other")


def ping(provider: str, key: str, model: str, base_url: str = "",
         post: Optional[Callable] = None) -> str:
    text = call(provider, key, "Reply with exactly: OK",
                [{"role": "user", "content": "ping"}], model=model,
                max_tokens=16, base_url=base_url, post=post)
    return f"{provider or 'omega'} reachable ({model}): {text.strip()[:24]!r}"
