"""Omega Plus gateway endpoint + base-URL override (2026-07-25).

The gateway moved from omega.kesarcloud.in to api.omegaplusapi.com (same vendor, new
host, unchanged Anthropic wire + Bearer auth + oc_ keys). While wiring it, a live bug
surfaced: providers.call ACCEPTED base_url and then dropped it for the omega provider,
so a URL typed into Settings was silently ignored. Both are locked here.
"""
import pytest

from maxgaffer.core import omega, providers


# ------------------------------------------------------------------ endpoint default
def test_default_endpoint_is_omega_plus():
    assert omega.GATEWAY_BASE_URL == "https://api.omegaplusapi.com/v1"
    assert omega.GATEWAY_URL == "https://api.omegaplusapi.com/v1/messages"
    assert "kesarcloud" not in omega.GATEWAY_URL


def test_default_model_is_available_on_the_gateway():
    # claude-opus-4-8 is in the Omega Plus model list and is vision-capable (the loop
    # shows it rendered plates)
    assert omega.DEFAULT_MODEL == "claude-opus-4-8"


# ------------------------------------------------------------------ resolve_url
@pytest.mark.parametrize("given,expected", [
    ("", omega.GATEWAY_URL),
    ("   ", omega.GATEWAY_URL),
    # the form the Omega Plus docs print as "Anthropic-compatible URL"
    ("https://api.omegaplusapi.com/v1", "https://api.omegaplusapi.com/v1/messages"),
    ("https://api.omegaplusapi.com/v1/", "https://api.omegaplusapi.com/v1/messages"),
    # a full endpoint pasted verbatim must not gain a second /messages
    ("https://api.omegaplusapi.com/v1/messages",
     "https://api.omegaplusapi.com/v1/messages"),
    # self-hosted / proxy forms
    ("http://localhost:8080/v1", "http://localhost:8080/v1/messages"),
])
def test_resolve_url_accepts_either_docs_form(given, expected):
    assert omega.resolve_url(given) == expected


# ------------------------------------------------------------------ override plumbing
class _Recorder:
    """Captures the URL the transport was actually handed."""

    def __init__(self):
        self.urls = []

    def __call__(self, url, headers, body, timeout):
        self.urls.append(url)
        return 200, '{"content":[{"type":"text","text":"OK"}]}'


def test_omega_call_uses_the_override_url():
    rec = _Recorder()
    omega.call("oc_test", "sys", [{"role": "user", "content": "hi"}],
               post=rec, base_url="https://gw.example.com/v1")
    assert rec.urls == ["https://gw.example.com/v1/messages"]


def test_omega_call_without_override_uses_the_shipped_default():
    rec = _Recorder()
    omega.call("oc_test", "sys", [{"role": "user", "content": "hi"}], post=rec)
    assert rec.urls == [omega.GATEWAY_URL]


def test_providers_threads_base_url_to_omega_instead_of_dropping_it():
    """The regression: base_url was accepted then ignored for the omega provider."""
    rec = _Recorder()
    providers.call("omega", "oc_test", "sys", [{"role": "user", "content": "hi"}],
                   model="claude-opus-4-8", base_url="https://gw.example.com/v1",
                   post=rec)
    assert rec.urls == ["https://gw.example.com/v1/messages"]


def test_providers_omega_aliases_all_route_to_the_gateway():
    for alias in ("omega", "omega_plus", "omegaplus", "kesarcloud", "OMEGA-PLUS"):
        rec = _Recorder()
        providers.call(alias, "oc_test", "sys", [{"role": "user", "content": "hi"}],
                       model="claude-opus-4-8", post=rec)
        assert rec.urls == [omega.GATEWAY_URL], alias


def test_ping_honours_the_override():
    rec = _Recorder()
    omega.ping("oc_test", "claude-opus-4-8", post=rec,
               base_url="https://gw.example.com/v1")
    assert rec.urls == ["https://gw.example.com/v1/messages"]
