"""Display-encode of OCIO-raw (linear) loop plates — the measured on-box fix (2026-07-24).

On an OCIO/ACES box Max's ``rt.save`` writes the LINEAR framebuffer (bitmap ``.gamma``
does not re-encode — measured), while metrics decodes plates as sRGB and the LLM is shown
the file. The fix: ``expose.display_encode_png`` (stdlib png_min read → sRGB OETF LUT →
png_min write) applied in ``_render_exposed`` when ``_verify_exposure_host`` measures the
~2× EV over-response that linear plates produce.
"""
import math
import os

import pytest

from maxgaffer.core import expose, png_min


# ------------------------------------------------------------------ png_min writer
def test_write_png_rgb_roundtrips_through_the_reader(tmp_path):
    rows = [[(x * 8 % 256, y * 16 % 256, (x + y) % 256) for x in range(20)]
            for y in range(10)]
    p = str(tmp_path / "rt.png")
    assert png_min.write_png_rgb(p, rows) == p
    back = png_min.read_png_rgb(p, max_dim=10_000)
    assert back == rows


def test_write_png_rgb_rejects_bad_input(tmp_path):
    p = str(tmp_path / "bad.png")
    assert png_min.write_png_rgb(p, []) is None
    assert png_min.write_png_rgb(p, [[]]) is None
    ragged = [[(0, 0, 0)] * 3, [(0, 0, 0)] * 2]
    assert png_min.write_png_rgb(p, ragged) is None


# ------------------------------------------------------------------ display encode
def test_display_encode_is_the_inverse_of_metrics_srgb_decode(tmp_path):
    """Encoding a linear plate then running metrics' sRGB→linear decode must recover the
    original linear values — that inverse relationship is the entire point of the fix."""
    from maxgaffer.core.metrics import _srgb_to_linear

    rows = [[(v, v, v) for v in (0, 12, 46, 118, 200, 255)]]
    p = str(tmp_path / "lin.png")
    assert png_min.write_png_rgb(p, rows)
    assert expose.display_encode_png(p, p) == p
    encoded = png_min.read_png_rgb(p, max_dim=10_000)
    for (orig, _, _), (enc, _, _) in zip(rows[0], encoded[0]):
        recovered = _srgb_to_linear(enc / 255.0) * 255.0
        assert abs(recovered - orig) <= 1.5, (orig, enc, recovered)


def test_display_encode_lifts_linear_midgray_and_pins_endpoints(tmp_path):
    rows = [[(0, 0, 0), (46, 46, 46), (255, 255, 255)]]
    p = str(tmp_path / "mid.png")
    assert png_min.write_png_rgb(p, rows)
    assert expose.display_encode_png(p, p) == p
    back = png_min.read_png_rgb(p, max_dim=10_000)[0]
    assert back[0][0] == 0
    assert 115 <= back[1][0] <= 130          # linear 0.18 → sRGB ≈ 0.46
    assert back[2][0] == 255


def test_display_encode_missing_file_degrades_to_none(tmp_path):
    assert expose.display_encode_png(str(tmp_path / "absent.png"), str(tmp_path / "o.png")) is None


# ------------------------------------------------------------------ controller branch
class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, m):
        self.lines.append(str(m))


def _controller():
    from maxgaffer.maxbridge.config import Config
    from maxgaffer.maxbridge.controller import Controller

    return Controller(Config(api_key=""))


def _fake_evcheck(monkeypatch, ctrl, stops_apart):
    """Route _verify_exposure_host's two probe renders through synthetic stats that sit
    ``stops_apart`` apart in log_key, with the exposure host itself faked out."""
    from maxgaffer.maxbridge import controller as cmod

    class _Host:
        def __init__(self, cam):
            pass

        def read_ev(self):
            return 10.0

        def write_ev(self, v):
            return "ev"

    keys = iter([0.30, 0.30 / (2.0 ** stops_apart)])
    monkeypatch.setattr(cmod.rd, "render_frame", lambda cam, p, w, h: p)
    monkeypatch.setattr(cmod.metrics, "compute_stats", lambda p: {"log_key": next(keys)})
    import maxgaffer.maxbridge.exposure as exp_mod

    monkeypatch.setattr(exp_mod, "ExposureHost", _Host)


def test_over_response_sets_plate_linear_not_software_exposure(monkeypatch, tmp_path):
    ctrl = _controller()
    _fake_evcheck(monkeypatch, ctrl, stops_apart=3.96)   # the measured OCIO value
    log = _Log()
    ctrl._verify_exposure_host(object(), str(tmp_path), log)
    assert getattr(ctrl, "_plate_linear", False) is True
    assert ctrl.cfg.software_exposure is False
    assert any("linear" in ln for ln in log.lines)


def test_healthy_response_sets_neither_flag(monkeypatch, tmp_path):
    ctrl = _controller()
    _fake_evcheck(monkeypatch, ctrl, stops_apart=2.0)    # textbook-healthy host
    log = _Log()
    ctrl._verify_exposure_host(object(), str(tmp_path), log)
    assert getattr(ctrl, "_plate_linear", False) is False
    assert ctrl.cfg.software_exposure is False
    assert log.lines == []


def test_inert_response_still_turns_software_exposure_on(monkeypatch, tmp_path):
    ctrl = _controller()
    _fake_evcheck(monkeypatch, ctrl, stops_apart=0.2)    # display-stage-only host
    log = _Log()
    ctrl._verify_exposure_host(object(), str(tmp_path), log)
    assert ctrl.cfg.software_exposure is True
    assert getattr(ctrl, "_plate_linear", False) is False


def test_render_exposed_encodes_only_when_plate_linear(monkeypatch, tmp_path):
    ctrl = _controller()
    calls = []
    from maxgaffer.maxbridge import controller as cmod

    monkeypatch.setattr(cmod.rd, "render_frame", lambda cam, p, w, h: p)
    monkeypatch.setattr(cmod.expose, "display_encode_png",
                        lambda s, d: calls.append((s, d)) or d)
    out = str(tmp_path / "f.png")
    ctrl._render_exposed(object(), out, 160, 90)
    assert calls == []                                   # flag off → raw path untouched
    ctrl._plate_linear = True
    ctrl._render_exposed(object(), out, 160, 90)
    assert calls == [(out, out)]                         # flag on → in-place encode
