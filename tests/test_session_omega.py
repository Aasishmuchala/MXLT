import json
import shutil

import pytest

from maxgaffer.core import omega
from maxgaffer.core.genome import LightingState
from maxgaffer.core.session import Session, sidecar_path


# --------------------------------------------------------------------------- session
def test_sidecar_path():
    assert sidecar_path("D:/jobs/tula/tula_v12.max") == "D:/jobs/tula/tula_v12.maxgaffer.json"
    assert sidecar_path("") is None


def test_session_roundtrip(tmp_path):
    p = str(tmp_path / "scene.maxgaffer.json")
    s = Session(p, now_fn=lambda: "2026-07-16T12:00:00")
    st = LightingState()
    st.set("sun.altitude_deg", 6.0)
    st.groups["practicals"] = 0.5
    s.set_reference("PhysCam_Hero", "D:/refs/dusk.jpg")
    s.entry("PhysCam_Hero").locks = {"dome.intensity"}
    s.entry("PhysCam_Hero").semantics = {"time_of_day": "golden_hour"}
    s.record_match("PhysCam_Hero", st, 84.2)
    assert s.save()

    s2 = Session.load(p)
    e = s2.entry("PhysCam_Hero")
    assert e.reference == "D:/refs/dusk.jpg"
    assert e.score == 84.2
    assert e.matched_at == "2026-07-16T12:00:00"
    assert e.locks == {"dome.intensity"}
    assert e.state.get("sun.altitude_deg") == 6.0
    assert e.state.groups["practicals"] == 0.5
    assert s2.cameras_with_states() == ["PhysCam_Hero"]


def test_new_reference_invalidates_cached_analysis(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "a.jpg")
    s.entry("cam").semantics = {"time_of_day": "night"}
    s.entry("cam").score = 90.0
    s.set_reference("cam", "b.jpg")
    assert s.entry("cam").semantics == {}
    assert s.entry("cam").score is None
    s.set_reference("cam", "b.jpg")  # same ref → no reset
    s.entry("cam").semantics = {"k": 1}
    s.set_reference("cam", "b.jpg")
    assert s.entry("cam").semantics == {"k": 1}


def test_overwritten_reference_at_same_path_invalidates_cached_analysis(tmp_path):
    ref = tmp_path / "reference.jpg"
    ref.write_bytes(b"first")
    s = Session()
    s.set_reference("cam", str(ref))
    e = s.entry("cam")
    e.semantics = {"time_of_day": "day"}
    e.score = 88.0

    ref.write_bytes(b"a different image payload")
    s.set_reference("cam", str(ref))

    assert e.semantics == {}
    assert e.score is None
    assert e.reference_signature


def test_camera_identity_survives_rename_and_disambiguates_duplicates():
    s = Session()
    first = s.entry("Camera", "101")
    first.score = 77.0
    assert s.entry("Renamed Camera", "101") is first
    assert first.camera_name == "Renamed Camera" and first.score == 77.0
    second = s.entry("Renamed Camera", "202")
    assert second is not first
    assert second.camera_id == "202"


def test_project_relative_reference_survives_folder_move(tmp_path):
    source = tmp_path / "source"
    (source / "refs").mkdir(parents=True)
    ref = source / "refs" / "look.jpg"
    ref.write_bytes(b"image")
    sidecar = source / "scene.maxgaffer.json"
    s = Session(str(sidecar))
    s.set_reference("Cam", str(ref), "42")
    assert s.entry("Cam", "42").reference_relative == str(
        ref.relative_to(source))
    assert s.save()

    moved = tmp_path / "moved"
    shutil.move(str(source), str(moved))
    loaded = Session.load(str(moved / "scene.maxgaffer.json"))
    assert loaded.entry("Cam", "42").reference == str(moved / "refs" / "look.jpg")


def test_session_survives_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    s = Session.load(str(p))
    assert s.cameras == {}
    assert s.settings["apply_on_select"] is True


# --------------------------------------------------------------------------- omega
def test_parse_json_from_text_balanced():
    assert omega.parse_json_from_text('x {"a": {"b": "}"}} y') == {"a": {"b": "}"}}
    assert omega.parse_json_from_text("no json here") is None
    assert omega.parse_json_from_text('{"broken": ') is None
    assert omega.parse_json_from_text('junk {bad} then {"ok": 1}') == {"ok": 1}


def _payload(text):
    return json.dumps({"content": [{"type": "text", "text": text}]})


def test_call_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(omega, "BACKOFF_S", (0.0, 0.0, 0.0))
    calls = []

    def post(url, headers, body, timeout):
        calls.append(json.loads(body.decode()))
        return (429, "slow down") if len(calls) < 3 else (200, _payload("hello"))

    out = omega.call("oc_test", "system", [{"role": "user", "content": "hi"}], post=post)
    assert out == "hello"
    assert len(calls) == 3
    assert calls[0]["model"] == omega.DEFAULT_MODEL
    assert calls[0]["stream"] is False
    assert "tools" not in calls[0]          # the gateway 500s on tools — never send them


def test_call_auth_errors_do_not_retry():
    attempts = []

    def post(url, headers, body, timeout):
        attempts.append(1)
        return 401, "bad key"

    with pytest.raises(omega.OmegaError) as e:
        omega.call("oc_bad", "s", [], post=post)
    assert e.value.kind == "auth"
    assert len(attempts) == 1
    with pytest.raises(omega.OmegaError) as e2:
        omega.call("", "s", [], post=post)
    assert e2.value.kind == "auth"


def test_call_exhausts_retries(monkeypatch):
    monkeypatch.setattr(omega, "BACKOFF_S", (0.0,))
    with pytest.raises(omega.OmegaError) as e:
        omega.call("oc_x", "s", [], post=lambda *a: (503, "down"))
    assert e.value.kind == "network"


def test_image_block_from_file(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    block = omega.image_block_from_file(str(p))
    assert block["source"]["media_type"] == "image/png"
    jpg = tmp_path / "photo.JPG"
    jpg.write_bytes(b"\xff\xd8fake")
    assert omega.image_block_from_file(str(jpg))["source"]["media_type"] == "image/jpeg"
    assert omega.image_block_from_file(str(tmp_path / "x.exr")) is None
    assert omega.image_block_from_file(str(tmp_path / "missing.png")) is None
