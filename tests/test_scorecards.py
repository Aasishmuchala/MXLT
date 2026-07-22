from maxgaffer.core import critic
from maxgaffer.core.genome import LightingState
from maxgaffer.core.session import Session


def test_artist_preference_profiles_materially_change_the_judge():
    balanced = critic.weights_for("balanced")
    directional = critic.weights_for("direction")
    color = critic.weights_for("color_mood")
    assert directional["direction"] > balanced["direction"]
    assert color["color"] > balanced["color"] and color["hue"] > balanced["hue"]


def test_scorecard_exposes_proxy_limits_and_content_gap():
    card = critic.scorecard(84.0, {
        "key": 0.92, "color": 0.88, "direction": 0.86,
        "envelope": 0.45, "histogram": 0.40, "hue": 0.85,
    }, ceiling_proven=True)
    assert card["content_gap"] is True
    assert "histogram" in card["weakest"]
    assert "not a guarantee" in card["disclaimer"]
    assert card["confidence"] == "high"


def test_artist_feedback_persists_the_human_verdict(tmp_path):
    path = tmp_path / "scene.maxgaffer.json"
    session = Session(str(path), now_fn=lambda: "2026-07-22T10:00:00")
    st = LightingState()
    st.set("sun.intensity", 1.0)
    session.record_match("Cam", st, 94.0)
    session.entry("Cam").scorecard = {"score": 94.0, "disclaimer": "proxy"}
    item = session.record_artist_feedback("Cam", False, rating=2, note="wrong mood")
    assert item["accepted"] is False and item["rating"] == 2
    assert session.save()
    loaded = Session.load(str(path), now_fn=lambda: "later")
    assert loaded.entry("Cam").artist_feedback[-1]["note"] == "wrong mood"
