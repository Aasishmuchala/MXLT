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
    full = {"key": 0.92, "color": 0.88, "direction": 0.86, "highlight": 0.90,
            "envelope": 0.45, "histogram": 0.40, "hue": 0.85}
    card = critic.scorecard(84.0, full, ceiling_proven=True)
    assert card["content_gap"] is True
    assert "histogram" in card["weakest"]
    assert "not a guarantee" in card["disclaimer"]
    assert card["confidence"] == "high"


def test_an_unmeasured_component_honestly_lowers_confidence():
    """Coverage is the share of weighted dimensions actually measured, so a reading that
    is missing one cannot claim the same confidence as a complete one. This fired for real
    when `highlight` joined the weights: stats produced before the sun-patch map exists
    carry six of seven components, and saying "high" about them would be a lie."""
    full = {"key": 0.92, "color": 0.88, "direction": 0.86, "highlight": 0.90,
            "envelope": 0.45, "histogram": 0.40, "hue": 0.85}
    partial = {k: v for k, v in full.items() if k != "highlight"}
    assert critic.scorecard(84.0, full)["confidence"] == "high"
    assert critic.scorecard(84.0, partial)["confidence"] == "medium"


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


def test_the_card_never_blames_content_while_the_sun_is_missing():
    """scorecard's diagnosis was computed from `direction` alone — the component critic.py
    itself documents as returning 0.922 for a sun 171 degrees out and 0.917 for one 13.5
    degrees out. For a render with NO directional light in it (highlight 0.0) but a healthy
    grid cosine, the card announced the residual was 'scene content/albedo/material
    distribution—not lighting alone' and set albedo_suspect, sending the artist off to fix
    materials at the exact moment the reference's sun was absent — which IS a lighting
    control, and a solvable one."""
    sunless = {"key": 0.95, "color": 0.90, "direction": 0.92, "highlight": 0.0,
               "envelope": 0.55, "histogram": 0.50, "hue": 0.90}
    card = critic.scorecard(70.0, sunless)
    assert any("direction" in reason for reason in card["likely_gap"]), card["likely_gap"]
    assert card["content_gap"] is False, (
        "a missing sun is not a content gap — it is the thing the optimizer exists to fix")

    # a genuine content gap — every LIGHTING axis healthy, tonal shape stubbornly wrong —
    # must still be reported as one
    real_gap = {"key": 0.95, "color": 0.90, "direction": 0.92, "highlight": 0.95,
                "envelope": 0.55, "histogram": 0.50, "hue": 0.90}
    assert critic.scorecard(70.0, real_gap)["content_gap"] is True
