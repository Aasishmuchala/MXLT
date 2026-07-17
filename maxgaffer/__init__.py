"""MaxGaffer — the gaffer to MaxDirector's director.

Reference-driven lighting matching for 3ds Max 2026 + V-Ray 7 + Chaos Vantage: pick a
camera, hand it a reference image, and the plugin analyzes, first-guesses, and iteratively
matches the scene lighting — exposure/WB by histogram math, sun geometry and mood by a
vision LLM through the Omega gateway, judged by a deterministic tonal critic — while the
Vantage live link mirrors every step in real time.
"""

__version__ = "0.9.4"
