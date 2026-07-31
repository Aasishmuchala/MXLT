"""Escalate uncertainty instead of guessing through it — the headless-safe question type.

Written 2026-07-31. On 2026-07-30 ANALYZE disagreed with itself about the sun's bearing by
±76° across samples, logged a warning, dropped its own trust to 25%, and then spent about
NINETY MINUTES probing 44 sun directions anyway. The artist was sitting right there and
could have answered in five seconds by looking at the photograph. Nothing in the pipeline
was able to ask.

This module is the type, not the dialog. It is pure: no Qt, no I/O, no imports from the
bridge. ``Controller.ask`` is a plain callable (``lambda q: ""`` by default) that the dock
replaces with a QMessageBox, exactly the way ``Controller.io`` is already wired — that
seam is proven and it keeps core headless. ``api.py`` and every existing script leave
``ask`` unset, and ``cfg.uncertainty_policy`` then decides:

    "ask"    the dock prompts; without a dock this degrades to the question's own default
    "assume" log the default and proceed — what match_all uses, so a 45-camera overnight
             queue cannot stop on the first dialog
    "abort"  raise PreflightBlocked

Unknown values read as "ask", the same defensive default ``probe_backend`` already uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class Question:
    """One decision the tool cannot make honestly on its own.

    ``detail`` goes into the transcript VERBATIM, whether or not anyone is there to answer
    — a question the artist never sees must still leave the measured facts behind it in
    the log, or the escalation is just another silent degradation.
    """
    key: str                                    # "cost" | "sun_bearing" | "basin_floor" | …
    headline: str                               # one line: the decision
    detail: str = ""                            # the measured facts behind it
    #: (value, button label) — value is what _escalate returns and what run.json records
    options: Tuple[Tuple[str, str], ...] = ()
    default: str = ""                           # what a headless run answers
    facts: Dict = field(default_factory=dict)   # machine-readable, into run.json

    def values(self) -> Tuple[str, ...]:
        return tuple(v for v, _label in self.options)

    def label_for(self, value: str) -> str:
        for v, label in self.options:
            if v == value:
                return label
        return value


#: The keys this release can raise, so run.json readers and the dock can switch on a
#: closed set rather than on free text.
DECISIONS = ("cost", "sun_bearing", "basin_floor", "frozen_plates", "preflight")


def cost_question(minutes: float, renders: int, probe_seconds: float,
                  options: Tuple[Tuple[str, str], ...], detail: str) -> Question:
    """The I2 question: this run will cost about ``minutes``; is that acceptable?"""
    return Question(
        key="cost",
        headline=(f"this match will take about {minutes:.0f} minutes "
                  f"({renders} renders at {probe_seconds:.0f} s each)"),
        detail=detail, options=options, default="proceed",
        facts={"minutes": round(float(minutes), 1), "renders": int(renders),
               "probe_seconds": round(float(probe_seconds), 2)})
