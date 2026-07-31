"""The two exceptions the honesty layer raises — and why they subclass ``RuntimeError``.

Written 2026-07-31 after the TULA post-mortem. Two things must be able to stop a match
from anywhere: the artist pressing ✕ (``MatchCancelled``), and a preflight finding that
makes every number the run could produce a non-measurement (``PreflightBlocked``).

Both subclass ``RuntimeError`` DELIBERATELY. Every existing caller in this plugin already
catches ``(OmegaError, RuntimeError)`` — ``api.py``, ``ui/dock.py``'s match/refine/board
handlers, ``controller._analyze_or_fallback`` — so a new exception type of its own would
escape as an unhandled crash through code that has been correct for months. Subclassing
keeps those handlers working unchanged; the three sites that MISREPORT a cancel (they say
"plan skipped", "gateway unavailable" and then carry on spending renders) each gained an
explicit ``except MatchCancelled: raise`` placed FIRST, which is a smaller and more
auditable change than re-typing the exception everywhere.

``MatchCancelled`` carries the reason string the latch was set with, so a legitimate
RuntimeError raised inside a probe can never be mistaken for the artist's ✕.
"""

from __future__ import annotations


class MatchCancelled(RuntimeError):
    """The artist asked the run to stop and a render boundary noticed.

    Raised from ``Controller._render_exposed`` — the ONE function every render in the
    plugin passes through — so a cancel is bounded by exactly one probe no matter which
    loop is running. See controller._render_exposed and ui/dock._cancel_match.
    """


class PreflightBlocked(RuntimeError):
    """The scene cannot produce a measurement, so the run must not spend the artist's time.

    Only six conditions raise this (preflight.py's BLOCK list): a missing / undecodable /
    100% black reference, distributed rendering pointed at a dead port, an unwritable run
    directory, and the three canary failures. Everything else warns and continues —
    ``cfg.preflight_level = "warn"`` downgrades even these, loudly.
    """
