# MaxGaffer — whole-concept stress test (2026-07-17, v0.9.5)

Attack the premise, the physics, the workflow, and the dependencies — then either fix,
pin to the on-box checklist, or schedule. Nothing here is hand-waved: every finding is
FIXED (with a regression test), CHECKED (an on-box measurement exists), SCOPED (a
documented limit), or SCHEDULED (v1.x with a reason it can wait).

## 1. Premise attacks

**The critic is a proxy.** Tonal envelope + color mood + coarse direction grids transfer
across scenes; "99/100" is a statement about statistics, not about a human saying "yes".
→ SCHEDULED + INSTRUMENTED NOW: every match writes `run.json` (scores per iteration,
params, stop reason) into its run folder — after P1/P2, correlate accepted shots vs
critic scores and recalibrate `target_score`/weights on real data. The proxy becomes a
promise only through that trail.

**Color management is the biggest untested assumption.** The whole solver stack assumes
the saved loop PNG looks like what the artist sees in the VFB. Max 2024+ ships OCIO
color management and V-Ray 7 can run ACEScg — if `rt.save(bitmap)` writes raw ACEScg
(no display transform), every histogram the solver sees is dark/flat vs the sRGB
reference, and EV/WB solve with a systematic bias while looking "converged".
→ CHECKED: on-box checklist **#19** (Spike E): render a probe, open the PNG next to the
VFB — if they differ, set Max's color management to apply the display transform on save
(or note the transform; a fixed OCIO correction in the stats loader is a one-liner once
measured). This is the first thing to eyeball on the box.

**Dome-seed physics: the double-sun.** A live VRaySun plus a baked sun disc in the seeded
dome doubles the direct energy and casts a second soft shadow — the exact mistake sunless
commercial HDRIs exist to avoid. → FIXED: hybrid rule — `parametric_sun_active` skips the
disc (ambient-only seed); disc-bearing seeds only for sunless/disabled-sun rigs.
Regression-tested.

**Seed in visible reflections** stays soft-focus by design (blurred illumination, not
picture). → SCOPED: README documents keep-dome-camera-invisible for glossy hero shots.

**Albedo trap** (white-room scene vs walnut reference) remains the honest irreducible
edge. → SCOPED since v0.2: leash + diagnosis + lock-EV workflow; deep match proves the
ceiling and says "content gap, not lighting".

## 2. Production-workflow attacks

**Undo flood.** A deep match + polish is 130+ scene applies; recording each as an undo
step pushes the artist's own work out of a finite undo stack. → FIXED: loop/probe/board
applies are now undo-free (`apply_state(undo=False)`); the official revert for
explorations is the pre-match snapshot + Restore (as designed). Manual paths (sliders,
presets, adopt, camera switch, plans) keep normal undo.

**Phone references arrive rotated.** EXIF orientation was ignored — a portrait reference
read sideways, poisoning the direction grid AND what the vision model saw.
→ FIXED: `ImageOps.exif_transpose` in both the stats loader and the LLM image path.
Regression-tested with a synthetic EXIF-rotated JPEG.

**Gateway down must degrade, never abort.** Previously ANALYZE raised and killed the run.
→ FIXED: `_analyze_or_fallback` (cached analysis → neutral base semantics), the deltas
hook short-circuits after the first failure (no 3×-backoff per iteration), the sun sweep
falls back to the **contrastive-metric winner** (direction still gets solved with zero
LLM), and the dock skips a failed plan and continues the loop. Analytic-only mode =
solver EV/WB + metric sweep + critic keep-best — a real match, minus semantics finesse.

**HEIC references** (iPhone default) aren't readable by Pillow-without-plugins or Max.
→ SCOPED: file dialog filters to jpg/png/webp; docs say convert. v1.x: optional
pillow-heif in the sidecar venv.

**Scene units.** Plan placement defaults (distance 200u) assume metric-ish units; an
inch-unit scene puts fills ~5m out, a mm scene 20cm. → SCHEDULED v1.x: scale placement
defaults by `rt.units.SystemScale`. The preview dialog shows distances, so today the
human gate catches it.

**Heavy-scene cost.** 480×270 loop renders on a GI-heavy interior can still run minutes;
a deep match is 10 iterations + ≤120 probes. → SCOPED: the levers exist (resolution,
iterations, sweep off, draft sampler, CPU/GPU); P1/P2 measure real per-shot cost before
any new machinery. Expectation set in plan.md.

**Animation is out of scope for 0.x.** Per-camera stills states; no keyframing, and light
jumps at camera cuts are by design (per-shot light, the TULA model). → SCHEDULED v1.x:
export a camera's state as keyframes / interpolate between states for sun studies.

**Multi-dome rigs, VRaySky-only environments, region renders, DR** — first-dome-wins,
dim-not-disable, and render-settings-untouched rules hold; DR/VRAM contention stays
checklist #14.

## 3. Dependency attacks

**KesarCloud gateway** is the only LLM path. → Mitigated by the offline mode above +
key borrowing from MaxDirector + `ping` in Settings. A second provider is out of scope
(same gateway serves the sibling tools).

**V-Ray build drift** (property spellings) — candidates tables + Spike A measure, one
cost per build. **Vantage parity**: the live link is a *monitor*; finals default to
V-Ray-in-Max. A Vantage-rendered final (batch queue) may shade slightly differently than
the matched V-Ray loop render — SCOPED in README; hero finals through V-Ray.

## 4. Fixed in this round (v0.9.5, all regression-tested)

1. EXIF orientation honored (stats + LLM images).
2. Hybrid double-sun rule (`parametric_sun_active` → ambient-only seed).
3. Undo-free loop applies.
4. Full gateway-down degradation (analyze fallback, deltas short-circuit, metric-only
   sweep, dock plan-skip).
5. `run.json` telemetry per run (`MatchResult.to_summary`).
6. `DEFAULT_SEMANTICS` completed to full ANALYZE shape (it doubles as the offline base).

## 5. On-box verification additions

- **#19 color pipeline** (see §1) — new, first priority on the box.
- #17 .hdr load without double-gamma, #18 spherical u-origin, two-camera seed-follow,
  re-seed cache defeat — all already in Spike E; Q/R automated in `onbox_spikes.py`.

## 6. v1.x ledger (needed, and why it can wait)

| Need | Why later |
|---|---|
| Critic↔human calibration pass over run.json corpus | needs P1/P2 real-shot data first |
| State→keyframe export / interpolation (sun studies, walkthroughs) | stills ship first |
| Unit-aware plan placement defaults | preview dialog gates it today |
| pillow-heif in sidecar (iPhone HEIC) | convert-on-import works today |
| Generative pano sidecar (DiffusionLight-class) behind `pano_path` | seam exists; floor seed is already useful |
| Preset cost profiles (Fast/Standard/Hero) | measure P1/P2 costs before naming numbers |
