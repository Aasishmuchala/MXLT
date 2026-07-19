"""Closed-loop convergence proof — the ENTIRE match engine minus pymxs, live.

A procedural world renders images as a pure function of a LightingState (EV → brightness,
WB → tint, sun azimuth/altitude → lobe position, intensity → lobe strength). A hidden
TARGET state renders the "reference"; the engine starts far away and must converge using
the real pipeline: ANALYZE (live LLM) → rules first-guess → sun SWEEP (live LLM pick) →
iterate with the real solver, critic, guards and live DELTAS calls.

    python scripts/sim_match.py [oc_key]      (falls back to scripted-LLM if offline)

PASS criteria (deterministic legs, asserted):  EV within 0.5 stop of target · WB within
700 K · critic score improves ≥ 15 points. LLM legs (azimuth direction, altitude) are
reported, not asserted — taste is bounded, not deterministic.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxgaffer.core import metrics, omega, parse, prompts, rules  # noqa: E402
from maxgaffer.core.director import Hooks, MatchConfig, run_match, run_sun_sweep  # noqa: E402
from maxgaffer.core.genome import LightingState  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_gateway_smoke import discover_key  # noqa: E402

TARGET = {"sun.enabled": 1, "sun.azimuth_deg": 210.0, "sun.altitude_deg": 8.0,
          "sun.intensity": 1.0, "sun.size": 4.0, "sun.turbidity": 4.0,
          "exposure.ev": 11.5, "exposure.wb_kelvin": 5200.0}
START = {"sun.enabled": 1, "sun.azimuth_deg": 40.0, "sun.altitude_deg": 55.0,
         "sun.intensity": 1.0, "sun.size": 1.0, "sun.turbidity": 3.0,
         "exposure.ev": 14.0, "exposure.wb_kelvin": 7500.0}
CAMERA_YAW = 180.0   # camera looks south → the target sun (az 210) sits just camera-right


def state_of(d):
    st = LightingState()
    for k, v in d.items():
        st.set(k, v)
    return st


class World:
    """Tiny physically-flavored renderer: consistent enough that matching the target state
    reproduces the reference exactly — so convergence is measurable, not vibes."""

    def __init__(self, out_dir):
        self.out = out_dir
        self.n = 0

    def render(self, st: LightingState, tag: str) -> str:
        from PIL import Image, ImageDraw

        ev = st.get("exposure.ev", 11.5)
        wb = st.get("exposure.wb_kelvin", 6500.0)
        az = st.get("sun.azimuth_deg", 0.0)
        alt = st.get("sun.altitude_deg", 30.0)
        inten = st.get("sun.intensity", 1.0)
        m = 2.0 ** (11.5 - ev)                     # exposure: higher EV = darker
        r_mul = 1.0 + (wb - 6500.0) * 6e-5        # higher kelvin = warmer render
        b_mul = 1.0 - (wb - 6500.0) * 6e-5
        rel = math.radians((az - CAMERA_YAW + 180.0) % 360.0 - 180.0)  # bearing from view
        sun_x = 160 + math.sin(rel) * 150
        sun_y = 130 - max(0.0, min(90.0, alt)) / 90.0 * 115
        sun_vis = abs(math.degrees(rel)) < 100 and alt > -2

        # low sun = golden sky, CONTINUOUS falloff (real V-Ray skies have no warmth cliff;
        # a hard cutoff at 18° created a false ridge that trapped coordinate descent)
        golden = max(0.0, (26.0 - alt)) * 2.2 if sun_vis else 0.0
        im = Image.new("RGB", (320, 180))
        px = im.load()
        for y in range(180):
            t = y / 179.0
            base = (200 - 120 * t + golden * (1 - t),
                    185 - 115 * t + golden * 0.45 * (1 - t),
                    190 - 100 * t)                                  # sky→haze gradient
            for x in range(320):
                glow = 0.0
                if sun_vis:
                    d2 = ((x - sun_x) ** 2 + (y - sun_y) ** 2) / (90.0 ** 2)
                    glow = inten * 200.0 * math.exp(-d2)
                r = (base[0] + glow * 1.15) * m * r_mul
                g = (base[1] + glow * 0.95) * m
                b = (base[2] + glow * 0.55) * m * b_mul
                px[x, y] = (int(min(255, max(0, r))), int(min(255, max(0, g))),
                            int(min(255, max(0, b))))
        d = ImageDraw.Draw(im)
        gnd = tuple(int(min(255, 55 * m * c)) for c in (r_mul, 1.0, b_mul))
        d.rectangle([0, 132, 320, 180], fill=gnd)
        bld = tuple(int(min(255, 85 * m * c)) for c in (r_mul, 1.0, b_mul))
        d.rectangle([210, 62, 285, 132], fill=bld)
        if sun_vis:   # cast shadow away from the sun — the direction cue real renders have
            shadow_len = min(140.0, 18.0 + (90.0 - alt) * 1.4)
            sx = -math.copysign(shadow_len, math.sin(rel))
            shade = tuple(int(c * 0.45) for c in gnd)
            d.polygon([(210, 132), (285, 132), (285 + sx, 152), (210 + sx, 152)],
                      fill=shade)
        self.n += 1
        path = os.path.join(self.out, f"{self.n:02d}_{tag}.png")
        im.save(path)
        return path


class ScriptedLLM:
    """Offline fallback so the deterministic legs still prove out without a gateway."""

    def analyze(self):
        return json.dumps({"time_of_day": "golden_hour", "sky": "hazy", "sun_active": True,
                           "sun_bearing_deg": 30.0, "sun_altitude_band": "golden",
                           "light_quality": "soft", "wb_kelvin_estimate": 5000.0,
                           "practicals_on": False, "atmosphere": "light_haze",
                           "contrast_character": "balanced", "key_notes": "scripted",
                           "confidence": 0.5})

    def deltas(self, ctx):
        return json.dumps({"assessment": "scripted no-op", "changes": [], "stop": False})

    def sweep(self, n):
        return json.dumps({"best_index": min(2, n - 1), "altitude_hint": "golden",
                           "why": "scripted"})


def phase_a_solver_only(world, ref_path, ref_stats) -> bool:
    """ASSERTED: proper experimental control — geometry already matches the target (the
    LLM's job is excluded), ONLY exposure + WB are wrong. The solver owns exactly those
    two, so it must land them on the hidden target with a no-op LLM."""
    print("\n===== PHASE A — solver-only convergence, geometry controlled (asserted) =====")
    a_start = dict(TARGET)
    a_start["exposure.ev"] = START["exposure.ev"]              # 14.0  (2.5 stops dark)
    a_start["exposure.wb_kelvin"] = START["exposure.wb_kelvin"]  # 7500 (too warm-set)
    scripted = ScriptedLLM()
    current = {"st": state_of(a_start)}
    hooks = Hooks(
        apply=lambda st: current.__setitem__("st", st.copy()),
        render=lambda tag: world.render(current["st"], "A_" + tag),
        stats=metrics.compute_stats,
        llm_deltas=lambda ctx: scripted.deltas(ctx),
        log=lambda m: print("   ·", m),
    )
    res = run_match(state_of(a_start), ref_stats, {}, hooks,
                    MatchConfig(max_iterations=6, target_score=99.0))
    if res.best_score is None:
        # cancelled/failed before any successful render — a clean FAIL, not a TypeError
        print("A: no score produced (render or stats failed before iteration 0) — FAIL ✗")
        return False
    ev_f = res.best_state.get("exposure.ev")
    wb_f = res.best_state.get("exposure.wb_kelvin")
    first = next((r.score for r in res.iterations if r.score is not None), 0.0)
    print(f"A: score {first:.1f} → {res.best_score:.1f} · EV {ev_f:.2f} (target 11.50) · "
          f"WB {wb_f:.0f}K (target 5200)")
    ok = (abs(ev_f - TARGET["exposure.ev"]) <= 0.5
          and abs(wb_f - TARGET["exposure.wb_kelvin"]) <= 700.0
          and res.best_score - first >= 15.0)
    print("PHASE A:", "PASS ✓" if ok else "FAIL ✗")
    return ok


def phase_c_deep_match_99(world, ref_path, ref_stats) -> bool:
    """ASSERTED — the 99 claim in the regime where it's true: the reference IS reachable
    (same world) and geometry sits within the convergence basin — which is what the live
    LLM loop demonstrably delivers (Phase B's model jumped altitude 55→4 in one iteration).
    From there the DETERMINISTIC legs alone (solver + adaptive polish, no-op LLM) must
    close to ≥99. Polish is a basin-finisher by design, not a global searcher."""
    print("\n===== PHASE C — deep match to 99 from the basin, deterministic only "
          "(asserted) =====")
    scripted = ScriptedLLM()
    basin = dict(TARGET)
    basin.update({"sun.azimuth_deg": 203.0,    # post-sweep + one LLM nudge off 210
                  "sun.altitude_deg": 14.0,    # golden band, target 8
                  "sun.size": 2.0,             # target 4
                  "exposure.ev": 13.2,         # 1.7 stops dark
                  "exposure.wb_kelvin": 6800.0})   # 1600K cool
    current = {"st": state_of(basin)}
    hooks = Hooks(
        apply=lambda st: current.__setitem__("st", st.copy()),
        render=lambda tag: world.render(current["st"], "C_" + tag),
        stats=metrics.compute_stats,
        llm_deltas=lambda ctx: scripted.deltas(ctx),
        log=lambda m: print("   ·", m),
    )
    res = run_match(state_of(basin), ref_stats, {}, hooks,
                    MatchConfig(max_iterations=5, target_score=99.0, stall_patience=3,
                                polish=True, polish_stop_at=99.0))
    print(f"C: {res.best_score:.2f} (polish +{res.polish_gain:.2f} over "
          f"{res.polish_probes} probes) · sun az {res.best_state.get('sun.azimuth_deg'):.0f}°"
          f" (target 210) · EV {res.best_state.get('exposure.ev'):.2f} · "
          f"WB {res.best_state.get('exposure.wb_kelvin'):.0f}K")
    ok = res.best_score is not None and res.best_score >= 99.0
    print("PHASE C:", "PASS ✓ (99 reached from the basin)" if ok else "FAIL ✗")
    return ok


def main() -> int:
    key = discover_key()
    live = bool(key)
    tmp = tempfile.mkdtemp(prefix="maxgaffer_sim_")
    world = World(tmp)
    target = state_of(TARGET)
    ref_path = world.render(target, "REFERENCE")
    ref_stats = metrics.compute_stats(ref_path)
    scripted = ScriptedLLM()
    print(f"mode: {'LIVE gateway' if live else 'scripted LLM (offline)'} · frames → {tmp}")
    print(f"hidden target: ev {TARGET['exposure.ev']} · wb {TARGET['exposure.wb_kelvin']:.0f}K "
          f"· sun az {TARGET['sun.azimuth_deg']:.0f}°/alt {TARGET['sun.altitude_deg']:.0f}°")

    phase_a_ok = phase_a_solver_only(world, ref_path, ref_stats)
    phase_c_ok = phase_c_deep_match_99(world, ref_path, ref_stats)
    if not live:
        return 0 if (phase_a_ok and phase_c_ok) else 1
    print("\n===== PHASE B — full pipeline with LIVE gateway (reported) =====")

    def img(path):
        return omega.image_block_from_file(path)

    # ---------- ① ANALYZE (3-sample self-consistency, as the controller now runs it)
    from maxgaffer.core.consensus import consolidate_analyses

    samples = []
    for _ in range(3 if live else 1):
        reply = (omega.call(key, prompts.ANALYZE_SYSTEM,
                            [{"role": "user", "content": [
                                img(ref_path),
                                omega.text_block(prompts.analyze_user_text())]}],
                            max_tokens=2048) if live else scripted.analyze())
        try:
            samples.append(parse.validate_analysis(reply))
        except parse.ParseError:
            continue
    semantics = consolidate_analyses(samples)
    agreement = semantics.pop("consensus_agreement", 1.0)
    print(f"\n① analyze ×{len(samples)} (agreement {agreement:.0%}): "
          f"{semantics['time_of_day']}, bearing {semantics['sun_bearing_deg']:+.0f}°, "
          f"band {semantics['sun_altitude_band']}, "
          f"wb~{semantics['wb_kelvin_estimate']:.0f}K")

    # ---------- ② RULES first guess
    start, why = rules.initial_state(semantics, state_of(START), CAMERA_YAW)
    print(f"② first guess: az {start.get('sun.azimuth_deg'):.0f}° "
          f"alt {start.get('sun.altitude_deg'):.0f}° wb {start.get('exposure.wb_kelvin'):.0f}K")

    logs = []

    def log(m):
        logs.append(m)
        print("   ·", m)

    current = {"st": start.copy()}
    hooks = Hooks(
        apply=lambda st: current.__setitem__("st", st.copy()),
        render=lambda tag: world.render(current["st"], tag),
        stats=metrics.compute_stats,
        llm_deltas=lambda ctx: (omega.call(
            key, prompts.DELTAS_SYSTEM,
            [{"role": "user", "content": [
                img(ref_path), img(ctx["render_path"]),
                omega.text_block(prompts.deltas_user_text(
                    ctx["state_table"], ctx["semantics"], ctx["score_history"],
                    ctx["analytic_applied"], ctx["iteration"], ctx["max_iterations"]))]}],
            max_tokens=2048) if live else scripted.deltas(ctx)),
        log=log,
    )

    # ---------- ③ SWEEP
    def sweep_pick(paths, azs):
        if not live:
            return scripted.sweep(len(paths))
        content = [img(ref_path)] + [img(p) for p in paths]
        content.append(omega.text_block(prompts.sweep_user_text(azs)))
        return omega.call(key, prompts.SWEEP_SYSTEM,
                          [{"role": "user", "content": content}], max_tokens=1024)

    az, hint, _why = run_sun_sweep(start, [0.0, 90.0, 180.0, 270.0], hooks, sweep_pick,
                                   ref_stats=ref_stats)
    if az is not None:
        start.set("sun.azimuth_deg", az)
        if hint != "na":
            start.set("sun.altitude_deg", rules.ALTITUDE_DEG[hint])
        err = abs((az - TARGET["sun.azimuth_deg"] + 180) % 360 - 180)
        print(f"③ sweep: picked az {az:.0f}° (target 210° → error {err:.0f}°), hint '{hint}'")

    # ---------- ④ ITERATE (real director, real solver, real critic, live deltas)
    print("④ iterate:")
    result = run_match(start, ref_stats, semantics, hooks,
                       MatchConfig(max_iterations=6, target_score=99.0, polish=True,
                                   polish_stop_at=99.0))

    # ---------- Phase B report (observed, not asserted — the LLM leg is bounded taste)
    first = next((r.score for r in result.iterations if r.score is not None), None)
    ev_f = result.best_state.get("exposure.ev")
    wb_f = result.best_state.get("exposure.wb_kelvin")
    az_f = result.best_state.get("sun.azimuth_deg")
    az_err = abs((az_f - TARGET["sun.azimuth_deg"] + 180) % 360 - 180)
    print(f"\n=== PHASE B report ({result.stop_reason}) ===")
    print(f"score:   {first} → {result.best_score}")
    print(f"EV:      {START['exposure.ev']:.1f} → {ev_f:.2f}   (target 11.50)")
    print(f"WB:      {START['exposure.wb_kelvin']:.0f} → {wb_f:.0f}K (target 5200)")
    print(f"sun az:  {START['sun.azimuth_deg']:.0f}° → {az_f:.0f}°  (target 210°, "
          f"error {az_err:.0f}°)")
    if result.polish_probes:
        print(f"polish:  +{result.polish_gain:.2f} over {result.polish_probes} probes"
              + (" · ceiling proven" if getattr(result, "ceiling_proven", False)
                 else (" · plateau" if result.ceiling_converged else "")))
    ok = phase_a_ok and phase_c_ok
    print("\nOVERALL:", "PASS ✓ (phases A + C asserted)" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
