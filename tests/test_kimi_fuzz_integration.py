"""Agent F — adversarial fuzz / integration tests for the audit+hardening round.

Deterministic-seeded fuzzing (random.Random(7) et al.) of every hostile-input surface:

  1. Persistence: session sidecar (<scene>.maxgaffer.json), maxbridge config.json, and
     lighting presets — truncations, byte mutations, type confusions, NaN/Infinity/1e999
     literals, bounded deep nesting, unicode/control chars, huge arrays.
     Contracts: Session.load / config.load / preset_loads never raise; a load failure
     quarantines the file to a timestamped .corrupt backup (original bytes preserved)
     and blocks auto-save until an explicit save(force=True); a successful load never
     mutates the on-disk bytes.
  2. LLM wire: core.parse.validate_analysis / validate_deltas and
     core.planner.validate_plan — non-JSON, markdown-fenced JSON, wrong types,
     NaN/Infinity literals, 10MB strings, missing/extra keys. Only the documented
     ParseError may escape; validated numbers must be finite; plans stay capped/grounded.
  3. Image codecs: byte-level fuzz of a valid stdlib-built PNG and a valid
     hdr_min-written .hdr through metrics.compute_stats, domeseed.build_seed and
     sidecar.metrics_cli — only documented handled paths (None / stats dict / SeedError
     / JSON null), never a hang, never allocation beyond the configured caps
     (declared 100000x100000 bomb must complete fast).
  4. Round-trips: LightingState.from_dict(to_dict) idempotent on random valid states;
     session save -> load preserves cameras/locks/baselines.

The fuzzing found recursion escape and quadratic brace-flood defects. Both are fixed and
their original reproducers run below as normal, timed regression tests.

Everything here is stdlib-only (runs under Max's Python 3.11, no Pillow/numpy) and every
fuzz loop is bounded (<= 300 iterations, each sub-second by construction).
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import math
import os
import random
import struct
import time
import zlib

import pytest

from maxgaffer.core import domeseed, hdr_min, metrics, png_min
from maxgaffer.core.genome import PARAMS, GROUP_BOUNDS, LightingState, spec_for
from maxgaffer.core.omega import parse_json_from_text
from maxgaffer.core.parse import ParseError, validate_analysis, validate_deltas
from maxgaffer.core.planner import CREATABLE_LIGHTS, MAX_OPS, PLACEMENT_LIMITS, validate_plan
from maxgaffer.core.session import Session, preset_dumps, preset_loads
from maxgaffer.maxbridge import config as config_mod
from maxgaffer.sidecar import metrics_cli

FIXED_NOW = "2026-01-02T03:04:05"
_NOW = lambda: FIXED_NOW  # noqa: E731 deterministic quarantine stamps

N_SESSION = 300
N_CONFIG = 200
N_PRESET = 200
N_REPLY = 300
N_IMAGE = 300
N_ROUNDTRIP = 300

# --------------------------------------------------------------------------- helpers


def _rand_text(rng: random.Random, n: int) -> str:
    """Random unicode string — accents, CJK, emoji, zalgo marks, control-free
    (control bytes are injected at the BYTE level separately). Surrogates excluded
    so the test's own utf-8 writes can never fail."""
    pools = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t",
        "äöüßéèê日本語テストΩπрусский",
        "☀🌙⭐💡🔥",
        "̴̵̶̷̸̡̢̧̨̛̖̗̘̙̜̝",
        "\\\"'{}[],:",
    )
    out = []
    for _ in range(n):
        pool = pools[rng.randrange(len(pools))]
        ch = pool[rng.randrange(len(pool))]
        if 0xD800 <= ord(ch) <= 0xDFFF:      # pragma: no cover - defensive
            continue
        out.append(ch)
    return "".join(out)


def _rand_scalar(rng: random.Random):
    return rng.choice([
        lambda: None,
        lambda: True,
        lambda: False,
        lambda: rng.randint(-10**12, 10**12),
        lambda: rng.uniform(-1e6, 1e6),
        lambda: float("nan"),                            # dumps as bare NaN literal
        lambda: float("inf"),                            # dumps as Infinity
        lambda: float("-inf"),
        lambda: float("1e999"),                          # == inf; text spliced raw too
        lambda: 1e-320,
        lambda: _rand_text(rng, rng.randint(0, 40)),
        lambda: "",
    ])()


def _rand_json(rng: random.Random, depth: int = 0):
    """Random JSON-shaped value; nesting HARD-BOUNDED (<= 12) so json never hits the
    recursion limit — the >limit crash is BUG-A, covered by its own skipped repro."""
    if depth >= 12 or rng.random() < 0.45:
        return _rand_scalar(rng)
    if rng.random() < 0.5:
        return [_rand_json(rng, depth + 1) for _ in range(rng.randint(0, 6))]
    return {_rand_text(rng, rng.randint(1, 10)): _rand_json(rng, depth + 1)
            for _ in range(rng.randint(0, 6))}


def _mutate_bytes(data: bytes, rng: random.Random):
    """Byte-level mutation operators. Returns (bytes, tag)."""
    if not data:
        return data, "noop"
    op = rng.randrange(6)
    b = bytearray(data)
    if op == 0:                                          # truncation at random offset
        return bytes(b[: rng.randint(0, len(b))]), "truncate"
    if op == 1:                                          # bit flips
        for _ in range(rng.randint(1, 8)):
            i = rng.randrange(len(b))
            b[i] ^= 1 << rng.randrange(8)
        return bytes(b), "bitflip"
    if op == 2:                                          # byte sets (incl. control bytes)
        for _ in range(rng.randint(1, 12)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        return bytes(b), "byteset"
    if op == 3:                                          # splice hostile literal
        lit = rng.choice([b"NaN", b"Infinity", b"-Infinity", b"1e999",
                          b"\x00\x01\x1f\x7f", b"\xff\xfe invalid-utf8 \x80"])
        i = rng.randint(0, len(b))
        return bytes(b[:i]) + lit + bytes(b[i:]), "literal"
    if op == 4:                                          # duplicate a slice (structure tear)
        i = rng.randrange(len(b))
        j = min(len(b), i + rng.randint(1, 64))
        k = rng.randint(0, len(b))
        return bytes(b[:k]) + bytes(b[i:j]) + bytes(b[k:]), "dup"
    for _ in range(rng.randint(1, 16)):                  # delete random bytes
        if len(b) > 1:
            del b[rng.randrange(len(b))]
    return bytes(b), "delete"


def _base_camera_state(rng: random.Random) -> LightingState:
    st = LightingState()
    for p in PARAMS:
        if rng.random() < 0.7:
            st.set(p.key, rng.randint(0, 1) if p.hi == 1 else rng.uniform(p.lo, p.hi))
    for _ in range(rng.randint(0, 4)):
        st.set("group." + _rand_text(rng, rng.randint(1, 8)) or "g",
               rng.uniform(*GROUP_BOUNDS))
    return st


def _base_sidecar(rng: random.Random) -> dict:
    """A valid v1 sidecar payload."""
    cams = {}
    for i in range(rng.randint(1, 3)):
        st = _base_camera_state(rng)
        cams[f"Cam_{i}"] = {
            "reference": f"refs/ref_{i}.png",
            "state": st.to_dict() if rng.random() < 0.8 else None,
            "score": round(rng.uniform(0, 100), 3),
            "matched_at": FIXED_NOW,
            "locks": sorted(p.key for p in PARAMS if rng.random() < 0.2),
            "semantics": {"scene_type": "exterior", "confidence": 0.9},
            "pre_match": None,
            "notes": ["note one", "note two"],
            "seed_hdri": "",
            "pre_seed": {},
        }
    return {"version": 1, "cameras": cams,
            "settings": {"apply_on_select": True},
            "baselines": {f"Light_{i}": round(rng.uniform(0.1, 20), 4)
                          for i in range(3)}}


def _quarantine_backup(path: str):
    backups = glob.glob(path + ".*.corrupt")
    return backups[0] if backups else None


# ------------------------------------------------------------- 1. persistence fuzz
class TestSidecarFuzz:
    """Session.load must never raise and must never lose the pre-existing file."""

    @pytest.mark.parametrize("iteration", range(N_SESSION))
    def test_session_load_fuzz(self, tmp_path, iteration):
        rng = random.Random(7 * 100000 + iteration)
        doc = _base_sidecar(rng)
        # per-key type confusion: swap whole sections for random JSON values
        for key in ("version", "cameras", "settings", "baselines"):
            if rng.random() < 0.15:
                doc[key] = _rand_json(rng)
        if rng.random() < 0.10:                          # deep-but-safe nesting inside
            doc.setdefault("settings", {})               # a tolerated slot
            if isinstance(doc["settings"], dict):
                doc["settings"]["blob"] = _rand_json(rng)
        if rng.random() < 0.08:                          # huge arrays / wide dicts
            doc["baselines"] = {f"L{i}": rng.uniform(0.1, 10) for i in range(20000)}
        raw = json.dumps(doc).encode("utf-8")
        if rng.random() < 0.15:                          # whole-doc type confusion
            raw = json.dumps(_rand_json(rng)).encode("utf-8")
        data, _tag = _mutate_bytes(raw, rng)
        path = str(tmp_path / f"side_{iteration}.maxgaffer.json")
        with open(path, "wb") as f:
            f.write(data)

        s = Session.load(path, now_fn=_NOW)              # must NEVER raise

        assert isinstance(s, Session)
        if not os.path.exists(path):
            # quarantined: backup holds the EXACT original bytes, auto-save blocked
            backup = _quarantine_backup(path)
            assert backup is not None, "load failure must leave a .corrupt backup"
            with open(backup, "rb") as f:
                assert f.read() == data, "quarantine must preserve the original bytes"
            assert s._protect_existing
            assert s.save() is False, "auto-save must stay blocked after a load failure"
            assert not os.path.exists(path), "blocked save must not recreate the file"
        else:
            with open(path, "rb") as f:                  # successful load never writes
                assert f.read() == data, "Session.load must not mutate the sidecar"
            if s._protect_existing:                      # newer-format guard
                assert s.save() is False
                with open(path, "rb") as f:
                    assert f.read() == data
        # explicit save is always allowed and must produce a parseable sidecar
        assert s.save(force=True) is True
        with open(path, "rb") as f:
            reparsed = json.load(f)
        assert isinstance(reparsed, dict) and "cameras" in reparsed

    def test_session_valid_file_roundtrip_load_is_clean(self, tmp_path):
        rng = random.Random(7)
        path = str(tmp_path / "clean.maxgaffer.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_base_sidecar(rng), f)
        with open(path, "rb") as f:
            before = f.read()
        s = Session.load(path, now_fn=_NOW)
        assert not s._protect_existing
        with open(path, "rb") as f:
            assert f.read() == before
        assert _quarantine_backup(path) is None


class TestConfigFuzz:
    """config.load must never raise, never write, and keep dataclass field types."""

    @pytest.mark.parametrize("iteration", range(N_CONFIG))
    def test_config_load_fuzz(self, tmp_path, monkeypatch, iteration):
        rng = random.Random(11 * 100000 + iteration)
        doc = json.loads(json.dumps(config_mod.Config().__dict__))
        for key in list(doc):
            if rng.random() < 0.35:
                doc[key] = _rand_json(rng)               # type confusion per field
        if rng.random() < 0.2:
            doc[_rand_text(rng, 8)] = _rand_json(rng)    # unknown extra keys
        raw = json.dumps(doc).encode("utf-8")
        if rng.random() < 0.15:
            raw = json.dumps(_rand_json(rng)).encode("utf-8")
        data, _tag = _mutate_bytes(raw, rng)
        path = str(tmp_path / f"cfg_{iteration}.json")
        with open(path, "wb") as f:
            f.write(data)
        monkeypatch.setattr(config_mod, "CONFIG_PATH", path)

        cfg = config_mod.load()                          # must NEVER raise

        assert isinstance(cfg, config_mod.Config)
        defaults = config_mod.Config()
        for f_name, default in defaults.__dict__.items():
            value = getattr(cfg, f_name)
            if isinstance(default, bool):
                assert isinstance(value, bool)
            elif isinstance(default, int):
                assert isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                assert isinstance(value, (int, float))
            elif isinstance(default, str):
                assert isinstance(value, str)
            elif isinstance(default, dict):
                assert isinstance(value, dict)
        with open(path, "rb") as f:                      # load must never write/rename
            assert f.read() == data


class TestPresetFuzz:
    """preset_loads returns a clamped LightingState or None — never raises."""

    @pytest.mark.parametrize("iteration", range(N_PRESET))
    def test_preset_loads_fuzz(self, iteration):
        rng = random.Random(13 * 100000 + iteration)
        base = preset_dumps(_base_camera_state(rng), name="fuzz", now=FIXED_NOW)
        if rng.random() < 0.2:
            base = json.dumps(_rand_json(rng))
        data, _tag = _mutate_bytes(base.encode("utf-8"), rng)
        text = data.decode("utf-8", errors="replace")

        st = preset_loads(text)                          # must NEVER raise

        assert st is None or isinstance(st, LightingState)
        if st is not None:
            for key, value in {**st.values,
                               **{f"group.{k}": v for k, v in st.groups.items()}}.items():
                spec = spec_for(key)
                assert spec is not None and math.isfinite(value)
                if spec.wrap:
                    assert 0.0 <= value < 360.0
                else:
                    assert spec.lo <= value <= spec.hi


# ------------------------------------------------------------------ 2. LLM wire fuzz
_PLAN_CAT = {
    "renderer": {"gi_on", "gi_multiplier", "environment_gi_on"},
    "environment": {"env_map", "env_multiplier"},
    "exposure": {"ev", "white_balance"},
    "node:VRaySun001": {"on", "multiplier", "turbidity", "size_multiplier"},
    "node:VRayLight001": {"on", "multiplier", "color"},
}

_ANALYSIS_BASE = {
    "scene_type": "exterior", "time_of_day": "golden_hour", "sky": "clear",
    "sun_active": True, "sun_bearing_deg": 45.0, "sun_altitude_band": "golden",
    "light_quality": "hard", "wb_kelvin_estimate": 4300.0, "practicals_on": False,
    "atmosphere": "light_haze", "contrast_character": "moody",
    "key_notes": "warm low sun from camera right", "confidence": 0.8,
}

_DELTAS_BASE = {
    "assessment": "render is flatter and cooler than the reference",
    "changes": [{"param": "sun.azimuth_deg", "value": 30.0, "why": "swing sun right"},
                {"param": "dome.intensity", "value": 0.5, "why": "less ambient"}],
    "stop": False,
}

_PLAN_BASE = {
    "read": "scene is overcast-flat, reference is warm golden hour",
    "ops": [{"op": "set", "target": "node:VRaySun001", "prop": "turbidity",
             "value": 5.0, "why": "hazier warm sky"},
            {"op": "create_light", "light_type": "VRayLight_plane", "name": "MG_fill",
             "placement": {"bearing_deg": -70, "distance": 250, "height": 120},
             "aim_at_camera_target": True,
             "props": {"multiplier": 8.0, "color": [255, 230, 200]}, "why": "warm fill"}],
    "expects": "warm directional key with soft fill",
}


def _fuzz_reply(rng: random.Random, base_obj) -> str:
    """Text-level LLM-reply fuzz: fences, prose, truncation, brace splices, literals,
    whole-object type confusion. Nesting bounded — deep-nesting crash is BUG-A."""
    obj = base_obj
    r = rng.random()
    if r < 0.10:
        obj = _rand_json(rng)                            # full type confusion
    elif r < 0.25 and isinstance(base_obj, dict):        # per-key type confusion
        obj = dict(base_obj)
        for key in list(obj):
            if rng.random() < 0.4:
                obj[key] = _rand_json(rng)
    elif r < 0.30 and isinstance(base_obj, dict):        # strip keys / add extras
        obj = {k: v for k, v in base_obj.items() if rng.random() < 0.6}
        obj[_rand_text(rng, 6)] = _rand_json(rng)
    text = json.dumps(obj)
    m = rng.randrange(8)
    if m == 0:
        text = f"```json\n{text}\n```"                   # markdown fence
    elif m == 1:
        text = f"Sure! Here is the analysis:\n{text}\nHope that helps."
    elif m == 2:
        text = text[: rng.randint(0, len(text))]         # truncated mid-stream
    elif m == 3:
        text = _rand_text(rng, rng.randint(0, 200))      # pure prose, maybe no JSON
    elif m == 4:
        i = rng.randint(0, len(text))                    # hostile literal splice
        text = text[:i] + rng.choice(["NaN", "Infinity", "-Infinity", "1e999"]) + text[i:]
    elif m == 5:
        text = text.replace(" ", "{" * rng.randint(1, 3), rng.randint(0, 4))
    elif m == 6:
        text = text + " " * rng.randint(0, 5000)
    return text


class TestLLMWireFuzz:
    @pytest.mark.parametrize("iteration", range(N_REPLY))
    def test_validate_analysis_fuzz(self, iteration):
        rng = random.Random(17 * 100000 + iteration)
        text = _fuzz_reply(rng, _ANALYSIS_BASE)
        try:
            out = validate_analysis(text)                # only ParseError may escape
        except ParseError:
            return
        assert isinstance(out, dict)
        for key in ("scene_type", "time_of_day", "sky", "sun_altitude_band",
                    "light_quality", "atmosphere", "contrast_character", "key_notes"):
            assert isinstance(out[key], str)
        for key in ("sun_bearing_deg", "wb_kelvin_estimate", "confidence"):
            assert math.isfinite(out[key])
        assert len(out["key_notes"]) <= 400

    @pytest.mark.parametrize("iteration", range(N_REPLY))
    def test_validate_deltas_fuzz(self, iteration):
        rng = random.Random(19 * 100000 + iteration)
        text = _fuzz_reply(rng, _DELTAS_BASE)
        try:
            out = validate_deltas(text)                  # only ParseError may escape
        except ParseError:
            return
        assert isinstance(out["assessment"], str) and len(out["assessment"]) <= 500
        assert isinstance(out["stop"], bool)
        assert len(out["changes"]) <= 4
        for key, value in out["changes"].items():
            assert isinstance(key, str) and math.isfinite(value)

    @pytest.mark.parametrize("iteration", range(N_REPLY))
    def test_validate_plan_fuzz(self, iteration):
        rng = random.Random(23 * 100000 + iteration)
        text = _fuzz_reply(rng, _PLAN_BASE)
        try:
            ops, rejected, meta = validate_plan(text, _PLAN_CAT)
        except ParseError:
            return
        assert len(ops) <= MAX_OPS
        assert isinstance(rejected, list) and isinstance(meta, dict)
        created = set()
        for op in ops:
            if op["op"] == "set":
                assert isinstance(op["target"], str) and isinstance(op["prop"], str)
                assert op["target"] in _PLAN_CAT or op["target"][5:] in created
                v = op["value"]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert math.isfinite(v)
            else:
                assert op["op"] == "create_light"
                assert op["light_type"] in CREATABLE_LIGHTS
                created.add(op["name"])
                place = op["placement"]
                for key, (lo, hi) in PLACEMENT_LIMITS.items():
                    if key in place:
                        assert lo <= place[key] <= hi

    # ------------------------------------------------- size extremes (deterministic)
    def test_ten_mb_valid_json_string_value(self):
        text = json.dumps({"assessment": "x" * 10_000_000,
                           "changes": [{"param": "dome.intensity", "value": 1.0}]})
        t0 = time.perf_counter()
        out = validate_deltas(text)
        assert time.perf_counter() - t0 < 5.0
        assert len(out["assessment"]) <= 500 and out["changes"] == {"dome.intensity": 1.0}

    def test_ten_mb_prose_no_braces_raises_parseerror_fast(self):
        text = "the quick brown fox " * 500_000          # 10MB, zero braces
        t0 = time.perf_counter()
        with pytest.raises(ParseError):
            validate_analysis(text)
        assert time.perf_counter() - t0 < 5.0

    def test_open_brace_flood_bounded_smoke(self):
        """Bounded version of BUG-B: 1000 unbalanced braces must still finish
        (quadratic cost here ~0.02s). The >5s repro is the skipped test below."""
        t0 = time.perf_counter()
        assert parse_json_from_text("{" * 1000) is None
        assert time.perf_counter() - t0 < 2.0

    def test_markdown_fenced_and_prose_wrapped_json_parse(self):
        fenced = f"```json\n{json.dumps(_ANALYSIS_BASE)}\n```"
        out = validate_analysis(fenced)
        assert out["scene_type"] == "exterior"
        wrapped = f"Thinking out loud… {json.dumps(_DELTAS_BASE)} — done."
        assert validate_deltas(wrapped)["changes"]


# ---------------------------------------------------------------- 3. image byte fuzz
_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def _png_bytes(width: int, height: int) -> bytes:
    """A real, valid 8-bit RGB PNG (filter-0 scanlines), stdlib-only."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(((x * 37 + y * 11) % 256, (x * 5 + y * 91) % 256,
                        (x * 13 + y * 3) % 256))
    return (_SIG + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
            + _png_chunk(b"IEND", b""))


def _corrupt_png_chunk_length(data: bytes, rng: random.Random) -> bytes:
    """Poke a random chunk's declared length field (IHDR/IDAT/IEND)."""
    b = bytearray(data)
    offsets = []
    pos = len(_SIG)
    while pos + 8 <= len(b):
        (length,) = struct.unpack(">I", bytes(b[pos:pos + 4]))
        offsets.append(pos)
        pos += 12 + max(0, length)
        if len(offsets) > 8:
            break
    if offsets:
        i = rng.choice(offsets)
        b[i:i + 4] = struct.pack(">I", rng.choice([0, 1, 7, 2**31 - 1, 2**32 - 1,
                                                   rng.randrange(2**32)]))
    return bytes(b)


def _mutate_png(data: bytes, rng: random.Random):
    if rng.random() < 0.2:
        return _corrupt_png_chunk_length(data, rng), "chunklen"
    if rng.random() < 0.25:                              # declared-dimension corruption
        b = bytearray(data)                              # IHDR width/height @ 16..23
        b[16:20] = struct.pack(">I", rng.choice([0, 1, 2**32 - 1, 100000,
                                                 rng.randrange(2**32)]))
        b[20:24] = struct.pack(">I", rng.choice([0, 1, 2**32 - 1, 100000,
                                                 rng.randrange(2**32)]))
        return bytes(b), "dims"
    return _mutate_bytes(data, rng)


class TestImageFuzz:
    @pytest.mark.parametrize("iteration", range(N_IMAGE))
    def test_png_fuzz(self, tmp_path, iteration):
        rng = random.Random(29 * 100000 + iteration)
        data, _tag = _mutate_png(_png_bytes(8, 6), rng)
        path = str(tmp_path / f"fuzz_{iteration}.png")
        with open(path, "wb") as f:
            f.write(data)
        stats = metrics.compute_stats(path)              # None or stats — never raises
        assert stats is None or (isinstance(stats, dict) and stats["count"] > 0)
        rows = png_min.read_png_rgb(path)
        assert rows is None or (rows and rows[0])

    @pytest.mark.parametrize("iteration", range(N_IMAGE))
    def test_hdr_fuzz(self, tmp_path, iteration):
        rng = random.Random(31 * 100000 + iteration)
        valid = str(tmp_path / "seed_src.hdr")
        rows = [[(0.2 + 0.01 * x, 0.3, 0.1 * y) for x in range(12)]
                for y in range(4)]
        assert hdr_min.write_hdr(valid, rows)
        with open(valid, "rb") as f:
            data = f.read()
        os.unlink(valid)
        if rng.random() < 0.2:                           # corrupt the dimension line
            data = data.replace(b"-Y 4 +X 12",
                                rng.choice([b"-Y 0 +X 12", b"-Y -3 +X 12",
                                            b"-Y 999999 +X 12", b"-Y 4 +X 0",
                                            b"-Y 4 +X 100000", b"+Y 4 +X 12"]), 1)
        data, _tag = _mutate_bytes(data, rng)
        path = str(tmp_path / f"fuzz_{iteration}.hdr")
        with open(path, "wb") as f:
            f.write(data)
        out_hdr = str(tmp_path / f"out_{iteration}.hdr")
        read = hdr_min.read_hdr(path)                    # None or rows — never raises
        assert read is None or (read and read[0])
        try:
            meta = domeseed.build_seed(out_hdr, pano_path=path)
        except domeseed.SeedError:                       # documented handled path
            meta = None
        assert meta is None or (isinstance(meta, dict)
                                and os.path.exists(meta["path"]))
        try:                                             # same bytes as an LDR ref
            meta2 = domeseed.build_seed(out_hdr + ".2", ref_path=path)
        except domeseed.SeedError:
            meta2 = None
        assert meta2 is None or isinstance(meta2, dict)

    @pytest.mark.parametrize("iteration", range(100))
    def test_metrics_cli_fuzz(self, tmp_path, iteration):
        rng = random.Random(37 * 100000 + iteration)
        data, _tag = _mutate_png(_png_bytes(8, 6), rng)
        path = str(tmp_path / f"cli_{iteration}.png")
        with open(path, "wb") as f:
            f.write(data)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = metrics_cli.main([path])                # one corrupt image never
        assert rc == 0                                   # kills the batch / the CLI
        out = json.loads(buf.getvalue())
        assert isinstance(out, list) and len(out) == 1
        assert out[0]["stats"] is None or isinstance(out[0]["stats"], dict)

    def test_metrics_cli_b64_flag_never_raises(self, tmp_path):
        rng = random.Random(41)
        path = str(tmp_path / "b64.png")
        with open(path, "wb") as f:
            f.write(_mutate_png(_png_bytes(8, 6), rng)[0])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = metrics_cli.main([path, "--b64"])
        assert rc == 0
        entry = json.loads(buf.getvalue())[0]
        assert "b64" in entry or "b64_error" in entry    # Pillow present or absent

    # ------------------------------------------------- bombs must fail FAST + BOUNDED
    def test_png_declared_dimension_bomb_completes_fast(self, tmp_path):
        """Declared 100000x100000 with a tiny payload: rejected by the dimension cap
        before any allocation/decompression."""
        ihdr = struct.pack(">IIBBBBB", 100000, 100000, 8, 2, 0, 0, 0)
        bomb = (_SIG + _png_chunk(b"IHDR", ihdr)
                + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 64))
                + _png_chunk(b"IEND", b""))
        path = str(tmp_path / "bomb.png")
        with open(path, "wb") as f:
            f.write(bomb)
        t0 = time.perf_counter()
        assert png_min.read_png_rgb(path) is None
        assert metrics.compute_stats(path) is None
        assert time.perf_counter() - t0 < 2.0

    def test_png_raw_byte_cap_bomb_completes_fast(self, tmp_path):
        """16384x16384 RGBA declares ~1.07GB of raw pixels (> 256MB cap) -> rejected
        without inflating."""
        ihdr = struct.pack(">IIBBBBB", 16384, 16384, 8, 6, 0, 0, 0)
        bomb = (_SIG + _png_chunk(b"IHDR", ihdr)
                + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 128))
                + _png_chunk(b"IEND", b""))
        path = str(tmp_path / "bomb2.png")
        with open(path, "wb") as f:
            f.write(bomb)
        t0 = time.perf_counter()
        assert png_min.read_png_rgb(path) is None
        assert time.perf_counter() - t0 < 2.0

    def test_png_inflate_bomb_stays_within_cap(self, tmp_path):
        """Within-cap geometry (4000x4000 RGB = 48MB) whose IDAT inflates PAST the
        declared size: the bounded decompress must stop at the cap and reject."""
        width = height = 4000
        expected = height * (width * 3 + 1)
        payload = zlib.compress(b"\x00" * (expected + 1_000_000), 1)
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        bomb = (_SIG + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", payload)
                + _png_chunk(b"IEND", b""))
        path = str(tmp_path / "bomb3.png")
        with open(path, "wb") as f:
            f.write(bomb)
        t0 = time.perf_counter()
        assert png_min.read_png_rgb(path) is None        # unconsumed tail -> reject
        assert time.perf_counter() - t0 < 5.0

    def test_hdr_declared_dimension_bomb_completes_fast(self, tmp_path):
        bomb = (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 100000 +X 100000\n"
                + b"\x00" * 32)
        path = str(tmp_path / "bomb.hdr")
        with open(path, "wb") as f:
            f.write(bomb)
        t0 = time.perf_counter()
        assert hdr_min.read_hdr(path) is None
        assert metrics.compute_stats(path) is None
        assert time.perf_counter() - t0 < 2.0


# ------------------------------------------------------------------- 4. round-trips
class TestRoundTrip:
    @pytest.mark.parametrize("iteration", range(N_ROUNDTRIP))
    def test_lightingstate_roundtrip_idempotent(self, iteration):
        rng = random.Random(43 * 100000 + iteration)
        st = _base_camera_state(rng)
        d = st.to_dict()
        again = LightingState.from_dict(d).to_dict()
        assert again == d, "from_dict(to_dict) must be idempotent"
        via_json = LightingState.from_dict(json.loads(json.dumps(d))).to_dict()
        assert via_json == d

    @pytest.mark.parametrize("iteration", range(100))
    def test_session_save_load_roundtrip(self, tmp_path, iteration):
        rng = random.Random(47 * 100000 + iteration)
        path = str(tmp_path / f"rt_{iteration}.maxgaffer.json")
        s = Session(path, now_fn=_NOW)
        for i in range(rng.randint(1, 5)):
            name = f"Cam_{i}_{_rand_text(rng, 4)}"
            e = s.entry(name)
            e.reference = f"refs/{_rand_text(rng, 8)}.png"
            e.state = _base_camera_state(rng) if rng.random() < 0.8 else None
            e.score = round(rng.uniform(0, 100), 4) if rng.random() < 0.7 else None
            e.matched_at = FIXED_NOW
            e.locks = {p.key for p in PARAMS if rng.random() < 0.25}
            e.semantics = {"scene_type": "interior",
                           "note": _rand_text(rng, 12)}
            e.pre_match = _base_camera_state(rng) if rng.random() < 0.4 else None
            e.notes = [_rand_text(rng, 10) for _ in range(rng.randint(0, 3))]
            e.seed_hdri = "seed.hdr" if rng.random() < 0.3 else ""
            e.pre_seed = {"texture": "old.hdr"} if rng.random() < 0.3 else {}
        s.baselines = {f"Light_{i}": round(rng.uniform(0.1, 20), 5)
                       for i in range(rng.randint(0, 5))}
        s.settings["apply_on_select"] = bool(rng.random() < 0.5)
        before = {n: e.to_dict() for n, e in s.cameras.items()}
        assert s.save() is True

        s2 = Session.load(path, now_fn=_NOW)
        assert not s2._protect_existing
        assert set(s2.cameras) == set(before)
        for n, e in s2.cameras.items():
            assert e.to_dict() == before[n], f"camera {n!r} changed across save/load"
            assert e.locks == set(before[n]["locks"])
        assert s2.baselines == s.baselines
        assert s2.settings == s.settings


# ------------------------------------------------------- parser hardening regressions


def test_regression_deep_nesting_recursionerror_is_contained(tmp_path):
    payload = '{"a":' * 5000 + "1" + "}" * 5000
    path = str(tmp_path / "deep.maxgaffer.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
    s = Session.load(path, now_fn=_NOW)
    assert s._protect_existing               # corrupt input is quarantined, never overwritten
    assert _quarantine_backup(path) is not None


def test_regression_brace_flood_is_linear_and_fast():
    t0 = time.perf_counter()
    parse_json_from_text("{" * 20000)
    assert time.perf_counter() - t0 < 2.0, "quadratic blowup in parse_json_from_text"
