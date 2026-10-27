# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The adversarial grid (plan §8.3): combined corners of the valid domain.

Every fixture is rendered by the shipping path and compared with its reference
at 128 × base (6.144 MHz for the 48 kHz family), whose convergence is checked
against 256 × base, and against 512 × base on every tenth fixture. The results
fix the internal rate and the constant of the validated-domain rule.

    uv run python -m tools.grid run --k 32 [--family NAME ...] [--out docs/grid/k32.csv]
    uv run python -m tools.grid analyze docs/grid/k16.csv docs/grid/k32.csv
    uv run python -m tools.grid list [--family NAME] [--verbose]

A run appends to its CSV and skips fixtures already there, so it can resume.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from brightwork_synth import carrier, cli, patch
from brightwork_synth.reference import CHECK, DEEP, REFERENCE, compare, reference
from brightwork_synth.render import LIMIT, QUIET, SATURATION, synthesize

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

type Knots = list[dict[str, float]]  # a curve as TOML parses it
type Source = dict[str, Any]  # a [[noise]] or [[tone]] table
type Edits = dict[str, Any]  # fields set by patch path
type Fixture = tuple[str, str, Edits]  # family, name, edits
type Row = dict[str, Any]  # one fixture's results
type Add = Callable[..., None]  # add(family, name, **edits)

ROOT = Path(__file__).resolve().parent.parent
INF = math.inf
FLOOR = 1e-7
LIMIT_DB = -100.0  # the rendering accuracy every fixture inside the domain meets
CONVERGED_DB = -120.0  # how far below that a reference must agree with its check
DEEP_FLOOR_DB = -200.0  # below which two references agree to rounding
ROUNDING = 1e-12  # of the peak: rounding, which neither term need cover; a stored sound's playback rounds near 1e-13
RANDOM_FIXTURES = 220
REPORT_EVERY = 25  # fixtures between progress lines
DEEP_EVERY = 10  # every tenth fixture is also checked at 512 × base
TOP_HZ = 20000.0
DAMPED = 0.2  # the share of random fixtures that a damper chokes
RATTLING = 0.3  # the share of random sources that rattle
STORING = 0.2  # the share of random fixtures that are stored


def say(text: str) -> None:
    """Print a line of progress at once."""
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()


def amp(*pts: tuple[float, float]) -> Knots:
    """Write knots as TOML parses them."""
    return [{"ms": float(m), "gain": float(g)} for m, g in pts]


def partials(*pairs: tuple[float, float]) -> Knots:
    """Write partials as TOML parses them."""
    return [{"hz": float(f), "level_db": float(v)} for f, v in pairs]


# Shapes: a closed hat's plates, the stick's tick, an open hat, and a crash that rings for seconds.
PLATES = amp((0, 0), (0.3, 1), (8, 0.45), (30, 0.14), (70, 0.04), (120, 0))
TICK = amp((0, 0), (0.03, 1), (0.13, 0.6), (1.5, 0.1), (5, 0))
OPEN = amp((0, 0), (0.6, 1), (60, 0.5), (250, 0.16), (520, 0.05), (850, 0.016), (1300, 0))
RING = amp((0, 0), (2, 1), (180, 0.55), (700, 0.2), (1500, 0.06), (2500, 0.018), (3500, 0.005), (4500, 0))
LONG = amp((0, 0), (2, 1), (300, 0.5), (1500, 0.15), (4000, 0.03), (6500, 0.006), (8000, 0))
ANALOG = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)
STRUCK = amp((0, 0), (0.02, 1), (0.12, 0.7), (20, 0.12), (60, 0))  # a tone's shape unless one is given


def noise(level: float, band: Sequence[float], spacing: float = 14.0, shape: Knots = PLATES, **fields: float) -> Source:
    """Write a noise source; `fields` sets damping (0.5 unless set), bloom, chatter, chatter_hz (220 Hz) and rattle."""
    source: Source = {
        "level_db": float(level),
        "band_hz": [float(band[0]), float(band[1])],
        "spacing_hz": float(spacing),
        "damping": 0.5,
        "bloom": 0.0,
        "chatter": 0.0,
        "chatter_hz": 220.0,
        "grains_hz": 0.0,
        "grain_ms": 10.0,
        "amp": copy.deepcopy(shape),
    }
    for key, value in fields.items():
        if key not in ("damping", "bloom", "chatter", "chatter_hz", "grains_hz", "grain_ms"):
            msg = f"a noise source has no field {key!r} to set"
            raise TypeError(msg)
        source[key] = float(value)
    return source


def tone(
    level: float,
    hz: Sequence[float],
    band: Sequence[float],
    shape: Knots | None = None,
    levels: Sequence[float] | None = None,
    **fields: object,
) -> Source:
    """Write a tone; `fields` sets wave ("square" unless set), phase ("free"), damping (0.3), bloom and rattle."""
    source: Source = {
        "level_db": float(level),
        "wave": "square",
        "phase": "free",
        "partials": partials(*zip(hz, levels or [0.0] * len(hz), strict=True)),
        "band_hz": [float(band[0]), float(band[1])],
        "damping": 0.3,
        "bloom": 0.0,
        "chatter": 0.0,
        "chatter_hz": 220.0,
        "grains_hz": 0.0,
        "grain_ms": 10.0,
        "amp": copy.deepcopy(shape or STRUCK),
    }
    for key, value in fields.items():
        if key in ("wave", "phase"):
            source[key] = str(value)
        elif key in ("damping", "bloom", "grains_hz", "grain_ms") and isinstance(value, (int, float)):
            source[key] = float(value)
        else:
            msg = f"a tone has no field {key!r} to set to {value!r}"
            raise TypeError(msg)
    return source


# The plan's closed hat at unity output gain.
BASE = {
    "instrument": "cymbal",
    "schema": 1,
    "seed": 1,
    "take": 1,
    "noise": [noise(-2, (6000, 16000)), noise(-10, (3000, 18000), 0.0, TICK, damping=0.0)],
    "storage": {"bits": 0, "rate_hz": 32000.0, "filter_hz": 16000.0},
    "damper": {"amp": []},
    "drive": {"db": 0.0},
    "output": {"sample_rate": 48000, "format": "float32", "mode": "gain", "db": 0.0, "polarity": 1},
}

BANDS = {
    "hat": (6000, 16000),
    "hat2": (8000, 18000),
    "crash": (300, 18000),
    "dark": (500, 5000),
    "top": (19000, 20000),
    "narrowest": (5000, 5001),
}


def set_path(d: dict[str, Any], path: str, value: object) -> None:
    """Set a field of parsed TOML by its patch path, counting from 1, such as "noise[2].amp"."""
    node: Any = d
    parts = path.split(".")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        if m is None:
            msg = f"{path}: not a patch path"
            raise ValueError(msg)
        name, index = m.group(1), m.group(2)
        last = i == len(parts) - 1
        if index is None:
            if last:
                node[name] = value
            else:
                node = node[name]
        elif last:
            node[name][int(index) - 1] = value
        else:
            node = node[name][int(index) - 1]


def patch_dict(edits: Edits) -> dict[str, Any]:
    """Return the base patch with the edits applied, as TOML parses it."""
    d = copy.deepcopy(BASE)
    for path, value in edits.items():
        set_path(d, path, copy.deepcopy(value))
    return d


def valid(edits: Edits) -> bool:
    """Tell whether the edits make a valid patch."""
    try:
        patch.from_dict(patch_dict(edits))
    except patch.PatchError:
        return False
    return True


def fixtures() -> list[Fixture]:
    """Return every fixture, (family, name, edits): combined corners and a random sample of the valid domain."""
    out: list[Fixture] = []

    def add(family: str, name: str, **edits: object) -> None:
        if valid(edits):
            out.append((family, name, edits))

    for family in (_noise, _damping, _chatter, _ticks, _tones, _chokes, _layers, _dynamics, _rattles, _storage):
        family(add)
    _seeds(add, out)
    _rates(add)
    _random(out)
    _longest(add)
    return out


def _noise(add: Add) -> None:
    """Add bands × spacing × (level, drive), one source with the closed hat's plates."""
    pairs = ((0, 0), (0, 12), (0, 18), (0, 24), (0, 36), (0, 48), (12, 0), (12, 12), (24, 0), (24, 24))
    for bname, band in BANDS.items():
        for spacing in (0, 1, 16, 200):
            for level, drive in pairs:
                add(
                    "noise",
                    f"{bname}-s{spacing}-L{level}-d{drive}",
                    noise=[noise(level, band, spacing)],
                    **{"drive.db": float(drive)},
                )


def _damping(add: Add) -> None:
    """Add damping and bloom 0–3 each, from 20 Hz, 300 Hz and 6 kHz, on short and long shapes.

    The valid domain's limit on compression at 20 kHz rejects the rest.
    """
    for f_ref in (20, 300, 6000):
        for damping in (0, 1, 1.6, 2, 3):
            for bloom in (0, 1, 1.6, 3):
                for sname, shape, spacing in (("short", PLATES, 0.0), ("long", LONG, 16.0)):
                    for drive in (0, 24):
                        add(
                            "damping",
                            f"f{f_ref}-c{damping:g}-b{bloom:g}-{sname}-d{drive}",
                            noise=[noise(0, (f_ref, 20000), spacing, shape, damping=damping, bloom=bloom)],
                            **{"drive.db": float(drive)},
                        )


def _chatter(add: Add) -> None:
    """Add chatter's depth × rate × drive over the hat's plates."""
    for depth in (0.25, 0.5, 1.0):
        for rate in (10, 220, 1000, 4000):
            for drive in (0, 12, 24):
                add(
                    "chatter",
                    f"c{depth:g}-r{rate}-d{drive}",
                    **{"noise[1].level_db": 0.0, "noise[1].chatter": depth, "noise[1].chatter_hz": float(rate)},
                    **{"noise[2].level_db": -INF, "drive.db": float(drive)},
                )


def _rattles(add: Add) -> None:
    """Add collision rates × rings over a shaker's shell, a tambourine's jingles and the whole band, level × drive."""
    shake = amp((0, 0), (20, 1), (60, 0.6), (150, 0.1), (250, 0))
    bands = {"shell": ((3000, 9000), 0.0), "jingles": ((2500, 12000), 120.0), "bright": ((1000, 20000), 0.0)}
    pairs = ((0, 0), (0, 12), (0, 24), (12, 0), (12, 12))
    for rate in (20, 200, 1000, 4000):
        for ring in (1, 6, 30, 200):
            for bname, (band, spacing) in bands.items():
                for level, drive in pairs:
                    add(
                        "rattles",
                        f"g{rate}-r{ring}-{bname}-L{level}-d{drive}",
                        noise=[noise(level, band, spacing, shake, damping=0.0, grains_hz=rate, grain_ms=ring)],
                        **{"drive.db": float(drive)},
                    )
    # A tambourine: jingles and a struck head sharing their collisions, under chatter.
    jingles = noise(0, (2500, 12000), 120.0, shake, damping=0.5, chatter=0.3, grains_hz=150, grain_ms=80)
    head = tone(-6, (180.0, 290.0), (100, 2000), shake, wave="sine", phase="strike", grains_hz=150, grain_ms=30)
    for drive in (0, 12, 24):
        add("rattles", f"tambourine-d{drive}", noise=[jingles], tone=[head], **{"drive.db": float(drive)})


def _storage(add: Add) -> None:
    """Add bits × storage rate × playback filter over the hat's plates, undriven and under 24 dB of drive.

    And a long ring, choked, and the analog squares, each stored.
    """
    for bits in (4, 6, 8, 12):
        for rate in (12000, 32000, 64000):
            for cutoff in (4000, 12000, 20000):
                for drive in (0, 24):
                    add(
                        "storage",
                        f"b{bits}-r{rate}-f{cutoff}-d{drive}",
                        noise=[noise(0, (6000, 16000))],
                        **{"storage.bits": bits, "storage.rate_hz": float(rate), "storage.filter_hz": float(cutoff)},
                        **{"drive.db": float(drive)},
                    )
    stored = {"storage.bits": 6, "storage.rate_hz": 32000.0, "storage.filter_hz": 12000.0}
    for drive in (0, 24):
        choked = amp((60, 1), (65, 0.3), (75, 0))
        add(
            "storage",
            f"choked-d{drive}",
            noise=[noise(0, (6000, 16000), shape=LONG)],
            **stored,
            **{"damper.amp": choked, "drive.db": float(drive)},
        )
        add(
            "storage",
            f"analog-d{drive}",
            noise=[],
            tone=[tone(0, ANALOG, (7000, 16000))],
            **stored,
            **{"drive.db": float(drive)},
        )


def _ticks(add: Add) -> None:
    """Add 10–50 µs rises under full-scale bright noise and dense modes."""
    for h in (0.01, 0.02, 0.05):
        for bname, band in (("bright", (1000, 20000)), ("hat", (6000, 16000)), ("hat2", (8000, 18000))):
            for spacing in (0, 1):
                for level in (0, 12):
                    for drive in (0, 12, 24):
                        shape = amp((0, 0), (h, 1), (h + 0.1, 0.7), (2, 0.1), (6, 0))
                        add(
                            "ticks",
                            f"h{h * 1000:g}us-{bname}-s{spacing}-L{level}-d{drive}",
                            noise=[noise(level, band, spacing, shape, damping=0.0)],
                            **{"drive.db": float(drive)},
                        )


def _tones(add: Add) -> None:
    """Add square banks at the analog frequencies and eight times higher, and struck sines up to 20 kHz."""
    for octave in (1, 8):
        hz = [f * octave for f in ANALOG]
        for bname, band in (("analog", (7000, 16000)), ("wide", (300, 20000)), ("mid", (2000, 9000))):
            for drive in (0, 12, 24, 36):
                add(
                    "tones",
                    f"square-x{octave}-{bname}-d{drive}",
                    noise=[],
                    tone=[tone(0, hz, band)],
                    **{"drive.db": float(drive)},
                )
    for rise in (0.01, 0.02, 0.05):
        for drive in (0, 12, 24):
            shape = amp((0, 0), (rise, 1), (rise + 0.1, 0.6), (8, 0.1), (40, 0))
            sines = tone(
                0,
                (5000, 11000, 19500),
                (20, 20000),
                shape,
                (0, -3, -6),
                wave="sine",
                phase="strike",
                damping=1.0,
            )
            add("tones", f"sine-r{rise * 1000:g}us-d{drive}", noise=[], tone=[sines], **{"drive.db": float(drive)})


def _chokes(add: Add) -> None:
    """Add dampers from `choke`, with releases of 1–100 ms, over ringing sources."""
    ringing = patch.from_dict(patch_dict({"noise": [noise(0, (400, 18000), 18.0, RING, damping=0.9, bloom=0.4)]}))
    for at in (50, 300):
        for release in (1, 5, 20, 100):
            choked = patch.choke(ringing, float(at), float(release))
            for drive in (0, 24):
                add(
                    "chokes",
                    f"at{at}-r{release}-d{drive}",
                    noise=patch.to_dict(choked)["noise"],
                    **{"damper.amp": patch.to_dict(choked)["damper"]["amp"], "drive.db": float(drive)},
                )


def _layers(add: Add) -> None:
    """Add four coherent sources at +24 dB, and crossfades between spacings and bands."""
    four = [
        noise(24, (6000, 16000), 14.0),
        noise(24, (3000, 18000), 0.0, TICK, damping=0.0),
        noise(24, (2000, 9000), 16.0, OPEN, damping=1.0),
        noise(24, (500, 5000), 200.0, PLATES, damping=0.0),
    ]
    for drive in (0, 12, 24, 36, 48):
        add("layers", f"four-L24-d{drive}", noise=four, **{"drive.db": float(drive)})
    fade_out = amp((0, 0), (1, 1), (40, 0.5), (120, 0))
    fade_in = amp((0, 0), (20, 0.1), (80, 1), (240, 0))
    for level in (0, 12):
        for drive in (0, 12, 24, 36):
            crossfade = [
                noise(level, (3000, 12000), 0.0, fade_out, damping=0.0),
                noise(level, (1000, 16000), 16.0, fade_in),
            ]
            add("layers", f"crossfade-L{level}-d{drive}", noise=crossfade, **{"drive.db": float(drive)})


def _dynamics(add: Add) -> None:
    """Add `softer` ladders from every reference, at unity output gain."""
    for name in cli.REFERENCES:
        ref = patch.loads(cli.reference_text(name))
        for db in (0, 6, 12, 18, 24, 30, 36):
            try:
                d = patch.to_dict(patch.softer(ref, db))
            except patch.PatchError:
                continue  # `softer` refuses rather than clamps; the references are voiced to take the ladder
            d["output"]["db"] = 0.0
            fields = {k: d[k] for k in ("seed", "take", "storage", "damper", "drive", "output")}
            add("dynamics", f"{name}-softer{db}", noise=d.get("noise", []), tone=d.get("tone", []), **fields)


def _seeds(add: Add, out: list[Fixture]) -> None:
    """Add one fixture of each family across seeds 1–4 × takes 1–4."""
    picks = [
        ("noise", "hat-s16-L0-d18"),
        ("damping", "f300-c1-b1-short-d24"),
        ("chatter", "c0.5-r220-d12"),
        ("ticks", "h20us-hat-s1-L0-d12"),
        ("tones", "square-x1-analog-d12"),
        ("layers", "crossfade-L0-d24"),
    ]
    named = {(f, n): e for f, n, e in out}
    for fam, name in picks:
        for seed in range(1, 5):
            for take in range(1, 5):
                if (seed, take) != (1, 1):
                    add("seeds", f"{fam}-{name}-s{seed}-t{take}", **named[(fam, name)], seed=seed, take=take)


def _rates(add: Add) -> None:
    """Add the export rates that share their family's internal rate."""
    for rate in (44100, 88200, 96000):
        for drive in (0, 24, 36):
            add("rates", f"sr{rate}-d{drive}", **{"output.sample_rate": rate, "drive.db": float(drive)})
            add(
                "rates",
                f"sr{rate}-hot-d{drive}",
                **{"output.sample_rate": rate, "drive.db": float(drive), "noise[1].level_db": 6.0},
            )


def _random(out: list[Fixture]) -> None:
    """Add a seeded random sample of the whole valid domain, every fixture sounding."""
    rng = np.random.default_rng(2026)
    rattles = np.random.default_rng(2027)  # its own stream, so the fixtures drawn before rattles keep their draws
    stores = np.random.default_rng(2028)  # and storage too
    made = 0
    while made < RANDOM_FIXTURES:
        e = random_edits(rng)
        if stores.uniform() < STORING:
            e["storage.bits"] = int(stores.integers(2, 17))
            e["storage.rate_hz"] = float(10 ** stores.uniform(math.log10(4000), math.log10(96000)))
            e["storage.filter_hz"] = float(10 ** stores.uniform(3, math.log10(20000)))
        for source in [*e["noise"], *e["tone"]]:
            if rattles.uniform() < RATTLING:
                ring = float(10 ** rattles.uniform(0, 2.3))
                source["grain_ms"] = ring
                source["grains_hz"] = min(float(10 ** rattles.uniform(1, 3.3)), 1000 * patch.MAX_OVERLAP / ring)
        if valid(e):
            out.append(("random", f"r{made:03d}", e))
            made += 1


def _longest(add: Add) -> None:
    """Add the longest sounds: 8 s rides and crashes with three sources, at 96 kHz."""
    long_sources = {
        "ride": [
            noise(-6, (500, 16000), 15.0, LONG, damping=0.8, bloom=0.15),
            noise(-4, (3000, 9000), 40.0, amp((0, 0), (0.1, 1), (0.25, 0.7), (25, 0.15), (80, 0)), damping=1.0),
            noise(-14, (2000, 18000), 0.0, TICK, damping=0.0),
        ],
        "crash": [
            noise(-1, (400, 18000), 18.0, LONG, damping=0.9, bloom=0.4),
            noise(-10, (2000, 18000), 0.0, amp((0, 0), (0.05, 1), (0.15, 0.6), (3, 0.1), (10, 0)), damping=0.0),
            noise(
                -12,
                (300, 6000),
                0.0,
                amp((0, 0), (5, 1), (400, 0.3), (2000, 0.05), (8000, 0)),
                damping=0.3,
                bloom=1.0,
            ),
        ],
    }
    for name, srcs in long_sources.items():
        for drive in (0, 12, 24):
            add("longest", f"{name}-8s-96k-d{drive}", noise=srcs, **{"output.sample_rate": 96000, "drive.db": drive})


def random_shape(rng: np.random.Generator, span: float) -> Knots:
    """Draw an amplitude shape: a rise of 10 µs to 3 ms, then one to four falling knots over about `span` ms."""
    rise = float(10 ** rng.uniform(-2, 0.5))
    knots = [(0.0, 0.0), (rise, 1.0)]
    t, g = rise, 1.0
    for _ in range(int(rng.integers(1, 5))):
        t += float(rng.uniform(0.05, 0.6)) * span
        g *= float(rng.uniform(0.1, 0.9))
        knots.append((t, g))
    knots.append((t + float(rng.uniform(0.1, 0.5)) * span, 0.0))
    return amp(*knots)


def _band(rng: np.random.Generator) -> tuple[float, float]:
    """Draw a band, log-uniformly placed, up to 2.5 decades wide."""
    lo = float(10 ** rng.uniform(math.log10(20), math.log10(19000)))
    return lo, float(min(20000.0, lo * 10 ** rng.uniform(0.001, 2.5)))


def _bends(rng: np.random.Generator, lo: float) -> tuple[float, float]:
    """Draw damping and bloom within what the valid domain allows from `lo`."""
    top = math.log(1000) / math.log(TOP_HZ / lo) if lo < TOP_HZ else 3.0
    damping = float(rng.choice([0.0, rng.uniform(0, min(3.0, top))]))
    bloom = float(rng.choice([0.0, 0.0, rng.uniform(0, min(3.0, top))]))
    return damping, bloom


def _level(rng: np.random.Generator) -> float:
    """Draw a source level: silent one time in seven or so, else −40 to +24 dB."""
    return float(rng.choice([-INF, rng.uniform(-40, 24)], p=[0.15, 0.85]))


def _random_tone(rng: np.random.Generator) -> Source:
    """Draw a tone of one to six partials, its loudest at 0 dB."""
    count = int(rng.integers(1, 7))
    f0 = float(10 ** rng.uniform(math.log10(20), math.log10(4000)))
    ratios = np.concatenate(([1.0], rng.uniform(1.01, 5.0, count - 1)))
    hz = [f for f in sorted({round(f0 * float(r), 3) for r in ratios}) if f <= TOP_HZ]
    levels = [float(x) for x in rng.uniform(-40, 0, len(hz))]
    levels[int(rng.integers(len(hz)))] = 0.0
    lo, hi = _band(rng)
    damping, bloom = _bends(rng, max(lo, hz[0]))
    span = float(10 ** rng.uniform(0.7, 3.4))
    level = _level(rng)
    wave, phase = str(rng.choice(["sine", "square"])), str(rng.choice(["strike", "free"]))
    shape = random_shape(rng, span)
    return tone(level, hz, (lo, hi), shape, levels, wave=wave, phase=phase, damping=damping, bloom=bloom)


def random_edits(rng: np.random.Generator) -> Edits:
    """Draw one fixture over the valid domain; shapes span up to about 2.5 s, which the longest family exceeds."""
    sources = []
    for _ in range(int(rng.integers(0, 5))):
        lo, hi = _band(rng)
        damping, bloom = _bends(rng, lo)
        spacing = float(rng.choice([0.0, rng.uniform(1, 200)]))
        chatter = float(rng.choice([0.0, rng.uniform(0, 1)], p=[0.7, 0.3]))
        span = float(10 ** rng.uniform(0.7, 3.4))
        level = _level(rng)
        shape = random_shape(rng, span)
        rate = float(rng.uniform(10, 4000))
        sources.append(
            noise(level, (lo, hi), spacing, shape, damping=damping, bloom=bloom, chatter=chatter, chatter_hz=rate),
        )
    tones = [_random_tone(rng) for _ in range(int(rng.choice([0, 0, 1, 2])))]
    if not sources and not tones:
        sources.append(noise(0.0, _band(rng), 14.0, PLATES))
    every = sources + tones
    if all(s["level_db"] == -INF for s in every):
        every[0]["level_db"] = 0.0  # a silent fixture would test nothing
    damper = []
    if rng.uniform() < DAMPED:
        start = float(rng.uniform(1, 400))
        fall = float(10 ** rng.uniform(-1.5, 2))
        damper = amp((start, 1), (start + fall / 2, 0.1), (start + fall, 0))
    return {
        "seed": int(rng.integers(0, 2**31)),
        "take": int(rng.integers(1, 1000)),
        "noise": sources,
        "tone": tones,
        "damper.amp": damper,
        "drive.db": float(rng.choice([0.0, rng.uniform(0, 48)])),
        "output.sample_rate": int(rng.choice([44100, 48000, 88200, 96000])),
    }


FIELDS = [
    "family",
    "name",
    "k",
    "seed",
    "take",
    "drive",
    "alpha",
    "e_rel",
    "e_max",
    "max_abs",
    "peak",
    "c_rel",
    "c_max",
    "c_abs",
    "c3_rel",
    "c3_max",
    "c3_abs",
    "knee",
    "kink",
    "length_ms",
    "ship_s",
    "total_s",
    "edits",
]


def run_one(job: tuple[str, str, Edits, int, bool]) -> Row:
    """Render one fixture by the shipping path and its references, and measure them."""
    family, name, edits, k, deep = job
    p = patch.from_dict(patch_dict(edits))
    t0 = time.perf_counter()
    ship = synthesize(p, k=k)
    t_ship = time.perf_counter() - t0
    ref = reference(p, REFERENCE, k=k, workers=1)
    check = reference(p, CHECK, k=k, workers=1)
    e, c = compare(ship.out, ref.out), compare(ref.out, check.out)
    row: Row = {
        "family": family,
        "name": name,
        "k": k,
        "seed": p.seed,
        "take": p.take,
        "drive": p.drive.db,
        "alpha": ship.alpha,
        "e_rel": e.rel_db,
        "e_max": e.max_db,
        "max_abs": e.max_abs,
        "peak": e.peak,
        "c_rel": c.rel_db,
        "c_max": c.max_db,
        "c_abs": c.max_abs,
        "knee": ship.slew.knee(ship.alpha),
        "kink": ship.kink,
        "length_ms": p.length_ms,
        "ship_s": t_ship,
        "edits": json.dumps(edits),
    }
    del ref
    if deep:
        c3 = compare(check.out, reference(p, DEEP, k=k, workers=1).out)
        row.update(c3_rel=c3.rel_db, c3_max=c3.max_db, c3_abs=c3.max_abs)
    row["total_s"] = time.perf_counter() - t0
    return row


def _write(out: Path, rows: Iterable[dict[str, str]]) -> None:
    """Write rows to a fresh CSV."""
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def run(args: argparse.Namespace) -> None:
    """Run every fixture not yet in the CSV, longest first, in parallel processes, appending each result."""
    todo = [f for f in fixtures() if not args.family or f[0] in args.family]
    out = Path(args.out or ROOT / f"docs/grid/k{args.k}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        # Rows whose fixture has changed since, such as a revoiced reference's ladder, are dropped and run again.
        current = {(fam, name): json.dumps(edits) for fam, name, edits in fixtures()}
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        kept = [r for r in rows if current.get((r["family"], r["name"])) == r["edits"]]
        if len(kept) < len(rows):
            say(f"dropping {len(rows) - len(kept)} rows whose fixtures changed")
            _write(out, kept)
        done = {(r["family"], r["name"]) for r in kept}
    jobs = [
        (fam, name, edits, args.k, i % DEEP_EVERY == 0 and fam != "longest")
        for i, (fam, name, edits) in enumerate(todo)
        if (fam, name) not in done
    ]
    # The longest fixtures first, so the pool never waits on a straggler at the end.
    jobs.sort(key=lambda j: -patch.from_dict(patch_dict(j[2])).length_ms)
    say(f"{len(jobs)} fixtures to run at K = {args.k} ({len(done)} already in {out})")
    start = time.perf_counter()
    fresh = not out.exists()
    with (
        out.open("a", newline="", encoding="utf-8") as f,
        ProcessPoolExecutor(args.workers, initializer=carrier.share_cores, initargs=(args.workers,)) as pool,
    ):
        w = csv.DictWriter(f, FIELDS)
        if fresh:
            w.writeheader()
        for i, row in enumerate(pool.map(run_one, jobs, chunksize=1)):
            w.writerow(row)
            f.flush()
            if (i + 1) % REPORT_EVERY == 0:
                say(f"  {i + 1}/{len(jobs)} after {time.perf_counter() - start:.0f} s")
    say(f"done in {time.perf_counter() - start:.0f} s")


def load(path: str | os.PathLike[str], saturation: float = SATURATION) -> list[Row]:
    """Read a grid's results, and judge each fixture by the rule at a saturation coefficient."""
    rows: list[Row] = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    for r in rows:
        for key in FIELDS[2:-1]:
            r[key] = float(r[key]) if r.get(key) not in (None, "") else math.nan
        near = r["peak"] < QUIET
        r["pass"] = r["max_abs"] <= FLOOR if near else (r["e_rel"] <= LIMIT_DB and r["e_max"] <= LIMIT_DB)
        converged = r["c_rel"] <= CONVERGED_DB and r["c_max"] <= CONVERGED_DB
        r["converged"] = r["c_abs"] <= FLOOR / 10 if near else converged
        q = r["knee"]
        r["saturation"] = math.exp(-(math.pi**2) / q) * r["peak"] if q > 0 else 0.0  # per unit coefficient
        r["allowed"] = LIMIT * r["peak"] if not near else FLOOR
        predicted = saturation * r["saturation"] + r["kink"]
        r["predicted"] = predicted
        r["inside"] = predicted <= r["allowed"]
        r["margin_db"] = 20 * math.log10(r["allowed"] / predicted) if predicted > 0 else math.inf
    return rows


def fit_saturation(rows: Iterable[Row]) -> tuple[float, Row | None]:
    """Return the smallest coefficient that covers every converged fixture with at least 6 dB of margin (plan §5).

    Where the measured error exceeds the kink bound, the saturation term must
    cover the rest twice over: C · e^(−π²/q) · P ≥ 2 · (measured − kink bound).
    """
    need, worst = 0.0, None
    for r in rows:
        if r["converged"] and r["saturation"] > 0 and r["max_abs"] > max(r["kink"], ROUNDING * r["peak"]):
            c = 2 * (r["max_abs"] - r["kink"]) / r["saturation"]
            if c > need:
                need, worst = c, r
    return need, worst


def _kinks(rows: list[Row]) -> None:
    """Report the undriven fixtures, and any whose measured error comes near the kink bound."""
    linear = [r for r in rows if r["alpha"] == 0.0 and r["converged"]]
    misses = [r for r in linear if r["max_abs"] > r["kink"] and r["max_abs"] > ROUNDING * r["peak"]]
    thin = [r for r in linear if 2 * r["max_abs"] > r["kink"] and r["max_abs"] > ROUNDING * r["peak"]]
    say(
        f"  undriven fixtures {len(linear)}: pass {sum(r['pass'] for r in linear)}; the kink bound is exceeded by "
        f"the measured error in {len(misses)}, and within 6 dB of it in {len(thin)}",
    )
    for r in misses + thin:
        say(f"    {r['family']}/{r['name']}: measured {r['max_abs']:.3g}, bound {r['kink']:.3g}")


def _inside(rows: list[Row]) -> None:
    """Report the fixtures inside the domain: any failing, the worst error and the least conservative prediction."""
    inside = [r for r in rows if r["inside"]]
    bad = [r for r in inside if not (r["pass"] and r["converged"])]
    say(f"  inside the validated domain: {len(inside)}; of those failing or unconverged: {len(bad)}")
    for r in bad:
        say(f"    {r['family']}/{r['name']}: E_max {r['e_max']:.1f} dB, convergence {r['c_max']:.1f} dB")
    loud = [r for r in inside if r["peak"] >= QUIET]
    if loud:
        w = max(loud, key=lambda r: r["e_max"])
        say(f"  worst error inside: E_max {w['e_max']:.1f} dB ({w['family']}/{w['name']})")
        measured = [r for r in loud if r["max_abs"] > ROUNDING * r["peak"]]  # an error above rounding
        if measured:
            m = min(measured, key=lambda r: r["predicted"] / r["max_abs"])
            above = 20 * math.log10(m["predicted"] / m["max_abs"])
            say(f"  least conservative inside: predicted {above:.1f} dB above measured ({m['family']}/{m['name']})")


def _references(rows: list[Row]) -> None:
    """Report the rule's caution, the unconverged references and the deeper checks."""
    caution = sum(1 for r in rows if not r["inside"] and r["pass"] and r["converged"])
    say(f"  outside but passing (the rule's caution): {caution}")
    for r in (r for r in rows if not r["converged"]):
        say(f"    unconverged: {r['family']}/{r['name']}: 128×/256× agree to {r['c_max']:.1f} dB")
    checks = [r for r in rows if not math.isnan(r["c3_max"])]
    if checks:
        better = sum(1 for r in checks if r["c3_max"] <= r["c_max"] + 1e-9 or r["c3_max"] <= DEEP_FLOOR_DB)
        say(f"  512× check on {len(checks)} fixtures: 256×/512× agree at least as well as 128×/256× in {better}")


def _families(rows: list[Row]) -> None:
    """Report each family's counts and its worst error inside, and the time per fixture."""
    say(f"  {'family':10s} {'n':>4s} {'inside':>7s} {'pass':>5s} {'unconverged':>12s} {'worst inside':>13s}")
    families: dict[str, list[float]] = {}
    for r in rows:
        f = families.setdefault(r["family"], [0, 0, 0, 0, -INF])
        f[0] += 1
        f[1] += r["inside"]
        f[2] += r["pass"] and r["converged"]
        f[3] += not r["converged"]
        if r["inside"] and r["peak"] >= QUIET:
            f[4] = max(f[4], r["e_max"])
    for name, (n, i, ok, unconv, w) in families.items():
        say(f"  {name:10s} {n:4.0f} {i:7.0f} {ok:5.0f} {unconv:12.0f} {w:13.1f}")
    times = sorted(r["total_s"] for r in rows)
    say(f"  time per fixture: median {times[len(times) // 2]:.1f} s, max {times[-1]:.0f} s")


def analyze(args: argparse.Namespace) -> None:
    """Report each grid: its fixtures, the kink bound, the coefficient the rule needs, and the domain it gives."""
    for path in args.csv:
        rows = load(path)
        conv = [r for r in rows if r["converged"]]
        need, worst = fit_saturation(rows)
        say(f"\n{path}: K = {int(rows[0]['k'])}, {len(rows)} fixtures")
        say(f"  pass {sum(r['pass'] for r in rows)}, converged references {len(conv)}")
        _kinks(rows)
        text = f"  saturation coefficient for 6 dB of margin on every converged fixture: {need:.3g}"
        if worst:
            text += f" ({worst['family']}/{worst['name']})"
        say(text + f"; the rule uses {SATURATION:g}")
        rows = load(path, max(need, SATURATION))
        _inside(rows)
        _references(rows)
        _families(rows)


def list_fixtures(args: argparse.Namespace) -> None:
    """List the fixtures, and count them by family."""
    rows = [f for f in fixtures() if not args.family or f[0] in args.family]
    counts: dict[str, int] = {}
    for fam, name, _ in rows:
        counts[fam] = counts.get(fam, 0) + 1
        if args.verbose:
            say(f"{fam}/{name}")
    say(f"{json.dumps(counts)} {sum(counts.values())}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid tool."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--k", type=int, default=32)
    r.add_argument("--family", action="append")
    r.add_argument("--out")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    a = sub.add_parser("analyze")
    a.add_argument("csv", nargs="+")
    lst = sub.add_parser("list")
    lst.add_argument("--family", action="append")
    lst.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    {"run": run, "analyze": analyze, "list": list_fixtures}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
