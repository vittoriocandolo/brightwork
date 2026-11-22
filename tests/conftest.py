# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the tests share: the documented example patch and tone, and ways to edit them by path."""

import copy
import os
import re
import tomllib
from typing import Any

import pytest
from hypothesis import settings

from brightwork_synth import wav
from brightwork_synth.patch import Patch, from_dict

# The illustrative patch of the patch reference (docs/patch.md): a closed hi-hat.
# Fuzzing derives its examples the same way on every run and keeps none between runs, so a run is reproducible
# from the code alone; a deadline would make the outcome depend on the machine.
settings.register_profile("reproducible", derandomize=True, database=None, deadline=None)
settings.load_profile("reproducible")

EXAMPLE = """\
instrument = "cymbal"
schema = 1
seed = 1                    # the cymbal: where its modes lie
take = 1                    # the strike: round robins use take, take + 1, …

[[noise]]                   # noise[1], the plates: modes about 14 Hz apart
level_db   = -2.0
band_hz    = [6000.0, 16000.0]
spacing_hz = 14.0           # mean distance between modes; 0 = continuous noise
damping    = 0.5            # content at f decays (f / 6 kHz)^0.5 times faster
bloom      = 0.0            # and rises (f / 6 kHz)^bloom times slower
chatter    = 0.0            # 0 … 1: gated at chatter_hz, as plates that slap
chatter_hz = 220.0
grains_hz  = 0.0            # collisions per second at the shape's peak, a rattle; 0 = none
grain_ms   = 10.0           # each collision rings down 60 dB in this time
amp = [                     # shape: 0 at both ends, highest point exactly 1
  { ms = 0.0,   gain = 0.0 },
  { ms = 0.3,   gain = 1.0 },
  { ms = 8.0,   gain = 0.45 },
  { ms = 30.0,  gain = 0.14 },
  { ms = 70.0,  gain = 0.04 },
  { ms = 120.0, gain = 0.0 },
]

[[noise]]                   # noise[2], the stick's tick
level_db   = -10.0
band_hz    = [3000.0, 18000.0]
spacing_hz = 0.0
damping    = 0.0
bloom      = 0.0
chatter    = 0.0
chatter_hz = 220.0
grains_hz  = 0.0
grain_ms   = 10.0
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.03, gain = 1.0 },
  { ms = 0.13, gain = 0.6 },  # a shoulder knot keeps the tick sharp
  { ms = 1.5,  gain = 0.1 },
  { ms = 5.0,  gain = 0.0 },
]

[storage]                   # low-resolution memory: the sources stored, then played back
bits      = 0               # 0 = off; else 2 … 16, quantized against the sound's peak
rate_hz   = 32000.0         # stored at this rate, as near as the internal rate allows
filter_hz = 16000.0         # played back through a low-pass, −3 dB here

[damper]                    # multiplies every source: 1 until damped, then down to 0
amp = []                    # no knots: never damped

[drive]
db = 0.0                    # small-signal gain; 0 = bypass

[output]
sample_rate = 48000
format      = "float32"     # "float32" | "pcm24" | "pcm16"
mode        = "gain"        # "gain" | "peak" | "true-peak"
db          = -6.0          # gain in dB, or peak target in dBFS
polarity    = 1
"""

# The tone of the patch reference (docs/patch.md): the analog preset's six square oscillators.
TONE = {
    "level_db": -3.0,
    "wave": "square",
    "phase": "free",
    "partials": [
        {"hz": 205.3, "level_db": 0.0},
        {"hz": 304.4, "level_db": 0.0},
        {"hz": 369.6, "level_db": 0.0},
        {"hz": 522.7, "level_db": 0.0},
        {"hz": 540.0, "level_db": 0.0},
        {"hz": 800.0, "level_db": 0.0},
    ],
    "band_hz": [7000.0, 16000.0],
    "damping": 0.3,
    "bloom": 0.0,
    "chatter": 0.0,
    "chatter_hz": 220.0,
    "grains_hz": 0.0,
    "grain_ms": 10.0,
    "amp": [
        {"ms": 0.0, "gain": 0.0},
        {"ms": 0.02, "gain": 1.0},
        {"ms": 0.12, "gain": 0.7},
        {"ms": 20.0, "gain": 0.12},
        {"ms": 60.0, "gain": 0.0},
    ],
}


@pytest.fixture
def example_dict() -> dict[str, Any]:
    """Return the example patch as TOML parses it, to edit freely."""
    return copy.deepcopy(tomllib.loads(EXAMPLE))


def edit(d: dict[str, Any], path: str, value: object) -> dict[str, Any]:
    """Set a field of parsed TOML by its patch path, counting from 1, such as "noise[2].amp[3].ms"."""
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
    return d


def example(*, tone: bool = False) -> dict[str, Any]:
    """Return the example patch as TOML parses it; with `tone`, the documented tone is added."""
    d = tomllib.loads(EXAMPLE)
    if tone:
        d["tone"] = [copy.deepcopy(TONE)]
    return d


def make(**edits: object) -> Patch:
    """Return the example patch with fields set by path, such as "noise[1].chatter"."""
    return _made(example(), edits)


def make_tone(**edits: object) -> Patch:
    """Return the example patch with the documented tone added, and fields set by path."""
    return _made(example(tone=True), edits)


def _made(d: dict[str, Any], edits: dict[str, object]) -> Patch:
    """Build the patch with the edits applied, by path."""
    for path, value in edits.items():
        edit(d, path, value)
    return from_dict(d)


def meta_of(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return what a WAV that Brightwork wrote records of how it was made."""
    meta = wav.read(path).meta
    assert meta is not None
    return meta
