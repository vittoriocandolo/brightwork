# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import re
import tomllib

import pytest

# The illustrative patch from the plan (§4): a closed hi-hat.
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
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.03, gain = 1.0 },
  { ms = 0.13, gain = 0.6 },  # a shoulder knot keeps the tick sharp
  { ms = 1.5,  gain = 0.1 },
  { ms = 5.0,  gain = 0.0 },
]

[damper]                    # multiplies every source: 1 until damped, then down to 0
amp = []                    # no knots: never damped

[drive]
db = 0.0                    # small-signal gain; 0 = bypass

[output]
sample_rate = 48000
format      = "float32"     # "float32" | "pcm24" | "pcm16"
mode        = "gain"        # "gain" | "peak"
db          = -6.0          # gain in dB, or peak target in dBFS
polarity    = 1
"""

# The plan's tone (§4): the analog reference's six square oscillators.
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
    "amp": [
        {"ms": 0.0, "gain": 0.0},
        {"ms": 0.02, "gain": 1.0},
        {"ms": 0.12, "gain": 0.7},
        {"ms": 20.0, "gain": 0.12},
        {"ms": 60.0, "gain": 0.0},
    ],
}


@pytest.fixture
def example_dict():
    return copy.deepcopy(tomllib.loads(EXAMPLE))


def edit(d: dict, path: str, value) -> dict:
    """Set a field by its patch path, counting from 1: edit(d, "noise[2].amp[3].ms", 5.0)."""
    node = d
    parts = path.split(".")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
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


def example(tone: bool = False) -> dict:
    d = tomllib.loads(EXAMPLE)
    if tone:
        d["tone"] = [copy.deepcopy(TONE)]
    return d


def make(tone: bool = False, **edits):
    """The example patch with edits by path, e.g. make(**{"noise[1].chatter": 0.3}); tone=True adds the plan's tone."""
    from brightwork_synth.patch import from_dict

    d = example(tone)
    for path, value in edits.items():
        edit(d, path, value)
    return from_dict(d)
