"""Golden renders of five frozen patches (plan §6 stability, §10 regression contract).

Across machines and library versions the output agrees within floating-point
rounding, so the comparison uses a tolerance rather than a hash (plan §6).
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from cymbal import wav
from cymbal.patch import loads
from cymbal.render import render_takes

GOLDEN = Path(__file__).parent / "golden"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from golden import SETS  # noqa: E402


@pytest.mark.parametrize("name", SETS)
def test_golden(name):
    files, stereo = SETS[name]
    p = loads((GOLDEN / f"{name}.toml").read_text(encoding="utf-8"))
    audio, report = render_takes(p, files, stereo=stereo)
    for a, f in zip(audio, report.files):
        golden = wav.read(GOLDEN / f"{name}_{f.number}.wav")
        first = report.takes[f.takes[0]].take
        assert loads(golden.meta["patch"]) == dataclasses.replace(p, take=first)
        assert golden.meta["takes"] == [report.takes[i].take for i in f.takes]
        assert golden.rate == report.sample_rate and golden.channels == (2 if stereo else 1)
        assert a.shape == golden.samples.shape
        assert np.max(np.abs(a - golden.samples)) <= 1e-6
