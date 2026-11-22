# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden renders of seven frozen patches (docs/design.md: Reproducibility).

Across machines and library versions the output agrees within floating-point
rounding, so the comparison uses a tolerance rather than a hash (docs/design.md: Reproducibility).
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from brightwork_synth import wav
from brightwork_synth.patch import loads
from brightwork_synth.render import render_takes
from tools.golden import SETS

GOLDEN = Path(__file__).parent / "golden"
ROUNDING = 1e-6  # of full scale: what another platform's rounding may change


@pytest.mark.parametrize("name", SETS)
def test_golden(name: str) -> None:
    """Render a golden set again, and find its patches, takes, format and samples as recorded, to a millionth."""
    files, stereo = SETS[name]
    p = loads((GOLDEN / f"{name}.toml").read_text(encoding="utf-8"))
    audio, report = render_takes(p, files, stereo=stereo)
    for a, f in zip(audio, report.files, strict=True):
        golden = wav.read(GOLDEN / f"{name}_{f.number}.wav")
        assert golden.meta is not None
        first = report.takes[f.takes[0]].take
        assert loads(golden.meta["patch"]) == dataclasses.replace(p, take=first)
        assert golden.meta["takes"] == [report.takes[i].take for i in f.takes]
        assert golden.rate == report.sample_rate
        assert golden.channels == (2 if stereo else 1)
        assert a.shape == golden.samples.shape
        assert np.max(np.abs(a - golden.samples)) <= ROUNDING
