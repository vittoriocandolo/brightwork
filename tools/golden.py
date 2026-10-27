# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write the golden renders that tests/test_golden.py compares against (plan §6, §10).

Golden files begin once the rendering policy is frozen (milestone 2). Each is
the render of a frozen patch in tests/golden, chosen to cover every stage:
modes and continuous noise, square tones with free phases, damping and bloom
over many sub-bands, chatter, a choke, drive, struck sine partials, the 44.1
and 96 kHz exports, a stereo pair, rattles: jingles under chatter and a head
sharing their collisions, and storage: an open hat stored in 6 bits at
24 kHz, choked and driven, at 96 kHz. They are not the references, whose
voicing may still change. Only regenerate them for a deliberate change to the
sound, and say so in the commit.

    uv run python -m tools.golden
"""

from __future__ import annotations

import sys
from pathlib import Path

from brightwork_synth.patch import loads
from brightwork_synth.render import export, render_takes

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"
SETS = {  # name: (files, stereo)
    "hat": (2, False),
    "analog": (2, False),
    "crash": (1, True),
    "choked": (2, False),
    "bell": (1, False),
    "rattle": (2, False),
    "stored": (2, False),
}


def main() -> int:
    """Render every golden set again, and write its files over the recorded ones."""
    for name, (files, stereo) in SETS.items():
        p = loads((GOLDEN / f"{name}.toml").read_text(encoding="utf-8"))
        audio, report = render_takes(p, files, stereo=stereo)
        for a, f in zip(audio, report.files, strict=True):
            export(GOLDEN / f"{name}_{f.number}.wav", a, p, report, f)
        sys.stdout.write(f"{name}: {len(audio)} file(s) of {audio[0].shape} at {p.output.sample_rate} Hz\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
