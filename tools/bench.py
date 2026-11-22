# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Performance check (docs/design.md, docs/validation.md): render time and memory on this machine.

Each case runs in a fresh interpreter, a spawned process, after a first render
has paid the libraries' first-use costs, so memory is measured above that
baseline. Every timed render uses a take not
rendered before, so it computes its tables, as every new take does; a repeat
of the same take finds them cached, as audition's re-renders do. Both are
reported, and the budget applies to the first. Times include the report.

    uv run python -m tools.bench
"""

from __future__ import annotations

import dataclasses
import functools
import multiprocessing
import os
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy

from brightwork_synth import cli
from brightwork_synth.pack import Options, build
from brightwork_synth.patch import Noise, loads, softer, stretch
from brightwork_synth.render import FAMILY, decimator, render

if TYPE_CHECKING:
    from collections.abc import Callable

# (label, preset, length in ms, a third source, the budget in s)
CASES = (
    ("closed hat 0.25 s, 2 sources", "closed", 250.0, False, 0.25),
    ("open hat 1.5 s, 2 sources", "open", 1500.0, False, 0.5),
    ("crash 5 s, 3 sources", "crash", 5000.0, True, 1.5),
    ("ride 8 s, 3 sources", "ride", 8000.0, False, 2.0),
)
RATES = (44100, 48000, 88200, 96000, 176400, 192000)
REPEAT = 3  # takes timed per case, the fastest counted
KIT_S, MEMORY_MIB = 30.0, 500
LOW_TAIL = Noise(  # a third source with a low band, under the crash
    -12.0,
    (300.0, 6000.0),
    0.0,
    0.3,
    1.0,
    0.0,
    220.0,
    0.0,
    10.0,
    ((0.0, 0.0), (5.0, 1.0), (400.0, 0.3), (2000.0, 0.05), (4500.0, 0.0)),
)


@dataclasses.dataclass(frozen=True)
class Hit:
    """One case's results: the fastest render of a new take, of a cached one, memory, length and sub-bands."""

    fresh: float
    cached: float
    mib: float
    subbands: int


def _memory() -> int:
    """Return this process's peak resident memory, in KiB."""
    if sys.platform != "win32":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sys.exit("the benchmark reads peak memory from the resource module, which Windows lacks")


def _hit(name: str, rate: int, ms: float, *, third: bool) -> Hit:
    """Time a preset, stretched to `ms`, at a rate: new takes and cached ones, above the libraries' baseline."""
    text = cli.preset_text(name).replace("sample_rate = 48000", f"sample_rate = {rate}")
    p = loads(text)
    if third:
        p = dataclasses.replace(p, noise=(*p.noise, LOW_TAIL))
    p = stretch(p, "all", ms / p.length_ms, after_ms=0.0)
    decimator(FAMILY[rate])  # loading the shipped coefficients is not part of a render
    _, report = render(dataclasses.replace(p, take=999))  # first-use costs of the libraries, not of this render
    base = _memory()
    fresh, cached = float("inf"), float("inf")
    for take in range(1, 1 + REPEAT):
        q = dataclasses.replace(p, take=take)
        t0 = time.perf_counter()
        render(q)
        fresh = min(fresh, time.perf_counter() - t0)
        t0 = time.perf_counter()
        render(q)
        cached = min(cached, time.perf_counter() - t0)
    peak = _memory()
    return Hit(fresh, cached, (peak - base) / 1024, sum(s.subbands for s in report.sources))


def _kit() -> tuple[float, float, int]:
    """Time the hi-hat kit of three articulations, four layers and eight takes: seconds, memory and files."""
    decimator(48000)
    groups = [
        (key, [(f"{name}{db}", softer(loads(cli.preset_text(name)), db)) for db in (0, 8, 16, 24)])
        for key, name in ((42, "closed"), (44, "pedal"), (46, "open"))
    ]
    base = _memory()
    t0 = time.perf_counter()
    kit = build(groups, Options(takes=8, chokes=((46, (42, 44)),)))
    seconds = time.perf_counter() - t0
    peak = _memory()
    return seconds, (peak - base) / 1024, sum(len(f) for k in kit.keys for f in k.audio)


def fresh[T](work: Callable[[], T]) -> T:
    """Run work in a fresh interpreter, so its memory is its own and nothing is cached before it."""
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(work).result()


def cpu() -> str:
    """Name the processor."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> int:
    """Time every case at every rate and the kit, against the budget, and say whether all of them fit."""
    out = sys.stdout
    out.write(
        f"machine: {cpu()}, {os.cpu_count()} cores, Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}\n",
    )
    out.write(
        f"{'sound':<30s} {'rate':>6s} {'sub-bands':>9s} {'new take':>9s} {'cached':>8s} {'limit':>7s} {'memory':>8s}\n",
    )
    ok = True
    for label, name, ms, third, limit in CASES:
        for rate in RATES:
            r = fresh(functools.partial(_hit, name, rate, ms, third=third))
            fine = r.fresh <= limit and r.mib <= MEMORY_MIB
            ok &= fine
            out.write(
                f"{label:<30s} {rate:6d} {r.subbands:9d} {1000 * r.fresh:7.0f}ms {1000 * r.cached:6.0f}ms "
                f"{1000 * limit:5.0f}ms {r.mib:5.0f}MiB {'' if fine else 'OVER BUDGET'}\n",
            )
    seconds, mib, files = fresh(_kit)
    fine = seconds <= KIT_S and mib <= MEMORY_MIB
    ok &= fine
    out.write(
        f"{'kit: 3 keys × 4 layers × 8 takes':<30s} {48000:6d} {'':>9s} {seconds:8.1f}s {'':>8s} {KIT_S:6.0f}s "
        f"{mib:5.0f}MiB {'' if fine else 'OVER BUDGET'}  ({files} files)\n",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
