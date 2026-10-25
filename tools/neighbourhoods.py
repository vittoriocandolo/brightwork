# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The validated extent of every reference along each of its controls (plan §5, §8.4).

The voiced regions are the references' neighbourhoods. Whether a sound in one
is useful is for listening to decide; this measures where each is validated.
One control at a time is moved on the reference's main source (its plates or
wash, or for the bell and the analog hat its tone), everything else as the
reference has it, and each candidate value is rendered over seeds × takes. A control's extent
is the run of candidates around the reference's own value at which every one
of those renders is inside the validated domain. Each end of the run says what
stops it: the validated domain (outside), the valid domain (the patch refuses
the value), or the end of the scan. Tail length is `stretch` of every shape
after its peak. Drive is not scanned: every reference renders without drive,
and its extent is the smallest drive headroom that tools/references.py
reports over all its takes, which is exact or cautious.

    uv run python -m tools.neighbourhoods [--seeds 4] [--takes 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from brightwork_synth.cli import REFERENCES, reference_text
from brightwork_synth.patch import PatchError, from_dict, loads, stretch, to_dict
from brightwork_synth.render import domain, synthesize
from tools.grid import set_path

STEPS = {
    "damping": [i / 4 for i in range(13)],
    "bloom": [i / 4 for i in range(13)],
    "chatter": [i / 10 for i in range(11)],
    "spacing_hz": [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0, 113.0, 160.0, 200.0],
}
OCTAVES = {"low": [2 ** (k / 2) for k in range(-6, 7)], "high": [2 ** (k / 4) for k in range(-8, 5)]}
LENGTHS = [2 ** (k / 2) for k in range(-4, 5)]  # tail length, by `stretch` of every shape after its peak
CONTROLS = ("band low", "band high", "spacing_hz", "damping", "bloom", "chatter", "length")


MAIN = {"bell": "tone[1]", "analog": "tone[1]"}  # the others' main source is noise[1]


def main_source(name: str) -> str:
    return MAIN.get(name, "noise[1]")


def candidates(name: str, control: str) -> tuple[list[float], float]:
    """The values scanned along a control, and the reference's own."""
    kind = main_source(name)[:-3]
    src = to_dict(loads(reference_text(name)))[kind][0]
    if control == "length":
        return LENGTHS, 1.0
    if control.startswith("band"):
        lo, hi = src["band_hz"]
        own = lo if control == "band low" else hi
        return [round(own * f, 1) for f in OCTAVES[control.split()[1]]], own
    if control == "spacing_hz" and kind == "tone":
        return [], math.nan
    own = float(src[control])
    return sorted(set(STEPS[control]) | {own}), own


def moved(name: str, control: str, value: float):
    """The reference with one control of its main source at `value`, or None where the patch refuses it."""
    p = loads(reference_text(name))
    try:
        if control == "length":
            return stretch(p, "all", value)
        d = to_dict(p)
        path = main_source(name)
        if control.startswith("band"):
            lo, hi = d[path[:-3]][0]["band_hz"]
            band = [value, hi] if control == "band low" else [lo, value]
            set_path(d, f"{path}.band_hz", band)
        else:
            set_path(d, f"{path}.{control}", value)
        return from_dict(d)
    except PatchError:
        return None


def status(job) -> tuple:
    """(job, inside), with None for inside where the patch refuses the value."""
    name, control, value, seed, take = job
    p = moved(name, control, value)
    if p is None:
        return job, None
    return job, domain(synthesize(dataclasses.replace(p, seed=seed, take=take))).inside


def extent(values: list[float], own: float, states: dict) -> tuple:
    """(low, high, why low stops, why high stops) of the run of values around `own` where every render is inside."""
    i = values.index(own)
    ok = [states[v] is True for v in values]
    if not ok[i]:
        return None, None, "outside", "outside"
    lo, hi = i, i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(values) - 1 and ok[hi + 1]:
        hi += 1

    def why(j: int) -> str:
        if j < 0 or j >= len(values):
            return "scan"
        return "refused" if states[values[j]] is None else "outside"

    return values[lo], values[hi], why(lo - 1), why(hi + 1)


def measure(seeds: int, takes: int, workers: int) -> dict:
    jobs, scans = [], {}
    for name in REFERENCES:
        for control in CONTROLS:
            values, own = candidates(name, control)
            if not values:
                continue
            if own not in values:
                values = sorted(set(values) | {own})
            scans[(name, control)] = (values, own)
            jobs += [(name, control, v, s, t) for v in values for s in range(1, seeds + 1) for t in range(1, takes + 1)]
    found: dict = {}
    with ProcessPoolExecutor(workers) as pool:
        for (name, control, value, _, _), inside in pool.map(status, jobs, chunksize=4):
            found.setdefault((name, control, value), []).append(inside)
    out = {}
    for (name, control), (values, own) in scans.items():
        # A value is refused (None) or inside (True) only if every render agrees; any render outside makes it False.
        states = {}
        for v in values:
            seen = found[(name, control, v)]
            states[v] = None if None in seen else all(seen)
        out[(name, control)] = extent(values, own, states)
    return out


def text(v: float, control: str) -> str:
    if control == "length":
        return f"{v:.2f}×"
    if control.startswith("band"):
        return f"{v / 1000:.3g} kHz" if v >= 1000 else f"{v:.3g} Hz"
    return f"{v:g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--takes", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    result = measure(args.seeds, args.takes, args.workers)
    print(f"seeds 1–{args.seeds} × takes 1–{args.takes}; an end in brackets says what stops it")
    for name in REFERENCES:
        print(f"{name} ({main_source(name)}):")
        for control in CONTROLS:
            if (name, control) not in result:
                continue
            lo, hi, why_lo, why_hi = result[(name, control)]
            own = candidates(name, control)[1]
            if lo is None:
                print(f"  {control:<11s} the reference's own value {text(own, control)} is outside")
                continue
            print(
                f"  {control:<11s} {text(lo, control)} [{why_lo}] – {text(hi, control)} [{why_hi}]"
                f"   (reference {text(own, control)})",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
