# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The seven references by measurement (plan §3, §8.4): seeds 1–4 × takes 1–8 at the shipping K.

For each reference it reports the loudness and sample peak of every take,
whether each take is inside the validated domain and its drive headroom, and
the loudness spread of takes 1–8 of each seed. The acceptance criteria are:

  * inside the validated domain with at least 3 dB of drive headroom, every seed and take;
  * loudness matched within 1 dB across the references, and every peak below −1 dBFS;
  * a loudness spread of at most 1 dB over takes 1–8 of each seed, or else a set that the report flags as
    uneven and that level matching brings within 1 dB.

    uv run python -m tools.references [--seeds 4] [--takes 8] [--workers N]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from brightwork_synth.cli import REFERENCES, reference_text
from brightwork_synth.loudness import momentary_max
from brightwork_synth.patch import loads
from brightwork_synth.render import cause_of, db, domain, drive_headroom, output_stage, render_takes, synthesize


def take_row(args) -> dict:
    name, seed, take = args
    p = dataclasses.replace(loads(reference_text(name)), seed=seed, take=take)
    s = synthesize(p)
    d = domain(s)
    h = drive_headroom(s, p.drive.db, d)
    a = output_stage(s.out, 10 ** (p.output.db / 20), p.output.polarity)
    return {
        "name": name,
        "seed": seed,
        "take": take,
        "lufs": momentary_max(a, p.output.sample_rate),
        "peak": db(float(max(abs(a.min()), a.max()))),
        "inside": d.inside,
        "margin": d.margin_db,
        "headroom": h.db if h else math.inf,
        "cause": cause_of(p, s, d).dominant,
    }


def summarize(name: str, rows: list[dict]) -> dict:
    spreads = {}
    for seed in sorted({r["seed"] for r in rows}):
        values = [r["lufs"] for r in rows if r["seed"] == seed]
        spreads[seed] = max(values) - min(values)
    first = next(r for r in rows if r["seed"] == 1 and r["take"] == 1)
    return {
        "name": name,
        "lufs": first["lufs"],
        "mean": sum(r["lufs"] for r in rows) / len(rows),
        "spread": max(spreads.values()),
        "spreads": spreads,
        "peak": max(r["peak"] for r in rows),
        "inside": all(r["inside"] for r in rows),
        "margin": min(r["margin"] for r in rows),
        "headroom": min(r["headroom"] for r in rows),
    }


def measure(seeds: int, takes: int, workers: int) -> list[dict]:
    jobs = [(n, s, t) for n in REFERENCES for s in range(1, seeds + 1) for t in range(1, takes + 1)]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(take_row, jobs, chunksize=1))
    return [summarize(n, [r for r in rows if r["name"] == n]) for n in REFERENCES]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--takes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    results = measure(args.seeds, args.takes, args.workers)
    print(f"seeds 1–{args.seeds} × takes 1–{args.takes}")
    print(
        f"{'reference':<9s} {'LUFS':>6s} {'mean':>6s} {'spread':>7s} {'max peak':>9s} {'inside':>7s} "
        f"{'min margin':>11s} {'min headroom':>13s}",
    )
    for r in results:
        print(
            f"{r['name']:<9s} {r['lufs']:6.1f} {r['mean']:6.1f} {r['spread']:5.2f}dB {r['peak']:7.2f}dB "
            f"{r['inside']!s:>7s} {r['margin']:9.1f}dB {r['headroom']:11.1f}dB",
        )
    means = [r["mean"] for r in results]
    # A set whose takes spread over 1 dB must be flagged by the report, and level matching must bring it within 1 dB.
    matched = set()
    for r in results:
        p = loads(reference_text(r["name"]))
        uneven = [s for s, spread in r["spreads"].items() if spread > 1.0]
        fine = True
        for s in uneven:
            q = dataclasses.replace(p, seed=s)
            fine &= render_takes(q, args.takes)[1].uneven
            fine &= render_takes(q, args.takes, level_match=True)[1].loudness_spread <= 1.0
        if uneven:
            state = "flagged; within 1 dB with --level-match" if fine else "NOT handled"
            print(f"{r['name']}: seeds {', '.join(map(str, uneven))} spread over 1 dB, {state}")
            if fine:
                matched.add(r["name"])
    ok = (
        max(means) - min(means) <= 1.0
        and all(r["peak"] < -1.0 for r in results)
        and all(r["inside"] and r["headroom"] >= 3.0 for r in results)
        and all(r["spread"] <= 1.0 or r["name"] in matched for r in results)
    )
    print(f"loudness range {max(means) - min(means):.2f} dB; {'all criteria met' if ok else 'criteria NOT met'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
