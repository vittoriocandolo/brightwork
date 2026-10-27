# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The report's drive headroom against brute force, on the adversarial grid (plan §5, §8.1).

For every grid fixture, the headroom that the report computes from one
render is compared with the first drive at which full renders change the
validated-domain status. The brute force renders every candidate the report
considers (every 0.5 dB, and both sides of every jump of the knee slew),
from the fixture's own drive until the status changes, then bisects that
bracket. With --dense it also scans every 0.05 dB up to the edge on fixtures
whose edge is within 6 dB, to show that no status change hides between the
candidates.

Drive enters only through the saturator, so the saturator's input x is
synthesized once per fixture. Each candidate is then a full render from x
on: the saturator, the decimator and the output peak, the knee slew
measured directly on x, and the kink bound at that drive. Nothing comes from
the report's frontier or its probes.

    uv run python -m tools.headroom run [--out docs/grid/headroom.csv] [--dense]
    uv run python -m tools.headroom analyze docs/grid/headroom.csv

A run appends to its CSV and skips fixtures already there, so it can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from brightwork_synth import carrier
from brightwork_synth.patch import from_dict
from brightwork_synth.render import (
    DRIVE_MAX,
    FAMILY,
    HEADROOM_STEP_DB,
    SATURATION,
    K,
    allowed,
    decimate,
    decimator,
    drives_db,
    interpolator_for,
    rates,
    render,
)
from brightwork_synth.voice import (
    Knees,
    drive_alpha,
    internal,
    kink_bound,
    kink_terms,
    sample_count,
    saturate,
    slew_of,
    tables,
    voice,
)
from tools import grid

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

type Row = dict[str, Any]

FIELDS = ["family", "name", "drive", "inside", "estimate", "reference", "error", "dense", "renders", "seconds", "edits"]
BISECT = 16  # halvings of a candidate bracket: 0.5 dB / 2¹⁶ ≈ 10⁻⁵ dB
DENSE_STEP_DB, DENSE_RANGE_DB = 0.05, 6.0
SIDE = 1e-12  # how far either side of a drop in the knee slew a candidate lies, relative to its α
EXACT_DB = 0.001  # a report this close to the brute force's edge is exact
NEAR_DB = 3.0  # how near the edge a fixture lies, for the report's caution there
REPORT_EVERY = 25  # fixtures between progress lines


def say(text: str) -> None:
    """Print a line of progress at once."""
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()


class Fixture:
    """One fixture's saturator input and kinks, from which a full render at any drive follows."""

    def __init__(self, edits: grid.Edits) -> None:
        """Synthesize the fixture's saturator input and its kinks, once."""
        p = from_dict(grid.patch_dict(edits))
        v = voice(p)
        self.rate, self.factor = rates(p.output.sample_rate)
        self.taps = decimator(FAMILY[p.output.sample_rate]).taps
        ip = interpolator_for(p.output.sample_rate)
        tabs = tables(v, ip.rate)
        self.x, _, _, stored = internal(v, self.rate, sample_count(v.t_src, self.rate), ip, tabs)
        self.kinks = kink_terms(v, ip, tabs, stored)
        self.band = FAMILY[p.output.sample_rate] / 2
        u = np.concatenate(([0.0], self.x))  # every step from silence on
        a, b = u[:-1], u[1:]
        self.step = np.abs(b - a)
        self.crosses = np.signbit(a) != np.signbit(b)
        self.reach = np.minimum(np.abs(a), np.abs(b))

    def knee(self, alpha: float) -> float:
        """Q at α, measured directly: the largest step of αx that crosses zero or touches |αx| ≤ 1."""
        if not alpha:
            return 0.0
        counts = self.crosses | (self.reach <= 1.0 / alpha)
        return alpha * float(np.max(self.step, where=counts, initial=0.0))

    def inside(self, drive: float) -> bool:
        """Return the validated-domain status of a full render at that drive."""
        alpha = drive_alpha(float(drive))
        out = decimate(self.taps, saturate(alpha, self.x), self.factor, K)
        peak = float(np.max(np.abs(out), initial=0.0))
        q = self.knee(alpha)
        saturation = SATURATION * math.exp(-(math.pi**2) / q) * peak if q else 0.0
        return saturation + kink_bound(self.kinks, alpha, self.rate, self.band) <= allowed(peak)


def candidates(knees: Knees, d0: float, *, up: bool) -> list[float]:
    """Return the drives the report's search visits, in search order."""
    drops = knees.drops()
    sides = np.concatenate((drops * (1.0 - SIDE), drops * (1.0 + SIDE)))
    jumps = [float(d) for d in drives_db(sides)]
    if up:
        steps = [*np.arange(d0 + HEADROOM_STEP_DB, DRIVE_MAX, HEADROOM_STEP_DB), DRIVE_MAX]
        return sorted({float(d) for d in steps + jumps if d0 < d <= DRIVE_MAX})
    steps = [*np.arange(d0 - HEADROOM_STEP_DB, 0.0, -HEADROOM_STEP_DB), 0.0]
    return sorted({float(d) for d in steps + jumps if 0.0 <= d < d0}, reverse=True)


def _edge(fixture: Fixture, d0: float, *, here: bool) -> tuple[float, int]:
    """Find, by full renders, the first drive from d0 at which the status changes, and count the renders."""
    reference, renders, prev = (math.inf if here else -math.inf), 0, d0
    for d in candidates(slew_of(fixture.x).knees, d0, up=here):
        renders += 1
        if fixture.inside(d) != here:
            inside, outside = (prev, d) if here else (d, prev)
            for _ in range(BISECT):
                mid = 0.5 * (inside + outside)
                renders += 1
                if fixture.inside(mid):
                    inside = mid
                else:
                    outside = mid
            return inside, renders
        prev = d
    return reference, renders


def run_one(job: tuple[str, str, grid.Edits, bool]) -> Row:
    """Compare one fixture's reported drive headroom with the edge full renders find."""
    family, name, edits, dense = job
    p = from_dict(grid.patch_dict(edits))
    t0 = time.perf_counter()
    take = render(p)[1].takes[0]
    d0, here = p.drive.db, take.inside
    h = take.headroom
    if h is None:
        estimate = math.nan
    elif math.isnan(h.edge_db):  # no edge within 0–48 dB
        estimate = math.inf if h.db > 0 else -math.inf
    else:
        estimate = h.edge_db
    reference, renders = (math.inf if here else -math.inf), 0
    fixture = Fixture(edits)
    if fixture.inside(d0) != here:
        msg = f"{family}/{name}: the brute force disagrees with the report at the fixture's own drive"
        raise RuntimeError(msg)
    if h is not None:
        reference, renders = _edge(fixture, d0, here=here)
    error = estimate - reference if math.isfinite(estimate) and math.isfinite(reference) else math.nan
    agree = math.isnan(error) and (estimate == reference or h is None)
    # Optionally: every 0.05 dB between the fixture's drive and the edge, where the edge is near.
    first_dense = math.nan
    if dense and h is not None and math.isfinite(reference) and abs(reference - d0) <= DENSE_RANGE_DB:
        step = DENSE_STEP_DB if here else -DENSE_STEP_DB
        d = d0 + step
        while (d < reference) if here else (d > reference):
            renders += 1
            if fixture.inside(d) != here:
                first_dense = d
                break
            d += step
        else:
            first_dense = reference
    return {
        "family": family,
        "name": name,
        "drive": d0,
        "inside": here,
        "estimate": estimate,
        "reference": reference,
        "error": 0.0 if agree else error,
        "dense": first_dense,
        "renders": renders,
        "seconds": time.perf_counter() - t0,
        "edits": json.dumps(edits),
    }


def run(args: argparse.Namespace) -> None:
    """Run every grid fixture not yet in the CSV, in parallel processes, appending each result."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fixtures = grid.fixtures()
    done = set()
    if out.exists():
        # Rows whose fixture has changed since, such as a revoiced reference's ladder, are dropped and run again.
        current = {(fam, name): json.dumps(edits) for fam, name, edits in fixtures}
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        kept = [r for r in rows if current.get((r["family"], r["name"])) == r["edits"]]
        if len(kept) < len(rows):
            say(f"dropping {len(rows) - len(kept)} rows whose fixtures changed")
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, FIELDS)
                w.writeheader()
                w.writerows(kept)
        done = {(r["family"], r["name"]) for r in kept}
    jobs = [(fam, name, edits, args.dense) for fam, name, edits in fixtures if (fam, name) not in done]
    say(f"{len(jobs)} fixtures to run ({len(done)} already in {out})")
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


def load(path: str | os.PathLike[str]) -> list[Row]:
    """Read a headroom comparison's results."""
    rows: list[Row] = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    for r in rows:
        for key in ("drive", "estimate", "reference", "error", "dense", "seconds"):
            r[key] = float(r[key])
        r["inside"] = r["inside"] == "True"
        r["renders"] = int(r["renders"])
    return rows


def split_infinite(rows: Iterable[Row]) -> tuple[list[Row], list[Row], list[Row]]:
    """Sort the fixtures whose status full renders never change within 0–48 dB by what the report says of them.

    They are the silent ones, for which no headroom is reported; the ones where
    the report is cautious, a finite edge where the status stays inside; and the
    ones where it is wrong, anything else that disagrees.
    """
    silent: list[Row] = []
    cautious: list[Row] = []
    wrong: list[Row] = []
    for r in rows:
        if r["estimate"] == r["reference"]:
            continue
        if math.isnan(r["estimate"]) and r["reference"] == math.inf:
            silent.append(r)
        elif r["reference"] == math.inf and math.isfinite(r["estimate"]) and r["estimate"] > r["drive"]:
            cautious.append(r)
        else:
            wrong.append(r)
    return silent, cautious, wrong


def _infinite(rows: list[Row]) -> None:
    """Report the fixtures whose status never changes within 0–48 dB, and what the report says of them."""
    silent, cautious, wrong = split_infinite(rows)
    agree = len(rows) - len(silent) - len(cautious) - len(wrong)
    say(
        f"  no change of status within 0–48 dB: {len(rows)}; the report agrees on {agree}, reports nothing for "
        f"{len(silent)} silent ones, is cautious on {len(cautious)} and wrong on {len(wrong)}",
    )
    for r in cautious + wrong:
        say(
            f"    {'cautious' if r in cautious else 'WRONG'}: {r['family']}/{r['name']} at {r['drive']:.1f} dB: "
            f"report {r['estimate']:.3f} dB, full renders {r['reference']}",
        )


def _finite(rows: list[Row]) -> None:
    """Report the error of the reported edge where full renders find one within 0–48 dB."""
    errors = np.array([r["error"] for r in rows])
    say(
        f"  an edge within 0–48 dB: {len(rows)}; error of the report (+ is optimistic): "
        f"max {errors.max():+.5f} dB, min {errors.min():+.4f} dB, median {np.median(errors):+.5f} dB",
    )
    say(f"    exact to 0.001 dB: {int(np.sum(np.abs(errors) < EXACT_DB))}")
    for lo, hi in ((0.001, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, math.inf)):
        n = sum(1 for e in errors if lo <= -e < hi)
        say(f"    conservative by {lo:g}–{hi:g} dB: {n}")
    near = [r for r in rows if abs(r["reference"] - r["drive"]) <= NEAR_DB]
    if near:
        worst = -min(r["error"] for r in near)
        say(f"  within 3 dB of the edge ({len(near)}): at most {worst:.3f} dB conservative")
    for side, rs in (("inside", [r for r in rows if r["inside"]]), ("outside", [r for r in rows if not r["inside"]])):
        if rs:
            e = np.array([r["error"] for r in rs])
            say(f"  {side} ({len(rs)}): max {e.max():+.5f} dB, min {e.min():+.4f} dB")
    for r in sorted(rows, key=lambda r: r["error"])[:5]:
        say(
            f"    most conservative: {r['family']}/{r['name']} at {r['drive']:.1f} dB: "
            f"report {r['estimate']:.3f}, full renders {r['reference']:.3f} dB",
        )


def analyze(args: argparse.Namespace) -> None:
    """Report how the reported drive headroom compares with full renders, on every fixture."""
    rows = load(args.csv)
    say(f"{args.csv}: {len(rows)} fixtures, {sum(r['renders'] for r in rows)} full renders")
    _infinite([r for r in rows if not math.isfinite(r["reference"])])
    finite = [r for r in rows if not math.isnan(r["error"]) and math.isfinite(r["reference"])]
    if finite:
        _finite(finite)
    dense = [r for r in rows if not math.isnan(r["dense"])]
    if dense:
        missed = [r for r in dense if abs(r["dense"] - r["reference"]) > DENSE_STEP_DB + 1e-9]
        say(f"  dense 0.05 dB scans: {len(dense)}; a change of status before the edge in {len(missed)}")
        for r in missed:
            say(f"    {r['family']}/{r['name']}: dense {r['dense']:.2f} dB, edge {r['reference']:.3f} dB")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the headroom tool."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(grid.ROOT / "docs/grid/headroom.csv"))
    r.add_argument("--dense", action="store_true")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    a = sub.add_parser("analyze")
    a.add_argument("csv")
    args = ap.parse_args(argv)
    (run if args.cmd == "run" else analyze)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
