"""The frozen validated-domain rule against the recorded adversarial grid (plan §8.3).

The grid itself takes a while (tools/grid.py run); this re-checks the rule's
constant against its committed results at the shipping K, so changing the
rule needs new evidence, and a fixture that has changed since its row was
recorded fails here until the grid is run again.
"""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
grid = pytest.importorskip("grid")

from cymbal.render import K, QUIET, SATURATION  # noqa: E402

FAMILIES = {
    "noise",
    "damping",
    "chatter",
    "ticks",
    "tones",
    "chokes",
    "layers",
    "dynamics",
    "seeds",
    "rates",
    "random",
    "longest",
}


@pytest.fixture(scope="module")
def rows():
    path = ROOT / "docs" / "grid" / f"k{K}.csv"
    if not path.exists():
        pytest.skip("no grid evidence recorded")
    return grid.load(path)


def test_the_grid_is_the_current_fixtures(rows):
    current = {(fam, name): json.dumps(edits) for fam, name, edits in grid.fixtures()}
    recorded = {(r["family"], r["name"]): r["edits"] for r in rows}
    assert recorded == current


def test_the_grid_covers_the_corners(rows):
    assert {r["family"] for r in rows} == FAMILIES
    assert len(rows) >= 900
    assert sum(r["family"] == "random" for r in rows) >= 200
    seeds = [r for r in rows if r["family"] == "seeds"]
    assert {(int(r["seed"]), int(r["take"])) for r in seeds} == {(s, t) for s in range(1, 5) for t in range(1, 5)} - {
        (1, 1)
    }
    assert max(r["length_ms"] for r in rows) == 8000.0


def test_every_fixture_inside_the_validated_domain_passes(rows):
    inside = [r for r in rows if r["inside"]]
    assert len(inside) > 450
    assert all(r["pass"] and r["converged"] for r in inside)
    assert max(r["e_max"] for r in inside if r["peak"] >= QUIET) <= -105


def test_the_kink_bound_holds_without_drive(rows):
    linear = [r for r in rows if r["alpha"] == 0.0 and r["converged"]]
    assert len(linear) > 300
    assert all(r["max_abs"] <= r["kink"] or r["max_abs"] <= 1e-15 * r["peak"] for r in linear)


def test_the_kink_bound_holds_under_drive(rows):
    driven = [r for r in rows if r["alpha"] > 0.0 and r["converged"] and r["kink"] >= 2 * SATURATION * r["saturation"]]
    assert len(driven) > 100
    assert all(r["max_abs"] <= r["kink"] or r["max_abs"] <= 1e-15 * r["peak"] for r in driven)


def test_the_saturation_coefficient_has_margin(rows):
    need, _ = grid.fit_saturation(rows)
    assert need <= SATURATION  # the fit already asks for 6 dB of margin over the worst fixture


def test_references_converge_wherever_the_rule_relies_on_them(rows):
    # Far outside the domain, heavy drive spreads harmonics so far up that even 6.144 MHz aliases: those
    # references do not converge, and the rule never needs them, since it puts those fixtures 20 dB or more outside.
    unconverged = [r for r in rows if not r["converged"]]
    assert all(r["margin_db"] < -20 for r in unconverged)
    assert len(unconverged) < len(rows) / 5
    # Every 512 × base check agrees at least as well as the 128/256 × base pair.
    checks = [r for r in rows if not math.isnan(r["c3_max"])]
    assert len(checks) >= 80
    assert all(r["c3_max"] <= r["c_max"] + 1e-9 or r["c3_max"] <= -200 for r in checks)


@pytest.fixture(scope="module")
def headroom_rows():
    headroom = pytest.importorskip("headroom")
    path = ROOT / "docs" / "grid" / "headroom.csv"
    if not path.exists():
        pytest.skip("no headroom evidence recorded")
    return headroom.load(path)


def test_the_reported_drive_headroom_is_never_optimistic(headroom_rows):
    # tools/headroom.py compares the report's headroom on every grid fixture with full renders.
    rows = headroom_rows
    assert {(r["family"], r["name"]) for r in rows} == {(f, n) for f, n, _ in grid.fixtures()}
    edges = [r for r in rows if math.isfinite(r["reference"])]
    assert len(edges) > 300
    assert max(r["error"] for r in edges) <= 1e-4  # the brute force's bisection resolution
    # Where full renders never change status within 0–48 dB, the report says so too, or is cautious: it may
    # place an edge where the status in fact stays inside, but never claims a way back in that is not there.
    headroom = pytest.importorskip("headroom")
    _, cautious, wrong = headroom.split_infinite([r for r in rows if not math.isfinite(r["reference"])])
    assert not wrong and len(cautious) <= 3


def test_no_change_of_status_hides_between_the_candidates(headroom_rows):
    dense = [r for r in headroom_rows if r["dense"] == r["dense"]]
    assert len(dense) >= 100
    assert all(abs(r["dense"] - r["reference"]) <= 0.05 + 1e-6 for r in dense)
