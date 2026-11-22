# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monotone cubic Hermite curves, their derivatives and jumps, and their time maps (docs/design.md: Shapes)."""

import math
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from brightwork_synth.curve import Curve, amplitude_curve, mapped, pchip_slopes, peak_index

type Knots = tuple[NDArray[np.float64], NDArray[np.float64]]

SOMETIMES = 0.3  # how often a random curve or shape takes an option
ENDS = 2  # a curve's first and last knots
COARSE_DB = 20  # knots this far apart are judged to a tenth of a dB, closer ones to a hundredth


def random_knots(rng: np.random.Generator) -> Knots:
    """Draw knots of one of four kinds: random values, monotone, flat runs, or amplitude-like shapes."""
    n = int(rng.integers(2, 33))
    t = np.cumsum(10 ** rng.uniform(-5, 0, n)) - 10 ** rng.uniform(-6, 0)
    match int(rng.integers(4)):
        case 0:
            v = rng.normal(size=n) * 10 ** rng.uniform(-3, 3)
        case 1:
            v = np.sort(rng.uniform(0, 20, n))[:: int(rng.choice([1, -1]))]
        case 2:  # flat runs and repeated values
            v = np.asarray(rng.choice([0.0, 0.5, 1.0, 2.0], n), dtype=np.float64)
        case _:  # amplitude-like shapes
            v = rng.uniform(0, 1, n)
            v[0] = v[-1] = 0.0
            v[rng.integers(1, n - 1) if n > ENDS else 0] = 1.0
    return t, v


def random_curves(seed: int, count: int) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64], Curve]]:
    """Draw curves through random knots, with the end slopes overridden or not."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        t, v = random_knots(rng)
        override = rng.integers(3)
        if override == 0:
            yield t, v, Curve.pchip(t, v)
        elif override == 1:
            yield t, v, Curve.pchip(t, v, first=0.0, last=0.0)
        else:
            yield t, v, Curve.pchip(t, v, last=0.0)


def test_slopes_match_scipy_pchip() -> None:
    """Compute PCHIP slopes, and find SciPy's on 2,000 random knot sets."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        t, v = random_knots(rng)
        ours = pchip_slopes(t, v)
        theirs = PchipInterpolator(t, v).derivative()(t)
        scale = np.max(np.abs(np.diff(v) / np.diff(t)))
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12 * scale)


def test_evaluation_matches_independent_hermite() -> None:
    """Evaluate a curve, and find SciPy's Hermite spline, with every knot reproduced exactly."""
    for t, v, c in random_curves(2, 500):
        x = np.linspace(t[0], t[-1], 997)
        ref = CubicHermiteSpline(t, v, c.d)(x)
        np.testing.assert_allclose(c(x), ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(v)) + 1e-300)
        assert np.array_equal(c(t), v)  # knots are reproduced exactly


def test_sliced_evaluation_is_the_same_curve() -> None:
    """Evaluate a curve segment by segment on sorted times, and find the very same values as point by point."""
    # Synthesis evaluates segment by segment on sorted times; it must give the very same values.
    rng = np.random.default_rng(14)
    for t, _, c in random_curves(15, 400):
        span = t[-1] - t[0]
        x = np.sort(np.concatenate((rng.uniform(t[0] - 0.3 * span, t[-1] + 0.3 * span, 3000), t)))
        assert np.array_equal(c.on(x), c(x))
        assert np.array_equal(c.on(x[:0]), x[:0])


def test_constant_outside_knots() -> None:
    """Hold a curve's end values outside its knots."""
    for t, v, c in random_curves(3, 200):
        span = t[-1] - t[0]
        assert np.all(c(t[0] - span * np.array([1e-9, 0.5, 3])) == v[0])
        assert np.all(c(t[-1] + span * np.array([1e-9, 0.5, 3])) == v[-1])


def test_segments_stay_in_their_knot_interval() -> None:
    """Keep every segment within its two knot values, as the slopes guarantee and clipping only rounds."""
    # Monotonicity is a property of the slopes, so check it with an independent
    # evaluator that does not clip, then check that clipping removes nothing else.
    worst = 0.0
    for t, v, c in random_curves(4, 1000):
        s = np.linspace(0, 1, 257)
        for i in range(t.size - 1):
            x = np.clip(t[i] + s * (t[i + 1] - t[i]), t[i], t[i + 1])
            y = CubicHermiteSpline(t, c.v, c.d)(x)
            lo, hi = min(v[i], v[i + 1]), max(v[i], v[i + 1])
            scale = max(abs(v[i]), abs(v[i + 1]), 1e-300)
            worst = max(worst, (lo - y.min()) / scale, (y.max() - hi) / scale)
            ours = c(x)
            assert ours.min() >= lo
            assert ours.max() <= hi
    assert worst <= 16 * np.finfo(float).eps  # the independent evaluator's own rounding


def test_amplitude_shape_confined_to_unit_interval() -> None:
    """Keep an amplitude shape within 0 … 1, with zero end slopes."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(3, 33))
        t = np.cumsum(10 ** rng.uniform(-5, -1, n))
        v = rng.uniform(0, 1, n)
        v[0] = v[-1] = 0.0
        v[rng.integers(1, n - 1)] = 1.0
        c = amplitude_curve(t, v)
        assert c.d[0] == 0.0
        assert c.d[-1] == 0.0
        y = c.on(np.linspace(t[0] - 0.01, t[-1] + 0.01, 20001))
        assert y.min() >= 0.0
        assert y.max() <= 1.0


def test_a_damper_never_increases() -> None:
    """Draw a damper from exactly 1 to exactly 0 that never increases between its knots."""
    # docs/design.md (Shapes): from exactly 1 down to exactly 0 through knots that never increase.
    rng = np.random.default_rng(16)
    for _ in range(300):
        n = int(rng.integers(2, 33))
        t = np.cumsum(10 ** rng.uniform(-5, -1, n))
        v = np.sort(rng.uniform(0, 1, n))[::-1]
        v[0], v[-1] = 1.0, 0.0
        if n > ENDS + 1 and rng.random() < SOMETIMES:
            v[1] = 1.0  # a flat start
        c = amplitude_curve(t, v)
        x = np.linspace(t[0] - 0.01, t[-1] + 0.01, 20001)
        y = c.on(x)
        assert np.all(np.diff(y) <= 0)
        assert y[0] == 1.0
        assert y[-1] == 0.0


def test_rejects_bad_knots() -> None:
    """Refuse knot times that do not increase, and a single knot."""
    with pytest.raises(ValueError, match="knot times must increase"):
        Curve([0.0, 0.0], [1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="at least two knots"):
        Curve([0.0], [1.0], [0.0])


def test_derivatives_match_independent_hermite() -> None:
    """Compute a curve's first and second derivatives, and find SciPy's, and 0 outside the knots."""
    for t, v, c in random_curves(12, 300):
        spline = CubicHermiteSpline(t, c.v, c.d)
        x = np.linspace(t[0], t[-1], 499)[:-1]
        scale = np.max(np.abs(np.diff(v) / np.diff(t))) * 3 + np.max(np.abs(c.d)) + 1e-300
        np.testing.assert_allclose(c.derivative(x), spline.derivative()(x), rtol=1e-9, atol=1e-9 * scale)
        assert np.all(c.derivative(t[-1] + np.array([1e-3, 1.0])) == 0)
        np.testing.assert_allclose(c.derivative(t), c.d, rtol=1e-9, atol=1e-9 * scale)
        scale2 = scale / np.min(np.diff(t)) * 10
        np.testing.assert_allclose(c.second(x), spline.derivative(2)(x), rtol=1e-8, atol=1e-9 * scale2)
        assert np.all(c.second(np.array([t[0] - 1.0, t[-1], t[-1] + 1.0])) == 0)
        knots = t[1:-1]  # where it jumps: the segment starting there, as a piecewise polynomial reads it too
        np.testing.assert_allclose(c.second(knots), spline.derivative(2)(knots), rtol=1e-8, atol=1e-9 * scale2)


def test_jumps_are_the_one_sided_derivative_differences() -> None:
    """Compute each knot's jumps in the second and third derivatives, from both sides."""
    for t, v, c in random_curves(13, 300):
        spline = CubicHermiteSpline(t, c.v, c.d)
        d2, d3 = spline.derivative(2), spline.derivative(3)
        j2, j3 = c.jumps()
        n = t.size
        for j in range(n):
            after2 = d2.c[-1, j] if j < n - 1 else 0.0
            before2 = d2.c[-1, j - 1] + d2.c[-2, j - 1] * (t[j] - t[j - 1]) if j > 0 else 0.0
            after3 = d3.c[-1, j] if j < n - 1 else 0.0
            before3 = d3.c[-1, j - 1] if j > 0 else 0.0
            scale = (np.max(np.abs(np.diff(v))) / np.min(np.diff(t)) ** 2) * 20 + 1e-300
            assert j2[j] == pytest.approx(after2 - before2, rel=1e-8, abs=1e-9 * scale)
            assert j3[j] == pytest.approx(after3 - before3, rel=1e-8, abs=1e-9 * scale / np.min(np.diff(t)))
            b2, a2, b3, a3 = c.sides(j)
            assert a2 - b2 == pytest.approx(j2[j], rel=1e-14, abs=1e-14 * scale)
            assert a3 - b3 == pytest.approx(j3[j], rel=1e-14, abs=1e-14 * scale / np.min(np.diff(t)))


@pytest.mark.parametrize(("spacing_db", "within_db"), [(6, 0.03), (10, 0.15), (20, 1.3)])
def test_points_every_few_db_reproduce_an_exponential_tail(spacing_db: float, within_db: float) -> None:
    """Follow an exponential tail within 0.03, 0.15 or 1.3 dB with knots every 6, 10 or 20 dB."""
    # docs/design.md (Shapes) authoring guidance, on the interior of a tail.
    tau = 0.1
    step = spacing_db * math.log(10) / 20 * tau
    knots = 0.001 + step * np.arange(12)
    c = amplitude_curve(
        np.concatenate(([0.0], knots, [knots[-1] + step])),
        np.concatenate(([0.0], np.exp(-(knots - 0.001) / tau), [0.0])),
    )
    x = np.linspace(knots[1], knots[-3], 100001)
    deviation = 20 * np.log10(c(x) / np.exp(-(x - 0.001) / tau))
    assert round(float(np.max(np.abs(deviation))), 2 if spacing_db < COARSE_DB else 1) <= within_db


@pytest.mark.parametrize(("spacing_db", "above_db"), [(6, 1.1), (10, 2.0), (20, 5.1)])
def test_the_first_segment_after_a_peak_is_rounder(spacing_db: float, above_db: float) -> None:
    """Start a decay flat from its peak, so its first segment lies 1.1 to 5.1 dB above the exponential."""
    # The peak's slope is zero, so the decay starts flat (docs/design.md: Shapes): a tick needs a shoulder knot.
    tau = 0.1
    step = spacing_db * math.log(10) / 20 * tau
    knots = 0.001 + step * np.arange(12)
    c = amplitude_curve(
        np.concatenate(([0.0], knots, [knots[-1] + step])),
        np.concatenate(([0.0], np.exp(-(knots - 0.001) / tau), [0.0])),
    )
    x = np.linspace(knots[0], knots[1], 100001)
    above = 20 * np.log10(c(x) / np.exp(-(x - 0.001) / tau))
    assert float(np.max(above)) == pytest.approx(above_db, abs=0.05)


# Time maps.


def random_shape(rng: np.random.Generator, span: float = 2.0) -> Curve:
    """Draw an amplitude shape with microsecond knots, sometimes a delayed onset or a plateau at its peak."""
    n = int(rng.integers(3, 33))
    h = 10 ** rng.uniform(-5, 0, n - 1)
    t = np.concatenate(([0.0], np.cumsum(h)))
    t *= min(1.0, span / t[-1])
    t = np.round(t * 1e6) / 1e6  # microsecond knots, as a patch writes them
    t = np.maximum.accumulate(t + np.arange(n) * 1e-6)
    if rng.random() < SOMETIMES:  # a delayed onset
        t = t + rng.uniform(0, 0.05)
    v = rng.uniform(0, 1, n)
    v[0] = v[-1] = 0.0
    v[rng.integers(1, n - 1)] = 1.0
    if rng.random() < SOMETIMES:  # a plateau at the peak
        k = int(np.flatnonzero(v == 1.0)[0])
        v[min(k + 1, n - 2)] = 1.0
    return amplitude_curve(t, v)


def mapped_time(e: Curve, x: NDArray[np.float64], r: float, s: float) -> NDArray[np.float64]:
    """Return τ(x), the time a sub-band reads its shape at, for a rise stretch r and a decay compression s."""
    t0, tpk = e.t[0], e.t[peak_index(e)]
    turn = t0 + r * (tpk - t0)
    out: NDArray[np.float64] = np.where(x <= turn, t0 + (x - t0) / r, tpk + s * (x - turn))
    return out


@pytest.mark.parametrize(("span", "top"), [(0.3, 20.0), (2.0, 1000.0)])
def test_a_mapped_shape_is_the_shape_at_mapped_time(span: float, top: float) -> None:
    """Map a shape, and find it the shape read at the mapped time, to the rounding of that time."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        e = random_shape(rng, span)
        r, s = top ** rng.uniform(0, 1), top ** rng.uniform(0, 1)
        if rng.random() < SOMETIMES:
            r = 1.0
        c = mapped(e, r, s)
        x = np.sort(np.concatenate((rng.uniform(-0.01, c.t[-1] * 1.1, 500), c.t)))
        tau = mapped_time(e, x, r, s)
        # Any representation carries the rounding of the mapped time, a few ulps of the time
        # times the map's slope, through the shape's slope, which is at most 3 times its steepest secant.
        slope = 3 * np.max(np.abs(np.diff(e.v)) / e.h)
        tol = 1e-15 + 16 * np.finfo(float).eps * (np.abs(x) + c.t[-1]) * max(s, 1 / r) * slope
        assert np.all(np.abs(c(x) - e(tau)) <= tol)
        np.testing.assert_allclose(c.on(x), c(x), rtol=0, atol=0)


def test_mapped_shapes_keep_their_peak_and_monotone_segments() -> None:
    """Map a shape, and keep its onset, its peak of exactly 1, and its segments monotone within 0 … 1."""
    rng = np.random.default_rng(6)
    for _ in range(300):
        e = random_shape(rng)
        r, s = 10 ** rng.uniform(0, 3), 10 ** rng.uniform(0, 3)
        c = mapped(e, r, s)
        k = peak_index(c)
        assert k == peak_index(e)
        assert c.v[k] == 1.0
        assert c(c.t[k]) == 1.0
        assert c.t[0] == e.t[0]  # the onset stays where it was
        assert c.t[k] == pytest.approx(e.t[0] + r * (e.t[k] - e.t[0]), rel=1e-15, abs=1e-18)
        assert np.all(np.diff(c.t) > 0)
        for i in range(c.t.size - 1):
            x = np.linspace(c.t[i], c.t[i + 1], 101)
            y = c(x)
            assert np.all(np.diff(y) * np.sign(c.v[i + 1] - c.v[i]) >= 0)
            assert np.all((y >= 0) & (y <= 1))


def test_mapped_jumps_scale_on_the_rise_and_the_decay() -> None:
    """Map a shape, and find its jumps scaled by r⁻², r⁻³ on the rise and s², s³ on the decay."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        e = random_shape(rng, span=0.3)
        r, s = 8 ** rng.uniform(0, 2), 8 ** rng.uniform(0, 3)
        c = mapped(e, r, s)
        k = peak_index(e)
        for j in range(e.t.size):
            b2, a2, b3, a3 = e.sides(j)
            if j < k:
                b2, a2, b3, a3 = b2 / r**2, a2 / r**2, b3 / r**3, a3 / r**3
            elif j == k:  # the stretched rise joins the compressed decay
                b2, b3 = b2 / r**2, b3 / r**3
                a2, a3 = a2 * s**2, a3 * s**3
            else:
                b2, a2, b3, a3 = b2 * s**2, a2 * s**2, b3 * s**3, a3 * s**3
            got = c.sides(j)
            scale = max(abs(b2), abs(a2), abs(b3) * np.min(c.h), abs(a3) * np.min(c.h), 1e-300)
            # A mapped interval carries the rounding of its two mapped times, relative to its length.
            rtol = max(1e-7, 64 * np.finfo(float).eps * c.t[-1] / np.min(c.h))
            np.testing.assert_allclose(got[:2], (b2, a2), rtol=rtol, atol=1e-9 * scale)
            np.testing.assert_allclose(got[2:], (b3, a3), rtol=rtol, atol=1e-9 * scale / np.min(c.h))


def test_the_identity_map_is_the_shape_itself() -> None:
    """Map a shape by r = s = 1, and get the shape itself."""
    e = amplitude_curve([0.0, 0.0004, 0.025, 0.18], [0.0, 1.0, 0.45, 0.0])
    assert mapped(e, 1.0, 1.0) is e
    assert mapped(e) is e


def test_a_map_needs_a_flat_peak() -> None:
    """Refuse to map a shape whose peak is not flat."""
    bent = Curve([0.0, 0.001, 0.01], [0.0, 1.0, 0.0], [0.0, 5.0, 0.0])
    with pytest.raises(ValueError, match="zero slope at the peak"):
        mapped(bent, 1.0, 2.0)


@pytest.mark.parametrize("gain", [5e-324, 1e-320, 2.2250738585072014e-308])
def test_a_subnormal_knot_has_a_slope_within_rounding_of_zero(gain: float) -> None:
    """Find the slope at a knot a subnormal gain away from its neighbours' 0, where a reciprocal overflows."""
    d = pchip_slopes([0.0, 1.0, 2.0, 3.0], [0.0, gain, 2 * gain, 0.0])
    assert np.all(np.isfinite(d))
    assert np.all(np.abs(d) <= 3 * 2 * gain)  # the shape-preserving rules: at most three times the steepest secant
