# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rattles: collisions seeded per take, thinned by the shape, at unit mean square, evaluated exactly."""

import itertools
import math

import numpy as np
import pytest

from brightwork_synth import patch, rattle, reference, render, voice
from brightwork_synth.curve import amplitude_curve

from .conftest import make
from .test_patch import amp, reject

RATE = 1536000  # 48 kHz at 32×
TAKES = 200
SHAKE = amplitude_curve([0.0, 0.01, 0.15, 0.3], [0.0, 1.0, 0.4, 0.0])  # one shake: a rise, a fall, a tail
SIGMAS = 4  # how far a count may fall from its expectation, in standard deviations
SAMPLED = 0.1  # how close 200 takes bring a mean square to its expectation
STEADY_DB = 0.3  # the standard deviation of a dense shake's energy over takes
ROUNDING = 1e-13  # how close the fast evaluation comes to the collisions summed one by one
SHAKER = {
    "noise[1].level_db": 0.0,
    "noise[1].band_hz": [3000.0, 9000.0],
    "noise[1].spacing_hz": 0.0,
    "noise[1].damping": 0.0,
    "noise[1].grains_hz": 800.0,
    "noise[1].grain_ms": 6.0,
    "noise[1].amp": amp((0, 0), (20, 1), (60, 0.6), (150, 0.1), (250, 0)),
    "noise[2].level_db": -math.inf,
}


def test_a_collision_rises_in_a_tenth_of_a_millisecond_and_falls_60_db() -> None:
    """Shape one collision: 0, 1 after 0.1 ms, then 10 dB lower at every sixth of the ring, and 0 at its end."""
    beta = rattle.collision(30.0)
    assert beta(np.array([0.0, rattle.ATTACK_S]))[1] == 1.0
    steps = rattle.ATTACK_S + 0.030 * np.arange(1, 6) / 6
    np.testing.assert_allclose(20 * np.log10(beta(steps)), -10.0 * np.arange(1, 6), rtol=0, atol=1e-12)
    assert beta(np.array([rattle.ATTACK_S + 0.030]))[0] == 0.0
    assert np.all(np.diff(beta(np.linspace(rattle.ATTACK_S, rattle.ATTACK_S + 0.03, 2001))) <= 0)


def test_collisions_are_the_takes_and_shared_by_sources_of_one_rate() -> None:
    """Draw the same collisions for the same seed, take and rate, others for another take, nested under shapes."""
    a = rattle.grains(SHAKE, 600.0, 8.0, 1, 1)
    assert a is not None
    same = rattle.grains(SHAKE, 600.0, 20.0, 1, 1)
    other = rattle.grains(SHAKE, 600.0, 8.0, 1, 2)
    assert same is not None
    assert other is not None
    assert np.array_equal(a.onsets, same.onsets)
    assert not np.array_equal(a.onsets[:10], other.onsets[:10])
    quieter = amplitude_curve(SHAKE.t, SHAKE.v * 0.5)
    fewer = rattle.grains(quieter, 600.0, 8.0, 1, 1)
    assert fewer is not None
    assert set(fewer.onsets) < set(a.onsets)  # a lower shape keeps a subset of the same collisions
    assert rattle.grains(SHAKE, 0.0, 8.0, 1, 1) is None


def test_collisions_crowd_at_the_peak_and_thin_out_with_the_shape() -> None:
    """Count collisions over many takes: in each stretch of the shake, the rate times the shape's area there."""
    edges = np.array([0.0, 0.01, 0.05, 0.15, 0.3])
    counts = np.zeros(edges.size - 1)
    for take in range(1, TAKES + 1):
        g = rattle.grains(SHAKE, 500.0, 5.0, 3, take)
        assert g is not None
        counts += np.histogram(g.onsets, edges)[0]
    t = np.linspace(0, 0.3, 300001)
    e = SHAKE(t)
    for (lo, hi), seen in zip(itertools.pairwise(edges), counts, strict=True):
        inside = (t >= lo) & (t < hi)
        expected = TAKES * 500.0 * float(np.sum(e[inside]) * (t[1] - t[0]))
        assert seen == pytest.approx(expected, abs=SIGMAS * math.sqrt(expected))  # a Poisson count


@pytest.mark.parametrize(("hz", "ms"), [(50.0, 4.0), (800.0, 6.0), (3000.0, 15.0)])
def test_a_rattle_has_unit_mean_square_at_the_peak_and_fewer_collisions_below_it(hz: float, ms: float) -> None:
    """Average R² over takes: 1 where the shape is 1, and the thinned stream's mean square where it is ½."""
    edge, end = 0.2, 0.4  # the shape is 1 until the edge, then ½ until the end
    steps = amplitude_curve([0.0, 0.001, edge, edge + 0.001, end, end + 0.001], [0.0, 1.0, 1.0, 0.5, 0.5, 0.0])
    t = np.linspace(0.0, end, 4001)
    total = np.zeros_like(t)
    for take in range(1, TAKES + 1):
        g = rattle.grains(steps, hz, ms, 5, take)
        assert g is not None
        total += g.value(t) ** 2
    mean = total / TAKES
    settled = 0.02 + ms / 1000  # past a step, once the collisions before it have rung out
    peak = (t > settled) & (t < edge - 0.01)
    half = (t > edge + settled) & (t < end - 0.01)
    assert float(np.mean(mean[peak])) == pytest.approx(1.0, abs=SAMPLED)
    # Keeping each collision with probability p scales single terms by p and pairs by p²: E = p² m + (p − p²) rate ∫β².
    beta = rattle.collision(ms)
    fine = np.arange(0.0, ms / 1000 + 2e-4, 1e-6)
    m = rattle.mean_square(beta, hz)
    thinned = m / 4 + hz * float(np.sum(beta(fine) ** 2) * 1e-6) / 4
    assert float(np.mean(mean[half])) == pytest.approx(thinned / m, rel=SAMPLED)


def test_one_shake_is_about_as_loud_as_the_next() -> None:
    """Find a shake's energy within a few tenths of a dB of the next: collisions fall one to a cell, never clumped."""
    t = np.linspace(0.0, 0.3, 30001)
    e = SHAKE(t)
    energy = []
    for take in range(1, TAKES + 1):
        g = rattle.grains(SHAKE, 3000.0, 1.5, 1, take)
        assert g is not None
        energy.append(10 * math.log10(float(np.sum((e * g.value(t)) ** 2))))
    assert float(np.std(energy)) < STEADY_DB


def test_the_fast_evaluation_is_the_collisions_summed() -> None:
    """Evaluate R on the sample grid in runs, and find each collision summed to rounding, however it is split."""
    g = rattle.grains(SHAKE, 2000.0, 25.0, 7, 1)
    assert g is not None
    n0, n1 = 1234, 1234 + 300_000
    fast = g.on_grid(n0, n1, RATE)
    np.testing.assert_allclose(fast, g.value(np.arange(n0, n1) / RATE), rtol=0, atol=ROUNDING)
    whole = g.on_grid(0, 460_800, RATE)
    parts = np.concatenate([g.on_grid(a, min(a + 9_999, 460_800), RATE) for a in range(0, 460_800, 9_999)])
    assert np.array_equal(parts, whole)
    assert g.on_grid(5, 5, RATE).size == 0
    assert g.value(np.zeros(0)).size == 0
    assert not np.any(g.on_grid(RATE, RATE + 100, RATE))  # after the last collision rings out
    slope = (g.value([0.05 + 1e-9]) - g.value([0.05 - 1e-9])) / 2e-9
    np.testing.assert_allclose(g.derivative(np.array([0.05])), slope, rtol=1e-5)


def test_a_shaker_renders_inside_the_domain_and_agrees_with_its_reference() -> None:
    """Render a shaker: inside the validated domain, its error under the kink bound, its collisions reported."""
    p = make(**SHAKER)
    _, report = render.render(p)
    assert report.inside
    text = str(report)
    assert "collisions ringing 6 ms" in text
    assert "at noise[1].grains;" in text
    v = reference.verify_take(p)
    assert v.passed
    assert v.error.max_abs <= render.synthesize(p).kink


def test_rattles_are_validated() -> None:
    """Refuse a rate or a ring out of range, and collisions that overlap past where a rattle stays a rattle."""
    assert any(m.startswith("noise[1].grains_hz") for m in reject(**{"noise[1].grains_hz": -1.0}))
    assert any(m.startswith("noise[1].grain_ms") for m in reject(**{"noise[1].grain_ms": 0.5}))
    found = reject(**{"noise[1].grains_hz": 4000.0, "noise[1].grain_ms": 20.0})
    assert "noise[1].grains_hz: 4000 collisions per second ringing 20 ms each overlap 80 at once" in found[0]
    for slow in (5e-324, 1e-300, 0.1):
        [problem] = reject(**{"noise[1].grains_hz": slow})
        assert problem.startswith(f"noise[1].grains_hz: {slow:g} collisions per second leave less than one in 8 s")
    assert make(**{"noise[1].grains_hz": patch.MIN_GRAINS_HZ}).noise[0].grains_hz == patch.MIN_GRAINS_HZ
    v = voice.voice(make(**SHAKER))
    assert v.sources[1].grains is None  # silent: nothing to rattle
