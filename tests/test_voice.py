# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math

import numpy as np
import pytest
from conftest import EXAMPLE, make

from brightwork_synth import carrier, spectrum
from brightwork_synth import voice as V
from brightwork_synth.curve import amplitude_curve, mapped
from brightwork_synth.patch import Noise, loads

BASE = 48000
RATE = 32 * BASE  # 48 kHz at 32×
IP = carrier.interpolator(BASE, 16)


def amp(*points):
    return [{"ms": m, "gain": g} for m, g in points]


def partials(*pairs):
    return [{"hz": f, "level_db": v} for f, v in pairs]


def shape(knots):
    """The amplitude curve through (ms, gain) knots, given as pairs or as patch dicts."""
    pairs = [(k["ms"], k["gain"]) if isinstance(k, dict) else k for k in knots]
    return amplitude_curve([m / 1000 for m, _ in pairs], [g for _, g in pairs])


def source_lines(p, path, s):
    """A source's calibrated lines, from the spectrum module alone."""
    if isinstance(s, Noise):
        found = spectrum.noise_lines(p.seed, p.take, s.spacing_hz)
        return found.bins, spectrum.calibrated(found.bins, found.amps, s.band_hz, random=True)
    index = int(path[path.index("[") + 1 : -1])
    phases = (
        np.zeros(len(s.partials))
        if s.phase == "strike"
        else spectrum.free_phases(p.seed, p.take, index, len(s.partials))
    )
    bins, a, args = spectrum.tone_lines(s.partials, s.wave, phases)
    return spectrum.merged(bins, spectrum.calibrated(bins, a * np.exp(1j * args), s.band_hz, random=False))


def x_exact(p, t=None, n=None, rate=RATE):
    """The plan's model (§2) evaluated directly: every line summed exactly, at times t or at samples n / rate."""
    if n is not None:
        n = np.asarray(n, dtype=np.int64)
        t = n / rate
    t = np.asarray(t, dtype=float)
    x = np.zeros_like(t)
    damper = shape(p.damper.amp)(t) if p.damper.amp else 1.0
    for path, s in p.sources:
        if s.level_db == -math.inf:
            continue
        bins, amps = source_lines(p, path, s)
        lay = spectrum.layout(s.f_ref, s.damping, s.bloom)
        w = lay.windows(bins / 8)
        e = shape(s.amp)
        total = np.zeros_like(t)
        for b, (c, r) in enumerate(zip(lay.compression(s.damping), lay.stretch(s.bloom))):
            keep = w[b] > 0
            k, a = bins[keep], amps[keep] * w[b][keep]
            if n is not None:  # phases reduced exactly in integers
                cycles = np.mod(np.outer(n, k), 8 * rate) / (8 * rate)
            else:
                cycles = np.mod(np.outer(t, k / 8), 1.0)
            carrier_t = np.cos(2 * np.pi * cycles + np.angle(a)) @ np.abs(a)
            total += mapped(e, float(r), float(c))(t) * carrier_t
        gate = 1 - s.chatter * e(t) * (1 - np.cos(2 * np.pi * s.chatter_hz * t)) / 2
        x += 10 ** (s.level_db / 20) * gate * damper * total
    return x


def internal_x(p, count=None):
    v = V.voice(p)
    count = V.sample_count(v.t_src, RATE) if count is None else count
    return V.synthesize_x(v, RATE, count, IP, V.tables(v, IP.rate))


# The saturator.


def test_drive_identities():
    assert V.drive_alpha(0.0) == 0.0
    for db in (0.5, 6.0, 18.0, 48.0):
        a = V.drive_alpha(db)
        assert a / math.tanh(a) == pytest.approx(10 ** (db / 20), rel=1e-14)
        x = np.linspace(-3, 3, 10001)
        y = V.saturate(a, x)
        assert V.saturate(a, np.array([1.0]))[0] == 1.0 and V.saturate(a, np.array([-1.0]))[0] == -1.0
        assert np.array_equal(V.saturate(a, -x), -y)
        assert V.saturate(a, np.array([1e-9]))[0] / 1e-9 == pytest.approx(10 ** (db / 20), rel=1e-6)
        assert np.all(np.diff(y) >= 0)
    x = np.linspace(-2, 2, 101)
    assert np.array_equal(V.saturate(0.0, x), x)


def test_slope_and_curvature_are_the_derivatives():
    for db in (3.0, 24.0):
        a = V.drive_alpha(db)
        x = np.linspace(-1.5, 1.5, 301)
        h = 1e-6
        np.testing.assert_allclose(
            V.slope(a, x), (V.saturate(a, x + h) - V.saturate(a, x - h)) / (2 * h), rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            V.curvature(a, x), (V.slope(a, x + h) - V.slope(a, x - h)) / (2 * h), rtol=1e-6, atol=1e-5
        )


# The model.

VARIANTS = [
    {},
    {"noise[1].damping": 1.2, "noise[1].bloom": 0.6, "noise[1].chatter": 0.7, "noise[1].chatter_hz": 310.0},
    {"noise[1].spacing_hz": 0.0, "noise[2].spacing_hz": 3.5, "damper.amp": amp((40, 1), (45, 0.2), (60, 0))},
    {"noise[1].amp": amp((2.5, 0), (3.0, 1), (40, 0.3), (110, 0)), "take": 9, "seed": 2**63 - 1},
]


@pytest.mark.parametrize("edits", VARIANTS)
def test_noise_sources_are_the_model(edits):
    p = make(**edits)
    count = V.sample_count(p.length_ms / 1000, RATE)
    n = np.unique(np.concatenate(([0, 1, 77], np.random.default_rng(1).integers(0, count, 120))))
    np.testing.assert_allclose(internal_x(p)[n], x_exact(p, n=n), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "edits",
    [
        {},
        {"tone[1].phase": "strike", "tone[1].damping": 1.5, "tone[1].bloom": 0.4},
        {
            "tone[1].wave": "sine",
            "tone[1].phase": "strike",
            "tone[1].partials": partials((690, 0), (1140, -4), (2280, -9)),
            "tone[1].band_hz": [20.0, 20000.0],
            "tone[1].chatter": 0.5,
        },
    ],
)
def test_tones_are_the_model(edits):
    p = make(tone=True, **{"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf, **edits})
    count = V.sample_count(p.length_ms / 1000, RATE)
    n = np.unique(np.concatenate(([0, 1], np.random.default_rng(2).integers(0, count, 150))))
    np.testing.assert_allclose(internal_x(p)[n], x_exact(p, n=n), rtol=0, atol=1e-12)


def test_blocks_threads_and_longer_renders_never_change_samples():
    p = make(**VARIANTS[1])
    v = V.voice(p)
    tabs = V.tables(v, IP.rate)
    count = V.sample_count(v.t_src, RATE)
    whole = V.synthesize_x(v, RATE, count, IP, tabs)
    assert np.array_equal(V.synthesize_x(v, RATE, count, IP, tabs, workers=1), whole)
    short = V.synthesize_x(v, RATE, count // 3, IP, tabs, workers=3)
    assert np.array_equal(short, whole[: count // 3])


def test_every_hit_starts_at_exactly_zero():
    for edits in ({}, {"drive.db": 30.0}, {"noise[2].amp": amp((0, 0), (0.02, 1), (5, 0))}):
        assert internal_x(make(**edits))[0] == 0.0
    assert internal_x(make(tone=True))[0] == 0.0


def test_identical_sources_sum_coherently():
    # Two identical sources at equal level make +6 dB, not +3 dB: they are the same cymbal.
    p = make()
    one = dataclasses.replace(p, noise=(p.noise[0],))
    two = dataclasses.replace(p, noise=(p.noise[0], p.noise[0]))
    np.testing.assert_allclose(internal_x(two), 2 * internal_x(one), rtol=0, atol=1e-12)


def test_takes_keep_struck_tones_and_redraw_the_rest():
    strike = {"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf, "tone[1].phase": "strike"}
    a, b = internal_x(make(tone=True, **strike)), internal_x(make(tone=True, take=2, **strike))
    assert np.array_equal(a, b)
    free = {**strike, "tone[1].phase": "free"}
    assert not np.array_equal(internal_x(make(tone=True, **free)), internal_x(make(tone=True, take=2, **free)))
    assert not np.array_equal(internal_x(make()), internal_x(make(take=2)))


@pytest.mark.parametrize(
    "damping, bloom, b", [(0.5, 0.0, 1), (0.5, 0.0, 2), (1.0, 0.0, 2), (2.0, 0.5, 4), (2.0, 0.5, 8)]
)
def test_each_sub_band_plays_the_shape_on_its_time_map(damping, bloom, b):
    # A sine at a sub-band's centre is in that sub-band alone, so its envelope is the shape on the sub-band's time
    # map, written here from the plan: the rise r times slower from the onset, the decay s times faster after it.
    f_ref, onset, peak = 1000.0, 0.0005, 0.004
    lay = spectrum.layout(f_ref, damping, bloom)
    f = float(lay.centres[b])
    assert f in (2000.0, 4000.0)  # whole octaves up: a grid bin, where the window is exactly 1
    knots = amp((0.5, 0), (4, 1), *[(4 + 30 * i, 10 ** (-i / 2)) for i in range(1, 7)], (220, 0))
    p = make(
        tone=True,
        **{
            "noise[1].level_db": -math.inf,
            "noise[2].level_db": -math.inf,
            "tone[1].wave": "sine",
            "tone[1].phase": "strike",
            "tone[1].partials": partials((f_ref, -60.0), (f, 0.0)),  # the quiet one sets f_ref
            "tone[1].band_hz": [f_ref, 20000.0],
            "tone[1].damping": damping,
            "tone[1].bloom": bloom,
            "tone[1].amp": knots,
        },
    )
    x = internal_x(p)
    n = np.arange(x.size, dtype=np.int64)
    t = n / RATE
    s, r = 2.0 ** (b * lay.step * damping), 2.0 ** (b * lay.step * bloom)
    rise_end = onset + r * (peak - onset)
    tau = np.where(t <= rise_end, onset + (t - onset) / r, peak + s * (t - rise_end))
    e = shape(knots)

    def quadrature(hz):  # sin and cos of 2π·hz·t, phases reduced exactly in integers
        cycles = np.mod(n * spectrum.grid_bin(hz), 8 * RATE) / (8 * RATE)
        return np.sin(2 * np.pi * cycles), np.cos(2 * np.pi * cycles)

    # x = e(τ(t))·(a sin + b cos)(2πft) + e(t)·(c sin + d cos)(2πf_ref t), the f_ref partial on sub-band 0's identity map.
    basis = np.stack([*(e(tau) * q for q in quadrature(f)), *(e(t) * q for q in quadrature(f_ref))], axis=1)
    coef = np.linalg.lstsq(basis, x, rcond=None)[0]
    band = lambda hz: (1 + (f_ref / hz) ** 4) ** -0.5 * (1 + (hz / 20000) ** 4) ** -0.5  # noqa: E731
    ratio = 10 ** (-60 / 20) * band(f_ref) / band(f)
    assert math.hypot(coef[2], coef[3]) == pytest.approx(ratio * math.hypot(coef[0], coef[1]), rel=1e-9)
    assert np.max(np.abs(x - basis @ coef)) < 1e-11 * np.max(np.abs(x))
    assert x.size == V.sample_count(0.220, RATE)  # the f_ref partial's sub-band ends last


def test_damping_scales_decay_rates_by_the_compression():
    # Plan §8.1: an exponential shape and a sine at each of three sub-band centres. Each centre's fitted decay rate
    # is s_b times the shape's within 0.1%. A quiet partial at f_ref keeps the layout where it is.
    f_ref, damping, decay = 1000.0, 1.0, 0.04
    lay = spectrum.layout(f_ref, damping, 0.0)
    knots = amp((0, 0), (1, 1), *[(1 + decay * 1000 * i * math.log(10) / 2, 10 ** (-i / 2)) for i in range(1, 8)])
    knots.append({"ms": knots[-1]["ms"] + 20.0, "gain": 0.0})
    width = int(0.002 * RATE)  # 2 ms: whole periods of every partial here, so the quiet one falls in Hann's nulls
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(width) / width)
    rates = {}
    for b in (0, 2, 4):
        f = float(lay.centres[b])
        p = make(
            tone=True,
            **{
                "noise[1].level_db": -math.inf,
                "noise[2].level_db": -math.inf,
                "tone[1].wave": "sine",
                "tone[1].phase": "strike",
                "tone[1].partials": partials((f, 0.0)) if b == 0 else partials((f_ref, -60.0), (f, 0.0)),
                "tone[1].band_hz": [f_ref, 20000.0],
                "tone[1].damping": damping,
                "tone[1].amp": knots,
            },
        )
        x = internal_x(p)
        n = np.arange(x.size, dtype=np.int64)
        z = x * np.exp(-2j * np.pi * np.mod(n * spectrum.grid_bin(f), 8 * RATE) / (8 * RATE))
        starts = np.arange(0, x.size - width, width // 4)
        envelope = np.abs(np.stack([z[a : a + width] for a in starts]) @ hann)
        centre = (starts + width / 2) / RATE
        s = float(lay.compression(damping)[b])
        tau = 0.001 + s * (centre - 0.001)  # the shape's time
        sel = (tau >= 0.001 + decay * math.log(10) / 2) & (tau <= 0.001 + 2 * decay * math.log(10))  # 10 to 40 dB down
        rates[b] = (s, -np.polyfit(centre[sel], np.log(envelope[sel]), 1)[0])
    s0, r0 = rates[0]
    assert s0 == 1.0 and r0 == pytest.approx(1 / decay, rel=0.01)
    for b in (2, 4):
        s, r = rates[b]
        assert s == 2.0 ** (b / 2)
        assert r / r0 == pytest.approx(s, rel=1e-3)


@pytest.mark.parametrize("depth", [0.3, 1.0])
def test_chatter_gates_at_its_rate_with_its_depth(depth):
    flat = amp((0, 0), (0.5, 1), (100, 1), (100.5, 0))
    base = {"noise[2].level_db": -math.inf, "noise[1].amp": flat, "noise[1].damping": 0.0, "noise[1].chatter_hz": 250.0}
    plain = internal_x(make(**base))
    gated = internal_x(make(**base, **{"noise[1].chatter": depth}))
    t = np.arange(plain.size) / RATE
    sel = (t > 0.002) & (t < 0.099) & (np.abs(plain) > 1e-3)
    gate = gated[sel] / plain[sel]
    want = 1 - depth * (1 - np.cos(2 * np.pi * 250.0 * t[sel])) / 2
    np.testing.assert_allclose(gate, want, rtol=0, atol=1e-9)
    assert np.array_equal(internal_x(make(**base, **{"noise[1].chatter": 0.0})), plain)


def test_the_damper_multiplies_every_source_and_ends_the_sound():
    knots = amp((30, 1), (35, 0.3), (50, 0))
    p, q = make(tone=True), make(tone=True, **{"damper.amp": knots})
    assert q.length_ms == 50.0
    xp, xq = internal_x(p), internal_x(q)
    t = np.arange(xq.size) / RATE
    np.testing.assert_allclose(xq, xp[: xq.size] * shape([(30, 1), (35, 0.3), (50, 0)])(t), rtol=0, atol=1e-13)
    assert np.array_equal(xq[t < 0.030], xp[: xq.size][t < 0.030])


def test_tables_depend_only_on_the_lines_and_the_layout():
    # Plan §3: a table depends on the seed, the take, the spectrum, the band and the sub-band layout, and never on
    # shapes, levels, chatter, the damper, the drive or the output, so the most common edits reuse every table.
    def keys(**edits):
        return [sb.key for s in V.voice(make(tone=True, **edits)).audible for sb in s.subbands]

    base = keys()
    for edits in (
        {"noise[1].amp": amp((0, 0), (1, 1), (50, 0.2), (300, 0)), "tone[1].amp": amp((0, 0), (5, 1), (90, 0))},
        {"noise[1].level_db": -20.0, "tone[1].level_db": 6.0},
        {"noise[1].chatter": 0.9, "noise[1].chatter_hz": 1000.0},
        {"damper.amp": amp((30, 1), (40, 0))},
        {"drive.db": 30.0},
        {"output.sample_rate": 96000, "output.db": 3.0, "output.polarity": -1, "output.format": "pcm16"},
    ):
        assert keys(**edits) == base
    for edits in (
        {"seed": 2},
        {"take": 2},
        {"noise[1].band_hz": [6000.0, 15000.0]},
        {"noise[1].spacing_hz": 15.0},
        {"noise[1].damping": 1.0},
        {"tone[1].partials": partials((205.3, 0.0), (304.4, -1.0))},
    ):
        assert keys(**edits) != base


def test_source_length_and_sample_count():
    for ms in (120.0, 0.05, 1000.0 / 3):
        n = V.sample_count(ms / 1000, RATE)
        assert (n - 1) / RATE <= ms / 1000 < n / RATE
    v = V.voice(make())
    assert v.t_src == 0.120


# The knee slew at every drive.


def brute_knee(x, alpha):
    u = np.concatenate(([0.0], x))
    a, b = u[:-1], u[1:]
    step = np.abs(b - a)
    counts = (np.signbit(a) != np.signbit(b)) | (np.minimum(np.abs(a), np.abs(b)) <= 1.0 / alpha)
    return alpha * float(np.max(step, where=counts, initial=0.0))


@pytest.mark.parametrize("edits", [{}, {"noise[1].level_db": 12.0, "drive.db": 20.0}, {"noise[2].level_db": 6.0}])
def test_knee_record_is_exact_at_every_drive(edits):
    x = internal_x(make(**edits))
    slew = V.slew_of(x)
    alphas = np.concatenate(
        [
            [V.drive_alpha(db) for db in np.arange(0.5, 48.01, 0.5)],
            slew.knees.drops(),
            np.nextafter(slew.knees.drops(), np.inf),
        ]
    )
    for a in alphas:
        assert slew.knee(a) == pytest.approx(brute_knee(x, a), rel=1e-15, abs=0)
    assert slew.crest == int(np.argmax(np.abs(x)))
    u = np.concatenate(([0.0], x))
    crossing = np.where(np.signbit(u[:-1]) != np.signbit(u[1:]), np.abs(np.diff(u)), -1.0)
    assert slew.steep == int(np.argmax(crossing))
    assert slew.peak == float(np.max(np.abs(np.diff(u))))


def test_knees_never_count_a_step_again_once_it_stops():
    k = V.Knees(0.1, np.array([0.2, 0.5]), np.array([0.3, 0.4]))
    assert k.at(1.0) == pytest.approx(0.4)
    assert k.at(1 / 0.5) == pytest.approx(2 * 0.4)
    assert k.at(np.nextafter(2.0, 3.0)) == pytest.approx(2 * 0.3, rel=1e-12)
    assert k.at(10.0) == pytest.approx(10 * 0.1)
    assert k.at(0.0) == 0.0
    np.testing.assert_allclose(k.drops(), [5.0, 2.0])


# The kink bound: every jump against one-sided derivatives of the model.


def one_sided(f, tau, delta, side):
    """(f″, f‴) at tau from one side, by a least-squares polynomial on [tau, tau ± delta]."""
    u = np.linspace(0, 1, 24)[1:] * side
    coef = np.polyfit(u, f(tau + u * delta), 8)
    p = np.poly1d(coef)
    return p.deriv(2)(0.0) / delta**2, p.deriv(3)(0.0) / delta**3


KINKY = {
    "noise[1].spacing_hz": 40.0,
    "noise[1].band_hz": [3000.0, 9000.0],
    "noise[1].damping": 1.0,
    "noise[1].bloom": 0.5,
    "noise[1].chatter": 0.6,
    "noise[1].amp": amp((0, 0), (0.4, 1), (3, 0.5), (9, 0.2), (30, 0)),
    "noise[2].spacing_hz": 20.0,
    "noise[2].band_hz": [2000.0, 6000.0],
    "noise[2].amp": amp((0.7, 0), (1.9, 1), (11, 0.4), (40, 0)),
    "damper.amp": amp((25, 1), (28, 0.4), (36, 0)),
}


def kinky_jumps(p):
    v = V.voice(p)
    return V.kink_terms(v, IP, V.tables(v, IP.rate))


def test_every_jump_matches_the_model():
    p = make(**KINKY)
    k = kinky_jumps(p)
    times = np.unique(k.t)
    assert times.size > 20
    for tau in times:
        rows = k.t == tau
        others = times[times != tau]
        delta = min(2e-6, 0.25 * float(np.min(np.abs(others - tau))))
        f = lambda t: x_exact(p, t)  # noqa: E731
        b2, b3 = one_sided(f, tau, delta, -1) if tau > 0 else (0.0, 0.0)
        a2, a3 = one_sided(f, tau, delta, +1)
        got2, got3 = float(np.sum(k.d2[rows])), float(np.sum(k.d3[rows]))
        scale2 = max(abs(a2), abs(b2), 1.0)
        scale3 = max(abs(a3), abs(b3), 1e5)
        assert got2 == pytest.approx(a2 - b2, abs=2e-3 * scale2), (tau, [k.paths[i] for i in np.flatnonzero(rows)])
        assert got3 == pytest.approx(a3 - b3, abs=2e-3 * scale3), (tau, [k.paths[i] for i in np.flatnonzero(rows)])
        np.testing.assert_allclose(k.x[rows], x_exact(p, np.full(rows.sum(), tau)), rtol=0, atol=1e-9)


def test_jumps_through_the_saturator():
    p = make(**KINKY, **{"drive.db": 18.0})
    k = kinky_jumps(p)
    a = V.drive_alpha(18.0)
    times = np.unique(k.t)
    for tau in times[1:]:
        rows = k.t == tau
        delta = min(1e-6, 0.25 * float(np.min(np.abs(times[times != tau] - tau))))
        f = lambda t: V.saturate(a, x_exact(p, t))  # noqa: E731
        b2, b3 = one_sided(f, tau, delta, -1)
        a2, a3 = one_sided(f, tau, delta, +1)
        g1, g2 = V.slope(a, k.x[rows]), V.curvature(a, k.x[rows])
        got2 = float(np.sum(g1 * k.d2[rows]))
        got3 = float(np.sum(g1 * k.d3[rows] + 3 * g2 * k.dx[rows] * k.d2[rows]))
        assert got2 == pytest.approx(a2 - b2, abs=5e-3 * max(abs(a2), abs(b2)) + 1e-10 / delta**2)
        assert got3 == pytest.approx(a3 - b3, abs=5e-3 * max(abs(a3), abs(b3)) + 1e-10 / delta**3)


def test_kink_bound_form():
    p = make(**KINKY, **{"drive.db": 12.0})
    k = kinky_jumps(p)
    alpha = V.drive_alpha(12.0)
    total = V.kink_bound(k, alpha, RATE, 24000.0)
    assert total > 0
    assert sum(c for _, c in V.kink_knots(k, alpha, RATE, 24000.0)) == pytest.approx(total, rel=1e-12)
    alphas = np.array([0.0, 0.5, alpha, 30.0])
    np.testing.assert_allclose(
        V.kink_bounds(k, alphas, RATE, 24000.0), [V.kink_bound(k, a, RATE, 24000.0) for a in alphas], rtol=1e-12
    )
    w2, w3 = V._weights(RATE, 24000.0)
    assert w2 == pytest.approx(1.01 * 4 * 1.2020569031595942 * 24000 / (2 * np.pi * RATE) ** 3, rel=1e-15)
    assert w3 == pytest.approx(1.01 * 4 * np.pi**4 / 90 * 24000 / (2 * np.pi * RATE) ** 4, rel=1e-15)
    paths = {path for path, _ in V.kink_knots(k, alpha, RATE, 24000.0)}
    assert {"noise[1].amp[1]", "noise[1].amp[5]", "noise[2].amp[3]", "damper.amp[1]", "damper.amp[3]"} <= paths


def test_silence_has_no_kinks():
    p = make(**{"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf})
    k = kinky_jumps(p)
    assert k.t.size == 0 and V.kink_bound(k, 0.0, RATE, 24000.0) == 0.0


def test_example_patch_kinks_are_dominated_by_the_stick():
    p = loads(EXAMPLE)
    top = V.kink_knots(kinky_jumps(p), 0.0, RATE, 24000.0)[0][0]
    assert top.startswith("noise[2].amp")
