# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from brightwork_synth import carrier, spectrum

DATA = Path(carrier.__file__).parent / "data"


def noise(seed=1, band=(20.0, 20000.0), spacing=0.0):
    lines = spectrum.noise_lines(seed, 1, spacing)
    return lines.bins, spectrum.calibrated(lines.bins, lines.amps, band, random=True)


def exact(bins, amps, n, rate):
    """The carrier at t = n / rate by direct summation, with the phase reduced exactly in integers."""
    period = spectrum.PERIOD * rate
    out = np.zeros(n.size)
    for i0 in range(0, bins.size, 4096):
        k = bins[i0 : i0 + 4096].astype(np.int64)
        cycles = np.mod(np.outer(n.astype(np.int64), k), period) / period
        a = amps[i0 : i0 + 4096]
        out += np.cos(2 * np.pi * cycles + np.angle(a)) @ np.abs(a)
    return out


def rms(amps):
    return math.sqrt(0.5 * float(np.sum(np.abs(amps) ** 2)))


def test_shipped_interpolators_are_the_design_and_match_their_checksums():
    entries = json.loads((DATA / "interpolators.json").read_text())["filters"]
    assert set(entries) == {f"{b}x{k}" for b in (44100, 48000) for k in (16, 32, 64)}
    for name, e in entries.items():
        raw = (DATA / e["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"]
        taps, q = carrier.design(e["base"], e["up"])
        assert q == e["q"] and taps.size == e["taps"] == 2 * e["up"] * q + 1
        np.testing.assert_allclose(np.frombuffer(raw, dtype="<f8"), taps, rtol=0, atol=1e-15)
        assert e["max_error_of_rms"] <= 1e-13
        ip = carrier.interpolator(e["base"], e["up"])
        assert ip.sha256 == e["sha256"] and ip.window == 2 * q + 2
    assert carrier.interpolator(48000, 64).sha256 is None  # the reference rates design on the fly


@pytest.mark.parametrize("base", [44100, 48000])
def test_tables_are_the_carrier_at_the_table_rate(base):
    bins, amps = noise(2, (300.0, 18000.0))
    rate = carrier.table_rate(base)
    tab = carrier.table(bins, amps, rate)
    assert tab.size == 8 * rate
    n = np.random.default_rng(3).integers(0, tab.size, 200)
    assert np.max(np.abs(tab[n] - exact(bins, amps, n, rate))) <= 1e-12 * rms(amps)


@pytest.mark.parametrize("base", [44100, 48000])
def test_interpolation_matches_direct_summation(base):
    ip = carrier.interpolator(base, 16)
    rate = ip.rate * ip.up
    for seed, band in ((4, (20.0, 20000.0)), (5, (6000.0, 16000.0))):
        bins, amps = noise(seed, band)
        tab = carrier.table(bins, amps, ip.rate)
        n = np.sort(np.random.default_rng(seed).integers(0, 8 * rate, 120))
        got = np.array(
            [carrier.block(tab, ip, int(i) // carrier.BLOCK * carrier.BLOCK)[int(i) % carrier.BLOCK] for i in n],
        )
        assert np.max(np.abs(got - exact(bins, amps, n, rate))) <= 1e-13 * rms(amps)


def test_interpolation_is_bit_identical_however_the_work_is_split():
    ip = carrier.interpolator(48000, 16)
    bins, amps = noise(6, (300.0, 18000.0))
    tab = carrier.table(bins, amps, ip.rate)
    n0, n1 = 123_457, 3 * carrier.BLOCK + 54_321

    def gather(a, b):
        out = np.empty(b - a)
        for x, y, v in carrier.span(tab, ip, a, b):
            out[x - a : y - a] = v
        return out

    whole = gather(n0, n1)
    cut = n0 + 2 * carrier.BLOCK - 7
    assert np.array_equal(whole, np.concatenate((gather(n0, cut), gather(cut, n1))))
    assert np.array_equal(whole[:1000], gather(n0, n0 + 1000))  # a shorter render never changes a sample
    starts = list(range(0, n1, carrier.BLOCK))
    with carrier.single_threaded(), ThreadPoolExecutor(4) as pool:
        blocks = list(pool.map(lambda b: carrier.block(tab, ip, b), starts))
    serial = np.concatenate([carrier.block(tab, ip, b) for b in starts])
    assert np.array_equal(np.concatenate(blocks), serial)
    for limit in (1, 4):
        with threadpool_limits(limits=limit, user_api="blas"):
            assert np.array_equal(carrier.block(tab, ip, starts[1]), serial[carrier.BLOCK : 2 * carrier.BLOCK])


def test_the_carrier_is_periodic_with_period_8_s():
    ip = carrier.interpolator(48000, 16)
    bins, amps = noise(7)
    tab = carrier.table(bins, amps, ip.rate)
    period = 8 * ip.rate * ip.up
    assert np.array_equal(carrier.block(tab, ip, 0), carrier.block(tab, ip, period))


@pytest.mark.parametrize("base", [44100, 48000])
def test_values_and_slopes_at_arbitrary_times(base):
    ip = carrier.interpolator(base, 16)
    bins, amps = noise(8, (1000.0, 20000.0))
    tab = carrier.table(bins, amps, ip.rate)
    # Dyadic times are exact in floating point, and f·t = k·m / 2^27 reduces exactly in integers,
    # so the reference has no rounding of its own and the comparison measures the kernel alone.
    m = np.random.default_rng(9).integers(-(2**17), int(8.5 * 2**24), 60)
    t = m / 2**24
    value, slope = carrier.at(tab, ip, t)
    cycles = np.mod(np.outer(m, bins.astype(np.int64)), 2**27) / 2**27
    phase = 2 * np.pi * cycles + np.angle(amps)
    w = 2 * np.pi * bins / spectrum.BINS_PER_HZ
    assert np.max(np.abs(value - np.cos(phase) @ np.abs(amps))) <= 1e-13 * rms(amps)
    # The kernel's derivative weights the top of the band most, so its accuracy is a little lower.
    assert np.max(np.abs(slope + np.sin(phase) @ (np.abs(amps) * w))) <= 1e-12 * rms(amps) * 2 * np.pi * 20000
    # At internal samples the continuous kernel agrees with the matrix product.
    n = np.arange(1000, 1100)
    v, _ = carrier.at(tab, ip, n / (ip.rate * ip.up))
    assert np.max(np.abs(v - carrier.block(tab, ip, 0)[n])) <= 1e-13 * rms(amps)


def test_common_instants_agree_between_the_families():
    bins, amps = noise(10, (300.0, 18000.0))
    out = {}
    for base in (44100, 48000):
        ip = carrier.interpolator(base, 16)
        tab = carrier.table(bins, amps, ip.rate)
        rate = ip.rate * ip.up
        step = rate // 4800  # common instants every 1/4800 s
        out[base] = carrier.block(tab, ip, 0)[::step][:50]
    assert np.max(np.abs(out[44100] - out[48000])) <= 1e-13 * rms(amps)


def test_tables_are_cached_and_never_depend_on_their_companions():
    carrier.clear_cache()
    rate = carrier.table_rate(48000)
    a = noise(11)
    b = noise(12, (6000.0, 16000.0))
    asked = []

    def lines(*which):
        def make(i):
            asked.append(i)
            return which[i]

        return make

    first = carrier.tables([("a",)], rate, lines(a))[0]
    assert asked == [0]
    asked.clear()
    both = carrier.tables([("b",), ("a",)], rate, lines(b, a))
    assert asked == [0]  # only the missing table's lines are asked for
    assert both[1] is first  # the cached array itself
    assert np.array_equal(both[0], carrier.table(*b, rate)) and not both[0].flags.writeable
    assert carrier.cached(("a",), rate) and not carrier.cached(("a",), carrier.table_rate(44100))
    carrier.clear_cache()
    assert np.array_equal(carrier.tables([("a",)], rate, lines(a))[0], first)


def test_the_cache_drops_the_oldest_but_never_what_a_render_needs(monkeypatch):
    carrier.clear_cache()
    rate = carrier.table_rate(48000)
    a = noise(11)
    monkeypatch.setattr(carrier, "CACHE_BYTES", 2 * carrier.table(*a, rate).nbytes)
    for name in ("x", "y", "z"):
        carrier.tables([(name,)], rate, lambda i: a)
    assert not carrier.cached(("x",), rate) and carrier.cached(("y",), rate) and carrier.cached(("z",), rate)
    carrier.tables([("y",)], rate, lambda i: a)  # a use makes it the newest
    carrier.tables([("w",)], rate, lambda i: a)
    assert carrier.cached(("y",), rate) and not carrier.cached(("z",), rate)
    # A render that needs more than the cap keeps all its tables.
    keys = [("p",), ("q",), ("r",)]
    assert len(carrier.tables(keys, rate, lambda i: a)) == 3 and all(carrier.cached(k, rate) for k in keys)
    carrier.clear_cache()
