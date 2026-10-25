# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tables, the frozen interpolators, carriers at arbitrary times and the table cache (plan §2.3, §3)."""

import hashlib
import json
import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from threadpoolctl import threadpool_info, threadpool_limits

from brightwork_synth import carrier, spectrum

DATA = Path(carrier.__file__).parent / "data"
TAPS_ROUNDING = 1e-15  # how close the shipped taps are to a fresh design
DESIGN_ERROR = 1e-13  # of the carrier's RMS: what each shipped design records, at most
TABLE_ERROR = 1e-12  # of the RMS: a table against direct summation
INTERPOLATION_ERROR = 1e-13  # of the RMS: an interpolated carrier against direct summation
SLOPE_ERROR = 1e-12  # of the RMS slope at 20 kHz: the kernel's derivative


def noise(seed: int = 1, band: tuple[float, float] = (20.0, 20000.0), spacing: float = 0.0) -> carrier.Lines:
    """Return the calibrated lines of a noise source."""
    lines = spectrum.noise_lines(seed, 1, spacing)
    return lines.bins, spectrum.calibrated(lines.bins, lines.amps, band, random=True)


def exact(
    bins: NDArray[np.int64],
    amps: NDArray[np.complex128],
    n: NDArray[np.int64],
    rate: int,
) -> NDArray[np.float64]:
    """Sum the carrier at t = n / rate directly, with the phase reduced exactly in integers."""
    period = spectrum.PERIOD * rate
    out = np.zeros(n.size)
    for i0 in range(0, bins.size, 4096):
        k = bins[i0 : i0 + 4096].astype(np.int64)
        cycles = np.mod(np.outer(n.astype(np.int64), k), period) / period
        a = amps[i0 : i0 + 4096]
        out += np.cos(2 * np.pi * cycles + np.angle(a)) @ np.abs(a)
    return out


def rms(amps: NDArray[np.complex128]) -> float:
    """Return a carrier's RMS from its lines."""
    return math.sqrt(0.5 * float(np.sum(np.abs(amps) ** 2)))


def test_shipped_interpolators_are_the_design_and_match_their_checksums() -> None:
    """Find every shipped interpolator the design, to rounding, and its checksum, its error within 10⁻¹³."""
    entries = json.loads((DATA / "interpolators.json").read_text(encoding="utf-8"))["filters"]
    assert set(entries) == {f"{b}x{k}" for b in (44100, 48000) for k in (16, 32, 64)}
    for e in entries.values():
        raw = (DATA / e["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"]
        taps, q = carrier.design(e["base"], e["up"])
        assert q == e["q"]
        assert taps.size == e["taps"] == 2 * e["up"] * q + 1
        np.testing.assert_allclose(np.frombuffer(raw, dtype="<f8"), taps, rtol=0, atol=TAPS_ROUNDING)
        assert e["max_error_of_rms"] <= DESIGN_ERROR
        ip = carrier.interpolator(e["base"], e["up"])
        assert ip.sha256 == e["sha256"]
        assert ip.window == 2 * q + 2
    assert carrier.interpolator(48000, 64).sha256 is None  # the reference rates design on the fly


@pytest.mark.parametrize("base", [44100, 48000])
def test_tables_are_the_carrier_at_the_table_rate(base: int) -> None:
    """Compute a table, and find the carrier summed directly at 200 instants, within 10⁻¹² of its RMS."""
    bins, amps = noise(2, (300.0, 18000.0))
    rate = carrier.table_rate(base)
    tab = carrier.table(bins, amps, rate)
    assert tab.size == 8 * rate
    n = np.random.default_rng(3).integers(0, tab.size, 200)
    assert np.max(np.abs(tab[n] - exact(bins, amps, n, rate))) <= TABLE_ERROR * rms(amps)


@pytest.mark.parametrize("base", [44100, 48000])
def test_interpolation_matches_direct_summation(base: int) -> None:
    """Interpolate a carrier to the internal rate, and find its direct sum within 10⁻¹³ of its RMS."""
    ip = carrier.interpolator(base, 16)
    rate = ip.rate * ip.up
    for seed, band in ((4, (20.0, 20000.0)), (5, (6000.0, 16000.0))):
        bins, amps = noise(seed, band)
        tab = carrier.table(bins, amps, ip.rate)
        n = np.sort(np.random.default_rng(seed).integers(0, 8 * rate, 120))
        got = np.array(
            [carrier.block(tab, ip, int(i) // carrier.BLOCK * carrier.BLOCK)[int(i) % carrier.BLOCK] for i in n],
        )
        assert np.max(np.abs(got - exact(bins, amps, n, rate))) <= INTERPOLATION_ERROR * rms(amps)


def test_interpolation_is_bit_identical_however_the_work_is_split() -> None:
    """Interpolate in pieces, in threads or with any BLAS threads, and find the same samples bit for bit."""
    ip = carrier.interpolator(48000, 16)
    bins, amps = noise(6, (300.0, 18000.0))
    tab = carrier.table(bins, amps, ip.rate)
    n0, n1 = 123_457, 3 * carrier.BLOCK + 54_321

    def gather(a: int, b: int) -> NDArray[np.float64]:
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


def test_the_carrier_is_periodic_with_period_8_s() -> None:
    """Find the carrier repeating after exactly 8 s."""
    ip = carrier.interpolator(48000, 16)
    bins, amps = noise(7)
    tab = carrier.table(bins, amps, ip.rate)
    period = 8 * ip.rate * ip.up
    assert np.array_equal(carrier.block(tab, ip, 0), carrier.block(tab, ip, period))


@pytest.mark.parametrize("base", [44100, 48000])
def test_values_and_slopes_at_arbitrary_times(base: int) -> None:
    """Evaluate a carrier and its slope at any time by the continuous kernel, to 10⁻¹³ and 10⁻¹² of the RMS."""
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
    assert np.max(np.abs(value - np.cos(phase) @ np.abs(amps))) <= INTERPOLATION_ERROR * rms(amps)
    # The kernel's derivative weights the top of the band most, so its accuracy is a little lower.
    assert np.max(np.abs(slope + np.sin(phase) @ (np.abs(amps) * w))) <= SLOPE_ERROR * rms(amps) * 2 * np.pi * 20000
    # At internal samples the continuous kernel agrees with the matrix product.
    n = np.arange(1000, 1100)
    v, _ = carrier.at(tab, ip, n / (ip.rate * ip.up))
    assert np.max(np.abs(v - carrier.block(tab, ip, 0)[n])) <= INTERPOLATION_ERROR * rms(amps)


def test_common_instants_agree_between_the_families() -> None:
    """Interpolate one carrier in both families, and find them agreeing at their common instants."""
    bins, amps = noise(10, (300.0, 18000.0))
    out = {}
    for base in (44100, 48000):
        ip = carrier.interpolator(base, 16)
        tab = carrier.table(bins, amps, ip.rate)
        rate = ip.rate * ip.up
        step = rate // 4800  # common instants every 1/4800 s
        out[base] = carrier.block(tab, ip, 0)[::step][:50]
    assert np.max(np.abs(out[44100] - out[48000])) <= INTERPOLATION_ERROR * rms(amps)


def test_tables_are_cached_and_never_depend_on_their_companions() -> None:
    """Cache tables, asking only for the missing ones' lines, and compute each the same alone or together."""
    carrier.clear_cache()
    rate = carrier.table_rate(48000)
    a = noise(11)
    b = noise(12, (6000.0, 16000.0))
    asked = []

    def lines(*which: carrier.Lines) -> Callable[[int], carrier.Lines]:
        def make(i: int) -> carrier.Lines:
            asked.append(i)
            return which[i]

        return make

    first = carrier.tables([("a",)], rate, lines(a))[0]
    assert asked == [0]
    asked.clear()
    both = carrier.tables([("b",), ("a",)], rate, lines(b, a))
    assert asked == [0]  # only the missing table's lines are asked for
    assert both[1] is first  # the cached array itself
    assert np.array_equal(both[0], carrier.table(*b, rate))
    assert not both[0].flags.writeable
    assert carrier.cached(("a",), rate)
    assert not carrier.cached(("a",), carrier.table_rate(44100))
    carrier.clear_cache()
    assert np.array_equal(carrier.tables([("a",)], rate, lines(a))[0], first)


def test_the_cache_drops_the_oldest_but_never_what_a_render_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the least recently used tables above the cap, but never one the render asking needs."""
    carrier.clear_cache()
    rate = carrier.table_rate(48000)
    a = noise(11)
    monkeypatch.setattr(carrier, "CACHE_BYTES", 2 * carrier.table(*a, rate).nbytes)
    for name in ("x", "y", "z"):
        carrier.tables([(name,)], rate, lambda _: a)
    assert not carrier.cached(("x",), rate)
    assert carrier.cached(("y",), rate)
    assert carrier.cached(("z",), rate)
    carrier.tables([("y",)], rate, lambda _: a)  # a use makes it the newest
    carrier.tables([("w",)], rate, lambda _: a)
    assert carrier.cached(("y",), rate)
    assert not carrier.cached(("z",), rate)
    # A render that needs more than the cap keeps all its tables.
    keys = [("p",), ("q",), ("r",)]
    assert len(carrier.tables(keys, rate, lambda _: a)) == len(keys)
    assert all(carrier.cached(k, rate) for k in keys)
    carrier.clear_cache()


def test_a_kit_worker_takes_its_share_of_the_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use every core alone, and an even share of them, at least one, while several processes render."""
    cores = 12
    monkeypatch.setattr(carrier.CORES, "share", None)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert carrier.threads() == cores
    with threadpool_limits():  # restores the linear algebra library's threads afterwards
        for processes, share in ((5, 2), (cores, 1), (2 * cores, 1)):
            carrier.share_cores(processes)
            assert carrier.threads() == share
            assert all(pool["num_threads"] == share for pool in threadpool_info())
