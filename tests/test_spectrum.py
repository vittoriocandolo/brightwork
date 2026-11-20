# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spectral grid, its seeded streams, modes, tones, calibration and sub-bands (plan §2.3–§2.6)."""

import math

import numpy as np
import pytest
from scipy import stats

from brightwork_synth import patch, render, wav
from brightwork_synth import spectrum as sp

from .conftest import make

SIGNIFICANCE = 1e-3  # a statistical test fails below this p-value
TOP_HZ = 20000.0
PERIOD_S = 8
ANALOG_BIN, CYMBAL_BIN = 1642, 6400  # 205.3 Hz and 800 Hz on the grid
CELL_BINS = 8  # bins in a cell, at least
SHOWN_SEEDS = 200  # seeds whose cells are checked bin by bin
WINDOW_SUM = 1e-15  # how close the windows sum to 1
OVERLAPPING = (0.5, 0.95)  # the two bands' cosine similarity: partly coherent
AT_CENTRE = 1e-12  # how close a centre's envelope is to the power law, which it plays exactly


def test_the_grid_reaches_20_khz_in_eighths_of_a_hertz() -> None:
    """Place partials on the grid of 1/8 Hz up to 20 kHz, each within 1/16 Hz of its frequency."""
    assert sp.BINS / sp.BINS_PER_HZ == TOP_HZ
    assert sp.PERIOD * sp.BINS_PER_HZ // sp.BINS_PER_HZ == PERIOD_S
    assert sp.grid_bin(205.3) == ANALOG_BIN
    assert sp.grid_bin(800.0) == CYMBAL_BIN
    for hz in np.random.default_rng(1).uniform(20, 20000, 1000):
        assert abs(sp.grid_bin(hz) / 8 - hz) <= 1 / 16


def test_streams_are_prefix_stable_and_separated() -> None:
    """Draw streams that extend without changing their start, unrelated across keys, seeds and takes."""
    a = sp.strike_excitation(1, 1, 100)
    b = sp.strike_excitation(1, 1, 1000)
    assert np.array_equal(a[0], b[0][:100])
    assert np.array_equal(a[1], b[1][:100])
    assert np.array_equal(sp.mode_offsets(1, 50), sp.mode_offsets(1, 500)[:50])
    assert np.array_equal(sp.dither(3, 2, 64), sp.dither(3, 2, 4096)[:64])
    # Different keys, seeds and takes give unrelated streams.
    assert not np.array_equal(sp.strike_excitation(1, 2, 100)[0], a[0])
    assert not np.array_equal(sp.strike_excitation(2, 1, 100)[0], a[0])
    assert not np.array_equal(sp.noise_excitation(1, 1)[0][:100], a[0])
    assert not np.array_equal(sp.dither(3, 2, 64), sp.dither(3, 3, 64))
    with pytest.raises(ValueError, match="take must be an integer"):
        sp.noise_excitation(1, 0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        sp.mode_offsets(-1, 10)


def test_excitation_is_rayleigh_and_uniform() -> None:
    """Draw Rayleigh amplitudes of mean square 2 and uniform phases, for noise and for modes."""
    rho, theta = sp.noise_excitation(7, 1)
    assert stats.kstest(rho, stats.rayleigh.cdf).pvalue > SIGNIFICANCE
    assert stats.kstest(theta / (2 * np.pi), "uniform").pvalue > SIGNIFICANCE
    assert np.mean(rho**2) == pytest.approx(2.0, rel=0.02)
    rho, theta = sp.strike_excitation(7, 1, 20000)
    assert stats.kstest(rho, stats.rayleigh.cdf).pvalue > SIGNIFICANCE
    assert stats.kstest(theta / (2 * np.pi), "uniform").pvalue > SIGNIFICANCE


def test_one_mode_per_cell_near_a_uniform_place_in_it() -> None:
    """Place one mode in each cell of at least 8 bins, at the bin nearest its place, uniform within the cell."""
    offsets = []
    for seed in range(2000):
        s = 1.0 + (seed % 7) * 13.37
        cells, bins = sp.mode_bins(seed, s)
        i = cells.astype(float)
        first = np.floor(8 * s * i) + 1
        last = np.minimum(np.floor(8 * s * (i + 1)), sp.BINS)
        u = sp.mode_offsets(seed, sp.cells_below_top(s))[cells]
        ideal = 8 * s * (i + u)
        assert np.all((bins >= first) & (bins <= last))  # inside its own cell
        assert np.all(np.abs(bins - ideal) <= 1.0)  # the nearest bin to its place, or the cell's edge
        assert np.all(np.diff(bins) > 0)  # never two modes on one bin
        assert np.all(ideal <= sp.BINS + 0.5)  # none above 20 kHz
        if seed < SHOWN_SEEDS:
            assert np.all(last - first + 1 >= CELL_BINS)
        offsets.append(u[:20])
    # Where in its cell each mode lies is uniform.
    assert stats.kstest(np.concatenate(offsets), "uniform").pvalue > SIGNIFICANCE


def test_takes_keep_the_modes_and_redraw_their_excitation() -> None:
    """Strike a cymbal again: the same modes, with Rayleigh amplitudes and uniform phases drawn anew."""
    a, b = sp.noise_lines(5, 1, 16.0), sp.noise_lines(5, 2, 16.0)
    assert np.array_equal(a.bins, b.bins)
    assert not np.array_equal(np.abs(a.amps), np.abs(b.amps))
    other = sp.noise_lines(6, 1, 16.0)
    assert not np.array_equal(a.bins, other.bins)
    # Across takes, each mode's amplitude is Rayleigh and its phase uniform.
    rows = np.array([sp.noise_lines(5, t, 16.0).amps[:40] for t in range(1, 401)])
    assert stats.kstest(np.abs(rows[:, 7]), stats.rayleigh.cdf).pvalue > SIGNIFICANCE
    assert stats.kstest((np.angle(rows[:, 7]) / (2 * np.pi)) % 1.0, "uniform").pvalue > SIGNIFICANCE


def test_pitch_scales_every_mode_to_within_a_bin() -> None:
    """Scale the spacing, and find every mode within a bin of its scaled place, with its own excitation."""
    for seed in (1, 2, 3):
        for s, ratio in ((16.0, 2.0**0.5), (12.5, 1.5), (30.0, 0.5)):
            u = sp.mode_offsets(seed, sp.cells_below_top(min(s, s * ratio)))
            for spacing in (s, s * ratio):
                cells, bins = sp.mode_bins(seed, spacing)
                ideal = 8 * spacing * (cells + u[cells])
                assert np.max(np.abs(bins - ideal)) <= 1.0
            # The excitation follows the cell, not the frequency.
            a = sp.noise_lines(seed, 1, s)
            b = sp.noise_lines(seed, 1, s * ratio)
            n = min(a.bins.size, b.bins.size)
            assert np.array_equal(a.amps[:n], b.amps[:n])


def test_square_partials_are_their_odd_harmonics_up_to_20_khz() -> None:
    """Build a square from its odd harmonics at 1/n up to 20 kHz, starting from zero; a sine from one line."""
    bins, amps, args = sp.tone_lines([(205.3, 0.0), (540.0, -6.0)], "square", np.array([0.0, 0.0]))
    k1, k2 = sp.grid_bin(205.3), sp.grid_bin(540.0)
    first = bins[: (sp.BINS // k1 + 1) // 2]
    n = np.arange(1, sp.BINS // k1 + 1, 2)
    assert np.array_equal(first, n * k1)
    assert first[-1] <= sp.BINS < first[-1] + 2 * k1
    np.testing.assert_allclose(amps[: n.size], 1 / n, rtol=1e-15)
    rest = bins[n.size :]
    assert np.array_equal(rest, np.arange(1, sp.BINS // k2 + 1, 2) * k2)
    np.testing.assert_allclose(amps[n.size :] * np.arange(1, sp.BINS // k2 + 1, 2), 10 ** (-6 / 20), rtol=1e-15)
    # A struck square starts at zero: every harmonic is sin(nωt).
    np.testing.assert_allclose(args, -np.pi / 2, rtol=0, atol=0)
    bins, amps, args = sp.tone_lines([(1000.0, 0.0)], "sine", np.array([1.0]))
    assert list(bins) == [8000]
    assert list(amps) == [1.0]
    assert args[0] == pytest.approx(1.0 - np.pi / 2)


def test_free_phases_are_uniform_across_takes() -> None:
    """Draw free partials' phases uniform across takes, and different for each tone."""
    phases = np.array([sp.free_phases(3, t, 1, 6) for t in range(1, 2001)])
    assert stats.kstest(phases[:, 2] / (2 * np.pi), "uniform").pvalue > SIGNIFICANCE
    assert not np.array_equal(sp.free_phases(3, 1, 1, 6), sp.free_phases(3, 1, 2, 6))


def test_calibration_sets_the_mean_square_to_one_half() -> None:
    """Calibrate noise to a mean square of ½ over takes, in any band, and a tone to exactly ½."""
    for spacing in (0.0, 1.0, 16.0, 200.0):
        lines = sp.noise_lines(3, 1, spacing)
        for band in ((20.0, 20000.0), (6000.0, 16000.0), (5000.0, 5001.0)):
            a = sp.calibrated(lines.bins, lines.amps, band, random=True)
            gain2 = (np.abs(a) / np.abs(lines.amps)) ** 2  # (γ|H|)²
            assert math.fsum(gain2.tolist()) == pytest.approx(0.5, rel=1e-14)
    # Over takes the mean square of a wide band is ½.
    ms = [
        0.5 * np.sum(np.abs(sp.calibrated(L.bins, L.amps, (2000.0, 9000.0), random=True)) ** 2)
        for L in (sp.noise_lines(4, t, 0.0) for t in range(1, 201))
    ]
    assert np.mean(ms) == pytest.approx(0.5, rel=0.005)
    # A tone's lines carry a mean square of exactly ½, whatever its band.
    bins, amps, args = sp.tone_lines([(205.3, 0.0), (304.4, -3.0)], "square", np.zeros(2))
    for band in ((20.0, 20000.0), (7000.0, 16000.0)):
        a = sp.calibrated(bins, np.asarray(amps * np.exp(1j * args), dtype=np.complex128), band, random=False)
        assert 0.5 * math.fsum((np.abs(a) ** 2).tolist()) == pytest.approx(0.5, rel=1e-14)


def test_editing_a_band_reweights_the_same_lines() -> None:
    """Change a band, and find the same lines with the same phases, reweighted by a real gain."""
    for spacing in (0.0, 14.0):
        lines = sp.noise_lines(2, 3, spacing)
        f = lines.bins / sp.BINS_PER_HZ
        a = sp.calibrated(lines.bins, lines.amps, (6000.0, 16000.0), random=True)
        b = sp.calibrated(lines.bins, lines.amps, (5000.0, 12000.0), random=True)
        weight = np.sqrt(sp.band_power(5000.0, 12000.0, f) / sp.band_power(6000.0, 16000.0, f))
        ratio = (b / a) / weight  # the same lines, phases and draws: only a real weight and one constant differ
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
        assert abs(ratio[0].imag) <= 1e-12 * abs(ratio[0])  # real, to the rounding of a complex division
        assert ratio[0].real > 0


def test_sources_correlate_as_their_bands_overlap() -> None:
    """Correlate two bands of one noise as their skirts overlap, and noise with modes not at all."""
    # Plan §2.4: noise sources share one noise, so two bands of it are coherent where their skirts overlap, and
    # noise and modes are independent draws. Over the period, the correlation of two carriers is Re Σ a·conj(b)
    # over their mean squares of ½; its mean over takes is the bands' cosine similarity, since E[ρ²] = 2.
    a_band, b_band = (2000.0, 4000.0), (3000.0, 9000.0)
    coherent, independent, spread = [], [], []
    n = sp.noise_lines(5, 1, 0.0)
    a = sp.calibrated(n.bins, n.amps, a_band, random=True)
    for take in range(1, 201):
        n = sp.noise_lines(5, take, 0.0)
        a = sp.calibrated(n.bins, n.amps, a_band, random=True)
        b = sp.calibrated(n.bins, n.amps, b_band, random=True)
        coherent.append(float(np.real(np.vdot(b, a))))
        m = sp.noise_lines(5, take, 14.0)
        c = np.zeros(sp.BINS + 1, dtype=complex)
        c[m.bins] = sp.calibrated(m.bins, m.amps, a_band, random=True)
        independent.append(float(np.real(np.vdot(c[n.bins], a))))
        spread.append(float(np.sqrt(0.5 * np.sum(np.abs(a * c[n.bins]) ** 2))))
    f = n.bins / sp.BINS_PER_HZ
    ha, hb = np.sqrt(sp.band_power(*a_band, f)), np.sqrt(sp.band_power(*b_band, f))
    expected = np.sum(ha * hb) / np.sqrt(np.sum(ha**2) * np.sum(hb**2))
    w = ha * hb / np.sqrt(4 * np.sum(ha**2) * np.sum(hb**2))  # each line's weight on ρ², whose variance is 4
    sigma = np.sqrt(4 * np.sum(w**2) / 200)
    assert OVERLAPPING[0] < expected < OVERLAPPING[1]
    assert abs(np.mean(coherent) - expected) < 4 * sigma
    assert abs(np.mean(independent)) < 4 * np.mean(spread) / np.sqrt(200)
    # One band is one noise: the same lines, so the same carrier.
    again = sp.calibrated(n.bins, n.amps, a_band, random=True)
    assert np.array_equal(again, a)


def test_layout_steps_and_windows() -> None:
    """Space sub-bands √2 apart in compression and stretch, up to 20 kHz, under windows that sum to 1."""
    for f_ref, damping, bloom in ((300.0, 1.0, 0.0), (6000.0, 0.5, 0.0), (20.0, 0.67, 0.2), (300.0, 0.3, 1.2)):
        lay = sp.layout(f_ref, damping, bloom)
        c = lay.centres
        assert c[0] == f_ref
        assert c[-1] >= sp.TOP_HZ
        assert lay.count == 1 or c[-2] < sp.TOP_HZ
        assert lay.count == 1 + math.ceil(2 * max(damping, bloom) * math.log2(20000 / f_ref) - 1e-12)
        top = max(damping, bloom)
        for row in (lay.compression(damping), lay.stretch(bloom)):
            assert row[0] == 1.0
            assert np.all(np.diff(np.log2(row)) <= 0.5 * (1 + 1e-12))
        np.testing.assert_allclose(lay.compression(damping), (c / f_ref) ** damping, rtol=1e-13)
        np.testing.assert_allclose(lay.stretch(bloom), (c / f_ref) ** bloom, rtol=1e-13)
        hz = np.geomspace(1 / 8, 20000.0, 50001)
        w = lay.windows(hz)
        assert np.max(np.abs(w.sum(axis=0) - 1.0)) <= WINDOW_SUM
        assert np.all(w >= 0)
        assert np.all(w[0, hz <= f_ref] == 1.0)
        assert np.all(w[1:, hz <= f_ref] == 0.0)
        assert top > 0
        assert lay.step == pytest.approx(1 / (2 * top))
    assert sp.layout(300.0, 0.0, 0.0).count == 1
    assert sp.layout(300.0, 5e-324, 0.0) == sp.layout(300.0, 0.0, 0.0)  # its next centre lies past every float
    assert sp.worst_compression(300.0, 1.0, 0.0) == pytest.approx(20000 / 300)
    assert sp.worst_compression(20000.0, 3.0, 3.0) == 1.0


def test_the_mix_between_centres_stays_near_the_power_law() -> None:
    """Mix the envelopes of neighbouring centres, and stay within plan §2.6's bounds of the power law."""
    # An exponential decay at f should be s(f) = (f/f_ref)^damping times faster; a line between two centres takes
    # the mix of their envelopes instead. Plan §2.6: when the line should be 20, 40 or 60 dB down, the mix is at
    # most 0.9, 2.4 or 4.5 dB louder and 0.6, 0.8 or 0.8 dB quieter, at any damping.
    bounds = {20: (0.888, -0.605), 40: (2.382, -0.807), 60: (4.466, -0.781)}
    for f_ref, damping in ((300.0, 1.0), (6000.0, 0.5), (40.0, 3.0)):
        lay = sp.layout(f_ref, damping, 0.0)
        hz = np.geomspace(f_ref, lay.centres[-2], 100_001)
        w, s = lay.windows(hz), lay.compression(damping)
        for down, (louder, quieter) in bounds.items():
            u = down / 20 * math.log(10) / (hz / f_ref) ** damping  # the shape's time to `down`, in decay times
            excess = 20 * np.log10(np.sum(w * np.exp(-np.outer(s, u)), axis=0)) + down
            assert louder - 0.002 < excess.max() <= louder + 0.0005
            assert quieter - 0.0005 <= excess.min() < quieter + 0.002
            at = np.isin(hz, lay.centres)
            assert np.all(np.abs(excess[at]) < AT_CENTRE)


def test_sub_bands_sum_to_the_source() -> None:
    """Split a source into sub-bands that sum to it, or keep it whole when it has one."""
    lines = sp.noise_lines(9, 1, 0.0)
    a = sp.calibrated(lines.bins, lines.amps, (300.0, 18000.0), random=True)
    parts = sp.split(lines.bins, a, sp.layout(300.0, 1.0, 0.0))
    total = np.zeros(sp.BINS + 1, dtype=complex)
    for p in parts:
        total[p.bins] += p.amps
    np.testing.assert_allclose(total[lines.bins], a, rtol=1e-15, atol=1e-15 * np.max(np.abs(a)))
    single = sp.split(lines.bins, a, sp.layout(300.0, 0.0, 0.0))
    assert len(single) == 1
    assert single[0].amps is a


def test_coincident_lines_add() -> None:
    """Add lines that share a bin."""
    bins, amps = sp.merged(np.array([5, 3, 5]), np.array([1 + 0j, 2j, 1j]))
    assert list(bins) == [3, 5]
    assert list(amps) == [2j, 1 + 1j]


def test_dither_is_triangular_and_never_reaches_one_lsb() -> None:
    """Draw triangular dither strictly within ±1 LSB."""
    d = sp.dither(11, 1, 200_000)
    assert np.all(np.abs(d) < 1)
    assert stats.kstest(d, stats.triang(c=0.5, loc=-1, scale=2).cdf).pvalue > SIGNIFICANCE


def test_the_names_that_seed_and_mark_a_render_never_change() -> None:
    """Pin the keys of every seeded stream, the patch's instrument and the WAV chunk: a new behavior takes a new key."""
    keys = (sp.MODES_KEY, sp.STRIKE_KEY, sp.NOISE_KEY, sp.PHASE_KEY, sp.GRAINS_KEY, sp.DITHER_KEY)
    assert keys == (
        b"cymbal/modes/v1",
        b"cymbal/strike/v1",
        b"cymbal/noise/v1",
        b"cymbal/phase/v1",
        b"cymbal/grains/v1",
        b"cymbal/dither/v1",
    )
    assert patch.INSTRUMENT == "cymbal"
    assert wav.PRIVATE == b"cymb"


@pytest.mark.parametrize("field", ["damping", "bloom"])
def test_a_subnormal_damping_or_bloom_sounds_as_none(field: str) -> None:
    """Render the smallest float of damping or bloom exactly as none, where its sub-bands' step would overflow."""
    edits = {f"noise[{i}].{f}": 0.0 for i in range(2) for f in ("damping", "bloom")}
    none = render.render(make(**edits))[0]
    tiny = render.render(make(**{**edits, f"noise[0].{field}": 5e-324}))[0]
    np.testing.assert_array_equal(tiny, none)
