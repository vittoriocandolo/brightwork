# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spectral grid, its seeded streams, and every source's lines and sub-bands (plan §2.3–§2.6).

Every source is a set of lines on one grid, f_k = k/8 Hz for k = 1 … 160,000,
so every carrier is periodic with period 8 s and no sound, which ends by 8 s,
ever repeats within itself. A line is a complex amplitude A_k, and the
carrier is Σ_k |A_k| cos(2π f_k t + arg A_k).

* **Noise** (spacing 0): every bin is a line, with a Rayleigh amplitude and a
  uniform phase drawn per take.
* **Modes** (spacing s ≥ 1 Hz): cell i is the run of whole bins in
  (i·s, (i + 1)·s]; it holds one mode near s·(i + u_i), where u_i is drawn
  from the cymbal's seed alone, with an amplitude and phase drawn per take.
* **Tones**: explicit partials, sines or the odd harmonics of a square wave,
  with zero phase at the strike or free phases drawn per take.

A band weights the lines with second-order Butterworth skirts, and γ scales
them so the mean square over takes is ½, that of a unit sine, whatever the
band: a band shapes a source, never attenuates it.

Each source is split into sub-bands whose cos² windows in log frequency sum to
exactly 1; sub-band b is played at the shape's time map with compression s_b
and stretch r_b (§2.6).

Every random value comes from SHAKE-256 over a domain-separated key, and is
indexed by bin, cell or partial, so each stream is platform-independent and
prefix-stable.
"""

from __future__ import annotations

import functools
import hashlib
import math
from dataclasses import dataclass

import numpy as np

BINS_PER_HZ = 8
PERIOD = 8  # seconds: one period of every carrier
BINS = 160_000  # bins 1 … 160,000 cover 1/8 Hz … 20 kHz
TOP_HZ = BINS / BINS_PER_HZ

MODES_KEY = b"cymbal/modes/v1"
STRIKE_KEY = b"cymbal/strike/v1"
NOISE_KEY = b"cymbal/noise/v1"
PHASE_KEY = b"cymbal/phase/v1"
DITHER_KEY = b"cymbal/dither/v1"

MAX_SEED = 2**63 - 1
MAX_TAKE = 2**31
MAX_COMPRESSION = 1000.0  # the largest compression or stretch at 20 kHz (plan §5)


def _check(seed: int, take: int | None = None) -> None:
    if not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be an integer from 0 to 2**63 - 1")
    if take is not None and not 1 <= take <= MAX_TAKE:
        raise ValueError("take must be an integer from 1 to 2**31")


def _stream(key: bytes, *ids: int, words: int) -> np.ndarray:
    """`words` 53-bit unsigned integers from SHAKE-256 over the key and the 8-byte little-endian ids."""
    data = hashlib.shake_256(key + b"".join(i.to_bytes(8, "little") for i in ids)).digest(8 * words)
    return np.frombuffer(data, dtype="<u8") >> np.uint64(11)


def _rayleigh(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box–Muller on pairs of 53-bit integers: Rayleigh amplitudes with mean square 2, uniform phases."""
    pairs = words.reshape(-1, 2)
    u = (pairs[:, 0] + np.uint64(1)).astype(float) * 2.0**-53  # (0, 1]
    v = pairs[:, 1].astype(float) * 2.0**-53  # [0, 1)
    return np.sqrt(-2.0 * np.log(u)), 2.0 * np.pi * v


def _readonly(*arrays):
    for a in arrays:
        a.flags.writeable = False
    return arrays


@functools.lru_cache(maxsize=8)
def noise_excitation(seed: int, take: int) -> tuple[np.ndarray, np.ndarray]:
    """ρ_k and θ_k for bins 1 … 160,000 (read-only)."""
    _check(seed, take)
    return _readonly(*_rayleigh(_stream(NOISE_KEY, seed, take, words=2 * BINS)))


@functools.lru_cache(maxsize=32)
def strike_excitation(seed: int, take: int, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """ρ_i and θ_i for cells 0 … cells − 1 (read-only); a prefix of any longer stream."""
    _check(seed, take)
    return _readonly(*_rayleigh(_stream(STRIKE_KEY, seed, take, words=2 * cells)))


@functools.lru_cache(maxsize=32)
def mode_offsets(seed: int, cells: int) -> np.ndarray:
    """u_i in [0, 1) for cells 0 … cells − 1: where each cell's mode lies (read-only)."""
    _check(seed)
    (u,) = _readonly(_stream(MODES_KEY, seed, words=cells).astype(float) * 2.0**-53)
    return u


def free_phases(seed: int, take: int, tone: int, count: int) -> np.ndarray:
    """Uniform phases for the partials of tone `tone` (numbered from 1)."""
    _check(seed, take)
    return 2.0 * np.pi * _stream(PHASE_KEY, seed, take, tone, words=count).astype(float) * 2.0**-53


def dither(seed: int, take: int, count: int) -> np.ndarray:
    """Triangular (TPDF) dither in LSB for `count` samples, from its own stream.

    Each value is the difference of two 32-bit uniforms on [0, 1), so it lies in
    (−1, 1) with a triangular density: the dither that makes the first two
    moments of the quantization error independent of the signal. The sequence
    is prefix-stable.
    """
    _check(seed, take)
    data = hashlib.shake_256(DITHER_KEY + seed.to_bytes(8, "little") + take.to_bytes(8, "little")).digest(8 * count)
    u = np.frombuffer(data, dtype="<u4").reshape(count, 2).astype(float) * 2.0**-32
    return u[:, 0] - u[:, 1]


def band_power(lo: float, hi: float, f) -> np.ndarray:
    """|H(f)|² of the band with half-power points lo and hi: second-order Butterworth skirts."""
    f = np.asarray(f, dtype=float)
    return 1.0 / ((1.0 + (lo / f) ** 4) * (1.0 + (f / hi) ** 4))


def grid_bin(hz: float) -> int:
    """The bin a partial sounds at: the nearest multiple of 1/8 Hz."""
    return int(round(hz * BINS_PER_HZ))


# Lines.


@dataclass(frozen=True)
class Lines:
    """A source's spectrum before its band and sub-bands: bins, complex amplitudes, and the cymbal's modes."""

    bins: np.ndarray  # ascending, 1 … BINS, unique
    amps: np.ndarray  # complex: |A| e^{iθ}, before γ and the band

    @property
    def hz(self) -> np.ndarray:
        return self.bins / BINS_PER_HZ


def cells_below_top(spacing: float) -> int:
    """How many cells of width `spacing` start below 20 kHz."""
    return int(math.ceil(TOP_HZ / spacing))


def mode_bins(seed: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """(cell, bin) of every mode: the nearest bin to s·(i + u_i), kept within cell i's own bins.

    Cell i holds the bins k with i·s < k/8 ≤ (i + 1)·s, at least 8 of them, so
    no two modes ever share a bin. The ideal position s·(i + u_i) scales exactly
    with the spacing, so `pitch` moves every mode by its ratio, to within a bin
    of the ideal on either side.
    """
    cells = cells_below_top(spacing)
    i = np.arange(cells, dtype=float)
    first = np.floor(BINS_PER_HZ * spacing * i).astype(np.int64) + 1
    last = np.minimum(np.floor(BINS_PER_HZ * spacing * (i + 1)).astype(np.int64), BINS)
    ideal = np.rint(BINS_PER_HZ * spacing * (i + mode_offsets(seed, cells))).astype(np.int64)
    keep = np.flatnonzero((first <= last) & (ideal <= BINS))  # a mode whose place lies above 20 kHz is left out
    return keep, np.clip(ideal, first, last)[keep]


def noise_lines(seed: int, take: int, spacing: float) -> Lines:
    """The lines of a noise source: every bin when the spacing is 0, else one mode per cell."""
    if spacing == 0:
        rho, theta = noise_excitation(seed, take)
        return Lines(np.arange(1, BINS + 1, dtype=np.int64), rho * np.exp(1j * theta))
    cells, bins = mode_bins(seed, spacing)
    rho, theta = strike_excitation(seed, take, cells_below_top(spacing))
    return Lines(bins, rho[cells] * np.exp(1j * theta[cells]))


def tone_lines(partials, wave: str, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lines of a tone, one row per line before coincident lines are added.

    partials: (hz, level_db) pairs; phases: φ_i per partial. A sine partial is
    one line; a square partial is its odd harmonics n = 1, 3, 5, … up to 20 kHz
    with amplitude 1/n. sin(n(ωt + φ)) is the line cos(nωt + nφ − π/2).
    Returns (bins, real amplitudes, phases).
    """
    bins, amps, args = [], [], []
    for (hz, level_db), phi in zip(partials, phases):
        k = grid_bin(hz)
        a = 10.0 ** (level_db / 20.0)
        n = np.array([1]) if wave == "sine" else np.arange(1, BINS // k + 1, 2)
        bins.append(n * k)
        amps.append(a / n)
        args.append(n * phi - np.pi / 2)
    return np.concatenate(bins), np.concatenate(amps), np.concatenate(args)


# Sub-bands (plan §2.6).


@dataclass(frozen=True)
class Layout:
    """A source's sub-bands: centres f_ref·2^(b·w), b = 0 … n, with their compressions and stretches."""

    f_ref: float
    step: float  # w, in octaves; 0 when there is a single sub-band
    count: int  # n + 1

    @property
    def centres(self) -> np.ndarray:
        return self.f_ref * 2.0 ** (np.arange(self.count) * self.step)

    def compression(self, damping: float) -> np.ndarray:
        return 2.0 ** (np.arange(self.count) * self.step * damping)

    def stretch(self, bloom: float) -> np.ndarray:
        return 2.0 ** (np.arange(self.count) * self.step * bloom)

    def windows(self, hz: np.ndarray) -> np.ndarray:
        """Rows of cos² crossfades in log frequency between neighbouring centres; the rows sum to 1."""
        hz = np.asarray(hz, dtype=float)
        w = np.zeros((self.count, hz.size))
        if self.count == 1:
            w[0] = 1.0
            return w
        n = self.count - 1
        x = np.clip(np.log2(hz / self.f_ref) / self.step, 0.0, n)
        i = np.minimum(np.floor(x).astype(np.int64), n - 1)
        frac = x - i
        cols = np.arange(hz.size)
        # At the ends one window is exactly 1 and its neighbour exactly 0.
        w[i, cols] = np.where(frac == 1.0, 0.0, np.cos(0.5 * np.pi * frac) ** 2)
        w[i + 1, cols] = np.where(frac == 0.0, 0.0, np.sin(0.5 * np.pi * frac) ** 2)
        return w


def layout(f_ref: float, damping: float, bloom: float) -> Layout:
    """Neighbouring sub-bands differ by at most √2 in compression and stretch, up to the first centre ≥ 20 kHz."""
    top = max(damping, bloom)
    if top == 0 or f_ref >= TOP_HZ:
        return Layout(f_ref, 0.0, 1)
    step = 1.0 / (2.0 * top)
    return Layout(f_ref, step, 1 + int(math.ceil(math.log2(TOP_HZ / f_ref) / step)))


def worst_compression(f_ref: float, damping: float, bloom: float) -> float:
    """The largest compression or stretch at 20 kHz, which the valid domain bounds."""
    return (max(TOP_HZ / f_ref, 1.0)) ** max(damping, bloom)


@dataclass(frozen=True)
class SubBand:
    """One sub-band's lines, calibrated and weighted: what its table is made of."""

    bins: np.ndarray
    amps: np.ndarray  # complex amplitudes A_k, including γ, the band and the window


def calibrated(bins: np.ndarray, amps: np.ndarray, band: tuple[float, float], random: bool) -> np.ndarray:
    """The lines weighted by the band and scaled by γ so the mean square is ½.

    Random lines (noise and modes) have E[ρ²] = 2, so a line's expected mean
    square is |H|² and γ²Σ|H|² = ½: the mean square over takes is exactly ½.
    A tone's line has mean square |A·H|²/2, so γ²Σ|A·H|² = 1, counting
    coincident lines separately.
    """
    gain = np.sqrt(band_power(band[0], band[1], bins / BINS_PER_HZ))
    if random:
        total = math.fsum((gain**2).tolist())
    else:
        total = math.fsum(((np.abs(amps) * gain) ** 2).tolist()) / 2
    gamma = math.sqrt(0.5 / total) if total > 0 else 0.0
    return gamma * gain * amps


def split(bins: np.ndarray, amps: np.ndarray, lay: Layout) -> list[SubBand]:
    """The lines of each sub-band; a line between two centres is shared by their windows."""
    if lay.count == 1:
        return [SubBand(bins, amps)]
    w = lay.windows(bins / BINS_PER_HZ)
    out = []
    for row in w:
        keep = row > 0
        out.append(SubBand(bins[keep], amps[keep] * row[keep]))
    return out


def merged(bins: np.ndarray, amps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coincident lines added: unique ascending bins with summed complex amplitudes."""
    unique, inverse = np.unique(bins, return_inverse=True)
    total = np.zeros(unique.size, dtype=complex)
    np.add.at(total, inverse, amps)
    return unique, total
