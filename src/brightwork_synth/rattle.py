# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rattles: a source struck by a stream of collisions, as a shaker's beads or a tambourine's jingles (docs/design.md).

    R(t) = Σ_i a_i β(t − t_i)

β is one collision: it rises in 0.1 ms and falls to −60 dB in `grain_ms`, a
Hermite curve with a knot every 10 dB, so each collision rings as the
source's band or modes would after a tap. The source is multiplied by R, as
it is by its chatter gate, so its band, modes or partials are what rings.

The collisions are `grains_hz` candidates a second, counted from the trigger
and drawn from the seed and the take, each kept with probability e(t_i), the
shape at its time: they crowd at the peak of a shake and thin out as it dies
away, as beads collide less often as the energy they share falls (Cook's
PhISEM). Sources with the same rate therefore share their collisions, as a
tambourine's jingles and head are struck by the same hand.

The candidates fall one in each cell of 1 / `grains_hz`, at a uniform place
in it, as the modes do in theirs: irregular, but never clumped as a Poisson
stream would be, whose clumps would make one shake twice as loud as the next.

Every collision has the same strength, 1 / √m(grains_hz), as in PhISEM, where
each collision adds the same energy and the noise it excites carries the
randomness; here that noise is the source's own carrier. m is the mean square
of R for unit collisions in every cell (`mean_square`), so at the shape's peak
R has a mean square of 1: a rattle is as loud there as the source it gates,
and as the rate grows it tends to that source. Away from the peak the shape
keeps fewer collisions, whose energy falls faster than the shape, as a
shake's does when fewer beads meet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import spectrum
from .curve import Curve, amplitude_curve

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]

ATTACK_S = 1e-4  # how fast a collision rises
RUN = 256  # samples over which a collision's segments are summed before they are evaluated
DECAY_STEPS = 6  # knots every 10 dB, down to −60 dB
_GAUSS = np.polynomial.legendre.leggauss(4)  # exact for the cubic's square, of degree 6


def collision(grain_ms: float) -> Curve:
    """Return β: a rise to 1 in 0.1 ms, then a fall through a knot every 10 dB to −60 dB at `grain_ms`, then 0."""
    decay = grain_ms / 1000
    steps = range(1, DECAY_STEPS + 1)
    times = [0.0, ATTACK_S, *(ATTACK_S + decay * k / DECAY_STEPS for k in steps)]
    values = [0.0, 1.0, *(10.0 ** (-k / 2) for k in steps[:-1]), 0.0]
    return amplitude_curve(times, values)


def _integrals(beta: Curve) -> tuple[float, float]:
    """Return ∫β and ∫β², exactly but for rounding."""
    nodes, weights = _GAUSS
    first = second = 0.0
    for a, h in zip(beta.t[:-1], beta.h, strict=True):
        y = beta(a + h * (nodes + 1) / 2)
        first += float(h / 2 * np.dot(weights, y))
        second += float(h / 2 * np.dot(weights, y * y))
    return first, second


@dataclass(frozen=True)
class Grains:
    """One take's collisions on one source: β, and each collision's onset and strength, onsets ascending."""

    beta: Curve
    onsets: Floats
    amps: Floats

    @property
    def length(self) -> float:
        """How long one collision lasts."""
        return float(self.beta.t[-1])

    def _each(self, t: Floats) -> list[tuple[int, int, int]]:
        """Return (collision, first, last) for every collision that reaches the ascending times t, in order."""
        if not t.size:
            return []
        first = int(np.searchsorted(self.onsets, t[0] - self.length, side="right"))
        last = int(np.searchsorted(self.onsets, t[-1], side="right"))
        out = []
        for i in range(first, last):
            a = int(np.searchsorted(t, self.onsets[i], side="left"))
            b = int(np.searchsorted(t, self.onsets[i] + self.length, side="right"))
            out.append((i, a, b))
        return out

    def value(self, at: ArrayLike) -> Floats:
        """Return R at ascending times, adding the collisions in order, so a sample never depends on the block."""
        t = np.asarray(at, dtype=float)
        out = np.zeros_like(t)
        for i, a, b in self._each(t):
            out[a:b] += self.amps[i] * self.beta.on(t[a:b] - self.onsets[i])
        return out

    def derivative(self, at: ArrayLike) -> Floats:
        """Return R′ at ascending times."""
        t = np.asarray(at, dtype=float)
        out = np.zeros_like(t)
        for i, a, b in self._each(t):
            out[a:b] += self.amps[i] * self.beta.derivative(t[a:b] - self.onsets[i])
        return out

    def on_grid(self, n0: int, n1: int, rate: int) -> Floats:
        """Return R(n / rate) for n0 ≤ n < n1, the same as `value` but for rounding, and much faster.

        R is piecewise cubic. Over each run of RUN samples, aligned to n = 0, that a collision's segment covers
        whole, the segment is re-expanded about the run's start, and the runs' sums of coefficients are
        evaluated once; the samples of a run a segment covers in part take its value directly. Each run's sum
        depends only on the collisions, not on the samples asked for, so a sample never depends on the block.
        """
        out = np.zeros(n1 - n0)
        if n1 <= n0:
            return out
        first = int(np.searchsorted(self.onsets, n0 / rate - self.length, side="right"))
        last = int(np.searchsorted(self.onsets, (n1 - 1) / rate, side="right"))
        if first >= last:
            return out
        beta = self.beta
        h = np.tile(beta.h, last - first)
        start = (self.onsets[first:last, None] + beta.t[None, :-1]).ravel()
        amp = np.repeat(self.amps[first:last], beta.h.size)
        k = np.tile(np.arange(beta.h.size), last - first)
        a1 = beta.h * beta.d[:-1]
        a2 = 3 * (beta.v[1:] - beta.v[:-1]) - 2 * a1 - beta.h * beta.d[1:]
        a3 = 2 * (beta.v[:-1] - beta.v[1:]) + a1 + beta.h * beta.d[1:]
        coef = np.stack((beta.v[:-1], a1, a2, a3), axis=1)[k] * amp[:, None]  # P(s) = Σ c_j s^j, s = (t − start) / h
        ma, mb = _first_at(start, rate), _first_at(start + h, rate)  # the samples each segment holds: [ma, mb)
        qa, qb = -(-ma // RUN), mb // RUN
        whole = qb > qa
        # Runs a segment covers whole: its cubic about the run's start, in samples from there.
        seg = np.repeat(np.flatnonzero(whole), (qb - qa)[whole])
        q = qa[seg] + _counting((qb - qa)[whole])
        s0 = (q * RUN / rate - start[seg]) / h[seg]
        c = coef[seg]
        w = 1.0 / (rate * h[seg])
        b = np.stack(
            (
                ((c[:, 3] * s0 + c[:, 2]) * s0 + c[:, 1]) * s0 + c[:, 0],
                ((3 * c[:, 3] * s0 + 2 * c[:, 2]) * s0 + c[:, 1]) * w,
                (3 * c[:, 3] * s0 + c[:, 2]) * w**2,
                c[:, 3] * w**3,
            ),
            axis=1,
        )
        q_lo, q_hi = n0 // RUN, (n1 - 1) // RUN + 1
        inside = (q >= q_lo) & (q < q_hi)
        sums = np.zeros((q_hi - q_lo, 4))
        for j in range(4):
            sums[:, j] = np.bincount(q[inside] - q_lo, weights=b[inside, j], minlength=q_hi - q_lo)
        j_of = np.arange(RUN, dtype=float)
        runs = ((sums[:, 3:4] * j_of + sums[:, 2:3]) * j_of + sums[:, 1:2]) * j_of + sums[:, 0:1]
        out += runs.ravel()[n0 - q_lo * RUN : n1 - q_lo * RUN]
        # The samples of runs a segment covers in part: its value directly.
        heads = (ma, np.where(whole, qa * RUN, mb))  # before its first whole run, or all of it
        tails = (np.where(whole, qb * RUN, 0), np.where(whole, mb, 0))  # after its last whole run
        for begin, end in (heads, tails):
            a, z = np.maximum(begin, n0), np.minimum(end, n1)
            keep = z > a
            pieces = np.repeat(np.flatnonzero(keep), (z - a)[keep])
            m = a[pieces] + _counting((z - a)[keep])
            sv = (m / rate - start[pieces]) / h[pieces]
            cp = coef[pieces]
            np.add.at(out, m - n0, ((cp[:, 3] * sv + cp[:, 2]) * sv + cp[:, 1]) * sv + cp[:, 0])
        return out

    def knots(self) -> tuple[Floats, Floats, Floats]:
        """Return every collision's knot times, and the jumps of R″ and R‴ there, collision by collision."""
        j2, j3 = self.beta.jumps()
        times = (self.onsets[:, None] + self.beta.t[None, :]).ravel()
        return times, (self.amps[:, None] * j2[None, :]).ravel(), (self.amps[:, None] * j3[None, :]).ravel()


def _first_at(t: Floats, rate: int) -> NDArray[np.int64]:
    """Return the first sample n with n / rate ≥ t, decided in float64."""
    n = np.ceil(t * rate).astype(np.int64)
    n = np.where((n - 1) / rate >= t, n - 1, n)
    return np.where(n / rate < t, n + 1, n)


def _counting(lengths: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return 0 … L − 1 for each length L, one after another."""
    total = int(lengths.sum())
    ends = np.cumsum(lengths)
    return np.arange(total) - np.repeat(ends - lengths, lengths)


def grains(shape: Curve, grains_hz: float, grain_ms: float, seed: int, take: int) -> Grains | None:
    """Draw one take's collisions under a shape; None when the rate is 0, a source that does not rattle."""
    if not grains_hz:
        return None
    cells = math.ceil(grains_hz * float(shape.t[-1]))
    place, keep = spectrum.grain_draws(seed, take, cells)
    times = (np.arange(cells) + place) / grains_hz
    e = shape(times)
    kept = keep < e
    beta = collision(grain_ms)
    strength = 1.0 / math.sqrt(mean_square(beta, grains_hz))
    return Grains(beta, times[kept], np.full(int(np.count_nonzero(kept)), strength))


def mean_square(beta: Curve, rate: float) -> float:
    """Return E[R²] of unit collisions β, one in each cell of 1 / rate at a uniform place in it.

    Pairs of such collisions lie τ apart with density rate² (1 − tri(rate · τ)), tri the unit triangle, against
    rate² for independent ones, so E[R²] = rate ∫β² + rate² ((∫β)² − ∫ (β ⋆ β)(τ) tri(rate · τ) dτ). The last
    integral is summed on a grid of 1 µs.
    """
    first, second = _integrals(beta)
    dt = 1e-6
    b = beta(np.arange(0.0, float(beta.t[-1]) + dt, dt))
    lags = min(b.size, math.ceil(1 / (rate * dt)))
    size = 1 << (2 * b.size - 1).bit_length()
    auto = np.fft.irfft(np.abs(np.fft.rfft(b, size)) ** 2, size)[:lags] * dt  # (β ⋆ β)(k dt), k ≥ 0
    tri = 1.0 - rate * dt * np.arange(lags)
    overlap = float(dt * (2 * np.sum(auto * tri) - auto[0]))  # both signs of τ, the lag 0 once
    return rate * second + rate**2 * (first**2 - overlap)
