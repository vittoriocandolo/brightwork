# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Storage: the sources stored at a low rate and bit depth, as in a drum machine's sample memory, then played back.

A drum machine of the early eighties stored its cymbals as samples: a few bits, at a few tens of kilohertz, read
out through a DAC that holds each value until the next, then filtered. Brightwork models that storage exactly:

* **Stored.** The sum of the sources, gated but not yet damped, is taken at every D-th internal sample, where D is
  the whole divider that makes the storage rate F / D the nearest to the one asked for. Nothing filters it
  first, so what lies above half that rate folds down, as it did.
* **Quantized.** Each stored value is rounded to `bits` bits against the stored sound's own peak, without dither:
  the noise this leaves sits a fixed distance under the peak, so a decaying cymbal turns grainy as it fades.
* **Played back.** Each value is held for one storage period and smoothed by a Gaussian low-pass whose −3 dB point
  is `filter_hz`, delayed by 8σ so that, as any real filter, it answers only after its input. Hold and Gaussian
  together are a closed form, g(t) = ½ [erf((t − L) / √2σ) − erf((t − L − T) / √2σ)] with L = 8σ, smooth,
  0 within 10⁻¹⁵ before 0 and after T + 2L, and past a few times the filter's frequency nil: the playback adds no
  kinks and nothing for the internal rate to alias, and the sound lasts T + 2L longer than its sources.

The damper multiplies the playback, as a drum machine's VCA shaped its stored sample, and the drive comes after.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import upfirdn
from scipy.special import erf

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .patch import Storage

    type Floats = NDArray[np.float64]

REACH = 8.0  # the Gaussian is kept to this many σ either side, where erf is 1 within 10⁻¹⁵
MAX_PERIOD_ROUNDING_S = 1e-6  # how much longer than 1 / rate_hz a storage period can be, its divider rounded


def divider(rate: int, rate_hz: float) -> int:
    """Return the whole number D that makes F / D the nearest storage rate to `rate_hz` at internal rate F.

    The nearer of the two whole dividers around F / rate_hz, judged by the rate
    each gives, not by the period: rounding the period would pick the farther
    rate between √(D(D + 1)) and D + ½.
    """
    ratio = rate / rate_hz
    candidates = {max(2, math.floor(ratio)), max(2, math.ceil(ratio))}
    return min(sorted(candidates), key=lambda d: abs(rate / d - rate_hz))


def tail_s(settings: Storage) -> float:
    """Return how much longer than its sources a stored sound lasts, at most: one held period and the filter's 2L."""
    return 1 / settings.rate_hz + MAX_PERIOD_ROUNDING_S + 2 * REACH * sigma(settings.filter_hz)


def sigma(filter_hz: float) -> float:
    """Return the Gaussian's σ in seconds: its response exp(−(2πfσ)² / 2) is 3 dB down at `filter_hz`."""
    return math.sqrt(math.log(2)) / (2 * math.pi * filter_hz)


@dataclass(frozen=True)
class Stored:
    """A sound as stored: its values after quantization, the storage rate and divider, and the playback filter's σ."""

    values: Floats  # at m / rate for m = 0, 1, …
    internal: int  # F, the internal rate it was stored from
    divider: int  # D: the storage rate is F / D
    sigma: float
    bits: int

    @property
    def rate(self) -> float:
        """The storage rate."""
        return self.internal / self.divider

    def kernel(self, t: ArrayLike) -> Floats:
        """Return g(t): one value held for a storage period, smoothed by the Gaussian."""
        x = np.asarray(t, dtype=float) - REACH * self.sigma
        s = math.sqrt(2) * self.sigma
        out: Floats = 0.5 * (erf(x / s) - erf((x - 1 / self.rate) / s))
        return out

    def slope(self, t: ArrayLike) -> Floats:
        """Return g′(t): the Gaussian at the hold's start, less the Gaussian at its end."""
        x = np.asarray(t, dtype=float) - REACH * self.sigma
        norm = 1 / (self.sigma * math.sqrt(2 * math.pi))
        start, end = x / self.sigma, (x - 1 / self.rate) / self.sigma
        out: Floats = norm * (np.exp(-0.5 * start**2) - np.exp(-0.5 * end**2))
        return out

    def _near(self, t: Floats) -> tuple[int, int]:
        """Return the stored values that reach any of the times t: [first, last)."""
        span = 2 * REACH * self.sigma + 1 / self.rate  # g is 0 outside [0, span]
        first = max(0, math.floor((float(np.min(t)) - span) * self.rate))
        last = min(self.values.size, math.ceil(float(np.max(t)) * self.rate) + 1)
        return first, last

    def at(self, t: ArrayLike) -> tuple[Floats, Floats]:
        """Return the playback and its derivative at arbitrary times, each summed over the values that reach it."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        y, dy = np.zeros_like(times), np.zeros_like(times)
        for i, ti in enumerate(times):  # a few knots, each reached by the values of one filter span, not all between
            first, last = self._near(np.array([ti]))
            m = np.arange(first, last)  # empty past the last value's reach, where the sums are 0
            y[i] = float(np.dot(self.values[m], self.kernel(ti - m / self.rate)))
            dy[i] = float(np.dot(self.values[m], self.slope(ti - m / self.rate)))
        return y, dy

    def play(self, rate: int, n0: int, n1: int) -> Floats:
        """Return the playback at n / rate for n0 ≤ n < n1, where `rate` is a whole multiple of the storage rate.

        g is sampled once on the grid of `rate`, and the stored values are held through it by one polyphase pass:
        y(n / rate) = Σ_m v[m] g((n − m·U) / rate), U = rate / storage rate.
        """
        up, rest = divmod(rate * self.divider, self.internal)
        if rest:
            msg = f"playback at {rate} Hz needs a whole multiple of the storage rate, {self.rate:g} Hz"
            raise ValueError(msg)
        tail = math.ceil((2 * REACH * self.sigma + 1 / self.rate) * rate)  # samples of g, which starts at t = 0
        g = self.kernel(np.arange(tail + 1) / rate)
        out = np.zeros(n1 - n0)
        if n1 <= n0 or not self.values.size:
            return out
        # The values that reach [n0, n1), and where their playback starts.
        first = max(0, (n0 - tail) // up)
        last = min(self.values.size, (n1 - 1) // up + 1)
        if first >= last:
            return out
        y = upfirdn(g, self.values[first:last], up=up)  # y[k] is the playback at (first·U + k) / rate
        start = first * up
        a, b = max(n0, start), min(n1, start + y.size)
        if b > a:
            out[a - n0 : b - n0] = y[a - start : b - start]
        return out


def store(x: Floats, rate: int, settings: Storage) -> Stored:
    """Store the internal signal x, sampled at `rate`: every D-th sample, quantized to the settings' bits."""
    d = divider(rate, settings.rate_hz)
    values = x[::d]
    peak = float(np.max(np.abs(values), initial=0.0))
    steps = 2 ** (settings.bits - 1) - 1  # codes −steps … steps, 0 among them
    if peak > 0:
        values = np.round(values / peak * steps) / steps * peak
    return Stored(values, rate, d, sigma(settings.filter_hz), settings.bits)
