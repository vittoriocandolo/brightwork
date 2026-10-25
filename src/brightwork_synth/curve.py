# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monotone cubic Hermite curves and their time maps (plan §2.2).

A curve passes through its knots with slopes from the PCHIP rule:

* interior knots take the weighted harmonic mean of the adjacent secants
  (Fritsch–Butland), or zero where the secants differ in sign or one is zero;
* end knots take the shape-preserving one-sided three-point rule.

End-slope overrides are applied before the curve is evaluated. Every slope
the rule or an override can produce keeps both normalized end slopes of each
segment in [0, 3], inside the Fritsch–Carlson monotonicity region, so each
segment is monotone and stays between its two knot values. Evaluation clips
to that interval, which removes only rounding excursions.

Outside its knots a curve holds its end values. Times are in seconds.

Synthesis evaluates curves on sorted time arrays, one segment at a time
(`on`), so each sample costs a few multiplications and no lookup.

A sub-band plays its source's shape on a piecewise-linear time map, the rise
stretched and the decay compressed (`mapped`), which is itself a curve of this
kind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

type Floats = NDArray[np.float64]
type Number = Floats | float
type Segment = tuple[Number, Number, Number, Number, Number]  # a segment's width, end values and end slopes


def pchip_slopes(times: ArrayLike, values: ArrayLike) -> Floats:
    """Return the knot slopes from the PCHIP rule, before any override."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    h = np.diff(t)
    m = np.diff(v) / h
    if h.size == 1:  # one segment: the secant at both ends
        return np.array([m[0], m[0]])
    ml, mr, hl, hr = m[:-1], m[1:], h[:-1], h[1:]
    agree = (np.sign(ml) == np.sign(mr)) & (ml != 0) & (mr != 0)
    w1, w2 = 2 * hr + hl, hr + 2 * hl
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (w1 + w2) / (w1 / ml + w2 / mr)
    d = np.empty(t.size)
    d[1:-1] = np.where(agree, mean, 0.0)
    d[0] = _end_slope(h[0], h[1], m[0], m[1])
    d[-1] = _end_slope(h[-1], h[-2], m[-1], m[-2])
    return d


def _end_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    """Return an end knot's slope by the shape-preserving three-point rule."""
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return float(d)


class Curve:
    """Cubic Hermite curve through knots (t, v) with knot slopes d."""

    def __init__(self, t: ArrayLike, v: ArrayLike, d: ArrayLike) -> None:
        """Make the curve through knot times t, values v and slopes d, read-only."""
        self.t = np.array(t, dtype=float)
        self.v = np.array(v, dtype=float)
        self.d = np.array(d, dtype=float)
        if not (self.t.ndim == 1 and self.t.size > 1 and self.v.shape == self.t.shape == self.d.shape):
            msg = "a curve needs at least two knots with one value and one slope each"
            raise ValueError(msg)
        self.h = np.diff(self.t)
        if not np.all(self.h > 0):
            msg = "knot times must increase"
            raise ValueError(msg)
        v, d, h = self.v, self.d, self.h
        m = np.diff(v) / h
        self._second_start = (6 * m - 4 * d[:-1] - 2 * d[1:]) / h  # second derivative just after each segment starts
        self._second_end = (-6 * m + 2 * d[:-1] + 4 * d[1:]) / h  # and just before it ends
        self._third = 6 * (d[:-1] + d[1:] - 2 * m) / h**2  # constant on each segment
        for a in (self.t, self.v, self.d, self.h, self._second_start, self._second_end, self._third):
            a.flags.writeable = False

    @classmethod
    def pchip(cls, t: ArrayLike, v: ArrayLike, *, first: float | None = None, last: float | None = None) -> Curve:
        """Make the PCHIP curve through (t, v); `first` and `last` override the end slopes."""
        d = pchip_slopes(t, v)
        if first is not None:
            d[0] = first
        if last is not None:
            d[-1] = last
        return cls(t, v, d)

    # Evaluation at arbitrary times.

    def _segment(self, x: Floats) -> tuple[NDArray[np.intp], Segment]:
        """Return the segment of each time x: its index, and its width, end values and end slopes."""
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, self.t.size - 2)
        return i, (self.h[i], self.v[i], self.v[i + 1], self.d[i], self.d[i + 1])

    def __call__(self, at: ArrayLike) -> Floats:
        """Evaluate the curve at times `at`, holding its end values outside its knots."""
        x = np.asarray(at, dtype=float)
        i, segment = self._segment(x)
        y = _hermite((x - self.t[i]) / self.h[i], segment)
        y = np.where(x < self.t[0], self.v[0], y)
        out: Floats = np.where(x >= self.t[-1], self.v[-1], y)
        return out

    def derivative(self, at: ArrayLike) -> Floats:
        """Return the first derivative at times `at`; 0 outside the knots."""
        x = np.asarray(at, dtype=float)
        i, (h, va, vb, da, db) = self._segment(x)
        s = (x - self.t[i]) / h
        y = 6 * (vb - va) / h * s * (1 - s) + da * (1 - s) * (1 - 3 * s) - db * s * (2 - 3 * s)
        out: Floats = np.where((x < self.t[0]) | (x > self.t[-1]), 0.0, y)
        return out

    def second(self, at: ArrayLike) -> Floats:
        """Return the second derivative at times `at`, one-sided from the right at knots; 0 outside."""
        x = np.asarray(at, dtype=float)
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, self.t.size - 2)
        s = (x - self.t[i]) / self.h[i]
        y = self._second_start[i] * (1 - s) + self._second_end[i] * s
        out: Floats = np.where((x < self.t[0]) | (x >= self.t[-1]), 0.0, y)
        return out

    # Evaluation on sorted times, one segment at a time.

    def on(self, at: ArrayLike) -> Floats:
        """Evaluate the curve at ascending times: the same values as calling it, sliced by segment."""
        x = np.asarray(at, dtype=float)
        y = np.empty_like(x)
        cut = np.searchsorted(x, self.t, side="left")  # x[cut[i]:cut[i + 1]] lies in segment i
        y[: cut[0]] = self.v[0]
        for i in range(self.t.size - 1):
            a, b = cut[i], cut[i + 1]
            if a < b:
                s = (x[a:b] - self.t[i]) / self.h[i]
                y[a:b] = _hermite(s, (self.h[i], self.v[i], self.v[i + 1], self.d[i], self.d[i + 1]))
        y[cut[-1] :] = self.v[-1]
        return y

    # The knots' discontinuities.

    def jumps(self) -> tuple[Floats, Floats]:
        """Return the jumps in the second and third derivatives at each knot; the curve is constant outside.

        The curve is only C¹, so these jumps set how fast its spectrum falls.
        """
        a2 = np.concatenate((self._second_start, [0.0]))
        b2 = np.concatenate(([0.0], self._second_end))
        a3 = np.concatenate((self._third, [0.0]))
        b3 = np.concatenate(([0.0], self._third))
        return a2 - b2, a3 - b3

    def sides(self, k: int) -> tuple[float, float, float, float]:
        """Return (x″ before, x″ after, x‴ before, x‴ after) at knot k; 0 outside the knots."""
        n = self.t.size
        before2 = float(self._second_end[k - 1]) if k > 0 else 0.0
        after2 = float(self._second_start[k]) if k < n - 1 else 0.0
        before3 = float(self._third[k - 1]) if k > 0 else 0.0
        after3 = float(self._third[k]) if k < n - 1 else 0.0
        return before2, after2, before3, after3


def _hermite(s: Floats, segment: Segment) -> Floats:
    """Evaluate a Hermite segment at normalized positions s, clipped to its knot values.

    Horner's form of the cubic: exact at s = 0, and five times faster than the
    basis form, which matters because synthesis spends most of its time here.
    """
    h, va, vb, da, db = segment
    a1 = h * da
    a2 = 3 * (vb - va) - 2 * a1 - h * db
    a3 = 2 * (va - vb) + a1 + h * db
    y = ((a3 * s + a2) * s + a1) * s + va
    out: Floats = np.clip(y, np.minimum(va, vb), np.maximum(va, vb))
    return out


def amplitude_curve(t: ArrayLike, v: ArrayLike) -> Curve:
    """Make an amplitude shape or a damper: zero end slopes, so it joins its constant ends smoothly."""
    return Curve.pchip(t, v, first=0.0, last=0.0)


def peak_index(shape: Curve) -> int:
    """Return the shape's first knot at gain 1, where its decay starts."""
    return int(np.flatnonzero(shape.v == 1.0)[0])


def mapped(shape: Curve, r: float = 1.0, s: float = 1.0) -> Curve:
    """Return the shape with its rise r times slower and its decay s times faster (plan §2.2, §2.6).

    With t₀ the first knot and t_pk the peak, the map is

        τ(t) = t₀ + (t − t₀)/r                 for t ≤ t₀ + r(t_pk − t₀),
        τ(t) = t_pk + s·(t − t₀ − r(t_pk − t₀))  after,

    and the result is e(τ(t)), again a Hermite curve. Knots up to the peak move
    to t₀ + r(t_k − t₀) with slopes divided by r, and knots after it move to
    t₀ + r(t_pk − t₀) + (t_k − t_pk)/s with slopes multiplied by s, so each
    segment is the original segment read at the mapped time: the peak is still
    exactly 1, every segment stays monotone, and the jumps scale by r⁻² and r⁻³
    on the rise and by s² and s³ on the decay. At the peak the stretched rise
    joins the compressed decay, which `sides` reports. The peak's slope is zero,
    so the result is C¹. For r = s = 1 the shape itself is returned.
    """
    if r == 1 and s == 1:
        return shape
    k = peak_index(shape)
    if shape.d[k] != 0:
        msg = "a time map needs a zero slope at the peak"
        raise ValueError(msg)
    t0, tpk = shape.t[0], shape.t[k]
    t, d = shape.t.copy(), shape.d.copy()
    t[: k + 1] = t0 + (shape.t[: k + 1] - t0) * r
    d[: k + 1] /= r
    t[k + 1 :] = t[k] + (shape.t[k + 1 :] - tpk) / s
    d[k + 1 :] *= s
    return Curve(t, shape.v, d)
