# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The sources, their sub-bands, chatter, the damper and the drive as pure functions of time (plan §2, §3, §5).

    x(t) = Σ_j g_j D(t) G_j(t) Σ_b e_j(τ_jb(t)) c_jb(t),    y(t) = S_α(x(t))

c_jb is sub-band b of source j's carrier, interpolated from its table; e_j∘τ_jb
is the source's shape on the sub-band's time map, itself a Hermite curve
(`curve.mapped`); G_j = 1 − d_j e_j (1 − cos 2π f_c t)/2 is the chatter gate,
and D the damper. Nothing here keeps state: sample n at rate F is computed
from t_n = n / F alone, so a block boundary, a longer render or another split
of the work never changes a sample's value.

What the validated-domain rule needs at every drive is recorded on the way:
the knee slew's frontier of steps of x, and the jumps of x″ and x‴ at every
knot, with x and x′ there.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import carrier, spectrum
from .curve import Curve, amplitude_curve, mapped
from .patch import Noise, Patch, Tone

TWO_PI = 2 * math.pi


def drive_alpha(db: float) -> float:
    """α with α / tanh α = 10^(db/20): the saturator's small-signal gain. 0 dB is bypass."""
    if db == 0.0:
        return 0.0
    g = 10.0 ** (db / 20.0)
    # x / tanh x lies between x and x + 1, so [g − 1, g] brackets the root.
    return brentq(lambda a: a / math.tanh(a) - g, g - 1.0, g, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)


# The voice: the patch reduced to what synthesis needs; times in seconds.


@dataclass(frozen=True)
class SubBand:
    key: tuple  # the table's cache key, without the rate
    env: Curve  # the source's shape on this sub-band's time map
    compression: float
    stretch: float


@dataclass(frozen=True)
class Source:
    path: str  # "noise[1]", "tone[2]"
    kind: str  # "noise", "modes" or "tone"
    spec: tuple  # what the lines are made of; the tables' common key
    gain: float  # g_j; 0 when silent
    shape: Curve  # e_j, which the chatter gate follows
    chatter: float
    chatter_hz: float
    layout: spectrum.Layout
    subbands: tuple  # (SubBand, ...); empty when silent

    @property
    def start(self) -> float:
        return float(self.shape.t[0])

    @property
    def end(self) -> float:
        return max((float(sb.env.t[-1]) for sb in self.subbands), default=self.start)


@dataclass(frozen=True)
class Voice:
    seed: int
    take: int
    sources: tuple  # every source, in patch order, silent ones included so paths keep their numbers
    damper: Curve | None
    alpha: float
    t_src: float

    @property
    def audible(self) -> tuple:
        return tuple(s for s in self.sources if s.gain)


def _seconds(knots) -> list[float]:
    return [ms / 1000 for ms, _ in knots]


def _shape(knots) -> Curve:
    return amplitude_curve(_seconds(knots), [g for _, g in knots])


def _spec(p: Patch, s, index: int) -> tuple:
    """Everything a source's lines depend on, and nothing else."""
    band = tuple(map(float, s.band_hz))
    if isinstance(s, Noise):
        return ("noise" if s.spacing_hz == 0 else "modes", p.seed, p.take, float(s.spacing_hz), band)
    free = s.phase == "free"
    # Struck tones are the same in every take, so takes share their tables.
    return (
        "tone",
        p.seed if free else 0,
        p.take if free else 0,
        index if free else 0,
        s.wave,
        s.phase,
        tuple(s.partials),
        band,
    )


def voice(p: Patch) -> Voice:
    sources = []
    for path, s in p.sources:
        index = int(path[path.index("[") + 1 : -1])
        kind = "tone" if isinstance(s, Tone) else ("noise" if s.spacing_hz == 0 else "modes")
        spec = _spec(p, s, index)
        shape = _shape(s.amp)
        lay = spectrum.layout(s.f_ref, s.damping, s.bloom)
        gain = 0.0 if s.level_db == -math.inf else 10.0 ** (s.level_db / 20.0)
        subbands = ()
        if gain:
            subbands = tuple(
                SubBand(
                    (spec, (lay.f_ref, lay.step, lay.count), b), mapped(shape, float(r), float(c)), float(c), float(r)
                )
                for b, (c, r) in enumerate(zip(lay.compression(s.damping), lay.stretch(s.bloom)))
            )
        sources.append(Source(path, kind, spec, gain, shape, float(s.chatter), float(s.chatter_hz), lay, subbands))
    damper = _shape(p.damper.amp) if p.damper.amp else None
    return Voice(p.seed, p.take, tuple(sources), damper, drive_alpha(p.drive.db), p.length_ms / 1000)


def lines(spec: tuple) -> tuple[np.ndarray, np.ndarray]:
    """A source's calibrated lines, coincident ones added: (bins, complex amplitudes)."""
    kind = spec[0]
    if kind in ("noise", "modes"):
        _, seed, take, spacing, band = spec
        found = spectrum.noise_lines(seed, take, spacing)
        return found.bins, spectrum.calibrated(found.bins, found.amps, band, random=True)
    _, seed, take, index, wave, phase, partials, band = spec
    phases = np.zeros(len(partials)) if phase == "strike" else spectrum.free_phases(seed, take, index, len(partials))
    bins, amps, args = spectrum.tone_lines(partials, wave, phases)
    calibrated = spectrum.calibrated(bins, amps * np.exp(1j * args), band, random=False)
    return spectrum.merged(bins, calibrated)


def tables(v: Voice, rate: int) -> dict[str, list[np.ndarray]]:
    """Each audible source's sub-band tables at the table rate, computed together and cached.

    A source's lines are drawn and split only when one of its tables is missing.
    """
    out = {}
    for s in v.audible:
        parts = []

        def make(i, s=s, parts=parts):
            if not parts:
                parts.extend(spectrum.split(*lines(s.spec), s.layout))
            return parts[i].bins, parts[i].amps

        out[s.path] = carrier.tables([sb.key for sb in s.subbands], rate, make)
    return out


def sample_count(t_src: float, rate: int) -> int:
    """N = ⌊T_src·F⌋ + 1: every internal sample with n / F ≤ T_src, decided in float64."""
    n = math.floor(t_src * rate)
    while (n + 1) / rate <= t_src:
        n += 1
    while n / rate > t_src:
        n -= 1
    return n + 1


def _span(t0: float, t1: float, rate: int, count: int) -> tuple[int, int]:
    """The internal samples n with t0 ≤ n / rate ≤ t1, within [0, count)."""
    a = max(0, math.ceil(t0 * rate))
    while a > 0 and (a - 1) / rate >= t0:
        a -= 1
    while a / rate < t0:
        a += 1
    b = min(count, sample_count(t1, rate)) if t1 >= 0 else 0
    return a, max(a, b)


def _gate(s: Source, t: np.ndarray) -> np.ndarray:
    """G_j(t) = 1 − d e_j(t) (1 − cos 2π f_c t) / 2."""
    if not s.chatter:
        return np.ones_like(t)
    return 1.0 - s.chatter * s.shape.on(t) * 0.5 * (1.0 - np.cos(TWO_PI * s.chatter_hz * t))


def synthesize_x(v: Voice, rate: int, count: int, ip: carrier.Interpolator, tabs, workers: int | None = None):
    """x[n] = x(n / rate) for 0 ≤ n < count.

    Sources are added in patch order. Within a source, each fixed block sums its
    sub-bands in order, then takes the gate, the damper and the gain, so a
    sample never depends on how the blocks are shared among threads.
    """
    x = np.zeros(count)
    if not count:
        return x
    workers = workers or os.cpu_count() or 1
    with carrier.single_threaded(), ThreadPoolExecutor(workers) as pool:
        for s in v.audible:
            lo, hi = _span(s.start, s.end, rate, count)
            if lo >= hi:
                continue
            spans = [_span(float(sb.env.t[0]), float(sb.env.t[-1]), rate, count) for sb in s.subbands]

            def run(block, s=s, lo=lo, hi=hi, spans=spans):
                b0 = block * carrier.BLOCK
                n0, n1 = max(lo, b0), min(hi, b0 + carrier.BLOCK)
                t = np.arange(n0, n1) / rate
                acc = np.zeros(n1 - n0)
                for sb, tab, (a, b) in zip(s.subbands, tabs[s.path], spans):
                    a, b = max(a, n0), min(b, n1)
                    if a < b:
                        c = carrier.block(tab, ip, b0)
                        acc[a - n0 : b - n0] += sb.env.on(t[a - n0 : b - n0]) * c[a - b0 : b - b0]
                factor = s.gain * _gate(s, t)
                if v.damper is not None:
                    factor = factor * v.damper.on(t)
                x[n0:n1] += factor * acc

            list(pool.map(run, range(lo // carrier.BLOCK, (hi - 1) // carrier.BLOCK + 1)))
    return x


def saturate(alpha: float, x: np.ndarray) -> np.ndarray:
    """S_α(x) = tanh(αx) / tanh α, odd by construction; the identity when α = 0."""
    if not alpha:
        return x
    norm = np.tanh(np.array([alpha]))[0]  # the same tanh as below, so S(1) = 1 exactly
    return np.copysign(np.tanh(alpha * np.abs(x)), x) / norm


def slope(alpha: float, x) -> np.ndarray:
    """S′_α(x), the saturator's local gain."""
    x = np.asarray(x, dtype=float)
    if not alpha:
        return np.ones_like(x)
    return alpha * (1.0 - np.tanh(alpha * x) ** 2) / np.tanh(alpha)


def curvature(alpha: float, x) -> np.ndarray:
    """S″_α(x) = −2α tanh(αx) S′_α(x)."""
    x = np.asarray(x, dtype=float)
    if not alpha:
        return np.zeros_like(x)
    return -2.0 * alpha * np.tanh(alpha * x) * slope(alpha, x)


# The knee slew at every drive.


@dataclass(frozen=True)
class Knees:
    """The knee slew at every drive, recorded in one pass (plan §5).

    Drive leaves the saturator's input x unchanged, so for any α

        q(α) = α · M(1/α),   M(m) = the largest step of x that crosses zero or has an end within |x| ≤ m.

    M is kept as its upper frontier: `crossing`, the largest step that crosses
    zero, which qualifies at every α, and the steps that beat it, each with its
    `reach`, the smaller |x| at its two ends. As α grows a step stops qualifying
    once α·reach > 1 and never qualifies again, so q grows at most in
    proportion to α and drops where α passes 1/reach.
    """

    crossing: float
    reach: np.ndarray  # ascending
    step: np.ndarray  # ascending: the largest step with at most that reach

    def largest(self, m) -> np.ndarray:
        """M(m) for m ≥ 0."""
        i = np.searchsorted(self.reach, np.asarray(m, dtype=float), side="right")
        best = self.step[np.maximum(i - 1, 0)] if self.step.size else np.zeros(np.shape(i))
        return np.maximum(self.crossing, np.where(i > 0, best, 0.0))

    def at(self, alpha) -> np.ndarray:
        """q(α); 0 without drive."""
        a = np.asarray(alpha, dtype=float)
        driven = a > 0
        return np.where(driven, a * self.largest(np.where(driven, 1.0 / np.where(driven, a, 1.0), 0.0)), 0.0)

    def drops(self) -> np.ndarray:
        """The α at which q drops: the last α at which each frontier step still qualifies."""
        return 1.0 / self.reach[self.reach > 0]


def _frontier(reach: np.ndarray, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The points that no other beats with a smaller or equal reach and a larger or equal step."""
    if not reach.size:
        return reach, step
    order = np.lexsort((-step, reach))
    r, s = reach[order], step[order]
    keep = np.empty(s.size, dtype=bool)
    keep[0], keep[1:] = True, s[1:] > np.maximum.accumulate(s)[:-1]
    return r[keep], s[keep]


@dataclass(frozen=True)
class Slew:
    """Steps of the saturator's input x between consecutive internal samples."""

    peak: float  # the largest step anywhere
    knees: Knees = field(compare=False, repr=False)
    crest: int = field(default=0, compare=False)  # the internal sample with the largest |x|
    steep: int = field(default=0, compare=False)  # the end of the largest step that crosses zero

    def knee(self, alpha: float) -> float:
        """q_knee at α: the largest step of αx that crosses zero or touches |αx| ≤ 1."""
        return float(self.knees.at(alpha))


SLEW_BLOCK = 2**18


def slew_of(x: np.ndarray) -> Slew:
    """The knee frontier, the largest step, the crest and the steepest crossing of x, from silence before it."""
    prev, peak = 0.0, 0.0
    crossing, reach, steps, crest, crest_x, steep = 0.0, [], [], 0, 0.0, 0
    for n0 in range(0, x.size, SLEW_BLOCK):
        xb = x[n0 : n0 + SLEW_BLOCK]
        u = np.concatenate(([prev], xb))
        step = np.abs(np.diff(u))
        a, b = u[:-1], u[1:]
        peak = max(peak, float(step.max()))
        crosses = np.signbit(a) != np.signbit(b)
        c = float(np.max(step, where=crosses, initial=0.0))
        if c > crossing:
            crossing, steep = c, n0 + int(np.argmax(np.where(crosses, step, -1.0)))
        beats = ~crosses & (step > crossing)
        if beats.any():
            r, s = _frontier(np.minimum(np.abs(a), np.abs(b))[beats], step[beats])
            reach.append(r)
            steps.append(s)
        i = int(np.argmax(np.abs(xb)))
        if abs(xb[i]) > crest_x:
            crest, crest_x = n0 + i, abs(float(xb[i]))
        prev = xb[-1]
    r = np.concatenate(reach) if reach else np.zeros(0)
    s = np.concatenate(steps) if steps else np.zeros(0)
    keep = s > crossing
    r, s = _frontier(r[keep], s[keep])
    return Slew(peak, Knees(crossing, r, s), crest, steep)


def internal(v: Voice, rate: int, count: int, ip: carrier.Interpolator, tabs=None):
    """(x, y, slew): the saturator's input and output at the internal samples 0 ≤ n < count."""
    if tabs is None:
        tabs = tables(v, ip.rate)
    x = synthesize_x(v, rate, count, ip, tabs)
    return x, saturate(v.alpha, x), slew_of(x)


# The kink bound.

ZETA3, ZETA4 = 1.2020569031595942, math.pi**4 / 90
INTERPOLATION_MARGIN = 1.01  # covers the carriers' evaluation at knots (plan §5)


@dataclass(frozen=True)
class State:
    """Everything x and x′ are made of, at some times."""

    x: np.ndarray
    dx: np.ndarray
    damper: tuple  # (D, D′)
    sources: dict  # path -> dict(gate=(G, G′), sum=(W, W′), bands=[(e, e′, c, c′), ...])


def state(v: Voice, t, ip: carrier.Interpolator, tabs) -> State:
    """x, x′ and their parts at arbitrary times, with the carriers from the interpolator's continuous kernel."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if v.damper is not None:
        d, dd = v.damper(t), v.damper.derivative(t)
    else:
        d, dd = np.ones_like(t), np.zeros_like(t)
    weights = carrier.weights(ip, t)
    x, dx = np.zeros_like(t), np.zeros_like(t)
    parts = {}
    for s in v.audible:
        w, dw = np.zeros_like(t), np.zeros_like(t)
        bands = []
        for sb, tab in zip(s.subbands, tabs[s.path]):
            c, dc = carrier.at_weights(tab, weights)
            e, de = sb.env(t), sb.env.derivative(t)
            w += e * c
            dw += de * c + e * dc
            bands.append((e, de, c, dc))
        if s.chatter:
            kappa = 0.5 * (1.0 - np.cos(TWO_PI * s.chatter_hz * t))
            dkappa = 0.5 * TWO_PI * s.chatter_hz * np.sin(TWO_PI * s.chatter_hz * t)
            ej, dej = s.shape(t), s.shape.derivative(t)
            g, dg = 1.0 - s.chatter * ej * kappa, -s.chatter * (dej * kappa + ej * dkappa)
        else:
            g, dg = np.ones_like(t), np.zeros_like(t)
        x += s.gain * d * g * w
        dx += s.gain * ((dd * g + d * dg) * w + d * g * dw)
        parts[s.path] = {"gate": (g, dg), "sum": (w, dw), "bands": bands}
    return State(x, dx, (d, dd), parts)


@dataclass(frozen=True)
class Kinks:
    """The jumps of x″ and x‴ at every knot, with x and x′ there: all the kink bound needs but α.

    Each row is one term's jump at one knot, named by the knot's path. Jumps at
    the same instant are bounded separately, which the triangle inequality
    allows. Drive acts only through the saturator, so these are computed once
    per render and the bound follows at any drive.
    """

    paths: tuple  # the knot's path, such as "noise[2].amp[3]", per row
    t: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    d2: np.ndarray  # Δx″
    d3: np.ndarray  # Δx‴


def kink_terms(v: Voice, ip: carrier.Interpolator, tabs=None) -> Kinks:
    """Every term's jumps at every knot (plan §5): sub-band shapes, the chatter gate and the damper."""
    if tabs is None:
        tabs = tables(v, ip.rate)
    end = v.t_src
    times = []
    for s in v.audible:
        for sb in s.subbands:
            times.append(sb.env.t)
        if s.chatter:
            times.append(s.shape.t)
    if v.damper is not None:
        times.append(v.damper.t)
    if not times:
        empty = np.zeros(0)
        return Kinks((), empty, empty, empty, empty, empty)
    grid = np.unique(np.concatenate(times))
    grid = grid[grid <= end]  # nothing after the sound's end is rendered
    st = state(v, grid, ip, tabs)
    where = {float(u): i for i, u in enumerate(grid)}
    d, dd = st.damper
    rows = []  # (path, index into grid, Δx″, Δx‴)

    def add(paths, knot_t, j2, j3):
        for path, tk, a, b in zip(paths, knot_t, j2, j3):
            i = where.get(float(tk))
            if i is not None and (a or b):
                rows.append((path, i, a, b))

    for s in v.audible:
        part = st.sources[s.path]
        g, dg = part["gate"]
        names = [f"{s.path}.amp[{k}]" for k in range(1, s.shape.t.size + 1)]
        for sb, (_, _, c, dc) in zip(s.subbands, part["bands"]):
            idx = np.array([where[float(u)] if float(u) in where else 0 for u in sb.env.t])
            j2, j3 = sb.env.jumps()
            f, df = s.gain * d[idx] * g[idx], s.gain * (dd[idx] * g[idx] + d[idx] * dg[idx])
            cc, dcc = c[idx], dc[idx]
            add(names, sb.env.t, f * cc * j2, f * cc * j3 + 3 * j2 * (df * cc + f * dcc))
        if s.chatter:
            idx = np.array([where[float(u)] if float(u) in where else 0 for u in s.shape.t])
            tk = grid[idx]
            kappa = 0.5 * (1.0 - np.cos(TWO_PI * s.chatter_hz * tk))
            dkappa = 0.5 * TWO_PI * s.chatter_hz * np.sin(TWO_PI * s.chatter_hz * tk)
            e2, e3 = s.shape.jumps()
            g2, g3 = -s.chatter * kappa * e2, -s.chatter * (kappa * e3 + 3 * dkappa * e2)
            w, dw = part["sum"]
            f, df = s.gain * d[idx] * w[idx], s.gain * (dd[idx] * w[idx] + d[idx] * dw[idx])
            add(names, s.shape.t, f * g2, f * g3 + 3 * g2 * df)
    if v.damper is not None:
        idx = np.array([where[float(u)] if float(u) in where else 0 for u in v.damper.t])
        whole, dwhole = np.zeros(grid.size), np.zeros(grid.size)
        for s in v.audible:
            (g, dg), (w, dw) = st.sources[s.path]["gate"], st.sources[s.path]["sum"]
            whole += s.gain * g * w
            dwhole += s.gain * (dg * w + g * dw)
        j2, j3 = v.damper.jumps()
        names = [f"damper.amp[{k}]" for k in range(1, v.damper.t.size + 1)]
        add(names, v.damper.t, j2 * whole[idx], j3 * whole[idx] + 3 * j2 * dwhole[idx])
    if not rows:
        empty = np.zeros(0)
        return Kinks((), empty, empty, empty, empty, empty)
    paths, i, a, b = zip(*rows)
    i = np.array(i)
    return Kinks(tuple(paths), grid[i], st.x[i], st.dx[i], np.array(a, dtype=float), np.array(b, dtype=float))


def _weights(rate: int, band: float) -> tuple[float, float]:
    return (
        INTERPOLATION_MARGIN * 4 * ZETA3 * band / (TWO_PI * rate) ** 3,
        INTERPOLATION_MARGIN * 4 * ZETA4 * band / (TWO_PI * rate) ** 4,
    )


def kink_contributions(k: Kinks, alpha: float, rate: int, band: float) -> np.ndarray:
    """Each row's share of the kink bound at one drive.

    A jump J in the n-th derivative of y gives a spectrum falling as
    J / (2πf)^(n+1); sampling at F folds its parts near every multiple of F into
    the band the decimator passes, below `band`, so the error is at most
    4 ζ(n+1) · band · |J| / (2πF)^(n+1). Through the saturator y″ jumps by
    S′·Δx″ and y‴ by S′·Δx‴ + 3·S″·x′·Δx″.
    """
    w2, w3 = _weights(rate, band)
    g1, g2 = slope(alpha, k.x), curvature(alpha, k.x)
    return w2 * np.abs(g1 * k.d2) + w3 * np.abs(g1 * k.d3 + 3 * g2 * k.dx * k.d2)


def kink_bound(k: Kinks, alpha: float, rate: int, band: float) -> float:
    """Upper bound on the output error from the curves' kinks, at unity gain."""
    return float(np.sum(kink_contributions(k, alpha, rate, band)))


def kink_bounds(k: Kinks, alphas, rate: int, band: float) -> np.ndarray:
    """The kink bound at many drives at once."""
    w2, w3 = _weights(rate, band)
    a = np.asarray(alphas, dtype=float)[:, None]
    driven = a > 0
    th = np.tanh(a * k.x)
    g1 = np.where(driven, a * (1.0 - th**2) / np.tanh(np.where(driven, a, 1.0)), 1.0)
    g2 = np.where(driven, -2.0 * a * th * g1, 0.0)
    return np.sum(w2 * np.abs(g1 * k.d2) + w3 * np.abs(g1 * k.d3 + 3 * g2 * k.dx * k.d2), axis=1)


def kink_knots(k: Kinks, alpha: float, rate: int, band: float) -> list[tuple[str, float]]:
    """Each knot's share of the kink bound, summed over its terms: (path, contribution), largest first."""
    shares: dict[str, float] = {}
    for path, c in zip(k.paths, kink_contributions(k, alpha, rate, band)):
        shares[path] = shares.get(path, 0.0) + float(c)
    return sorted(shares.items(), key=lambda row: -row[1])
