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
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.optimize import brentq

from . import carrier, spectrum
from .curve import Curve, amplitude_curve, mapped
from .patch import Noise, Patch, Tone

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

type Floats = NDArray[np.float64]
type Tables = dict[str, list[Floats]]  # each audible source's sub-band tables, by its path
type Pair = tuple[Floats, Floats]  # a quantity and its time derivative
type Band = tuple[Floats, Floats, Floats, Floats]  # a sub-band's e, e′, c and c′
type Span = tuple[int, int]  # internal samples a ≤ n < b
type Row = tuple[str, int, float, float]  # a knot's path, its time's index, Δx″ and Δx‴

TWO_PI = 2 * math.pi


def drive_alpha(db: float) -> float:
    """Return α with α / tanh α = 10^(db/20): the saturator's small-signal gain. 0 dB is bypass."""
    if db == 0.0:
        return 0.0
    g = 10.0 ** (db / 20.0)
    # x / tanh x lies between x and x + 1, so [g − 1, g] brackets the root.
    root: float = brentq(
        lambda a: a / math.tanh(a) - g,
        g - 1.0,
        g,
        xtol=1e-300,
        rtol=4 * np.finfo(float).eps,
        maxiter=500,
    )
    return root


# The voice: the patch reduced to what synthesis needs; times in seconds.


@dataclass(frozen=True)
class SubBand:
    """One sub-band of a source: its table's key, its envelope, and the compression and stretch it plays at."""

    key: carrier.Key  # the table's cache key, without the rate
    env: Curve  # the source's shape on this sub-band's time map
    compression: float
    stretch: float


@dataclass(frozen=True)
class Source:
    """A source as synthesis needs it: its lines' key, gain, shape, chatter, and sub-bands."""

    path: str  # "noise[1]", "tone[2]"
    kind: str  # "noise", "modes" or "tone"
    spec: NoiseSpec | ToneSpec  # what the lines are made of; the tables' common key
    gain: float  # g_j; 0 when silent
    shape: Curve  # e_j, which the chatter gate follows
    chatter: float
    chatter_hz: float
    layout: spectrum.Layout
    subbands: tuple[SubBand, ...]  # empty when silent

    @property
    def start(self) -> float:
        """When the source starts: its shape's first knot."""
        return float(self.shape.t[0])

    @property
    def end(self) -> float:
        """When its last sub-band ends."""
        return max((float(sb.env.t[-1]) for sb in self.subbands), default=self.start)


@dataclass(frozen=True)
class Voice:
    """The patch reduced to what synthesis needs; times in seconds."""

    seed: int
    take: int
    sources: tuple[Source, ...]  # every source, in patch order, silent ones included so paths keep their numbers
    damper: Curve | None
    alpha: float
    t_src: float

    @property
    def audible(self) -> tuple[Source, ...]:
        """The sources that sound: those whose level is finite."""
        return tuple(s for s in self.sources if s.gain)


class Model(NamedTuple):
    """What the internal signal is read from: the voice, the interpolator to the rate it is read at, and the tables."""

    voice: Voice
    ip: carrier.Interpolator
    tables: Tables


def _shape(knots: Sequence[tuple[float, float]]) -> Curve:
    """Make an amplitude curve from a patch's (ms, gain) knots, in seconds."""
    return amplitude_curve([ms / 1000 for ms, _ in knots], [g for _, g in knots])


@dataclass(frozen=True)
class NoiseSpec:
    """Everything a noise source's lines depend on: its cymbal, its strike, its spacing and its band."""

    kind: str  # "noise" or "modes"
    seed: int
    take: int
    spacing_hz: float
    band: tuple[float, float]


@dataclass(frozen=True)
class ToneSpec:
    """Everything a tone's lines depend on; struck tones are the same in every take, so takes share their tables."""

    seed: int  # 0 for a struck tone
    take: int  # 0 for a struck tone
    index: int  # the tone's number, which keys its free phases; 0 for a struck tone
    wave: str
    phase: str
    partials: tuple[tuple[float, float], ...]
    band: tuple[float, float]


def _spec(p: Patch, s: Noise | Tone, index: int) -> NoiseSpec | ToneSpec:
    """Return everything a source's lines depend on, and nothing else."""
    band = (float(s.band_hz[0]), float(s.band_hz[1]))
    if isinstance(s, Noise):
        return NoiseSpec("noise" if s.spacing_hz == 0 else "modes", p.seed, p.take, float(s.spacing_hz), band)
    free = s.phase == "free"
    return ToneSpec(
        p.seed if free else 0,
        p.take if free else 0,
        index if free else 0,
        s.wave,
        s.phase,
        tuple(s.partials),
        band,
    )


def voice(p: Patch) -> Voice:
    """Reduce the patch to what synthesis needs."""
    sources = []
    for path, s in p.sources:
        index = int(path[path.index("[") + 1 : -1])
        kind = "tone" if isinstance(s, Tone) else ("noise" if s.spacing_hz == 0 else "modes")
        spec = _spec(p, s, index)
        shape = _shape(s.amp)
        lay = spectrum.layout(s.f_ref, s.damping, s.bloom)
        gain = 0.0 if s.level_db == -math.inf else 10.0 ** (s.level_db / 20.0)
        subbands: tuple[SubBand, ...] = ()
        if gain:
            subbands = tuple(
                SubBand(
                    (spec, (lay.f_ref, lay.step, lay.count), b),
                    mapped(shape, float(r), float(c)),
                    float(c),
                    float(r),
                )
                for b, (c, r) in enumerate(zip(lay.compression(s.damping), lay.stretch(s.bloom), strict=True))
            )
        sources.append(Source(path, kind, spec, gain, shape, float(s.chatter), float(s.chatter_hz), lay, subbands))
    damper = _shape(p.damper.amp) if p.damper.amp else None
    return Voice(p.seed, p.take, tuple(sources), damper, drive_alpha(p.drive.db), p.length_ms / 1000)


def lines(spec: NoiseSpec | ToneSpec) -> carrier.Lines:
    """Return a source's calibrated lines, coincident ones added: (bins, complex amplitudes)."""
    if isinstance(spec, NoiseSpec):
        found = spectrum.noise_lines(spec.seed, spec.take, spec.spacing_hz)
        return found.bins, spectrum.calibrated(found.bins, found.amps, spec.band, random=True)
    count = len(spec.partials)
    strike = spec.phase == "strike"
    phases = np.zeros(count) if strike else spectrum.free_phases(spec.seed, spec.take, spec.index, count)
    bins, amps, args = spectrum.tone_lines(spec.partials, spec.wave, phases)
    phasors = np.asarray(amps * np.exp(1j * args), dtype=np.complex128)
    return spectrum.merged(bins, spectrum.calibrated(bins, phasors, spec.band, random=False))


def tables(v: Voice, rate: int) -> Tables:
    """Return each audible source's sub-band tables at the table rate, computed together and cached.

    A source's lines are drawn and split only when one of its tables is missing.
    """
    out = {}
    for s in v.audible:
        parts: list[spectrum.SubBand] = []

        def make(i: int, s: Source = s, parts: list[spectrum.SubBand] = parts) -> carrier.Lines:
            if not parts:
                parts.extend(spectrum.split(*lines(s.spec), s.layout))
            return parts[i].bins, parts[i].amps

        out[s.path] = carrier.tables([sb.key for sb in s.subbands], rate, make)
    return out


def sample_count(t_src: float, rate: int) -> int:
    """Count N = ⌊T_src·F⌋ + 1: every internal sample with n / F ≤ T_src, decided in float64."""
    n = math.floor(t_src * rate)
    while (n + 1) / rate <= t_src:
        n += 1
    while n / rate > t_src:
        n -= 1
    return n + 1


def _span(t0: float, t1: float, rate: int, count: int) -> tuple[int, int]:
    """Return the internal samples n with t0 ≤ n / rate ≤ t1, within [0, count)."""
    a = max(0, math.ceil(t0 * rate))
    while a > 0 and (a - 1) / rate >= t0:
        a -= 1
    while a / rate < t0:
        a += 1
    b = min(count, sample_count(t1, rate)) if t1 >= 0 else 0
    return a, max(a, b)


def _gate(s: Source, t: Floats) -> Floats:
    """Return G_j(t) = 1 − d e_j(t) (1 − cos 2π f_c t) / 2."""
    if not s.chatter:
        return np.ones_like(t)
    gate: Floats = 1.0 - s.chatter * s.shape.on(t) * 0.5 * (1.0 - np.cos(TWO_PI * s.chatter_hz * t))
    return gate


def synthesize_x(model: Model, rate: int, count: int, workers: int | None = None) -> Floats:
    """Return x[n] = x(n / rate) for 0 ≤ n < count.

    Sources are added in patch order. Within a source, each fixed block sums its
    sub-bands in order, then takes the gate, the damper and the gain, so a
    sample never depends on how the blocks are shared among threads.
    """
    v, ip, tabs = model
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

            def run(
                block: int,
                s: Source = s,
                lo: int = lo,
                hi: int = hi,
                spans: list[Span] = spans,
            ) -> None:
                b0 = block * carrier.BLOCK
                n0, n1 = max(lo, b0), min(hi, b0 + carrier.BLOCK)
                t = np.arange(n0, n1) / rate
                acc = np.zeros(n1 - n0)
                for sb, tab, (first, last) in zip(s.subbands, tabs[s.path], spans, strict=True):
                    a, b = max(first, n0), min(last, n1)
                    if a < b:
                        c = carrier.block(tab, ip, b0)
                        acc[a - n0 : b - n0] += sb.env.on(t[a - n0 : b - n0]) * c[a - b0 : b - b0]
                factor = s.gain * _gate(s, t)
                if v.damper is not None:
                    factor = factor * v.damper.on(t)
                x[n0:n1] += factor * acc

            list(pool.map(run, range(lo // carrier.BLOCK, (hi - 1) // carrier.BLOCK + 1)))
    return x


def saturate(alpha: float, x: Floats) -> Floats:
    """Return S_α(x) = tanh(αx) / tanh α, odd by construction; the identity when α = 0."""
    if not alpha:
        return x
    norm = np.tanh(np.array([alpha]))[0]  # the same tanh as below, so S(1) = 1 exactly
    out: Floats = np.copysign(np.tanh(alpha * np.abs(x)), x) / norm
    return out


def slope(alpha: float, at: ArrayLike) -> Floats:
    """Return S′_α(x), the saturator's local gain."""
    x = np.asarray(at, dtype=float)
    if not alpha:
        return np.ones_like(x)
    out: Floats = alpha * (1.0 - np.tanh(alpha * x) ** 2) / np.tanh(alpha)
    return out


def curvature(alpha: float, at: ArrayLike) -> Floats:
    """Return S″_α(x) = −2α tanh(αx) S′_α(x)."""
    x = np.asarray(at, dtype=float)
    if not alpha:
        return np.zeros_like(x)
    out: Floats = -2.0 * alpha * np.tanh(alpha * x) * slope(alpha, x)
    return out


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
    reach: Floats  # ascending
    step: Floats  # ascending: the largest step with at most that reach

    def largest(self, m: ArrayLike) -> Floats:
        """Return M(m) for m ≥ 0."""
        i = np.searchsorted(self.reach, np.asarray(m, dtype=float), side="right")
        best = self.step[np.maximum(i - 1, 0)] if self.step.size else np.zeros(np.shape(i))
        out: Floats = np.maximum(self.crossing, np.where(i > 0, best, 0.0))
        return out

    def at(self, alpha: ArrayLike) -> Floats:
        """Return q(α); 0 without drive."""
        a = np.asarray(alpha, dtype=float)
        driven = a > 0
        out: Floats = np.where(driven, a * self.largest(np.where(driven, 1.0 / np.where(driven, a, 1.0), 0.0)), 0.0)
        return out

    def drops(self) -> Floats:
        """Return the α at which q drops: the last α at which each frontier step still qualifies."""
        out: Floats = 1.0 / self.reach[self.reach > 0]
        return out


def _frontier(reach: Floats, step: Floats) -> tuple[Floats, Floats]:
    """Keep the points that no other beats with a smaller or equal reach and a larger or equal step."""
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


def slew_of(x: Floats) -> Slew:
    """Measure the knee frontier, the largest step, the crest and the steepest crossing of x, from silence before it."""
    prev, peak = 0.0, 0.0
    crossing, crest, crest_x, steep = 0.0, 0, 0.0, 0
    reach: list[Floats] = []
    steps: list[Floats] = []
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


def internal(
    v: Voice,
    rate: int,
    count: int,
    ip: carrier.Interpolator,
    tabs: Tables | None = None,
) -> tuple[Floats, Floats, Slew]:
    """Return (x, y, slew): the saturator's input and output at the internal samples 0 ≤ n < count."""
    if tabs is None:
        tabs = tables(v, ip.rate)
    x = synthesize_x(Model(v, ip, tabs), rate, count)
    return x, saturate(v.alpha, x), slew_of(x)


# The kink bound.

ZETA3, ZETA4 = 1.2020569031595942, math.pi**4 / 90
INTERPOLATION_MARGIN = 1.01  # covers the carriers' evaluation at knots (plan §5)


@dataclass(frozen=True)
class Part:
    """One source's parts of x and x′ at some times."""

    gate: Pair  # (G, G′)
    sum: Pair  # (W, W′): its sub-bands' envelopes times their carriers, summed
    bands: list[Band]  # each sub-band's (e, e′, c, c′)


@dataclass(frozen=True)
class State:
    """Everything x and x′ are made of, at some times."""

    x: Floats
    dx: Floats
    damper: Pair  # (D, D′)
    sources: dict[str, Part]  # by the source's path


def state(v: Voice, times: ArrayLike, ip: carrier.Interpolator, tabs: Tables) -> State:
    """Return x, x′ and their parts at arbitrary times, with the carriers from the interpolator's continuous kernel."""
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if v.damper is not None:
        d, dd = v.damper(t), v.damper.derivative(t)
    else:
        d, dd = np.ones_like(t), np.zeros_like(t)
    weights = carrier.weights(ip, t)
    x, dx = np.zeros_like(t), np.zeros_like(t)
    parts = {}
    for s in v.audible:
        w, dw = np.zeros_like(t), np.zeros_like(t)
        bands: list[Band] = []
        for sb, tab in zip(s.subbands, tabs[s.path], strict=True):
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
        parts[s.path] = Part((g, dg), (w, dw), bands)
    return State(x, dx, (d, dd), parts)


@dataclass(frozen=True)
class Kinks:
    """The jumps of x″ and x‴ at every knot, with x and x′ there: all the kink bound needs but α.

    Each row is one term's jump at one knot, named by the knot's path. Jumps at
    the same instant are bounded separately, which the triangle inequality
    allows. Drive acts only through the saturator, so these are computed once
    per render and the bound follows at any drive.
    """

    paths: tuple[str, ...]  # the knot's path, such as "noise[2].amp[3]", per row
    t: Floats
    x: Floats
    dx: Floats
    d2: Floats  # Δx″
    d3: Floats  # Δx‴


def _no_kinks() -> Kinks:
    """Return the kinks of a render with none."""
    empty = np.zeros(0)
    return Kinks((), empty, empty, empty, empty, empty)


def kink_terms(v: Voice, ip: carrier.Interpolator, tabs: Tables | None = None) -> Kinks:
    """Return every term's jumps at every knot (plan §5): sub-band shapes, the chatter gate and the damper."""
    if tabs is None:
        tabs = tables(v, ip.rate)
    times = [sb.env.t for s in v.audible for sb in s.subbands]
    times += [s.shape.t for s in v.audible if s.chatter]
    if v.damper is not None:
        times.append(v.damper.t)
    if not times:
        return _no_kinks()
    grid = np.unique(np.concatenate(times))
    grid = grid[grid <= v.t_src]  # nothing after the sound's end is rendered
    st = state(v, grid, ip, tabs)
    where = {float(u): i for i, u in enumerate(grid)}
    rows: list[Row] = []
    for s in v.audible:
        rows += _source_kinks(s, st, where, grid)
    if v.damper is not None:
        rows += _damper_kinks(v, v.damper, st, where)
    if not rows:
        return _no_kinks()
    i = np.array([row[1] for row in rows])
    return Kinks(
        tuple(row[0] for row in rows),
        grid[i],
        st.x[i],
        st.dx[i],
        np.array([row[2] for row in rows], dtype=float),
        np.array([row[3] for row in rows], dtype=float),
    )


def _rows(where: dict[float, int], paths: list[str], knots: Floats, j2: Floats, j3: Floats) -> list[Row]:
    """Return the rows of the knots that lie on the grid and jump at all."""
    rows = []
    for path, tk, a, b in zip(paths, knots, j2, j3, strict=True):
        i = where.get(float(tk))
        if i is not None and (a or b):
            rows.append((path, i, float(a), float(b)))
    return rows


def _index(where: dict[float, int], knots: Floats) -> NDArray[np.intp]:
    """Return each knot's index on the grid, or 0 for one past the sound's end, which `_rows` leaves out."""
    return np.array([where.get(float(u), 0) for u in knots], dtype=np.intp)


def _source_kinks(s: Source, st: State, where: dict[float, int], grid: Floats) -> list[Row]:
    """Return one source's rows: its sub-band envelopes' jumps, then its chatter gate's."""
    d, dd = st.damper
    part = st.sources[s.path]
    g, dg = part.gate
    names = [f"{s.path}.amp[{k}]" for k in range(1, s.shape.t.size + 1)]
    rows = []
    for sb, (_, _, c, dc) in zip(s.subbands, part.bands, strict=True):
        idx = _index(where, sb.env.t)
        j2, j3 = sb.env.jumps()
        f, df = s.gain * d[idx] * g[idx], s.gain * (dd[idx] * g[idx] + d[idx] * dg[idx])
        cc, dcc = c[idx], dc[idx]
        rows += _rows(where, names, sb.env.t, f * cc * j2, f * cc * j3 + 3 * j2 * (df * cc + f * dcc))
    if s.chatter:
        idx = _index(where, s.shape.t)
        tk = grid[idx]
        kappa = 0.5 * (1.0 - np.cos(TWO_PI * s.chatter_hz * tk))
        dkappa = 0.5 * TWO_PI * s.chatter_hz * np.sin(TWO_PI * s.chatter_hz * tk)
        e2, e3 = s.shape.jumps()
        g2, g3 = -s.chatter * kappa * e2, -s.chatter * (kappa * e3 + 3 * dkappa * e2)
        w, dw = part.sum
        f, df = s.gain * d[idx] * w[idx], s.gain * (dd[idx] * w[idx] + d[idx] * dw[idx])
        rows += _rows(where, names, s.shape.t, f * g2, f * g3 + 3 * g2 * df)
    return rows


def _damper_kinks(v: Voice, damper: Curve, st: State, where: dict[float, int]) -> list[Row]:
    """Return the damper's rows: each knot of it is a kink of every source it multiplies."""
    idx = _index(where, damper.t)
    whole, dwhole = np.zeros(st.x.size), np.zeros(st.x.size)
    for s in v.audible:
        (g, dg), (w, dw) = st.sources[s.path].gate, st.sources[s.path].sum
        whole += s.gain * g * w
        dwhole += s.gain * (dg * w + g * dw)
    j2, j3 = damper.jumps()
    names = [f"damper.amp[{k}]" for k in range(1, damper.t.size + 1)]
    return _rows(where, names, damper.t, j2 * whole[idx], j3 * whole[idx] + 3 * j2 * dwhole[idx])


def kink_weights(rate: int, band: float) -> tuple[float, float]:
    """Return the kink bound's weights of a jump in x″ and in x‴ at rate F, for the band the decimator passes."""
    return (
        INTERPOLATION_MARGIN * 4 * ZETA3 * band / (TWO_PI * rate) ** 3,
        INTERPOLATION_MARGIN * 4 * ZETA4 * band / (TWO_PI * rate) ** 4,
    )


def kink_contributions(k: Kinks, alpha: float, rate: int, band: float) -> Floats:
    """Return each row's share of the kink bound at one drive.

    A jump J in the n-th derivative of y gives a spectrum falling as
    J / (2πf)^(n+1); sampling at F folds its parts near every multiple of F into
    the band the decimator passes, below `band`, so the error is at most
    4 ζ(n+1) · band · |J| / (2πF)^(n+1). Through the saturator y″ jumps by
    S′·Δx″ and y‴ by S′·Δx‴ + 3·S″·x′·Δx″.
    """
    w2, w3 = kink_weights(rate, band)
    g1, g2 = slope(alpha, k.x), curvature(alpha, k.x)
    out: Floats = w2 * np.abs(g1 * k.d2) + w3 * np.abs(g1 * k.d3 + 3 * g2 * k.dx * k.d2)
    return out


def kink_bound(k: Kinks, alpha: float, rate: int, band: float) -> float:
    """Bound the output error from the curves' kinks, at unity gain."""
    return float(np.sum(kink_contributions(k, alpha, rate, band)))


def kink_bounds(k: Kinks, alphas: ArrayLike, rate: int, band: float) -> Floats:
    """Return the kink bound at many drives at once."""
    w2, w3 = kink_weights(rate, band)
    a = np.asarray(alphas, dtype=float)[:, None]
    driven = a > 0
    th = np.tanh(a * k.x)
    g1 = np.where(driven, a * (1.0 - th**2) / np.tanh(np.where(driven, a, 1.0)), 1.0)
    g2 = np.where(driven, -2.0 * a * th * g1, 0.0)
    out: Floats = np.sum(w2 * np.abs(g1 * k.d2) + w3 * np.abs(g1 * k.d3 + 3 * g2 * k.dx * k.d2), axis=1)
    return out


def kink_knots(k: Kinks, alpha: float, rate: int, band: float) -> list[tuple[str, float]]:
    """Return each knot's share of the kink bound, summed over its terms: (path, contribution), largest first."""
    shares: dict[str, float] = {}
    for path, c in zip(k.paths, kink_contributions(k, alpha, rate, band), strict=True):
        shares[path] = shares.get(path, 0.0) + float(c)
    return sorted(shares.items(), key=lambda row: -row[1])
