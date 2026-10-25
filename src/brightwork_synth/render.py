# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering policy, the validated domain, takes, stereo pairs and the output stage (plan §2.10, §3, §5, §7).

    out[m] = p · g · Σ_{k<L} h[k] y[mK − k],    y[n] = y(n / F) for 0 ≤ n < N, else 0

The internal rate is F = K × base, with base 44.1 or 48 kHz. The 88.2 and
96 kHz exports reuse their family's F and filter with factor K/2, so a 96 kHz
file is exactly the 48 kHz file with the in-between samples filled in.
Nothing trims, fades, high-passes or removes DC, and the whole filter tail
is kept.

Each take is judged on its own: the validated-domain status, what the
predicted aliasing comes from and how much drive would change the status,
all from the take's own render. A set of takes shares one gain, and a stereo
file is two consecutive takes, one per channel.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any, override

import numpy as np
import scipy
from scipy.signal import oaconvolve, upfirdn

from . import __version__, carrier, spectrum, wav
from .loudness import momentary_max
from .patch import Noise, Patch, Tone, describe_pitch, dumps
from .spectrum import MAX_TAKE
from .voice import (
    Kinks,
    Slew,
    drive_alpha,
    internal,
    kink_bound,
    kink_bounds,
    kink_knots,
    kink_terms,
    sample_count,
    tables,
    voice,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from .patch import Output

type Floats = NDArray[np.float64]

K = 32  # frozen by the adversarial grid (plan §3, docs/VALIDATION.md)
PEAK_BLOCK = 1 << 20  # internal samples of the reconstruction measured at once
FAMILY = {44100: 44100, 88200: 44100, 48000: 48000, 96000: 48000}

# The validated domain (plan §5). The predicted aliasing error at unity gain is
#     SATURATION · exp(−π² / q) · P  +  the kink bound
# where q is the knee slew and P the output peak. A render is inside when that
# is at most LIMIT · P, or FLOOR when it is near-silent. SATURATION is the
# smallest value that covers every converged fixture of the adversarial grid at
# K = 32 with 6 dB of margin, 0.356, rounded up; the same form would need 0.85
# at 16×, so it belongs to this K.
SATURATION = 0.36
LIMIT = 1e-5  # −100 dB of the peak
FLOOR = 1e-7  # the absolute bound for near-silent renders
QUIET = 1e-2  # below this peak the absolute bound applies

DRIVE_MAX = 48.0  # the valid domain's largest drive
MAX_RENDERS = 128  # takes in one set; a stereo file uses two

# Export formats: float32, or PCM with seeded TPDF dither (the value is the bit depth).
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}
SHOWN_PARTIALS = 6  # the report lists a tone's first partials, then an ellipsis


class RenderError(RuntimeError):
    """A render that cannot be made or written, with what to change."""


def ceiling(fmt: str) -> float:
    """Return the largest sample a format takes: full scale for float, 2 LSB below it for PCM, so dither never clips."""
    bits = FORMATS[fmt]
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


def db(x: float) -> float:
    """Return x in decibels, -inf for 0."""
    return 20 * math.log10(x) if x > 0 else -math.inf


# Frozen decimators.


@dataclass(frozen=True)
class Decimator:
    """A frozen decimator: its family, its factor K, its taps, their checksum and its measured stopband."""

    base: int
    k: int
    taps: Floats
    sha256: str
    stopband_db: float


def manifest() -> dict[str, Any]:
    """Read the frozen decimators' manifest: each one's file, taps, checksum and stopband."""
    text = resources.files("brightwork_synth").joinpath("data", "decimators.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


@functools.cache
def decimator(base: int, k: int = K) -> Decimator:
    """Return a frozen decimator from brightwork_synth/data, checked against its checksum."""
    entry = manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("brightwork_synth").joinpath("data", entry["file"]).read_bytes()
    taps = carrier.load_taps(raw, entry, f"decimator {base}x{k}")
    return Decimator(base, k, taps, entry["sha256"], entry["stopband_db"])


# The decimator as matrix products.

ROW = 256  # input samples per row of a block's matrix
OUTPUTS = 2**13  # outputs per block


def polyphase(taps: Floats, stride: int, row: int) -> tuple[Floats, int]:
    """Return (G, J), with G[r·(J + 1) + j, t] = h[j·row + r·stride − t], and 0 where that is outside the filter."""
    phases, rows = row // stride, (taps.size + row - 2) // row
    k = np.arange(rows + 1)[:, None] * row + (np.arange(phases) * stride)[:, None, None] - np.arange(row)
    g = np.where((k >= 0) & (k < taps.size), taps[np.clip(k, 0, taps.size - 1)], 0.0)
    return g.reshape(phases * (rows + 1), row), rows


def decimate(
    taps: Floats,
    y: Floats,
    factor: int,
    stride: int | None = None,
    workers: int | None = None,
) -> Floats:
    """Return out[m] = Σ_k h[k] y[m·factor − k], y zero outside the render, each output always computed alike.

    The outputs form stride / factor interleaved streams, each a decimation by
    `stride` of the input advanced by a multiple of the factor. A 2× export
    passes its family's factor as the stride, so its even outputs are the base
    rate's own stream and a 96 kHz file is the 48 kHz file with samples filled
    in, bit for bit.

    A stream runs in blocks of OUTPUTS outputs aligned to m = 0. A block's
    inputs are laid out in rows of ROW samples, and one matrix product with G
    gives, for every row and every output phase, the filter's partial sum over
    that row; each output then adds its J + 1 rows' partial sums in a fixed
    order. Every block is computed at full size and each product runs on one
    thread, as in the carriers, so an output never depends on the render's
    length or on the split of the work, and inputs outside its window meet
    exact zeros of G, so it never depends on them.
    """
    stride = stride or factor
    if stride % factor:
        msg = f"a stride of {stride} does not interleave decimations by {factor}"
        raise ValueError(msg)
    phases = max(1, ROW // stride)
    row, per = phases * stride, max(1, OUTPUTS // phases)  # output rows per block
    g, reach = polyphase(taps, stride, row)  # J: the rows before its own that an output reaches
    n_out = (y.size + taps.size - 2) // factor + 1
    streams = stride // factor
    counts = [len(range(s, n_out, streams)) for s in range(streams)]

    def block(task: tuple[int, int]) -> Floats:
        s, b = task
        first = (b * per - reach) * row + s * factor  # input index of the block's first row
        z = np.zeros((per + reach) * row)
        a, e = max(first, 0), min(first + z.size, y.size)  # a block always reaches the render's last input
        z[a - first : e - first] = y[a:e]
        w = (z.reshape(per + reach, row) @ g.T).reshape(per + reach, phases, reach + 1)
        out = w[reach : reach + per, :, 0].copy()
        for i in range(1, reach + 1):
            out += w[reach - i : reach - i + per, :, i]
        flat: Floats = out.reshape(-1)
        return flat

    tasks = [(s, b) for s in range(streams) for b in range(-(-counts[s] // (per * phases)))]
    with carrier.single_threaded(), ThreadPoolExecutor(min(workers or carrier.threads(), len(tasks))) as pool:
        blocks = list(pool.map(block, tasks))
    out = np.empty(n_out)
    for s in range(streams):
        out[s::streams] = np.concatenate([v for (t, _), v in zip(tasks, blocks, strict=True) if t == s] or [out[:0]])[
            : counts[s]
        ]
    return out


def rates(sample_rate: int, k: int = K) -> tuple[int, int]:
    """Return (internal rate F, decimation factor) for an export rate."""
    base = FAMILY[sample_rate]
    return base * k, k * base // sample_rate


# Synthesis, and what the domain rule needs at every drive.

PROBES = 4  # output positions kept for every drive, in each of three places


@dataclass(frozen=True)
class Probes:
    """The exact output at a few chosen positions, at any drive.

    Every internal sample is a pure function of time, so the probes keep the
    saturator's input over their filter windows, and the output there follows
    exactly for any α. The largest of them never exceeds that drive's output
    peak, and at the render's own drive it is the peak. They sit at the
    loudest outputs of the render itself; at the loudest undriven outputs just
    after the crest of x, where a weak drive peaks; and at the loudest outputs
    at full drive just after the steepest zero crossing, where the overshoot
    of a saturated edge peaks. Nearby windows overlap almost entirely, so x is
    kept once over their union.
    """

    x: Floats  # the saturator input over the union of the windows
    windows: tuple[tuple[int, Floats], ...]  # per probe: (offset into x, the taps that weight its samples)

    def peak(self, alphas: ArrayLike) -> Floats:
        """Return the largest probe at each α: a lower bound on that drive's output peak."""
        a = np.atleast_1d(np.asarray(alphas, dtype=float))
        out = np.empty(a.size)
        for i in range(0, a.size, 32):  # bounded temporaries
            c = a[i : i + 32, None]
            y = np.where(c > 0, np.tanh(c * self.x) / np.tanh(np.where(c > 0, c, 1.0)), self.x)
            out[i : i + 32] = np.max(np.abs(np.stack([y[:, o : o + h.size] @ h for o, h in self.windows])), axis=0)
        return out


@dataclass(frozen=True)
class Synthesis:
    """Everything before the output stage, at unity gain, for one take."""

    out: Floats
    rate: int
    factor: int
    source_samples: int
    slew: Slew
    kinks: Kinks = field(compare=False, repr=False)
    kink: float  # bound on the error from the curves' kinks, at unity gain and the patch's drive
    alpha: float
    band: float  # the decimator's passband edge, for the kink bound
    true_peak: float  # the peak between the file's samples, at unity gain
    probes: Probes | None = field(default=None, compare=False, repr=False)


def _loudest(mags: Floats, first: int, n: int, taken: list[int]) -> list[int]:
    """Return up to n indices of the largest nonzero values, apart from each other and from `taken`."""
    picked: list[int] = []
    for i in np.argsort(mags)[::-1][:256]:
        m = first + int(i)
        if mags[i] and len(picked) < n and all(abs(m - q) > 1 for q in taken + picked):
            picked.append(m)
    return picked


def _probes(x: Floats, factor: int, taps: Floats, out: Floats, slew: Slew) -> Probes | None:
    """Place the probes, and keep x over their windows; None when no output can be probed."""
    count = x.size
    picked = _loudest(np.abs(out), 0, PROBES, [])
    # The output just after the crest without drive, and just after the steepest crossing at full drive,
    # computed exactly from x (the decimator's delay at low frequencies is a few outputs).
    a_max = drive_alpha(DRIVE_MAX)
    for n, alpha in ((slew.crest, 0.0), (slew.steep, a_max)):
        m0, m1 = max(0, n // factor - 8), min(out.size, n // factor + 24)  # n lies inside the render
        n0 = max(0, m0 * factor - taps.size + 1) // factor * factor
        n1 = min(count, (m1 - 1) * factor + 1)
        xs = x[n0:n1]
        y = np.tanh(alpha * xs) / np.tanh(alpha) if alpha else xs
        near = upfirdn(taps, y, down=factor)[m0 - n0 // factor : m1 - n0 // factor]
        picked += _loudest(np.abs(near), m0, PROBES, picked)
    # out[m] = Σ_n h[mK − n] · y[n] over the internal samples 0 ≤ n < count that the filter reaches; every output
    # reaches at least one.
    spans = [(m, max(0, m * factor - taps.size + 1), min(m * factor, count - 1) + 1) for m in picked]
    if not spans:
        return None
    merged: list[list[int]] = []
    for _, n0, n1 in sorted(spans, key=lambda s: s[1]):
        if merged and n0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], n1)
        else:
            merged.append([n0, n1])
    pieces: list[Floats] = []
    where: list[tuple[int, int, int]] = []
    pos = 0
    for n0, n1 in merged:
        pieces.append(x[n0:n1])
        where.append((n0, n1, pos))
        pos += n1 - n0
    windows = []
    for m, n0, n1 in spans:
        base = next(p0 + n0 - a for a, b, p0 in where if a <= n0 and n1 <= b)
        top = m * factor
        windows.append((base, np.ascontiguousarray(taps[top - np.arange(n0, n1)])))
    return Probes(np.concatenate(pieces), tuple(windows))


def interpolator_for(sample_rate: int, k: int = K) -> carrier.Interpolator:
    """Return the frozen interpolator from an export rate's tables to its internal rate."""
    return carrier.interpolator(FAMILY[sample_rate], k // 2)


def true_peak(y: Floats, taps: Floats, workers: int | None = None) -> float:
    """Return the file's true peak from the saturator's output y: the highest crest of its reconstruction.

    The decimator's output at every internal sample, before it keeps one in K, is the file's own band-limited
    reconstruction, true to its stopband, so its peak is the peak between the file's samples, as an ideal
    converter would play them, found to within 0.000001 dB. Every rate of a family shares it. It is measured in
    blocks, in parallel, so a long cymbal costs no more memory than a short one.
    """
    n = y.size + taps.size - 1 if y.size else 0

    def block(b0: int) -> float:
        b1 = min(n, b0 + PEAK_BLOCK)
        lo, hi = max(0, b0 - 1), min(n, b1 + 1)  # a sample either side, for the parabolas through its crests
        s0 = max(0, lo - taps.size + 1)  # the first input that reaches output lo
        w = oaconvolve(y[s0 : min(hi, y.size)], taps)[lo - s0 : hi - s0]
        return float(np.max(crests(w)[b0 - lo : b1 - lo]))

    starts = range(0, n, PEAK_BLOCK)
    with ThreadPoolExecutor(min(workers or carrier.threads(), max(len(starts), 1))) as pool:
        return max(pool.map(block, starts), default=0.0)


def crests(w: Floats) -> Floats:
    """Return |w|, each local peak raised to the top of the parabola through it and its two neighbours.

    w is a reconstruction at K times the base rate, where a parabola finds a peak between its samples to within
    0.000001 dB; the largest value is the true peak.
    """
    level = np.abs(w)
    out = level.copy()
    left, mid, right = level[:-2], level[1:-1], level[2:]
    bend = left - 2 * mid + right
    top = (mid >= left) & (mid >= right) & (bend < 0)
    with np.errstate(divide="ignore", invalid="ignore"):  # only where it is a peak, where the bend is negative
        out[1:-1] = np.where(top, mid - (left - right) ** 2 / (8 * bend), mid)
    return out


def synthesize(p: Patch, *, k: int = K, taps: Floats | None = None) -> Synthesis:
    """Synthesize the take that p's seed and take name, at unity gain, with everything the domain rule needs."""
    v = voice(p)
    rate, factor = rates(p.output.sample_rate, k)
    if taps is None:
        taps = decimator(FAMILY[p.output.sample_rate], k).taps
    ip = interpolator_for(p.output.sample_rate, k)
    tabs = tables(v, ip.rate)
    n = sample_count(v.t_src, rate)
    x, y, slew = internal(v, rate, n, ip, tabs)
    band = FAMILY[p.output.sample_rate] / 2
    kinks = kink_terms(v, ip, tabs)
    out = decimate(taps, y, factor, k)
    peak = true_peak(y, taps)
    del y
    return Synthesis(
        out,
        rate,
        factor,
        n,
        slew,
        kinks,
        kink_bound(kinks, v.alpha, rate, band),
        v.alpha,
        band,
        peak,
        _probes(x, factor, taps, out, slew),
    )


# The validated domain.


@dataclass(frozen=True)
class Domain:
    """A take's validated-domain status: its predicted aliasing, what is allowed, and its peak, at unity gain."""

    predicted: float  # predicted aliasing error, at unity gain
    allowed: float
    peak: float

    @property
    def inside(self) -> bool:
        """Whether the take lies inside the validated domain."""
        return self.predicted <= self.allowed

    @property
    def margin_db(self) -> float:
        """How far below what is allowed the prediction lies; negative outside."""
        return db(self.allowed / self.predicted) if self.predicted else math.inf

    @property
    def predicted_db(self) -> float:
        """The predicted aliasing relative to the peak."""
        return db(self.predicted / self.peak) if self.peak else -math.inf


def allowed(peak: float) -> float:
    """Return the aliasing allowed at a peak: −100 dB of it, or the absolute bound when near silent."""
    return LIMIT * peak if peak >= QUIET else FLOOR


def domain(s: Synthesis) -> Domain:
    """Judge a take by the validated-domain rule (plan §5)."""
    peak = float(np.max(np.abs(s.out), initial=0.0))
    q = s.slew.knee(s.alpha)
    saturation = SATURATION * math.exp(-(math.pi**2) / q) * peak if q else 0.0
    return Domain(saturation + s.kink, allowed(peak), peak)


@dataclass(frozen=True)
class Cause:
    """Where the predicted aliasing comes from, at unity gain."""

    saturation: float  # the saturation term of the rule
    kinks: float  # the kink bound
    knot: str | None = None  # the knot that contributes most to the kink bound, e.g. "noise[1].amp[2]"
    knot_ms: float | None = None
    share: float = 0.0  # its share of the kink bound

    @property
    def dominant(self) -> str | None:
        """Which term dominates the prediction: "saturation" or "kinks"; None when nothing is predicted."""
        if not (self.saturation or self.kinks):
            return None
        return "saturation" if self.saturation >= self.kinks else "kinks"

    @property
    def where(self) -> str:
        """Describe the knot that contributes most to the kink bound."""
        return f"{100 * self.share:.0f}% at {self.knot} ({self.knot_ms:g} ms)"


def knot_ms(p: Patch, path: str) -> float:
    """Return the time of a knot named by its path, such as "noise[2].amp[3]" or "damper.amp[1]", as written."""
    curve, index = path[: path.rindex("[")], int(path[path.rindex("[") + 1 : -1])
    if curve == "damper.amp":
        knots = p.damper.amp
    else:
        name = curve[: -len(".amp")]
        knots = dict(p.sources)[name].amp
    return float(knots[index - 1][0])


def cause_of(p: Patch, s: Synthesis, d: Domain) -> Cause:
    """Find where a take's predicted aliasing comes from."""
    saturation = d.predicted - s.kink
    knots = kink_knots(s.kinks, s.alpha, s.rate, s.band)
    total = math.fsum(c for _, c in knots)
    if not total:
        return Cause(saturation, s.kink)
    path, c = knots[0]
    return Cause(saturation, s.kink, path, knot_ms(p, path), c / total)


HEADROOM_STEP_DB = 0.5  # the scan's grid; every jump of the knee slew is a candidate as well


@dataclass(frozen=True)
class Headroom:
    """How much the drive can change before the validated-domain status does, estimated from one render.

    `db` is positive inside and negative outside; +inf when the take stays
    inside at every drive up to 48 dB, −inf when it is outside even without
    drive. `edge_db` is the drive at the domain's edge, NaN when there is none.
    """

    db: float
    edge_db: float = math.nan


def alphas(drives: ArrayLike) -> Floats:
    """Return α with α / tanh α = 10^(drive/20), for many drives at once: bisection on [G − 1, G]."""
    g = 10.0 ** (np.asarray(drives, dtype=float) / 20.0)
    lo, hi = np.maximum(g - 1.0, 0.0), g.copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = mid / np.tanh(np.where(mid > 0, mid, 1.0)) < g
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    out: Floats = np.where(g > 1.0, 0.5 * (lo + hi), 0.0)
    return out


def drives_db(alphas: ArrayLike) -> Floats:
    """Return the drive in dB of each α: 20 log10(α / tanh α), 0 for no drive."""
    a = np.asarray(alphas, dtype=float)
    out: Floats = np.where(a > 0, 20.0 * np.log10(a / np.tanh(np.where(a > 0, a, 1.0))), 0.0)
    return out


def ratios(s: Synthesis, alphas: ArrayLike) -> Floats:
    """Return the rule's prediction over its limit at many α: above 1 is outside.

    The knee slew and the kink bound are exact at every α. The output peak is
    bounded from below by the probes, which can only raise the ratio, so the
    headroom errs on the safe side in both directions.
    """
    a = np.atleast_1d(np.asarray(alphas, dtype=float))
    p = s.probes.peak(a) if s.probes is not None else np.zeros(a.size)
    q = s.slew.knees.at(a)
    saturation = np.where(q > 0, SATURATION * np.exp(-(math.pi**2) / np.where(q > 0, q, 1.0)) * p, 0.0)
    out: Floats = (saturation + kink_bounds(s.kinks, a, s.rate, s.band)) / np.where(p >= QUIET, LIMIT * p, FLOOR)
    return out


def _edge(s: Synthesis, inside: float, outside: float) -> float:
    """Refine an inside/outside bracket in α, 16 points at a time, and return its inside end.

    Four rounds narrow a 0.5 dB bracket below 10⁻⁵ dB.
    """
    for _ in range(4):
        a = np.linspace(inside, outside, 17)[1:-1]
        out = np.flatnonzero(ratios(s, a) > 1.0)  # a runs from the inside end to the outside end
        if out.size:
            j = out[0]
            inside, outside = (a[j - 1] if j else inside), a[j]
        else:
            inside = a[-1]
    return float(inside)


def drive_headroom(s: Synthesis, drive_db: float, d: Domain) -> Headroom | None:
    """Find the drive at which the validated-domain status first changes, searched from the patch's own drive.

    The rule is not monotone in drive: a step near the knee counts only while
    α·reach ≤ 1, so just below the α where it stops counting the domain can
    have a narrow outside island. Candidates are every 0.5 dB of drive and
    both sides of every α at which the knee slew drops, so such an island is
    never skipped; the first change is then refined.
    """
    if d.peak == 0:
        return None
    a_max = drive_alpha(DRIVE_MAX)
    drops = s.slew.knees.drops()
    if d.inside:
        grid = np.append(np.arange(drive_db + HEADROOM_STEP_DB, DRIVE_MAX, HEADROOM_STEP_DB), DRIVE_MAX)
        below = drops * (1.0 - 1e-12)  # just before each drop, where the step still counts
        cand = np.concatenate((alphas(grid), below[(below > s.alpha) & (below <= a_max)]))
        cand = np.sort(cand[cand > s.alpha])
        out = np.flatnonzero(ratios(s, cand) > 1.0)
        if not out.size:
            return Headroom(math.inf)
        j = out[0]
        edge = _edge(s, cand[j - 1] if j else s.alpha, cand[j])
    else:
        grid = np.append(np.arange(drive_db - HEADROOM_STEP_DB, 0.0, -HEADROOM_STEP_DB), 0.0)
        above = drops * (1.0 + 1e-12)  # just past each drop, where the step no longer counts
        cand = np.concatenate((alphas(grid), above[above < s.alpha]))
        cand = np.sort(cand[cand < s.alpha])[::-1]
        inside = np.flatnonzero(ratios(s, cand) <= 1.0)
        if not inside.size:
            return Headroom(-math.inf)
        j = inside[0]
        edge = _edge(s, cand[j], cand[j - 1] if j else s.alpha)
    edge_db = float(drives_db(edge))
    return Headroom(edge_db - drive_db, edge_db)


def _floor1(x: float) -> float:
    """Round down to 0.1, so a suggested drive errs on the inside."""
    return math.floor(x * 10 + 1e-9) / 10


# Takes, files, the set gain and the report.


@dataclass(frozen=True)
class Take:
    """One take's own judgement and levels; the levels are those of the exported take."""

    number: int  # 1-based, within the set
    take: int  # the patch's take value: its strike
    peak: float
    loudness: float
    domain: Domain
    cause: Cause
    headroom: Headroom | None

    @property
    def inside(self) -> bool:
        """Whether the take lies inside the validated domain."""
        return self.domain.inside

    def judgement(self) -> str:
        """Describe the take's validated-domain status."""
        d = self.domain
        if not d.predicted:
            return "inside the validated domain (no predicted aliasing)"
        if d.inside:
            return (
                f"inside the validated domain: predicted aliasing {d.predicted_db:.0f} dB, margin {d.margin_db:.1f} dB"
            )
        return (
            f"OUTSIDE the validated domain by {-d.margin_db:.1f} dB (predicted aliasing {d.predicted_db:.0f} dB); "
            "check it with `brightwork verify`"
        )

    def detail(self, drive_db: float) -> str | None:
        """Describe what the prediction comes from, and what drive would change the status."""
        c, h, d = self.cause, self.headroom, self.domain
        if c.dominant is None or not d.predicted or h is None:  # no headroom: silent
            return None
        text = "mostly saturation" if c.dominant == "saturation" else f"mostly kinks, {c.where}"
        if h.db == math.inf:
            return f"{text}; inside at every drive up to {DRIVE_MAX:g} dB"
        if h.db == -math.inf:
            knot = "" if c.dominant == "kinks" or c.knot is None else f", most at {c.knot} ({c.knot_ms:g} ms)"
            return f"{text}; outside even without drive: the kinks alone exceed the limit{knot}"
        if h.db >= 0:
            return f"{text}; drive headroom {_floor1(h.db):+.1f} dB"
        to = _floor1(h.edge_db)
        return f"{text}; drive headroom {to - drive_db:+.1f} dB: lower drive.db to {to:.1f} dB"


@dataclass(frozen=True)
class File:
    """One exported file: a take, or with --stereo two consecutive takes as left and right."""

    number: int  # 1-based
    takes: tuple[int, ...]  # indices into Report.takes
    gain_db: float  # the file gain from level matching; 0 without
    peak: float
    true_peak: float
    loudness: float


@dataclass(frozen=True)
class SourceInfo:
    """What the report says of a source: its kind, band, sub-bands, and its realized modes or partials."""

    path: str
    kind: str
    level_db: float
    band_hz: tuple[float, float]
    f_ref: float
    subbands: int
    modes: int | None = None  # modes within the band's edges
    spacing_hz: float | None = None  # their realized mean spacing
    partials_hz: tuple[float, ...] = ()  # realized partials
    wave: str | None = None

    def line(self) -> str:
        """Describe the source in one line."""
        if self.level_db == -math.inf:
            return f"{self.path} silent"
        lo, hi = self.band_hz
        bands = f"{self.subbands} sub-bands from {self.f_ref:g} Hz" if self.subbands > 1 else "one band"
        if self.kind == "noise":
            what = f"continuous noise, {lo:g}–{hi:g} Hz"
        elif self.kind == "modes":
            spacing = f", mean spacing {self.spacing_hz:.2f} Hz" if self.spacing_hz else ""
            what = f"{self.modes} modes in {lo:g}–{hi:g} Hz{spacing}"
        else:
            hz = ", ".join(f"{f:.3f}" for f in self.partials_hz[:SHOWN_PARTIALS]) + (
                " …" if len(self.partials_hz) > SHOWN_PARTIALS else ""
            )
            what = f"{self.wave} partials at {hz} Hz"
            if self.wave == "sine":
                what += f" (the lowest {describe_pitch(self.partials_hz[0])})"
            what += f", band {lo:g}–{hi:g} Hz"
        return f"{self.path} {what}; {bands}"


def source_info(p: Patch) -> tuple[SourceInfo, ...]:
    """Describe every source for the report, with its realized modes or partials."""
    out = []
    v = voice(p)
    for (path, s), src in zip(p.sources, v.sources, strict=True):
        info = SourceInfo(path, src.kind, s.level_db, (s.band_hz[0], s.band_hz[1]), s.f_ref, src.layout.count)
        if isinstance(s, Noise) and src.kind == "modes":
            _, bins = spectrum.mode_bins(p.seed, s.spacing_hz)
            hz = bins / spectrum.BINS_PER_HZ
            inside = hz[(hz >= s.band_hz[0]) & (hz <= s.band_hz[1])]
            spacing = float(np.mean(np.diff(inside))) if inside.size > 1 else None
            info = dataclasses.replace(info, modes=int(inside.size), spacing_hz=spacing)
        elif isinstance(s, Tone):
            partials = tuple(spectrum.grid_bin(f) / spectrum.BINS_PER_HZ for f, _ in s.partials)
            info = dataclasses.replace(info, partials_hz=partials, wave=s.wave)
        out.append(info)
    return tuple(out)


@dataclass(frozen=True)
class Report:
    """What a render made, as `brightwork render` reports it."""

    sources: tuple[SourceInfo, ...]
    source_ms: float
    samples: int
    sample_rate: int
    format: str
    mode: str
    drive_db: float
    set_gain_db: float
    ceiling: float
    level_match: bool
    stereo: bool
    seed: int
    takes: tuple[Take, ...]  # every rendered take
    files: tuple[File, ...]
    internal_rate: int
    factor: int
    decimator: str
    interpolator: str | None
    render_s: float

    @property
    def peak(self) -> float:
        """The highest sample of the set."""
        return max(f.peak for f in self.files)

    @property
    def over(self) -> bool:
        """A gain-mode render past the format's ceiling."""
        return self.mode == "gain" and self.peak > self.ceiling

    @property
    def inside(self) -> bool:
        """Whether every take lies inside the validated domain."""
        return all(t.inside for t in self.takes)

    @property
    def exported_ms(self) -> float:
        """The exported length, decimator tail included."""
        return 1000 * self.samples / self.sample_rate

    @property
    def loudness_spread(self) -> float:
        """How far apart the files' loudnesses lie."""
        values = [f.loudness for f in self.files if f.loudness > -math.inf]
        return max(values) - min(values) if values else 0.0

    @property
    def peak_spread(self) -> float:
        """How far apart the files' sample peaks lie, in dB."""
        values = [db(f.peak) for f in self.files if f.peak > 0]
        return max(values) - min(values) if values else 0.0

    @property
    def uneven(self) -> bool:
        """The report flags a set whose loudness spreads over more than 1 dB (plan §7.1)."""
        return self.loudness_spread > 1.0

    def _over(self) -> str:
        """Describe an over, and what to lower."""
        peak = self.peak
        if FORMATS[self.format] is None:
            return (
                f"{db(peak):+.2f} dBFS sample: over full scale; lower output.db by at least "
                f"{db(peak):.2f} dB, or pass --allow-over"
            )
        return (
            f"{db(peak):+.4f} dBFS sample: over the {FORMATS[self.format]}-bit ceiling of "
            f"{db(self.ceiling):.4f} dBFS, which keeps 2 LSB for dither; lower output.db by at least "
            f"{db(peak / self.ceiling):.4f} dB"
        )

    def _levels(self, f: File) -> tuple[str, str]:
        """Describe a file's peak and loudness."""
        if f.peak == 0:
            peak = "silent"
        elif self.over and f.peak == self.peak:
            peak = self._over()
        else:
            peak = f"{db(f.peak):.2f} dBFS sample, {db(f.true_peak):.2f} dBTP true"
        loudness = (
            f"{f.loudness:.1f} LUFS, the largest 400 ms momentary loudness (BS.1770)"
            if f.loudness > -math.inf
            else "silent"
        )
        return peak, loudness

    def _take_label(self, t: Take) -> str:
        """Name a take, and its channel in a stereo file."""
        if not self.stereo:
            return f"take {t.take}"
        return f"take {t.take} ({'left' if (t.number - 1) % 2 == 0 else 'right'})"

    def lines(self) -> list[str]:
        """Return the report's lines."""
        pcm = self.format if FORMATS[self.format] is None else f"{FORMATS[self.format]}-bit PCM"
        first, last = self.takes[0].take, self.takes[-1].take
        takes = f"take {first}" if len(self.takes) == 1 else f"takes {first}–{last}"
        channels = "stereo, takes as left and right" if self.stereo else "mono"
        rows = [(("sources" if i == 0 else ""), s.line()) for i, s in enumerate(self.sources)]
        length = (
            f"{self.source_ms:.1f} ms source, {self.exported_ms:.1f} ms exported "
            f"({self.samples} samples at {self.sample_rate} Hz, {pcm}, {channels}), seed {self.seed}, {takes}"
        )
        rows.append(("length", length))
        if len(self.files) == 1:
            peak, loudness = self._levels(self.files[0])
            rows += [("peak", peak), ("loudness", loudness)]
            for t in self.takes:
                label = "domain" if len(self.takes) == 1 else self._take_label(t)
                rows.append((label, t.judgement()))
                detail = t.detail(self.drive_db)
                if detail:
                    rows.append(("", detail))
        else:
            match = ", level-matched to the first" if self.level_match else ""
            rows.append(("files", f"{len(self.files)} files sharing one gain of {self.set_gain_db:+.2f} dB{match}"))
            spread = f"loudness spread {self.loudness_spread:.2f} dB, sample-peak spread {self.peak_spread:.2f} dB"
            if self.uneven:
                spread += "; the files vary by more than 1 dB: consider --level-match"
            rows.append(("", spread))
            if self.over:
                rows.append(("peak", self._over()))
            for f in self.files:
                gain = f", file gain {f.gain_db:+.2f} dB" if self.level_match else ""
                level = "silent" if f.peak == 0 else f"{db(f.peak):+.2f} dBFS, {f.loudness:.1f} LUFS"
                rows.append((f"file {f.number}", f"{level}{gain}"))
                for i in f.takes:
                    t = self.takes[i]
                    rows.append(("", f"{self._take_label(t)}: {t.judgement()}"))
                    detail = t.detail(self.drive_db)
                    if detail:
                        rows.append(("", detail))
        rows.append(("render", f"{1000 * self.render_s:.0f} ms"))
        width = max(9, max(len(label) for label, _ in rows) + 1)
        return [f"{label:<{width}}{text}" for label, text in rows]

    @override
    def __str__(self) -> str:
        """Format the report as `brightwork render` prints it."""
        return "\n".join(self.lines())


def take_numbers(p: Patch, files: int, *, stereo: bool = False) -> list[int]:
    """Return the takes a set renders: take, take + 1, …; two per file with stereo (plan §6)."""
    renders = files * (2 if stereo else 1)
    if isinstance(files, bool) or not 1 <= renders <= MAX_RENDERS:
        top = MAX_RENDERS // (2 if stereo else 1)
        msg = f"--takes: needs 1 to {top} files{' with --stereo' if stereo else ''}, found {files!r}"
        raise RenderError(msg)
    if p.take + renders - 1 > MAX_TAKE:
        msg = f"--takes: takes {p.take} … {p.take + renders - 1} pass 2**31; lower the patch's take"
        raise RenderError(msg)
    return [p.take + j for j in range(renders)]


def file_gains(outs: list[Floats], rate: int, *, level_match: bool) -> list[float]:
    """Return each file's gain from level matching to the first: 1 without, and for silence."""
    if not level_match:
        return [1.0] * len(outs)
    loud = [momentary_max(o, rate) for o in outs]
    return [10.0 ** ((loud[0] - x) / 20.0) if loud[0] > -math.inf and x > -math.inf else 1.0 for x in loud]


def set_gain(o: Output, peaks: Sequence[float], true_peaks: Sequence[float]) -> float:
    """Return the gain every file of a set shares, from its files' sample and true peaks after their own gains.

    It is `db` itself, or what brings the loudest file's sample peak or true peak to `db`, never putting a
    sample over the format's ceiling.
    """
    target = float(10.0 ** (o.db / 20.0))
    if o.mode == "gain":
        return target
    peak = max(peaks, default=0.0)
    if peak == 0:
        return 1.0
    top = ceiling(o.format)
    if o.mode == "peak":
        return min(target, top) / peak
    return min(target / max(true_peaks), top / peak)


def output_stage(out: Floats, gain: float, polarity: int) -> NDArray[np.float32]:
    """Apply the gain, then the polarity, then one conversion to float32."""
    scaled = out * gain
    if polarity < 0:
        scaled = -scaled
    return scaled.astype(np.float32)


def render_takes(
    p: Patch,
    takes: int = 1,
    *,
    level_match: bool = False,
    stereo: bool = False,
) -> tuple[list[NDArray[np.float32]], Report]:
    """Render a set: the float32 audio of each file, before any PCM quantization, and the set's report.

    Each file is mono, or (samples, 2) for stereo.
    """
    start = time.perf_counter()
    numbers = take_numbers(p, takes, stereo=stereo)
    top = ceiling(p.output.format)
    synths = [synthesize(dataclasses.replace(p, take=t)) if t != p.take else synthesize(p) for t in numbers]
    width = 2 if stereo else 1
    groups = [list(range(i, i + width)) for i in range(0, len(synths), width)]
    raw = [np.stack([synths[i].out for i in g], axis=1) if stereo else synths[g[0]].out for g in groups]
    gains = file_gains(raw, p.output.sample_rate, level_match=level_match)
    true_peaks = [max(synths[i].true_peak for i in g) * fg for g, fg in zip(groups, gains, strict=True)]
    peaks = [float(np.max(np.abs(r), initial=0.0)) * fg for r, fg in zip(raw, gains, strict=True)]
    gain = set_gain(p.output, peaks, true_peaks)
    audio, take_rows, file_rows = [], [], []
    for j, (g, r, fg, tp) in enumerate(zip(groups, raw, gains, true_peaks, strict=True), 1):
        a = output_stage(r, gain * fg, p.output.polarity)
        if not np.all(np.isfinite(a)):
            msg = "the render produced non-finite samples"
            raise RenderError(msg)
        audio.append(a)
        for i in g:
            s = synths[i]
            channel = a[:, i - g[0]] if stereo else a
            d = domain(s)
            take_rows.append(
                Take(
                    i + 1,
                    numbers[i],
                    float(np.max(np.abs(channel), initial=0.0)),
                    momentary_max(channel, p.output.sample_rate),
                    d,
                    cause_of(p, s, d),
                    drive_headroom(s, p.drive.db, d),
                ),
            )
        peak = float(np.max(np.abs(a), initial=0.0))
        file_rows.append(
            File(
                j,
                tuple(g),
                db(fg),
                peak,
                max(gain * tp, peak),  # the float32 rounding of the samples aside
                momentary_max(a, p.output.sample_rate),
            ),
        )
    base = FAMILY[p.output.sample_rate]
    report = Report(
        sources=source_info(p),
        source_ms=p.length_ms,
        samples=audio[0].shape[0],
        sample_rate=p.output.sample_rate,
        format=p.output.format,
        mode=p.output.mode,
        drive_db=p.drive.db,
        set_gain_db=db(gain),
        ceiling=top,
        level_match=level_match,
        stereo=stereo,
        seed=p.seed,
        takes=tuple(take_rows),
        files=tuple(file_rows),
        internal_rate=synths[0].rate,
        factor=synths[0].factor,
        decimator=decimator(base).sha256,
        interpolator=interpolator_for(p.output.sample_rate).sha256,
        render_s=time.perf_counter() - start,
    )
    return audio, report


def render(p: Patch) -> tuple[NDArray[np.float32], Report]:
    """Render the float32 audio of the patch's own take, and its report."""
    audio, report = render_takes(p, 1)
    return audio[0], report


# Export.


def quantize(audio: NDArray[np.float32], bits: int, seed: int, take: int) -> NDArray[np.int32]:
    """Round to `bits`-bit PCM with the take's TPDF dither of ±1 LSB.

    Exact zeros stay zero, so the silent first sample and any digital silence
    stay silent. The output stage keeps the peak 2 LSB below full scale, so
    the dithered value always fits.
    """
    top = 2 ** (bits - 1)
    x = audio.astype(float) * top
    q = np.round(np.where(x != 0.0, x + spectrum.dither(seed, take, x.size), 0.0))
    if q.size and (q.max() > top - 1 or q.min() < -top):
        msg = f"the render does not fit {bits}-bit PCM"
        raise RenderError(msg)
    return q.astype(np.int32)


@dataclass(frozen=True)
class Placement:
    """Where a file sits in its set: its takes, how many files the set has, and the set's and the file's gains."""

    takes: tuple[int, ...]
    files: int
    set_gain_db: float
    file_gain_db: float


def provenance(p: Patch, placed: Placement, k: int = K) -> dict[str, Any]:
    """Return what a file records of how it was made, beside its patch."""
    rate, factor = rates(p.output.sample_rate, k)
    base = FAMILY[p.output.sample_rate]
    meta: dict[str, Any] = {
        "renderer": f"brightwork {__version__}",
        "seed": p.seed,
        "takes": list(placed.takes),
        "files": placed.files,
        "set_gain_db": placed.set_gain_db,
        "file_gain_db": placed.file_gain_db,
        "decimator": {"sha256": decimator(base, k).sha256, "internal_rate": rate, "factor": factor},
        "interpolator": {"sha256": interpolator_for(p.output.sample_rate, k).sha256, "table_rate": 2 * base},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    bits = FORMATS[p.output.format]
    if bits is not None:
        meta["pcm"] = {
            "bits": bits,
            "dither": "TPDF, ±1 LSB; exact zeros kept",
            "dither_key": spectrum.DITHER_KEY.decode(),
            "seed": p.seed,
            "takes": list(placed.takes),
        }
    return meta


def root(p: Patch) -> tuple[int, int] | None:
    """Return the smpl root: tone[1]'s lowest partial, when that tone sounds; unpitched patches have none."""
    if p.tone and p.tone[0].level_db > -math.inf:
        return wav.root_note(spectrum.grid_bin(p.tone[0].partials[0][0]) / spectrum.BINS_PER_HZ)
    return None


def write_file(
    path: str | os.PathLike[str],
    audio: NDArray[np.float32],
    p: Patch,
    placed: Placement,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write one file in the patch's format, with its root note, its patch text and provenance.

    The patch text is the first take's: `brightwork extract` recovers exactly the
    patch that renders this file, or its left channel.
    """
    bits = FORMATS[p.output.format]
    takes = placed.takes
    samples: NDArray[np.float32] | NDArray[np.int32]
    if bits is None:
        samples = audio
    elif audio.ndim == 1:
        samples = quantize(audio, bits, p.seed, takes[0])
    else:
        samples = np.stack([quantize(audio[:, c], bits, p.seed, t) for c, t in enumerate(takes)], axis=1)
    meta = {"patch": dumps(dataclasses.replace(p, take=takes[0])), **provenance(p, placed), **(extra or {})}
    wav.write(path, samples, wav.Format(p.output.sample_rate, bits), root=root(p), meta=meta)


def export(path: str | os.PathLike[str], audio: NDArray[np.float32], p: Patch, report: Report, f: File) -> None:
    """Write one file of a render."""
    takes = tuple(report.takes[i].take for i in f.takes)
    write_file(path, audio, p, Placement(takes, len(report.files), report.set_gain_db, f.gain_db))
