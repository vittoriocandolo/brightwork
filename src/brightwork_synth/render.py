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
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import scipy
from scipy.signal import upfirdn

from . import __version__, carrier, spectrum, wav
from .loudness import momentary_max, true_peak
from .patch import MAX_TAKE, Patch, describe_pitch, dumps
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

K = 32  # frozen by the adversarial grid (plan §3, docs/VALIDATION.md)
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


class RenderError(RuntimeError):
    pass


def ceiling(fmt: str) -> float:
    """The largest sample a format takes: full scale for float, and 2 LSB below it for PCM, so dither never clips."""
    bits = FORMATS[fmt]
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


def db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


# Frozen decimators.


@dataclass(frozen=True)
class Decimator:
    base: int
    k: int
    taps: np.ndarray
    sha256: str
    stopband_db: float


def manifest() -> dict:
    return json.loads(resources.files("brightwork_synth").joinpath("data", "decimators.json").read_text())


def load_taps(raw: bytes, entry: dict) -> np.ndarray:
    """Coefficients from their shipped bytes, refused unless they match the checksum."""
    if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != 8 * entry["taps"]:
        raise RenderError(f"decimator {entry['base']}x{entry['K']} does not match its checksum")
    taps = np.frombuffer(raw, dtype="<f8").astype(float)
    taps.flags.writeable = False
    return taps


@functools.cache
def decimator(base: int, k: int = K) -> Decimator:
    """A frozen decimator from brightwork_synth/data, verified against its checksum."""
    entry = manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("brightwork_synth").joinpath("data", entry["file"]).read_bytes()
    return Decimator(base, k, load_taps(raw, entry), entry["sha256"], entry["stopband_db"])


# The decimator as matrix products.

ROW = 256  # input samples per row of a block's matrix
OUTPUTS = 2**13  # outputs per block


def polyphase(taps: np.ndarray, stride: int, row: int) -> tuple[np.ndarray, int]:
    """(G, J) with G[r·(J + 1) + j, t] = h[j·row + r·stride − t], and 0 where that index is outside the filter."""
    phases, J = row // stride, (taps.size + row - 2) // row
    k = np.arange(J + 1)[:, None] * row + (np.arange(phases) * stride)[:, None, None] - np.arange(row)
    g = np.where((k >= 0) & (k < taps.size), taps[np.clip(k, 0, taps.size - 1)], 0.0)
    return g.reshape(phases * (J + 1), row), J


def decimate(
    taps: np.ndarray, y: np.ndarray, factor: int, stride: int | None = None, workers: int | None = None
) -> np.ndarray:
    """out[m] = Σ_k h[k] y[m·factor − k], with y zero outside the render, every output computed the same way always.

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
        raise ValueError(f"a stride of {stride} does not interleave decimations by {factor}")
    phases = max(1, ROW // stride)
    row, per = phases * stride, max(1, OUTPUTS // phases)  # output rows per block
    g, J = polyphase(taps, stride, row)
    n_out = (y.size + taps.size - 2) // factor + 1
    streams = stride // factor
    counts = [len(range(s, n_out, streams)) for s in range(streams)]

    def block(task: tuple[int, int]) -> np.ndarray:
        s, b = task
        first = (b * per - J) * row + s * factor  # input index of the block's first row
        z = np.zeros((per + J) * row)
        a, e = max(first, 0), min(first + z.size, y.size)
        if e > a:
            z[a - first : e - first] = y[a:e]
        w = (z.reshape(per + J, row) @ g.T).reshape(per + J, phases, J + 1)
        out = w[J : J + per, :, 0].copy()
        for i in range(1, J + 1):
            out += w[J - i : J - i + per, :, i]
        return out.reshape(-1)

    tasks = [(s, b) for s in range(streams) for b in range(-(-counts[s] // (per * phases)))]
    with carrier.single_threaded(), ThreadPoolExecutor(min(workers or os.cpu_count() or 1, len(tasks))) as pool:
        blocks = list(pool.map(block, tasks))
    out = np.empty(n_out)
    for s in range(streams):
        out[s::streams] = np.concatenate([v for (t, _), v in zip(tasks, blocks) if t == s] or [out[:0]])[: counts[s]]
    return out


def rates(sample_rate: int, k: int = K) -> tuple[int, int]:
    """(internal rate F, decimation factor) for an export rate."""
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

    x: np.ndarray  # the saturator input over the union of the windows
    windows: tuple  # per probe: (offset into x, the taps that weight its samples)

    def peak(self, alphas) -> np.ndarray:
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

    out: np.ndarray
    rate: int
    factor: int
    source_samples: int
    slew: Slew
    kinks: Kinks = field(compare=False, repr=False)
    kink: float  # bound on the error from the curves' kinks, at unity gain and the patch's drive
    alpha: float
    band: float  # the decimator's passband edge, for the kink bound
    probes: Probes | None = field(default=None, compare=False, repr=False)


def _loudest(mags: np.ndarray, first: int, n: int, taken: list[int]) -> list[int]:
    """Up to n indices of the largest nonzero values, apart from each other and from `taken`."""
    picked: list[int] = []
    for i in np.argsort(mags)[::-1][:256]:
        m = first + int(i)
        if mags[i] == 0 or len(picked) == n:
            break
        if all(abs(m - q) > 1 for q in taken + picked):
            picked.append(m)
    return picked


def _probes(x: np.ndarray, factor: int, taps: np.ndarray, out: np.ndarray, slew: Slew) -> Probes | None:
    count = x.size
    picked = _loudest(np.abs(out), 0, PROBES, [])
    # The output just after the crest without drive, and just after the steepest crossing at full drive,
    # computed exactly from x (the decimator's delay at low frequencies is a few outputs).
    a_max = drive_alpha(DRIVE_MAX)
    for n, alpha in ((slew.crest, 0.0), (slew.steep, a_max)):
        m0, m1 = max(0, n // factor - 8), min(out.size, n // factor + 24)
        if m1 <= m0:
            continue
        n0 = max(0, m0 * factor - taps.size + 1) // factor * factor
        n1 = min(count, (m1 - 1) * factor + 1)
        xs = x[n0:n1]
        y = np.tanh(alpha * xs) / np.tanh(alpha) if alpha else xs
        near = upfirdn(taps, y, down=factor)[m0 - n0 // factor : m1 - n0 // factor]
        picked += _loudest(np.abs(near), m0, PROBES, picked)
    # out[m] = Σ_n h[mK − n] · y[n] over the internal samples 0 ≤ n < count that the filter reaches.
    spans = []
    for m in picked:
        top = m * factor
        n0, n1 = max(0, top - taps.size + 1), min(top, count - 1) + 1
        if n1 > n0:
            spans.append((m, n0, n1))
    if not spans:
        return None
    merged: list[list[int]] = []
    for _, n0, n1 in sorted(spans, key=lambda s: s[1]):
        if merged and n0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], n1)
        else:
            merged.append([n0, n1])
    xs, where, pos = [], [], 0
    for n0, n1 in merged:
        xs.append(x[n0:n1])
        where.append((n0, n1, pos))
        pos += n1 - n0
    windows = []
    for m, n0, n1 in spans:
        base = next(p0 + n0 - a for a, b, p0 in where if a <= n0 and n1 <= b)
        top = m * factor
        windows.append((base, np.ascontiguousarray(taps[top - np.arange(n0, n1)])))
    return Probes(np.concatenate(xs), tuple(windows))


def interpolator_for(sample_rate: int, k: int = K) -> carrier.Interpolator:
    return carrier.interpolator(FAMILY[sample_rate], k // 2)


def synthesize(p: Patch, *, k: int = K, taps: np.ndarray | None = None) -> Synthesis:
    """The take that p's seed and take name, at unity gain, with everything the domain rule needs."""
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
        _probes(x, factor, taps, out, slew),
    )


# The validated domain.


@dataclass(frozen=True)
class Domain:
    predicted: float  # predicted aliasing error, at unity gain
    allowed: float
    peak: float

    @property
    def inside(self) -> bool:
        return self.predicted <= self.allowed

    @property
    def margin_db(self) -> float:
        return db(self.allowed / self.predicted) if self.predicted else math.inf

    @property
    def predicted_db(self) -> float:
        """Relative to the peak."""
        return db(self.predicted / self.peak) if self.peak else -math.inf


def allowed(peak: float) -> float:
    return LIMIT * peak if peak >= QUIET else FLOOR


def domain(s: Synthesis) -> Domain:
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
        if not (self.saturation or self.kinks):
            return None
        return "saturation" if self.saturation >= self.kinks else "kinks"

    @property
    def where(self) -> str:
        return f"{100 * self.share:.0f}% at {self.knot} ({self.knot_ms:g} ms)"


def knot_ms(p: Patch, path: str) -> float:
    """The time of a knot named by its path, such as "noise[2].amp[3]" or "damper.amp[1]", as the patch writes it."""
    curve, index = path[: path.rindex("[")], int(path[path.rindex("[") + 1 : -1])
    if curve == "damper.amp":
        knots = p.damper.amp
    else:
        name = curve[: -len(".amp")]
        knots = dict(p.sources)[name].amp
    return knots[index - 1][0]


def cause_of(p: Patch, s: Synthesis, d: Domain) -> Cause:
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
    drive. `edge_db` is the drive at the domain's edge.
    """

    db: float
    edge_db: float | None = None


def _alphas(drives) -> np.ndarray:
    """α with α / tanh α = 10^(drive/20), for many drives at once: bisection on [G − 1, G]."""
    g = 10.0 ** (np.asarray(drives, dtype=float) / 20.0)
    lo, hi = np.maximum(g - 1.0, 0.0), g.copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = mid / np.tanh(np.where(mid > 0, mid, 1.0)) < g
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(g > 1.0, 0.5 * (lo + hi), 0.0)


def _drives(alphas) -> np.ndarray:
    a = np.asarray(alphas, dtype=float)
    return np.where(a > 0, 20.0 * np.log10(a / np.tanh(np.where(a > 0, a, 1.0))), 0.0)


def ratios(s: Synthesis, alphas) -> np.ndarray:
    """The rule's prediction over its limit at many α: above 1 is outside.

    The knee slew and the kink bound are exact at every α. The output peak is
    bounded from below by the probes, which can only raise the ratio, so the
    headroom errs on the safe side in both directions.
    """
    a = np.atleast_1d(np.asarray(alphas, dtype=float))
    p = s.probes.peak(a) if s.probes is not None else np.zeros(a.size)
    q = s.slew.knees.at(a)
    saturation = np.where(q > 0, SATURATION * np.exp(-(math.pi**2) / np.where(q > 0, q, 1.0)) * p, 0.0)
    return (saturation + kink_bounds(s.kinks, a, s.rate, s.band)) / np.where(p >= QUIET, LIMIT * p, FLOOR)


def _edge(s: Synthesis, inside: float, outside: float) -> float:
    """Refine an inside/outside bracket in α, 16 points at a time; returns the inside end.

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
    """The drive at which the validated-domain status first changes, searched from the patch's own drive.

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
        cand = np.concatenate((_alphas(grid), below[(below > s.alpha) & (below <= a_max)]))
        cand = np.sort(cand[cand > s.alpha])
        out = np.flatnonzero(ratios(s, cand) > 1.0)
        if not out.size:
            return Headroom(math.inf)
        j = out[0]
        edge = _edge(s, cand[j - 1] if j else s.alpha, cand[j])
    else:
        grid = np.append(np.arange(drive_db - HEADROOM_STEP_DB, 0.0, -HEADROOM_STEP_DB), 0.0)
        above = drops * (1.0 + 1e-12)  # just past each drop, where the step no longer counts
        cand = np.concatenate((_alphas(grid), above[above < s.alpha]))
        cand = np.sort(cand[cand < s.alpha])[::-1]
        inside = np.flatnonzero(ratios(s, cand) <= 1.0)
        if not inside.size:
            return Headroom(-math.inf)
        j = inside[0]
        edge = _edge(s, cand[j], cand[j - 1] if j else s.alpha)
    edge_db = float(_drives(edge))
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
        return self.domain.inside

    def judgement(self) -> str:
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
        """What the prediction comes from, and what drive would change the status."""
        c, h, d = self.cause, self.headroom, self.domain
        if c.dominant is None or not d.predicted:
            return None
        text = "mostly saturation" if c.dominant == "saturation" else f"mostly kinks, {c.where}"
        if h is None:
            return text
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
    takes: tuple  # indices into Report.takes
    gain_db: float  # the file gain from level matching; 0 without
    peak: float
    true_peak: float
    loudness: float


@dataclass(frozen=True)
class SourceInfo:
    path: str
    kind: str
    level_db: float
    band_hz: tuple
    f_ref: float
    subbands: int
    modes: int | None = None  # modes within the band's edges
    spacing_hz: float | None = None  # their realized mean spacing
    partials_hz: tuple = ()  # realized partials
    wave: str | None = None

    def line(self) -> str:
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
            hz = ", ".join(f"{f:.3f}" for f in self.partials_hz[:6]) + (" …" if len(self.partials_hz) > 6 else "")
            what = f"{self.wave} partials at {hz} Hz"
            if self.wave == "sine":
                what += f" (the lowest {describe_pitch(self.partials_hz[0])})"
            what += f", band {lo:g}–{hi:g} Hz"
        return f"{self.path} {what}; {bands}"


def source_info(p: Patch) -> tuple:
    out = []
    v = voice(p)
    for (path, s), src in zip(p.sources, v.sources):
        info = dict(
            path=path,
            kind=src.kind,
            level_db=s.level_db,
            band_hz=tuple(s.band_hz),
            f_ref=s.f_ref,
            subbands=src.layout.count,
        )
        if src.kind == "modes":
            _, bins = spectrum.mode_bins(p.seed, s.spacing_hz)
            hz = bins / spectrum.BINS_PER_HZ
            inside = hz[(hz >= s.band_hz[0]) & (hz <= s.band_hz[1])]
            info.update(modes=int(inside.size), spacing_hz=float(np.mean(np.diff(inside))) if inside.size > 1 else None)
        elif src.kind == "tone":
            info.update(partials_hz=tuple(spectrum.grid_bin(f) / spectrum.BINS_PER_HZ for f, _ in s.partials))
            info.update(wave=s.wave)
        out.append(SourceInfo(**info))
    return tuple(out)


@dataclass(frozen=True)
class Report:
    sources: tuple
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
    takes: tuple  # (Take, ...): every rendered take
    files: tuple  # (File, ...)
    internal_rate: int
    factor: int
    decimator: str
    interpolator: str
    render_s: float

    @property
    def peak(self) -> float:
        return max(f.peak for f in self.files)

    @property
    def over(self) -> bool:
        """A gain-mode render past the format's ceiling."""
        return self.mode == "gain" and self.peak > self.ceiling

    @property
    def inside(self) -> bool:
        return all(t.inside for t in self.takes)

    @property
    def exported_ms(self) -> float:
        return 1000 * self.samples / self.sample_rate

    @property
    def loudness_spread(self) -> float:
        values = [f.loudness for f in self.files if f.loudness > -math.inf]
        return max(values) - min(values) if values else 0.0

    @property
    def peak_spread(self) -> float:
        values = [db(f.peak) for f in self.files if f.peak > 0]
        return max(values) - min(values) if values else 0.0

    @property
    def uneven(self) -> bool:
        """The report flags a set whose loudness spreads over more than 1 dB (plan §7.1)."""
        return self.loudness_spread > 1.0

    def _over(self) -> str:
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
        if f.peak == 0:
            peak = "silent"
        elif self.over and f.peak == self.peak:
            peak = self._over()
        else:
            peak = f"{db(f.peak):.2f} dBFS sample, {db(f.true_peak):.2f} dBTP true (4× estimate, information only)"
        loudness = (
            f"{f.loudness:.1f} LUFS, the largest 400 ms momentary loudness (BS.1770)"
            if f.loudness > -math.inf
            else "silent"
        )
        return peak, loudness

    def _take_label(self, t: Take) -> str:
        if not self.stereo:
            return f"take {t.take}"
        return f"take {t.take} ({'left' if (t.number - 1) % 2 == 0 else 'right'})"

    def lines(self) -> list[str]:
        pcm = self.format if FORMATS[self.format] is None else f"{FORMATS[self.format]}-bit PCM"
        first, last = self.takes[0].take, self.takes[-1].take
        takes = f"take {first}" if len(self.takes) == 1 else f"takes {first}–{last}"
        channels = "stereo, takes as left and right" if self.stereo else "mono"
        rows = [(("sources" if i == 0 else ""), s.line()) for i, s in enumerate(self.sources)]
        rows.append(
            (
                "length",
                f"{self.source_ms:.1f} ms source, {self.exported_ms:.1f} ms exported "
                f"({self.samples} samples at {self.sample_rate} Hz, {pcm}, {channels}), seed {self.seed}, {takes}",
            )
        )
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

    def __str__(self) -> str:
        return "\n".join(self.lines())


def take_numbers(p: Patch, files: int, stereo: bool = False) -> list[int]:
    """The takes a set renders: take, take + 1, …; two per file with stereo (plan §6)."""
    renders = files * (2 if stereo else 1)
    if isinstance(files, bool) or not isinstance(files, int) or not 1 <= renders <= MAX_RENDERS:
        top = MAX_RENDERS // (2 if stereo else 1)
        raise RenderError(f"--takes: needs 1 to {top} files{' with --stereo' if stereo else ''}, found {files!r}")
    if p.take + renders - 1 > MAX_TAKE:
        raise RenderError(f"--takes: takes {p.take} … {p.take + renders - 1} pass 2**31; lower the patch's take")
    return [p.take + j for j in range(renders)]


def set_gains(outs: list[np.ndarray], p: Patch, level_match: bool, top: float) -> tuple[float, list[float]]:
    """One gain for the whole set, and each file's gain from level matching (1 without)."""
    rate = p.output.sample_rate
    gains = [1.0] * len(outs)
    if level_match:
        loud = [momentary_max(o, rate) for o in outs]
        gains = [10.0 ** ((loud[0] - x) / 20.0) if loud[0] > -math.inf and x > -math.inf else 1.0 for x in loud]
    if p.output.mode == "gain":
        return 10.0 ** (p.output.db / 20.0), gains
    peak = max(float(np.max(np.abs(o), initial=0.0)) * g for o, g in zip(outs, gains))
    target = min(10.0 ** (p.output.db / 20.0), top)
    return (target / peak if peak > 0 else 1.0), gains


def output_stage(out: np.ndarray, gain: float, polarity: int) -> np.ndarray:
    """The gain, then the polarity, then one conversion to float32."""
    scaled = out * gain
    if polarity < 0:
        scaled = -scaled
    return scaled.astype(np.float32)


def render_takes(
    p: Patch, takes: int = 1, level_match: bool = False, stereo: bool = False
) -> tuple[list[np.ndarray], Report]:
    """The float32 audio of each file, mono or as (samples, 2) for stereo, before any PCM quantization, and the
    report of the set."""
    start = time.perf_counter()
    numbers = take_numbers(p, takes, stereo)
    top = ceiling(p.output.format)
    synths = [synthesize(dataclasses.replace(p, take=t)) if t != p.take else synthesize(p) for t in numbers]
    width = 2 if stereo else 1
    groups = [list(range(i, i + width)) for i in range(0, len(synths), width)]
    raw = [np.stack([synths[i].out for i in g], axis=1) if stereo else synths[g[0]].out for g in groups]
    gain, file_gains = set_gains(raw, p, level_match, top)
    audio, take_rows, file_rows = [], [], []
    for j, (g, r, fg) in enumerate(zip(groups, raw, file_gains), 1):
        a = output_stage(r, gain * fg, p.output.polarity)
        if not np.all(np.isfinite(a)):
            raise RenderError("the render produced non-finite samples")
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
                )
            )
        file_rows.append(
            File(
                j,
                tuple(g),
                db(fg),
                float(np.max(np.abs(a), initial=0.0)),
                max(true_peak(a[:, c]) for c in range(width)) if stereo else true_peak(a),
                momentary_max(a, p.output.sample_rate),
            )
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


def render(p: Patch) -> tuple[np.ndarray, Report]:
    """The float32 audio of the patch's own take, and its report."""
    audio, report = render_takes(p, 1)
    return audio[0], report


# Export.


def quantize(audio: np.ndarray, bits: int, seed: int, take: int) -> np.ndarray:
    """Round to `bits`-bit PCM with the take's TPDF dither of ±1 LSB.

    Exact zeros stay zero, so the silent first sample and any digital silence
    stay silent. The output stage keeps the peak 2 LSB below full scale, so
    the dithered value always fits.
    """
    top = 2 ** (bits - 1)
    x = audio.astype(float) * top
    q = np.round(np.where(x != 0.0, x + spectrum.dither(seed, take, x.size), 0.0))
    if q.size and (q.max() > top - 1 or q.min() < -top):
        raise RenderError(f"the render does not fit {bits}-bit PCM")
    return q.astype(np.int32)


def provenance(p: Patch, *, takes: tuple, files: int, set_gain_db: float, file_gain_db: float, k: int = K) -> dict:
    rate, factor = rates(p.output.sample_rate, k)
    base = FAMILY[p.output.sample_rate]
    meta = {
        "renderer": f"brightwork {__version__}",
        "seed": p.seed,
        "takes": list(takes),
        "files": files,
        "set_gain_db": set_gain_db,
        "file_gain_db": file_gain_db,
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
            "takes": list(takes),
        }
    return meta


def root(p: Patch) -> tuple[int, int] | None:
    """The smpl root: tone[1]'s lowest partial, when that tone sounds; unpitched patches have none."""
    if p.tone and p.tone[0].level_db > -math.inf:
        return wav.root_note(spectrum.grid_bin(p.tone[0].partials[0][0]) / spectrum.BINS_PER_HZ)
    return None


def write_file(
    path,
    audio: np.ndarray,
    p: Patch,
    *,
    takes: tuple,
    files: int,
    set_gain_db: float,
    file_gain_db: float,
    extra: dict | None = None,
) -> None:
    """Write one file in the patch's format, with its root note, its patch text and provenance.

    The patch text is the first take's: `brightwork extract` recovers exactly the
    patch that renders this file, or its left channel.
    """
    bits = FORMATS[p.output.format]
    if bits is None:
        samples = audio
    elif audio.ndim == 1:
        samples = quantize(audio, bits, p.seed, takes[0])
    else:
        samples = np.stack([quantize(audio[:, c], bits, p.seed, t) for c, t in enumerate(takes)], axis=1)
    meta = {
        "patch": dumps(dataclasses.replace(p, take=takes[0])),
        **provenance(p, takes=takes, files=files, set_gain_db=set_gain_db, file_gain_db=file_gain_db),
        **(extra or {}),
    }
    wav.write(path, samples, p.output.sample_rate, bits=bits, root=root(p), meta=meta)


def export(path, audio: np.ndarray, p: Patch, report: Report, f: File) -> None:
    """Write one file of a render."""
    write_file(
        path,
        audio,
        p,
        takes=tuple(report.takes[i].take for i in f.takes),
        files=len(report.files),
        set_gain_db=report.set_gain_db,
        file_gain_db=f.gain_db,
    )
