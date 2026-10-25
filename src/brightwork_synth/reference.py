"""The reference method behind `brightwork verify` and the adversarial grid (plan §8.2).

The shipping output is D(y_F), where y_F[n] = y(n / F) and D is the frozen
decimator. The reference renders the same take at a reference rate R, removes
everything above F/2 with a zero-phase FFT mask (flat to 0.4F, then a raised
cosine down to zero at 0.5F), keeps every (R/F)-th sample to get ŷ_F, an
alias-free version of the same internal signal, and passes that through the
same frozen filter. Both outputs share the filter, so its ripple and phase
cancel exactly; the difference is the aliasing caused by the finite internal
rate. There is no time shifting, no gain matching, and the output gain is
fixed at unity.

Reference rates are absolute, not multiples of F: 128 × base (6.144 MHz for
the 48 kHz family), with a convergence check at 256 × base and, on a subset,
512 × base. A reference counts only if the pair agrees 20 dB below the limits.
The reference's carriers come from the same tables, through interpolators of
the same design at its own rates, whose exactness §8.1 tests separately; so
the reference tests the rendering policy: the saturator's aliasing, the kinks
and the decimator.

The mask is applied in FFT segments with guard bands wide enough for its
kernel to decay far below the error budget, and ŷ_F includes the kernel's
ringing just before the strike and after the end. Segments run in parallel,
each computing its own stretch of the internal signal, so memory stays a few
segments' worth however long the sound.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import upfirdn

from . import carrier
from .patch import Patch
from .render import FAMILY, K, Domain, decimator, domain, rates, synthesize, take_numbers
from .voice import Voice, _gate, sample_count, saturate, tables, voice

REFERENCE = 128  # the reference rate, in multiples of the base rate
CHECK = 256  # its convergence check
DEEP = 512  # the further check on a subset
SEGMENT = 2**15  # output samples at F per FFT segment
GUARD = 2**12  # samples at F on each side of a segment, and before and after the signal
LIMIT_DB = -100.0
NEAR_SILENT = 1e-2  # reference peak below which the absolute bound applies instead
ABS_LIMIT = 1e-7
MARGIN_DB = 20.0  # references must agree this far below the limits


@functools.lru_cache(maxsize=8)
def _mask(length: int, hi_rate: int, rate: int) -> np.ndarray:
    f = np.arange(length // 2 + 1) * (hi_rate / length)
    edge = np.clip((f - 0.4 * rate) / (0.1 * rate), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * edge))


def _carrier(tab: np.ndarray, ip: carrier.Interpolator, a: int, b: int) -> np.ndarray:
    """Internal samples a ≤ n < b of a carrier, at any a and b."""
    a0 = a // ip.up * ip.up
    count = -(-(b - a0) // ip.up) * ip.up
    return carrier.block(tab, ip, a0, count)[a - a0 : b - a0]


def x_range(v: Voice, rate: int, ip: carrier.Interpolator, tabs, n0: int, n1: int, count: int) -> np.ndarray:
    """x(n / rate) for n0 ≤ n < n1, and 0 outside 0 ≤ n < count: the same model as the shipping synthesis."""
    x = np.zeros(n1 - n0)
    lo, hi = max(n0, 0), min(n1, count)
    if lo >= hi:
        return x
    t = np.arange(lo, hi) / rate
    for s in v.audible:
        acc = np.zeros(hi - lo)
        for sb, tab in zip(s.subbands, tabs[s.path]):
            a = max(lo, math.ceil(float(sb.env.t[0]) * rate) - 1)
            b = min(hi, math.floor(float(sb.env.t[-1]) * rate) + 2)
            if a < b:
                acc[a - lo : b - lo] += sb.env.on(t[a - lo : b - lo]) * _carrier(tab, ip, a, b)
        factor = s.gain * _gate(s, t)
        if v.damper is not None:
            factor = factor * v.damper.on(t)
        x[lo - n0 : hi - n0] += factor * acc
    return x


def band_limited(
    v: Voice, rate: int, ip: carrier.Interpolator, tabs, n_start: int, n_stop: int, m: int, workers: int = 1
) -> np.ndarray:
    """ŷ_F[n] for n_start ≤ n < n_stop: y rendered at m·F, masked above F/2, every m-th sample."""
    hi = m * rate
    count = sample_count(v.t_src, hi)
    out = np.empty(n_stop - n_start)

    def segment(a: int) -> None:
        b = min(a + SEGMENT, n_stop)
        length = next_fast_len((b - a + 2 * GUARD) * m, real=True)
        start = (a - GUARD) * m
        y = saturate(v.alpha, x_range(v, hi, ip, tabs, start, start + length, count))
        z = irfft(rfft(y) * _mask(length, hi, rate), length)
        out[a - n_start : b - n_start] = z[GUARD * m : (GUARD + b - a) * m : m]

    with carrier.single_threaded(), ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(segment, range(n_start, n_stop, SEGMENT)))
    return out


@dataclass(frozen=True)
class Reference:
    out: np.ndarray  # D(ŷ_F), aligned with and as long as the shipping output
    internal: np.ndarray  # ŷ_F from −GUARD to N + GUARD


def reference(
    p: Patch, multiple: int = REFERENCE, *, k: int = K, taps: np.ndarray | None = None, workers: int | None = None
) -> Reference:
    """The reference output of p's take at `multiple` × base."""
    v = voice(p)
    rate, factor = rates(p.output.sample_rate, k)
    if multiple % k:
        raise ValueError(f"a reference at {multiple}× base needs K dividing it, not {k}")
    base = FAMILY[p.output.sample_rate]
    if taps is None:
        taps = decimator(base, k).taps
    ip = carrier.interpolator(base, multiple // 2)
    tabs = tables(v, ip.rate)
    n = sample_count(v.t_src, rate)
    z = band_limited(v, rate, ip, tabs, -GUARD, n + GUARD, multiple // k, workers or os.cpu_count() or 1)
    first = GUARD // factor  # GUARD is a multiple of every factor, so the output grid stays aligned
    out = upfirdn(taps, z, down=factor)[first : first + (n + taps.size - 2) // factor + 1]
    return Reference(out, z)


@dataclass(frozen=True)
class Errors:
    rel_db: float
    max_db: float
    max_abs: float
    peak: float

    @property
    def near_silent(self) -> bool:
        return self.peak < NEAR_SILENT

    def passes(self, limit_db: float = LIMIT_DB, abs_limit: float = ABS_LIMIT) -> bool:
        if self.near_silent:
            return self.max_abs <= abs_limit
        return self.rel_db <= limit_db and self.max_db <= limit_db


def _db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


def compare(test: np.ndarray, ref: np.ndarray) -> Errors:
    err = test - ref
    peak = float(np.max(np.abs(ref), initial=0.0))
    max_abs = float(np.max(np.abs(err), initial=0.0))
    norm = float(np.linalg.norm(ref))
    return Errors(
        _db(np.linalg.norm(err) / norm) if norm else -math.inf,
        _db(max_abs / peak) if peak else -math.inf,
        max_abs,
        peak,
    )


def leakage_db(internal: np.ndarray, out: np.ndarray, rate: int, base: int, stopband_db: float) -> float:
    """Bound on the decimator's own leakage, relative to the output: stopband gain × out-of-band norm."""
    X = np.fft.rfft(internal)
    f = np.arange(X.size) * rate / internal.size
    oob = math.sqrt(2 * float(np.sum(np.abs(X[f >= base / 2]) ** 2)) / internal.size)
    norm = float(np.linalg.norm(out))
    return _db(10 ** (-stopband_db / 20) * oob / norm) if norm else -math.inf


@dataclass(frozen=True)
class Verification:
    take: int
    error: Errors  # shipping against the reference
    convergence: Errors  # the reference against its check
    reference_hz: int
    check_hz: int
    leakage_db: float
    domain: Domain

    @property
    def converged(self) -> bool:
        return self.convergence.passes(LIMIT_DB - MARGIN_DB, ABS_LIMIT * 10 ** (-MARGIN_DB / 20))

    @property
    def passed(self) -> bool:
        return self.converged and self.error.passes()

    def lines(self) -> list[str]:
        e, c = self.error, self.convergence
        if e.near_silent:
            error = f"max error {e.max_abs:.2e} (near-silent: limit {ABS_LIMIT:g})"
            conv = f"agree to {c.max_abs:.2e} (limit {ABS_LIMIT * 10 ** (-MARGIN_DB / 20):g})"
        else:
            error = f"E_rel {e.rel_db:.1f} dB, E_max {e.max_db:.1f} dB (limit {LIMIT_DB:g} dB)"
            conv = f"agree to E_rel {c.rel_db:.1f} dB, E_max {c.max_db:.1f} dB (limit {LIMIT_DB - MARGIN_DB:g} dB)"
        d = self.domain
        inside = f"inside, margin {d.margin_db:.1f} dB" if d.inside else f"OUTSIDE by {-d.margin_db:.1f} dB"
        predicted = f"predicted aliasing {d.predicted_db:.0f} dB" if d.predicted else "no predicted aliasing"
        verdict = "PASS" if self.passed else ("FAIL: the reference did not converge" if not self.converged else "FAIL")
        return [
            f"error       {error}",
            f"reference   {self.reference_hz / 1e6:g} and {self.check_hz / 1e6:g} MHz {conv}",
            f"leakage     ≤ {self.leakage_db:.0f} dB of the output (the same in both, not counted)",
            f"domain      {predicted}: {inside}",
            f"result      {verdict}",
        ]

    def __str__(self) -> str:
        return "\n".join(self.lines())


def verify_take(
    p: Patch,
    *,
    k: int = K,
    taps: np.ndarray | None = None,
    multiple: int = REFERENCE,
    check: int = CHECK,
    workers: int | None = None,
) -> Verification:
    """Compare the shipping render of p's take with its reference (plan §8.2)."""
    base = FAMILY[p.output.sample_rate]
    stopband = decimator(base, k).stopband_db if taps is None else 110.0
    ship = synthesize(p, k=k, taps=taps)
    ref = reference(p, multiple, k=k, taps=taps, workers=workers)
    finer = reference(p, check, k=k, taps=taps, workers=workers)
    return Verification(
        take=p.take,
        error=compare(ship.out, ref.out),
        convergence=compare(ref.out, finer.out),
        reference_hz=multiple * base,
        check_hz=check * base,
        leakage_db=leakage_db(finer.internal, finer.out, ship.rate, base, stopband),
        domain=domain(ship),
    )


@dataclass(frozen=True)
class Verifications:
    takes: tuple[Verification, ...]

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.takes)

    def __str__(self) -> str:
        if len(self.takes) == 1:
            return str(self.takes[0])
        lines = []
        for v in self.takes:
            lines.append(f"take {v.take}")
            lines += [f"  {line}" for line in v.lines()]
        passed = sum(v.passed for v in self.takes)
        lines.append(f"result      {passed} of {len(self.takes)} takes pass")
        return "\n".join(lines)


def verify(p: Patch, *, takes: int = 1, k: int = K) -> Verifications:
    """Verify each take of a patch: take, take + 1, …, each judged on its own as each channel of a pair is."""
    return Verifications(tuple(verify_take(dataclasses.replace(p, take=t), k=k) for t in take_numbers(p, takes)))
