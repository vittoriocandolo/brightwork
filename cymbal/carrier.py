"""Carriers: each sub-band's table at the table rate, interpolated to the internal rate (plan §2.3, §3).

A sub-band's carrier is a line spectrum on the 1/8 Hz grid, so it is periodic
with period 8 s and band-limited to 20 kHz. Its exact samples at the table rate
R = 2 × base come from one inverse real FFT of R × 8 s points. Its samples at
the internal rate F = (K/2) × R come from a frozen interpolator: a
Kaiser-windowed sinc cut off at R/2, flat to 20 kHz and stopped from
R − 20 kHz, with 34 taps per phase at 96 kHz and 36 at 88.2 kHz. It is linear
phase, since it reconstructs a steady carrier; the shapes are applied after it,
at F, so it never filters a transient.

The interpolation is one matrix product per block: a sliding window of the
table times the polyphase matrix. Blocks have a fixed size and are aligned to
n = 0, every block is computed at full size, and each product runs on one
thread, so every sample is computed the same way at any render length and any
split of the work.

Values and slopes at arbitrary times, which the kink bound needs at knots, come
from the same windowed sinc evaluated as a continuous function.

Tables are computed together, one transform each in threads, and cached up to
a memory cap. A table depends only on its lines and the table rate.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources

import numpy as np
import scipy.fft
import scipy.special
from threadpoolctl import threadpool_limits

from .spectrum import BINS_PER_HZ, PERIOD

PASS_HZ = 20000.0
ATTENUATION_DB = 280.0
BLOCK = 2**18  # internal samples per block; a multiple of every interpolation factor


def table_rate(base: int) -> int:
    """R = 2 × base: 88.2 or 96 kHz."""
    return 2 * base


def table(bins: np.ndarray, amps: np.ndarray, rate: int) -> np.ndarray:
    """The carrier Σ |A_k| cos(2π k t/8 + arg A_k) at t = j / rate for one period, j = 0 … 8·rate − 1."""
    n = rate * PERIOD
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    spectrum[bins] = (n / 2) * amps
    return scipy.fft.irfft(spectrum, n=n)


# Interpolators.


def design(base: int, up: int, attenuation_db: float = ATTENUATION_DB) -> tuple[np.ndarray, int]:
    """The interpolator from 2 × base to up × 2 × base: (taps, q), with 2·up·q + 1 taps and gain up.

    A Kaiser window for the attenuation, sized by Kaiser's formula for the
    transition from 20 kHz to R − 20 kHz, on a sinc cut off at R/2; q is the
    number of table samples each side of an output that the kernel reaches.
    """
    rate = table_rate(base)
    dw = 2 * math.pi * (rate - 2 * PASS_HZ) / (up * rate)
    n = int(math.ceil((attenuation_db - 7.95) / (2.285 * dw))) + 1
    q = (n + 2 * up - 1) // (2 * up)
    n = 2 * up * q + 1
    m = np.arange(n) - (n - 1) / 2
    return np.sinc(m / up) * np.kaiser(n, kaiser_beta(attenuation_db)), q


def kaiser_beta(attenuation_db: float) -> float:
    return 0.1102 * (attenuation_db - 8.7)


@dataclass(frozen=True)
class Interpolator:
    base: int
    up: int
    q: int
    taps: np.ndarray
    matrix: np.ndarray  # (2q + 2) × up polyphase matrix
    sha256: str | None  # None for a design made on the fly (the reference rates)
    attenuation_db: float = ATTENUATION_DB

    @property
    def rate(self) -> int:
        """The table rate it reads."""
        return table_rate(self.base)

    @property
    def window(self) -> int:
        return self.matrix.shape[0]


def polyphase(taps: np.ndarray, q: int, up: int) -> np.ndarray:
    """P with output sample up·j + p = Σ_i x[j − q − 1 + i] · P[i, p]."""
    off = (taps.size - 1) // 2
    rows = 2 * q + 2
    p = np.zeros((rows, up))
    for phase in range(up):
        for i in range(rows):
            k = phase - (i - (q + 1)) * up + off
            if 0 <= k < taps.size:
                p[i, phase] = taps[k]
    p.flags.writeable = False
    return p


def manifest() -> dict:
    return json.loads(resources.files("cymbal").joinpath("data", "interpolators.json").read_text())


@functools.cache
def interpolator(base: int, up: int) -> Interpolator:
    """The frozen interpolator for a shipped factor, verified against its checksum, or a fresh design otherwise."""
    entry = manifest()["filters"].get(f"{base}x{2 * up}")
    if entry is None:
        taps, q = design(base, up)
        taps.flags.writeable = False
        return Interpolator(base, up, q, taps, polyphase(taps, q, up), None)
    raw = resources.files("cymbal").joinpath("data", entry["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != 8 * entry["taps"]:
        raise RuntimeError(f"interpolator {base}x{2 * up} does not match its checksum")
    taps = np.frombuffer(raw, dtype="<f8").astype(float)
    taps.flags.writeable = False
    return Interpolator(base, up, entry["q"], taps, polyphase(taps, entry["q"], up), entry["sha256"])


def block(tab: np.ndarray, ip: Interpolator, n0: int, count: int = BLOCK) -> np.ndarray:
    """Internal samples n0 … n0 + count − 1 of the carrier whose table is `tab`; n0 and count are multiples of up."""
    up, q = ip.up, ip.q
    j0, rows = n0 // up, count // up
    idx = np.arange(j0 - q - 1, j0 + rows + q + 1) % tab.size
    windows = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(tab[idx], ip.window)[:rows])
    return (windows @ ip.matrix).reshape(-1)


def span(tab: np.ndarray, ip: Interpolator, n0: int, n1: int):
    """Yield (a, b, samples) for the fixed blocks covering internal samples n0 ≤ n < n1, clipped to them."""
    first = n0 // BLOCK * BLOCK
    for b0 in range(first, n1, BLOCK):
        values = block(tab, ip, b0)
        a, b = max(n0, b0), min(n1, b0 + BLOCK)
        yield a, b, values[a - b0 : b - b0]


def single_threaded():
    """Keep every matrix product on its caller's thread, so a sample never depends on how work is split."""
    return threadpool_limits(limits=1, user_api="blas")


# Values and slopes at arbitrary times, from the continuous windowed sinc.


def _kernel(tau: np.ndarray, q: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """h(τ) = sinc(τ)·w(τ) and h′(τ), for |τ| ≤ q in table samples; w is the Kaiser window."""
    xi = np.sqrt(np.maximum(0.0, 1.0 - (tau / q) ** 2))
    i0 = scipy.special.i0(beta)
    w = scipy.special.i0(beta * xi) / i0
    # w′ = I1(βξ)·β·dξ/dτ / I0(β), with dξ/dτ = −τ / (q² ξ); I1(βξ)/ξ → β/2 as ξ → 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(xi > 1e-12, scipy.special.i1(beta * xi) / xi, beta / 2)
    dw = -beta * ratio * tau / (q * q * i0)
    s = np.sinc(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = np.where(tau != 0, (np.cos(np.pi * tau) - s) / tau, 0.0)
    return s * w, ds * w + s * dw


@dataclass(frozen=True)
class Weights:
    """The continuous kernel's taps for some times: shared by every table at the same rate."""

    index: np.ndarray  # table indices, one row per time
    value: np.ndarray  # h(x − m)
    slope: np.ndarray  # h′(x − m) · rate


def weights(ip: Interpolator, t) -> Weights:
    t = np.atleast_1d(np.asarray(t, dtype=float))
    rate = ip.rate
    x = np.mod(t, PERIOD) * rate
    j = np.floor(x).astype(np.int64)
    m = j[:, None] + np.arange(-ip.q + 1, ip.q + 1)[None, :]
    h, dh = _kernel(x[:, None] - m, ip.q, kaiser_beta(ip.attenuation_db))
    return Weights(m % (PERIOD * rate), h, dh * rate)


def at_weights(tab: np.ndarray, w: Weights) -> tuple[np.ndarray, np.ndarray]:
    samples = tab[w.index]
    return np.sum(samples * w.value, axis=1), np.sum(samples * w.slope, axis=1)


def at(tab: np.ndarray, ip: Interpolator, t) -> tuple[np.ndarray, np.ndarray]:
    """The carrier and its time derivative at arbitrary times t, in seconds."""
    return at_weights(tab, weights(ip, t))


# The table cache.

CACHE_BYTES = 256 * 2**20
_CACHE: dict[tuple, np.ndarray] = {}  # insertion order is recency: the most recently used last
_LOCK = threading.Lock()


def tables(keys: list[tuple], rate: int, make) -> list[np.ndarray]:
    """One read-only table per key at `rate`, computed in parallel and cached; make(i) gives the i-th key's
    (bins, amps) when its table is missing."""
    ks = [(key, rate) for key in keys]
    with _LOCK:
        todo, seen = [], set()
        for i, k in enumerate(ks):
            if k not in _CACHE and k not in seen:
                seen.add(k)
                todo.append((k, *make(i)))
        if todo:
            with ThreadPoolExecutor(max_workers=min(len(todo), os.cpu_count() or 1)) as pool:
                for (k, _, _), row in zip(todo, pool.map(lambda r: table(r[1], r[2], rate), todo)):
                    row.flags.writeable = False
                    _CACHE[k] = row
        for k in ks:
            _CACHE[k] = _CACHE.pop(k)
        needed = set(ks)
        size = sum(a.nbytes for a in _CACHE.values())
        for k in list(_CACHE):
            if size <= CACHE_BYTES:
                break
            if k not in needed:
                size -= _CACHE.pop(k).nbytes
        return [_CACHE[k] for k in ks]


def cached(key: tuple, rate: int) -> bool:
    with _LOCK:
        return (key, rate) in _CACHE


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def frequencies(bins: np.ndarray) -> np.ndarray:
    return bins / BINS_PER_HZ
