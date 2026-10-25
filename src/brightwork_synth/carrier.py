# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

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
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.fft
import scipy.special
from threadpoolctl import threadpool_limits

from .spectrum import PERIOD

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from contextlib import AbstractContextManager

    from numpy.typing import ArrayLike, NDArray

type Floats = NDArray[np.float64]
type Lines = tuple[NDArray[np.int64], NDArray[np.complex128]]  # a table's bins and complex amplitudes
type Key = tuple[object, ...]  # what a table depends on

PASS_HZ = 20000.0
ATTENUATION_DB = 280.0
BLOCK = 2**18  # internal samples per block; a multiple of every interpolation factor
XI_FLOOR = 1e-12  # below it, I1(βξ)/ξ takes its limit β/2


def table_rate(base: int) -> int:
    """R = 2 × base: 88.2 or 96 kHz."""
    return 2 * base


def table(bins: NDArray[np.int64], amps: NDArray[np.complex128], rate: int) -> Floats:
    """Return the carrier Σ |A_k| cos(2π k t/8 + arg A_k) at t = j / rate for one period, j = 0 … 8·rate − 1."""
    n = rate * PERIOD
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    spectrum[bins] = (n / 2) * amps
    out: Floats = scipy.fft.irfft(spectrum, n=n)
    return out


# Interpolators.


def design(base: int, up: int, attenuation_db: float = ATTENUATION_DB) -> tuple[Floats, int]:
    """Design the interpolator from 2 × base to up × 2 × base: (taps, q), with 2·up·q + 1 taps and gain up.

    A Kaiser window for the attenuation, sized by Kaiser's formula for the
    transition from 20 kHz to R − 20 kHz, on a sinc cut off at R/2; q is the
    number of table samples each side of an output that the kernel reaches.
    """
    rate = table_rate(base)
    dw = 2 * math.pi * (rate - 2 * PASS_HZ) / (up * rate)
    n = math.ceil((attenuation_db - 7.95) / (2.285 * dw)) + 1
    q = (n + 2 * up - 1) // (2 * up)
    n = 2 * up * q + 1
    m = np.arange(n) - (n - 1) / 2
    return np.sinc(m / up) * np.kaiser(n, kaiser_beta(attenuation_db)), q


def kaiser_beta(attenuation_db: float) -> float:
    """Return the Kaiser window's β for an attenuation above 50 dB, by Kaiser's formula."""
    return 0.1102 * (attenuation_db - 8.7)


@dataclass(frozen=True)
class Interpolator:
    """An interpolator from a table at 2 × base up by `up`: its taps, reach q and polyphase matrix."""

    base: int
    up: int
    q: int
    taps: Floats
    matrix: Floats  # (2q + 2) × up polyphase matrix
    sha256: str | None  # None for a design made on the fly (the reference rates)
    attenuation_db: float = ATTENUATION_DB

    @property
    def rate(self) -> int:
        """The table rate it reads."""
        return table_rate(self.base)

    @property
    def window(self) -> int:
        """How many table samples one output row reads."""
        rows: int = self.matrix.shape[0]
        return rows


def polyphase(taps: Floats, q: int, up: int) -> Floats:
    """Return P, with output sample up·j + p = Σ_i x[j − q − 1 + i] · P[i, p]."""
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


def manifest() -> dict[str, Any]:
    """Read the frozen interpolators' manifest: each one's file, taps, reach and checksum."""
    text = resources.files("brightwork_synth").joinpath("data", "interpolators.json").read_text()
    data: dict[str, Any] = json.loads(text)
    return data


def load_taps(raw: bytes, entry: dict[str, Any], name: str) -> Floats:
    """Return a frozen filter's coefficients from their shipped bytes, refused unless they match the checksum."""
    if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != 8 * entry["taps"]:
        msg = f"{name} does not match its checksum: reinstall Brightwork"
        raise RuntimeError(msg)
    taps = np.frombuffer(raw, dtype="<f8").astype(float)
    taps.flags.writeable = False
    return taps


@functools.cache
def interpolator(base: int, up: int) -> Interpolator:
    """Return the frozen interpolator for a shipped factor, checked against its checksum, or else a fresh design."""
    entry = manifest()["filters"].get(f"{base}x{2 * up}")
    if entry is None:
        taps, q = design(base, up)
        taps.flags.writeable = False
        return Interpolator(base, up, q, taps, polyphase(taps, q, up), None)
    raw = resources.files("brightwork_synth").joinpath("data", entry["file"]).read_bytes()
    taps = load_taps(raw, entry, f"interpolator {base}x{2 * up}")
    return Interpolator(base, up, entry["q"], taps, polyphase(taps, entry["q"], up), entry["sha256"])


def block(tab: Floats, ip: Interpolator, n0: int, count: int = BLOCK) -> Floats:
    """Return internal samples n0 … n0 + count − 1 of the carrier of table `tab`; n0 and count are multiples of up."""
    up, q = ip.up, ip.q
    j0, rows = n0 // up, count // up
    idx = np.arange(j0 - q - 1, j0 + rows + q + 1) % tab.size
    windows = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(tab[idx], ip.window)[:rows])
    out: Floats = (windows @ ip.matrix).reshape(-1)
    return out


def span(tab: Floats, ip: Interpolator, n0: int, n1: int) -> Iterator[tuple[int, int, Floats]]:
    """Yield (a, b, samples) for the fixed blocks covering internal samples n0 ≤ n < n1, clipped to them."""
    first = n0 // BLOCK * BLOCK
    for b0 in range(first, n1, BLOCK):
        values = block(tab, ip, b0)
        a, b = max(n0, b0), min(n1, b0 + BLOCK)
        yield a, b, values[a - b0 : b - b0]


def single_threaded() -> AbstractContextManager[object]:
    """Keep every matrix product on its caller's thread, so a sample never depends on how work is split."""
    limits: AbstractContextManager[object] = threadpool_limits(limits=1, user_api="blas")
    return limits


# Values and slopes at arbitrary times, from the continuous windowed sinc.


def _kernel(tau: Floats, q: int, beta: float) -> tuple[Floats, Floats]:
    """Return h(τ) = sinc(τ)·w(τ) and h′(τ), for |τ| ≤ q in table samples; w is the Kaiser window."""
    xi = np.sqrt(np.maximum(0.0, 1.0 - (tau / q) ** 2))
    i0 = scipy.special.i0(beta)
    w = scipy.special.i0(beta * xi) / i0
    # w′ = I1(βξ)·β·dξ/dτ / I0(β), with dξ/dτ = −τ / (q² ξ); I1(βξ)/ξ → β/2 as ξ → 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(xi > XI_FLOOR, scipy.special.i1(beta * xi) / xi, beta / 2)
    dw = -beta * ratio * tau / (q * q * i0)
    s = np.sinc(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = np.where(tau != 0, (np.cos(np.pi * tau) - s) / tau, 0.0)
    return s * w, ds * w + s * dw


@dataclass(frozen=True)
class Weights:
    """The continuous kernel's taps for some times: shared by every table at the same rate."""

    index: NDArray[np.int64]  # table indices, one row per time
    value: Floats  # h(x − m)
    slope: Floats  # h′(x − m) · rate


def weights(ip: Interpolator, times: ArrayLike) -> Weights:
    """Return the continuous kernel's taps at times in seconds, for any table at the interpolator's rate."""
    t = np.atleast_1d(np.asarray(times, dtype=float))
    rate = ip.rate
    x = np.mod(t, PERIOD) * rate
    j = np.floor(x).astype(np.int64)
    m = j[:, None] + np.arange(-ip.q + 1, ip.q + 1)[None, :]
    h, dh = _kernel(x[:, None] - m, ip.q, kaiser_beta(ip.attenuation_db))
    return Weights(m % (PERIOD * rate), h, dh * rate)


def at_weights(tab: Floats, w: Weights) -> tuple[Floats, Floats]:
    """Return a table's carrier and its time derivative at the times the weights were made for."""
    samples = tab[w.index]
    return np.sum(samples * w.value, axis=1), np.sum(samples * w.slope, axis=1)


def at(tab: Floats, ip: Interpolator, times: ArrayLike) -> tuple[Floats, Floats]:
    """Return the carrier and its time derivative at arbitrary times, in seconds."""
    return at_weights(tab, weights(ip, times))


# The table cache.

CACHE_BYTES = 256 * 2**20
_CACHE: dict[tuple[Key, int], Floats] = {}  # insertion order is recency: the most recently used last
_LOCK = threading.Lock()


def tables(keys: Sequence[Key], rate: int, make: Callable[[int], Lines]) -> list[Floats]:
    """Return one read-only table per key at `rate`, computed in parallel and cached.

    make(i) gives the i-th key's bins and amplitudes when its table is missing.
    """
    ks = [(key, rate) for key in keys]
    with _LOCK:
        todo: list[tuple[tuple[Key, int], NDArray[np.int64], NDArray[np.complex128]]] = []
        seen: set[tuple[Key, int]] = set()
        for i, k in enumerate(ks):
            if k not in _CACHE and k not in seen:
                seen.add(k)
                todo.append((k, *make(i)))
        if todo:
            with ThreadPoolExecutor(max_workers=min(len(todo), os.cpu_count() or 1)) as pool:
                for (k, _, _), row in zip(todo, pool.map(lambda r: table(r[1], r[2], rate), todo), strict=True):
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


def cached(key: Key, rate: int) -> bool:
    """Tell whether the table of `key` at `rate` is in the cache."""
    with _LOCK:
        return (key, rate) in _CACHE


def clear_cache() -> None:
    """Empty the table cache."""
    with _LOCK:
        _CACHE.clear()
