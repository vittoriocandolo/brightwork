# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design the frozen decimators (plan §3) and write them to src/brightwork_synth/data.

The renderer never designs filters: it loads these coefficients and checks
their SHA-256. Run this only to change the rendering policy.

    uv run python -m tools.design_decimator            # write the shipped designs
    uv run python -m tools.design_decimator --check    # compare a fresh design with the shipped data

Design: a Kaiser-window linear-phase prototype at 116 dB, converted to minimum
phase by SciPy's full-length, magnitude-matching homomorphic method with a
2^20-point FFT, then renormalized so that the taps sum to exactly 1. The
renderer ships K = 32; 16× and 64× are shipped too, for the adversarial grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.signal import firwin, kaiserord, minimum_phase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

type Floats = NDArray[np.float64]

PASS_HZ = 20000.0
ATTENUATION_DB = 116.0
N_FFT = 2**20
BASES = (44100, 48000)
SHIPPED_K = (16, 32, 64)
DATA = Path(__file__).resolve().parent.parent / "src" / "brightwork_synth" / "data"
SAME = 1e-12  # the largest difference between a fresh design and the shipped one that counts as none
NUDGES = 64  # how often the largest tap may be nudged before normalizing gives up


def design(base: int, k: int) -> Floats:
    """Design a decimator for a family at K times its base rate: a Kaiser prototype turned minimum phase."""
    rate = base * k
    numtaps, beta = kaiserord(ATTENUATION_DB, (base / 2 - PASS_HZ) / (rate / 2))
    numtaps += 1 - numtaps % 2  # odd length: a type I prototype
    prototype = firwin(numtaps, (PASS_HZ + base / 2) / 2, window=("kaiser", beta), fs=rate)
    taps: Floats = minimum_phase(prototype, method="homomorphic", n_fft=N_FFT, half=False)
    return normalize(taps)


def normalize(taps: Floats) -> Floats:
    """Give the taps unit DC gain: nudge the largest tap until the exactly rounded sum is 1."""
    h = taps / math.fsum(taps)
    k = int(np.argmax(np.abs(h)))
    for _ in range(NUDGES):
        residual = 1.0 - math.fsum(h)
        if residual == 0.0:
            return h
        h[k] += residual
    msg = "could not normalize the taps"
    raise RuntimeError(msg)


def measure(h: Floats, base: int, k: int) -> dict[str, float]:
    """Measure a decimator: its passband ripple, stopband, length and the time holding 99% of its energy."""
    rate = base * k
    n = 2**23
    gain = np.abs(np.fft.rfft(h, n))
    f = np.arange(gain.size) * rate / n
    energy = np.cumsum(h**2) / np.sum(h**2)
    return {
        "passband_db": float(np.max(np.abs(20 * np.log10(gain[f <= PASS_HZ])))),
        "stopband_db": float(-20 * np.log10(np.max(gain[f >= base / 2]))),
        "length_ms": 1000 * h.size / rate,
        "energy99_ms": 1000 * int(np.searchsorted(energy, 0.99)) / rate,
    }


def checksum(h: Floats) -> str:
    """Return the SHA-256 of taps as little-endian float64, as they ship."""
    return hashlib.sha256(np.asarray(h, dtype="<f8").tobytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Design every shipped decimator, and write it with its checksum, or compare it with the shipped one."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare a fresh design with the shipped data")
    args = ap.parse_args(argv)
    filters: dict[str, dict[str, Any]] = {}
    manifest = {
        "design": {
            "method": "Kaiser linear-phase prototype, minimum_phase(method='homomorphic', half=False), "
            "taps renormalized to sum to 1",
            "passband_hz": PASS_HZ,
            "attenuation_db": ATTENUATION_DB,
            "n_fft": N_FFT,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "filters": filters,
    }
    status = 0
    for k in SHIPPED_K:
        for base in BASES:
            h = design(base, k)
            name = f"{base}x{k}"
            entry = {
                "base": base,
                "K": k,
                "rate": base * k,
                "taps": int(h.size),
                "file": f"decimator-{name}.f64",
                "sha256": checksum(h),
                **measure(h, base, k),
            }
            filters[name] = entry
            sys.stdout.write(
                f"{name}: {h.size} taps, {entry['length_ms']:.2f} ms, 99% energy in {entry['energy99_ms']:.2f} ms, "
                f"passband {entry['passband_db']:.1e} dB, stopband {entry['stopband_db']:.1f} dB\n",
            )
            if args.check:
                shipped = np.fromfile(DATA / f"decimator-{name}.f64", dtype="<f8")
                diff = np.max(np.abs(shipped - h)) if shipped.size == h.size else math.inf
                sys.stdout.write(f"  shipped vs fresh design: max difference {diff:.1e}\n")
                status |= int(diff > SAME)
            else:
                h.astype("<f8").tofile(DATA / f"decimator-{name}.f64")
    if not args.check:
        (DATA / "decimators.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
