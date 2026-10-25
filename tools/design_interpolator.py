# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design the frozen interpolators (plan §3) and write them to src/brightwork_synth/data.

The renderer never designs its interpolators: it loads these coefficients and
checks their SHA-256. Run this only to change the rendering policy.

    python tools/design_interpolator.py            # write the shipped designs
    python tools/design_interpolator.py --check    # compare a fresh design with the shipped data

Design: a Kaiser-windowed sinc at 280 dB, cut off at R/2 for the table rate
R = 2 × base, flat to 20 kHz and stopped from R − 20 kHz, sized by Kaiser's
formula and rounded up to a whole number of taps per phase. The accuracy is
measured against direct summation of the lines with exactly reduced integer
phases, on three noises, and recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brightwork_synth import carrier, spectrum

BASES = (44100, 48000)
SHIPPED_K = (16, 32, 64)
DATA = Path(__file__).resolve().parent.parent / "src" / "brightwork_synth" / "data"
BANDS = ((20.0, 20000.0), (6000.0, 16000.0), (300.0, 18000.0))
INSTANTS = 400


def checksum(h: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(h, dtype="<f8").tobytes()).hexdigest()


def exact(bins: np.ndarray, amps: np.ndarray, n: np.ndarray, rate: int) -> np.ndarray:
    """Σ |A_k| cos(2π k n / (8·rate) + arg A_k), with the phase reduced exactly in integers."""
    period = spectrum.PERIOD * rate
    out = np.zeros(n.size)
    for i0 in range(0, bins.size, 4096):
        k = bins[i0 : i0 + 4096].astype(np.int64)
        cycles = np.mod(np.outer(n.astype(np.int64), k), period) / period
        a = amps[i0 : i0 + 4096]
        out += np.cos(2 * np.pi * cycles + np.angle(a)) @ np.abs(a)
    return out


def accuracy(ip: carrier.Interpolator, seed: int = 1) -> float:
    """The largest error relative to the carrier's RMS, at random internal samples, over three noises."""
    rate = ip.rate * ip.up
    rng = np.random.default_rng(seed)
    worst = 0.0
    for band in BANDS:
        lines = spectrum.noise_lines(seed, 1, 0.0)
        amps = spectrum.calibrated(lines.bins, lines.amps, band, random=True)
        tab = carrier.table(lines.bins, amps, ip.rate)
        n = np.sort(rng.integers(0, spectrum.PERIOD * rate, INSTANTS))
        got = np.array(
            [carrier.block(tab, ip, int(i) // carrier.BLOCK * carrier.BLOCK)[int(i) % carrier.BLOCK] for i in n],
        )
        rms = math.sqrt(0.5 * float(np.sum(np.abs(amps) ** 2)))
        worst = max(worst, float(np.max(np.abs(got - exact(lines.bins, amps, n, rate)))) / rms)
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare a fresh design with the shipped data")
    args = ap.parse_args(argv)
    manifest = {
        "design": {
            "method": "Kaiser-windowed sinc cut off at R/2, R = 2 x base, linear phase, gain = factor",
            "passband_hz": carrier.PASS_HZ,
            "attenuation_db": carrier.ATTENUATION_DB,
            "numpy": np.__version__,
        },
        "filters": {},
    }
    status = 0
    for K in SHIPPED_K:
        for base in BASES:
            up = K // 2
            taps, q = carrier.design(base, up)
            name = f"{base}x{K}"
            ip = carrier.Interpolator(base, up, q, taps, carrier.polyphase(taps, q, up), None)
            err = accuracy(ip)
            entry = {
                "base": base,
                "K": K,
                "up": up,
                "q": q,
                "taps": int(taps.size),
                "file": f"interpolator-{name}.f64",
                "sha256": checksum(taps),
                "max_error_of_rms": err,
            }
            manifest["filters"][name] = entry
            print(f"{name}: {taps.size} taps, {2 * q} per phase, error at most {err:.1e} of the carrier's RMS")
            if args.check:
                shipped = np.fromfile(DATA / entry["file"], dtype="<f8")
                diff = np.max(np.abs(shipped - taps)) if shipped.size == taps.size else math.inf
                print(f"  shipped vs fresh design: max difference {diff:.1e}")
                status |= int(diff > 1e-15)
            else:
                taps.astype("<f8").tofile(DATA / entry["file"])
    if not args.check:
        (DATA / "interpolators.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
