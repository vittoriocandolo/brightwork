# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The frozen decimators, and decimation by matrix products (plan §3)."""

import hashlib
import math
from importlib import resources

import numpy as np
import pytest
from scipy.signal import upfirdn

from brightwork_synth import carrier, render

from .conftest import make_tone

# (base, K, taps, length in ms, 99% of the energy within, in ms), as plan §3 records them for 32× and the grid's
# 16× and 64× designs repeat.
FILTERS = [
    (44100, 16, 2593, 3.67, 0.85),
    (48000, 16, 1447, 1.88, 0.52),
    (44100, 32, 5183, 3.67, 0.85),
    (48000, 32, 2891, 1.88, 0.52),
    (44100, 64, 10363, 3.67, 0.85),
    (48000, 64, 5781, 1.88, 0.52),
]
PASS_HZ = 20000.0
RIPPLE_DB = 0.001  # the passband's specified flatness
STOPBAND_DB = 115.0  # the stopband, at least, from half the base rate
DESIGN_DB = 116.0  # the Kaiser prototype's attenuation
LEADING = 20  # the peak tap lies within the first twentieth: minimum phase
FROZEN_K = 32  # the internal rate's factor, frozen by the adversarial grid
SUM_ROUNDING = 1e-15  # of a sum of rounded products, against its exactly rounded value


@pytest.mark.parametrize(("base", "k", "taps", "length_ms", "energy_ms"), FILTERS)
def test_frozen_designs(base: int, k: int, taps: int, length_ms: float, energy_ms: float) -> None:
    """Measure each frozen decimator: its taps, length and energy, its passband, stopband, DC gain and minimum phase."""
    d = render.decimator(base, k)
    h = d.taps
    rate = base * k
    assert h.size == taps
    assert 1000 * h.size / rate == pytest.approx(length_ms, abs=0.01)
    energy = np.cumsum(h**2) / np.sum(h**2)
    assert 1000 * np.searchsorted(energy, 0.99) / rate == pytest.approx(energy_ms, abs=0.01)
    n = 2**23
    gain = np.abs(np.fft.rfft(h, n))
    f = np.arange(gain.size) * rate / n
    assert np.max(np.abs(20 * np.log10(gain[f <= PASS_HZ]))) <= RIPPLE_DB
    assert -20 * np.log10(np.max(gain[f >= base / 2])) >= STOPBAND_DB
    assert d.stopband_db >= STOPBAND_DB
    # DC gain exactly 1: the exactly rounded sum of the taps is 1.
    assert math.fsum(h) == 1.0
    # Minimum phase: the energy arrives first, so transients never pre-ring.
    assert np.argmax(np.abs(h)) < h.size // LEADING


# The snare's frozen decimators, by their checksums: plan §3 ships them bit for bit.
SNARE_SHA256 = {
    "44100x16": "0932afce82f960c811e1a3bf74ccc4636a5aefd2161f5d69466b0a522955c7d0",
    "48000x16": "753e10eadb9b6550d7dfc81847d7a3057b578c8d3bfe07278d0b474d7e29b676",
    "44100x32": "7b063d24485ffd4a2802f042ea867c347463022aee2faf69a711bcd2f27639c4",
    "48000x32": "69797574530979e57d55849bcc2bc84907f3d336d670e5b85a579544d6ef5dd0",
    "44100x64": "b6108201a1aebbb83ac1957d730543699e6c44b8cc0efaf823aab9507f6a7aff",
    "48000x64": "caa6ec2ebc9ae36fe62b0ce79efe77507661224283c1852807073b8d036e0acd",
}


def test_the_snare_designs_ship_unchanged() -> None:
    """Find the snare's designs shipped bit for bit, by their checksums, with the design they record."""
    m = render.manifest()
    assert {name: entry["sha256"] for name, entry in m["filters"].items()} == SNARE_SHA256
    for name, sha256 in SNARE_SHA256.items():
        base, k = map(int, name.split("x"))
        assert render.decimator(base, k).sha256 == sha256
    design = m["design"]
    assert design["attenuation_db"] == DESIGN_DB
    assert design["n_fft"] == 2**20
    assert design["passband_hz"] == PASS_HZ


def tampered(raw: bytes) -> bytes:
    """Flip one bit of a filter's bytes."""
    out = bytearray(raw)
    out[100] ^= 1
    return bytes(out)


@pytest.mark.parametrize(("base", "k"), [(44100, 32), (48000, 32), (48000, 16)])
def test_checksum_is_enforced(base: int, k: int) -> None:
    """Load a decimator's taps only when they match their checksum, and keep them read-only."""
    entry = render.manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("brightwork_synth").joinpath("data", entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert np.array_equal(carrier.load_taps(raw, entry, "decimator"), render.decimator(base, k).taps)
    for bad in (tampered(raw), raw[:-8]):
        with pytest.raises(RuntimeError, match="decimator does not match its checksum"):
            carrier.load_taps(bad, entry, "decimator")
    assert not render.decimator(base, k).taps.flags.writeable


@pytest.mark.parametrize("name", list(carrier.manifest()["filters"]))
def test_interpolator_checksums_are_enforced(name: str) -> None:
    """Load a shipped interpolator's taps only when they match their checksum."""
    entry = carrier.manifest()["filters"][name]
    raw = resources.files("brightwork_synth").joinpath("data", entry["file"]).read_bytes()
    base, factor = map(int, name.split("x"))
    assert np.array_equal(carrier.load_taps(raw, entry, name), carrier.interpolator(base, factor // 2).taps)
    with pytest.raises(RuntimeError, match=f"{name} does not match its checksum"):
        carrier.load_taps(tampered(raw), entry, name)


def test_timing_contract_on_synthetic_input() -> None:
    """Decimate any length, and find every output the plan's sum over the inputs its window reaches."""
    h = render.decimator(48000).taps
    taps, factor = h.size, render.K
    rng = np.random.default_rng(2)
    for count in (1, 2, 31, 32, 33, 1000, 12345):
        y = rng.normal(size=count)
        out = upfirdn(h, y, down=factor)
        assert out.size == (count + taps - 2) // factor + 1
        assert out[0] == h[0] * y[0]
        for m in {0, 1, out.size // 2, out.size - 1}:
            k = np.arange(taps)
            n = m * factor - k
            ok = (n >= 0) & (n < count)
            assert out[m] == pytest.approx(math.fsum(h[k[ok]] * y[n[ok]]), abs=SUM_ROUNDING)
        # The last output still sees the last input sample.
        assert (out.size - 1) * factor - (count - 1) <= taps - 1


@pytest.mark.parametrize(("low", "high"), [(48000, 96000), (44100, 88200)])
def test_double_rate_is_the_base_rate_with_samples_filled_in(low: int, high: int) -> None:
    """Render at a double rate, and find the base rate's samples, bit for bit, every second sample."""
    edits = {"drive.db": 12.0, "noise[1].chatter": 0.5, "noise[1].damping": 1.0}
    a, _ = render.render(make_tone(**edits, **{"output.sample_rate": low}))
    b, _ = render.render(make_tone(**edits, **{"output.sample_rate": high}))
    assert a.size == (b.size + 1) // 2
    assert np.array_equal(b[::2], a)


def test_rates() -> None:
    """Derive each export rate's internal rate and factor, and the interpolator that reaches it from the tables."""
    assert render.K == FROZEN_K
    assert render.rates(44100) == (1411200, 32)
    assert render.rates(88200) == (1411200, 16)
    assert render.rates(48000) == (1536000, 32)
    assert render.rates(96000) == (1536000, 16)
    assert render.rates(48000, 16) == (768000, 16)
    # The carriers reach F from their tables at 2 × base.
    for rate in (44100, 48000, 88200, 96000):
        for k in (16, 32, 64):
            ip = render.interpolator_for(rate, k)
            assert ip.rate == 2 * render.FAMILY[rate]
            assert ip.rate * ip.up == render.rates(rate, k)[0]


def test_decimation_agrees_with_upfirdn_within_rounding() -> None:
    """Decimate by matrix products, and find SciPy's upfirdn to a thousandth of their rounding bound."""
    rng = np.random.default_rng(3)
    for base in (44100, 48000):
        for k in (16, 32, 64):
            h = render.decimator(base, k).taps
            for n in (1, 33, 255, 257, 8191, 140001):
                y = rng.normal(size=n)
                for factor in (k, k // 2):
                    out = render.decimate(h, y, factor, k)
                    reference = upfirdn(h, y, down=factor)
                    assert out.shape == reference.shape
                    # Both are sums of at most 2L rounded products: each is within 2L·ε·Σ|h|·max|y| of the exact
                    # value. They agree a thousand times closer than that.
                    scale = np.sum(np.abs(h)) * np.max(np.abs(y))
                    assert np.max(np.abs(out - reference)) <= SUM_ROUNDING * scale


def test_decimation_never_depends_on_how_the_work_is_split_or_on_the_render_length() -> None:
    """Decimate with any number of workers, or a shorter render, and find the same outputs bit for bit."""
    rng = np.random.default_rng(4)
    for base in (44100, 48000):
        h = render.decimator(base).taps
        y = rng.normal(size=600000)
        for factor in (32, 16):
            whole = render.decimate(h, y, factor, 32)
            for workers in (1, 2, 3, 5):
                assert np.array_equal(render.decimate(h, y, factor, 32, workers=workers), whole)
            # A shorter render never changes an output whose window it covers, at any cut, block edges included.
            for n in (1, 257, 32 * render.OUTPUTS, 32 * render.OUTPUTS + 1, 333333):
                m = (n - 1) // factor + 1
                assert np.array_equal(render.decimate(h, y[:n], factor, 32)[:m], whole[:m])


def test_an_output_depends_only_on_the_inputs_in_its_window() -> None:
    """Change one input, and find exactly the outputs whose windows reach it changed, either side of a block edge."""
    rng = np.random.default_rng(5)
    h = render.decimator(44100).taps
    y = rng.normal(size=300001)
    for factor in (32, 16):
        whole = render.decimate(h, y, factor, 32)
        m = np.arange(whole.size) * factor
        for n in (0, 5000, 32 * render.OUTPUTS - 1, 32 * render.OUTPUTS, 299999):  # either side of a block edge
            moved = y.copy()
            moved[n] += 1.0
            out = render.decimate(h, moved, factor, 32)
            window = (m >= n) & (m - n < h.size)
            assert np.array_equal(out[~window], whole[~window])
            assert np.all(out[window] != whole[window])


def test_the_double_rate_interleaves_the_base_rate() -> None:
    """Decimate by half the factor, and find the base rate's outputs every second sample; refuse a bad stride."""
    rng = np.random.default_rng(6)
    for base in (44100, 48000):
        for k in (16, 32, 64):
            h = render.decimator(base, k).taps
            for n in (1, 2, 1000, 140001):
                y = rng.normal(size=n)
                low, high = render.decimate(h, y, k, k), render.decimate(h, y, k // 2, k)
                assert low.size == (high.size + 1) // 2
                assert np.array_equal(high[::2], low)
    with pytest.raises(ValueError, match="does not interleave decimations by 12"):
        render.decimate(render.decimator(48000).taps, np.ones(100), 12, 32)
