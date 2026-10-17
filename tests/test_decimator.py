import hashlib
import math
from importlib import resources

import numpy as np
import pytest
from conftest import make
from scipy.signal import upfirdn

from cymbal import render as R

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


@pytest.mark.parametrize("base, k, taps, length_ms, energy_ms", FILTERS)
def test_frozen_designs(base, k, taps, length_ms, energy_ms):
    d = R.decimator(base, k)
    h = d.taps
    rate = base * k
    assert h.size == taps
    assert 1000 * h.size / rate == pytest.approx(length_ms, abs=0.01)
    energy = np.cumsum(h**2) / np.sum(h**2)
    assert 1000 * np.searchsorted(energy, 0.99) / rate == pytest.approx(energy_ms, abs=0.01)
    n = 2**23
    H = np.abs(np.fft.rfft(h, n))
    f = np.arange(H.size) * rate / n
    assert np.max(np.abs(20 * np.log10(H[f <= 20000]))) <= 0.001
    assert -20 * np.log10(np.max(H[f >= base / 2])) >= 115
    assert d.stopband_db >= 115
    # DC gain exactly 1: the exactly rounded sum of the taps is 1.
    assert math.fsum(h) == 1.0
    # Minimum phase: the energy arrives first, so transients never pre-ring.
    assert np.argmax(np.abs(h)) < h.size // 20


# The snare's frozen decimators, by their checksums: plan §3 ships them bit for bit.
SNARE_SHA256 = {
    "44100x16": "0932afce82f960c811e1a3bf74ccc4636a5aefd2161f5d69466b0a522955c7d0",
    "48000x16": "753e10eadb9b6550d7dfc81847d7a3057b578c8d3bfe07278d0b474d7e29b676",
    "44100x32": "7b063d24485ffd4a2802f042ea867c347463022aee2faf69a711bcd2f27639c4",
    "48000x32": "69797574530979e57d55849bcc2bc84907f3d336d670e5b85a579544d6ef5dd0",
    "44100x64": "b6108201a1aebbb83ac1957d730543699e6c44b8cc0efaf823aab9507f6a7aff",
    "48000x64": "caa6ec2ebc9ae36fe62b0ce79efe77507661224283c1852807073b8d036e0acd",
}


def test_the_snare_designs_ship_unchanged():
    m = R.manifest()
    assert {name: entry["sha256"] for name, entry in m["filters"].items()} == SNARE_SHA256
    for name in SNARE_SHA256:
        base, k = map(int, name.split("x"))
        assert R.decimator(base, k).sha256 == SNARE_SHA256[name]
    design = m["design"]
    assert design["attenuation_db"] == 116.0 and design["n_fft"] == 2**20 and design["passband_hz"] == 20000.0


@pytest.mark.parametrize("base, k", [(44100, 32), (48000, 32), (48000, 16)])
def test_checksum_is_enforced(base, k):
    entry = R.manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("cymbal").joinpath("data", entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert np.array_equal(R.load_taps(raw, entry), R.decimator(base, k).taps)
    tampered = bytearray(raw)
    tampered[100] ^= 1
    with pytest.raises(R.RenderError):
        R.load_taps(bytes(tampered), entry)
    with pytest.raises(R.RenderError):
        R.load_taps(raw[:-8], entry)
    assert not R.decimator(base, k).taps.flags.writeable


def test_timing_contract_on_synthetic_input():
    h = R.decimator(48000).taps
    L, K = h.size, R.K
    rng = np.random.default_rng(2)
    for N in (1, 2, 31, 32, 33, 1000, 12345):
        y = rng.normal(size=N)
        out = upfirdn(h, y, down=K)
        assert out.size == (N + L - 2) // K + 1
        assert out[0] == h[0] * y[0]
        for m in {0, 1, out.size // 2, out.size - 1}:
            k = np.arange(L)
            n = m * K - k
            ok = (n >= 0) & (n < N)
            assert out[m] == pytest.approx(math.fsum(h[k[ok]] * y[n[ok]]), abs=1e-15)
        # The last output still sees the last input sample.
        assert (out.size - 1) * K - (N - 1) <= L - 1


@pytest.mark.parametrize("low, high", [(48000, 96000), (44100, 88200)])
def test_double_rate_is_the_base_rate_with_samples_filled_in(low, high):
    edits = {"drive.db": 12.0, "noise[1].chatter": 0.5, "noise[1].damping": 1.0}
    a, _ = R.render(make(tone=True, **edits, **{"output.sample_rate": low}))
    b, _ = R.render(make(tone=True, **edits, **{"output.sample_rate": high}))
    assert a.size == (b.size + 1) // 2
    assert np.array_equal(b[::2], a)


def test_rates():
    assert R.K == 32
    assert R.rates(44100) == (1411200, 32)
    assert R.rates(88200) == (1411200, 16)
    assert R.rates(48000) == (1536000, 32)
    assert R.rates(96000) == (1536000, 16)
    assert R.rates(48000, 16) == (768000, 16)
    # The carriers reach F from their tables at 2 × base.
    for rate in (44100, 48000, 88200, 96000):
        for k in (16, 32, 64):
            ip = R.interpolator_for(rate, k)
            assert ip.rate == 2 * R.FAMILY[rate] and ip.rate * ip.up == R.rates(rate, k)[0]


def test_decimation_agrees_with_upfirdn_within_rounding():
    rng = np.random.default_rng(3)
    for base in (44100, 48000):
        for k in (16, 32, 64):
            h = R.decimator(base, k).taps
            for n in (1, 33, 255, 257, 8191, 140001):
                y = rng.normal(size=n)
                for factor in (k, k // 2):
                    out = R.decimate(h, y, factor, k)
                    reference = upfirdn(h, y, down=factor)
                    assert out.shape == reference.shape
                    # Both are sums of at most 2L rounded products: each is within 2L·ε·Σ|h|·max|y| of the exact
                    # value. They agree a thousand times closer than that.
                    scale = np.sum(np.abs(h)) * np.max(np.abs(y))
                    assert np.max(np.abs(out - reference)) <= 1e-15 * scale


def test_decimation_never_depends_on_how_the_work_is_split_or_on_the_render_length():
    rng = np.random.default_rng(4)
    for base in (44100, 48000):
        h = R.decimator(base).taps
        y = rng.normal(size=600000)
        for factor in (32, 16):
            whole = R.decimate(h, y, factor, 32)
            for workers in (1, 2, 3, 5):
                assert np.array_equal(R.decimate(h, y, factor, 32, workers=workers), whole)
            # A shorter render never changes an output whose window it covers, at any cut, block edges included.
            for n in (1, 257, 32 * R.OUTPUTS, 32 * R.OUTPUTS + 1, 333333):
                m = (n - 1) // factor + 1
                assert np.array_equal(R.decimate(h, y[:n], factor, 32)[:m], whole[:m])


def test_an_output_depends_only_on_the_inputs_in_its_window():
    rng = np.random.default_rng(5)
    h = R.decimator(44100).taps
    y = rng.normal(size=300001)
    for factor in (32, 16):
        whole = R.decimate(h, y, factor, 32)
        m = np.arange(whole.size) * factor
        for n in (0, 5000, 32 * R.OUTPUTS - 1, 32 * R.OUTPUTS, 299999):  # either side of a block edge
            moved = y.copy()
            moved[n] += 1.0
            out = R.decimate(h, moved, factor, 32)
            window = (m >= n) & (m - n < h.size)
            assert np.array_equal(out[~window], whole[~window])
            assert np.all(out[window] != whole[window])


def test_the_double_rate_interleaves_the_base_rate():
    rng = np.random.default_rng(6)
    for base in (44100, 48000):
        for k in (16, 32, 64):
            h = R.decimator(base, k).taps
            for n in (1, 2, 1000, 140001):
                y = rng.normal(size=n)
                low, high = R.decimate(h, y, k, k), R.decimate(h, y, k // 2, k)
                assert low.size == (high.size + 1) // 2 and np.array_equal(high[::2], low)
    with pytest.raises(ValueError):
        R.decimate(h, y, 12, 32)
