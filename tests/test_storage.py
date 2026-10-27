# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Storage: the sources stored at a low rate and bit depth, then played back smoothly and exactly (plan §2.13)."""

import dataclasses
import math

import numpy as np
import pytest

from brightwork_synth import cli, reference, render, storage, voice
from brightwork_synth.patch import Drive, Patch, Storage, choke, loads

from .test_patch import reject

F = 1536000  # the 48 kHz family's internal rate
SIX_BITS = Storage(6, 32000.0, 12000.0)
ROUNDING = 1e-12  # how close the playback's sums come to their exact values
SILENT = 1e-15  # the Gaussian's tails, kept to 8σ
FASTEST = 2  # the smallest divider
ROUNDING_DB = -200.0  # below which an error against the reference is rounding


def stored(values: list[float], divider: int = 48, filter_hz: float = 12000.0) -> storage.Stored:
    """Make a stored sound directly from its values."""
    return storage.Stored(np.array(values, dtype=float), F, divider, storage.sigma(filter_hz), 6)


def closed(settings: Storage = SIX_BITS) -> Patch:
    """Return the closed hat, stored."""
    return dataclasses.replace(loads(cli.reference_text("closed")), storage=settings)


def test_the_storage_rate_is_the_internal_rate_over_the_nearest_whole_divider() -> None:
    """Store at F / D, D the nearest whole number to F / rate_hz, in both families."""
    assert storage.divider(F, 32000.0) == F // 32000
    assert storage.divider(1411200, 32000.0) == round(1411200 / 32000)
    assert storage.divider(F, 1e9) == FASTEST  # never faster than every other internal sample


def test_values_are_rounded_to_their_bits_against_the_peak() -> None:
    """Keep every stored value on a grid of 2^(bits−1) − 1 steps either side of 0, its peak exactly on it."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=48 * 500)
    for bits in (2, 4, 6, 16):
        s = storage.store(x, F, Storage(bits, 32000.0, 12000.0))
        steps = 2 ** (bits - 1) - 1
        codes = s.values / np.max(np.abs(x[::48])) * steps
        np.testing.assert_allclose(codes, np.round(codes), rtol=0, atol=1e-9)
        assert np.max(np.abs(s.values)) == pytest.approx(np.max(np.abs(x[::48])), rel=1e-15)
        assert len(np.unique(np.round(codes))) <= 2 * steps + 1
    silence = storage.store(np.zeros(4800), F, SIX_BITS)
    assert not np.any(silence.values)


def test_the_filter_is_3_db_down_at_its_frequency() -> None:
    """Make the Gaussian's response exp(−(2πfσ)²/2) half power at filter_hz."""
    for hz in (1000.0, 12000.0, 20000.0):
        s = storage.sigma(hz)
        assert math.exp(-((2 * math.pi * hz * s) ** 2) / 2) == pytest.approx(1 / math.sqrt(2), rel=1e-14)


def test_playback_holds_each_value_smoothly_and_only_after_it() -> None:
    """Play a constant as a constant, nothing before the first value, and nothing once the last has rung out."""
    s = stored([1.0] * 200)
    reach = storage.REACH * s.sigma
    y = s.play(F, 0, 300 * 48)
    t = np.arange(y.size) / F
    middle = (t > 2 * reach + 1 / s.rate) & (t < 199 / s.rate)
    np.testing.assert_allclose(y[middle], 1.0, rtol=0, atol=ROUNDING)
    assert abs(y[0]) < SILENT  # a filter answers only after its input
    end = 200 / s.rate + 2 * reach
    assert np.max(np.abs(y[t > end])) < SILENT
    assert np.max(np.abs(s.kernel(np.array([-1e-6, 1 / s.rate + 2 * reach + 1e-6])))) < SILENT
    # Asked only for times after the playback has ended, just after or long after, it is silent.
    rung_out = math.ceil(end * F) + 1
    assert not np.any(s.play(F, rung_out, rung_out + 100))
    assert not np.any(s.play(F, 10**6, 10**6 + 100))


def test_playback_is_the_same_at_every_rate_and_however_it_is_split() -> None:
    """Play at F and at 4F, in pieces or whole, and summed value by value: the same playback to rounding."""
    rng = np.random.default_rng(2)
    s = stored(list(rng.normal(size=300)), filter_hz=8000.0)
    n = 300 * 48
    whole = s.play(F, 0, n)
    np.testing.assert_allclose(s.play(4 * F, 0, 4 * n)[::4], whole, rtol=0, atol=ROUNDING)
    parts = np.concatenate([s.play(F, a, min(a + 997, n)) for a in range(0, n, 997)])
    np.testing.assert_allclose(parts, whole, rtol=0, atol=ROUNDING)
    times = np.arange(0, n, 911) / F
    direct, slope = s.at(times)
    np.testing.assert_allclose(direct, whole[::911], rtol=0, atol=ROUNDING)
    h = 1e-8
    numeric = (s.at(times + h)[0] - s.at(times - h)[0]) / (2 * h)
    np.testing.assert_allclose(slope, numeric, rtol=1e-5, atol=1e-3)
    assert s.play(F, 10, 10).size == 0
    assert s.at(np.zeros(0))[0].size == 0
    with pytest.raises(ValueError, match="whole multiple of the storage rate"):
        s.play(44100, 0, 10)


def test_a_stored_sound_plays_its_stored_values_and_lasts_their_tail() -> None:
    """Render the stored closed hat as its stored values played back, longer by a period and the filter's delay."""
    p = closed()
    plain = loads(cli.reference_text("closed"))
    assert p.length_ms == pytest.approx(plain.length_ms + 1000 * storage.tail_s(SIX_BITS))
    v = voice.voice(p)
    ip = render.interpolator_for(48000)
    tabs = voice.tables(v, ip.rate)
    count = voice.sample_count(v.t_src, F)
    x, _, _, kept = voice.internal(v, F, count, ip, tabs)
    assert kept is not None
    assert kept.divider == F // 32000
    np.testing.assert_array_equal(x, kept.play(F, 0, count))
    raw = voice.synthesize_x(voice.Model(v, ip, tabs), F, count, damped=False)
    np.testing.assert_array_equal(kept.values, storage.store(raw, F, SIX_BITS).values)
    assert voice.voice(plain).storage is None
    text = str(render.render(p)[1])
    assert "storage  6 bits at 32000 Hz (the internal rate / 48)" in text


@pytest.mark.parametrize(("name", "drive", "choked"), [("closed", 18.0, None), ("open", 12.0, 250.0)])
def test_a_stored_sound_is_validated_driven_and_choked(name: str, drive: float, choked: float | None) -> None:
    """Find a stored sound inside the validated domain and as its reference says, its only kinks the damper's."""
    p = dataclasses.replace(loads(cli.reference_text(name)), storage=SIX_BITS, drive=Drive(drive))
    if choked is not None:
        p = choke(p, choked)
    s = render.synthesize(p)
    assert render.domain(s).inside
    assert all(path.startswith("damper.amp[") for path in s.kinks.paths)
    assert (s.kinks.t.size > 0) == (choked is not None)
    v = reference.verify_take(p)
    assert v.passed
    assert v.error.max_db < ROUNDING_DB  # the playback is smooth: only rounding is left


def test_storage_is_validated() -> None:
    """Refuse 1 bit, a rate or a filter out of range, and anything but whole bits."""
    assert any(m.startswith("storage.bits") for m in reject(**{"storage.bits": 1}))
    assert any(m.startswith("storage.bits") for m in reject(**{"storage.bits": 6.0}))
    assert any(m.startswith("storage.rate_hz") for m in reject(**{"storage.rate_hz": 3000.0}))
    assert any(m.startswith("storage.filter_hz") for m in reject(**{"storage.filter_hz": 30000.0}))
    on = loads(cli.reference_text("closed"))
    assert on.storage.bits == 0  # every reference ships without storage
