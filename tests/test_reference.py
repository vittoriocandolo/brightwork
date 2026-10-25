# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference method behind `brightwork verify` (plan §8.2)."""

import math

import numpy as np
import pytest

from brightwork_synth import carrier, reference, render, voice
from brightwork_synth.patch import loads

from .conftest import EXAMPLE, make, make_tone

SHIPPING_ROUNDING = 1e-14  # the matrix products' other partitions, against the shipping signal
LIMIT_DB = -100.0  # the rendering accuracy every fixture inside the domain meets
CONVERGED_DB = -120.0  # how far below that a reference must agree with its check
LEAKAGE_DB = -110.0  # the decimator's own leakage, bounded
DEEP_DB = -150.0  # the reference against a third, higher rate
SHIFT_GAIN = 1000  # how much worse a one-sample shift matches than no shift, at least
GUARD_RINGING = 1e-4  # of the peak: the ideal band-limited signal's ringing left at its guard's start
EDGE_RINGING = 1e-10  # of the peak: and at the very ends of the guards
BOUND_USE = 100  # how far above the measured error the kink bound may lie, at most
REFERENCE_HZ, CHECK_HZ = 6144000, 12288000  # 128 and 256 times 48 kHz


def amp(*pts: tuple[float, float]) -> list[dict[str, float]]:
    """Write knots as TOML parses them."""
    return [{"ms": m, "gain": g} for m, g in pts]


def test_mask_is_flat_then_raised_cosine() -> None:
    """Build the reference's mask: flat up to 0.4 F, falling in a raised cosine, and 0 from F/2 on."""
    rate, m = 1536000, 4
    length = 1 << 16
    mask = reference.mask(length, m * rate, rate)
    f = np.arange(mask.size) * (m * rate / length)
    assert np.all(mask[f <= 0.4 * rate] == 1.0)
    assert np.all(mask[f >= 0.5 * rate] == 0.0)
    band = (f > 0.4 * rate) & (f < 0.5 * rate)
    assert np.all(np.diff(mask[band]) < 0)


def test_the_reference_signal_is_the_shipping_model() -> None:
    """Synthesize any stretch of samples by the reference's model at the shipping rate, and find the shipping signal."""
    p = make_tone(**{"noise[1].chatter": 0.6, "noise[1].bloom": 0.5, "damper.amp": amp((60, 1), (70, 0))})
    v = voice.voice(p)
    rate, ip = 1536000, carrier.interpolator(48000, 16)
    tabs = voice.tables(v, ip.rate)
    count = voice.sample_count(v.t_src, rate)
    whole = voice.synthesize_x(voice.Model(v, ip, tabs), rate, count)
    for n0, n1 in ((-100, 5000), (12345, 99999), (count - 3000, count + 700), (-700, -100), (count + 10, count + 900)):
        got = reference.x_range(voice.Model(v, ip, tabs), rate, n0, n1, count)
        want = np.zeros(n1 - n0)
        a = max(n0, 0)
        b = max(min(n1, count), a)
        want[a - n0 : b - n0] = whole[a:b]
        np.testing.assert_allclose(got, want, rtol=0, atol=SHIPPING_ROUNDING)


def test_reference_rates_are_absolute() -> None:
    """Render references at 128 and 256 times the base rate, with interpolators designed for those rates."""
    v = reference.verify_take(loads(EXAMPLE))
    assert (v.reference_hz, v.check_hz) == (REFERENCE_HZ, CHECK_HZ)
    assert "6.144 and 12.288 MHz" in str(v)
    with pytest.raises(ValueError, match="needs K dividing it"):
        reference.reference(loads(EXAMPLE), 100)  # 100 × base is not a multiple of K
    # The reference interpolators are designed at their own rates, from the same tables.
    ip = carrier.interpolator(48000, reference.REFERENCE // 2)
    assert ip.sha256 is None
    assert ip.rate == 2 * 48000
    assert ip.rate * ip.up == REFERENCE_HZ


def test_example_passes_with_a_converged_reference() -> None:
    """Verify the example: within −100 dB of a reference that agrees with its check, and with a third rate."""
    v = reference.verify(loads(EXAMPLE))
    t = v.takes[0]
    assert v.passed
    assert t.converged
    assert t.error.rel_db <= LIMIT_DB
    assert t.error.max_db <= LIMIT_DB
    assert t.convergence.max_db <= CONVERGED_DB
    assert t.leakage_db <= LEAKAGE_DB
    assert t.domain.inside
    assert "PASS" in str(v)
    # A third rate agrees at least as well.
    p = loads(EXAMPLE)
    deep = reference.compare(reference.reference(p, reference.CHECK).out, reference.reference(p, reference.DEEP).out)
    assert deep.max_db <= DEEP_DB


def test_reference_is_aligned_with_the_shipping_output() -> None:
    """Align the reference with the shipping output, with no shift, its ringing dead within its guard."""
    p = make_tone()
    ship = render.synthesize(p).out
    ref = reference.reference(p)
    assert ref.out.size == ship.size
    # Same time origin: no shift improves the match.
    errs = [np.max(np.abs(np.roll(ref.out, s) - ship)) for s in (-1, 0, 1)]
    assert errs[1] < min(errs[0], errs[2]) / SHIFT_GAIN
    # The band-limited signal rings a little before the strike (it is ideal, not causal),
    # and the guard is wide enough for that ringing to die out.
    peak = np.max(np.abs(ref.internal))
    assert np.max(np.abs(ref.internal[: reference.GUARD])) <= GUARD_RINGING * peak
    assert np.max(np.abs(ref.internal[:64])) <= EDGE_RINGING * peak
    assert np.max(np.abs(ref.internal[-64:])) <= EDGE_RINGING * peak


def test_aliasing_is_caught_and_predicted() -> None:
    """Drive loud hat noise by 24 dB, and find it aliasing beyond −100 dB, as the domain rule predicts."""
    v = reference.verify_take(make(**{"noise[1].level_db": 6.0, "drive.db": 24.0}))
    assert v.converged
    assert not v.passed
    assert v.error.max_db > LIMIT_DB
    assert not v.domain.inside


@pytest.mark.parametrize("rise_ms", [0.01, 0.02, 0.05])
def test_kink_bound_is_an_upper_bound(rise_ms: float) -> None:
    """Measure a fast tick's aliasing below the kink bound, and within 40 dB of it."""
    p = make(
        **{
            "noise[1].level_db": -math.inf,
            "noise[2].level_db": 0.0,
            "noise[2].band_hz": [1000.0, 20000.0],
            "noise[2].amp": amp((0, 0), (rise_ms, 1), (rise_ms + 0.1, 0.7), (2, 0.1), (6, 0)),
        },
    )
    v = reference.verify_take(p)
    bound = render.synthesize(p).kink
    assert v.error.max_abs <= bound
    assert v.error.max_abs >= bound / BOUND_USE  # and a useful one


def test_the_kink_bound_covers_mapped_knots_chatter_and_the_damper() -> None:
    """Measure aliasing below the kink bound with sub-bands, chatter and a damper's knots 10 µs apart."""
    p = make_tone(
        **{
            "noise[1].damping": 2.0,
            "noise[1].bloom": 1.0,
            "noise[1].chatter": 1.0,
            "noise[1].chatter_hz": 3000.0,
            "noise[1].level_db": 6.0,
            "damper.amp": amp((20, 1), (20.01, 0.5), (20.02, 0)),
        },
    )
    v = reference.verify_take(p)
    assert v.converged
    assert v.error.max_abs <= render.synthesize(p).kink


def test_verify_judges_every_take() -> None:
    """Verify two takes, each on its own."""
    v = reference.verify(make(**{"noise[2].level_db": 0.0, "drive.db": 12.0}), takes=2)
    assert [t.take for t in v.takes] == [1, 2]
    assert "take 2" in str(v)
    assert "2 of 2 takes pass" in str(v)


def test_compare() -> None:
    """Compare renders relatively, or absolutely when the reference is near silent."""
    ref = np.array([0.0, 1.0, -0.5])
    e = reference.compare(ref + np.array([0.0, 1e-6, 0.0]), ref)
    assert e.max_db == pytest.approx(-120.0)
    assert e.passes()
    assert not reference.compare(ref + 1e-4, ref).passes()
    silent = reference.compare(np.zeros(3), np.zeros(3))
    assert silent.near_silent
    assert silent.passes()
    assert silent.max_abs == 0
    quiet = np.array([0.0, 1e-3, 0.0])
    assert reference.compare(quiet + 5e-8, quiet).passes()  # near-silent: the absolute bound applies
    assert not reference.compare(quiet + 5e-7, quiet).passes()


def test_a_near_silent_take_is_judged_absolutely() -> None:
    """Judge a take whose reference peaks below 1e-2 by the absolute bound, and say so."""
    quiet = make(**{"noise[1].level_db": -50.0, "noise[2].level_db": -50.0, "output.db": 0.0})
    v = reference.verify_take(quiet)
    assert v.error.near_silent
    assert v.passed
    text = str(v)
    assert "near-silent: limit 1e-07" in text
    assert "agree to" in text


def test_the_shipping_taps_given_explicitly_change_nothing() -> None:
    """Render and reference alike whether the shipping decimator is given or looked up."""
    p = loads(EXAMPLE)
    taps = render.decimator(48000).taps
    assert np.array_equal(render.synthesize(p, taps=taps).out, render.synthesize(p).out)
    given = reference.reference(p, reference.REFERENCE, taps=taps)
    assert np.array_equal(given.out, reference.reference(p, reference.REFERENCE).out)
