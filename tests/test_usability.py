# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The usability tasks of plan §8.4, from each reference patch.

Each task takes at most three field edits or one command, and must move the
sound in the expected direction, judged here by a simple measurement.
Measurements fade their windows out rather than cutting them, so the onset
stays and the cut does not leak. Listening decides the rest; this checks that
the edits are short and predictable.
"""

import dataclasses
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from brightwork_synth import patch
from brightwork_synth.cli import REFERENCES, reference_text
from brightwork_synth.loudness import momentary_max
from brightwork_synth.render import render, render_takes

from .conftest import edit as set_path

type Floats = NDArray[np.float64]

RATE = 48000
EDITS = 3  # field edits a task may take, at most
TAKES = 4  # a rattle's measurements average this many takes
HEADROOM_DB = 3.0  # of drive headroom, at least, on every take of every reference
MATCHED_DB = 1.0  # the references' loudnesses lie within this of each other
CHOKED_S = 0.3  # the closed open hat lasts less than this
BEAT_MS = 250.0  # where the foot closes it
FALL_MS = 297.0  # and where its 60 dB fall has ended, by then


def ref(name: str) -> patch.Patch:
    """Return a reference patch."""
    return patch.loads(reference_text(name))


def edit(p: patch.Patch, **fields: object) -> patch.Patch:
    """Set at most three fields by patch path, such as "noise[2].level_db"; the caller counts them."""
    assert len(fields) <= EDITS
    d = patch.to_dict(p)
    for path, value in fields.items():
        set_path(d, path, value)
    return patch.from_dict(d)


def audio(p: patch.Patch) -> Floats:
    """Render a patch, which must stay inside the validated domain and under full scale, as floats."""
    a, report = render(p)
    assert report.inside, str(report)
    assert not report.over, str(report)
    return a.astype(float)


def faded(x: Floats, start_ms: float, length_ms: float) -> Floats:
    """Cut a window from start_ms, flat for its first half and faded out over the second."""
    a, n = int(start_ms / 1000 * RATE), int(length_ms / 1000 * RATE)
    seg = np.zeros(n)
    part = x[a : a + n]
    seg[: part.size] = part
    fade = np.ones(n)
    fade[n // 2 :] = 0.5 * (1 + np.cos(np.pi * np.arange(n - n // 2) / (n - n // 2)))
    return seg * fade


def spectrum(seg: Floats) -> tuple[Floats, Floats]:
    """Return a segment's power spectrum and its frequencies."""
    power = np.abs(np.fft.rfft(seg, 1 << 17)) ** 2
    return np.fft.rfftfreq(1 << 17, 1 / RATE), power


def centroid(x: Floats, start_ms: float = 0.0, length_ms: float = 60.0) -> float:
    """Return the spectral centroid of a faded window: brightness."""
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(f * power) / np.sum(power))


def band_energy(x: Floats, lo: float, hi: float, start_ms: float = 0.0, length_ms: float = 60.0) -> float:
    """Return a faded window's energy between lo and hi Hz."""
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(power[(f >= lo) & (f < hi)]))


def decay_ms(x: Floats, drop_db: float = 40) -> float:
    """Return when the 5 ms RMS envelope last sits within drop_db of its peak."""
    w = RATE // 200
    rms = np.sqrt(np.convolve(x**2, np.ones(w) / w, mode="same"))
    return 1000 * float(np.flatnonzero(rms >= rms.max() * 10 ** (-drop_db / 20))[-1]) / RATE


def modulation(
    x: Floats,
    lo: float = 20.0,
    hi: float = 300.0,
    start_ms: float = 1.0,
    length_ms: float = 200.0,
) -> float:
    """Measure the envelope's fluctuation between lo and hi Hz, relative to its smooth course: roughness."""
    seg = x[int(start_ms / 1000 * RATE) : int((start_ms + length_ms) / 1000 * RATE)]
    hop = RATE // 2000  # a 0.5 ms envelope
    env = np.sqrt(np.convolve(seg**2, np.ones(hop) / hop, mode="same"))[::hop]
    smooth = np.convolve(env, np.ones(40) / 40, mode="same")  # 20 ms
    ripple = env / np.maximum(smooth, 1e-12) - 1.0
    f = np.fft.rfftfreq(ripple.size, hop / RATE)
    power = np.abs(np.fft.rfft(ripple * np.hanning(ripple.size))) ** 2
    return float(np.sqrt(np.sum(power[(f >= lo) & (f < hi)]) / ripple.size))


def loudness(x: Floats) -> float:
    """Return the largest momentary loudness."""
    return momentary_max(x, RATE)


def log_shift_semitones(a: Floats, b: Floats, lo: float = 6000.0, hi: float = 16000.0) -> float:
    """Find the shift, in semitones, that best aligns b's spectrum with a's over lo … hi, in log frequency."""
    grid = np.geomspace(lo, hi, 4000)
    step = math.log2(hi / lo) / 3999 * 12
    curves = []
    for x in (a, b):
        f, power = spectrum(faded(x, 0.0, 60.0))
        curves.append(np.log(np.interp(grid, f, power) + 1e-30))
    ca, cb = (c - c.mean() for c in curves)
    lags = range(-1200, 1201)  # ±5 semitones
    score = [np.dot(ca[max(0, -k) : ca.size - max(0, k)], cb[max(0, k) : cb.size - max(0, -k)]) for k in lags]
    return float(lags[int(np.argmax(score))] * step)


def peak_hz(x: Floats, near: float, start_ms: float = 5.0, length_ms: float = 400.0) -> float:
    """Return the frequency of the strongest partial near `near`, refined by a parabola."""
    f, power = spectrum(faded(x, start_ms, length_ms) * np.hanning(int(length_ms / 1000 * RATE)))
    sel = (f > near / 1.06) & (f < near * 1.06)
    i = np.flatnonzero(sel)[np.argmax(power[sel])]
    a, b, c = np.log(power[i - 1 : i + 2])
    return float(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


# The references themselves (tools.references measures seeds 1–4 × takes 1–8; this checks the first take).


def test_the_references_are_matched_inside_and_combine_into_kits() -> None:
    """Find every reference inside the domain with 3 dB of headroom, under −1 dBFS, matched within 1 dB."""
    loud = {}
    for name in REFERENCES:
        _, report = render_takes(ref(name), 4)
        assert report.inside, name
        assert min(t.headroom.db if t.headroom else -math.inf for t in report.takes) >= HEADROOM_DB, name
        assert report.peak < 10 ** (-1 / 20), name
        loud[name] = sum(f.loudness for f in report.files) / len(report.files)  # matched over takes
    assert max(loud.values()) - min(loud.values()) <= MATCHED_DB
    assert len({ref(name).output for name in REFERENCES}) == 1  # one output setting: any of them make one kit


# From the closed hat.


def test_tighten_and_loosen_the_closed_hat() -> None:
    """Tighten and loosen the closed hat with one command each."""
    p = ref("closed")
    base = decay_ms(audio(p))
    assert decay_ms(audio(patch.stretch(p, "all", 0.6))) < 0.8 * base  # one command
    assert decay_ms(audio(patch.stretch(p, "all", 1.6))) > 1.25 * base  # one command


def test_darken_and_brighten_the_closed_hat() -> None:
    """Darken and brighten the closed hat by its bands."""
    p = ref("closed")
    base = centroid(audio(p))
    darker = edit(p, **{"noise[1].band_hz": [5000.0, 11000.0], "noise[2].band_hz": [3000.0, 11000.0]})
    brighter = edit(p, **{"noise[1].band_hz": [8000.0, 18000.0]})
    assert centroid(audio(darker)) < 0.9 * base
    assert centroid(audio(brighter)) > 1.05 * base


def test_add_stick_to_the_closed_hat() -> None:
    """Add stick to the closed hat with one field: the tick's level."""
    p = ref("closed")
    louder = edit(p, **{"noise[2].level_db": p.noise[1].level_db + 6})  # one field: the tick
    # The tick reaches lower than the plates, so it shows below 6 kHz in the first milliseconds.
    assert band_energy(audio(louder), 2000, 6000, 0, 2) > 1.25 * band_energy(audio(p), 2000, 6000, 0, 2)


def test_make_a_ghost_note() -> None:
    """Make a ghost note with one command: 18 dB quieter, and darker by the contact rule."""
    p = ref("closed")
    ghost = patch.softer(p, 18)  # one command
    x, g = audio(p), audio(ghost)
    assert loudness(g) == pytest.approx(loudness(x) - 18, abs=1.0)
    assert centroid(g) < centroid(x)  # the contact rule: a softer strike excites less high end


# From the open hat.


def test_shorten_and_half_open_the_open_hat() -> None:
    """Shorten the open hat with one command, and half-open it with two fields."""
    p = ref("open")
    x = audio(p)
    assert decay_ms(audio(patch.stretch(p, "all", 0.6))) < 0.8 * decay_ms(x)  # one command
    # Two fields: the plates touch, so they slap more and damp each other over the whole ring.
    half = edit(p, **{"noise[1].chatter": 0.6, "damper.amp": [{"ms": 0.0, "gain": 1.0}, {"ms": 700.0, "gain": 0.0}]})
    h = audio(half)
    assert decay_ms(h) < decay_ms(x)
    assert modulation(h) > 1.2 * modulation(x)


def test_close_the_open_hat_on_a_beat() -> None:
    """Close the open hat on a beat with one command: it rings until the beat, then falls in 47 ms."""
    p = ref("open")
    closed = patch.choke(p, 250.0)  # one command: closed by the foot on the beat at 250 ms
    c = audio(closed)
    assert closed.length_ms == pytest.approx(250.0 + 40.0 * 7 / 6, abs=0.01)
    assert c.size / RATE < CHOKED_S
    assert decay_ms(c, 60) < FALL_MS
    assert decay_ms(c, 60) > BEAT_MS  # and it rang until the beat


def test_darken_the_open_hat() -> None:
    """Darken the open hat by its bands."""
    p = ref("open")
    darker = edit(p, **{"noise[1].band_hz": [5000.0, 11000.0], "noise[2].band_hz": [3000.0, 11000.0]})
    assert centroid(audio(darker), 0, 200) < 0.9 * centroid(audio(p), 0, 200)


# From the pedal.


def test_shorten_and_darken_the_pedal() -> None:
    """Shorten the pedal with one command, and darken it by its bands."""
    p = ref("pedal")
    x = audio(p)
    assert decay_ms(audio(patch.stretch(p, "all", 0.6))) < 0.8 * decay_ms(x)  # one command
    darker = edit(p, **{"noise[1].band_hz": [2500.0, 8000.0], "noise[2].band_hz": [900.0, 4000.0]})
    assert centroid(audio(darker)) < 0.9 * centroid(x)


# From the crash.


def test_lengthen_and_choke_the_crash() -> None:
    """Lengthen the crash and choke it, with one command each."""
    p = ref("crash")
    x = audio(p)
    assert decay_ms(audio(patch.stretch(p, "all", 1.5)), 50) > 1.25 * decay_ms(x, 50)  # one command
    choked = patch.choke(p, 450.0)  # one command
    assert choked.length_ms == pytest.approx(450.0 + 40.0 * 7 / 6, abs=0.01)
    assert decay_ms(audio(choked), 50) < 0.6 * decay_ms(x, 50)


def test_darken_the_crash_tail() -> None:
    """Darken the crash's tail with one field, its strike unchanged."""
    p = ref("crash")
    darker = edit(p, **{"noise[1].damping": 1.4})  # one field: the highs die sooner
    x, d = audio(p), audio(darker)
    assert centroid(d, 1000, 1000) < 0.9 * centroid(x, 1000, 1000)
    assert centroid(d, 0, 20) == pytest.approx(centroid(x, 0, 20), rel=0.1)  # the strike itself stays


def test_make_the_crash_bloom_more() -> None:
    """Make the crash bloom more with one field."""
    p = ref("crash")
    more = edit(p, **{"noise[1].bloom": 1.0})  # one field

    def build_up(x: Floats) -> float:  # the highs after the strike against the highs at it
        return band_energy(x, 8000, 18000, 20, 60) / band_energy(x, 8000, 18000, 0, 4)

    assert build_up(audio(more)) > 1.5 * build_up(audio(p))


def test_make_a_splash_and_a_china_from_the_crash() -> None:
    """Make a splash from the crash with two fields, and a china with three."""
    p = ref("crash")
    x = audio(p)
    # Brighter and shorter (two fields): a higher band, and a damper that fades the whole ring by 1.2 s.
    splash = edit(
        p,
        **{
            "noise[1].band_hz": [1200.0, 18000.0],
            "damper.amp": [{"ms": 200.0, "gain": 1.0}, {"ms": 1200.0, "gain": 0.0}],
        },
    )
    s = audio(splash)
    assert centroid(s, 0, 50) > 1.05 * centroid(x, 0, 50)
    assert decay_ms(s, 50) < 0.8 * decay_ms(x, 50)
    # Darker, shorter and rougher (three fields): a lower band, the plates slapping at chatter_hz, 220 Hz, and a
    # damper over the ring. A darker band alone lengthens the measured decay, since what remains is the slow low end.
    damper = [{"ms": 150.0, "gain": 1.0}, {"ms": 1500.0, "gain": 0.0}]
    china = edit(p, **{"noise[1].band_hz": [400.0, 9000.0], "noise[1].chatter": 0.7, "damper.amp": damper})
    c = audio(china)
    assert centroid(c, 0, 200) < 0.9 * centroid(x, 0, 200)
    assert decay_ms(c, 50) < 0.8 * decay_ms(x, 50)
    assert modulation(c) > 1.4 * modulation(x)  # over the noise envelope's own fluctuation


# From the ride.


def test_add_ping_and_wash_to_the_ride() -> None:
    """Add ping and wash to the ride with one field each."""
    p = ref("ride")
    x = audio(p)
    ping = edit(p, **{"noise[2].level_db": p.noise[1].level_db + 6})  # one field: the stick's definition
    wash = edit(p, **{"noise[1].level_db": p.noise[0].level_db + 6})  # one field: the cymbal's spread
    assert band_energy(audio(ping), 3000, 9000, 0, 30) > 1.5 * band_energy(x, 3000, 9000, 0, 30)
    assert band_energy(audio(wash), 500, 16000, 1000, 1000) > 2 * band_energy(x, 500, 16000, 1000, 1000)


def test_darken_the_ride() -> None:
    """Darken the ride by its bands."""
    p = ref("ride")
    darker = edit(
        p,
        **{
            "noise[1].band_hz": [500.0, 9000.0],
            "noise[2].band_hz": [3000.0, 7000.0],
            "noise[3].band_hz": [2000.0, 9000.0],
        },
    )
    assert centroid(audio(darker), 0, 500) < 0.85 * centroid(audio(p), 0, 500)


# From the bell.


def test_retune_and_shorten_the_bell() -> None:
    """Retune the bell to D5 and shorten it, with one command each."""
    p = ref("bell")
    x = audio(p)
    lowest = p.tone[0].partials[0][0]
    assert peak_hz(x, lowest) == pytest.approx(lowest, abs=0.5)
    d5 = patch.pitch(p, "D5")  # one command
    assert peak_hz(audio(d5), 587.33) == pytest.approx(587.33, abs=0.5)
    assert decay_ms(audio(patch.stretch(p, "all", 0.6)), 50) < 0.8 * decay_ms(x, 50)  # one command


# From the analog hat.


def test_open_the_analog_hat_and_turn_it_into_its_cymbal() -> None:
    """Open the analog hat with one command, and turn it into its cymbal with three fields."""
    p = ref("analog")
    x = audio(p)
    opened = patch.stretch(p, "all", 8.0)  # one command
    assert decay_ms(audio(opened)) > 4 * decay_ms(x)
    # Its cymbal: the same oscillators through a lower band, ringing for a second (three fields).
    cymbal = edit(p, **{"tone[1].band_hz": [3500.0, 18000.0], "tone[1].amp[4].ms": 300.0, "tone[1].amp[5].ms": 1200.0})
    c = audio(cymbal)
    assert decay_ms(c) > 8 * decay_ms(x)
    assert centroid(c) < 0.9 * centroid(x)


def test_retune_and_darken_the_analog_hat() -> None:
    """Retune the analog hat 3 semitones with one command, and darken it with one field."""
    p = ref("analog")
    x = audio(p)
    up = patch.pitch(p, "+3")  # one command
    assert log_shift_semitones(x, audio(up)) == pytest.approx(3.0, abs=0.2)
    darker = edit(p, **{"tone[1].band_hz": [6000.0, 11000.0]})  # one field
    assert centroid(audio(darker)) < 0.9 * centroid(x)


# From the shaker.


def granularity(p: patch.Patch, start_ms: float = 10.0, end_ms: float = 100.0) -> float:
    """Measure how granular a rattle sounds: its 2 ms envelope's fluctuation about its 20 ms one, over four takes."""

    def envelope(x: Floats, ms: float) -> Floats:
        n = round(ms * RATE / 1000)
        out: Floats = np.sqrt(np.convolve(x * x, np.ones(n) / n, "same"))
        return out

    a, b = round(start_ms * RATE / 1000), round(end_ms * RATE / 1000)
    found = []
    for take in range(1, TAKES + 1):
        x = audio(dataclasses.replace(p, take=take))
        fine, coarse = envelope(x, 2)[a:b], envelope(x, 20)[a:b]
        sounding = coarse > 0  # a sparse rattle can fall silent between collisions
        found.append(float(np.std(fine[sounding] / coarse[sounding])))
    return float(np.mean(found))


def test_loosen_and_smooth_the_shaker() -> None:
    """Loosen the shaker's grains with two fields, and smooth them with two."""
    p = ref("shaker")
    base = granularity(p)
    looser = edit(p, **{"noise[1].grains_hz": 600.0, "noise[1].grain_ms": 4.0})
    smoother = edit(p, **{"noise[1].grains_hz": 4000.0, "noise[2].level_db": p.noise[1].level_db - 12})
    assert granularity(looser) > 1.2 * base
    assert granularity(smoother) < 0.9 * base


def test_lengthen_the_stroke_and_brighten_the_shaker() -> None:
    """Lengthen the stroke with one command, and brighten the shell with one field."""
    p = ref("shaker")
    x = audio(p)
    assert decay_ms(audio(patch.stretch(p, "all", 1.6))) > 1.25 * decay_ms(x)  # one command
    brighter = edit(p, **{"noise[1].band_hz": [4000.0, 16000.0]})  # one field: a cabasa's
    assert centroid(audio(brighter)) > 1.05 * centroid(x)


# From the tambourine.


def test_shake_the_tambourine_longer_and_choke_it() -> None:
    """Shake the tambourine longer with one command, and choke it with another."""
    p = ref("tambourine")
    x = audio(p)
    assert decay_ms(audio(patch.stretch(p, "noise[2].amp", 2.0))) > 1.25 * decay_ms(x)  # one command: the shiver
    choked = audio(patch.choke(p, 60.0, release=20.0))  # one command
    assert decay_ms(choked) < 0.6 * decay_ms(x)


def test_thin_the_tambourine_shiver() -> None:
    """Thin the shiver to fewer, drier jingle collisions with two fields."""
    p = ref("tambourine")
    thinner = edit(p, **{"noise[2].grains_hz": 120.0, "noise[2].grain_ms": 15.0})
    assert granularity(thinner, 30, 200) > 1.2 * granularity(p, 30, 200)
