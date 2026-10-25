"""The usability tasks of plan §8.4, from each reference patch.

Each task takes at most three field edits or one command, and must move the
sound in the expected direction, judged here by a simple measurement.
Measurements fade their windows out rather than cutting them, so the onset
stays and the cut does not leak. Listening decides the rest; this checks that
the edits are short and predictable.
"""

import math

import numpy as np
import pytest
from conftest import edit as set_path

from brightwork_synth import patch as P
from brightwork_synth.cli import reference_text
from brightwork_synth.loudness import momentary_max
from brightwork_synth.render import render

RATE = 48000


def ref(name):
    return P.loads(reference_text(name))


def edit(p, **fields):
    """Field edits by patch path, such as edit(p, **{"noise[2].level_db": -3.0}); the caller counts them."""
    assert len(fields) <= 3
    d = P.to_dict(p)
    for path, value in fields.items():
        set_path(d, path, value)
    return P.from_dict(d)


def audio(p):
    a, report = render(p)
    assert report.inside and not report.over, str(report)
    return a.astype(float)


def faded(x, start_ms, length_ms):
    """A window from start_ms, flat for its first half and faded out over the second."""
    a, n = int(start_ms / 1000 * RATE), int(length_ms / 1000 * RATE)
    seg = np.zeros(n)
    part = x[a : a + n]
    seg[: part.size] = part
    fade = np.ones(n)
    fade[n // 2 :] = 0.5 * (1 + np.cos(np.pi * np.arange(n - n // 2) / (n - n // 2)))
    return seg * fade


def spectrum(seg):
    power = np.abs(np.fft.rfft(seg, 1 << 17)) ** 2
    return np.fft.rfftfreq(1 << 17, 1 / RATE), power


def centroid(x, start_ms=0.0, length_ms=60.0):
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(f * power) / np.sum(power))


def band_energy(x, lo, hi, start_ms=0.0, length_ms=60.0):
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(power[(f >= lo) & (f < hi)]))


def decay_ms(x, drop_db=40):
    """When the 5 ms RMS envelope last sits within drop_db of its peak."""
    w = RATE // 200
    rms = np.sqrt(np.convolve(x**2, np.ones(w) / w, mode="same"))
    return 1000 * np.flatnonzero(rms >= rms.max() * 10 ** (-drop_db / 20))[-1] / RATE


def modulation(x, lo=20.0, hi=300.0, start_ms=1.0, length_ms=200.0):
    """The envelope's fluctuation between lo and hi Hz, relative to its smooth course: roughness."""
    seg = x[int(start_ms / 1000 * RATE) : int((start_ms + length_ms) / 1000 * RATE)]
    hop = RATE // 2000  # a 0.5 ms envelope
    env = np.sqrt(np.convolve(seg**2, np.ones(hop) / hop, mode="same"))[::hop]
    smooth = np.convolve(env, np.ones(40) / 40, mode="same")  # 20 ms
    ripple = env / np.maximum(smooth, 1e-12) - 1.0
    f = np.fft.rfftfreq(ripple.size, hop / RATE)
    power = np.abs(np.fft.rfft(ripple * np.hanning(ripple.size))) ** 2
    return float(np.sqrt(np.sum(power[(f >= lo) & (f < hi)]) / ripple.size))


def loudness(x):
    return momentary_max(x, RATE)


def log_shift_semitones(a, b, lo=6000.0, hi=16000.0):
    """The shift, in semitones, that best aligns b's spectrum with a's over lo … hi, in log frequency."""
    grid = np.geomspace(lo, hi, 4000)
    step = math.log2(hi / lo) / 3999 * 12
    curves = []
    for x in (a, b):
        f, power = spectrum(faded(x, 0.0, 60.0))
        curves.append(np.log(np.interp(grid, f, power) + 1e-30))
    ca, cb = (c - c.mean() for c in curves)
    lags = np.arange(-1200, 1201)  # ±5 semitones
    score = [np.dot(ca[max(0, -k) : ca.size - max(0, k)], cb[max(0, k) : cb.size - max(0, -k)]) for k in lags]
    return float(lags[int(np.argmax(score))] * step)


def peak_hz(x, near, start_ms=5.0, length_ms=400.0):
    f, power = spectrum(faded(x, start_ms, length_ms) * np.hanning(int(length_ms / 1000 * RATE)))
    sel = (f > near / 1.06) & (f < near * 1.06)
    i = np.flatnonzero(sel)[np.argmax(power[sel])]
    a, b, c = np.log(power[i - 1 : i + 2])
    return float(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


# The references themselves (tools/references.py measures seeds 1–4 × takes 1–8; this checks the first take).


def test_the_references_are_matched_inside_and_combine_into_kits():
    from brightwork_synth.cli import REFERENCES
    from brightwork_synth.render import render_takes

    loud = {}
    for name in REFERENCES:
        _, report = render_takes(ref(name), 4)
        assert report.inside and min(t.headroom.db for t in report.takes) >= 3.0, name
        assert report.peak < 10 ** (-1 / 20), name
        loud[name] = sum(f.loudness for f in report.files) / len(report.files)  # matched over takes
    assert max(loud.values()) - min(loud.values()) <= 1.0
    assert len({ref(name).output for name in REFERENCES}) == 1  # one output setting: any of them make one kit


# From the closed hat.


def test_tighten_and_loosen_the_closed_hat():
    p = ref("closed")
    base = decay_ms(audio(p))
    assert decay_ms(audio(P.stretch(p, "all", 0.6))) < 0.8 * base  # one command
    assert decay_ms(audio(P.stretch(p, "all", 1.6))) > 1.25 * base  # one command


def test_darken_and_brighten_the_closed_hat():
    p = ref("closed")
    base = centroid(audio(p))
    darker = edit(p, **{"noise[1].band_hz": [5000.0, 11000.0], "noise[2].band_hz": [3000.0, 11000.0]})
    brighter = edit(p, **{"noise[1].band_hz": [8000.0, 18000.0]})
    assert centroid(audio(darker)) < 0.9 * base
    assert centroid(audio(brighter)) > 1.05 * base


def test_add_stick_to_the_closed_hat():
    p = ref("closed")
    louder = edit(p, **{"noise[2].level_db": p.noise[1].level_db + 6})  # one field: the tick
    # The tick reaches lower than the plates, so it shows below 6 kHz in the first milliseconds.
    assert band_energy(audio(louder), 2000, 6000, 0, 2) > 1.25 * band_energy(audio(p), 2000, 6000, 0, 2)


def test_make_a_ghost_note():
    p = ref("closed")
    ghost = P.softer(p, 18)  # one command
    x, g = audio(p), audio(ghost)
    assert loudness(g) == pytest.approx(loudness(x) - 18, abs=1.0)
    assert centroid(g) < centroid(x)  # the contact rule: a softer strike excites less high end


# From the open hat.


def test_shorten_and_half_open_the_open_hat():
    p = ref("open")
    x = audio(p)
    assert decay_ms(audio(P.stretch(p, "all", 0.6))) < 0.8 * decay_ms(x)  # one command
    # Two fields: the plates touch, so they slap more and damp each other over the whole ring.
    half = edit(p, **{"noise[1].chatter": 0.6, "damper.amp": [{"ms": 0.0, "gain": 1.0}, {"ms": 700.0, "gain": 0.0}]})
    h = audio(half)
    assert decay_ms(h) < decay_ms(x)
    assert modulation(h) > 1.2 * modulation(x)


def test_close_the_open_hat_on_a_beat():
    p = ref("open")
    closed = P.choke(p, 250.0)  # one command: closed by the foot on the beat at 250 ms
    c = audio(closed)
    assert closed.length_ms == pytest.approx(250.0 + 40.0 * 7 / 6, abs=0.01)
    assert c.size / RATE < 0.3 and decay_ms(c, 60) < 297  # silent within the release, the file ends with it
    assert decay_ms(c, 60) > 250  # and it rang until the beat


def test_darken_the_open_hat():
    p = ref("open")
    darker = edit(p, **{"noise[1].band_hz": [5000.0, 11000.0], "noise[2].band_hz": [3000.0, 11000.0]})
    assert centroid(audio(darker), 0, 200) < 0.9 * centroid(audio(p), 0, 200)


# From the pedal.


def test_shorten_and_darken_the_pedal():
    p = ref("pedal")
    x = audio(p)
    assert decay_ms(audio(P.stretch(p, "all", 0.6))) < 0.8 * decay_ms(x)  # one command
    darker = edit(p, **{"noise[1].band_hz": [2500.0, 8000.0], "noise[2].band_hz": [900.0, 4000.0]})
    assert centroid(audio(darker)) < 0.9 * centroid(x)


# From the crash.


def test_lengthen_and_choke_the_crash():
    p = ref("crash")
    x = audio(p)
    assert decay_ms(audio(P.stretch(p, "all", 1.5)), 50) > 1.25 * decay_ms(x, 50)  # one command
    choked = P.choke(p, 450.0)  # one command
    assert choked.length_ms == pytest.approx(450.0 + 40.0 * 7 / 6, abs=0.01)
    assert decay_ms(audio(choked), 50) < 0.6 * decay_ms(x, 50)


def test_darken_the_crash_tail():
    p = ref("crash")
    darker = edit(p, **{"noise[1].damping": 1.4})  # one field: the highs die sooner
    x, d = audio(p), audio(darker)
    assert centroid(d, 1000, 1000) < 0.9 * centroid(x, 1000, 1000)
    assert centroid(d, 0, 20) == pytest.approx(centroid(x, 0, 20), rel=0.1)  # the strike itself stays


def test_make_the_crash_bloom_more():
    p = ref("crash")
    more = edit(p, **{"noise[1].bloom": 1.0})  # one field

    def build_up(x):  # the highs after the strike against the highs at it
        return band_energy(x, 8000, 18000, 20, 60) / band_energy(x, 8000, 18000, 0, 4)

    assert build_up(audio(more)) > 1.5 * build_up(audio(p))


def test_make_a_splash_and_a_china_from_the_crash():
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
    assert centroid(s, 0, 50) > 1.05 * centroid(x, 0, 50) and decay_ms(s, 50) < 0.8 * decay_ms(x, 50)
    # Darker, shorter and rougher (three fields): a lower band, the plates slapping at chatter_hz, 220 Hz, and a
    # damper over the ring. A darker band alone lengthens the measured decay, since what remains is the slow low end.
    damper = [{"ms": 150.0, "gain": 1.0}, {"ms": 1500.0, "gain": 0.0}]
    china = edit(p, **{"noise[1].band_hz": [400.0, 9000.0], "noise[1].chatter": 0.7, "damper.amp": damper})
    c = audio(china)
    assert centroid(c, 0, 200) < 0.9 * centroid(x, 0, 200)
    assert decay_ms(c, 50) < 0.8 * decay_ms(x, 50)
    assert modulation(c) > 1.4 * modulation(x)  # over the noise envelope's own fluctuation


# From the ride.


def test_add_ping_and_wash_to_the_ride():
    p = ref("ride")
    x = audio(p)
    ping = edit(p, **{"noise[2].level_db": p.noise[1].level_db + 6})  # one field: the stick's definition
    wash = edit(p, **{"noise[1].level_db": p.noise[0].level_db + 6})  # one field: the cymbal's spread
    assert band_energy(audio(ping), 3000, 9000, 0, 30) > 1.5 * band_energy(x, 3000, 9000, 0, 30)
    assert band_energy(audio(wash), 500, 16000, 1000, 1000) > 2 * band_energy(x, 500, 16000, 1000, 1000)


def test_darken_the_ride():
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


def test_retune_and_shorten_the_bell():
    p = ref("bell")
    x = audio(p)
    lowest = p.tone[0].partials[0][0]
    assert peak_hz(x, lowest) == pytest.approx(lowest, abs=0.5)
    d5 = P.pitch(p, "D5")  # one command
    assert peak_hz(audio(d5), 587.33) == pytest.approx(587.33, abs=0.5)
    assert decay_ms(audio(P.stretch(p, "all", 0.6)), 50) < 0.8 * decay_ms(x, 50)  # one command


# From the analog hat.


def test_open_the_analog_hat_and_turn_it_into_its_cymbal():
    p = ref("analog")
    x = audio(p)
    opened = P.stretch(p, "all", 8.0)  # one command
    assert decay_ms(audio(opened)) > 4 * decay_ms(x)
    # Its cymbal: the same oscillators through a lower band, ringing for a second (three fields).
    cymbal = edit(p, **{"tone[1].band_hz": [3500.0, 18000.0], "tone[1].amp[4].ms": 300.0, "tone[1].amp[5].ms": 1200.0})
    c = audio(cymbal)
    assert decay_ms(c) > 8 * decay_ms(x) and centroid(c) < 0.9 * centroid(x)


def test_retune_and_darken_the_analog_hat():
    p = ref("analog")
    x = audio(p)
    up = P.pitch(p, "+3")  # one command
    assert log_shift_semitones(x, audio(up)) == pytest.approx(3.0, abs=0.2)
    darker = edit(p, **{"tone[1].band_hz": [6000.0, 11000.0]})  # one field
    assert centroid(audio(darker)) < 0.9 * centroid(x)
