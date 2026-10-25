import math

import numpy as np
import pytest
from conftest import EXAMPLE, make
from scipy.signal import sosfreqz

from brightwork_synth import render as R
from brightwork_synth.loudness import k_weighting, momentary_max
from brightwork_synth.patch import loads

# ITU-R BS.1770-4, table 1 and table 2: the K-weighting biquads at 48 kHz.
SHELF_48K = [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585]
HIGHPASS_48K = [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]
RATES = (44100, 48000, 88200, 96000)


def test_k_weighting_reproduces_the_published_coefficients():
    sos = k_weighting(48000)
    assert np.max(np.abs(sos[0] - SHELF_48K)) < 1e-14
    assert np.max(np.abs(sos[1] - HIGHPASS_48K)) < 1e-14


@pytest.mark.parametrize("rate", [44100, 88200, 96000])
def test_k_weighting_is_the_same_filter_at_every_rate(rate):
    f = np.geomspace(20, 20000, 400)
    ref = np.abs(sosfreqz(k_weighting(48000), worN=f, fs=48000)[1])
    got = np.abs(sosfreqz(k_weighting(rate), worN=f, fs=rate)[1])
    assert np.max(np.abs(20 * np.log10(got / ref))) < 0.03


@pytest.mark.parametrize("rate", RATES)
def test_ebu_tech_3341_sine(rate):
    # EBU Tech 3341, test case 1: a 1 kHz sine at −23 dBFS in both channels reads −23.0 ± 0.1 LUFS.
    # One channel alone reads 10·log10(2) lower; the level scales linearly.
    t = np.arange(4 * rate) / rate
    for level in (-23.0, -33.0, -3.0):
        x = 10 ** (level / 20) * np.sin(2 * np.pi * 1000 * t)
        assert momentary_max(np.stack([x, x], axis=1), rate) == pytest.approx(level, abs=0.1)
        assert momentary_max(x, rate) + 10 * math.log10(2) == pytest.approx(level, abs=0.1)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_ebu_tech_3341_channels_add_their_energies(rate):
    # Test case 3 of EBU Tech 3341 in spirit: a sine in one channel only reads its own energy, and uncorrelated
    # channels add in power, never in amplitude.
    t = np.arange(4 * rate) / rate
    x = 10 ** (-23 / 20) * np.sin(2 * np.pi * 1000 * t)
    silent = np.zeros_like(x)
    assert momentary_max(np.stack([x, silent], axis=1), rate) == pytest.approx(momentary_max(x, rate), abs=1e-9)
    y = 10 ** (-23 / 20) * np.sin(2 * np.pi * 1500 * t + 0.3)
    both = momentary_max(np.stack([x, y], axis=1), rate)
    alone = 10 ** (momentary_max(x, rate) / 10) + 10 ** (momentary_max(y, rate) / 10)
    assert both == pytest.approx(10 * math.log10(alone), abs=0.01)


def test_the_largest_window_counts_every_position():
    rate = 48000
    t = np.arange(2 * rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    steady = momentary_max(tone, rate)
    # A 100 ms burst fills a quarter of the 400 ms window.
    burst = np.concatenate((tone[: rate // 10], np.zeros(rate)))
    assert momentary_max(burst, rate) == pytest.approx(steady + 10 * math.log10(0.25), abs=0.02)
    # A burst at the very end of a file reads the same: windows run past the end.
    assert momentary_max(np.concatenate((np.zeros(rate), tone[: rate // 10])), rate) == pytest.approx(
        momentary_max(burst, rate), abs=1e-9
    )
    assert momentary_max(np.zeros(1000), rate) == -math.inf


def test_the_report_gives_the_loudness_of_the_export():
    audio, report = R.render(loads(EXAMPLE))
    assert report.files[0].loudness == report.takes[0].loudness == momentary_max(audio, 48000)
    assert "LUFS, the largest 400 ms momentary loudness (BS.1770)" in str(report)
    quieter = R.render(make(**{"output.db": -18.0}))[1]
    assert quieter.files[0].loudness == pytest.approx(report.files[0].loudness - 12.0, abs=1e-5)
    silent = R.render(make(**{"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}))[1]
    assert silent.files[0].loudness == -math.inf and "loudness silent" in str(silent)


def test_a_stereo_file_reads_both_channels():
    audio, report = R.render_takes(loads(EXAMPLE), 1, stereo=True)
    f = report.files[0]
    left, right = report.takes[0].loudness, report.takes[1].loudness
    assert f.loudness == momentary_max(audio[0], 48000)
    # The channels are different strikes, so their largest windows may fall apart: the pair reads at most the sum.
    assert max(left, right) < f.loudness <= 10 * math.log10(10 ** (left / 10) + 10 ** (right / 10)) + 1e-9


def test_loudness_agrees_across_export_rates():
    # The same sound reads the same at every rate: the K-weighting is one analog filter.
    levels = [R.render(make(**{"output.sample_rate": rate}))[1].files[0].loudness for rate in RATES]
    assert max(levels) - min(levels) < 0.05
