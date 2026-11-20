# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering: the timing contract, output stage, domain rule, drive headroom, takes and pairs (plan §3, §5, §7)."""

import dataclasses
import math
import multiprocessing
import sys
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from brightwork_synth import carrier, cli, render, spectrum, voice
from brightwork_synth.loudness import momentary_max
from brightwork_synth.patch import SAMPLE_RATES, loads

from .conftest import EXAMPLE, make, make_tone


def amp(*points: tuple[float, float]) -> list[dict[str, float]]:
    """Write knots as TOML parses them."""
    return [{"ms": m, "gain": g} for m, g in points]


def headroom_of(take: render.Take) -> render.Headroom:
    """Return a take's drive headroom, which every sounding take has."""
    headroom = take.headroom
    assert headroom is not None
    return headroom


LENGTH_MS = 120.0  # the example's length: its plates' last knot
TICK_MS = 5.0  # its tick's
DAMPED_MS = 50.0  # a damper's last knot, where it ends the sound
SAMPLES = 5851  # the example's exported samples at 48 kHz
INTERNAL_RATE, FACTOR = 1536000, 32  # the 48 kHz family's
MARGIN_DB = 20.0  # the example's margin inside the domain, at least
FLOOR = 1e-7  # the absolute bound near silence
MATCHED_DB = 0.01  # how close level matching brings a set's files
DECORRELATED = 0.2  # a stereo pair's channels correlate less than this
TOP_COMPRESSION = 2.0  # the closed hat's top sub-band plays its decay √2·√2 = 2 times faster
TRUE_PEAK_DB = 1e-4  # how close the true peak comes to a sine's amplitude: the passband's flatness


SILENT = {"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}
FAST = {
    "noise[2].level_db": 6.0,
    "noise[2].band_hz": [1000.0, 20000.0],
    "noise[2].amp": amp((0, 0), (0.01, 1), (0.11, 0.7), (2, 0.1), (6, 0)),
}
HOT = {"noise[1].level_db": 6.0, "drive.db": 24.0}


def test_output_length_follows_the_timing_contract() -> None:
    """Render at every rate as long as the timing contract says, from exactly zero."""
    for rate in SAMPLE_RATES:
        p = make_tone(**{"output.sample_rate": rate})
        s = render.synthesize(p)
        internal, factor = render.rates(rate)
        count = voice.sample_count(0.120, internal)
        taps = render.decimator(render.FAMILY[rate]).taps.size
        assert s.source_samples == count
        assert s.out.size == (count + taps - 2) // factor + 1
        audio, report = render.render(p)
        assert audio.size == s.out.size == report.samples
        assert audio.dtype == np.float32
        # Every shape starts at 0, so every file starts at exactly zero.
        assert audio[0] == 0.0


def test_the_length_is_the_latest_audible_sub_band_or_the_damper() -> None:
    """End the sound at its latest audible sub-band, silent sources aside, or at the damper's last knot."""
    assert render.render(make())[1].source_ms == LENGTH_MS
    # A silent source never lengthens the file, however long its shape.
    assert render.render(make(**{"noise[1].level_db": -math.inf}))[1].source_ms == TICK_MS
    long_tone = {"tone[1].level_db": -math.inf, "tone[1].amp": amp((0, 0), (1, 1), (900, 0))}
    assert render.render(make_tone(**long_tone))[1].source_ms == LENGTH_MS
    # Bloom without damping stretches the rise of the top sub-band, which then ends last.
    bloom = make(**{"noise[1].damping": 0.0, "noise[1].bloom": 1.0})
    lay = spectrum.layout(6000.0, 0.0, 1.0)
    end = float(lay.stretch(1.0)[-1]) * 0.3 + (120.0 - 0.3)
    assert bloom.length_ms == pytest.approx(end, rel=1e-12)
    s = render.synthesize(bloom)
    assert s.source_samples == voice.sample_count(bloom.length_ms / 1000, s.rate)
    # The damper ends the sound at its last knot.
    damped = render.render(make(**{"damper.amp": amp((30, 1), (36, 0.1), (50, 0))}))[1]
    assert damped.source_ms == DAMPED_MS


def test_repeating_a_render_changes_nothing() -> None:
    """Render twice, and with the table cache cleared, and find the same samples."""
    p = make_tone(**{"drive.db": 18.0, "noise[1].chatter": 0.8})
    assert np.array_equal(render.render(p)[0], render.render(p)[0])
    carrier_cache_cleared = render.render(p)[0]
    carrier.clear_cache()
    assert np.array_equal(render.render(p)[0], carrier_cache_cleared)


def test_output_gain_and_polarity_never_touch_synthesis() -> None:
    """Change the output gain, polarity, mode or format, and find the synthesis untouched."""
    base = render.synthesize(make()).out
    for edits in (
        {"output.db": 3.0},
        {"output.polarity": -1},
        {"output.mode": "peak", "output.db": -1.0},
        {"output.format": "pcm24"},
    ):
        assert np.array_equal(render.synthesize(make(**edits)).out, base)
    pos, _ = render.render(make(**{"output.db": 0.0}))
    neg, _ = render.render(make(**{"output.db": 0.0, "output.polarity": -1}))
    assert np.array_equal(neg, -pos)
    assert np.array_equal(pos, base.astype(np.float32))
    quiet, _ = render.render(make(**{"output.db": -20.0}))
    assert np.array_equal(quiet, (base * 10 ** (-20 / 20)).astype(np.float32))


@pytest.mark.parametrize("target", [0.0, -0.5, -12.0, -60.0])
def test_peak_mode_hits_its_target(target: float) -> None:
    """Bring the peak to its target in peak mode, from 0 to −60 dBFS."""
    audio, report = render.render(make(**{"output.mode": "peak", "output.db": target, "drive.db": 6.0}))
    assert 20 * math.log10(np.max(np.abs(audio))) == pytest.approx(target, abs=1e-6)
    assert np.max(np.abs(audio)) <= 1.0
    assert not report.over


def test_silence_stays_silent() -> None:
    """Render silence as silence in every mode, reported as silent, with no headroom."""
    for mode in ("gain", "peak"):
        audio, report = render.render(make(**SILENT, **{"output.mode": mode, "output.db": 0.0}))
        assert np.all(audio == 0.0)
        assert report.peak == 0.0
        assert "silent" in str(report)
        assert report.takes[0].headroom is None


def test_overs_are_reported_never_clipped() -> None:
    """Report a render over full scale, and what to lower, without clipping it."""
    audio, report = render.render(make(**{"output.db": 12.0, "drive.db": 12.0}))
    assert report.over
    assert report.peak > 1.0
    assert np.max(np.abs(audio)) == pytest.approx(report.peak)
    assert "over full scale" in str(report)
    assert "lower output.db by at least" in str(report)
    assert not render.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[1].over


def test_a_later_knot_never_changes_earlier_samples() -> None:
    """Add a knot after the end, and find every sample before its reach unchanged."""
    short = make()
    long = make(**{"noise[1].amp": amp((0, 0), (0.3, 1), (8, 0.45), (30, 0.14), (70, 0.04), (120, 0.01), (400, 0))})
    a, b = render.synthesize(short).out, render.synthesize(long).out
    # PCHIP slopes are local: the new value at 120 ms moves the slope at 70 ms, which reaches back only to the knot
    # at 30 ms. The top sub-band plays the decay √2·√2 = 2 times faster, so that knot sounds at 0.3 + 29.7 / 2 ms
    # there, and the decimator is causal.
    s = float(spectrum.layout(6000.0, 0.5, 0.0).compression(0.5)[-1])
    assert s == TOP_COMPRESSION
    m = int((0.0003 + 0.0297 / s) * 48000)
    assert np.array_equal(a[:m], b[:m])
    assert not np.array_equal(a[: m + 48], b[: m + 48])
    assert b.size > a.size


def test_report() -> None:
    """Report the example: its length, rates, sources, levels and domain, and a tone's realized partials."""
    audio, report = render.render(loads(EXAMPLE))
    assert report.samples == audio.size == SAMPLES
    assert report.exported_ms == pytest.approx(1000 * SAMPLES / 48000)
    assert report.source_ms == LENGTH_MS
    assert report.internal_rate == INTERNAL_RATE
    assert report.factor == FACTOR
    t = report.takes[0]
    f = report.files[0]
    assert f.true_peak >= f.peak == t.peak
    assert t.take == 1
    assert f.gain_db == 0.0
    assert report.inside
    assert t.domain.margin_db > MARGIN_DB
    text = str(report)
    assert "noise[1] 714 modes in 6000–16000 Hz, mean spacing 14.00 Hz; 3 sub-bands from 6000 Hz" in text
    assert "noise[2] continuous noise, 3000–18000 Hz; one band" in text
    assert "120.0 ms source, 121.9 ms exported (5851 samples at 48000 Hz, float32, mono), seed 1, take 1" in text
    assert "predicted aliasing" in text
    assert "margin" in text
    assert "render" in text
    far = render.render(make(**HOT))[1]
    assert not far.inside
    assert "OUTSIDE" in str(far)
    assert "brightwork verify" in str(far)
    # Output gain never moves the domain status: it is judged at unity gain.
    assert render.render(make(**{"output.db": -40.0}))[1].takes[0].domain == t.domain
    tone = str(render.render(make_tone())[1])
    assert (
        "tone[1] square partials at 205.250, 304.375, 369.625, 522.750, 540.000, 800.000 Hz, band 7000–16000 Hz; "
        "2 sub-bands from 7000 Hz" in tone
    )
    sine = make_tone(**{"tone[1].wave": "sine", "tone[1].partials": [{"hz": 440.0, "level_db": 0.0}]})
    assert "sine partials at 440.000 Hz (the lowest A4 +0.0 cents)" in str(render.render(sine)[1])
    assert "tone[1] silent" in str(render.render(make_tone(**{"tone[1].level_db": -math.inf}))[1])


def test_domain_rule() -> None:
    """Judge renders by the rule: the saturation term's edge, kinks adding, and the floor near silence."""
    kinks = voice.Kinks((), *(np.zeros(0),) * 5)

    def synth(knee_step: float, kink: float, peak: float) -> render.Synthesis:
        knees = voice.Knees(knee_step, np.zeros(0), np.zeros(0))
        return render.Synthesis(
            np.array([peak]),
            1536000,
            32,
            1,
            voice.Slew(knee_step, knees),
            kinks,
            kink,
            1.0,
            24000.0,
            peak,
        )

    # A linear path with no kinks predicts nothing.
    d = render.domain(synth(0.0, 0.0, 1.0))
    assert d.inside
    assert d.margin_db == math.inf
    # The saturation term alone crosses the limit at q = π² / ln(SATURATION / LIMIT); here α = 1, so q is the step.
    edge = math.pi**2 / math.log(render.SATURATION / render.LIMIT)
    assert render.domain(synth(edge * 0.999, 0.0, 1.0)).inside
    assert not render.domain(synth(edge * 1.001, 0.0, 1.0)).inside
    # Kinks add; near-silent renders use the absolute floor.
    assert not render.domain(synth(0.0, 2e-5, 1.0)).inside
    assert render.domain(synth(0.0, 0.9e-7, 1e-3)).inside
    assert not render.domain(synth(0.0, 1.1e-7, 1e-3)).inside
    assert render.allowed(0.02) == pytest.approx(2e-7)
    assert render.allowed(0.0099) == FLOOR


@pytest.mark.parametrize("base", [44100, 48000])
def test_the_true_peak_is_read_between_the_samples_exactly(base: int) -> None:
    """Measure a 19 kHz sine, whose peaks fall between the file's samples, and find its amplitude to 0.0001 dB."""
    rate = render.K * base
    n = np.arange(round(0.05 * rate))
    amplitude = 0.7
    sine = amplitude * np.sin(2 * math.pi * 19000 * n / rate + 0.3) * np.sin(math.pi * n / n.size) ** 2  # faded
    peak = render.true_peak(sine, render.decimator(base).taps)
    assert 20 * math.log10(peak / amplitude) == pytest.approx(0.0, abs=TRUE_PEAK_DB)
    file = render.decimate(render.decimator(base).taps, sine, render.K)
    assert np.max(np.abs(file)) < peak  # the file's samples miss the crest


def test_a_render_reports_the_true_peak_of_its_files() -> None:
    """Report each file's true peak at its gain: above its sample peak, and the same at both rates of a family."""
    p = make(**{"noise[2].level_db": 0.0})
    report = render.render(p)[1]
    double = render.render(make(**{"noise[2].level_db": 0.0, "output.sample_rate": 96000}))[1]
    f = report.files[0]
    assert f.true_peak > f.peak
    assert f.true_peak == pytest.approx(10 ** (-6 / 20) * render.synthesize(p).true_peak, rel=1e-6)
    assert double.files[0].true_peak == pytest.approx(f.true_peak, rel=1e-6)
    assert f"{20 * math.log10(f.true_peak):.2f} dBTP true" in str(report)


@pytest.mark.parametrize(("db", "fmt"), [(-1.0, "float32"), (-0.1, "float32"), (0.0, "pcm16"), (0.0, "pcm24")])
def test_true_peak_mode_reaches_its_target_within_the_pcm_ceiling(db: float, fmt: str) -> None:
    """Normalize a set to its loudest true peak, and reach either the target or the format's ceiling, never beyond."""
    p = make(**{"output.mode": "true-peak", "output.db": db, "output.format": fmt, "noise[2].level_db": 0.0})
    _, report = render.render_takes(p, 3)
    top = max(f.true_peak for f in report.files)
    assert top <= 10 ** (db / 20) * (1 + 1e-6)
    assert report.peak <= report.ceiling
    hit_target = math.isclose(20 * math.log10(top), db, rel_tol=0, abs_tol=1e-4)
    hit_ceiling = report.peak == pytest.approx(report.ceiling, rel=2**-24)
    assert hit_target or hit_ceiling  # one of the two limits binds


# One render knows the rule at every drive.


def rule_inside(edits: dict[str, object], drive: float, *, tone: bool = False) -> bool:
    """Return the validated-domain status of a full render at a drive."""
    return render.domain(render.synthesize((make_tone if tone else make)(**{**edits, "drive.db": drive}))).inside


@pytest.mark.parametrize(
    ("edits", "tone"),
    [
        ({}, False),
        ({"noise[1].level_db": 6.0}, False),
        ({"noise[1].chatter": 0.8, "damper.amp": amp((40, 1), (60, 0))}, True),
    ],
)
def test_one_render_knows_the_knee_slew_and_kink_bound_at_every_drive(edits: dict[str, object], *, tone: bool) -> None:
    """Predict the knee slew, kink bound and peak at any drive from one render, as full renders find them."""
    base = render.synthesize((make_tone if tone else make)(**edits, **{"drive.db": 0.0}))
    for d in (0.5, 6.0, 18.0, 30.0, 48.0):
        s = render.synthesize((make_tone if tone else make)(**edits, **{"drive.db": d}))
        a = voice.drive_alpha(d)
        assert base.slew.knee(a) == pytest.approx(s.slew.knee(a), rel=1e-12)
        assert voice.kink_bound(base.kinks, a, s.rate, s.band) == s.kink
        assert voice.kink_bounds(base.kinks, [a], s.rate, s.band)[0] == pytest.approx(s.kink, rel=1e-12)
        # The probes give the exact output at the render's own drive, so their peak is the output peak.
        assert s.probes is not None
        assert base.probes is not None
        assert s.probes.peak([a])[0] == pytest.approx(np.max(np.abs(s.out)), rel=1e-12)
        assert base.probes.peak([a])[0] <= np.max(np.abs(s.out)) * (1 + 1e-12)  # elsewhere, a lower bound


@pytest.mark.parametrize(
    "edits",
    [{"drive.db": 6.0}, {**HOT}, {**FAST, "drive.db": 0.0}, {"noise[2].level_db": 3.0, "drive.db": 10.0}],
)
def test_drive_headroom_finds_where_the_status_changes(edits: dict[str, object]) -> None:
    """Report the drive at which the status changes, and find full renders either side of it."""
    report = render.render(make(**edits))[1]
    drive = make(**edits).drive.db
    h = headroom_of(report.takes[0])
    assert h.db == pytest.approx(h.edge_db - drive)
    # Full renders on either side of the reported edge.
    assert rule_inside(edits, h.edge_db - 0.01)
    assert not rule_inside(edits, h.edge_db + 0.01)
    assert (h.db > 0) == report.inside


def test_drive_headroom_never_skips_an_outside_island() -> None:
    """Find the narrow outside island just below a drop of the knee slew, from inside it and from below."""
    # A frontier step with reach 0.1 counts only while α ≤ 10, so with q = α·0.1 up to there and α·0.05 after,
    # the rule is outside from α = 10·q* (q* = π²/ln(SATURATION/LIMIT)), inside again past α = 10, and outside
    # for good from α = 20·q*.
    knees = voice.Knees(0.05, np.array([0.1]), np.array([0.1]))
    kinks = voice.Kinks((), *(np.zeros(0),) * 5)
    probes = render.Probes(np.array([50.0]), ((0, np.array([1.0])),))  # a saturated probe: the peak is 1 at every drive
    alpha = voice.drive_alpha(12.0)

    def at(a: float) -> render.Synthesis:
        return render.Synthesis(
            np.array([1.0]),
            1536000,
            32,
            1,
            voice.Slew(0.1, knees),
            kinks,
            0.0,
            a,
            24000.0,
            1.0,
            probes,
        )

    threshold = math.pi**2 / math.log(render.SATURATION / render.LIMIT)
    island = threshold / 0.1
    assert render.ratios(at(alpha), [island * 0.999])[0] < 1 < render.ratios(at(alpha), [island * 1.001])[0]
    assert render.ratios(at(alpha), [10.0 * 1.0001])[0] < 1  # inside again just past the drop
    d = render.domain(at(alpha))
    assert d.inside
    h = render.drive_headroom(at(alpha), 12.0, d)
    assert h is not None
    assert render.alphas([h.edge_db])[0] == pytest.approx(island, rel=1e-5)
    # From inside the island, the way back in is below it.
    inside_island = 0.5 * (island + 10.0)
    drive = float(render.drives_db(inside_island))
    s = at(inside_island)
    assert not render.domain(s).inside
    h = render.drive_headroom(s, drive, render.domain(s))
    assert h is not None
    assert h.db < 0
    assert render.alphas([h.edge_db])[0] == pytest.approx(island, rel=1e-5)


def test_drive_headroom_at_the_extremes() -> None:
    """Report inside at every drive for a quiet hit, and outside without drive for a fast spike."""
    quiet = render.render(make(**{"noise[1].level_db": -45.0, "noise[2].level_db": -45.0}))[1]
    assert headroom_of(quiet.takes[0]).db == math.inf
    assert "inside at every drive up to 48 dB" in str(quiet)
    # A 30 µs spike of full-band noise aliases above the limit with no drive at all.
    spike = {
        "noise[1].level_db": -math.inf,
        "drive.db": 0.0,
        "noise[2].level_db": 6.0,
        "noise[2].band_hz": [1000.0, 20000.0],
        "noise[2].amp": amp((0, 0), (0.01, 1), (0.02, 0.3), (0.03, 0)),
    }
    fast = render.render(make(**spike))[1]
    assert not fast.inside
    assert headroom_of(fast.takes[0]).db == -math.inf
    assert "outside even without drive: the kinks alone exceed the limit" in str(fast)


def test_the_cause_names_what_to_change() -> None:
    """Name what the predicted aliasing comes from: the knot that contributes most, or the saturation."""
    fast = render.render(make(**FAST, **{"drive.db": 0.0}))[1]
    c = fast.takes[0].cause
    assert c.dominant == "kinks"
    assert c.knot is not None
    assert c.knot.startswith("noise[2].amp[")
    assert 0 < c.share <= 1
    assert c.knot_ms == {"noise[2].amp[1]": 0.0, "noise[2].amp[2]": 0.01, "noise[2].amp[3]": 0.11}[c.knot]
    assert "mostly kinks" in str(fast)
    assert f"at {c.knot} ({c.knot_ms:g} ms)" in str(fast)
    assert c.saturation == 0.0
    assert c.kinks == pytest.approx(fast.takes[0].domain.predicted)
    hot = render.render(make(**HOT))[1]
    c = hot.takes[0].cause
    assert c.dominant == "saturation"
    assert "mostly saturation" in str(hot)
    assert c.saturation + c.kinks == pytest.approx(hot.takes[0].domain.predicted)
    assert "lower drive.db to" in str(hot)
    # A damper's knots are named like any other.
    choked = make(**{"damper.amp": amp((20, 1), (20.01, 0))})
    c = render.render(choked)[1].takes[0].cause
    assert c.knot is not None
    assert c.knot.startswith("damper.amp[")
    assert c.knot_ms in (20.0, 20.01)


def test_knot_times_by_path() -> None:
    """Find a knot's time by its path, as the patch writes it."""
    p = make_tone(**{"damper.amp": amp((30, 1), (35, 0.3), (50, 0))})
    for path, ms in (
        ("noise[1].amp[2]", 0.3),
        ("noise[2].amp[5]", 5.0),
        ("tone[1].amp[4]", 20.0),
        ("damper.amp[3]", 50.0),
    ):
        assert render.knot_ms(p, path) == ms


# Takes and stereo pairs.


def test_takes_are_the_same_cymbal_struck_again() -> None:
    """Render takes as the patch struck again, and struck tones alone the same in every take."""
    p = make_tone(**{"drive.db": 9.0})
    audio, report = render.render_takes(p, 3)
    assert [t.take for t in report.takes] == [1, 2, 3]
    assert [t.number for t in report.takes] == [1, 2, 3]
    for j, a in enumerate(audio):
        assert np.array_equal(a, render.render(dataclasses.replace(p, take=1 + j))[0])
    assert not np.array_equal(audio[0], audio[1])
    assert report.seed == 1
    assert "seed 1, takes 1–3" in str(report)
    # Struck tones alone are the same sound in every take.
    struck = make_tone(**SILENT, **{"tone[1].phase": "strike"})
    pair = render.render_takes(struck, 2)[0]
    assert np.array_equal(pair[0], pair[1])


def test_a_set_shares_one_gain() -> None:
    """Bring a peak-mode set's loudest take to the target, and the others in proportion."""
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = render.render_takes(p, 4)
    peaks = [float(np.max(np.abs(a))) for a in audio]
    assert 20 * math.log10(max(peaks)) == pytest.approx(-1.0, abs=1e-6)
    assert len(set(peaks)) == len(audio)  # the others keep their natural levels
    unity = [np.max(np.abs(render.synthesize(dataclasses.replace(p, take=t)).out)) for t in (1, 2, 3, 4)]
    np.testing.assert_allclose(np.array(peaks) / max(peaks), np.array(unity) / max(unity), rtol=1e-6)
    assert report.set_gain_db == pytest.approx(-1.0 - 20 * math.log10(max(unity)), abs=1e-9)


def test_level_matching_scales_each_file_to_the_first() -> None:
    """Level-match a set's files to the first, and say so."""
    p = loads(cli.reference_text("closed"))
    _, report = render.render_takes(p, 4)
    matched, mreport = render.render_takes(p, 4, level_match=True)
    rate = p.output.sample_rate
    first = momentary_max(matched[0], rate)
    for a in matched:
        assert momentary_max(a, rate) == pytest.approx(first, abs=MATCHED_DB)
    assert mreport.loudness_spread <= MATCHED_DB < report.loudness_spread
    assert mreport.files[0].gain_db == 0.0
    assert any(f.gain_db != 0.0 for f in mreport.files)
    assert "level-matched to the first" in str(mreport)


def test_the_report_flags_uneven_sets() -> None:
    """Flag a set whose takes spread over 1 dB, and suggest level matching."""
    click = make(
        **{
            "noise[1].level_db": -math.inf,
            "noise[2].level_db": 0.0,
            "noise[2].band_hz": [3000.0, 6000.0],
            "noise[2].amp": amp((0, 0), (0.05, 1), (0.15, 0.7), (0.5, 0.1), (1, 0)),
        },
    )
    report = render.render_takes(click, 8)[1]
    assert report.uneven
    assert "consider --level-match" in str(report)
    text = str(report)
    assert "8 files sharing one gain" in text
    assert "take 8" in text
    assert "takes 1–8" in text


def test_take_numbers_stay_in_range() -> None:
    """Number a set's takes from the patch's take, and refuse takes past 2**31 or a count out of range."""
    with pytest.raises(render.RenderError):
        render.take_numbers(make(take=2**31 - 1), 3)
    assert render.take_numbers(make(take=2**31 - 2), 3)[-1] == 2**31
    assert render.take_numbers(make(take=5), 2, stereo=True) == [5, 6, 7, 8]
    for bad in (0, 129, True):
        with pytest.raises(render.RenderError):
            render.take_numbers(make(), bad)
    with pytest.raises(render.RenderError, match="1 to 64 files with --stereo"):
        render.take_numbers(make(), 65, stereo=True)


def test_a_stereo_file_is_two_consecutive_takes() -> None:
    """Pair consecutive takes as stereo files, the channels unlike in detail, sharing the set's gain."""
    p = make_tone(take=3)
    audio, report = render.render_takes(p, 2, stereo=True)
    assert len(audio) == len(report.files)
    assert all(a.shape == (report.samples, 2) and a.dtype == np.float32 for a in audio)
    assert [t.take for t in report.takes] == [3, 4, 5, 6]
    assert [f.takes for f in report.files] == [(0, 1), (2, 3)]
    for j, a in enumerate(audio):
        for c in (0, 1):
            mono = render.render(dataclasses.replace(p, take=3 + 2 * j + c))[0]
            assert np.array_equal(a[:, c], mono)
    text = str(report)
    assert "stereo, takes as left and right" in text
    assert "take 3 (left)" in text
    assert "take 6 (right)" in text
    # The channels are the same cymbal struck twice: alike in level, unlike in detail.
    left, right = audio[0][:, 0].astype(float), audio[0][:, 1].astype(float)
    assert abs(np.dot(left, right)) / math.sqrt(np.dot(left, left) * np.dot(right, right)) < DECORRELATED
    assert report.files[0].loudness == pytest.approx(momentary_max(audio[0], 48000))
    # In peak mode the pair shares the set gain, and the loudest channel of any file meets the target.
    peaked, preport = render.render_takes(make_tone(**{"output.mode": "peak", "output.db": -1.0}), 2, stereo=True)
    assert 20 * math.log10(max(float(np.max(np.abs(a))) for a in peaked)) == pytest.approx(-1.0, abs=1e-6)
    assert preport.files[0].gain_db == preport.files[1].gain_db == 0.0


def test_a_set_over_full_scale_is_reported() -> None:
    """Report a gain-mode set over full scale, and what to lower, above its files."""
    _, report = render.render_takes(make(**{"output.db": 12.0, "drive.db": 12.0}), 2)
    assert report.over
    text = str(report)
    assert text.index("over full scale") < text.index("file 1")


def test_a_render_never_returns_non_finite_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a render whose output stage yields a non-finite sample, rather than return it."""

    def broken(out: render.Floats, gain: float, polarity: int) -> NDArray[np.float32]:
        return np.full(out.shape, math.inf * gain * polarity, dtype=np.float32)

    monkeypatch.setattr(render, "output_stage", broken)
    with pytest.raises(render.RenderError, match="non-finite"):
        render.render(make())


def test_quantization_refuses_what_its_format_cannot_hold() -> None:
    """Refuse audio at full scale, which the dither could carry past the largest PCM value."""
    with pytest.raises(render.RenderError, match="does not fit 16-bit PCM"):
        render.quantize(np.ones(64, dtype=np.float32), 16, 1, 1)


def test_a_silent_set_reports_each_file_silent() -> None:
    """Report every file of a silent set as silent, with nothing to say about its domain."""
    text = str(render.render_takes(make(**SILENT), 2)[1])
    assert "file 1   silent" in text
    assert "file 2   silent" in text
    assert "mostly" not in text


def _renders_as(text: str, expected: bytes) -> None:
    """Render a patch in this process and fail unless it is exactly `expected`."""
    if render.render(loads(text))[0].tobytes() != expected:
        raise AssertionError


def test_a_process_forked_after_a_render_renders_too() -> None:
    """Render, fork as multiprocessing does by default on Linux before Python 3.14, and render the same in the child.

    The child inherits the parent's caches and locks but none of its threads, so nothing may wait on those threads.
    """
    if sys.platform == "linux":  # a check the type checker reads too, unlike a skip mark
        expected = render.render(loads(EXAMPLE))[0].tobytes()
        with warnings.catch_warnings():  # Python warns that forking a process with threads may deadlock: the point here
            warnings.simplefilter("ignore", DeprecationWarning)
            child = multiprocessing.get_context("fork").Process(target=_renders_as, args=(EXAMPLE, expected))
            child.start()
        child.join(timeout=60)
        alive = child.is_alive()
        child.kill()
        child.join()
        assert not alive, "the forked child hung"
        assert child.exitcode == 0
        return
    pytest.skip("fork is the default only on Linux: Windows has none, and macOS's system libraries make it unsafe")
