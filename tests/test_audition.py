import os
import random
import threading
import time

import numpy as np
import pytest
from conftest import EXAMPLE

from cymbal import audition as A
from cymbal.loudness import momentary_max
from cymbal.patch import choke_knots, loads
from cymbal.render import quantize, render, render_takes

RATE = 1000  # a toy rate keeps the hit arithmetic readable


def play(hits, frames, block):
    out = []
    while frames > 0:
        n = min(block, frames)
        out.append(hits.fill(n))
        frames -= n
    return np.concatenate(out)


def ones(n, value=1.0):
    return np.full(n, value, dtype=np.float32)


class Collect:
    def __init__(self, sink):
        self.sink = sink

    def write(self, s):
        self.sink.append(s)

    def flush(self):
        pass


# The hit scheduler.


def test_hits_repeat_after_each_hit_plus_half_a_second():
    buf = np.arange(1, 301, dtype=np.float32)
    hits = A.Hits(RATE)
    hits.offer(0, 0, [buf])
    out = play(hits, 2000, 64)
    period = 300 + 500
    for start in (0, period, 2 * period):
        assert np.array_equal(out[start : start + 300], buf)
        assert np.all(out[start + 300 : start + period] == 0)


def test_nothing_plays_until_a_buffer_arrives():
    hits = A.Hits(RATE)
    assert np.all(play(hits, 500, 100) == 0)
    hits.offer(0, 0, [ones(10)])
    out = play(hits, 100, 100)
    assert np.all(out[:10] == 1) and np.all(out[10:] == 0)


def test_block_size_never_moves_a_hit():
    buf = np.random.default_rng(0).normal(size=333).astype(np.float32)
    outs = []
    for block in (1, 7, 64, 1000, 4096):
        hits = A.Hits(RATE, bpm=97.0, every=0.5, pattern=(1, 0, 1, 1), chokes={0: {0}})
        hits.offer(0, 0, [buf])
        outs.append(play(hits, 5000, block))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])


def test_new_buffer_takes_over_at_the_next_hit():
    hits = A.Hits(RATE)
    hits.offer(0, 0, [ones(300)])
    first = play(hits, 150, 50)
    hits.offer(0, 1, [ones(100, 2.0)])  # arrives mid-hit
    out = np.concatenate((first, play(hits, 2000, 50)))
    assert np.all(out[:300] == 1.0)  # the playing hit finishes with its own buffer
    assert np.all(out[300:800] == 0.0)
    assert np.all(out[800:900] == 2.0)  # the next hit plays the new buffer...
    assert np.all(out[900:1400] == 0.0)
    assert np.all(out[1400:1500] == 2.0)  # ...and the period follows its length


def test_older_generations_never_replace_newer_ones():
    hits = A.Hits(RATE)
    hits.offer(0, 2, [ones(10, 2.0)])
    hits.offer(0, 1, [ones(10, 1.0)])
    assert np.all(play(hits, 10, 10) == 2.0)
    # Within a generation, the whole set (stage 1) replaces its first take (stage 0), and never the reverse.
    hits = A.Hits(RATE)
    hits.offer(0, (3, 1), [ones(10, 5.0)])
    hits.offer(0, (3, 0), [ones(10, 4.0)])
    assert np.all(play(hits, 10, 10) == 5.0)
    hits.offer(0, (4, 0), [ones(10, 6.0)])
    assert np.all(play(hits, 1000, 10)[500:510] == 6.0)  # the next hit: 10 samples played, then 0.5 s


def test_steps_of_any_note_value_never_drift():
    # At 120 bpm a beat is 500 samples: eighths every 250, a crash on 4/1 every four bars.
    for every, step in ((0.5, 250), (0.25, 125), (1.0, 500), (16.0, 8000)):
        hits = A.Hits(RATE, bpm=120.0, every=every)
        hits.offer(0, 0, [ones(1)])
        out = play(hits, 4 * step + 1, 333)
        assert np.array_equal(np.flatnonzero(out), [0, step, 2 * step, 3 * step, 4 * step])
    # Fractional step lengths never drift: step n starts at round(n · every · rate · 60 / bpm).
    hits = A.Hits(RATE, bpm=140.0, every=0.25)
    hits.offer(0, 0, [ones(1)])
    out = play(hits, 20000, 333)
    assert np.array_equal(np.flatnonzero(out), [round(i * 0.25 * RATE * 60 / 140) for i in range(187)])


def test_overlapping_tails_sum():
    hits = A.Hits(RATE, bpm=120.0)
    hits.offer(0, 0, [ones(900)])
    out = play(hits, 2000, 128)
    assert np.all(out[:500] == 1) and np.all(out[500:900] == 2) and np.all(out[900:1000] == 1)


def test_a_pattern_names_the_patch_of_every_step():
    hits = A.Hits(RATE, 2, bpm=120.0, every=0.5, pattern=(1, 1, 2, 0))
    hits.offer(0, 0, [ones(1, 1.0)])
    hits.offer(1, 0, [ones(1, 2.0)])
    out = play(hits, 2000, 100)
    assert [out[250 * i] for i in range(8)] == [1, 1, 2, 0, 1, 1, 2, 0]
    # A patch that has not rendered yet rests on its steps.
    hits = A.Hits(RATE, 2, bpm=120.0, every=0.5, pattern=(1, 2))
    hits.offer(0, 0, [ones(1, 1.0)])
    out = play(hits, 1000, 100)
    assert [out[250 * i] for i in range(4)] == [1, 0, 1, 0]


def test_patches_cycle_hit_by_hit_and_takes_cycle_within_each():
    hits = A.Hits(RATE, 2, bpm=120.0)
    hits.offer(0, 0, [ones(1, 1.0), ones(1, 2.0)])
    hits.offer(1, 0, [ones(1, 10.0), ones(1, 20.0), ones(1, 30.0)])
    out = play(hits, 6000, 250)
    played = [out[500 * i] for i in range(12)]
    assert played == [1, 10, 2, 20, 1, 30, 2, 10, 1, 20, 2, 30]
    assert hits.log[:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_a_choke_fades_the_ringing_hit_with_the_curve_choke_writes():
    # Patch 1 rings for 2 s; patch 2 chokes it at its step, 250 samples in, over a 40 ms release.
    ring, close = ones(2000, 1.0), ones(1, 0.0)
    hits = A.Hits(RATE, 2, bpm=120.0, every=0.5, pattern=(1, 2, 0, 0, 0, 0, 0, 0), chokes={0: {1}}, release_ms=40.0)
    hits.offer(0, 0, [ring])
    hits.offer(1, 0, [close])
    out = play(hits, 2000, 64)
    assert np.all(out[:250] == 1.0)
    knots = choke_knots(0.0, 40.0)
    curve = A.fade(40.0, RATE)[0]
    t = np.arange(250, 2000) - 250
    np.testing.assert_allclose(out[250:2000], curve.on(t / RATE), rtol=0, atol=1e-7)
    assert out[250 + 20] == pytest.approx(10 ** (-3 / 2), rel=1e-3)  # −30 dB halfway through the release
    end = round(knots[-1][0])
    assert np.all(out[250 + end :] == 0.0)
    # A patch never chokes itself unless asked, and a choke only reaches hits already sounding.
    hits = A.Hits(RATE, 2, bpm=120.0, every=0.5, pattern=(1, 1), chokes={0: {1}})
    hits.offer(0, 0, [ones(600, 1.0)])
    hits.offer(1, 0, [ones(1, 0.0)])
    out = play(hits, 1000, 100)
    assert np.all(out[250:500] == 2.0) and np.all(out[500:600] == 3.0)  # every hit of patch 1 rings on


def test_stereo_takes_play_in_stereo_and_mono_hits_in_the_centre():
    pair = np.stack([ones(10, 1.0), ones(10, -1.0)], axis=1)
    hits = A.Hits(RATE, 2, bpm=120.0, channels=2)
    hits.offer(0, 0, [pair])
    hits.offer(1, 0, [ones(10, 0.5)])
    out = play(hits, 1000, 64)
    assert out.shape == (1000, 2)
    assert np.array_equal(out[:10], pair) and np.all(out[500:510] == 0.5)


# Beds and samples.


def test_a_bed_loops_under_the_hits_on_the_tempo_grid():
    bpm = 120.0
    bed = (np.arange(1, 4001, dtype=np.float32) / 10000)[:, None]  # two bars, mono
    hat = ones(10, 0.5)
    outs = []
    for block in (1, 64, 333, 4096):
        hits = A.Hits(RATE, bpm=bpm, every=0.5, bed=A.Bed(bed, RATE, bpm, 2))
        hits.offer(0, 0, [hat])
        outs.append(play(hits, 12000, block))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])
    expected = np.tile(bed[:, 0], 3)
    for step in range(48):
        expected[250 * step : 250 * step + 10] += 0.5
    assert outs[0].ndim == 1 and np.allclose(outs[0], expected, atol=1e-7)


def test_a_bed_a_little_short_of_whole_bars_never_drifts():
    bed = np.ones((3999, 1), dtype=np.float32)  # one sample short of two bars at 120 bpm
    hits = A.Hits(RATE, bpm=120.0, bed=A.Bed(bed, RATE, 120.0, 2))
    hits.offer(0, 0, [np.zeros(1, dtype=np.float32)])
    out = play(hits, 12000, 256)
    # Each loop restarts on the grid: the missing sample is a gap at the end of every period.
    assert np.array_equal(np.flatnonzero(out == 0), [3999, 7999, 11999])


def test_a_stereo_bed_opens_a_stereo_stream_with_the_hits_centred():
    bed = np.stack([np.full(2000, 0.1), np.full(2000, -0.1)], axis=1).astype(np.float32)
    hits = A.Hits(RATE, bpm=120.0, bed=A.Bed(bed, RATE, 120.0, 1))
    hat = np.linspace(0, 1, 50, dtype=np.float32)
    hits.offer(0, 0, [hat])
    out = play(hits, 4000, 100)
    assert hits.channels == 2 and out.shape == (4000, 2)
    centre = out - np.tile(bed, (2, 1))
    assert np.allclose(centre[:, 0], centre[:, 1]) and np.allclose(centre[500:550, 0], hat)


def test_hits_alternate_between_the_render_and_the_sample():
    hits = A.Hits(RATE)
    hits.offer(0, 0, [ones(100, 1.0)], ones(200, 2.0))
    out = play(hits, 3000, 64)
    # Without a tempo each hit repeats after its own length plus 0.5 s: render, sample, render, …
    assert np.all(out[:100] == 1.0) and np.all(out[600:800] == 2.0) and np.all(out[1300:1400] == 1.0)
    assert np.all(out[100:600] == 0.0) and np.all(out[800:1300] == 0.0)
    assert hits.order == [True, True, True]


def test_blind_pairs_play_in_the_recorded_random_order():
    hits = A.Hits(RATE, bpm=120.0, rng=random.Random(7))
    hits.offer(0, 0, [ones(10, 1.0)], ones(10, 2.0))
    out = play(hits, 500 * 40, 128)
    played = [out[500 * i] for i in range(40)]
    for pair, render_first in enumerate(hits.order):
        assert played[2 * pair : 2 * pair + 2] == ([1.0, 2.0] if render_first else [2.0, 1.0])
    assert 5 < sum(hits.order) < 15  # both orders occur


# Sessions.


def write_wav(path, data, rate, subtype="FLOAT"):
    import soundfile

    soundfile.write(path, data, rate, subtype=subtype)
    return path


@pytest.fixture
def example(tmp_path):
    path = tmp_path / "hat.toml"
    path.write_text(EXAMPLE)
    return path


def wait_for(condition, timeout=60.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def said(log, generation):
    """Whether audition has reported on a generation: its messages start with the tag."""
    return any(chunk.startswith(f"[{generation}] ") for chunk in log)


def edit(path, text):
    # Bump the modification time so the change is seen even within one clock tick.
    path.write_text(text)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


OTHER = EXAMPLE.replace("spacing_hz = 14.0", "spacing_hz = 25.0")


def test_audition_plays_the_exact_render_buffer(example):
    expected, _ = render(loads(EXAMPLE))
    session = A.Session(example, out=Collect([]))
    assert session.rate == 48000 and session.hits.channels == 1
    heard = np.concatenate([session.hits.fill(512) for _ in range(40)])
    assert np.array_equal(heard[: expected.size], expected)


def test_pcm_patches_play_their_quantized_samples(tmp_path):
    path = tmp_path / "pcm.toml"
    text = EXAMPLE.replace('"float32"     #', '"pcm16"       #')
    path.write_text(text)
    audio, report = render(loads(text))
    expected = quantize(audio, 16, 1, 1) / 2.0**15
    heard = A.Session(path, out=Collect([])).hits.fill(20000)
    np.testing.assert_array_equal(heard[: expected.size], expected.astype(np.float32))


def test_the_first_take_plays_while_the_others_render(example):
    audio, _ = render_takes(loads(EXAMPLE), 3)
    session = A.Session(example, takes=3, out=Collect([]))
    # The session starts with the first take alone; the set follows from the same generation.
    wait_for(lambda: session.hits._offers[0] is not None and session.hits._offers[0][0] == (0, 1))
    period = audio[0].size + 24000
    heard = np.concatenate([session.hits.fill(4800) for _ in range(60)])
    for j in range(3):
        start = j * period
        assert np.array_equal(heard[start : start + audio[j].size], audio[j])


def test_a_peak_mode_set_waits_for_all_its_takes(tmp_path):
    path = tmp_path / "peak.toml"
    text = EXAMPLE.replace('mode        = "gain"', 'mode        = "peak"').replace("db          = -6.0", "db = -1.0")
    path.write_text(text)
    audio, _ = render_takes(loads(text), 3)
    session = A.Session(path, takes=3, out=Collect([]))
    assert session.hits._offers[0][0] == (0, 0) and len(session.hits._offers[0][1]) == 3
    heard = session.hits.fill(audio[0].size)
    assert np.array_equal(heard, audio[0])  # the set's own gain from the first hit


def test_stereo_sessions_play_take_pairs(example):
    audio, _ = render_takes(loads(EXAMPLE), 1, stereo=True)
    session = A.Session(example, stereo=True, out=Collect([]))
    assert session.hits.channels == 2
    heard = session.hits.fill(audio[0].shape[0])
    assert heard.shape == audio[0].shape and np.array_equal(heard, audio[0])


def test_several_patches_render_behind_the_first(tmp_path, example):
    other = tmp_path / "other.toml"
    other.write_text(OTHER)
    first, _ = render(loads(EXAMPLE))
    second, _ = render(loads(OTHER))
    session = A.Session([example, other], bpm=120.0, every=0.5, pattern=(1, 2), out=Collect([]))
    wait_for(lambda: session.buffers[1] is not None)
    heard = np.concatenate([session.hits.fill(4800) for _ in range(10)])
    assert np.array_equal(heard[: first.size], first)  # step 1: the first patch
    step = 12000  # an eighth at 120 bpm
    np.testing.assert_allclose(heard[step : step + second.size], second, atol=1e-7)


def test_a_session_chokes_one_patch_with_another(tmp_path, example):
    ring = tmp_path / "open.toml"
    ring.write_text(EXAMPLE.replace("{ ms = 120.0, gain = 0.0 }", "{ ms = 900.0, gain = 0.0 }"))
    session = A.Session([ring, example], bpm=120.0, every=0.5, pattern=(1, 2), chokes=[(1, (2,))], out=Collect([]))
    wait_for(lambda: session.buffers[1] is not None)
    long, _ = render(loads(ring.read_text()))
    hat, _ = render(loads(EXAMPLE))
    heard = np.concatenate([session.hits.fill(4800) for _ in range(10)])
    step = 12000
    curve = A.fade(40.0, 48000)[0]
    want = long[step : step + 4800] * curve.on(np.arange(4800) / 48000) + hat[:4800]
    np.testing.assert_allclose(heard[step : step + 4800], want, atol=1e-6)


def test_audition_runs_until_interrupted_and_reports_what_it_cannot_play(example):
    seen = {}

    class Stream:
        def __init__(self, rate, channels, fill):
            seen["rate"], seen["channels"], seen["first"] = rate, channels, fill(64)

        def __enter__(self):
            raise KeyboardInterrupt

        def __exit__(self, *exc):
            return False

    assert A.audition([example], stream_factory=Stream, out=Collect([])) == 0
    assert seen["rate"] == 48000 and seen["channels"] == 1
    assert A.audition([example], stereo=True, stream_factory=Stream, out=Collect([])) == 0
    assert seen["channels"] == 2
    example.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"))
    assert A.audition([example], stream_factory=Stream, out=Collect([])) == 1  # over full scale
    example.write_text(EXAMPLE.replace("damping    = 0.5", "damping    = 9.0"))
    assert A.audition([example], stream_factory=Stream, out=Collect([])) == 2  # invalid

    def unsupported(rate, channels, fill):
        raise RuntimeError(f"the output device does not support {rate} Hz")

    example.write_text(EXAMPLE)
    assert A.audition([example], stream_factory=unsupported, out=Collect([])) == 1


def test_watcher_renders_only_the_newest_version(tmp_path):
    path = tmp_path / "hat.toml"
    path.write_text("v0")
    gate = threading.Event()
    rendered, results = [], []

    def work(text, generation):
        rendered.append(text)
        gate.wait(5)
        return text

    w = A.Watcher(path, work, lambda g, r, e, newest: results.append((g, r, newest)), text="v0")
    w.poll()
    assert w.generation == 0  # unchanged content starts nothing
    for i in range(1, 5):
        path.write_text(f"v{i}")
        w.poll()
        time.sleep(0.05)
    gate.set()
    wait_for(lambda: results and results[-1][0] == 4, 5)
    # v1 was already rendering; v2 and v3 were superseded before they started.
    assert rendered == ["v1", "v4"]
    assert results == [(1, "v1", False), (4, "v4", True)]


def test_watching_swaps_buffers_and_keeps_the_last_valid_one(example):
    log = []
    session = A.Session(example, watch=True, out=Collect(log))
    original = session.hits.fill(48000 // 2)  # the first hit starts playing
    assert session.hits.generation == 0

    edit(example, EXAMPLE.replace("damping    = 0.5", "damping    = 9.0"))
    session.poll()
    wait_for(lambda: said(log, 1))
    assert "[1] invalid patch, still playing the last valid one" in "".join(log)

    quieter = EXAMPLE.replace("db          = -6.0", "db          = -12.0")
    edit(example, quieter)
    session.poll()
    wait_for(lambda: said(log, 2))
    assert "[2] rendered in" in "".join(log) and "plays from its next hit" in "".join(log)
    assert session.hits.generation == 0  # nothing changes until the next hit
    expected, _ = render(loads(quieter))
    rest = np.concatenate([session.hits.fill(1000) for _ in range(60)])
    heard = np.concatenate((original, rest))
    old, _ = render(loads(EXAMPLE))
    assert np.array_equal(heard[: old.size], old)
    start = old.size + 24000  # the next hit: the rendered length plus 0.5 s
    assert np.all(heard[old.size : start] == 0)
    assert np.array_equal(heard[start : start + expected.size], expected)
    assert session.hits.generation == 2


def test_a_session_matches_the_sample_and_plays_over_the_bed(tmp_path, example):
    audio, report = render(loads(EXAMPLE))
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", np.stack([other, other * 0.5], axis=1) * 0.3, 48000, "PCM_24")
    bed = write_wav(tmp_path / "bed.wav", np.full((96000, 2), 0.01), 48000)  # one bar at 120 bpm, stereo
    log = []
    session = A.Session(example, bpm=120.0, bed=bed, against=sample, out=Collect(log))
    text = "".join(log)
    assert "bed      " in text and "1 bar at 120 bpm, stereo" in text and "sample   plays at" in text
    assert session.hits.channels == 2
    matched = session.hits._offers[0][2]
    assert matched.ndim == 1  # a mono render: the sample plays as its mid channel
    assert momentary_max(matched, 48000) == pytest.approx(report.files[0].loudness, abs=0.01)
    heard = session.hits.fill(48000)  # step 1: the render, exact, over the bed
    assert np.allclose(heard[: audio.size, 0] - 0.01, audio, atol=1e-7)
    assert np.allclose(heard[audio.size + 100 : 24000], 0.01)


def test_a_stereo_session_plays_a_stereo_sample_in_stereo(tmp_path, example):
    pair, _ = render_takes(loads(OTHER), 1, stereo=True)
    sample = write_wav(tmp_path / "sample.wav", pair[0] * 0.3, 48000)
    session = A.Session(example, stereo=True, against=sample, out=Collect([]))
    matched = session.hits._offers[0][2]
    assert matched.shape == pair[0].shape
    mine = render_takes(loads(EXAMPLE), 1, stereo=True)[1].files[0].loudness
    assert momentary_max(matched, 48000) == pytest.approx(mine, abs=0.01)
    mono = write_wav(tmp_path / "mono.wav", pair[0][:, 0] * 0.3, 48000)
    centred = A.Session(example, stereo=True, against=mono, out=Collect([])).hits._offers[0][2]
    assert centred.shape[1] == 2 and np.array_equal(centred[:, 0], centred[:, 1])


def test_the_sample_is_rematched_on_every_re_render(tmp_path, example):
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", other, 48000)
    log = []
    session = A.Session(example, watch=True, against=sample, out=Collect(log))
    before = session.hits._offers[0][2]
    quieter = EXAMPLE.replace("db          = -6.0", "db          = -18.0")
    edit(example, quieter)
    session.poll()
    wait_for(lambda: said(log, 1))
    after = session.hits._offers[0][2]
    assert 20 * np.log10(np.max(np.abs(before)) / np.max(np.abs(after))) == pytest.approx(12.0, abs=0.01)


def test_a_session_refuses_what_it_cannot_play_faithfully(tmp_path, example):
    wrong_rate = write_wav(tmp_path / "44k.wav", np.full((88200, 1), 0.1), 44100)
    uneven = write_wav(tmp_path / "uneven.wav", np.full((144000, 1), 0.1), 48000)  # 1.5 bars at 120 bpm
    near = write_wav(tmp_path / "near.wav", np.full((96000 + 96, 1), 0.1), 48000)  # 2 ms past one bar
    silent = write_wav(tmp_path / "silent.wav", np.zeros(4800), 48000)
    for kwargs, message in (
        ({"bpm": 120.0, "bed": wrong_rate}, "never resamples"),
        ({"bpm": 120.0, "bed": uneven}, "1.500 bars at 120 bpm; a bed must be within 1 ms of a whole number of bars"),
        ({"bpm": 120.0, "bed": near}, "within 1 ms"),
        ({"bed": uneven}, "--with needs --bpm"),
        ({"against": wrong_rate}, "never resamples"),
        ({"against": silent}, "the sample is silent"),
        ({"blind": True}, "--blind needs --against"),
        ({"against": tmp_path / "missing.wav"}, "missing.wav"),
        ({"pattern": (1, 0)}, "--pattern needs --bpm"),
        ({"bpm": 120.0, "pattern": (0, 0)}, "at least one hit"),
        ({"bpm": 120.0, "pattern": (1, 2)}, "names patch 2, but there are 1"),
        ({"chokes": [(1, (1,))]}, "cannot choke itself"),
        ({"chokes": [(1, (3,))]}, "patch 3 is not among the 1"),
    ):
        with pytest.raises(A.AuditionError, match=message):
            A.Session(example, out=Collect([]), **kwargs)
    with pytest.raises(A.AuditionError, match="give one patch"):
        A.Session([example, example], against=silent, out=Collect([]))
    # 0.5 ms off whole bars is within tolerance.
    close = write_wav(tmp_path / "close.wav", np.full((96000 + 24, 1), 0.01), 48000)
    A.Session(example, bpm=120.0, bed=close, out=Collect([]))


def test_audition_reveals_the_blind_order_when_it_stops(tmp_path, example):
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", other, 48000)
    log = []

    class Stream:
        def __init__(self, rate, channels, fill):
            for _ in range(24):  # six seconds: six quarter-note hits per two seconds at 120 bpm
                fill(12000)

        def __enter__(self):
            raise KeyboardInterrupt

        def __exit__(self, *exc):
            return False

    code = A.audition(
        [example], bpm=120.0, against=sample, blind=True, stream_factory=Stream, out=Collect(log), rng=random.Random(3)
    )
    text = "".join(log)
    assert code == 0 and "blind order (R = render, S = sample): 1:" in text
    assert text.count(":RS") + text.count(":SR") == 6  # twelve quarter-note hits in six seconds: six pairs


def test_clipping_is_flagged(tmp_path, example):
    loud = write_wav(tmp_path / "loud.wav", np.full((96000, 1), 0.9), 48000)
    quiet = write_wav(tmp_path / "quiet.wav", np.full((96000, 1), 0.01), 48000)
    log = []
    A.Session(example, bpm=120.0, bed=loud, out=Collect(log))
    assert "warning: the hits over the bed peak at" in "".join(log)
    log = []
    A.Session(example, bpm=120.0, bed=quiet, out=Collect(log))
    assert "warning" not in "".join(log)
    # A quiet sample matched to a loud render would clip the device.
    other, _ = render(loads(OTHER))
    faint = write_wav(tmp_path / "faint.wav", other * 0.01 + np.pad([0.5], (0, other.size - 1)), 48000)
    log = []
    A.Session(example, against=faint, out=Collect(log))
    assert "which the device will clip" in "".join(log)


def test_a_closed_session_renders_and_says_nothing_more(example):
    log = []
    session = A.Session(example, watch=True, takes=2, out=Collect(log))
    session.close()
    count = len(log)
    edit(example, EXAMPLE.replace("db          = -6.0", "db          = -12.0"))
    session.poll()
    time.sleep(1.0)
    assert len(log) == count and not any(w._pending for w in session.watchers)
