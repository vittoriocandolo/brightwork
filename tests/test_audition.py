# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audition: the hit scheduler, patterns, chokes, beds, samples, watching and sessions (plan §7.5)."""

import importlib
import os
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
from numpy.typing import ArrayLike, NDArray

from brightwork_synth import audition
from brightwork_synth.loudness import momentary_max
from brightwork_synth.patch import choke_knots, loads
from brightwork_synth.render import quantize, render, render_takes

from .conftest import EXAMPLE

type Audio = NDArray[np.float32]

RATE = 1000  # a toy rate keeps the hit arithmetic readable
SESSION_RATE = 48000  # the example patch's
STEREO = 2
REFUSED = 1  # the exit status of an audition that cannot play
BAD_INPUT = 2  # the exit status of an invalid patch or setting


def play(hits: audition.Hits, frames: int, block: int) -> Audio:
    """Fill `frames` samples in blocks of `block`."""
    out = []
    while frames > 0:
        n = min(block, frames)
        out.append(hits.fill(n))
        frames -= n
    return np.concatenate(out)


def ones(n: int, value: float = 1.0) -> Audio:
    """Return n samples of a constant."""
    return np.full(n, value, dtype=np.float32)


def sample_of(session: audition.Session) -> NDArray[np.float32]:
    """Return the sample waiting for slot 0's next hit, matched to the render."""
    offer = session.hits.offered(0)
    assert offer is not None
    assert offer.sample is not None
    return offer.sample


class Collect:
    """A writer that keeps what audition says, chunk by chunk."""

    def __init__(self, sink: list[str]) -> None:
        """Keep chunks in `sink`."""
        self.sink = sink

    def write(self, s: str) -> None:
        """Keep a chunk."""
        self.sink.append(s)


# The hit scheduler.


def test_hits_repeat_after_each_hit_plus_half_a_second() -> None:
    """Repeat hits after each hit's length plus half a second, without a tempo."""
    buf = np.arange(1, 301, dtype=np.float32)
    hits = audition.Hits(RATE)
    hits.offer(0, 0, [buf])
    out = play(hits, 2000, 64)
    period = 300 + 500
    for start in (0, period, 2 * period):
        assert np.array_equal(out[start : start + 300], buf)
        assert np.all(out[start + 300 : start + period] == 0)


def test_nothing_plays_until_a_buffer_arrives() -> None:
    """Play silence until the first buffer arrives, then play it at once."""
    hits = audition.Hits(RATE)
    assert np.all(play(hits, 500, 100) == 0)
    hits.offer(0, 0, [ones(10)])
    out = play(hits, 100, 100)
    assert np.all(out[:10] == 1)
    assert np.all(out[10:] == 0)


def test_block_size_never_moves_a_hit() -> None:
    """Fill in blocks of any size, and find the same output."""
    buf = np.random.default_rng(0).normal(size=333).astype(np.float32)
    outs = []
    for block in (1, 7, 64, 1000, 4096):
        hits = audition.Hits(RATE, audition.Setup(bpm=97.0, every=0.5, pattern=(1, 0, 1, 1), chokes={0: {0}}))
        hits.offer(0, 0, [buf])
        outs.append(play(hits, 5000, block))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])


def test_new_buffer_takes_over_at_the_next_hit() -> None:
    """Let a new buffer take over at the next hit, the sounding one finishing with its own."""
    hits = audition.Hits(RATE)
    hits.offer(0, 0, [ones(300)])
    first = play(hits, 150, 50)
    new = 2.0
    hits.offer(0, 1, [ones(100, new)])  # arrives mid-hit
    out = np.concatenate((first, play(hits, 2000, 50)))
    assert np.all(out[:300] == 1.0)  # the playing hit finishes with its own buffer
    assert np.all(out[300:800] == 0.0)
    assert np.all(out[800:900] == new)  # the next hit plays the new buffer...
    assert np.all(out[900:1400] == 0.0)
    assert np.all(out[1400:1500] == new)  # ...and the period follows its length


def test_older_generations_never_replace_newer_ones() -> None:
    """Never let an older generation replace a newer one, nor a first take the whole set."""
    hits = audition.Hits(RATE)
    newer, older = 2.0, 1.0
    hits.offer(0, 2, [ones(10, newer)])
    hits.offer(0, 1, [ones(10, older)])
    assert np.all(play(hits, 10, 10) == newer)
    # Within a generation, the whole set (stage 1) replaces its first take (stage 0), and never the reverse.
    hits = audition.Hits(RATE)
    whole, first, later = 5.0, 4.0, 6.0
    hits.offer(0, (3, 1), [ones(10, whole)])
    hits.offer(0, (3, 0), [ones(10, first)])
    assert np.all(play(hits, 10, 10) == whole)
    hits.offer(0, (4, 0), [ones(10, later)])
    assert np.all(play(hits, 1000, 10)[500:510] == later)  # the next hit: 10 samples played, then 0.5 s


def test_steps_of_any_note_value_never_drift() -> None:
    """Place steps of any note value on the grid, rounded from the first bar, never drifting."""
    # At 120 bpm a beat is 500 samples: eighths every 250, a crash on 4/1 every four bars.
    for every, step in ((0.5, 250), (0.25, 125), (1.0, 500), (16.0, 8000)):
        hits = audition.Hits(RATE, audition.Setup(bpm=120.0, every=every))
        hits.offer(0, 0, [ones(1)])
        out = play(hits, 4 * step + 1, 333)
        assert np.array_equal(np.flatnonzero(out), [0, step, 2 * step, 3 * step, 4 * step])
    # Fractional step lengths never drift: step n starts at round(n · every · rate · 60 / bpm).
    hits = audition.Hits(RATE, audition.Setup(bpm=140.0, every=0.25))
    hits.offer(0, 0, [ones(1)])
    out = play(hits, 20000, 333)
    assert np.array_equal(np.flatnonzero(out), [round(i * 0.25 * RATE * 60 / 140) for i in range(187)])


def test_overlapping_tails_sum() -> None:
    """Sum overlapping tails."""
    hits = audition.Hits(RATE, audition.Setup(bpm=120.0))
    hits.offer(0, 0, [ones(900)])
    out = play(hits, 2000, 128)
    assert np.all(out[:500] == 1)
    assert np.all(out[500:900] == 2 * 1)  # two hits sound together
    assert np.all(out[900:1000] == 1)


def test_a_pattern_names_the_patch_of_every_step() -> None:
    """Play each step's patch from the pattern, resting where a patch has not rendered yet."""
    hits = audition.Hits(RATE, audition.Setup(2, bpm=120.0, every=0.5, pattern=(1, 1, 2, 0)))
    hits.offer(0, 0, [ones(1, 1.0)])
    hits.offer(1, 0, [ones(1, 2.0)])
    out = play(hits, 2000, 100)
    assert [out[250 * i] for i in range(8)] == [1, 1, 2, 0, 1, 1, 2, 0]
    # A patch that has not rendered yet rests on its steps.
    hits = audition.Hits(RATE, audition.Setup(2, bpm=120.0, every=0.5, pattern=(1, 2)))
    hits.offer(0, 0, [ones(1, 1.0)])
    out = play(hits, 1000, 100)
    assert [out[250 * i] for i in range(4)] == [1, 0, 1, 0]


def test_patches_cycle_hit_by_hit_and_takes_cycle_within_each() -> None:
    """Cycle patches hit by hit, and each patch's takes over its own hits."""
    hits = audition.Hits(RATE, audition.Setup(2, bpm=120.0))
    hits.offer(0, 0, [ones(1, 1.0), ones(1, 2.0)])
    hits.offer(1, 0, [ones(1, 10.0), ones(1, 20.0), ones(1, 30.0)])
    out = play(hits, 6000, 250)
    played = [out[500 * i] for i in range(12)]
    assert played == [1, 10, 2, 20, 1, 30, 2, 10, 1, 20, 2, 30]
    assert hits.log[:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_a_choke_fades_the_ringing_hit_with_the_curve_choke_writes() -> None:
    """Fade a choked hit by `choke`'s curve, and never a hit that starts after the choke."""
    # Patch 1 rings for 2 s; patch 2 chokes it at its step, 250 samples in, over a 40 ms release.
    ring, close = ones(2000, 1.0), ones(1, 0.0)
    hits = audition.Hits(
        RATE,
        audition.Setup(2, bpm=120.0, every=0.5, pattern=(1, 2, 0, 0, 0, 0, 0, 0), chokes={0: {1}}, release_ms=40.0),
    )
    hits.offer(0, 0, [ring])
    hits.offer(1, 0, [close])
    out = play(hits, 2000, 64)
    assert np.all(out[:250] == 1.0)
    knots = choke_knots(0.0, 40.0)
    curve = audition.fade(40.0, RATE)[0]
    t = np.arange(250, 2000) - 250
    np.testing.assert_allclose(out[250:2000], curve.on(t / RATE), rtol=0, atol=1e-7)
    assert out[250 + 20] == pytest.approx(10 ** (-3 / 2), rel=1e-3)  # −30 dB halfway through the release
    end = round(knots[-1][0])
    assert np.all(out[250 + end :] == 0.0)
    # A patch never chokes itself unless asked, and a choke only reaches hits already sounding.
    hits = audition.Hits(RATE, audition.Setup(2, bpm=120.0, every=0.5, pattern=(1, 1), chokes={0: {1}}))
    hits.offer(0, 0, [ones(600, 1.0)])
    hits.offer(1, 0, [ones(1, 0.0)])
    out = play(hits, 1000, 100)
    assert np.all(out[250:500] == 2 * 1.0)  # two hits of the first patch, neither choked
    assert np.all(out[500:600] == 3 * 1.0)


def test_stereo_takes_play_in_stereo_and_mono_hits_in_the_centre() -> None:
    """Play stereo take pairs in stereo and mono hits in the centre."""
    pair = np.stack([ones(10, 1.0), ones(10, -1.0)], axis=1)
    hits = audition.Hits(RATE, audition.Setup(2, bpm=120.0, channels=2))
    hits.offer(0, 0, [pair])
    centre = 0.5
    hits.offer(1, 0, [ones(10, centre)])
    out = play(hits, 1000, 64)
    assert out.shape == (1000, 2)
    assert np.array_equal(out[:10], pair)
    assert np.all(out[500:510] == centre)


# Beds and samples.


def test_a_bed_loops_under_the_hits_on_the_tempo_grid() -> None:
    """Loop a bed under the hits on the tempo grid, in blocks of any size."""
    bpm = 120.0
    bed = (np.arange(1, 4001, dtype=np.float32) / 10000)[:, None]  # two bars, mono
    hat = ones(10, 0.5)
    outs = []
    for block in (1, 64, 333, 4096):
        hits = audition.Hits(RATE, audition.Setup(bpm=bpm, every=0.5, bed=audition.Bed(bed, RATE, bpm, 2)))
        hits.offer(0, 0, [hat])
        outs.append(play(hits, 12000, block))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])
    expected = np.tile(bed[:, 0], 3)
    for step in range(48):
        expected[250 * step : 250 * step + 10] += 0.5
    assert outs[0].ndim == 1
    assert np.allclose(outs[0], expected, atol=1e-7)


def test_a_bed_a_little_short_of_whole_bars_never_drifts() -> None:
    """Restart a bed a sample short of whole bars on the grid, leaving the gap at its end."""
    bed = np.ones((3999, 1), dtype=np.float32)  # one sample short of two bars at 120 bpm
    hits = audition.Hits(RATE, audition.Setup(bpm=120.0, bed=audition.Bed(bed, RATE, 120.0, 2)))
    hits.offer(0, 0, [np.zeros(1, dtype=np.float32)])
    out = play(hits, 12000, 256)
    # Each loop restarts on the grid: the missing sample is a gap at the end of every period.
    assert np.array_equal(np.flatnonzero(out == 0), [3999, 7999, 11999])


def test_a_stereo_bed_opens_a_stereo_stream_with_the_hits_centred() -> None:
    """Open a stereo stream for a stereo bed, with mono hits in the centre."""
    bed = np.stack([np.full(2000, 0.1), np.full(2000, -0.1)], axis=1).astype(np.float32)
    hits = audition.Hits(RATE, audition.Setup(bpm=120.0, bed=audition.Bed(bed, RATE, 120.0, 1)))
    hat = np.linspace(0, 1, 50, dtype=np.float32)
    hits.offer(0, 0, [hat])
    out = play(hits, 4000, 100)
    assert hits.channels == STEREO
    assert out.shape == (4000, 2)
    centre = out - np.tile(bed, (2, 1))
    assert np.allclose(centre[:, 0], centre[:, 1])
    assert np.allclose(centre[500:550, 0], hat)


def test_hits_alternate_between_the_render_and_the_sample() -> None:
    """Alternate the render and the sample hit by hit."""
    hits = audition.Hits(RATE)
    rendered, sample = 1.0, 2.0
    hits.offer(0, 0, [ones(100, rendered)], ones(200, sample))
    out = play(hits, 3000, 64)
    # Without a tempo each hit repeats after its own length plus 0.5 s: render, sample, render, …
    assert np.all(out[:100] == rendered)
    assert np.all(out[600:800] == sample)
    assert np.all(out[1300:1400] == rendered)
    assert np.all(out[100:600] == 0.0)
    assert np.all(out[800:1300] == 0.0)
    assert hits.order == [True, True, True]


def test_blind_pairs_play_in_the_recorded_random_order() -> None:
    """Play each blind pair in a random order, as recorded, both orders occurring."""
    hits = audition.Hits(RATE, audition.Setup(bpm=120.0, rng=np.random.default_rng(7)))
    rendered, sample = 1.0, 2.0
    hits.offer(0, 0, [ones(10, rendered)], ones(10, sample))
    out = play(hits, 500 * 40, 128)
    played = [out[500 * i] for i in range(40)]
    for pair, render_first in enumerate(hits.order):
        assert played[2 * pair : 2 * pair + 2] == ([rendered, sample] if render_first else [sample, rendered])
    assert 0 < sum(hits.order) < len(hits.order)  # both orders occur


# Sessions.


def write_wav(path: Path, data: ArrayLike, rate: int, subtype: str = "FLOAT") -> Path:
    """Write a WAV with soundfile, and return its path."""
    soundfile.write(path, data, rate, subtype=subtype)
    return path


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Write the plan's example patch to a file."""
    path = tmp_path / "hat.toml"
    path.write_text(EXAMPLE, encoding="utf-8")
    return path


def wait_for(condition: Callable[[], object], timeout: float = 60.0) -> None:
    """Wait until a condition holds, failing after the timeout."""
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def said(log: list[str], generation: int) -> bool:
    """Tell whether audition has reported on a generation: its messages start with the tag."""
    return any(chunk.startswith(f"[{generation}] ") for chunk in log)


def edit(path: Path, text: str) -> None:
    """Write new text to a patch file, bumping its time so the change is seen within one clock tick."""
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


OTHER = EXAMPLE.replace("spacing_hz = 14.0", "spacing_hz = 25.0")


def test_audition_plays_the_exact_render_buffer(example: Path) -> None:
    """Play the exact buffer that `render` writes."""
    expected, _ = render(loads(EXAMPLE))
    session = audition.Session(example, out=Collect([]))
    assert session.rate == SESSION_RATE
    assert session.hits.channels == 1
    heard = np.concatenate([session.hits.fill(512) for _ in range(40)])
    assert np.array_equal(heard[: expected.size], expected)


def test_pcm_patches_play_their_quantized_samples(tmp_path: Path) -> None:
    """Play a PCM patch's dithered samples, as `render` writes them."""
    path = tmp_path / "pcm.toml"
    text = EXAMPLE.replace('"float32"     #', '"pcm16"       #')
    path.write_text(text, encoding="utf-8")
    audio, _ = render(loads(text))
    expected = quantize(audio, 16, 1, 1) / 2.0**15
    heard = audition.Session(path, out=Collect([])).hits.fill(20000)
    np.testing.assert_array_equal(heard[: expected.size], expected.astype(np.float32))


def test_the_first_take_plays_while_the_others_render(example: Path) -> None:
    """Play the first take at once, and the others as they render, from the same generation."""
    audio, _ = render_takes(loads(EXAMPLE), 3)
    session = audition.Session(example, audition.Settings(takes=3), out=Collect([]))
    # The session starts with the first take alone; the set follows from the same generation.
    wait_for(lambda: (offer := session.hits.offered(0)) is not None and offer.generation == (0, 1))
    period = audio[0].size + 24000
    heard = np.concatenate([session.hits.fill(4800) for _ in range(60)])
    for j in range(3):
        start = j * period
        assert np.array_equal(heard[start : start + audio[j].size], audio[j])


def test_a_peak_mode_set_waits_for_all_its_takes(tmp_path: Path) -> None:
    """Wait for every take of a peak-mode set, whose gain depends on all of them."""
    path = tmp_path / "peak.toml"
    text = EXAMPLE.replace('mode        = "gain"', 'mode        = "peak"').replace("db          = -6.0", "db = -1.0")
    path.write_text(text, encoding="utf-8")
    audio, _ = render_takes(loads(text), 3)
    session = audition.Session(path, audition.Settings(takes=3), out=Collect([]))
    offer = session.hits.offered(0)
    assert offer is not None
    assert offer.generation == (0, 0)
    assert len(offer.buffers) == len(audio)
    heard = session.hits.fill(audio[0].size)
    assert np.array_equal(heard, audio[0])  # the set's own gain from the first hit


def test_stereo_sessions_play_take_pairs(example: Path) -> None:
    """Play take pairs in stereo."""
    audio, _ = render_takes(loads(EXAMPLE), 1, stereo=True)
    session = audition.Session(example, audition.Settings(stereo=True), out=Collect([]))
    assert session.hits.channels == STEREO
    heard = session.hits.fill(audio[0].shape[0])
    assert heard.shape == audio[0].shape
    assert np.array_equal(heard, audio[0])


def test_several_patches_render_behind_the_first(tmp_path: Path, example: Path) -> None:
    """Render the other patches behind the first, each playing on its steps."""
    other = tmp_path / "other.toml"
    other.write_text(OTHER, encoding="utf-8")
    first, _ = render(loads(EXAMPLE))
    second, _ = render(loads(OTHER))
    session = audition.Session(
        [example, other],
        audition.Settings(bpm=120.0, every=0.5, pattern=(1, 2)),
        out=Collect([]),
    )
    wait_for(lambda: session.buffers[1])
    heard = np.concatenate([session.hits.fill(4800) for _ in range(10)])
    assert np.array_equal(heard[: first.size], first)  # step 1: the first patch
    step = 12000  # an eighth at 120 bpm
    np.testing.assert_allclose(heard[step : step + second.size], second, atol=1e-7)


def test_a_session_chokes_one_patch_with_another(tmp_path: Path, example: Path) -> None:
    """Choke one patch's hits with another's in a session."""
    ring = tmp_path / "open.toml"
    ring.write_text(EXAMPLE.replace("{ ms = 120.0, gain = 0.0 }", "{ ms = 900.0, gain = 0.0 }"), encoding="utf-8")
    session = audition.Session(
        [ring, example],
        audition.Settings(bpm=120.0, every=0.5, pattern=(1, 2), chokes=((1, (2,)),)),
        out=Collect([]),
    )
    wait_for(lambda: session.buffers[1])
    long, _ = render(loads(ring.read_text(encoding="utf-8")))
    hat, _ = render(loads(EXAMPLE))
    heard = np.concatenate([session.hits.fill(4800) for _ in range(10)])
    step = 12000
    curve = audition.fade(40.0, 48000)[0]
    want = long[step : step + 4800] * curve.on(np.arange(4800) / 48000) + hat[:4800]
    np.testing.assert_allclose(heard[step : step + 4800], want, atol=1e-6)


def test_audition_runs_until_interrupted_and_reports_what_it_cannot_play(example: Path) -> None:
    """Play until interrupted, and refuse an over, an invalid patch or an unsupported rate with its status."""
    seen: dict[str, object] = {}

    class Stream:
        def __init__(self, rate: int, channels: int, fill: audition.Fill) -> None:
            seen["rate"], seen["channels"], seen["first"] = rate, channels, fill(64)

        def __enter__(self) -> None:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            return None

    assert audition.audition([example], stream_factory=Stream, out=Collect([])) == 0
    assert seen["rate"] == SESSION_RATE
    assert seen["channels"] == 1
    assert audition.audition([example], audition.Settings(stereo=True), stream_factory=Stream, out=Collect([])) == 0
    assert seen["channels"] == STEREO
    example.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"), encoding="utf-8")
    assert audition.audition([example], stream_factory=Stream, out=Collect([])) == REFUSED  # over full scale
    example.write_text(EXAMPLE.replace("damping    = 0.5", "damping    = 9.0"), encoding="utf-8")
    assert audition.audition([example], stream_factory=Stream, out=Collect([])) == BAD_INPUT  # invalid

    def unsupported(rate: int, _channels: int, _fill: audition.Fill) -> Stream:
        msg = f"the output device does not support {rate} Hz"
        raise RuntimeError(msg)

    example.write_text(EXAMPLE, encoding="utf-8")
    assert audition.audition([example], stream_factory=unsupported, out=Collect([])) == REFUSED


def test_watcher_renders_only_the_newest_version(tmp_path: Path) -> None:
    """Render only the newest version of a file, dropping one superseded while it rendered."""
    path = tmp_path / "hat.toml"
    path.write_text("v0", encoding="utf-8")
    gate = threading.Event()
    rendered: list[str] = []
    results: list[tuple[int, object, bool]] = []

    def work(text: str, _generation: int) -> object:
        rendered.append(text)
        gate.wait(5)
        return text

    def done(generation: int, result: object, _elapsed: float, *, newest: bool) -> None:
        results.append((generation, result, newest))

    w = audition.Watcher(path, work, done, text="v0")
    w.poll()
    assert w.generation == 0  # unchanged content starts nothing
    for i in range(1, 5):
        path.write_text(f"v{i}", encoding="utf-8")
        w.poll()
        time.sleep(0.05)
    gate.set()
    wait_for(lambda: results and results[-1][0] == len(rendered) + 2, 5)
    # v1 was already rendering; v2 and v3 were superseded before they started.
    assert rendered == ["v1", "v4"]
    assert results == [(1, "v1", False), (4, "v4", True)]


def test_watching_swaps_buffers_and_keeps_the_last_valid_one(example: Path) -> None:
    """Swap in each valid edit at the next hit, keeping the last valid buffer through an invalid one."""
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True), out=Collect(log))
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
    assert "[2] rendered in" in "".join(log)
    assert "plays from its next hit" in "".join(log)
    assert session.hits.generation == 0  # nothing changes until the next hit
    expected, _ = render(loads(quieter))
    rest = np.concatenate([session.hits.fill(1000) for _ in range(60)])
    heard = np.concatenate((original, rest))
    old, _ = render(loads(EXAMPLE))
    assert np.array_equal(heard[: old.size], old)
    start = old.size + 24000  # the next hit: the rendered length plus 0.5 s
    assert np.all(heard[old.size : start] == 0)
    assert np.array_equal(heard[start : start + expected.size], expected)
    assert session.hits.generation == len(("invalid", "quieter"))


def test_a_session_matches_the_sample_and_plays_over_the_bed(tmp_path: Path, example: Path) -> None:
    """Match the sample to the render's loudness, and play the render exactly over the bed."""
    audio, report = render(loads(EXAMPLE))
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", np.stack([other, other * 0.5], axis=1) * 0.3, 48000, "PCM_24")
    bed = write_wav(tmp_path / "bed.wav", np.full((96000, 2), 0.01), 48000)  # one bar at 120 bpm, stereo
    log: list[str] = []
    session = audition.Session(example, audition.Settings(bpm=120.0, bed=bed, against=sample), out=Collect(log))
    text = "".join(log)
    assert "bed      " in text
    assert "1 bar at 120 bpm, stereo" in text
    assert "sample   plays at" in text
    assert session.hits.channels == STEREO
    matched = sample_of(session)
    assert matched.ndim == 1  # a mono render: the sample plays as its mid channel
    assert momentary_max(matched, SESSION_RATE) == pytest.approx(report.files[0].loudness, abs=0.01)
    heard = session.hits.fill(48000)  # step 1: the render, exact, over the bed
    assert np.allclose(heard[: audio.size, 0] - 0.01, audio, atol=1e-7)
    assert np.allclose(heard[audio.size + 100 : 24000], 0.01)


def test_a_stereo_session_plays_a_stereo_sample_in_stereo(tmp_path: Path, example: Path) -> None:
    """Play a stereo sample in stereo, and a mono one centred, matched to the pair."""
    pair, _ = render_takes(loads(OTHER), 1, stereo=True)
    sample = write_wav(tmp_path / "sample.wav", pair[0] * 0.3, 48000)
    session = audition.Session(example, audition.Settings(stereo=True, against=sample), out=Collect([]))
    matched = sample_of(session)
    assert matched.shape == pair[0].shape
    mine = render_takes(loads(EXAMPLE), 1, stereo=True)[1].files[0].loudness
    assert momentary_max(matched, SESSION_RATE) == pytest.approx(mine, abs=0.01)
    mono = write_wav(tmp_path / "mono.wav", pair[0][:, 0] * 0.3, 48000)
    centred = sample_of(audition.Session(example, audition.Settings(stereo=True, against=mono), out=Collect([])))
    assert centred.shape[1] == STEREO
    assert np.array_equal(centred[:, 0], centred[:, 1])


def test_the_sample_is_rematched_on_every_re_render(tmp_path: Path, example: Path) -> None:
    """Match the sample again whenever the render changes."""
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", other, 48000)
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True, against=sample), out=Collect(log))
    before = sample_of(session)
    quieter = EXAMPLE.replace("db          = -6.0", "db          = -18.0")
    edit(example, quieter)
    session.poll()
    wait_for(lambda: said(log, 1))
    after = sample_of(session)
    assert 20 * np.log10(np.max(np.abs(before)) / np.max(np.abs(after))) == pytest.approx(12.0, abs=0.01)


def test_a_session_refuses_what_it_cannot_play_faithfully(tmp_path: Path, example: Path) -> None:
    """Refuse beds and samples at another rate or off whole bars, and settings that cannot go together."""
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
        ({"against": write_wav(tmp_path / "nan.wav", np.full(4800, np.nan), 48000)}, "not numbers or are infinite"),
        ({"bpm": 120.0, "bed": write_wav(tmp_path / "inf.wav", np.full(96000, np.inf), 48000)}, "are infinite"),
        ({"pattern": (1, 0)}, "--pattern needs --bpm"),
        ({"bpm": 120.0, "pattern": (0, 0)}, "at least one hit"),
        ({"bpm": 120.0, "pattern": (1, 2)}, "names patch 2, but there are 1"),
        ({"chokes": ((1, (1,)),)}, "cannot choke itself"),
        ({"chokes": ((1, (3,)),)}, "patch 3 is not among the 1"),
    ):
        with pytest.raises(audition.AuditionError, match=message):
            audition.Session(example, audition.Settings(**kwargs), out=Collect([]))
    with pytest.raises(audition.AuditionError, match="give one patch"):
        audition.Session([example, example], audition.Settings(against=silent), out=Collect([]))
    # 0.5 ms off whole bars is within tolerance.
    close = write_wav(tmp_path / "close.wav", np.full((96000 + 24, 1), 0.01), 48000)
    audition.Session(example, audition.Settings(bpm=120.0, bed=close), out=Collect([]))


def test_audition_reveals_the_blind_order_when_it_stops(tmp_path: Path, example: Path) -> None:
    """Reveal the blind order of every pair when audition stops."""
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", other, 48000)
    log: list[str] = []

    class Stream:
        def __init__(self, _rate: int, _channels: int, fill: audition.Fill) -> None:
            for _ in range(24):  # six seconds: six quarter-note hits per two seconds at 120 bpm
                fill(12000)

        def __enter__(self) -> None:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            return None

    code = audition.audition(
        [example],
        audition.Settings(bpm=120.0, against=sample, blind=True),
        stream_factory=Stream,
        out=Collect(log),
        rng=np.random.default_rng(3),
    )
    text = "".join(log)
    assert code == 0
    assert "blind order (R = render, S = sample): 1:" in text
    assert text.count(":RS") + text.count(":SR") == 12 // 2  # twelve quarter-note hits in six seconds: six pairs


def test_clipping_is_flagged(tmp_path: Path, example: Path) -> None:
    """Warn when the hits over the bed, or a matched sample, would clip the device."""
    loud = write_wav(tmp_path / "loud.wav", np.full((96000, 1), 0.9), 48000)
    quiet = write_wav(tmp_path / "quiet.wav", np.full((96000, 1), 0.01), 48000)
    log: list[str] = []
    audition.Session(example, audition.Settings(bpm=120.0, bed=loud), out=Collect(log))
    assert "warning: the hits over the bed peak at" in "".join(log)
    log = []
    audition.Session(example, audition.Settings(bpm=120.0, bed=quiet), out=Collect(log))
    assert "warning" not in "".join(log)
    # A quiet sample matched to a loud render would clip the device.
    other, _ = render(loads(OTHER))
    faint = write_wav(tmp_path / "faint.wav", other * 0.01 + np.pad([0.5], (0, other.size - 1)), 48000)
    log = []
    audition.Session(example, audition.Settings(against=faint), out=Collect(log))
    assert "which the device will clip" in "".join(log)


def test_a_closed_session_renders_and_says_nothing_more(example: Path) -> None:
    """Render and say nothing more once a session is closed."""
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True, takes=2), out=Collect(log))
    session.close()
    count = len(log)
    edit(example, EXAMPLE.replace("db          = -6.0", "db          = -12.0"))
    session.poll()
    time.sleep(1.0)
    assert len(log) == count
    assert not any(w.pending() for w in session.watchers)


def test_stereo_pcm_patches_dither_each_channel_with_its_take(tmp_path: Path) -> None:
    """Play a stereo PCM pair's dithered samples, each channel dithered as its own take."""
    path = tmp_path / "pcm.toml"
    text = EXAMPLE.replace('"float32"     #', '"pcm24"       #')
    path.write_text(text, encoding="utf-8")
    pair, _ = render_takes(loads(text), 1, stereo=True)
    expected = np.stack([quantize(pair[0][:, c], 24, 1, take) for c, take in enumerate((1, 2))], axis=1) / 2.0**23
    heard = audition.Session(path, audition.Settings(stereo=True), out=Collect([])).hits.fill(pair[0].shape[0])
    np.testing.assert_array_equal(heard, expected.astype(np.float32))


def test_an_empty_offer_silences_its_slot() -> None:
    """Rest on every step while the only slot holds no buffers, and play again once it does."""
    hits = audition.Hits(RATE, audition.Setup(bpm=120.0))
    hits.offer(0, 0, [ones(10)])
    assert np.all(play(hits, 500, 100)[:10] == 1)
    hits.offer(0, 1, [])
    assert np.all(play(hits, 1000, 100) == 0)
    hits.offer(0, 2, [ones(10)])
    assert np.all(play(hits, 500, 100)[:10] == 1)


def test_a_watcher_waits_out_unreadable_files_and_stops_cleanly(tmp_path: Path) -> None:
    """Start nothing for an unchanged, missing or undecodable file, and drop a render finished after stopping."""
    path = tmp_path / "hat.toml"
    path.write_text("v0", encoding="utf-8")
    gate, started = threading.Event(), threading.Event()
    results: list[int] = []

    def work(text: str, _generation: int) -> object:
        started.set()
        gate.wait(5)
        return text

    def done(generation: int, _result: object, _elapsed: float, *, newest: bool) -> None:
        results.append(generation if newest else -generation)

    w = audition.Watcher(path, work, done, text="v0")
    w.poll()
    w.poll()  # the same stamp: not even read
    path.write_bytes(b"\xff\xfe\xfd")  # a UTF-16 BOM, then half a character
    w.poll()
    path.unlink()
    w.poll()
    assert w.generation == 0
    path.write_text("v1", encoding="utf-8")
    w.poll()
    started.wait(5)
    w.stop()
    gate.set()
    w.finish("v2")
    time.sleep(0.2)
    assert results == []
    assert not w.pending()


def test_a_session_says_why_a_render_does_not_play(example: Path, tmp_path: Path) -> None:
    """Say a render was dropped, failed, is over full scale, or renders at another rate."""
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True), out=Collect(log))
    done = session.watchers[0].done

    def rendered(text: str) -> audition.Rendered:
        audio, report = render_takes(loads(text), 1)
        return audition.playable(audio, report), report

    done(1, rendered(EXAMPLE), 0.1, newest=False)
    done(2, RuntimeError("out of memory"), 0.1, newest=True)
    done(3, rendered(EXAMPLE.replace("db          = -6.0", "db          = 12.0")), 0.1, newest=True)
    done(4, rendered(EXAMPLE.replace("sample_rate = 48000", "sample_rate = 44100")), 0.1, newest=True)
    text = "".join(log)
    assert "[1] dropped: a newer edit is rendering" in text
    assert "[2] render failed, still playing the last valid one: out of memory" in text
    assert "[3] over full scale by" in text
    assert "[4] sample rate 44100 Hz, but audition plays at 48000 Hz" in text
    waiting = session.hits.offered(0)
    assert waiting is not None
    assert waiting.generation[0] == 0  # the first render still waits to play
    garbled = tmp_path / "garbled.toml"
    garbled.write_bytes(b"seed = \xff\n")
    with pytest.raises(audition.AuditionError, match="not UTF-8 or UTF-16 text"):
        audition.Session(garbled, out=Collect([]))
    with pytest.raises(audition.AuditionError, match="No such file"):
        audition.Session(tmp_path / "missing.toml", out=Collect([]))


def test_an_edit_plays_its_first_take_while_the_others_render(
    example: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offer an edit's first take on its own, then the whole set, from the same generation; nothing when over."""
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True, takes=2), out=Collect(log))
    wait_for(lambda: (offer := session.hits.offered(0)) is not None and offer.generation == (0, 1))
    offered: list[object] = []
    offer = session.hits.offer

    def recorded(
        slot: int,
        generation: int | audition.Generation,
        buffers: list[audition.Audio],
        sample: audition.Audio | None = None,
    ) -> None:
        offered.append(generation)
        offer(slot, generation, buffers, sample)

    monkeypatch.setattr(session.hits, "offer", recorded)
    edit(example, EXAMPLE.replace("db          = -6.0", "db          = -12.0"))
    session.poll()
    wait_for(lambda: (1, 1) in offered)
    assert offered == [(1, 0), (1, 1)]
    # An edit over full scale offers nothing, not even its first take.
    edit(example, EXAMPLE.replace("db          = -6.0", "db          = 12.0"))
    session.poll()
    wait_for(lambda: said(log, 2))
    assert "[2] over full scale by" in "".join(log)
    assert offered == [(1, 0), (1, 1)]


def test_a_bed_must_be_mono_or_stereo(tmp_path: Path, example: Path) -> None:
    """Refuse a bed of more than two channels."""
    surround = write_wav(tmp_path / "surround.wav", np.full((96000, 3), 0.1), 48000)
    with pytest.raises(audition.AuditionError, match="3 channels; a bed must be mono or stereo"):
        audition.Session(example, audition.Settings(bpm=120.0, bed=surround), out=Collect([]))


def test_each_blind_pair_is_announced_as_it_starts(tmp_path: Path, example: Path) -> None:
    """Announce every blind pair once, as its first hit starts."""
    other, _ = render(loads(OTHER))
    sample = write_wav(tmp_path / "sample.wav", other, 48000)
    log: list[str] = []
    settings = audition.Settings(bpm=120.0, against=sample, blind=True)
    session = audition.Session(example, settings, out=Collect(log), rng=np.random.default_rng(1))
    for _ in range(3):
        session.hits.fill(24000)  # three quarter notes at 120 bpm: pairs 1 and 2 have started
        session.poll()
    assert [chunk for chunk in log if chunk.startswith("pair")] == ["pair 1\n", "pair 2\n"]


def fake_device(rate: int, opened: dict[str, object]) -> SimpleNamespace:
    """Stand in for the sounddevice module: an output device that supports one rate, keeping what it opens."""

    class PortAudioError(Exception):
        """PortAudio's error."""

    def check_output_settings(*, samplerate: int, channels: int, dtype: str) -> None:
        if samplerate != rate:
            msg = f"Invalid sample rate {samplerate} for {channels} channels of {dtype}"
            raise PortAudioError(msg)

    def output_stream(**settings: object) -> dict[str, object]:
        opened.update(settings)
        return settings

    return SimpleNamespace(
        PortAudioError=PortAudioError,
        check_output_settings=check_output_settings,
        OutputStream=output_stream,
    )


def test_the_device_stream_plays_what_the_hits_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the device at the patch's rate with a callback that mixes; refuse a missing package or another rate."""
    opened: dict[str, object] = {}
    device = fake_device(SESSION_RATE, opened)
    real = importlib.import_module

    def importer(name: str, package: str | None = None) -> object:
        return device if name == "sounddevice" else real(name, package)

    monkeypatch.setattr(importlib, "import_module", importer)
    hits = audition.Hits(SESSION_RATE, audition.Setup(channels=2))
    hits.offer(0, 0, [np.stack([ones(8), ones(8, 0.5)], axis=1)])
    audition.device_stream(SESSION_RATE, 2, hits.fill)
    assert opened["samplerate"] == SESSION_RATE
    callback = opened["callback"]
    assert callable(callback)
    out = np.zeros((4, 2), dtype=np.float32)
    callback(out, 4, None, None)
    assert np.array_equal(out, [[1.0, 0.5]] * 4)
    with pytest.raises(RuntimeError, match="does not support 44100 Hz"):
        audition.device_stream(44100, 1, hits.fill)

    def missing(name: str, package: str | None = None) -> object:
        if name != "sounddevice":
            return real(name, package)
        msg = "No module named 'sounddevice'"
        raise ImportError(msg)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="audition needs the sounddevice package and PortAudio"):
        audition.device_stream(SESSION_RATE, 1, hits.fill)


def test_audition_polls_for_edits_until_interrupted(example: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll the session between sleeps while the stream plays, until Ctrl+C."""
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == len(("poll", "interrupt")):
            raise KeyboardInterrupt

    class Stream:
        def __init__(self, _rate: int, _channels: int, _fill: audition.Fill) -> None:
            pass

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(time, "sleep", sleep)
    settings = audition.Settings(watch=True)
    assert audition.audition([example], settings, stream_factory=Stream, out=Collect([])) == 0
    assert sleeps == [audition.POLL_S, audition.POLL_S]


def test_a_pcm_over_is_refused_with_what_to_lower(tmp_path: Path) -> None:
    """Refuse a 16-bit patch over its ceiling as an over, saying what to lower, not as a failed quantization."""
    path = tmp_path / "loud.toml"
    loud = EXAMPLE.replace('"float32"     #', '"pcm16"       #').replace("db          = -6.0", "db          = 12.0")
    path.write_text(loud, encoding="utf-8")
    with pytest.raises(audition.OverError, match=r"lower output\.db"):
        audition.Session(path, out=Collect([]))


def test_a_refused_device_stops_every_watcher(tmp_path: Path, example: Path) -> None:
    """Stop rendering and saying anything once the device refuses the stream."""
    other = tmp_path / "other.toml"
    other.write_text(OTHER, encoding="utf-8")
    log: list[str] = []

    def refused(_rate: int, _channels: int, _fill: audition.Fill) -> AbstractContextManager[object]:
        msg = "no device"
        raise RuntimeError(msg)

    code = audition.audition([example, other], stream_factory=refused, out=Collect(log))
    assert code == REFUSED
    said_then = len(log)
    time.sleep(1.0)  # the second patch rendered behind the first would report here
    assert len(log) == said_then
