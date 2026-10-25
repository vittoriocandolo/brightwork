# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audition: play the exact render buffers in context, and re-render on every edit (plan §7.5).

Hits are one-shots: each plays its buffer to the end, so overlapping tails
sum as in a DAW, unless a choke fades it. With a tempo, hits land on a grid
of note values, a 1/8 or 1/16 for hats, a 1/4 for a ride, 4/1 for a crash
once every four bars; without one, they repeat every rendered length plus
0.5 s. A pattern says which patch plays on each step, with rests; without
one, patches cycle hit by hit. Each patch's takes cycle over its own hits. A
new buffer takes over at the next hit that plays its patch.

A choke (`--choke A:B,…`) fades every hit of patch A still sounding when B
plays, over the release, with the curve `brightwork choke` writes: 60 dB in six
even steps of 10 dB, then silence one step later. A kit asks its sampler for
the same (off_mode=normal with ampeg_release), so a hi-hat pattern is heard
as it will play; samplers' release curves differ in detail.

The watcher tags each file change with an increasing generation and renders
it in the background. A finished render is used only if its generation is
still the newest, so stale results are dropped; an invalid edit, or a
gain-mode render over full scale, leaves the last valid buffer playing. With
several takes, the first file plays as soon as it is ready while the others
render behind it, when the set's gain does not depend on them.

A bed (`--with`), such as the track's loop, plays under the hits and restarts
on the tempo grid, so it never drifts from them. A sample (`--against`)
alternates with the render hit by hit, matched to the render's loudness and
rematched on every re-render: the sample is scaled, never the render. With
`--blind` each pair plays in random order, revealed when audition stops.
With `--stereo`, renders play as take pairs and a stereo sample in stereo;
otherwise a stereo sample plays as its mid channel. Nothing is ever
resampled.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import wav
from .curve import amplitude_curve
from .loudness import momentary_max
from .patch import DEFAULT_RELEASE_MS, PatchError, choke_knots, loads
from .render import FORMATS, RenderError, db, quantize, render_takes

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars


class AuditionError(Exception):
    pass


def load_audio(path) -> tuple[int, np.ndarray]:
    """A float or PCM WAV as float64 frames × channels, never resampled."""
    try:
        w = wav.read(path)
    except (OSError, ValueError) as e:
        raise AuditionError(f"{path}: {e}") from None
    x = w.as_float()
    return w.rate, x if x.ndim == 2 else x[:, None]


def playable(audio: list[np.ndarray], report) -> list[np.ndarray]:
    """What `render` writes, as float32: PCM formats add only their quantization, each channel with its take's."""
    bits = FORMATS[report.format]
    if bits is None:
        return audio
    scale = float(2 ** (bits - 1))
    out = []
    for a, f in zip(audio, report.files):
        takes = [report.takes[i].take for i in f.takes]
        if a.ndim == 1:
            q = quantize(a, bits, report.seed, takes[0]) / scale
        else:
            q = np.stack([quantize(a[:, c], bits, report.seed, t) / scale for c, t in enumerate(takes)], axis=1)
        out.append(q.astype(np.float32))
    return out


def fade(release_ms: float, rate: int):
    """The gain of a choked hit from the moment it is choked, in samples: `choke`'s damper from 0 ms."""
    knots = choke_knots(0.0, release_ms)
    curve = amplitude_curve([ms / 1000 for ms, _ in knots], [g for _, g in knots])
    return curve, math.ceil(knots[-1][0] / 1000 * rate)


class Bed:
    """A loop under the hits, restarted every `bars` bars on the tempo grid.

    Loop k starts at round(k · bars · 4 · 60 / bpm · rate), the same rounding as
    the hits, and plays as much of the bed as fits before the next start.
    """

    def __init__(self, frames: np.ndarray, rate: int, bpm: float, bars: int):
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.channels = self.frames.shape[1]
        self.rate, self.bpm, self.bars = rate, bpm, bars

    def start(self, k: int) -> int:
        return round(k * self.bars * 4 * self.rate * 60 / self.bpm)

    def fill(self, clock: int, frames: int) -> np.ndarray:
        """The bed over [clock, clock + frames), with the clock counted from the first bar."""
        out = np.zeros((frames, self.channels), dtype=np.float32)
        k = max(0, int(clock // (self.bars * 4 * self.rate * 60 / self.bpm)) - 1)
        while self.start(k) < clock + frames:
            s, e = self.start(k), min(self.start(k + 1), self.start(k) + len(self.frames))
            a, b = max(clock, s), min(clock + frames, e)
            if a < b:
                out[a - clock : b - clock] += self.frames[a - s : b - s]
            k += 1
        return out


class Voice:
    """One hit that is sounding: its buffer from `start`, faded from `choked` on if a choke caught it."""

    __slots__ = ("start", "buffer", "slot", "choked")

    def __init__(self, start: int, buffer: np.ndarray, slot: int):
        self.start, self.buffer, self.slot, self.choked = start, buffer, slot, None

    @property
    def end(self) -> int:
        return self.start + len(self.buffer)


class Hits:
    """Mixes hits into output blocks. `offer` may be called from any thread.

    Slot i holds patch i's take buffers, mono or (frames, 2). With a tempo,
    step n starts at round(n · every · 60 / bpm · rate), every in beats of a
    4/4 bar; the pattern names the slot of each step, from 1, with 0 for a
    rest, and without one the slots that have a buffer take turns. `chokes`
    maps a slot to the slots that choke it, from 0. With a sample (slot 0
    only), hits alternate between the render and the sample in pairs; `rng`
    makes each pair's order random, and `order` records it (True when the
    render played first).
    """

    def __init__(
        self,
        rate: int,
        slots: int = 1,
        bpm: float | None = None,
        every: float = 1.0,
        pattern=None,
        chokes=None,
        release_ms: float = DEFAULT_RELEASE_MS,
        *,
        channels: int = 1,
        bed: Bed | None = None,
        rng=None,
    ):
        self.rate, self.bpm, self.every, self.bed, self.rng = rate, bpm, every, bed, rng
        self.pattern = tuple(pattern) if pattern else None
        self.chokes = {a: frozenset(bs) for a, bs in (chokes or {}).items()}
        self.fade, self.fade_length = fade(release_ms, rate)
        self.channels = max(channels, bed.channels if bed is not None else 1)
        self._offers: list[tuple | None] = [None] * slots
        self._lock = threading.Lock()
        self.generations: list[tuple] = [(-1, 0)] * slots
        self.buffers: list[list[np.ndarray] | None] = [None] * slots
        self.sample: np.ndarray | None = None
        self.played = [0] * slots  # hits per slot, which pick its take
        self.last_slot = -1
        self.voices: list[Voice] = []
        self.clock = 0  # index of the next output sample, counted from the first bar
        self.started = False
        self.step = 0
        self.next_hit: int | None = None
        self.count = 0  # hits started
        self.order: list[bool] = []
        self.log: list[tuple[int, int]] = []  # (slot, take) of each hit, the sample being slot −1

    @property
    def generation(self) -> int:
        return self.generations[0][0]

    def offer(self, slot: int, generation, buffers: list[np.ndarray], sample: np.ndarray | None = None) -> None:
        """Queue a slot's take buffers, and the sample matched to them, for that slot's next hit.

        `generation` is an int or a (generation, stage) pair; an older one never replaces a newer one.
        """
        key = generation if isinstance(generation, tuple) else (generation, 0)
        with self._lock:
            current = self._offers[slot]
            if current is None or key > current[0]:
                self._offers[slot] = (key, list(buffers), sample)

    def _take_offers(self) -> None:
        with self._lock:
            for i, offered in enumerate(self._offers):
                if offered is not None and offered[0] > self.generations[i]:
                    self.generations[i], self.buffers[i] = offered[0], offered[1]
                    if i == 0:
                        self.sample = offered[2]

    def _step_time(self, n: int) -> int:
        return round(n * self.every * self.rate * 60 / self.bpm)

    def _slot_for_step(self, n: int) -> int | None:
        """The slot that plays step n, or None for a rest or a patch not rendered yet."""
        if self.pattern is not None:
            slot = self.pattern[n % len(self.pattern)] - 1
            return slot if slot >= 0 and self.buffers[slot] else None
        ready = [i for i, b in enumerate(self.buffers) if b]
        if not ready:
            return None
        slot = next((i for i in ready if i > self.last_slot), ready[0])
        self.last_slot = slot
        return slot

    def _buffer(self, slot: int) -> tuple[int, np.ndarray]:
        """The slot's next take, or the sample in its turn; returns the slot that sounds (−1 for the sample)."""
        if slot == 0 and self.sample is not None:
            if self.count % 2 == 0:
                self.order.append(True if self.rng is None else self.rng.random() < 0.5)
            if (self.count % 2 == 0) != self.order[-1]:
                self.log.append((-1, 0))
                return -1, self.sample
        takes = self.buffers[slot]
        take = self.played[slot] % len(takes)
        self.played[slot] += 1
        self.log.append((slot, take))
        return slot, takes[take]

    def _start_hit(self, at: int, slot: int) -> np.ndarray:
        sounding, buf = self._buffer(slot)
        if sounding >= 0:
            for v in self.voices:
                if v.choked is None and v.start < at and sounding in self.chokes.get(v.slot, ()):
                    v.choked = at
        self.voices.append(Voice(at, buf, sounding))
        self.count += 1
        return buf

    def fill(self, frames: int) -> np.ndarray:
        if not self.started:
            self._take_offers()
            if not any(self.buffers):
                return np.zeros(frames if self.channels == 1 else (frames, self.channels), dtype=np.float32)
            self.started = True
            self.next_hit = 0
        out = np.zeros((frames, self.channels), dtype=np.float32)
        end = self.clock + frames
        while self.next_hit is not None and self.next_hit < end:
            self._take_offers()
            slot = self._slot_for_step(self.step)
            buf = self._start_hit(self.next_hit, slot) if slot is not None else None
            self.step += 1
            if self.bpm:
                self.next_hit = self._step_time(self.step)
            elif buf is not None:
                self.next_hit += len(buf) + round(GAP_S * self.rate)
            else:
                self.next_hit += round(GAP_S * self.rate)
        playing = []
        for v in self.voices:
            stop = v.end if v.choked is None else min(v.end, v.choked + self.fade_length)
            a, b = max(self.clock, v.start), min(end, stop)
            if a < b:
                part = v.buffer[a - v.start : b - v.start]
                if v.choked is not None:
                    gain = self.fade.on((np.arange(a, b) - v.choked) / self.rate).astype(np.float32)
                    part = part * (gain if part.ndim == 1 else gain[:, None])
                out[a - self.clock : b - self.clock] += part if part.ndim == 2 else part[:, None]
            if stop > end:
                playing.append(v)
        self.voices = playing
        if self.bed is not None:
            out += self.bed.fill(self.clock, frames)  # a mono bed in a stereo stream plays in the centre
        self.clock = end
        return out[:, 0] if self.channels == 1 else out


class Watcher:
    """Renders the newest version of a file in the background."""

    def __init__(
        self,
        path,
        work: Callable[[str, int], object],
        done: Callable[[int, object, float, bool], None],
        text: str | None = None,
    ):
        self.path, self.work, self.done = path, work, done
        self.generation = 0
        self._stamp, self._text = None, text  # `text`: the version already rendered, if any
        self._pending: tuple[int, str] | None = None
        self._stopped = False
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def poll(self) -> None:
        """Start a new generation if the file's content changed."""
        try:
            st = os.stat(self.path)
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._stamp:
                return
            with open(self.path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        self._stamp = stamp
        if text == self._text:
            return
        self._text = text
        self.request(text)

    def request(self, text: str) -> None:
        with self._cv:
            if self._stopped:
                return
            self.generation += 1
            self._pending = (self.generation, text)  # an older request not yet started is superseded
            self._cv.notify()

    def finish(self, text: str) -> None:
        """Render the current generation again in the background, unless an edit supersedes it."""
        with self._cv:
            if self._stopped:
                return
            self._pending = (self.generation, text)
            self._cv.notify()

    def newest(self, generation: int) -> bool:
        with self._cv:
            return generation == self.generation and not self._stopped

    def stop(self) -> None:
        """Start nothing more; a render already running finishes, and its result is dropped."""
        with self._cv:
            self._stopped = True
            self._pending = None
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    return
                generation, text = self._pending
                self._pending = None
            start = time.perf_counter()
            try:
                result = self.work(text, generation)
            except Exception as e:  # reported by `done`; the last valid buffer keeps playing
                result = e
            elapsed = time.perf_counter() - start
            with self._cv:
                if self._stopped:
                    return
            self.done(generation, result, elapsed, self.newest(generation))


class OverError(Exception):
    def __init__(self, path, report):
        super().__init__(f"{path}: over full scale by {db(report.peak):.2f} dB: lower output.db to audition it")


def _chokes(rules, slots: int) -> dict[int, set[int]]:
    """--choke A:B,… rules, numbered from 1, as slot → the slots that choke it."""
    out: dict[int, set[int]] = {}
    for target, by in rules:
        for n in (target, *by):
            if not 1 <= n <= slots:
                raise AuditionError(f"--choke {target}:{','.join(map(str, by))}: patch {n} is not among the {slots}")
        if target in by:
            raise AuditionError(f"--choke {target}:{','.join(map(str, by))}: a patch cannot choke itself")
        out.setdefault(target - 1, set()).update(b - 1 for b in by)
    return out


class Session:
    """One audition: the hits, the renders behind them, and the watchers. `poll` is called about every 0.1 s."""

    def __init__(
        self,
        paths,
        *,
        watch: bool = False,
        bpm: float | None = None,
        every: float = 1.0,
        pattern=None,
        chokes=(),
        release_ms: float = DEFAULT_RELEASE_MS,
        bed=None,
        against=None,
        blind: bool = False,
        takes: int = 1,
        stereo: bool = False,
        rng=None,
        out=None,
    ):
        out = out or sys.stdout
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        self.out, self.takes, self.stereo, self.bpm = out, takes, stereo, bpm
        if against is not None and len(self.paths) > 1:
            raise AuditionError("--against compares one patch with a sample; give one patch")
        if blind and against is None:
            raise AuditionError("--blind needs --against: a blind comparison needs a sample")
        if bed is not None and bpm is None:
            raise AuditionError("--with needs --bpm, so the hits land on the bed's beats")
        if pattern is not None:
            if bpm is None:
                raise AuditionError("--pattern needs --bpm: a pattern's steps are on the tempo grid")
            if not any(pattern):
                raise AuditionError("--pattern needs at least one hit")
            if max(pattern) > len(self.paths):
                raise AuditionError(f"--pattern names patch {max(pattern)}, but there are {len(self.paths)}")
        choke_map = _chokes(chokes, len(self.paths))
        texts = [self._text(p) for p in self.paths]
        audio, report = self._render(texts[0], first=True)
        self._say(0, None, str(report))
        if report.over:
            raise OverError(self.paths[0], report)
        self.rate = report.sample_rate
        self.bed = self._bed(bed, bpm) if bed is not None else None
        self.sample, self.sample_lufs = self._sample(against) if against is not None else (None, None)
        self.blind = blind
        self.hits = Hits(
            self.rate,
            len(self.paths),
            bpm,
            every,
            pattern,
            choke_map,
            release_ms,
            channels=2 if stereo else 1,
            bed=self.bed,
            rng=(rng or random.SystemRandom()) if blind else None,
        )
        self.buffers: list[list[np.ndarray] | None] = [None] * len(self.paths)
        for note in self._accept(0, (0, 0), audio, report):
            print(note, file=self.out)
        self._pairs_said = 0
        self.watchers = [
            Watcher(p, self._work_for(i), self._done_for(i), t) for i, (p, t) in enumerate(zip(self.paths, texts))
        ]
        if len(report.files) < takes:
            self.watchers[0].finish(texts[0])  # the other takes render behind the first
        # The first patch plays as soon as it is ready; the others render behind it.
        for i, text in enumerate(texts[1:], 1):
            self.watchers[i].request(text)
        self.watch = watch

    def _text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise AuditionError(f"{path}: {e.strerror}") from None

    def _render(self, text: str, first: bool = False):
        """The set of takes, or with `first` only its first file when the set's gain cannot depend on the rest."""
        p = loads(text)
        files = 1 if first and p.output.mode == "gain" else self.takes
        audio, report = render_takes(p, files, stereo=self.stereo)
        return playable(audio, report), report

    def _work_for(self, slot: int):
        def work(text: str, generation: int):
            p = loads(text)
            if self.takes > 1 and p.output.mode == "gain" and generation > 0:
                early = self._render(text, first=True)
                if self.watchers[slot].newest(generation) and not early[1].over:
                    self._accept(slot, (generation, 0), *early)
            return self._render(text)

        return work

    def _say(self, slot: int, generation: int | None, message: str) -> None:
        if generation is None:
            tag = "" if len(self.paths) == 1 else f"{self.paths[slot].name}\n"
            print(tag + message, file=self.out)
            return
        tag = f"[{generation}]" if len(self.paths) == 1 else f"[{self.paths[slot].name} {generation}]"
        print(f"{tag} {message}", file=self.out)

    def _bed(self, path, bpm) -> Bed:
        rate, frames = load_audio(path)
        if rate != self.rate:
            raise AuditionError(f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples")
        if frames.shape[1] > 2:
            raise AuditionError(f"{path}: {frames.shape[1]} channels; a bed must be mono or stereo")
        bar = self.rate * 4 * 60 / bpm
        bars = round(frames.shape[0] / bar)
        if bars < 1 or abs(frames.shape[0] - bars * bar) > BAR_TOLERANCE_S * self.rate:
            raise AuditionError(
                f"{path}: {frames.shape[0] / bar:.3f} bars at {bpm:g} bpm; "
                "a bed must be within 1 ms of a whole number of bars"
            )
        print(
            f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, "
            f"{'stereo' if frames.shape[1] == 2 else 'mono'}",
            file=self.out,
        )
        return Bed(frames, self.rate, bpm, bars)

    def _sample(self, path) -> tuple[np.ndarray, float]:
        """The sample as it will play: in stereo with --stereo, else its mid channel; and its loudness so."""
        rate, frames = load_audio(path)
        if rate != self.rate:
            raise AuditionError(f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples")
        if self.stereo:
            played = frames if frames.shape[1] == 2 else np.repeat(frames[:, :1], 2, axis=1)
        else:
            played = frames.mean(axis=1)
        level = momentary_max(played, rate)
        if level == -math.inf:
            raise AuditionError(f"{path}: the sample is silent")
        return played, level

    def _matched(self, report) -> tuple[np.ndarray | None, list[str]]:
        """The sample matched to this render's loudness (its first file's), and what to say about it."""
        if self.sample is None:
            return None, []
        loud = report.files[0].loudness
        gain = loud - self.sample_lufs if loud > -math.inf else -math.inf
        matched = (self.sample * 10.0 ** (gain / 20.0)).astype(np.float32)
        peak = float(np.max(np.abs(matched)))
        clip = (
            f"; its peak is then {db(peak):+.1f} dBFS, which the device will clip, so lower output.db"
            if peak > 1.0
            else ""
        )
        return matched, [
            f"{'sample':<9}plays at {gain:+.1f} dB, matched to the render's loudness "
            f"(the largest 400 ms momentary loudness, BS.1770){clip}"
        ]

    def _mix_warning(self) -> str | None:
        """Whether the hits, with the bed under them and their chokes, would clip the device over the pattern."""
        if not self.bpm or not any(self.buffers):
            return None
        h = self.hits
        probe = Hits(
            self.rate,
            len(self.buffers),
            self.bpm,
            h.every,
            h.pattern,
            h.chokes,
            channels=h.channels,
            bed=self.bed,
        )
        probe.fade, probe.fade_length = h.fade, h.fade_length
        for i, b in enumerate(self.buffers):
            if b:
                probe.offer(i, 0, b)
        longest = max(len(x) for b in self.buffers if b for x in b)
        steps = len(h.pattern) if h.pattern else 1
        rounds = max(1, sum(len(b) for b in self.buffers if b))  # every take of every patch at least once
        span = probe._step_time(steps * rounds)
        if self.bed is not None:
            span = max(span, self.bed.start(1))
        peak = float(np.max(np.abs(probe.fill(span + longest))))
        if peak <= 1.0:
            return None
        under = "over the bed " if self.bed is not None else ""
        return f"warning: the hits {under}peak at {db(peak):+.1f} dBFS, which the device will clip; lower " + (
            "the bed or output.db" if self.bed is not None else "output.db"
        )

    def _accept(self, slot: int, generation: tuple, audio, report) -> list[str]:
        """Offer a render to its slot; returns what to say about the sample and the mix."""
        matched, notes = self._matched(report) if slot == 0 else (None, [])
        self.buffers[slot] = audio
        self.hits.offer(slot, generation, audio, matched)
        warning = self._mix_warning()
        return notes + ([warning] if warning else [])

    def poll(self) -> None:
        if self.watch:
            for w in self.watchers:
                w.poll()
        if self.blind:
            started = (self.hits.count + 1) // 2
            while self._pairs_said < started:
                self._pairs_said += 1
                print(f"pair {self._pairs_said}", file=self.out)

    def close(self) -> None:
        """Stop watching and rendering: nothing is offered or printed after this."""
        for w in self.watchers:
            w.stop()

    def reveal(self) -> str | None:
        """The blind order, once audition stops: R is the render, S the sample."""
        if not self.blind:
            return None
        pairs = " ".join(f"{i + 1}:{'RS' if first else 'SR'}" for i, first in enumerate(self.hits.order))
        return f"blind order (R = render, S = sample): {pairs or 'no pair played'}"

    def _done_for(self, slot: int):
        def done(generation, result, elapsed, newest) -> None:
            say = lambda msg: self._say(slot, generation, msg)  # noqa: E731
            if not newest:
                say("dropped: a newer edit is rendering")
            elif isinstance(result, PatchError):
                say(f"invalid patch, still playing the last valid one:\n{result}")
            elif isinstance(result, (RenderError, Exception)):
                say(f"render failed, still playing the last valid one: {result}")
            else:
                audio, report = result
                if report.over:
                    say(
                        f"over full scale by {db(report.peak):.2f} dB; lower output.db. "
                        "Still playing the last valid buffer."
                    )
                elif report.sample_rate != self.rate:
                    say(
                        f"sample rate {report.sample_rate} Hz, but audition plays at {self.rate} Hz; restart audition "
                        "to hear it. Still playing the last valid buffer."
                    )
                else:
                    notes = self._accept(slot, (generation, 1), audio, report)
                    what = f"{len(report.files)} files" if len(report.files) > 1 else "rendered"
                    say("\n".join([f"{what} in {1000 * elapsed:.0f} ms; plays from its next hit", str(report), *notes]))

        return done


def audition(
    paths,
    *,
    watch: bool = False,
    bpm: float | None = None,
    every: float = 1.0,
    pattern=None,
    chokes=(),
    release_ms: float = DEFAULT_RELEASE_MS,
    bed=None,
    against=None,
    blind: bool = False,
    takes: int = 1,
    stereo: bool = False,
    out=None,
    stream_factory=None,
    rng=None,
) -> int:
    """Play the patches until interrupted; with `watch`, re-render on every edit."""
    out = out or sys.stdout
    try:
        session = Session(
            paths,
            watch=watch,
            bpm=bpm,
            every=every,
            pattern=pattern,
            chokes=chokes,
            release_ms=release_ms,
            bed=bed,
            against=against,
            blind=blind,
            takes=takes,
            stereo=stereo,
            rng=rng,
            out=out,
        )
    except PatchError as e:
        print("invalid patch\n" + "\n".join(f"  {m}" for m in e.problems), file=sys.stderr)
        return 2
    except (AuditionError, RenderError) as e:
        print(e, file=sys.stderr)
        return 2
    except OverError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        stream = (stream_factory or _sounddevice_stream)(session.rate, session.hits.channels, session.hits.fill)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        with stream:
            while True:
                time.sleep(0.1)
                session.poll()
    except KeyboardInterrupt:
        return 0
    finally:
        session.close()
        revealed = session.reveal()
        if revealed:
            print(revealed, file=out)


def _sounddevice_stream(rate: int, channels: int, fill):
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise RuntimeError(f"audition needs the sounddevice package and PortAudio ({e})") from None
    try:
        sd.check_output_settings(samplerate=rate, channels=channels, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"the output device does not support {rate} Hz ({e}); audition never resamples") from None

    def callback(outdata, frames, _time, _status):
        outdata[:] = fill(frames).reshape(frames, channels)

    return sd.OutputStream(samplerate=rate, channels=channels, dtype="float32", callback=callback)
