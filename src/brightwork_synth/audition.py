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

import dataclasses
import importlib
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np

from . import wav
from .curve import Curve, amplitude_curve
from .loudness import momentary_max
from .patch import DEFAULT_RELEASE_MS, PatchError, choke_knots, loads
from .render import FORMATS, RenderError, Report, db, quantize, render_takes

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from contextlib import AbstractContextManager

    from numpy.typing import ArrayLike, NDArray

type Floats = NDArray[np.float64]
type Audio = NDArray[np.float32]  # a buffer, mono or (frames, 2)
type Generation = tuple[int, int]  # an edit's generation, and the stage of its render: 0 for an early first file
type Rendered = tuple[list[Audio], Report]
type Fill = Callable[[int], Audio]

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars
STEREO = 2  # channels
EVEN_ODDS = 0.5  # a fair coin's: which of a blind pair plays first
POLL_S = 0.1  # how often the session looks at its files and its pairs
FAILURES = (ValueError, RuntimeError, OSError, ArithmeticError, MemoryError)  # what a render can fail with


class Offer(NamedTuple):
    """What waits for a slot's next hit: its generation, its take buffers, and the sample matched to them."""

    generation: Generation
    buffers: list[Audio]
    sample: Audio | None


class Coin(Protocol):
    """What decides each blind pair's order: anything with random() in [0, 1), such as a seeded generator."""

    def random(self) -> float:
        """Draw a number in [0, 1)."""
        ...


class Writer(Protocol):
    """Where audition says what it does: standard output, or a test's collector."""

    def write(self, s: str, /) -> object:
        """Write text."""
        ...


class Done(Protocol):
    """What a watcher calls with each finished render: its generation, result and time, and whether it is newest."""

    def __call__(self, generation: int, result: object, elapsed: float, /, *, newest: bool) -> None:
        """Take a finished render."""
        ...


class AuditionError(Exception):
    """An audition that cannot start, with what to change."""


def load_audio(path: str | os.PathLike[str]) -> tuple[int, Floats]:
    """Read a float or PCM WAV as float64 frames × channels, never resampled."""
    try:
        w = wav.read(path)
    except (OSError, ValueError) as e:
        msg = f"{path}: {e}"
        raise AuditionError(msg) from None
    x = w.as_float()
    return w.rate, x if x.ndim == STEREO else x[:, None]


def playable(audio: list[Audio], report: Report) -> list[Audio]:
    """Return what `render` writes, as float32: PCM formats add only their quantization, each channel its take's."""
    bits = FORMATS[report.format]
    if bits is None:
        return audio
    scale = float(2 ** (bits - 1))
    out = []
    for a, f in zip(audio, report.files, strict=True):
        takes = [report.takes[i].take for i in f.takes]
        if a.ndim == 1:
            q = quantize(a, bits, report.seed, takes[0]) / scale
        else:
            q = np.stack([quantize(a[:, c], bits, report.seed, t) / scale for c, t in enumerate(takes)], axis=1)
        out.append(q.astype(np.float32))
    return out


def fade(release_ms: float, rate: int) -> tuple[Curve, int]:
    """Return the gain of a choked hit from the moment it is choked, `choke`'s damper from 0 ms, and its length."""
    knots = choke_knots(0.0, release_ms)
    curve = amplitude_curve([ms / 1000 for ms, _ in knots], [g for _, g in knots])
    return curve, math.ceil(knots[-1][0] / 1000 * rate)


class Bed:
    """A loop under the hits, restarted every `bars` bars on the tempo grid.

    Loop k starts at round(k · bars · 4 · 60 / bpm · rate), the same rounding as
    the hits, and plays as much of the bed as fits before the next start.
    """

    def __init__(self, frames: ArrayLike, rate: int, bpm: float, bars: int) -> None:
        """Keep the bed's frames, as float32, and its length in bars at a tempo."""
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.channels: int = self.frames.shape[1]
        self.rate, self.bpm, self.bars = rate, bpm, bars

    def start(self, k: int) -> int:
        """Return where loop k starts, in samples from the first bar."""
        return round(k * self.bars * 4 * self.rate * 60 / self.bpm)

    def fill(self, clock: int, frames: int) -> Audio:
        """Return the bed over [clock, clock + frames), with the clock counted from the first bar."""
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

    __slots__ = ("buffer", "choked", "slot", "start")

    def __init__(self, start: int, buffer: Audio, slot: int) -> None:
        """Start a hit of a slot's buffer at a sample."""
        self.start, self.buffer, self.slot = start, buffer, slot
        self.choked: int | None = None

    @property
    def end(self) -> int:
        """Where its buffer ends."""
        return self.start + len(self.buffer)


@dataclass(frozen=True)
class Tempo:
    """The grid hits land on: a tempo, a note value in beats of a 4/4 bar, and the pattern of slots, if any."""

    bpm: float
    every: float = 1.0
    pattern: tuple[int, ...] | None = None  # the slot of each step, from 1, with 0 for a rest

    def time(self, n: int, rate: int) -> int:
        """Return where step n starts, in samples from the first bar."""
        return round(n * self.every * rate * 60 / self.bpm)


@dataclass(frozen=True)
class Setup:
    """How hits are played: the slots, the grid, the chokes and their release, the channels, the bed and the coin.

    `chokes` maps a slot to the slots that choke it, from 0; `rng` makes each
    pair of a comparison with a sample play in random order.
    """

    slots: int = 1
    bpm: float | None = None
    every: float = 1.0
    pattern: Sequence[int] | None = None
    chokes: Mapping[int, Collection[int]] = field(default_factory=dict)
    release_ms: float = DEFAULT_RELEASE_MS
    channels: int = 1
    bed: Bed | None = None
    rng: Coin | None = None


class Hits:
    """Mixes hits into output blocks. `offer` may be called from any thread.

    Slot i holds patch i's take buffers, mono or (frames, 2). With a tempo,
    step n starts at round(n · every · 60 / bpm · rate), every in beats of a
    4/4 bar; the pattern names the slot of each step, from 1, with 0 for a
    rest, and without one the slots that have a buffer take turns. With a
    sample (slot 0 only), hits alternate between the render and the sample in
    pairs, and `order` records each pair's order (True when the render played
    first).
    """

    def __init__(self, rate: int, setup: Setup | None = None) -> None:
        """Prepare to play at a rate, with nothing to play until a slot is offered its buffers."""
        s = setup or Setup()
        self.rate, self.bed, self.rng = rate, s.bed, s.rng
        pattern = tuple(s.pattern) if s.pattern else None
        self.tempo = None if s.bpm is None else Tempo(s.bpm, s.every, pattern)
        self.chokes = {a: frozenset(bs) for a, bs in s.chokes.items()}
        self.fade, self.fade_length = fade(s.release_ms, rate)
        self.channels = max(s.channels, s.bed.channels if s.bed is not None else 1)
        self._offers: list[Offer | None] = [None] * s.slots
        self._lock = threading.Lock()
        self.generations: list[Generation] = [(-1, 0)] * s.slots
        self.buffers: list[list[Audio]] = [[] for _ in range(s.slots)]  # empty until rendered
        self.sample: Audio | None = None
        self.played = [0] * s.slots  # hits per slot, which pick its take
        self.last_slot = -1
        self.voices: list[Voice] = []
        self.clock = 0  # index of the next output sample, counted from the first bar
        self.started = False
        self.step = 0
        self.next_hit = 0
        self.count = 0  # hits started
        self.order: list[bool] = []
        self.log: list[tuple[int, int]] = []  # (slot, take) of each hit, the sample being slot −1

    @property
    def generation(self) -> int:
        """The generation of slot 0's buffers."""
        return self.generations[0][0]

    def offer(self, slot: int, generation: int | Generation, buffers: list[Audio], sample: Audio | None = None) -> None:
        """Queue a slot's take buffers, and the sample matched to them, for that slot's next hit.

        `generation` is an int or a (generation, stage) pair; an older one never replaces a newer one.
        """
        key = generation if isinstance(generation, tuple) else (generation, 0)
        with self._lock:
            current = self._offers[slot]
            if current is None or key > current.generation:
                self._offers[slot] = Offer(key, list(buffers), sample)

    def offered(self, slot: int) -> Offer | None:
        """Return what waits for a slot's next hit, if anything."""
        with self._lock:
            return self._offers[slot]

    def _take_offers(self) -> None:
        """Take every offer newer than what its slot plays."""
        with self._lock:
            for i, offered in enumerate(self._offers):
                if offered is not None and offered.generation > self.generations[i]:
                    self.generations[i], self.buffers[i] = offered.generation, offered.buffers
                    if i == 0:
                        self.sample = offered.sample

    def _slot_for_step(self, n: int) -> int | None:
        """Return the slot that plays step n, or None for a rest or a patch not rendered yet."""
        if self.tempo is not None and self.tempo.pattern is not None:
            slot = self.tempo.pattern[n % len(self.tempo.pattern)] - 1
            return slot if slot >= 0 and self.buffers[slot] else None
        ready = [i for i, b in enumerate(self.buffers) if b]
        if not ready:
            return None
        slot = next((i for i in ready if i > self.last_slot), ready[0])
        self.last_slot = slot
        return slot

    def _buffer(self, slot: int) -> tuple[int, Audio]:
        """Return the slot that sounds, −1 for the sample, and its buffer: the slot's next take, or the sample."""
        if slot == 0 and self.sample is not None:
            if self.count % 2 == 0:
                self.order.append(True if self.rng is None else self.rng.random() < EVEN_ODDS)
            if (self.count % 2 == 0) != self.order[-1]:
                self.log.append((-1, 0))
                return -1, self.sample
        takes = self.buffers[slot]
        take = self.played[slot] % len(takes)
        self.played[slot] += 1
        self.log.append((slot, take))
        return slot, takes[take]

    def _start_hit(self, at: int, slot: int) -> Audio:
        """Start a hit of a slot at a sample, choking what that slot chokes, and return its buffer."""
        sounding, buf = self._buffer(slot)
        if sounding >= 0:
            for v in self.voices:
                if v.choked is None and v.start < at and sounding in self.chokes.get(v.slot, ()):
                    v.choked = at
        self.voices.append(Voice(at, buf, sounding))
        self.count += 1
        return buf

    def _schedule(self, end: int) -> None:
        """Start every hit due before sample `end`."""
        while self.next_hit < end:
            self._take_offers()
            slot = self._slot_for_step(self.step)
            buf = self._start_hit(self.next_hit, slot) if slot is not None else None
            self.step += 1
            if self.tempo is not None:
                self.next_hit = self.tempo.time(self.step, self.rate)
            else:
                self.next_hit += (len(buf) if buf is not None else 0) + round(GAP_S * self.rate)

    def _mix(self, out: Audio, end: int) -> None:
        """Add the sounding hits over [clock, end) into out, faded where choked, and keep those that sound on."""
        playing = []
        for v in self.voices:
            stop = v.end if v.choked is None else min(v.end, v.choked + self.fade_length)
            a, b = max(self.clock, v.start), min(end, stop)
            if a < b:
                part = v.buffer[a - v.start : b - v.start]
                if v.choked is not None:
                    gain = self.fade.on((np.arange(a, b) - v.choked) / self.rate).astype(np.float32)
                    part = part * (gain if part.ndim == 1 else gain[:, None])
                out[a - self.clock : b - self.clock] += part if part.ndim == STEREO else part[:, None]
            if stop > end:
                playing.append(v)
        self.voices = playing

    def fill(self, frames: int) -> Audio:
        """Return the next `frames` samples of the output, silence until the first buffers arrive."""
        if not self.started:
            self._take_offers()
            if not any(self.buffers):
                return np.zeros(frames if self.channels == 1 else (frames, self.channels), dtype=np.float32)
            self.started = True
        out = np.zeros((frames, self.channels), dtype=np.float32)
        end = self.clock + frames
        self._schedule(end)
        self._mix(out, end)
        if self.bed is not None:
            out += self.bed.fill(self.clock, frames)  # a mono bed in a stereo stream plays in the centre
        self.clock = end
        return out[:, 0] if self.channels == 1 else out


class Watcher:
    """Renders the newest version of a file in the background."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        work: Callable[[str, int], object],
        done: Done,
        text: str | None = None,
    ) -> None:
        """Start the watcher's thread; `text` is the version already rendered, if any."""
        self.path, self.work, self.done = Path(path), work, done
        self.generation = 0
        self._stamp: tuple[int, int] | None = None
        self._text = text
        self._pending: tuple[int, str] | None = None
        self._stopped = False
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def poll(self) -> None:
        """Start a new generation if the file's content changed."""
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._stamp:
                return
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        self._stamp = stamp
        if text == self._text:
            return
        self._text = text
        self.request(text)

    def request(self, text: str) -> None:
        """Render a new generation of the text, superseding any request not yet started."""
        with self._cv:
            if self._stopped:
                return
            self.generation += 1
            self._pending = (self.generation, text)
            self._cv.notify()

    def finish(self, text: str) -> None:
        """Render the current generation again in the background, unless an edit supersedes it."""
        with self._cv:
            if self._stopped:
                return
            self._pending = (self.generation, text)
            self._cv.notify()

    def pending(self) -> bool:
        """Tell whether a request waits to be rendered."""
        with self._cv:
            return self._pending is not None

    def newest(self, generation: int) -> bool:
        """Tell whether a generation is still the newest, and the watcher still running."""
        with self._cv:
            return generation == self.generation and not self._stopped

    def stop(self) -> None:
        """Start nothing more; a render already running finishes, and its result is dropped."""
        with self._cv:
            self._stopped = True
            self._pending = None
            self._cv.notify()

    def _run(self) -> None:
        """Render each pending request, and report it, until stopped."""
        while True:
            with self._cv:
                while self._pending is None and not self._stopped:
                    self._cv.wait()
                if self._pending is None:  # stopped
                    return
                generation, text = self._pending
                self._pending = None
            start = time.perf_counter()
            try:
                result = self.work(text, generation)
            except FAILURES as e:  # reported by `done`; the last valid buffer keeps playing
                result = e
            elapsed = time.perf_counter() - start
            with self._cv:
                if self._stopped:
                    return
            self.done(generation, result, elapsed, newest=self.newest(generation))


class OverError(Exception):
    """A first render over full scale, which audition cannot play."""

    def __init__(self, path: Path, report: Report) -> None:
        """Say how far over the render is, and what to lower."""
        super().__init__(f"{path}: over full scale by {db(report.peak):.2f} dB: lower output.db to audition it")


def _chokes(rules: Sequence[tuple[int, tuple[int, ...]]], slots: int) -> dict[int, set[int]]:
    """Return --choke A:B,… rules, numbered from 1, as slot → the slots that choke it."""
    out: dict[int, set[int]] = {}
    for target, by in rules:
        rule = f"--choke {target}:{','.join(map(str, by))}"
        for n in (target, *by):
            if not 1 <= n <= slots:
                msg = f"{rule}: patch {n} is not among the {slots}"
                raise AuditionError(msg)
        if target in by:
            msg = f"{rule}: a patch cannot choke itself"
            raise AuditionError(msg)
        out.setdefault(target - 1, set()).update(b - 1 for b in by)
    return out


@dataclass(frozen=True)
class Settings:
    """How to audition: watching, the grid, the chokes, a bed, a sample to compare with, and the takes."""

    watch: bool = False
    bpm: float | None = None
    every: float = 1.0
    pattern: tuple[int, ...] | None = None
    chokes: tuple[tuple[int, tuple[int, ...]], ...] = ()
    release_ms: float = DEFAULT_RELEASE_MS
    bed: str | os.PathLike[str] | None = None
    against: str | os.PathLike[str] | None = None
    blind: bool = False
    takes: int = 1
    stereo: bool = False


DEFAULTS = Settings()


def _check(settings: Settings, patches: int) -> None:
    """Refuse settings that cannot go together."""
    s = settings
    if s.against is not None and patches > 1:
        msg = "--against compares one patch with a sample; give one patch"
        raise AuditionError(msg)
    if s.blind and s.against is None:
        msg = "--blind needs --against: a blind comparison needs a sample"
        raise AuditionError(msg)
    if s.bed is not None and s.bpm is None:
        msg = "--with needs --bpm, so the hits land on the bed's beats"
        raise AuditionError(msg)
    if s.pattern is None:
        return
    if s.bpm is None:
        msg = "--pattern needs --bpm: a pattern's steps are on the tempo grid"
        raise AuditionError(msg)
    if not any(s.pattern):
        msg = "--pattern needs at least one hit"
        raise AuditionError(msg)
    if max(s.pattern) > patches:
        msg = f"--pattern names patch {max(s.pattern)}, but there are {patches}"
        raise AuditionError(msg)


class Session:
    """One audition: the hits, the renders behind them, and the watchers. `poll` is called about every 0.1 s."""

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        settings: Settings = DEFAULTS,
        *,
        rng: Coin | None = None,
        out: Writer | None = None,
    ) -> None:
        """Render the first patch, open the bed and the sample, and start watching every patch."""
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, (str, os.PathLike)) else paths)]
        s = self.settings = settings
        self.out = out or sys.stdout
        _check(s, len(self.paths))
        choke_map = _chokes(s.chokes, len(self.paths))
        texts = [self._text(p) for p in self.paths]
        audio, report = self._render(texts[0], first=True)
        self._say(0, None, str(report))
        if report.over:
            raise OverError(self.paths[0], report)
        self.rate = report.sample_rate
        self.bed = None if s.bed is None or s.bpm is None else self._bed(s.bed, s.bpm)
        self.sample: Audio | None = None
        self.sample_lufs = -math.inf
        if s.against is not None:
            self.sample, self.sample_lufs = self._sample(s.against)
        setup = Setup(
            len(self.paths),
            s.bpm,
            s.every,
            s.pattern,
            choke_map,
            s.release_ms,
            channels=STEREO if s.stereo else 1,
            bed=self.bed,
            rng=(rng or random.SystemRandom()) if s.blind else None,
        )
        self.hits = Hits(self.rate, setup)
        self.buffers: list[list[Audio]] = [[] for _ in self.paths]  # empty until rendered
        for note in self._accept(0, (0, 0), audio, report):
            self.out.write(f"{note}\n")
        self._pairs_said = 0
        self.watchers = [
            Watcher(p, self._work_for(i), self._done_for(i), t)
            for i, (p, t) in enumerate(zip(self.paths, texts, strict=True))
        ]
        if len(report.files) < s.takes:
            self.watchers[0].finish(texts[0])  # the other takes render behind the first
        # The first patch plays as soon as it is ready; the others render behind it.
        for i, text in enumerate(texts[1:], 1):
            self.watchers[i].request(text)

    def _text(self, path: Path) -> str:
        """Read a patch file's text."""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            msg = f"{path}: {e.strerror}"
            raise AuditionError(msg) from None

    def _render(self, text: str, *, first: bool = False) -> Rendered:
        """Render the takes; with `first`, the first file alone when the set's gain cannot depend on the rest."""
        p = loads(text)
        files = 1 if first and p.output.mode == "gain" else self.settings.takes
        audio, report = render_takes(p, files, stereo=self.settings.stereo)
        return playable(audio, report), report

    def _work_for(self, slot: int) -> Callable[[str, int], object]:
        """Make the work of a slot's watcher: its first file early when it can play alone, then every take."""

        def work(text: str, generation: int) -> object:
            p = loads(text)
            if self.settings.takes > 1 and p.output.mode == "gain" and generation > 0:
                early = self._render(text, first=True)
                if self.watchers[slot].newest(generation) and not early[1].over:
                    self._accept(slot, (generation, 0), *early)
            return self._render(text)

        return work

    def _say(self, slot: int, generation: int | None, message: str) -> None:
        """Say something about a slot's render, tagged with its patch and generation when that tells them apart."""
        if generation is None:
            tag = "" if len(self.paths) == 1 else f"{self.paths[slot].name}\n"
            self.out.write(f"{tag}{message}\n")
            return
        tag = f"[{generation}]" if len(self.paths) == 1 else f"[{self.paths[slot].name} {generation}]"
        self.out.write(f"{tag} {message}\n")

    def _bed(self, path: str | os.PathLike[str], bpm: float) -> Bed:
        """Open the bed: at the patch's rate, mono or stereo, and within 1 ms of a whole number of bars."""
        rate, frames = load_audio(path)
        if rate != self.rate:
            msg = f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples"
            raise AuditionError(msg)
        if frames.shape[1] > STEREO:
            msg = f"{path}: {frames.shape[1]} channels; a bed must be mono or stereo"
            raise AuditionError(msg)
        bar = self.rate * 4 * 60 / bpm
        bars = round(frames.shape[0] / bar)
        if bars < 1 or abs(frames.shape[0] - bars * bar) > BAR_TOLERANCE_S * self.rate:
            msg = (
                f"{path}: {frames.shape[0] / bar:.3f} bars at {bpm:g} bpm; "
                "a bed must be within 1 ms of a whole number of bars"
            )
            raise AuditionError(msg)
        channels = "stereo" if frames.shape[1] == STEREO else "mono"
        self.out.write(f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, {channels}\n")
        return Bed(frames, self.rate, bpm, bars)

    def _sample(self, path: str | os.PathLike[str]) -> tuple[Audio, float]:
        """Open the sample as it will play, in stereo with --stereo or else its mid channel, and its loudness so."""
        rate, frames = load_audio(path)
        if rate != self.rate:
            msg = f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples"
            raise AuditionError(msg)
        if self.settings.stereo:
            played = frames if frames.shape[1] == STEREO else np.repeat(frames[:, :1], STEREO, axis=1)
        else:
            played = frames.mean(axis=1)
        level = momentary_max(played, rate)
        if level == -math.inf:
            msg = f"{path}: the sample is silent"
            raise AuditionError(msg)
        return played.astype(np.float32), level

    def _matched(self, report: Report) -> tuple[Audio | None, list[str]]:
        """Return the sample matched to this render's loudness (its first file's), and what to say about it."""
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
        note = (
            f"{'sample':<9}plays at {gain:+.1f} dB, matched to the render's loudness "
            f"(the largest 400 ms momentary loudness, BS.1770){clip}"
        )
        return matched, [note]

    def _mix_warning(self) -> str | None:
        """Warn if the hits, with the bed under them and their chokes, would clip the device over the pattern."""
        h = self.hits
        if h.tempo is None or not any(self.buffers):
            return None
        setup = Setup(len(self.buffers), h.tempo.bpm, h.tempo.every, h.tempo.pattern, h.chokes, channels=h.channels)
        probe = Hits(self.rate, dataclasses.replace(setup, bed=self.bed))
        probe.fade, probe.fade_length = h.fade, h.fade_length
        for i, b in enumerate(self.buffers):
            if b:
                probe.offer(i, 0, b)
        longest = max(len(x) for b in self.buffers for x in b)
        steps = len(h.tempo.pattern) if h.tempo.pattern else 1
        rounds = max(1, sum(len(b) for b in self.buffers))  # every take of every patch at least once
        span = h.tempo.time(steps * rounds, self.rate)
        if self.bed is not None:
            span = max(span, self.bed.start(1))
        peak = float(np.max(np.abs(probe.fill(span + longest))))
        if peak <= 1.0:
            return None
        under = "over the bed " if self.bed is not None else ""
        lower = "the bed or output.db" if self.bed is not None else "output.db"
        return f"warning: the hits {under}peak at {db(peak):+.1f} dBFS, which the device will clip; lower {lower}"

    def _accept(self, slot: int, generation: Generation, audio: list[Audio], report: Report) -> list[str]:
        """Offer a render to its slot, and return what to say about the sample and the mix."""
        matched, notes = self._matched(report) if slot == 0 else (None, [])
        self.buffers[slot] = audio
        self.hits.offer(slot, generation, audio, matched)
        warning = self._mix_warning()
        return notes + ([warning] if warning else [])

    def poll(self) -> None:
        """Look for edits, when watching, and announce each blind pair as it starts."""
        if self.settings.watch:
            for w in self.watchers:
                w.poll()
        if self.settings.blind:
            started = (self.hits.count + 1) // 2
            while self._pairs_said < started:
                self._pairs_said += 1
                self.out.write(f"pair {self._pairs_said}\n")

    def close(self) -> None:
        """Stop watching and rendering: nothing is offered or printed after this."""
        for w in self.watchers:
            w.stop()

    def reveal(self) -> str | None:
        """Return the blind order, once audition stops: R is the render, S the sample."""
        if not self.settings.blind:
            return None
        pairs = " ".join(f"{i + 1}:{'RS' if first else 'SR'}" for i, first in enumerate(self.hits.order))
        return f"blind order (R = render, S = sample): {pairs or 'no pair played'}"

    def _done_for(self, slot: int) -> Done:
        """Make what a slot's watcher does with a finished render: offer it, or say why not."""

        def done(generation: int, result: object, elapsed: float, *, newest: bool) -> None:
            def say(message: str) -> None:
                self._say(slot, generation, message)

            if not newest:
                say("dropped: a newer edit is rendering")
            elif isinstance(result, PatchError):
                say(f"invalid patch, still playing the last valid one:\n{result}")
            elif isinstance(result, Exception):
                say(f"render failed, still playing the last valid one: {result}")
            else:
                audio, report = cast("Rendered", result)
                if report.over:
                    say(
                        f"over full scale by {db(report.peak):.2f} dB; lower output.db. "
                        "Still playing the last valid buffer.",
                    )
                elif report.sample_rate != self.rate:
                    say(
                        f"sample rate {report.sample_rate} Hz, but audition plays at {self.rate} Hz; restart audition "
                        "to hear it. Still playing the last valid buffer.",
                    )
                else:
                    notes = self._accept(slot, (generation, 1), audio, report)
                    what = f"{len(report.files)} files" if len(report.files) > 1 else "rendered"
                    say("\n".join([f"{what} in {1000 * elapsed:.0f} ms; plays from its next hit", str(report), *notes]))

        return done


def audition(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    settings: Settings = DEFAULTS,
    *,
    out: Writer | None = None,
    stream_factory: Callable[[int, int, Fill], AbstractContextManager[object]] | None = None,
    rng: Coin | None = None,
) -> int:
    """Play the patches until interrupted; with `settings.watch`, re-render on every edit."""
    report = out or sys.stdout
    try:
        session = Session(paths, settings, rng=rng, out=report)
    except PatchError as e:
        sys.stderr.write("invalid patch\n" + "".join(f"  {m}\n" for m in e.problems))
        return 2
    except (AuditionError, RenderError) as e:
        sys.stderr.write(f"{e}\n")
        return 2
    except OverError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    try:
        stream = (stream_factory or _sounddevice_stream)(session.rate, session.hits.channels, session.hits.fill)
    except RuntimeError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    try:
        with stream:
            while True:
                time.sleep(POLL_S)
                session.poll()
    except KeyboardInterrupt:
        return 0
    finally:
        session.close()
        revealed = session.reveal()
        if revealed:
            report.write(f"{revealed}\n")


def _sounddevice_stream(rate: int, channels: int, fill: Fill) -> AbstractContextManager[object]:
    """Open an output stream at exactly `rate` that plays what `fill` mixes, or say why the device cannot."""
    try:
        # Imported here: importing sounddevice starts PortAudio, which only audition needs.
        sd = importlib.import_module("sounddevice")
    except (ImportError, OSError) as e:
        msg = f"audition needs the sounddevice package and PortAudio ({e})"
        raise RuntimeError(msg) from None
    try:
        sd.check_output_settings(samplerate=rate, channels=channels, dtype="float32")
    except (ValueError, sd.PortAudioError) as e:
        msg = f"the output device does not support {rate} Hz ({e}); audition never resamples"
        raise RuntimeError(msg) from None

    def callback(outdata: Audio, frames: int, _time: object, _status: object) -> None:
        outdata[:] = fill(frames).reshape(frames, channels)

    stream: AbstractContextManager[object] = sd.OutputStream(
        samplerate=rate,
        channels=channels,
        dtype="float32",
        callback=callback,
    )
    return stream
