# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate the figures, the reference sounds and the reference gallery in docs/, with brightwork itself.

    uv run python docs/build.py

Every figure is drawn from the code in this repository, as SVG that follows the reader's light or dark
theme. Every sound is a reference rendered and exported as `brightwork render` would, and
docs/references.md is written from the shipped references, so the gallery always shows them as they are.
"""

from __future__ import annotations

import dataclasses
import html
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from brightwork_synth import cli, pack, render, voice
from brightwork_synth.patch import Patch, loads, softer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]

DOCS = Path(__file__).resolve().parent
FIGURES = DOCS / "figures"
AUDIO = DOCS / "audio"
RATE = 48000
INTERNAL = render.rates(RATE)[0]  # the 48 kHz family's internal rate
WIDTH = 760
POINTS = 600  # the most points a curve is drawn with
FLOOR_DB = -80.0  # the lowest level a figure shows
SMOOTH_OCTAVE = 1 / 6  # spectra are smoothed over this much of an octave
LOW_HZ, HIGH_HZ = 100.0, 20000.0
MAX_TICKS = 6  # time ticks on an axis, at most
SECONDS_FROM_MS = 2000.0  # axes this long are marked in seconds
MS_PER_S = 1000.0

DESCRIPTIONS = {
    "closed": "A closed hi-hat: plates of modes 14 Hz apart over 6–16 kHz, and a stick tick.",
    "open": "The open hat: the same plates ringing for 1.3 s, slapping a little.",
    "pedal": "The foot's chick: plates closing, with a darker chick below them.",
    "crash": "A crash: a bright wash that blooms after the strike and darkens as it decays.",
    "ride": "A ride: a defined ping over a sustained wash.",
    "bell": "A ride's bell: a few strong partials, long and clearly pitched.",
    "analog": "The drum machine's closed hat: six square oscillators through a band.",
    "shaker": "One stroke of a shaker: fine grains rattling against the shell, larger beads clicking over them.",
    "tambourine": "A headless tambourine: jingles ringing from the strike, then shivering as they rattle down.",
    "offbeat": "Trance's off-beat open hat: brighter and shorter than the open hat, slapping lightly.",
    "tight": "Tech house's closed hat: shorter and a little darker than the closed hat.",
    "dead": "A damped hat, as with a towel over the plates: dark, its highs gone at once.",
    "wash": "A ride washed with a soft mallet: its highs swell after the strike and ring for seconds.",
    "house-open": "House's off-beat open hat: darker and longer than trance's, sizzling as its plates slap.",
    "house-ride": "House's ride: a bright ping over a short wash.",
    "house-tamb": "A crisp tambourine stroke for the off-beat: jingles ringing, then a short shiver.",
    "shaker-16": "A short shaker stroke for sixteenths.",
    "top": "Tech house's top: a closed hat whose plates rattle.",
    "trance-crash": "The crash on the drop: long and bright, its highs blooming after the strike.",
    "trance-ride": "A driving ride: bright wash, bright ping.",
    "analog-open": "The drum machine's analog squares, open.",
    "analog-cymbal": "The analog squares as their cymbal, through a wider band.",
    "machine-closed": "A closed hat stored as a drum machine stored it: 6 bits at 32 kHz, its decay the damper.",
    "machine-open": "The same machine's open hat: 6 bits at 32 kHz, a fading tail turning grainy.",
    "half-open": "A half-open hat: its plates slapping as they touch.",
    "sizzle": "A ride with rivets buzzing against the plate through its wash.",
    "china": "A china: trashy and quick, its plates slapping.",
    "splash": "A splash: small and bright.",
}

STYLE = """<style>
.bg{fill:#ffffff}.fg{fill:#1f2328}.dim{fill:#59636e}.axis{stroke:#59636e;stroke-width:1}
.grid{stroke:#d1d9e0;stroke-width:1}.box{fill:#f6f8fa;stroke:#59636e;stroke-width:1}
.c0{stroke:#0969da;fill:none;stroke-width:1.6}.c1{stroke:#bf3989;fill:none;stroke-width:1.6}
.c2{stroke:#1a7f37;fill:none;stroke-width:1.6}.c3{stroke:#9a6700;fill:none;stroke-width:1.6}
.f0{fill:#0969da}.f1{fill:#bf3989}.f2{fill:#1a7f37}.f3{fill:#9a6700}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:12px}
.title{font-size:15px;font-weight:600}.small{font-size:11px}
@media (prefers-color-scheme:dark){.bg{fill:#0d1117}.fg{fill:#f0f6fc}.dim{fill:#9198a1}
.axis{stroke:#9198a1}.grid{stroke:#3d444d}.box{fill:#151b23;stroke:#9198a1}
.c0{stroke:#4493f8}.c1{stroke:#db61a2}.c2{stroke:#3fb950}.c3{stroke:#d29922}
.f0{fill:#4493f8}.f1{fill:#db61a2}.f2{fill:#3fb950}.f3{fill:#d29922}}
</style>"""


def svg(height: float, title: str, subtitle: str, body: str, description: str) -> str:
    """Wrap a figure's body: its background, title and subtitle, styled for light and dark themes."""
    head = (
        f'<text class="fg title" x="20" y="28">{html.escape(title)}</text>'
        f'<text class="dim" x="20" y="47">{html.escape(subtitle)}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" '
        f'height="{height:g}" role="img" aria-label="{html.escape(description)}">'
        f"<title>{html.escape(description)}</title>{STYLE}"
        f'<rect class="bg" width="{WIDTH}" height="{height:g}" rx="10"/>{head}{body}</svg>\n'
    )


def text(x: float, y: float, words: str, cls: str = "dim", anchor: str = "start") -> str:
    """Place a label."""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{html.escape(words)}</text>'


@dataclass(frozen=True)
class Panel:
    """A plot area: its place on the figure, its ranges, and whether its x axis is logarithmic."""

    x: float
    y: float
    w: float
    h: float
    xs: tuple[float, float]
    ys: tuple[float, float]
    xlog: bool = False

    def px(self, v: Floats) -> Floats:
        """Map x values to the figure."""
        lo, hi = self.xs
        u = (np.log(v / lo) / math.log(hi / lo)) if self.xlog else (v - lo) / (hi - lo)
        out: Floats = self.x + self.w * u
        return out

    def py(self, v: Floats) -> Floats:
        """Map y values to the figure, clipped to the panel."""
        lo, hi = self.ys
        out: Floats = self.y + self.h * (1 - (np.clip(v, lo, hi) - lo) / (hi - lo))
        return out

    def curve(self, x: Floats, y: Floats, cls: str) -> str:
        """Draw a curve through (x, y), at most POINTS points, keeping each stretch's extremes."""
        if x.size > POINTS:
            edges = np.linspace(0, x.size, POINTS // 2 + 1).astype(int)
            keep = []
            for a, b in itertools.pairwise(edges):
                if b > a:
                    seg = y[a:b]
                    keep += sorted({a + int(np.argmin(seg)), a + int(np.argmax(seg))})
            x, y = x[keep], y[keep]
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(self.px(x), self.py(y), strict=True))
        return f'<polyline class="{cls}" points="{pts}"/>'

    def runs(self, x: Floats, y: Floats, keep: NDArray[np.bool_], cls: str) -> str:
        """Draw the curve through (x, y) where `keep` holds, one stretch at a time, never across a gap."""
        edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(int), [0]))))
        return "".join(
            self.curve(x[a:b], y[a:b], cls) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a > 1
        )

    def frame(self, xticks: Sequence[tuple[float, str]], yticks: Sequence[tuple[float, str]], xlabel: str) -> str:
        """Draw the panel's grid, axes and tick labels."""
        out = []
        for v, label in xticks:
            x = float(self.px(np.array([v]))[0])
            out.append(f'<line class="grid" x1="{x:.1f}" y1="{self.y}" x2="{x:.1f}" y2="{self.y + self.h}"/>')
            out.append(text(x, self.y + self.h + 15, label, anchor="middle"))
        for v, label in yticks:
            y = float(self.py(np.array([v]))[0])
            out.append(f'<line class="grid" x1="{self.x}" y1="{y:.1f}" x2="{self.x + self.w}" y2="{y:.1f}"/>')
            out.append(text(self.x - 6, y + 4, label, anchor="end"))
        out.append(
            f'<rect class="axis" fill="none" x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}"/>',
        )
        out.append(text(self.x + self.w / 2, self.y + self.h + 32, xlabel, anchor="middle"))
        return "".join(out)


def signed(x: float) -> str:
    """Write a level to 0.1 dB with a true minus sign."""
    return f"{x:.1f}".replace("-", "−")


def db(x: Floats) -> Floats:
    """Return x in dB, with silence at the figures' floor."""
    out: Floats = 20 * np.log10(np.maximum(np.abs(x), 10 ** (FLOOR_DB / 20)))
    return out


def envelope_db(a: Floats, rate: int, ms: float = 1.0) -> Floats:
    """Return the RMS envelope over windows of `ms`, in dB."""
    n = max(1, round(ms * rate / 1000))
    return db(np.sqrt(np.convolve(a * a, np.ones(n) / n, "same")))


def spectrum_db(a: Floats, rate: int) -> tuple[Floats, Floats]:
    """Return the power spectrum, smoothed over a sixth of an octave and peaking at 0 dB, from 100 Hz to 20 kHz."""
    power = np.abs(np.fft.rfft(a)) ** 2
    f = np.fft.rfftfreq(a.size, 1 / rate)
    centres = np.geomspace(LOW_HZ, HIGH_HZ, 240)
    half = 2 ** (SMOOTH_OCTAVE / 2)
    cum = np.concatenate(([0.0], np.cumsum(power)))
    lo, hi = np.searchsorted(f, centres / half), np.searchsorted(f, centres * half)
    mean = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
    level = 10 * np.log10(np.maximum(mean / np.max(mean), 10 ** (FLOOR_DB / 10)))
    return centres, level


def time_ticks(end_ms: float) -> list[tuple[float, str]]:
    """Return about five round ticks from 0 to end_ms."""
    step = 10 ** math.floor(math.log10(end_ms / 5))
    step *= next(k for k in (1, 2, 5, 10) if end_ms / (step * k) <= MAX_TICKS)
    ticks = np.arange(0, end_ms * (1 + 1e-9), step)
    return [(float(t), f"{t / 1000:g} s" if end_ms >= SECONDS_FROM_MS else f"{t:g} ms") for t in ticks]


FREQ_TICKS = [(100.0, "100"), (300.0, "300"), (1000.0, "1k"), (3000.0, "3k"), (10000.0, "10k"), (20000.0, "20k")]
DB_TICKS = [(0.0, "0"), (-20.0, "−20"), (-40.0, "−40"), (-60.0, "−60"), (-80.0, "−80")]


# The references.


@dataclass(frozen=True)
class Measured:
    """A reference as rendered: its patch, audio and report, and what the gallery says of it."""

    name: str
    patch: Patch
    audio: Floats
    report: render.Report

    @property
    def length_ms(self) -> float:
        """How long it sounds, to 40 dB below its loudest millisecond."""
        env = envelope_db(self.audio, RATE)
        above = np.flatnonzero(env > np.max(env) - 40)
        return 1000 * float(above[-1]) / RATE

    @property
    def loudness(self) -> float:
        """Its largest momentary loudness, in LUFS."""
        return self.report.files[0].loudness

    @property
    def centroid_hz(self) -> float:
        """Its spectral centroid."""
        power = np.abs(np.fft.rfft(self.audio)) ** 2
        f = np.fft.rfftfreq(self.audio.size, 1 / RATE)
        return float(np.sum(f * power) / np.sum(power))


def measure(name: str) -> Measured:
    """Render a reference as it ships."""
    p = loads(cli.reference_text(name))
    audio, report = render.render(p)
    return Measured(name, p, audio.astype(float), report)


def reference_figure(m: Measured) -> str:
    """Draw a reference's level over time and its spectrum."""
    env = envelope_db(m.audio, RATE, max(1.0, m.length_ms / 500))
    top = float(np.max(env))
    end_ms = min(1000 * m.audio.size / RATE, 2 * m.length_ms)
    t = np.arange(m.audio.size) * 1000 / RATE
    shown = (t <= end_ms) & (env > FLOOR_DB + 1)  # not where the sound falls silent
    left = Panel(70, 72, 290, 200, (0.0, end_ms), (FLOOR_DB, 0.0))
    right = Panel(440, 72, 290, 200, (LOW_HZ, HIGH_HZ), (-60.0, 0.0), xlog=True)
    f, level = spectrum_db(m.audio, RATE)
    body = (
        left.frame(time_ticks(end_ms), DB_TICKS, "level, dB below its loudest millisecond")
        + left.runs(t, env - top, shown, "c0")
        + right.frame(FREQ_TICKS, DB_TICKS[:4], "spectrum, Hz (1/6 octave)")
        + right.curve(f, level, "c1")
    )
    subtitle = f"{m.length_ms:.0f} ms to −40 dB, centroid {m.centroid_hz / 1000:.1f} kHz, {signed(m.loudness)} LUFS"
    return svg(320, m.name, subtitle, body, f"The {m.name} reference's level over time and its spectrum")


# How the instrument works.


def flow_figure() -> str:
    """Draw the signal flow, from the patch's sources to the file."""
    stages = [
        ("sources", "noise: modes or\nnoise; tones:\npartials"),
        ("sub-bands", "each on its\ntime map:\ndamping, bloom"),
        ("gates", "chatter, and\na rattle's\ncollisions"),
        ("damper", "a hand or\na foot on\nevery source"),
        ("drive", "tanh, at 32×\nthe sample\nrate"),
        ("decimator", "frozen, 116 dB\ndown, to the\nfile's rate"),
        ("file", "gain or peak;\nfloat or\ndithered PCM"),
    ]
    w, gap, y = 96, 8.67, 70
    body = []
    for i, (name, words) in enumerate(stages):
        x = 20 + i * (w + gap)
        body.append(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="104" rx="6"/>')
        body.append(text(x + w / 2, y + 22, name, "fg", "middle"))
        for k, line in enumerate(words.split("\n")):
            body.append(text(x + w / 2, y + 46 + 15 * k, line, "dim small", "middle"))
        if i:
            ax = x - gap
            body.append(f'<line class="axis" x1="{ax}" y1="{y + 52}" x2="{x - 2}" y2="{y + 52}"/>')
    note = "Every stage before the decimator is a pure function of time; every render says whether it is validated."
    body.append(text(20, 205, note))
    return svg(225, "Signal flow", "from a patch to a WAV", "".join(body), "The signal flow from patch to file")


def subbands_figure() -> str:
    """Draw the crash's sub-band envelopes: higher sub-bands rise later (bloom) and fall sooner (damping)."""
    v = voice.voice(loads(cli.reference_text("crash")))
    plates = v.sources[0]
    t = np.linspace(0.0, 2.5, 2501)
    panel = Panel(70, 72, 660, 230, (0.0, 2500.0), (-60.0, 0.0))
    body = [panel.frame(time_ticks(2500.0), DB_TICKS[:4], "time")]
    shown = plates.subbands[:: max(1, len(plates.subbands) // 4)]
    for k, sb in enumerate(shown):
        body.append(panel.curve(1000 * t, db(sb.env(t)), f"c{k % 4}"))
    centres = plates.layout.centres
    labels = ", ".join(f"{centres[plates.subbands.index(sb)] / 1000:.1f}" for sb in shown)
    subtitle = (
        f"crash plates, sub-bands centred at {labels} kHz: the higher, the later the peak and the sooner the fall"
    )
    return svg(350, "Damping and bloom", subtitle, "".join(body), "The crash's sub-band envelopes over time")


def modes_figure() -> str:
    """Draw a stretch of the closed hat's line spectrum, exactly as synthesis builds it: modes against noise."""
    p = loads(cli.reference_text("closed"))
    plates = voice.voice(p).sources[0]
    modes = plates.spec
    if not isinstance(modes, voice.NoiseSpec):
        msg = "the closed hat's plates are no longer noise"
        raise TypeError(msg)
    noise = dataclasses.replace(modes, kind="noise", spacing_hz=0.0)
    lo, hi = 8000.0, 8300.0
    ticks = [(f, f"{f / 1000:g}k") for f in np.arange(lo, hi + 1, 50)]
    body = []
    for row, (spec, cls, label) in enumerate(((modes, "c0", "modes"), (noise, "c3", "continuous noise"))):
        panel = Panel(70, 80 + 150 * row, 660, 100, (lo, hi), (-40.0, 0.0))
        body.append(panel.frame(ticks, [(0.0, "0"), (-20.0, "−20"), (-40.0, "−40")], ""))
        body.append(text(panel.x + panel.w, panel.y - 6, label, "fg", "end"))
        bins, amps = voice.lines(spec)
        hz = bins / 8
        keep = (hz >= lo) & (hz <= hi)
        level = db(np.abs(amps[keep]) / np.max(np.abs(amps[keep])))
        base = float(panel.py(np.array([-40.0]))[0])
        stems = "".join(
            f"M{x:.2f} {base:.1f}V{y:.1f}" for x, y in zip(panel.px(hz[keep]), panel.py(level), strict=True)
        )
        body.append(f'<path class="{cls}" d="{stems}"/>')
    subtitle = "closed hat plates, 8–8.3 kHz, every line: one mode to each 14 Hz cell, or a line on every 1/8 Hz"
    return svg(390, "Modes", subtitle, "".join(body), "Modes against continuous noise in the closed hat's lines")


def rattle_figure() -> str:
    """Draw the shaker's grains: its shape, and the collisions that sound under it."""
    v = voice.voice(loads(cli.reference_text("shaker")))
    grains = v.sources[0]
    if grains.grains is None:
        msg = "the shaker reference no longer rattles"
        raise RuntimeError(msg)
    end = float(grains.shape.t[-1])
    n = round(end * INTERNAL)
    t = np.arange(n) / INTERNAL
    e = grains.shape(t)
    r = grains.grains.on_grid(0, n, INTERNAL)
    panel = Panel(70, 72, 660, 200, (0.0, 1000 * end), (0.0, float(np.max(e * r)) * 1.05))
    body = [
        panel.frame(time_ticks(1000 * end), [], "time"),
        panel.curve(1000 * t, e * r, "c0"),
        panel.curve(1000 * t, e, "c1"),
    ]
    subtitle = (
        f"shaker grains: {grains.grains.onsets.size} collisions (blue) under the shape (pink), one to each cell "
        "of time, kept as the shape falls"
    )
    return svg(320, "A rattle", subtitle, "".join(body), "The shaker's collisions under its shape")


def kit_figure() -> str:
    """Draw how a kit's velocity layers follow the square law, from the closed hat struck softer."""
    base = loads(cli.reference_text("closed"))
    steps = (0, 8, 16, 24)
    loud = [render.render(softer(base, s))[1].files[0].loudness for s in steps]
    relative = [x - loud[0] for x in loud]
    ranges = pack.velocity_ranges(relative)
    v = np.arange(1, 128, dtype=float)
    panel = Panel(70, 72, 660, 220, (1.0, 127.0), (-42.0, 0.0))
    ticks = [(float(k), str(k)) for k in (1, 32, 64, 96, 127)]
    body = [panel.frame(ticks, [(0.0, "0"), (-12.0, "−12"), (-24.0, "−24"), (-36.0, "−36")], "velocity")]
    body.append(panel.curve(v, 40 * np.log10(v / 127), "c1"))
    for k, ((lo, hi), rel) in enumerate(zip(ranges, relative, strict=True)):
        x0, x1 = panel.px(np.array([float(lo), float(hi)]))
        y = float(panel.py(np.array([rel]))[0])
        body.append(f'<rect class="f{k % 4}" x="{x0:.1f}" y="{y - 3:.1f}" width="{x1 - x0:.1f}" height="6" rx="2"/>')
        body.append(text(float(x0), y - 8, f"layer {k + 1}: {steps[k]} dB softer", "dim"))
    subtitle = "four layers of the closed hat by `softer`, each placed where 40·log₁₀(v/127) reaches its loudness"
    return svg(340, "A kit's velocity layers", subtitle, "".join(body), "Velocity layers on the square law")


def headroom_figure(measured: Sequence[Measured]) -> str:
    """Draw every reference's drive headroom, the drive it takes before leaving the validated domain."""
    rows = []
    for m in measured:
        h = m.report.takes[0].headroom
        rows.append((m.name, 48.0 if h is None or h.db == math.inf else min(h.db, 48.0)))
    height = 80 + 24 * len(rows)
    panel = Panel(120, 64, 600, 24 * len(rows), (0.0, 48.0), (0.0, 1.0))
    body = [panel.frame([(float(d), f"{d} dB") for d in (0, 12, 24, 36, 48)], [], "")]
    for k, (name, value) in enumerate(rows):
        y = 64 + 24 * k + 5
        x1 = float(panel.px(np.array([value]))[0])
        body.append(text(112, y + 11, name, "fg", "end"))
        body.append(f'<rect class="f0" x="{panel.x}" y="{y}" width="{x1 - panel.x:.1f}" height="14" rx="2"/>')
    subtitle = "seed 1, take 1: how much drive each reference takes before its aliasing could pass −100 dB"
    return svg(height + 30, "Drive headroom", subtitle, "".join(body), "Every reference's drive headroom")


def overview_figure(measured: Sequence[Measured]) -> str:
    """Draw every reference's level over time, side by side, each over twice its length to −40 dB."""
    columns, cw, ch = 4, 172, 92
    body = []
    for k, m in enumerate(measured):
        col, row = k % columns, k // columns
        x, y = 24 + col * (cw + 10), 70 + row * (ch + 34)
        env = envelope_db(m.audio, RATE, max(1.0, m.length_ms / 500))
        end_ms = min(1000 * m.audio.size / RATE, 2 * m.length_ms)
        t = np.arange(m.audio.size) * 1000 / RATE
        panel = Panel(x, y, cw, ch, (0.0, end_ms), (-60.0, 0.0))
        body.append(f'<rect class="axis" fill="none" x="{x}" y="{y}" width="{cw}" height="{ch}"/>')
        body.append(panel.runs(t, env - float(np.max(env)), (t <= end_ms) & (env > FLOOR_DB + 1), f"c{k % 4}"))
        length = f"{m.length_ms / 1000:.1f} s" if m.length_ms >= MS_PER_S else f"{m.length_ms:.0f} ms"
        body.append(text(x, y + ch + 16, m.name, "fg"))
        body.append(text(x + cw, y + ch + 16, f"{length}, {m.centroid_hz / 1000:.1f} kHz", "dim small", "end"))
    rows = -(-len(measured) // columns)
    subtitle = "level over time, 60 dB deep, each over twice its length to −40 dB; its length and spectral centroid"
    return svg(70 + rows * (ch + 34), "The references", subtitle, "".join(body), "Every reference's level over time")


# The gallery.


def gallery(measured: Sequence[Measured]) -> str:
    """Write docs/references.md: every reference with its sound, its figure and what it measures."""
    lines = [
        "# The references",
        "",
        "<!-- Written by docs/build.py from the shipped references; edit those, then run it. -->",
        "",
        "Every reference is a complete starting patch: `brightwork init NAME > NAME.toml`. They are matched in",
        "loudness and share one output setting, so any of them combine into one kit. Each sound below is the",
        "reference as it ships, seed 1, take 1, as `brightwork render` writes it. [GUIDE.md](GUIDE.md) says what to",
        "try on each.",
        "",
        "| Reference | Length to −40 dB | Centroid | Loudness | Peak | Drive headroom |",
        "|---|---|---|---|---|---|",
    ]
    for m in measured:
        h = m.report.takes[0].headroom
        head = "any" if h is None or h.db == math.inf else f"{h.db:.1f} dB"
        lines.append(
            f"| [{m.name}](#{m.name}) | {m.length_ms:.0f} ms | {m.centroid_hz / 1000:.1f} kHz | "
            f"{signed(m.loudness)} LUFS | {signed(render.db(m.report.peak))} dBFS | {head} |",
        )
    for m in measured:
        lines += [
            "",
            f"## {m.name}",
            "",
            DESCRIPTIONS[m.name],
            "",
            f"[Listen: {m.name}.wav](audio/{m.name}.wav)",
            "",
            f"![{m.name}: level over time and spectrum](figures/{m.name}.svg)",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write every figure, sound and the gallery."""
    FIGURES.mkdir(exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    measured = [measure(name) for name in cli.REFERENCES]
    figures: dict[str, Callable[[], str]] = {
        "flow": flow_figure,
        "subbands": subbands_figure,
        "modes": modes_figure,
        "rattle": rattle_figure,
        "kit": kit_figure,
    }
    for name, draw in figures.items():
        (FIGURES / f"{name}.svg").write_text(draw(), encoding="utf-8", newline="\n")
    (FIGURES / "headroom.svg").write_text(headroom_figure(measured), encoding="utf-8", newline="\n")
    (FIGURES / "references.svg").write_text(overview_figure(measured), encoding="utf-8", newline="\n")
    for m in measured:
        (FIGURES / f"{m.name}.svg").write_text(reference_figure(m), encoding="utf-8", newline="\n")
        audio, report = render.render(m.patch)
        render.export(AUDIO / f"{m.name}.wav", audio, m.patch, report, report.files[0])
    (DOCS / "references.md").write_text(gallery(measured), encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote {len(figures) + 2 + len(measured)} figures, {len(measured)} sounds and the gallery\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
