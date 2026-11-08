# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Kits: articulations on keys, velocity layers with takes, chokes, and an SFZ mapping (plan §7.3).

Each `KEY=PATCH[,PATCH…]` group is one articulation on one key; its patches
are velocity layers, loudest first, and each layer gets the same number of
files. File j of a layer plays its patch's take + j − 1, or with --stereo the
pair take + 2(j − 1) and the one after it. Every file of a kit shares one gain,
so layers and articulations keep their natural balance; in peak mode that gain
brings the loudest file to the target. Level matching instead scales each file
to the loudness of its layer's first.

The velocity mapping follows a square law, 40·log₁₀(v/127) dB, per key. Each
layer's velocities run up to the one where the law reaches its loudness
relative to the key's loudest layer, from just above the softer layer's, and
its group carries a curve point at every velocity in its range, so the level
follows the law at every velocity and no take ever plays louder than it was
rendered.

Chokes: `--choke 46:42,44` puts keys 42 and 44 in one SFZ group and turns key
46 off whenever one of them plays (`group`, `off_by`), fading over the release
rather than cutting (`off_mode=normal`, `ampeg_release`). A key can belong to
one choke group only, so rules that would need two are refused.
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from . import __version__, carrier
from .loudness import momentary_max
from .patch import DEFAULT_RELEASE_MS, Patch, PatchError, load
from .render import (
    MAX_RENDERS,
    Domain,
    Placement,
    RenderError,
    ceiling,
    db,
    domain,
    matched_gains,
    output_stage,
    set_gain,
    synthesize,
    take_numbers,
    write_file,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

type Floats = NDArray[np.float64]
type Audio = NDArray[np.float32]
type Rule = tuple[int, tuple[int, ...]]  # a choke rule: the key choked, and the keys that choke it
type Group = tuple[int, list[tuple[str, Patch]]]  # a key, and its layers' names and patches, loudest first

MAX_FILES = 4096  # files in one kit


class PackError(RuntimeError):
    """A kit that cannot be built, with what to change."""


SFZ_SYNTAX = ("<", ">", "=", "$", "\\", "//", "/*")  # what a sampler reads as syntax even inside a sample's name


def sfz_name_problem(name: str) -> str | None:
    """Return why `name` cannot stand for itself in an SFZ map, or None when it can.

    A sampler reads a map line by line: `<` opens a header, `word=` an opcode,
    `$` a variable, `//` and `/*` a comment and a backslash a folder, and the
    space around a value is trimmed. A name holding any of these, a control
    character or a line break would be read as something else: a crafted file
    name could add regions or opcodes to the map.
    """
    breaks = sorted({f"U+{ord(c):04X}" for c in name if unicodedata.category(c) in {"Cc", "Zl", "Zp"}})
    found = [*breaks, *(s for s in SFZ_SYNTAX if s in name)]
    if found:
        return f"{name!r} cannot name a file in an SFZ map: it holds {' '.join(found)}; rename it"
    if name != name.strip():
        return f"{name!r} cannot name a file in an SFZ map: a sampler trims the space around it; rename it"
    return None


@dataclass(frozen=True)
class Options:
    """How a kit is rendered: files per layer, stereo pairs, level matching, its chokes and their release."""

    takes: int = 1
    stereo: bool = False
    level_match: bool = False
    chokes: tuple[Rule, ...] = ()
    release_ms: float = DEFAULT_RELEASE_MS


DEFAULTS = Options()


@dataclass(frozen=True)
class Layer:
    """One velocity layer of a key: its patch, loudness, velocities, curve, files and their domains."""

    name: str  # the patch file's stem
    patch: Patch
    loudness: float  # its first file's, LUFS, in the kit's files
    relative_db: float  # to the key's loudest layer
    lovel: int
    hivel: int
    curve: tuple[float, ...]  # amplitude at lovel … hivel, at most 1
    takes: tuple[tuple[int, ...], ...]  # per file, the takes it holds: one, or two for a stereo pair
    file_loudness: tuple[float, ...]
    file_gain_db: tuple[float, ...]
    domains: tuple[Domain, ...]  # per take

    @property
    def spread(self) -> float:
        """How far apart its files' loudnesses lie."""
        values = [x for x in self.file_loudness if x > -math.inf]
        return max(values) - min(values) if values else 0.0

    def files(self, key: int) -> list[str]:
        """Name its files on a key."""
        return [f"{key}_{self.name}_v{self.hivel}_rr{j}.wav" for j in range(1, len(self.takes) + 1)]


@dataclass(frozen=True)
class Key:
    """One articulation on one key: its layers, their audio, and its choke groups."""

    key: int
    layers: tuple[Layer, ...]
    audio: tuple[tuple[Audio, ...], ...]  # per layer, per file
    group: int | None  # its choke group, when it chokes another key
    off_by: int | None  # the choke group that turns it off


def velocity_ranges(relative_db: list[float]) -> list[tuple[int, int]]:
    """Return (lovel, hivel) per layer, loudest first, from each layer's loudness relative to the loudest.

    A layer's velocities run up to its own, the velocity where 40·log₁₀(v/127)
    reaches its level, from just above the softer layer's own. So the map only
    ever lowers a take, never plays it louder than it was rendered, and the
    law holds at every velocity.
    """
    own = [math.floor(127.0 * 10.0 ** (d / 40.0) + 1e-9) for d in relative_db]  # the law's inverse
    lows = [*(hi + 1 for hi in own[1:]), 1]
    ranges = list(zip(lows, own, strict=True))
    for i, (lo, hi) in enumerate(ranges):
        if lo > hi:
            msg = (
                f"layers {i + 1} and {i + 2} are within a velocity step of each other in loudness"
                if i + 1 < len(ranges)
                else f"layer {i + 1} is {-relative_db[i]:.1f} dB under the loudest, below the square law at velocity 1"
            )
            raise PackError(msg)
    return ranges


def curve_for(relative_db: float, lovel: int, hivel: int) -> tuple[float, ...]:
    """Return the amplitude over lovel … hivel that puts a layer on the square law, never above 1.

    At velocity v the level should be 40·log₁₀(v/127) dB, and the layer's own
    level is relative_db, so its gain is (v/127)² / 10^(relative_db/20): at most
    1 up to the layer's own velocity, where its range ends.
    """
    return tuple((v / 127.0) ** 2 / 10.0 ** (relative_db / 20.0) for v in range(lovel, hivel + 1))


def choke_groups(rules: Iterable[Rule], kit_keys: Iterable[int]) -> dict[int, tuple[int | None, int | None]]:
    """Return key → (its choke group, the group that turns it off), from rules (target, (choker, …)).

    The chokers of one target form one group; rules for the same target add to
    it. Each distinct group gets a number, and a key may belong to one only.
    """
    keys = set(kit_keys)
    by_target: dict[int, set[int]] = {}
    for target, chokers in rules:
        text = f"--choke {target}:{','.join(map(str, chokers))}"
        for k in (target, *chokers):
            if k not in keys:
                msg = f"{text}: key {k} is not in the kit"
                raise PackError(msg)
        if target in chokers:
            msg = f"{text}: a key cannot choke itself"
            raise PackError(msg)
        by_target.setdefault(target, set()).update(chokers)
    numbers: dict[frozenset[int], int] = {}
    for members in by_target.values():
        numbers.setdefault(frozenset(members), len(numbers) + 1)
    member: dict[int, frozenset[int]] = {}
    for group in numbers:
        for k in group:
            if k in member and member[k] != group:
                a, b = sorted(member[k]), sorted(group)
                msg = (
                    f"--choke: key {k} would need two choke groups, {','.join(map(str, a))} and "
                    f"{','.join(map(str, b))}; SFZ gives a region one group, so choke both targets with the same keys"
                )
                raise PackError(msg)
            member[k] = group
    return {
        k: (numbers[member[k]] if k in member else None, numbers[frozenset(by_target[k])] if k in by_target else None)
        for k in keys
    }


def check_outputs(patches: list[tuple[str, Patch]]) -> None:
    """Refuse patches whose output settings differ: the patches of a kit share one output."""
    first_name, first = patches[0]
    for name, p in patches[1:]:
        if p.output != first.output:
            fields = [
                f.name
                for f in dataclasses.fields(p.output)
                if getattr(p.output, f.name) != getattr(first.output, f.name)
            ]
            differ = ", ".join(f"output.{f}" for f in fields)
            msg = f"{name}: its output settings differ from {first_name}'s ({differ}); a kit's patches share one output"
            raise PackError(msg)


def _synthesize(p: Patch) -> tuple[Floats, float, Domain]:
    """Synthesize one take, with its true peak, and judge its domain, in a worker."""
    s = synthesize(p)
    return s.out, s.true_peak, domain(s)


@dataclass(frozen=True)
class Kit:
    """A kit built and ready to write: its keys, layers and audio, its gain, and how it was made."""

    stem: str
    files_per_layer: int
    stereo: bool
    set_gain_db: float
    keys: tuple[Key, ...]
    level_match: bool
    release_ms: float
    render_s: float

    def sfz(self) -> str:
        """Write the SFZ mapping."""
        layers = sum(len(k.layers) for k in self.keys)
        title = (
            f"// {self.stem}: {len(self.keys)} key{'s' if len(self.keys) > 1 else ''}, {layers} velocity "
            f"layer{'s' if layers > 1 else ''} × {self.files_per_layer} take{'s' if self.files_per_layer > 1 else ''}"
            f"{' (stereo pairs)' if self.stereo else ''}, written by brightwork {__version__}"
        )
        lines = [
            title,
            "// Velocity follows a square law, 40·log10(v/127) dB, across layer boundaries.",
            "",
            "<global>",
            "pitch_keytrack=0",
            "amp_veltrack=100",
            "loop_mode=one_shot",
        ]
        for k in self.keys:
            choke = []
            if k.group is not None:
                choke.append(f"group={k.group}")
            if k.off_by is not None:
                choke += [f"off_by={k.off_by}", "off_mode=normal", f"ampeg_release={self.release_ms / 1000:.6g}"]
            for layer in k.layers:
                lines += [
                    "",
                    f"// key {k.key}, {layer.name}: {layer.loudness:.1f} LUFS, {layer.relative_db:+.2f} dB",
                    "<group>",
                    f"key={k.key}",
                    f"lovel={layer.lovel}",
                    f"hivel={layer.hivel}",
                    f"seq_length={len(layer.takes)}",
                    *choke,
                ]
                # Ten decimals keep six significant digits down to velocity 1 of a 127-wide layer, (1/127)².
                lines += [
                    f"amp_velcurve_{v}={c:.10f}"
                    for v, c in zip(range(layer.lovel, layer.hivel + 1), layer.curve, strict=True)
                ]
                for j, name in enumerate(layer.files(k.key), 1):
                    lines.append(f"<region> sample={name} seq_position={j}")
        return "\n".join(lines) + "\n"

    def lines(self, out_dir: str | os.PathLike[str]) -> list[str]:
        """Return the kit's report, as `brightwork pack` prints it."""
        files = sum(len(layer.takes) for k in self.keys for layer in k.layers)
        match = ", level-matched within each layer" if self.level_match else ""
        pairs = " stereo" if self.stereo else ""
        gain = f"one gain of {self.set_gain_db:+.2f} dB{match}"
        rows = [("kit", f"{len(self.keys)} keys, {files}{pairs} files in {out_dir}, {gain}")]
        groups: dict[int, list[int]] = {}
        for k in self.keys:
            if k.group is not None:
                groups.setdefault(k.group, []).append(k.key)
        for k in self.keys:
            for i, layer in enumerate(k.layers, 1):
                worst = min(d.margin_db for d in layer.domains)
                inside = all(d.inside for d in layer.domains)
                outside = sum(not d.inside for d in layer.domains)
                status = (
                    f"inside, worst margin {worst:.1f} dB"
                    if inside
                    else f"{outside} take(s) OUTSIDE the validated domain; check with `brightwork verify`"
                )
                spread = f", take spread {layer.spread:.2f} dB" + (
                    " (over 1 dB: consider --level-match)" if layer.spread > 1.0 and not self.level_match else ""
                )
                text = (
                    f"layer {i} {layer.name}: {layer.loudness:.1f} LUFS ({layer.relative_db:+.1f} dB), "
                    f"velocities {layer.lovel}–{layer.hivel}{spread}; {status}"
                )
                rows.append((f"key {k.key}" if i == 1 else "", text))
            if k.off_by is not None:
                by = ", ".join(map(str, groups[k.off_by]))
                rows.append(
                    ("", f"choked by key{'s' if len(groups[k.off_by]) > 1 else ''} {by}, over {self.release_ms:g} ms"),
                )
        rows.append(("render", f"{self.render_s:.1f} s"))
        width = max(9, max(len(label) for label, _ in rows) + 1)
        return [f"{label:<{width}}{text}" for label, text in rows]


@dataclass(frozen=True)
class Rendered:
    """Every patch's files at unity gain: per patch, per file, the output, takes, peaks, loudness and level match."""

    raw: list[list[Floats]]  # a stereo pair stacked as (samples, 2)
    held: list[tuple[tuple[int, ...], ...]]  # the takes each file holds
    domains: list[tuple[Domain, ...]]  # per take
    true_peaks: list[list[float]]  # per file, at unity gain
    loud: list[list[float]]  # LUFS
    gains: list[list[float]]  # from level matching, 1 without


def _check(groups: Sequence[Group], options: Options, every: list[tuple[str, Patch]]) -> None:
    """Refuse a kit that cannot be built, before anything renders."""
    if not groups or not all(patches for _, patches in groups):
        msg = "a kit needs at least one key with at least one patch"
        raise PackError(msg)
    seen = set()
    for key, _ in groups:
        if key in seen:
            msg = f"key {key} is given twice; give each key's layers in one KEY=PATCH,… group"
            raise PackError(msg)
        seen.add(key)
    if not (math.isfinite(options.release_ms) and options.release_ms > 0):
        msg = f"--release: needs a positive time in ms, found {options.release_ms!r}"
        raise PackError(msg)
    check_outputs(every)
    takes, stereo = options.takes, options.stereo
    if isinstance(takes, bool) or not 1 <= takes * (2 if stereo else 1) <= MAX_RENDERS:
        top = MAX_RENDERS // (2 if stereo else 1)
        msg = f"--takes: needs 1 to {top} files per layer{' with --stereo' if stereo else ''}, found {takes!r}"
        raise PackError(msg)
    if len(every) * takes > MAX_FILES:
        msg = f"the kit would have {len(every) * takes} files; at most {MAX_FILES}"
        raise PackError(msg)


def _render(every: list[tuple[str, Patch]], options: Options, workers: int | None) -> Rendered:
    """Render every take of every patch, in parallel processes, and gather them into files."""
    numbers = [take_numbers(p, options.takes, stereo=options.stereo) for _, p in every]
    jobs = [dataclasses.replace(p, take=t) for (_, p), ts in zip(every, numbers, strict=True) for t in ts]
    n = min(len(jobs), workers or os.cpu_count() or 1)
    if n > 1:
        # Workers start from a fork server, or fresh where there is none (Windows), never by forking the caller: a
        # thread of the caller holding a lock, such as an audition watcher rendering, would leave a worker deadlocked.
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        context = multiprocessing.get_context(method)
        with ProcessPoolExecutor(n, mp_context=context, initializer=carrier.share_cores, initargs=(n,)) as pool:
            done = list(pool.map(_synthesize, jobs, chunksize=1))
    else:
        done = [_synthesize(p) for p in jobs]
    width = 2 if options.stereo else 1
    raw: list[list[Floats]] = []
    held: list[tuple[tuple[int, ...], ...]] = []
    doms: list[tuple[Domain, ...]] = []
    true_peaks: list[list[float]] = []
    i = 0
    for ts in numbers:
        files, which, tps = [], [], []
        ds: list[Domain] = []
        for j in range(0, len(ts), width):
            outs, peaks, takes = zip(*done[i + j : i + j + width], strict=True)
            files.append(np.stack(outs, axis=1) if options.stereo else outs[0])
            which.append(tuple(ts[j : j + width]))
            ds += takes
            tps.append(max(peaks))
        raw.append(files)
        held.append(tuple(which))
        doms.append(tuple(ds))
        true_peaks.append(tps)
        i += len(ts)
    rate = every[0][1].output.sample_rate
    loud = [[momentary_max(f, rate) for f in files] for files in raw]
    gains = [matched_gains(row) if options.level_match else [1.0] * len(row) for row in loud]
    return Rendered(raw, held, doms, true_peaks, loud, gains)


def _set_gain(first: Patch, r: Rendered) -> float:
    """Return the gain every file shares, from every file's peaks after its level match."""
    gained = [
        (f, tp, g) for row in zip(r.raw, r.true_peaks, r.gains, strict=True) for f, tp, g in zip(*row, strict=True)
    ]
    peaks = [float(np.max(np.abs(f), initial=0.0)) * g for f, _, g in gained]
    return set_gain(first.output, peaks, [tp * g for _, tp, g in gained])


def _layers(group: Group, rows: range, r: Rendered, set_gain: float) -> tuple[list[Layer], list[tuple[Audio, ...]]]:
    """Map one key's layers onto velocities and scale their files by the kit's gain."""
    key, patches = group
    level = db(set_gain)
    firsts = [r.loud[i][0] for i in rows]
    names = [name for name, _ in patches]
    if any(x == -math.inf for x in firsts):
        msg = f"key {key}: {names[firsts.index(-math.inf)]} is silent and cannot be mapped"
        raise PackError(msg)
    for a in range(1, len(firsts)):
        if not firsts[a] < firsts[a - 1]:
            msg = (
                f"key {key}: {names[a]} ({firsts[a]:.1f} LUFS) is not softer than {names[a - 1]} "
                f"({firsts[a - 1]:.1f} LUFS): give each key's layers loudest first"
            )
            raise PackError(msg)
    relative = [x - firsts[0] for x in firsts]
    layers, audio = [], []
    for a, (i, (name, p), (lo, hi)) in enumerate(zip(rows, patches, velocity_ranges(relative), strict=True)):
        curve = curve_for(relative[a], lo, hi)
        files = tuple(
            output_stage(f, set_gain * g, p.output.polarity) for f, g in zip(r.raw[i], r.gains[i], strict=True)
        )
        if not all(np.all(np.isfinite(f)) for f in files):
            msg = "the render produced non-finite samples"
            raise RenderError(msg)
        audio.append(files)
        loudness = tuple(x + level + db(g) for x, g in zip(r.loud[i], r.gains[i], strict=True))
        gains = tuple(db(g) for g in r.gains[i])
        layers.append(
            Layer(
                name,
                p,
                firsts[a] + level,
                relative[a],
                lo,
                hi,
                curve,
                r.held[i],
                loudness,
                gains,
                r.domains[i],
            ),
        )
    return layers, audio


def build(
    groups: Sequence[Group],
    options: Options = DEFAULTS,
    *,
    stem: str = "kit",
    workers: int | None = None,
) -> Kit:
    """Render every key's layers and takes and derive the mapping; nothing is written."""
    start = time.perf_counter()
    every = [pair for _, patches in groups for pair in patches]
    names = dict.fromkeys([stem, *(name for name, _ in every)])
    if unnamable := [problem for name in names if (problem := sfz_name_problem(name))]:
        raise PackError("\n".join(unnamable))
    _check(groups, options, every)
    roles = choke_groups(options.chokes, [k for k, _ in groups])
    rendered = _render(every, options, workers)
    set_gain = _set_gain(every[0][1], rendered)
    keys, index = [], 0
    for group in groups:
        rows = range(index, index + len(group[1]))
        index += len(group[1])
        layers, audio = _layers(group, rows, rendered, set_gain)
        keys.append(Key(group[0], tuple(layers), tuple(audio), *roles[group[0]]))
    elapsed = time.perf_counter() - start
    o = options
    return Kit(stem, o.takes, o.stereo, db(set_gain), tuple(keys), o.level_match, o.release_ms, elapsed)


def over(kit: Kit) -> float:
    """Return the largest sample over the format's ceiling, or 0 when the kit fits."""
    top = ceiling(kit.keys[0].layers[0].patch.output.format)
    peak = max(float(np.max(np.abs(f), initial=0.0)) for k in kit.keys for files in k.audio for f in files)
    return peak if peak > top else 0.0


def write(kit: Kit, folder: str | os.PathLike[str]) -> list[Path]:
    """Write every file of the kit and its SFZ mapping into a folder, and return their paths."""
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(layer.takes) for k in kit.keys for layer in k.layers)
    written = []
    for k in kit.keys:
        for i, (layer, files) in enumerate(zip(k.layers, k.audio, strict=True), 1):
            for j, (name, a) in enumerate(zip(layer.files(k.key), files, strict=True)):
                path = out_dir / name
                write_file(
                    path,
                    a,
                    layer.patch,
                    Placement(layer.takes[j], total, kit.set_gain_db, layer.file_gain_db[j]),
                    extra={
                        "kit": {
                            "key": k.key,
                            "layer": i,
                            "layers": len(k.layers),
                            "lovel": layer.lovel,
                            "hivel": layer.hivel,
                            "group": k.group,
                            "off_by": k.off_by,
                        },
                    },
                )
                written.append(path)
    sfz = out_dir / f"{kit.stem}.sfz"
    sfz.write_text(kit.sfz(), encoding="utf-8", newline="\n")  # the same bytes on every platform
    written.append(sfz)
    return written


def pack(
    groups: Iterable[tuple[int, Iterable[str]]],
    out_dir: str | os.PathLike[str],
    options: Options = DEFAULTS,
    *,
    out: TextIO | None = None,
) -> int:
    """Run `brightwork pack`: render the keys, layers and takes, and write the files and the SFZ mapping."""
    report = out or sys.stdout
    loaded: list[Group] = []
    for key, paths in groups:
        patches = []
        for path in paths:
            try:
                patches.append((Path(path).stem, load(path)))
            except OSError as e:
                sys.stderr.write(f"{path}: {e.strerror}\n")
                return 2
            except PatchError as e:
                sys.stderr.write(f"{path}: invalid patch\n" + "".join(f"  {m}\n" for m in e.problems))
                return 2
        loaded.append((key, patches))
    stem = Path(out_dir).resolve().name or "kit"
    try:
        kit = build(loaded, options, stem=stem)
    except (PackError, RenderError) as e:
        sys.stderr.write(f"{e}\n")
        return 2
    peak = over(kit)
    if peak:
        report.write("\n".join(kit.lines(out_dir)) + "\n")
        top = ceiling(kit.keys[0].layers[0].patch.output.format)
        sys.stderr.write(
            f"not written: the loudest file peaks at {db(peak):+.2f} dBFS, over {db(top):.2f} dBFS; lower output.db by "
            f"at least {db(peak / top):.2f} dB in every patch\n",
        )
        return 1
    written = write(kit, out_dir)  # every file first, so a closed output never leaves a kit half written
    report.write("\n".join(kit.lines(out_dir)) + "\n")
    report.write("".join(f"wrote {path}\n" for path in written))
    return 0
