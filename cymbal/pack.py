"""Kits: articulations on keys, velocity layers with takes, chokes, and an SFZ mapping (plan §7.3).

Each `KEY=PATCH[,PATCH…]` group is one articulation on one key; its patches
are velocity layers, loudest first, and each layer gets the same number of
files. File j of a layer plays its patch's take + j − 1, or with --stereo the
pair take + 2(j − 1) and the one after it. Every file of a kit shares one gain,
so layers and articulations keep their natural balance; in peak mode that gain
brings the loudest file to the target. Level matching instead scales each file
to the loudness of its layer's first.

The velocity mapping follows a square law, 40·log₁₀(v/127) dB, per key. Each
layer sits at the velocity where the law reaches its loudness relative to the
key's loudest layer, and layers meet halfway between, in dB. Each layer's group
carries a curve point at every velocity in its range, so the level follows
the law across layer boundaries.

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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .loudness import momentary_max
from .patch import DEFAULT_RELEASE_MS, Patch, PatchError, load
from .render import (
    MAX_RENDERS,
    Domain,
    RenderError,
    ceiling,
    db,
    domain,
    output_stage,
    synthesize,
    take_numbers,
    write_file,
)

MAX_FILES = 4096  # files in one kit


class PackError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layer:
    name: str  # the patch file's stem
    patch: Patch
    loudness: float  # its first file's, LUFS, in the kit's files
    relative_db: float  # to the key's loudest layer
    lovel: int
    hivel: int
    volume_db: float  # the group's volume, which with the curve places the layer on the square law
    curve: tuple[float, ...]  # amplitude at lovel … hivel, at most 1
    takes: tuple[tuple[int, ...], ...]  # per file, the takes it holds: one, or two for a stereo pair
    file_loudness: tuple[float, ...]
    file_gain_db: tuple[float, ...]
    domains: tuple[Domain, ...]  # per take

    @property
    def spread(self) -> float:
        values = [x for x in self.file_loudness if x > -math.inf]
        return max(values) - min(values) if values else 0.0

    def files(self, key: int) -> list[str]:
        return [f"{key}_{self.name}_v{self.hivel}_rr{j}.wav" for j in range(1, len(self.takes) + 1)]


@dataclass(frozen=True)
class Key:
    key: int
    layers: tuple[Layer, ...]
    audio: tuple[tuple[np.ndarray, ...], ...]  # per layer, per file
    group: int | None  # its choke group, when it chokes another key
    off_by: int | None  # the choke group that turns it off


def velocity_ranges(relative_db: list[float]) -> list[tuple[int, int]]:
    """(lovel, hivel) per layer, loudest first, from each layer's loudness relative to the loudest.

    Layer i sits where 40·log₁₀(v/127) reaches its level; neighbours meet at the
    geometric mean of their velocities, halfway in dB.
    """
    natural = [127.0 * 10.0 ** (d / 40.0) for d in relative_db]
    bounds = [127.0] + [math.sqrt(a * b) for a, b in zip(natural, natural[1:])] + [0.0]
    ranges = [(math.floor(bounds[i + 1]) + 1, math.floor(bounds[i])) for i in range(len(natural))]
    for i, (lo, hi) in enumerate(ranges):
        if lo > hi:
            raise PackError(f"layers {i} and {i + 1} are too close in loudness to get velocities of their own")
    return ranges


def curve_for(relative_db: float, lovel: int, hivel: int) -> tuple[float, tuple[float, ...]]:
    """The group volume in dB and the amplitude curve over lovel … hivel that put a layer on the square law.

    At velocity v the level should be 40·log₁₀(v/127) dB, and the layer's own
    level is relative_db, so its gain is (v/127)² / 10^(relative_db/20). The
    curve is that gain over its largest value, which becomes the volume.
    """
    gains = [(v / 127.0) ** 2 / 10.0 ** (relative_db / 20.0) for v in range(lovel, hivel + 1)]
    top = max(gains)
    return 20.0 * math.log10(top), tuple(g / top for g in gains)


def choke_groups(rules, keys) -> dict[int, tuple[int | None, int | None]]:
    """key → (its choke group, the group that turns it off), from rules (target, (choker, …)).

    The chokers of one target form one group; rules for the same target add to
    it. Each distinct group gets a number, and a key may belong to one only.
    """
    keys = set(keys)
    by_target: dict[int, set[int]] = {}
    for target, chokers in rules:
        text = f"--choke {target}:{','.join(map(str, chokers))}"
        for k in (target, *chokers):
            if k not in keys:
                raise PackError(f"{text}: key {k} is not in the kit")
        if target in chokers:
            raise PackError(f"{text}: a key cannot choke itself")
        by_target.setdefault(target, set()).update(chokers)
    numbers: dict[frozenset, int] = {}
    for chokers in by_target.values():
        numbers.setdefault(frozenset(chokers), len(numbers) + 1)
    member: dict[int, frozenset] = {}
    for group in numbers:
        for k in group:
            if k in member and member[k] != group:
                a, b = sorted(member[k]), sorted(group)
                raise PackError(
                    f"--choke: key {k} would need two choke groups, {','.join(map(str, a))} and "
                    f"{','.join(map(str, b))}; SFZ gives a region one group, so choke both targets with the same keys"
                )
            member[k] = group
    return {
        k: (numbers[member[k]] if k in member else None, numbers[frozenset(by_target[k])] if k in by_target else None)
        for k in keys
    }


def check_outputs(patches: list[tuple[str, Patch]]) -> None:
    first_name, first = patches[0]
    for name, p in patches[1:]:
        if p.output != first.output:
            fields = [
                f.name
                for f in dataclasses.fields(p.output)
                if getattr(p.output, f.name) != getattr(first.output, f.name)
            ]
            raise PackError(
                f"{name}: its output settings differ from {first_name}'s ({', '.join(f'output.{f}' for f in fields)}); "
                "the patches of a kit share one output"
            )


def _synthesize(p: Patch) -> tuple[np.ndarray, Domain]:
    s = synthesize(p)
    return s.out, domain(s)


@dataclass(frozen=True)
class Kit:
    stem: str
    files_per_layer: int
    stereo: bool
    set_gain_db: float
    keys: tuple[Key, ...]
    level_match: bool
    release_ms: float
    render_s: float

    def sfz(self) -> str:
        layers = sum(len(k.layers) for k in self.keys)
        lines = [
            f"// {self.stem}: {len(self.keys)} key{'s' if len(self.keys) > 1 else ''}, {layers} velocity "
            f"layer{'s' if layers > 1 else ''} × {self.files_per_layer} take{'s' if self.files_per_layer > 1 else ''}"
            f"{' (stereo pairs)' if self.stereo else ''}, written by cymbal {__version__}",
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
                    f"volume={layer.volume_db:.4f}",
                    *choke,
                ]
                # Ten decimals keep six significant digits down to velocity 1 of a 127-wide layer, (1/127)².
                lines += [
                    f"amp_velcurve_{v}={c:.10f}" for v, c in zip(range(layer.lovel, layer.hivel + 1), layer.curve)
                ]
                for j, name in enumerate(layer.files(k.key), 1):
                    lines.append(f"<region> sample={name} seq_position={j}")
        return "\n".join(lines) + "\n"

    def lines(self, out_dir) -> list[str]:
        files = sum(len(layer.takes) for k in self.keys for layer in k.layers)
        match = ", level-matched within each layer" if self.level_match else ""
        pairs = " stereo" if self.stereo else ""
        rows = [
            (
                "kit",
                f"{len(self.keys)} keys, {files}{pairs} files in {out_dir}, one gain of {self.set_gain_db:+.2f} dB{match}",
            )
        ]
        groups: dict[int, list[int]] = {}
        for k in self.keys:
            if k.group is not None:
                groups.setdefault(k.group, []).append(k.key)
        for k in self.keys:
            for i, layer in enumerate(k.layers, 1):
                worst = min(d.margin_db for d in layer.domains)
                inside = all(d.inside for d in layer.domains)
                status = (
                    f"inside, worst margin {worst:.1f} dB"
                    if inside
                    else f"{sum(not d.inside for d in layer.domains)} take(s) OUTSIDE the validated domain; "
                    "check with `cymbal verify`"
                )
                spread = f", take spread {layer.spread:.2f} dB" + (
                    " (over 1 dB: consider --level-match)" if layer.spread > 1.0 and not self.level_match else ""
                )
                rows.append(
                    (
                        f"key {k.key}" if i == 1 else "",
                        f"layer {i} {layer.name}: {layer.loudness:.1f} LUFS ({layer.relative_db:+.1f} dB), "
                        f"velocities {layer.lovel}–{layer.hivel}{spread}; {status}",
                    )
                )
                if layer.volume_db > 6.0:
                    rows.append(
                        (
                            "",
                            f"its group volume is {layer.volume_db:+.1f} dB; some players stop at +6 dB, so "
                            "keep neighbouring layers within 12 dB",
                        )
                    )
            if k.off_by is not None:
                by = ", ".join(map(str, groups[k.off_by]))
                rows.append(
                    ("", f"choked by key{'s' if len(groups[k.off_by]) > 1 else ''} {by}, over {self.release_ms:g} ms")
                )
        rows.append(("render", f"{self.render_s:.1f} s"))
        width = max(9, max(len(label) for label, _ in rows) + 1)
        return [f"{label:<{width}}{text}" for label, text in rows]


def build(
    groups: list[tuple[int, list[tuple[str, Patch]]]],
    *,
    takes: int = 1,
    stereo: bool = False,
    level_match: bool = False,
    chokes=(),
    release_ms: float = DEFAULT_RELEASE_MS,
    stem: str = "kit",
    workers: int | None = None,
) -> Kit:
    """Render every key's layers and takes and derive the mapping; nothing is written."""
    start = time.perf_counter()
    if not groups or not all(patches for _, patches in groups):
        raise PackError("a kit needs at least one key with at least one patch")
    seen = set()
    for key, _ in groups:
        if key in seen:
            raise PackError(f"key {key} is given twice; give each key's layers in one KEY=PATCH,… group")
        seen.add(key)
    if not (math.isfinite(release_ms) and release_ms > 0):
        raise PackError(f"--release: needs a positive time in ms, found {release_ms!r}")
    every = [pair for _, patches in groups for pair in patches]
    check_outputs(every)
    renders = takes * (2 if stereo else 1)
    if isinstance(takes, bool) or not isinstance(takes, int) or not 1 <= renders <= MAX_RENDERS:
        top = MAX_RENDERS // (2 if stereo else 1)
        raise PackError(
            f"--takes: needs 1 to {top} files per layer{' with --stereo' if stereo else ''}, found {takes!r}"
        )
    if len(every) * takes > MAX_FILES:
        raise PackError(f"the kit would have {len(every) * takes} files; at most {MAX_FILES}")
    roles = choke_groups(chokes, [k for k, _ in groups])
    first = every[0][1]
    numbers = [take_numbers(p, takes, stereo) for _, p in every]
    jobs = [dataclasses.replace(p, take=t) for (_, p), ts in zip(every, numbers) for t in ts]
    n = min(len(jobs), workers or os.cpu_count() or 1)
    if n > 1:
        # Workers start from a fork server, never by forking the caller: a thread of the caller holding a lock,
        # such as an audition watcher rendering, would otherwise leave a worker deadlocked on it.
        with ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("forkserver")) as pool:
            done = list(pool.map(_synthesize, jobs, chunksize=1))
    else:
        done = [_synthesize(p) for p in jobs]
    rate = first.output.sample_rate
    width = 2 if stereo else 1
    # Per patch, per file: the raw output (a pair stacked with --stereo), the takes it holds, and its takes' domains.
    raw, held, doms, i = [], [], [], 0
    for (_, p), ts in zip(every, numbers):
        files, which, ds = [], [], []
        for j in range(0, len(ts), width):
            outs = [done[i + j + c][0] for c in range(width)]
            files.append(np.stack(outs, axis=1) if stereo else outs[0])
            which.append(tuple(ts[j : j + width]))
            ds += [done[i + j + c][1] for c in range(width)]
        raw.append(files)
        held.append(tuple(which))
        doms.append(tuple(ds))
        i += len(ts)
    loud = [[momentary_max(f, rate) for f in files] for files in raw]
    gains = [
        [(10.0 ** ((row[0] - x) / 20.0) if level_match and row[0] > -math.inf and x > -math.inf else 1.0) for x in row]
        for row in loud
    ]
    if first.output.mode == "gain":
        set_gain = 10.0 ** (first.output.db / 20.0)
    else:
        peak = max(
            float(np.max(np.abs(f), initial=0.0)) * g for files, gs in zip(raw, gains) for f, g in zip(files, gs)
        )
        target = min(10.0 ** (first.output.db / 20.0), ceiling(first.output.format))
        set_gain = target / peak if peak > 0 else 1.0
    level = db(set_gain)
    keys, index = [], 0
    for key, patches in groups:
        rows = range(index, index + len(patches))
        index += len(patches)
        firsts = [loud[r][0] for r in rows]
        names = [name for name, _ in patches]
        if any(x == -math.inf for x in firsts):
            raise PackError(f"key {key}: {names[firsts.index(-math.inf)]} is silent and cannot be mapped")
        for a in range(1, len(firsts)):
            if not firsts[a] < firsts[a - 1]:
                raise PackError(
                    f"key {key}: {names[a]} ({firsts[a]:.1f} LUFS) is not softer than {names[a - 1]} "
                    f"({firsts[a - 1]:.1f} LUFS): give each key's layers loudest first"
                )
        relative = [x - firsts[0] for x in firsts]
        ranges = velocity_ranges(relative)
        layers, audio = [], []
        for a, (r, (name, p), (lo, hi)) in enumerate(zip(rows, patches, ranges)):
            volume, curve = curve_for(relative[a], lo, hi)
            files = tuple(output_stage(f, set_gain * g, p.output.polarity) for f, g in zip(raw[r], gains[r]))
            for f in files:
                if not np.all(np.isfinite(f)):
                    raise RenderError("the render produced non-finite samples")
            audio.append(files)
            layers.append(
                Layer(
                    name,
                    p,
                    firsts[a] + level,
                    relative[a],
                    lo,
                    hi,
                    volume,
                    curve,
                    held[r],
                    tuple(x + level + db(g) for x, g in zip(loud[r], gains[r])),
                    tuple(db(g) for g in gains[r]),
                    doms[r],
                )
            )
        group, off_by = roles[key]
        keys.append(Key(key, tuple(layers), tuple(audio), group, off_by))
    return Kit(stem, takes, stereo, level, tuple(keys), level_match, release_ms, time.perf_counter() - start)


def over(kit: Kit) -> float:
    """The largest sample over the format's ceiling, or 0 when the kit fits."""
    top = ceiling(kit.keys[0].layers[0].patch.output.format)
    peak = max(float(np.max(np.abs(f), initial=0.0)) for k in kit.keys for files in k.audio for f in files)
    return peak if peak > top else 0.0


def write(kit: Kit, out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(layer.takes) for k in kit.keys for layer in k.layers)
    written = []
    for k in kit.keys:
        for i, (layer, files) in enumerate(zip(k.layers, k.audio), 1):
            for j, (name, a) in enumerate(zip(layer.files(k.key), files)):
                path = out_dir / name
                write_file(
                    path,
                    a,
                    layer.patch,
                    takes=layer.takes[j],
                    files=total,
                    set_gain_db=kit.set_gain_db,
                    file_gain_db=layer.file_gain_db[j],
                    extra={
                        "kit": {
                            "key": k.key,
                            "layer": i,
                            "layers": len(k.layers),
                            "lovel": layer.lovel,
                            "hivel": layer.hivel,
                            "group": k.group,
                            "off_by": k.off_by,
                        }
                    },
                )
                written.append(path)
    sfz = out_dir / f"{kit.stem}.sfz"
    sfz.write_text(kit.sfz(), encoding="utf-8")
    written.append(sfz)
    return written


def pack(
    groups,
    out_dir,
    *,
    takes: int = 1,
    stereo: bool = False,
    level_match: bool = False,
    chokes=(),
    release_ms: float = DEFAULT_RELEASE_MS,
    out=None,
) -> int:
    """`cymbal pack`: render the keys, layers and takes, and write the files and the SFZ mapping."""
    out = out or sys.stdout
    loaded = []
    for key, paths in groups:
        patches = []
        for path in paths:
            try:
                patches.append((Path(path).stem, load(path)))
            except OSError as e:
                print(f"{path}: {e.strerror}", file=sys.stderr)
                return 2
            except PatchError as e:
                print(f"{path}: invalid patch\n" + "\n".join(f"  {m}" for m in e.problems), file=sys.stderr)
                return 2
        loaded.append((key, patches))
    stem = Path(out_dir).resolve().name or "kit"
    try:
        kit = build(
            loaded, takes=takes, stereo=stereo, level_match=level_match, chokes=chokes, release_ms=release_ms, stem=stem
        )
    except (PackError, RenderError) as e:
        print(e, file=sys.stderr)
        return 2
    peak = over(kit)
    if peak:
        print("\n".join(kit.lines(out_dir)), file=out)
        top = ceiling(kit.keys[0].layers[0].patch.output.format)
        print(
            f"not written: the loudest file peaks at {db(peak):+.2f} dBFS, over {db(top):.2f} dBFS; lower output.db by "
            f"at least {db(peak / top):.2f} dB in every patch",
            file=sys.stderr,
        )
        return 1
    written = write(kit, out_dir)  # every file first, so a closed output never leaves a kit half written
    print("\n".join(kit.lines(out_dir)), file=out)
    for path in written:
        print(f"wrote {path}", file=out)
    return 0
