# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: a complete, readable description of one hit (plan §4, §5).

Every field is required and the renderer holds no defaults, so a future
default can never change an old patch; the source arrays, `[[noise]]` and
`[[tone]]`, may be absent, meaning none. A Patch validates itself whenever it
is constructed, by the parser or by an edit, so every Patch object lies inside
the valid domain; the parser and the editing commands reject everything else
and never clamp or repair. Problems name their fields by path, counting from 1:
`noise[2].amp[3].ms` is the time of the third knot of the second noise source.

The editing commands (`pitch`, `stretch`, `choke`, `softer`) rewrite points and
never add a parameter. They compute on the decimal values as written, so the
patch they print stays readable.
"""

from __future__ import annotations

import codecs
import dataclasses
import datetime as dt
import math
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .spectrum import MAX_COMPRESSION, MAX_SEED, MAX_TAKE, TOP_HZ, grid_bin, layout, worst_compression
from .storage import tail_s

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterable, Sequence

type Knots = tuple[tuple[float, float], ...]  # a curve's (ms, value) knots
type Row = tuple[str, str, str]  # a canonical line's key, value and comment

INSTRUMENT = "cymbal"
SCHEMA = 1
SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)
FORMATS = ("float32", "pcm24", "pcm16")
WAVES = ("sine", "square")
PHASES = ("strike", "free")
STORAGE_BITS = (0, *range(2, 17))  # 0: no storage
STORAGE_RATES_HZ = (4000.0, 96000.0)
STORAGE_FILTERS_HZ = (1000.0, 20000.0)
MAX_GRAINS_HZ = 4000.0  # collisions per second
MIN_GRAIN_MS, MAX_GRAIN_MS = 1.0, 1000.0  # how long a collision rings
MAX_OVERLAP = 64.0  # collisions ringing at once, at the shape's peak
MODES = ("gain", "peak", "true-peak")  # the output's: a gain, or a sample or true peak target at or below full scale
MAX_MS = 8000.0  # every carrier repeats after 8 s
MIN_GRAINS_HZ = 1000.0 / MAX_MS  # slower, even the longest sound holds less than one collision: no rattle
MAX_NOISE = 4
MAX_TONES = 2
MAX_PARTIALS = 16
MAX_KNOTS = 32
MAX_SPACING = 200.0
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written
PAIR = 2  # a knot, a partial and a band are two numbers each


class PatchError(ValueError):
    """An invalid patch or edit. `problems` lists every reason, each with its field path."""

    def __init__(self, problems: Iterable[str]) -> None:
        """Keep every problem, and make the message one problem per line."""
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Noise:
    """A noise source: continuous noise, or seeded modes at a mean spacing, through a band, under a shape."""

    level_db: float
    band_hz: tuple[float, float]  # (low, high)
    spacing_hz: float  # 0: continuous noise
    damping: float
    bloom: float
    chatter: float
    chatter_hz: float
    grains_hz: float  # 0: no rattle
    grain_ms: float
    amp: Knots  # ((ms, gain), ...)

    @property
    def f_ref(self) -> float:
        """The band's lower edge, from which damping and bloom count (plan §2.6)."""
        return self.band_hz[0]


@dataclass(frozen=True)
class Tone:
    """A tone: sine or band-limited square partials at fixed frequencies, through a band, under a shape."""

    level_db: float
    wave: str
    phase: str
    partials: tuple[tuple[float, float], ...]  # ((hz, level_db), ...)
    band_hz: tuple[float, float]
    damping: float
    bloom: float
    chatter: float
    chatter_hz: float
    grains_hz: float
    grain_ms: float
    amp: Knots

    @property
    def f_ref(self) -> float:
        """The higher of the band's lower edge and the lowest partial (plan §2.6)."""
        return max(self.band_hz[0], self.partials[0][0])


@dataclass(frozen=True)
class Storage:
    """Low-resolution storage, as in a drum machine's sample memory: off at 0 bits."""

    bits: int  # 0: off; else 2 … 16
    rate_hz: float  # the rate the sound is stored at, as near as the internal rate's divisors allow
    filter_hz: float  # the playback low-pass's −3 dB point


@dataclass(frozen=True)
class Damper:
    """The damper, which multiplies every source: no knots, or a curve from 1 down to 0."""

    amp: Knots  # ((ms, gain), ...)


@dataclass(frozen=True)
class Drive:
    """The saturator's small-signal gain in dB; 0 is bypass."""

    db: float


@dataclass(frozen=True)
class Output:
    """The exported file: its sample rate, format, level and polarity."""

    sample_rate: int
    format: str
    mode: str
    db: float
    polarity: int


@dataclass(frozen=True)
class Patch:
    """One hit: the cymbal, which is its seed, the strike, which is its take, its sources and its sections."""

    seed: int
    take: int
    noise: tuple[Noise, ...]
    tone: tuple[Tone, ...]
    storage: Storage
    damper: Damper
    drive: Drive
    output: Output

    def __post_init__(self) -> None:
        """Validate the patch, so that every Patch lies inside the valid domain."""
        problems = _Problems()
        _validate(self, problems)
        if problems:
            raise PatchError(problems)

    @property
    def sources(self) -> tuple[tuple[str, Noise | Tone], ...]:
        """Every source with its path prefix, noise first, in patch order."""
        noise = tuple((f"noise[{j}]", s) for j, s in enumerate(self.noise, 1))
        return noise + tuple((f"tone[{j}]", s) for j, s in enumerate(self.tone, 1))

    @property
    def length_ms(self) -> float:
        """T_src: the latest end among the sub-bands of the audible sources, never after the damper (plan §3).

        Silent sources never lengthen the sound. With nothing audible the length is 0. Storage plays back a held
        period and its filter's delay longer than the sources.
        """
        ends = [max(sub_band_ends(s)) for _, s in self.sources if s.level_db > -math.inf]
        end = max(ends, default=0.0)
        if ends and self.storage.bits:
            end += 1000 * tail_s(self.storage)
        if self.damper.amp:
            end = min(end, self.damper.amp[-1][0])
        return end


def peak_ms(amp: Knots) -> float:
    """Return the time of a shape's first knot at gain 1."""
    return next(ms for ms, g in amp if g == 1.0)


def sub_band_ends(source: Noise | Tone) -> list[float]:
    """Where each of a source's sub-bands ends, in ms: t₀ + r_b(t_pk − t₀) + (t_end − t_pk)/s_b."""
    lay = layout(source.f_ref, source.damping, source.bloom)
    t0, tpk, tend = source.amp[0][0], peak_ms(source.amp), source.amp[-1][0]
    r, s = lay.stretch(source.bloom), lay.compression(source.damping)
    return [float(t0 + rb * (tpk - t0) + (tend - tpk) / sb) for rb, sb in zip(r, s, strict=True)]


# Validation. Each check reports every problem it finds, with its path.


class _Problems(list[str]):
    """The problems found so far, each with its field path, and the checks that find them."""

    def add(self, path: str, message: str) -> None:
        """Record a problem at a path."""
        self.append(f"{path}: {message}")

    def number(
        self,
        path: str,
        v: object,
        lo: float | None = None,
        hi: float | None = None,
        *,
        silence: bool = False,
    ) -> float | None:
        """Return v as a finite number within [lo, hi], or -inf where silence is allowed; None if invalid."""
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.add(path, f"expected a number, found {_kind(v)}")
            return None
        try:
            x = float(v)
        except OverflowError:
            self.add(path, "is too large")
            return None
        if silence and x == -math.inf:
            return x
        if not math.isfinite(x):
            self.add(path, "must be finite" + (" (or -inf for silence)" if silence else ""))
            return None
        if lo is not None and hi is not None and not lo <= x <= hi:
            self.add(path, f"{x:g} is outside {lo:g} … {hi:g}" + (" (or -inf for silence)" if silence else ""))
            return None
        return x

    def integer(self, path: str, v: object, allowed: tuple[int, ...]) -> None:
        """Check that v is one of the allowed integers."""
        if isinstance(v, bool) or not isinstance(v, int):
            self.add(path, f"expected an integer, found {_kind(v)}")
        elif v not in allowed:
            self.add(path, f"{v} is not one of {', '.join(map(str, allowed))}")

    def choice(self, path: str, v: object, allowed: tuple[str, ...]) -> str | None:
        """Return v if it is one of the allowed strings; None if not."""
        if not isinstance(v, str) or v not in allowed:
            self.add(path, f"expected one of {', '.join(repr(a) for a in allowed)}, found {v!r}")
            return None
        return v

    def knots(self, path: str, knots: object, key: str, count: int, max_ms: float) -> list[tuple[float, float]] | None:
        """Return the knot pairs, if their times are valid; the values are only checked to be numbers."""
        if not isinstance(knots, tuple) or not all(isinstance(k, tuple) and len(k) == PAIR for k in knots):
            self.add(path, f"expected (ms, {key}) knots")
            return None
        ok = count <= len(knots) <= MAX_KNOTS
        if not ok:
            self.add(path, f"needs {count} to {MAX_KNOTS} knots, found {len(knots)}")
        times = [self.number(f"{path}[{i}].ms", ms, 0.0, max_ms) for i, (ms, _) in enumerate(knots, 1)]
        values = [self.number(f"{path}[{i}].{key}", v) for i, (_, v) in enumerate(knots, 1)]
        if not ok or None in times or None in values:
            return None
        times_ = cast("list[float]", times)
        for i in range(1, len(times)):
            a, b = Decimal(repr(times_[i - 1])), Decimal(repr(times_[i]))
            where = f"{path}[{i + 1}].ms"
            if b == a:
                self.add(where, "repeats the previous knot time")
            elif b < a:
                self.add(where, "is earlier than the previous knot")
            elif b - a < MIN_SPACING_MS:
                self.add(where, f"is {b - a} ms after the previous knot; knots need 0.01 ms (10 µs)")
        return list(zip(times_, cast("list[float]", values), strict=True))

    def shape(self, path: str, amp: object) -> list[tuple[float, float]] | None:
        """Return an amplitude shape's knots if valid: gains in [0, 1], 0 at both ends, highest point exactly 1."""
        knots = self.knots(path, amp, "gain", 3, MAX_MS)
        if knots is None:
            return None
        gains = [g for _, g in knots]
        bad = False
        for i, g in enumerate(gains, 1):
            if not 0.0 <= g <= 1.0:
                self.add(f"{path}[{i}].gain", f"{g:g} is outside 0 … 1")
                bad = True
        if gains[0] != 0.0:
            self.add(f"{path}[1].gain", "an amplitude shape must start at 0")
        if gains[-1] != 0.0:
            self.add(f"{path}[{len(gains)}].gain", "an amplitude shape must end at 0")
        if not bad and max(gains) != 1.0:
            self.add(path, f"the highest gain must be exactly 1, found {max(gains)!r}")
            return None
        return None if bad or gains[0] != 0.0 or gains[-1] != 0.0 else knots

    def band(self, path: str, band: object) -> tuple[float, float] | None:
        """Return a band's edges if valid: two frequencies, the low one below the high one."""
        if not isinstance(band, tuple) or len(band) != PAIR:
            self.add(path, "expected [low, high] in Hz")
            return None
        lo = self.number(f"{path}[1]", band[0], 20.0, TOP_HZ)
        hi = self.number(f"{path}[2]", band[1], 20.0, TOP_HZ)
        if lo is not None and hi is not None and not lo < hi:
            self.add(path, f"the low edge {lo:g} Hz must be below the high edge {hi:g} Hz")
            return None
        return None if lo is None or hi is None else (lo, hi)


_KINDS = {
    bool: "a boolean",
    int: "an integer",
    float: "a float",
    str: "a string",
    dt.datetime: "a date-time",
    dt.date: "a date",
    dt.time: "a time",
    list: "an array",
    tuple: "an array",
    dict: "a table",
}


def _kind(v: object) -> str:
    """Name the kind of a value as TOML would."""
    return _KINDS.get(type(v), type(v).__name__)


def _validate(p: Patch, problems: _Problems) -> None:
    """Check every field of a patch."""
    _validate_seed(p.seed, problems)
    _validate_take(p.take, problems)
    ok = True
    for name, kind, top in (("noise", Noise, MAX_NOISE), ("tone", Tone, MAX_TONES)):
        items = getattr(p, name)
        if not isinstance(items, tuple) or not all(isinstance(s, kind) for s in items):
            problems.add(name, f"expected {name} sources")
            ok = False
        elif len(items) > top:
            problems.add(name, f"at most {top} {name} sources, found {len(items)}")
    if ok:
        if not p.noise and not p.tone:
            problems.add("noise", "a patch needs at least one source, [[noise]] or [[tone]]")
        for j, s in enumerate(p.noise, 1):
            _validate_noise(s, f"noise[{j}]", problems)
        for j, t in enumerate(p.tone, 1):
            _validate_tone(t, f"tone[{j}]", problems)
    _validate_storage(p.storage, problems)
    _validate_damper(p.damper, problems)
    _validate_drive(p.drive, problems)
    _validate_output(p.output, problems)


def _validate_seed(seed: object, problems: _Problems) -> None:
    """Check the seed: an integer from 0 to 2**63 - 1."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.add("seed", f"expected an integer, found {_kind(seed)}")
    elif not 0 <= seed <= MAX_SEED:
        problems.add("seed", f"{seed} is outside 0 … 2**63 - 1")


def _validate_take(take: object, problems: _Problems) -> None:
    """Check the take: an integer from 1 to 2**31."""
    if isinstance(take, bool) or not isinstance(take, int):
        problems.add("take", f"expected an integer, found {_kind(take)}")
    elif not 1 <= take <= MAX_TAKE:
        problems.add("take", f"{take} is outside 1 … 2**31")


def _validate_common(s: Noise | Tone, path: str, f_ref: float | None, problems: _Problems) -> None:
    """Check level, damping, bloom, chatter, rattle and shape, and the sub-bands they imply."""
    problems.number(f"{path}.level_db", s.level_db, -90.0, 24.0, silence=True)
    damping = problems.number(f"{path}.damping", s.damping, 0.0, 3.0)
    bloom = problems.number(f"{path}.bloom", s.bloom, 0.0, 3.0)
    problems.number(f"{path}.chatter", s.chatter, 0.0, 1.0)
    problems.number(f"{path}.chatter_hz", s.chatter_hz, 10.0, 4000.0)
    rate = problems.number(f"{path}.grains_hz", s.grains_hz, 0.0, MAX_GRAINS_HZ)
    if rate is not None and 0 < rate < MIN_GRAINS_HZ:
        problems.add(
            f"{path}.grains_hz",
            f"{rate:g} collisions per second leave less than one in {MAX_MS / 1000:g} s, the longest sound, which is "
            f"no rattle; at least {MIN_GRAINS_HZ:g}, or 0 for none",
        )
        rate = None
    ring = problems.number(f"{path}.grain_ms", s.grain_ms, MIN_GRAIN_MS, MAX_GRAIN_MS)
    if rate is not None and ring is not None and rate * ring / 1000 > MAX_OVERLAP:
        problems.add(
            f"{path}.grains_hz",
            f"{rate:g} collisions per second ringing {ring:g} ms each overlap {rate * ring / 1000:.4g} at once; "
            f"at most {MAX_OVERLAP:g}, past which a rattle sounds as the plain source",
        )
    amp = problems.shape(f"{path}.amp", s.amp)
    if f_ref is None or damping is None or bloom is None:
        return
    worst = worst_compression(f_ref, damping, bloom)
    if worst > MAX_COMPRESSION:
        problems.add(
            path,
            f"damping {damping:g} and bloom {bloom:g} compress or stretch the content at 20 kHz {worst:.4g} times "
            f"relative to {f_ref:g} Hz; at most {MAX_COMPRESSION:g}",
        )
        return
    if amp is not None:
        last = max(sub_band_ends(s))
        if last > MAX_MS:
            problems.add(f"{path}.bloom", f"its slowest sub-band would end at {last:.6g} ms, after {MAX_MS:g} ms")


def _validate_noise(s: Noise, path: str, problems: _Problems) -> None:
    """Check a noise source."""
    band = problems.band(f"{path}.band_hz", s.band_hz)
    spacing = problems.number(f"{path}.spacing_hz", s.spacing_hz, 0.0, MAX_SPACING)
    if spacing is not None and 0 < spacing < 1:
        problems.add(f"{path}.spacing_hz", f"{spacing:g} Hz: modes need 1 … {MAX_SPACING:g} Hz, or 0 for noise")
    _validate_common(s, path, None if band is None else band[0], problems)


def _validate_tone(s: Tone, path: str, problems: _Problems) -> None:
    """Check a tone."""
    problems.choice(f"{path}.wave", s.wave, WAVES)
    problems.choice(f"{path}.phase", s.phase, PHASES)
    band = problems.band(f"{path}.band_hz", s.band_hz)
    lowest = _validate_partials(s.partials, f"{path}.partials", problems)
    f_ref = None if band is None or lowest is None else max(band[0], lowest)
    _validate_common(s, path, f_ref, problems)


def _validate_partials(partials: object, path: str, problems: _Problems) -> float | None:
    """Return the lowest partial if the partials are valid, else None."""
    if not isinstance(partials, tuple) or not all(isinstance(q, tuple) and len(q) == PAIR for q in partials):
        problems.add(path, "expected (hz, level_db) partials")
        return None
    if not 1 <= len(partials) <= MAX_PARTIALS:
        problems.add(path, f"needs 1 to {MAX_PARTIALS} partials, found {len(partials)}")
        return None
    hz = [problems.number(f"{path}[{i}].hz", f, 20.0, TOP_HZ) for i, (f, _) in enumerate(partials, 1)]
    levels = [problems.number(f"{path}[{i}].level_db", v, -60.0, 0.0) for i, (_, v) in enumerate(partials, 1)]
    if None in hz:
        return None
    freqs = cast("list[float]", hz)
    for i in range(1, len(freqs)):
        if not grid_bin(freqs[i]) > grid_bin(freqs[i - 1]):
            problems.add(f"{path}[{i + 1}].hz", "partials must increase by at least 1/8 Hz, the grid's step")
    if None not in levels and (loudest := max(cast("list[float]", levels))) != 0.0:
        problems.add(path, f"the loudest partial must be exactly 0 dB, found {loudest!r}")
    return freqs[0]


def _validate_damper(d: object, problems: _Problems) -> None:
    """Check the damper: no knots, or from exactly 1 down to exactly 0, never increasing."""
    if not isinstance(d, Damper):
        problems.add("damper", "expected a damper section")
        return
    if d.amp == ():
        return
    knots = problems.knots("damper.amp", d.amp, "gain", 2, MAX_MS)
    if knots is None:
        return
    gains = [g for _, g in knots]
    if gains[0] != 1.0:
        problems.add("damper.amp[1].gain", "a damper must start at exactly 1")
    if gains[-1] != 0.0:
        problems.add(f"damper.amp[{len(gains)}].gain", "a damper must end at exactly 0")
    for i in range(1, len(gains)):
        if gains[i] > gains[i - 1]:
            problems.add(f"damper.amp[{i + 1}].gain", "a damper never increases")


def _validate_storage(s: object, problems: _Problems) -> None:
    """Check the storage: off, or a bit depth with its rate and its playback filter."""
    if not isinstance(s, Storage):
        problems.add("storage", "expected a storage section")
        return
    problems.integer("storage.bits", s.bits, STORAGE_BITS)
    problems.number("storage.rate_hz", s.rate_hz, *STORAGE_RATES_HZ)
    problems.number("storage.filter_hz", s.filter_hz, *STORAGE_FILTERS_HZ)


def _validate_drive(d: object, problems: _Problems) -> None:
    """Check the drive."""
    if not isinstance(d, Drive):
        problems.add("drive", "expected a drive section")
    else:
        problems.number("drive.db", d.db, 0.0, 48.0)


def _validate_output(o: object, problems: _Problems) -> None:
    """Check the output section."""
    if not isinstance(o, Output):
        problems.add("output", "expected an output section")
        return
    problems.integer("output.sample_rate", o.sample_rate, SAMPLE_RATES)
    problems.choice("output.format", o.format, FORMATS)
    mode = problems.choice("output.mode", o.mode, MODES)
    problems.number("output.db", o.db, -60.0, 24.0 if mode == "gain" else 0.0)
    problems.integer("output.polarity", o.polarity, (1, -1))


# Parsing.

_COMMON = ("band_hz", "damping", "bloom", "chatter", "chatter_hz", "grains_hz", "grain_ms", "amp")
_FIELDS = {
    "": ("instrument", "schema", "seed", "take", "noise", "tone", "storage", "damper", "drive", "output"),
    "noise": ("level_db", "band_hz", "spacing_hz", *_COMMON[1:]),
    "tone": ("level_db", "wave", "phase", "partials", *_COMMON),
    "storage": ("bits", "rate_hz", "filter_hz"),
    "damper": ("amp",),
    "drive": ("db",),
    "output": ("sample_rate", "format", "mode", "db", "polarity"),
}
_OPTIONAL = ("noise", "tone")


def _keys(
    problems: _Problems,
    prefix: str,
    table: dict[str, Any],
    expected: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> bool:
    """Report unknown keys, and missing ones of those expected; True when none is missing."""
    for key in table:
        if key not in expected:
            problems.add(f"{prefix}{key}", "unknown field")
    missing = [key for key in expected if key not in table and key not in optional]
    for key in missing:
        problems.add(f"{prefix}{key}", "missing")
    return not missing


def _float[T](v: T) -> T | float:
    """Return a TOML integer as a float, which floats accept; anything else is left for validation to name."""
    if isinstance(v, int) and not isinstance(v, bool):
        try:
            return float(v)
        except OverflowError:
            return v
    return v


def _pairs(problems: _Problems, path: str, items: object, keys: tuple[str, str]) -> tuple[tuple[Any, ...], ...] | None:
    """Return an array of inline tables with exactly these keys as tuples, their numbers as floats."""
    if not isinstance(items, list):
        problems.add(path, f"expected an array of {{ {', '.join(keys)} }} tables, found {_kind(items)}")
        return None
    out = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.add(f"{path}[{i}]", f"expected a {{ {', '.join(keys)} }} table, found {_kind(item)}")
            return None
        if not _keys(problems, f"{path}[{i}].", item, keys):
            return None
        out.append(tuple(_float(item[k]) for k in keys))
    return tuple(out)


def _section(problems: _Problems, path: str, table: object, fields: tuple[str, ...]) -> dict[str, Any] | None:
    """Return a section's table if it has exactly these fields; None after reporting what is wrong."""
    if not isinstance(table, dict):
        problems.add(path, f"expected a table, found {_kind(table)}")
        return None
    return table if _keys(problems, f"{path}.", table, fields) else None


def _band[T](v: T) -> T | tuple[Any, ...]:
    """Return a TOML array as a tuple, its integers as floats; anything else is left for validation to name."""
    return tuple(map(_float, v)) if isinstance(v, list) else v


def _noise(problems: _Problems, path: str, table: object) -> Noise | None:
    """Build a noise source from its table; None after reporting what is missing."""
    t = _section(problems, path, table, _FIELDS["noise"])
    if t is None:
        return None
    amp = _pairs(problems, f"{path}.amp", t["amp"], ("ms", "gain"))
    if amp is None:
        return None
    return Noise(
        _float(t["level_db"]),
        _band(t["band_hz"]),
        _float(t["spacing_hz"]),
        _float(t["damping"]),
        _float(t["bloom"]),
        _float(t["chatter"]),
        _float(t["chatter_hz"]),
        _float(t["grains_hz"]),
        _float(t["grain_ms"]),
        amp,
    )


def _tone(problems: _Problems, path: str, table: object) -> Tone | None:
    """Build a tone from its table; None after reporting what is missing."""
    t = _section(problems, path, table, _FIELDS["tone"])
    if t is None:
        return None
    partials = _pairs(problems, f"{path}.partials", t["partials"], ("hz", "level_db"))
    amp = _pairs(problems, f"{path}.amp", t["amp"], ("ms", "gain"))
    if partials is None or amp is None:
        return None
    return Tone(
        _float(t["level_db"]),
        t["wave"],
        t["phase"],
        partials,
        _band(t["band_hz"]),
        _float(t["damping"]),
        _float(t["bloom"]),
        _float(t["chatter"]),
        _float(t["chatter_hz"]),
        _float(t["grains_hz"]),
        _float(t["grain_ms"]),
        amp,
    )


def _header(data: dict[str, Any]) -> None:
    """Check the instrument and the schema, which come first: another instrument or schema may share nothing else."""
    instrument = data.get("instrument")
    if instrument is None:
        raise PatchError([f'instrument: missing; a cymbal patch starts with instrument = "{INSTRUMENT}"'])
    if instrument != INSTRUMENT:
        raise PatchError([f'instrument: this is a {instrument!r} patch, not a "{INSTRUMENT}" one'])
    schema = data.get("schema")
    if schema is None:
        raise PatchError(["schema: missing"])
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise PatchError([f"schema: expected an integer, found {_kind(schema)}"])
    if schema != SCHEMA:
        raise PatchError([f"schema: unsupported version {schema}; this renderer reads schema {SCHEMA}"])


def _sources[S: (Noise, Tone)](
    problems: _Problems,
    data: dict[str, Any],
    name: str,
    build: Callable[[_Problems, str, object], S | None],
) -> tuple[S, ...] | None:
    """Build the sources of one kind, [[noise]] or [[tone]]; None after reporting what is wrong."""
    tables = data.get(name, [])
    if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
        problems.add(name, f"expected an array of tables ([[{name}]]), found {_kind(tables)}")
        return None
    built = [build(problems, f"{name}[{j}]", t) for j, t in enumerate(tables, 1)]
    return None if any(b is None for b in built) else tuple(b for b in built if b is not None)


def _storage(problems: _Problems, data: dict[str, Any]) -> Storage | None:
    """Build the storage section; None after reporting what is wrong."""
    if "storage" not in data or (t := _section(problems, "storage", data["storage"], _FIELDS["storage"])) is None:
        return None
    return Storage(t["bits"], _float(t["rate_hz"]), _float(t["filter_hz"]))


def _damper(problems: _Problems, data: dict[str, Any]) -> Damper | None:
    """Build the damper section; None after reporting what is wrong."""
    if "damper" not in data or (t := _section(problems, "damper", data["damper"], _FIELDS["damper"])) is None:
        return None
    amp = _pairs(problems, "damper.amp", t["amp"], ("ms", "gain"))
    return None if amp is None else Damper(amp)


def _drive(problems: _Problems, data: dict[str, Any]) -> Drive | None:
    """Build the drive section; None after reporting what is wrong."""
    if "drive" not in data or (t := _section(problems, "drive", data["drive"], _FIELDS["drive"])) is None:
        return None
    return Drive(_float(t["db"]))


def _output(problems: _Problems, data: dict[str, Any]) -> Output | None:
    """Build the output section; None after reporting what is wrong."""
    if "output" not in data or (t := _section(problems, "output", data["output"], _FIELDS["output"])) is None:
        return None
    return Output(t["sample_rate"], t["format"], t["mode"], _float(t["db"]), t["polarity"])


def from_dict(data: object) -> Patch:
    """Build a patch from parsed TOML, reporting every problem at once."""
    if not isinstance(data, dict):
        raise PatchError(["expected a table"])
    _header(data)
    problems = _Problems()
    _keys(problems, "", data, _FIELDS[""], _OPTIONAL)
    noise = _sources(problems, data, "noise", _noise)
    tone = _sources(problems, data, "tone", _tone)
    storage, damper = _storage(problems, data), _damper(problems, data)
    drive, output = _drive(problems, data), _output(problems, data)
    if noise is None or tone is None or storage is None or damper is None or drive is None or output is None:
        raise PatchError(problems or ["invalid patch"])
    if "seed" not in data or "take" not in data:
        raise PatchError(problems)
    # Every part is there, so every value is checked as well, and all the problems are reported together.
    try:
        p = Patch(data["seed"], data["take"], noise, tone, storage, damper, drive, output)
    except PatchError as e:
        raise PatchError(problems + list(e.problems)) from None
    if problems:
        raise PatchError(problems)
    return p


def loads(text: str) -> Patch:
    """Read a patch from TOML text."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    except RecursionError:  # arrays or tables nested deeper than any patch, made to exhaust the parser
        raise PatchError(["TOML: nested too deeply to be a patch"]) from None
    return from_dict(data)


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a patch file's text: UTF-8 with or without a BOM, or UTF-16 with one, as Windows PowerShell's `>` writes."""
    raw = Path(path).read_bytes()
    return raw.decode("utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig")


def load(path: str | os.PathLike[str]) -> Patch:
    """Read a patch from a TOML file, in UTF-8 or UTF-16."""
    try:
        text = read_text(path)
    except UnicodeDecodeError:
        raise PatchError(["not UTF-8 or UTF-16 text"]) from None
    return loads(text)


def _knots_dict(knots: Knots) -> list[dict[str, float]]:
    """Return knots as TOML parses them: an array of { ms, gain } tables."""
    return [{"ms": m, "gain": g} for m, g in knots]


def to_dict(p: Patch) -> dict[str, Any]:
    """Return the plain data that TOML parses to, so that from_dict(to_dict(p)) == p."""
    out: dict[str, Any] = {"instrument": INSTRUMENT, "schema": SCHEMA, "seed": p.seed, "take": p.take}
    if p.noise:
        out["noise"] = [
            {
                "level_db": s.level_db,
                "band_hz": list(s.band_hz),
                "spacing_hz": s.spacing_hz,
                "damping": s.damping,
                "bloom": s.bloom,
                "chatter": s.chatter,
                "chatter_hz": s.chatter_hz,
                "grains_hz": s.grains_hz,
                "grain_ms": s.grain_ms,
                "amp": _knots_dict(s.amp),
            }
            for s in p.noise
        ]
    if p.tone:
        out["tone"] = [
            {
                "level_db": s.level_db,
                "wave": s.wave,
                "phase": s.phase,
                "partials": [{"hz": f, "level_db": v} for f, v in s.partials],
                "band_hz": list(s.band_hz),
                "damping": s.damping,
                "bloom": s.bloom,
                "chatter": s.chatter,
                "chatter_hz": s.chatter_hz,
                "grains_hz": s.grains_hz,
                "grain_ms": s.grain_ms,
                "amp": _knots_dict(s.amp),
            }
            for s in p.tone
        ]
    st = p.storage
    out["storage"] = {"bits": st.bits, "rate_hz": st.rate_hz, "filter_hz": st.filter_hz}
    out["damper"] = {"amp": _knots_dict(p.damper.amp)}
    out["drive"] = {"db": p.drive.db}
    o = p.output
    out["output"] = {
        "sample_rate": o.sample_rate,
        "format": o.format,
        "mode": o.mode,
        "db": o.db,
        "polarity": o.polarity,
    }
    return out


# Canonical TOML.

_COMMENT_COLUMN = 28


def dumps(p: Patch) -> str:
    """Write the complete patch as TOML with the canonical comments; loads(dumps(p)) == p exactly."""
    o = p.output
    lines = [
        f'instrument = "{INSTRUMENT}"',
        f"schema = {SCHEMA}",
        _note(f"seed = {p.seed}", "the cymbal: where its modes lie"),
        _note(f"take = {p.take}", "the strike: round robins use take, take + 1, …"),
    ]
    for j, s in enumerate(p.noise, 1):
        lines += ["", _note("[[noise]]", f"noise[{j}]")]
        lines += _rows(
            [
                ("level_db", _f(s.level_db), ""),
                ("band_hz", _band_text(s.band_hz), ""),
                ("spacing_hz", _f(s.spacing_hz), "mean distance between modes; 0 = continuous noise"),
                *_common_rows(s),
            ],
        )
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in s.amp], "shape: 0 at both ends, peak exactly 1")
    for j, t in enumerate(p.tone, 1):
        lines += ["", _note("[[tone]]", f"tone[{j}]")]
        lines += _rows(
            [
                ("level_db", _f(t.level_db), ""),
                ("wave", f'"{t.wave}"', '"sine" | "square"'),
                ("phase", f'"{t.phase}"', '"strike" | "free"'),
            ],
        )
        lines += _array(
            "partials",
            [(("hz", f), ("level_db", v)) for f, v in t.partials],
            "loudest exactly 0 dB; at the nearest 1/8 Hz",
        )
        lines += _rows([("band_hz", _band_text(t.band_hz), ""), *_common_rows(t)])
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in t.amp], "shape: 0 at both ends, peak exactly 1")
    st = p.storage
    lines += ["", _note("[storage]", "a drum machine's sample memory: the sources stored, then played back")]
    lines += _rows(
        [
            ("bits", str(st.bits), "0 = off; else 2 … 16, quantized against the sound's peak"),
            ("rate_hz", _f(st.rate_hz), "stored at this rate, as near as the internal rate allows"),
            ("filter_hz", _f(st.filter_hz), "played back through a low-pass, −3 dB here"),
        ],
    )
    lines += ["", _note("[damper]", "multiplies every source: 1 until damped, then 0")]
    if p.damper.amp:
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in p.damper.amp], "from 1 down to 0")
    else:
        lines += [_note("amp = []", "no knots: never damped")]
    lines += ["", "[drive]", *_rows([("db", _f(p.drive.db), "small-signal gain; 0 = bypass")])]
    lines += ["", "[output]"]
    lines += _rows(
        [
            ("sample_rate", str(o.sample_rate), ""),
            ("format", f'"{o.format}"', '"float32" | "pcm24" | "pcm16"'),
            ("mode", f'"{o.mode}"', '"gain" | "peak" | "true-peak"'),
            ("db", _f(o.db), "gain in dB, or peak target in dBFS"),
            ("polarity", str(o.polarity), ""),
        ],
    )
    return "\n".join(lines) + "\n"


def _common_rows(s: Noise | Tone) -> list[Row]:
    """Return the rows every source has: damping, bloom, chatter and rattle."""
    return [
        ("damping", _f(s.damping), "content at f decays (f / f_ref)^damping times faster"),
        ("bloom", _f(s.bloom), "and rises (f / f_ref)^bloom times slower"),
        ("chatter", _f(s.chatter), "0 … 1: gated at chatter_hz, as plates that slap"),
        ("chatter_hz", _f(s.chatter_hz), ""),
        ("grains_hz", _f(s.grains_hz), "collisions per second at the shape's peak, a rattle; 0 = none"),
        ("grain_ms", _f(s.grain_ms), "each collision rings down 60 dB in this time"),
    ]


def _f(x: float) -> str:
    """Write a number as a TOML float, exactly."""
    return repr(float(x))


def _band_text(band: tuple[float, float]) -> str:
    """Write a band as a TOML array."""
    return f"[{_f(band[0])}, {_f(band[1])}]"


def _note(line: str, note: str) -> str:
    """Add a comment to a line, in the comment column."""
    return f"{(line + ' ').ljust(_COMMENT_COLUMN)}# {note}" if note else line


def _rows(rows: list[Row]) -> list[str]:
    """Write key = value lines, their equals signs aligned, each with its comment."""
    width = max(len(k) for k, _, _ in rows)
    return [_note(f"{k.ljust(width)} = {v}", note) for k, v, note in rows]


def _array(name: str, items: list[tuple[tuple[str, float], tuple[str, float]]], note: str) -> list[str]:
    """Write an array of two-key inline tables, with the first column aligned."""
    firsts = [f"{items[0][0][0]} = {_f(a[1])}," for a, _ in items] if items else []
    width = max(map(len, firsts), default=0)
    body = [
        f"  {{ {first.ljust(width)} {key} = {_f(v)} }}," for first, (_, (key, v)) in zip(firsts, items, strict=True)
    ]
    return [_note(f"{name} = [", note), *body, "]"]


# Pitch vocabulary shared by `brightwork pitch`, the report and the WAV root note.

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_pitch(hz: float) -> float:
    """Return the fractional MIDI note number of a frequency (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(midi: int) -> str:
    """Return the name of MIDI note `midi`, such as 'F#3'."""
    return f"{_NOTES[midi % 12]}{midi // 12 - 1}"


def describe_pitch(hz: float) -> str:
    """Describe a frequency by its nearest note and its offset in cents, such as 'F#3 -3.2 cents'."""
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_note(text: str) -> float | None:
    """Return a note name such as D5, C#4 or Eb3 in Hz; None if the text is not a note name."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if not m:
        return None
    midi = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1)
    return 440.0 * float(2.0 ** ((midi - 69) / 12))


# Editing commands. Each returns a new, validated patch, or raises PatchError.

_US = Decimal("0.001")  # knot times are rewritten to 1 µs
_MHZ = Decimal("0.001")  # frequencies are rewritten to 1 mHz


def _dec(x: float) -> Decimal:
    """Return the decimal value as written."""
    return Decimal(repr(float(x)))


def _ms(x: Decimal) -> float:
    """Round a time to 1 µs."""
    return float(x.quantize(_US))


def _hz(x: Decimal) -> float:
    """Round a frequency to 1 mHz."""
    return float(x.quantize(_MHZ))


def _significant(x: float, digits: int = 4) -> float:
    """Round to `digits` significant digits."""
    return float(f"{x:.{digits}g}")


def pitch_ratio(p: Patch, text: str) -> float:
    """Return the ratio a `pitch` argument names: ±semitones, or a note for tone[1]'s lowest partial."""
    hz = parse_note(text)
    if hz is not None:
        if not p.tone:
            raise PatchError([f"pitch {text}: a note sets the lowest partial of tone[1], and this patch has no tone"])
        return hz / p.tone[0].partials[0][0]
    try:
        semitones = float(text)
    except ValueError:
        raise PatchError(
            [f"pitch: expected semitones such as +2 or -3.5, or a note such as D5, found {text!r}"],
        ) from None
    if not math.isfinite(semitones):
        raise PatchError([f"pitch: expected a finite number of semitones, found {text!r}"])
    return float(2.0 ** (semitones / 12.0))


def pitch(p: Patch, text: str) -> Patch:
    """Multiply every frequency by one ratio: band edges, spacings, partials and chatter rates (plan §4).

    Each mode keeps its place in its cell and its excitation, so the same cymbal
    is resized; times are unchanged. Frequencies are written to 1 mHz. The
    result is validated, so the command fails rather than clamping.
    """
    ratio = _dec(pitch_ratio(p, text))

    def scale(x: float) -> float:
        return _hz(_dec(x) * ratio)

    noise = tuple(
        dataclasses.replace(
            s,
            band_hz=(scale(s.band_hz[0]), scale(s.band_hz[1])),
            spacing_hz=scale(s.spacing_hz),
            chatter_hz=scale(s.chatter_hz),
        )
        for s in p.noise
    )
    tone = tuple(
        dataclasses.replace(
            s,
            partials=tuple((scale(f), v) for f, v in s.partials),
            band_hz=(scale(s.band_hz[0]), scale(s.band_hz[1])),
            chatter_hz=scale(s.chatter_hz),
        )
        for s in p.tone
    )
    return dataclasses.replace(p, noise=noise, tone=tone)


def curve_names(p: Patch) -> tuple[str, ...]:
    """Name the patch's curves: every source's shape, then the damper."""
    names = [f"{prefix}.amp" for prefix, _ in p.sources]
    return (*names, "damper.amp")


def _get_curve(p: Patch, name: str) -> Knots:
    """Return a curve's knots, by one of its curve names."""
    if name == "damper.amp":
        return p.damper.amp
    return dict(p.sources)[name.removesuffix(".amp")].amp


def _set_curves(p: Patch, curves: dict[str, Knots]) -> Patch:
    """Return the patch with these curves replaced, by name."""
    noise = tuple(
        dataclasses.replace(s, amp=curves[f"noise[{j}].amp"]) if f"noise[{j}].amp" in curves else s
        for j, s in enumerate(p.noise, 1)
    )
    tone = tuple(
        dataclasses.replace(s, amp=curves[f"tone[{j}].amp"]) if f"tone[{j}].amp" in curves else s
        for j, s in enumerate(p.tone, 1)
    )
    damper = Damper(curves["damper.amp"]) if "damper.amp" in curves else p.damper
    return dataclasses.replace(p, noise=noise, tone=tone, damper=damper)


def stretch(p: Patch, curves: str | Sequence[str], factor: float, after_ms: float | None = None) -> Patch:
    """Rescale knot times after each shape's peak, or after `after_ms`, by `factor`.

    `curves` names curves by path (`noise[1].amp`, `tone[1].amp`, `damper.amp`);
    `all` names every shape and the damper. The damper's times scale from 0 ms
    unless `after_ms` is given. Times are rewritten to 1 µs.
    """
    if not (math.isfinite(factor) and factor > 0):
        raise PatchError([f"stretch: the factor must be a positive number, found {factor!r}"])
    if after_ms is not None and not (math.isfinite(after_ms) and after_ms >= 0):
        raise PatchError([f"--after: must be a time in ms from 0 on, found {after_ms!r}"])
    names: list[str] = []
    for name in [curves] if isinstance(curves, str) else curves:
        if name == "all":
            names += [n for n in curve_names(p) if n != "damper.amp" or p.damper.amp]
        elif name in curve_names(p):
            names.append(name)
        else:
            raise PatchError([f"{name}: not a curve of this patch; expected all or one of {', '.join(curve_names(p))}"])
    out = {}
    for name in dict.fromkeys(names):
        knots = _get_curve(p, name)
        if not knots:
            continue
        default = 0.0 if name == "damper.amp" else peak_ms(knots)
        anchor = default if after_ms is None else after_ms
        a, f = _dec(anchor), Decimal(repr(float(factor)))
        out[name] = tuple((ms if ms <= anchor else _ms(a + f * (_dec(ms) - a)), v) for ms, v in knots)
    return _set_curves(p, out)


CHOKE_DB = 60.0  # a choke's release falls this far, in knots every 10 dB
DEFAULT_RELEASE_MS = 40.0


def choke_knots(ms: float, release: float = DEFAULT_RELEASE_MS) -> Knots:
    """Return a choke's damper: 1 at `ms`, knots every 10 dB evenly over `release` down to −60 dB, then 0 a step on.

    Times are written to 1 µs and gains to 4 significant digits. Audition fades a choked hit the same way.
    """
    steps = int(CHOKE_DB // 10)
    if not (math.isfinite(ms) and ms >= 0):
        raise PatchError([f"choke: the time must be in ms from 0 on, found {ms!r}"])
    if not (math.isfinite(release) and release / steps >= float(MIN_SPACING_MS)):
        raise PatchError([f"--release: needs at least {steps * float(MIN_SPACING_MS):g} ms, found {release!r}"])
    start, step = _dec(ms), _dec(release) / steps
    knots = [(_ms(start), 1.0)]
    knots += [(_ms(start + step * k), _significant(10 ** (-k / 2))) for k in range(1, steps + 1)]
    knots.append((_ms(start + step * (steps + 1)), 0.0))
    return tuple(knots)


def choke(p: Patch, ms: float, release: float = DEFAULT_RELEASE_MS) -> Patch:
    """Replace the damper with a choke at `ms` that falls 60 dB over `release` (plan §4)."""
    return dataclasses.replace(p, damper=Damper(choke_knots(ms, release)))


def rises(values: Sequence[float]) -> list[tuple[int, int]]:
    """Return the (first, last) knot indices of each rise: a maximal run of strictly increasing knots.

    Both ends of a rise have zero slope, the first because nothing before it is
    lower and the last because nothing after it is higher, so a rise can be
    lengthened without changing the rest of the curve.
    """
    out: list[tuple[int, int]] = []
    i, n = 0, len(values)
    while i < n - 1:
        if values[i + 1] > values[i]:
            k = i
            while k < n - 1 and values[k + 1] > values[k]:
                k += 1
            out.append((i, k))
            i = k
        else:
            i += 1
    return out


def _lengthen_rises(knots: Knots, factor: Decimal) -> Knots:
    """Lengthen every rise by `factor` and shift later knots by the added time, all to 1 µs.

    The shift after each rise is the exact added time so far, rounded once, so
    the rounding never accumulates over several rises, and knots between rises
    move together, keeping their spacing exactly.
    """
    values = [v for _, v in knots]
    times = [_dec(ms) for ms, _ in knots]
    new, added, shift, done = list(times), Decimal(0), Decimal(0), 0
    for a, b in rises(values):
        for k in range(done, a + 1):
            new[k] = times[k] + shift
        added += (factor - 1) * (times[b] - times[a])
        shift = added.quantize(_US)
        new[b] = times[b] + shift
        for k in range(a + 1, b):  # knots inside a rise keep their place in it
            new[k] = (new[a] + (new[b] - new[a]) * (times[k] - times[a]) / (times[b] - times[a])).quantize(_US)
        done = b + 1
    for k in range(done, len(times)):
        new[k] = times[k] + shift
    return tuple((float(t), v) for t, (_, v) in zip(new, knots, strict=True))


def softer(p: Patch, db: float, contact: float = 0.2) -> Patch:
    """Return the same cymbal struck `db` softer, by two rules of plan §2.11.

    With v = 10^(−db/20):
      * Level: every source level falls by db (silent sources stay silent).
      * Contact: every rise lengthens by v^(−contact), later knots shifting by
        the added time, and every source's upper band edge scales by v^contact.
    Nothing else changes. An exponent of 0 turns the contact rule off. Values
    are written to 1 µs, band edges to 0.1 Hz and levels to 6 decimals of a dB.
    """
    for name, x in (("DB", db), ("--contact", contact)):
        if not math.isfinite(x):
            raise PatchError([f"{name}: must be a finite number, found {x!r}"])
    if db == 0:
        return p
    v = 10 ** (-db / 20)

    def level(x: float) -> float:
        return x if x == -math.inf else float(round(_dec(x) - Decimal(repr(float(db))), 6))

    def soften[S: (Noise, Tone)](s: S) -> S:
        amp, band = s.amp, s.band_hz
        if contact:
            amp = _lengthen_rises(amp, Decimal(repr(v**-contact)))
            band = (band[0], float(round(_dec(band[1]) * Decimal(repr(v**contact)), 1)))
        return dataclasses.replace(s, level_db=level(s.level_db), band_hz=band, amp=amp)

    return dataclasses.replace(p, noise=tuple(map(soften, p.noise)), tone=tuple(map(soften, p.tone)))
