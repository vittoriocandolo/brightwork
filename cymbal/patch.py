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

import dataclasses
import math
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal

from .spectrum import MAX_COMPRESSION, MAX_SEED, MAX_TAKE, TOP_HZ, grid_bin, layout, worst_compression

INSTRUMENT = "cymbal"
SCHEMA = 1
SAMPLE_RATES = (44100, 48000, 88200, 96000)
FORMATS = ("float32", "pcm24", "pcm16")
WAVES = ("sine", "square")
PHASES = ("strike", "free")
MAX_MS = 8000.0  # every carrier repeats after 8 s
MAX_NOISE = 4
MAX_TONES = 2
MAX_PARTIALS = 16
MAX_KNOTS = 32
MAX_SPACING = 200.0
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written


class PatchError(ValueError):
    """An invalid patch or edit. `problems` lists every reason, each with its field path."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Noise:
    level_db: float
    band_hz: tuple  # (low, high)
    spacing_hz: float  # 0: continuous noise
    damping: float
    bloom: float
    chatter: float
    chatter_hz: float
    amp: tuple  # ((ms, gain), ...)

    @property
    def f_ref(self) -> float:
        return self.band_hz[0]


@dataclass(frozen=True)
class Tone:
    level_db: float
    wave: str
    phase: str
    partials: tuple  # ((hz, level_db), ...)
    band_hz: tuple
    damping: float
    bloom: float
    chatter: float
    chatter_hz: float
    amp: tuple

    @property
    def f_ref(self) -> float:
        """The higher of the band's lower edge and the lowest partial (plan §2.6)."""
        return max(self.band_hz[0], self.partials[0][0])


@dataclass(frozen=True)
class Damper:
    amp: tuple  # ((ms, gain), ...): none, or from 1 down to 0


@dataclass(frozen=True)
class Drive:
    db: float


@dataclass(frozen=True)
class Output:
    sample_rate: int
    format: str
    mode: str
    db: float
    polarity: int


@dataclass(frozen=True)
class Patch:
    seed: int
    take: int
    noise: tuple  # (Noise, ...)
    tone: tuple  # (Tone, ...)
    damper: Damper
    drive: Drive
    output: Output

    def __post_init__(self):
        problems = _Problems()
        _validate(self, problems)
        if problems:
            raise PatchError(problems)

    @property
    def sources(self) -> tuple:
        """Every source with its path prefix, noise first, in patch order."""
        return tuple((f"noise[{j}]", s) for j, s in enumerate(self.noise, 1)) + tuple(
            (f"tone[{j}]", s) for j, s in enumerate(self.tone, 1)
        )

    @property
    def length_ms(self) -> float:
        """T_src: the latest end among the sub-bands of the audible sources, never after the damper (plan §3).

        Silent sources never lengthen the sound. With nothing audible the length is 0.
        """
        ends = [max(sub_band_ends(s)) for _, s in self.sources if s.level_db > -math.inf]
        end = max(ends, default=0.0)
        if self.damper.amp:
            end = min(end, self.damper.amp[-1][0])
        return end


def peak_ms(amp: tuple) -> float:
    return next(ms for ms, g in amp if g == 1.0)


def sub_band_ends(source) -> list[float]:
    """Where each of a source's sub-bands ends, in ms: t₀ + r_b(t_pk − t₀) + (t_end − t_pk)/s_b."""
    lay = layout(source.f_ref, source.damping, source.bloom)
    t0, tpk, tend = source.amp[0][0], peak_ms(source.amp), source.amp[-1][0]
    r, s = lay.stretch(source.bloom), lay.compression(source.damping)
    return [float(t0 + rb * (tpk - t0) + (tend - tpk) / sb) for rb, sb in zip(r, s)]


# Validation. Each check reports every problem it finds, with its path.


class _Problems(list):
    def add(self, path: str, message: str) -> None:
        self.append(f"{path}: {message}")

    def number(self, path, v, lo=None, hi=None, *, silence=False):
        """A finite number within [lo, hi], or -inf where silence is allowed; None if invalid."""
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
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            self.add(path, f"{x:g} is outside {lo:g} … {hi:g}" + (" (or -inf for silence)" if silence else ""))
            return None
        return x

    def integer(self, path, v, allowed):
        if isinstance(v, bool) or not isinstance(v, int):
            self.add(path, f"expected an integer, found {_kind(v)}")
        elif v not in allowed:
            self.add(path, f"{v} is not one of {', '.join(map(str, allowed))}")

    def choice(self, path, v, allowed):
        if not isinstance(v, str) or v not in allowed:
            self.add(path, f"expected one of {', '.join(repr(a) for a in allowed)}, found {v!r}")
            return None
        return v

    def knots(self, path, knots, key, count, max_ms):
        """Knot pairs with valid times; the values are only checked to be numbers."""
        if not isinstance(knots, tuple) or not all(isinstance(k, tuple) and len(k) == 2 for k in knots):
            self.add(path, f"expected (ms, {key}) knots")
            return None
        ok = count <= len(knots) <= MAX_KNOTS
        if not ok:
            self.add(path, f"needs {count} to {MAX_KNOTS} knots, found {len(knots)}")
        times = [self.number(f"{path}[{i}].ms", ms, 0.0, max_ms) for i, (ms, _) in enumerate(knots, 1)]
        values = [self.number(f"{path}[{i}].{key}", v) for i, (_, v) in enumerate(knots, 1)]
        if not ok or None in times or None in values:
            return None
        for i in range(1, len(times)):
            a, b = Decimal(repr(times[i - 1])), Decimal(repr(times[i]))
            where = f"{path}[{i + 1}].ms"
            if b == a:
                self.add(where, "repeats the previous knot time")
            elif b < a:
                self.add(where, "is earlier than the previous knot")
            elif b - a < MIN_SPACING_MS:
                self.add(where, f"is {b - a} ms after the previous knot; knots need 0.01 ms (10 µs)")
        return list(zip(times, values))

    def shape(self, path, knots):
        """An amplitude shape: gains in [0, 1], 0 at both ends, highest point exactly 1."""
        knots = self.knots(path, knots, "gain", 3, MAX_MS)
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

    def band(self, path, band):
        if not isinstance(band, tuple) or len(band) != 2:
            self.add(path, "expected [low, high] in Hz")
            return None
        lo = self.number(f"{path}[1]", band[0], 20.0, TOP_HZ)
        hi = self.number(f"{path}[2]", band[1], 20.0, TOP_HZ)
        if lo is not None and hi is not None and not lo < hi:
            self.add(path, f"the low edge {lo:g} Hz must be below the high edge {hi:g} Hz")
            return None
        return None if lo is None or hi is None else (lo, hi)


def _kind(v) -> str:
    return {bool: "a boolean", str: "a string", list: "an array", tuple: "an array", dict: "a table"}.get(
        type(v), type(v).__name__
    )


def _validate(p: Patch, problems: _Problems) -> None:
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
        for j, s in enumerate(p.tone, 1):
            _validate_tone(s, f"tone[{j}]", problems)
    _validate_damper(p.damper, problems)
    _validate_drive(p.drive, problems)
    _validate_output(p.output, problems)


def _validate_seed(seed, problems: _Problems) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.add("seed", f"expected an integer, found {_kind(seed)}")
    elif not 0 <= seed <= MAX_SEED:
        problems.add("seed", f"{seed} is outside 0 … 2**63 - 1")


def _validate_take(take, problems: _Problems) -> None:
    if isinstance(take, bool) or not isinstance(take, int):
        problems.add("take", f"expected an integer, found {_kind(take)}")
    elif not 1 <= take <= MAX_TAKE:
        problems.add("take", f"{take} is outside 1 … 2**31")


def _validate_common(s, path: str, f_ref, problems: _Problems) -> None:
    """Level, damping, bloom, chatter and shape, and the sub-bands they imply."""
    problems.number(f"{path}.level_db", s.level_db, -90.0, 24.0, silence=True)
    damping = problems.number(f"{path}.damping", s.damping, 0.0, 3.0)
    bloom = problems.number(f"{path}.bloom", s.bloom, 0.0, 3.0)
    problems.number(f"{path}.chatter", s.chatter, 0.0, 1.0)
    problems.number(f"{path}.chatter_hz", s.chatter_hz, 10.0, 4000.0)
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
    band = problems.band(f"{path}.band_hz", s.band_hz)
    spacing = problems.number(f"{path}.spacing_hz", s.spacing_hz, 0.0, MAX_SPACING)
    if spacing is not None and 0 < spacing < 1:
        problems.add(f"{path}.spacing_hz", f"{spacing:g} Hz: modes need 1 … {MAX_SPACING:g} Hz, or 0 for noise")
    _validate_common(s, path, None if band is None else band[0], problems)


def _validate_tone(s: Tone, path: str, problems: _Problems) -> None:
    problems.choice(f"{path}.wave", s.wave, WAVES)
    problems.choice(f"{path}.phase", s.phase, PHASES)
    band = problems.band(f"{path}.band_hz", s.band_hz)
    lowest = _validate_partials(s.partials, f"{path}.partials", problems)
    f_ref = None if band is None or lowest is None else max(band[0], lowest)
    _validate_common(s, path, f_ref, problems)


def _validate_partials(partials, path: str, problems: _Problems):
    """The lowest partial if the partials are valid, else None."""
    if not isinstance(partials, tuple) or not all(isinstance(q, tuple) and len(q) == 2 for q in partials):
        problems.add(path, "expected (hz, level_db) partials")
        return None
    if not 1 <= len(partials) <= MAX_PARTIALS:
        problems.add(path, f"needs 1 to {MAX_PARTIALS} partials, found {len(partials)}")
        return None
    hz = [problems.number(f"{path}[{i}].hz", f, 20.0, TOP_HZ) for i, (f, _) in enumerate(partials, 1)]
    levels = [problems.number(f"{path}[{i}].level_db", v, -60.0, 0.0) for i, (_, v) in enumerate(partials, 1)]
    if None in hz:
        return None
    for i in range(1, len(hz)):
        if not grid_bin(hz[i]) > grid_bin(hz[i - 1]):
            problems.add(f"{path}[{i + 1}].hz", "partials must increase by at least 1/8 Hz, the grid's step")
    if None not in levels and max(levels) != 0.0:
        problems.add(path, f"the loudest partial must be exactly 0 dB, found {max(levels)!r}")
    return hz[0]


def _validate_damper(d: Damper, problems: _Problems) -> None:
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


def _validate_drive(d: Drive, problems: _Problems) -> None:
    if not isinstance(d, Drive):
        problems.add("drive", "expected a drive section")
    else:
        problems.number("drive.db", d.db, 0.0, 48.0)


def _validate_output(o: Output, problems: _Problems) -> None:
    if not isinstance(o, Output):
        problems.add("output", "expected an output section")
        return
    problems.integer("output.sample_rate", o.sample_rate, SAMPLE_RATES)
    problems.choice("output.format", o.format, FORMATS)
    mode = problems.choice("output.mode", o.mode, ("gain", "peak"))
    problems.number("output.db", o.db, -60.0, 0.0 if mode == "peak" else 24.0)
    problems.integer("output.polarity", o.polarity, (1, -1))


# Parsing.

_COMMON = ("band_hz", "damping", "bloom", "chatter", "chatter_hz", "amp")
_FIELDS = {
    "": ("instrument", "schema", "seed", "take", "noise", "tone", "damper", "drive", "output"),
    "noise": ("level_db", "band_hz", "spacing_hz", "damping", "bloom", "chatter", "chatter_hz", "amp"),
    "tone": ("level_db", "wave", "phase", "partials", *_COMMON),
    "damper": ("amp",),
    "drive": ("db",),
    "output": ("sample_rate", "format", "mode", "db", "polarity"),
}
_OPTIONAL = ("noise", "tone")


def _keys(problems: _Problems, prefix: str, table: dict, expected, optional=()) -> bool:
    for key in table:
        if key not in expected:
            problems.add(f"{prefix}{key}", "unknown field")
    missing = [key for key in expected if key not in table and key not in optional]
    for key in missing:
        problems.add(f"{prefix}{key}", "missing")
    return not missing


def _float(v):
    """TOML integers are accepted for floats; anything else is left for validation to name."""
    if isinstance(v, int) and not isinstance(v, bool):
        try:
            return float(v)
        except OverflowError:
            return v
    return v


def _pairs(problems: _Problems, path: str, items, keys):
    """An array of inline tables with exactly these keys, as tuples of numbers."""
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


def _section(problems: _Problems, path: str, table, fields):
    if not isinstance(table, dict):
        problems.add(path, f"expected a table, found {_kind(table)}")
        return None
    return table if _keys(problems, f"{path}.", table, fields) else None


def _band(v):
    return tuple(map(_float, v)) if isinstance(v, list) else v


def _noise(problems: _Problems, path: str, table):
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
        amp,
    )


def _tone(problems: _Problems, path: str, table):
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
        amp,
    )


def from_dict(data) -> Patch:
    """Build a patch from parsed TOML, reporting every problem at once."""
    if not isinstance(data, dict):
        raise PatchError(["expected a table"])
    # The instrument and schema come first: another instrument or a newer schema may share nothing else.
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

    problems = _Problems()
    _keys(problems, "", data, _FIELDS[""], _OPTIONAL)
    seed, take = data.get("seed"), data.get("take")
    sources = {}
    for name, build in (("noise", _noise), ("tone", _tone)):
        tables = data.get(name, [])
        if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
            problems.add(name, f"expected an array of tables ([[{name}]]), found {_kind(tables)}")
            sources[name] = None
            continue
        built = [build(problems, f"{name}[{j}]", t) for j, t in enumerate(tables, 1)]
        sources[name] = None if None in built else tuple(built)
    damper = drive = output = None
    if "damper" in data and (t := _section(problems, "damper", data["damper"], _FIELDS["damper"])) is not None:
        amp = _pairs(problems, "damper.amp", t["amp"], ("ms", "gain"))
        if amp is not None:
            damper = Damper(amp)
    if "drive" in data and (t := _section(problems, "drive", data["drive"], _FIELDS["drive"])) is not None:
        drive = Drive(_float(t["db"]))
    if "output" in data and (t := _section(problems, "output", data["output"], _FIELDS["output"])) is not None:
        output = Output(t["sample_rate"], t["format"], t["mode"], _float(t["db"]), t["polarity"])
    if None in (sources["noise"], sources["tone"], damper, drive, output) or "seed" not in data or "take" not in data:
        raise PatchError(problems or ["invalid patch"])
    # Every part is there, so every value is checked as well, and all the problems are reported together.
    try:
        p = Patch(seed, take, sources["noise"], sources["tone"], damper, drive, output)
    except PatchError as e:
        raise PatchError(problems + list(e.problems)) from None
    if problems:
        raise PatchError(problems)
    return p


def loads(text: str) -> Patch:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    return from_dict(data)


def load(path) -> Patch:
    with open(path, encoding="utf-8") as f:
        return loads(f.read())


def _knots_dict(knots) -> list:
    return [{"ms": m, "gain": g} for m, g in knots]


def to_dict(p: Patch) -> dict:
    out = {"instrument": INSTRUMENT, "schema": SCHEMA, "seed": p.seed, "take": p.take}
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
                "amp": _knots_dict(s.amp),
            }
            for s in p.tone
        ]
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
    """The complete patch as TOML with the canonical comments; loads(dumps(p)) == p exactly."""
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
            ]
        )
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in s.amp], "shape: 0 at both ends, peak exactly 1")
    for j, s in enumerate(p.tone, 1):
        lines += ["", _note("[[tone]]", f"tone[{j}]")]
        lines += _rows(
            [
                ("level_db", _f(s.level_db), ""),
                ("wave", f'"{s.wave}"', '"sine" | "square"'),
                ("phase", f'"{s.phase}"', '"strike" | "free"'),
            ]
        )
        lines += _array(
            "partials",
            [(("hz", f), ("level_db", v)) for f, v in s.partials],
            "loudest exactly 0 dB; at the nearest 1/8 Hz",
        )
        lines += _rows([("band_hz", _band_text(s.band_hz), ""), *_common_rows(s)])
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in s.amp], "shape: 0 at both ends, peak exactly 1")
    lines += ["", _note("[damper]", "multiplies every source: 1 until damped, then 0")]
    if p.damper.amp:
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in p.damper.amp], "from 1 down to 0")
    else:
        lines += [_note("amp = []", "no knots: never damped")]
    lines += ["", "[drive]"] + _rows([("db", _f(p.drive.db), "small-signal gain; 0 = bypass")])
    lines += ["", "[output]"]
    lines += _rows(
        [
            ("sample_rate", str(o.sample_rate), ""),
            ("format", f'"{o.format}"', '"float32" | "pcm24" | "pcm16"'),
            ("mode", f'"{o.mode}"', '"gain" | "peak"'),
            ("db", _f(o.db), "gain in dB, or peak target in dBFS"),
            ("polarity", str(o.polarity), ""),
        ]
    )
    return "\n".join(lines) + "\n"


def _common_rows(s) -> list:
    return [
        ("damping", _f(s.damping), "content at f decays (f / f_ref)^damping times faster"),
        ("bloom", _f(s.bloom), "and rises (f / f_ref)^bloom times slower"),
        ("chatter", _f(s.chatter), "0 … 1: gated at chatter_hz, as plates that slap"),
        ("chatter_hz", _f(s.chatter_hz), ""),
    ]


def _f(x: float) -> str:
    return repr(float(x))


def _band_text(band) -> str:
    return f"[{_f(band[0])}, {_f(band[1])}]"


def _note(line: str, note: str) -> str:
    return f"{(line + ' ').ljust(_COMMENT_COLUMN)}# {note}" if note else line


def _rows(rows) -> list[str]:
    width = max(len(k) for k, _, _ in rows)
    return [_note(f"{k.ljust(width)} = {v}", note) for k, v, note in rows]


def _array(name, items, note) -> list[str]:
    """An array of inline tables, with the first column aligned."""
    firsts = [f"{items[0][0][0]} = {_f(a[1])}," for a, _ in items] if items else []
    width = max(map(len, firsts), default=0)
    body = [f"  {{ {first.ljust(width)} {key} = {_f(v)} }}," for first, (_, (key, v)) in zip(firsts, items)]
    return [_note(f"{name} = [", note), *body, "]"]


# Pitch vocabulary shared by `cymbal pitch`, the report and the WAV root note.

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_pitch(hz: float) -> float:
    """Fractional MIDI note number (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(midi: int) -> str:
    return f"{_NOTES[midi % 12]}{midi // 12 - 1}"


def describe_pitch(hz: float) -> str:
    """For example 'F#3 -3.2 cents'."""
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_note(text: str) -> float | None:
    """A note name such as D5, C#4 or Eb3, in Hz; None if the text is not a note name."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if not m:
        return None
    midi = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1)
    return 440.0 * 2.0 ** ((midi - 69) / 12)


# Editing commands. Each returns a new, validated patch, or raises PatchError.

_US = Decimal("0.001")  # knot times are rewritten to 1 µs
_MHZ = Decimal("0.001")  # frequencies are rewritten to 1 mHz


def _dec(x: float) -> Decimal:
    """The decimal value as written."""
    return Decimal(repr(float(x)))


def _ms(x) -> float:
    return float(Decimal(x).quantize(_US))


def _hz(x: Decimal) -> float:
    return float(x.quantize(_MHZ))


def _significant(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def pitch_ratio(p: Patch, text: str) -> float:
    """The ratio a `pitch` argument names: ±semitones, or a note for tone[1]'s lowest partial."""
    hz = parse_note(text)
    if hz is not None:
        if not p.tone:
            raise PatchError([f"pitch {text}: a note sets the lowest partial of tone[1], and this patch has no tone"])
        return hz / p.tone[0].partials[0][0]
    try:
        semitones = float(text)
    except ValueError:
        raise PatchError(
            [f"pitch: expected semitones such as +2 or -3.5, or a note such as D5, found {text!r}"]
        ) from None
    if not math.isfinite(semitones):
        raise PatchError([f"pitch: expected a finite number of semitones, found {text!r}"])
    return 2.0 ** (semitones / 12.0)


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
    names = [f"{prefix}.amp" for prefix, _ in p.sources]
    return (*names, "damper.amp")


def _get_curve(p: Patch, name: str) -> tuple:
    if name == "damper.amp":
        return p.damper.amp
    return _source(p, name[: -len(".amp")]).amp


def _source(p: Patch, name: str):
    for prefix, s in p.sources:
        if prefix == name:
            return s
    names = ", ".join(prefix for prefix, _ in p.sources)
    raise PatchError([f"{name}: not a source of this patch; it has {names}"])


def _set_curves(p: Patch, curves: dict) -> Patch:
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


def stretch(p: Patch, curves, factor: float, after_ms: float | None = None) -> Patch:
    """Rescale knot times after each shape's peak, or after `after_ms`, by `factor`.

    `curves` names curves by path (`noise[1].amp`, `tone[1].amp`, `damper.amp`);
    `all` names every shape and the damper. The damper's times scale from 0 ms
    unless `after_ms` is given. Times are rewritten to 1 µs.
    """
    if not (math.isfinite(factor) and factor > 0):
        raise PatchError([f"stretch: the factor must be a positive number, found {factor!r}"])
    if after_ms is not None and not (math.isfinite(after_ms) and after_ms >= 0):
        raise PatchError([f"--after: must be a time in ms from 0 on, found {after_ms!r}"])
    names = []
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
        if after_ms is not None:
            anchor = after_ms
        else:
            anchor = 0.0 if name == "damper.amp" else peak_ms(knots)
        a, f = _dec(anchor), Decimal(repr(float(factor)))
        out[name] = tuple((ms if ms <= anchor else _ms(a + f * (_dec(ms) - a)), v) for ms, v in knots)
    return _set_curves(p, out)


CHOKE_DB = 60.0  # a choke's release falls this far, in knots every 10 dB
DEFAULT_RELEASE_MS = 40.0


def choke_knots(ms: float, release: float = DEFAULT_RELEASE_MS) -> tuple:
    """A choke's damper: gain 1 at `ms`, knots every 10 dB evenly spaced over `release` down to −60 dB, and 0 one
    step later. Times are written to 1 µs and gains to 4 significant digits. Audition fades a choked hit the same way.
    """
    steps = int(CHOKE_DB // 10)
    if not (math.isfinite(ms) and ms >= 0):
        raise PatchError([f"choke: the time must be in ms from 0 on, found {ms!r}"])
    if not (math.isfinite(release) and release / steps >= float(MIN_SPACING_MS)):
        raise PatchError([f"--release: needs at least {steps * float(MIN_SPACING_MS):g} ms, found {release!r}"])
    start, step = _dec(ms), _dec(release) / steps
    knots = [(_ms(start), 1.0)]
    for k in range(1, steps + 1):
        knots.append((_ms(start + step * k), _significant(10 ** (-k / 2))))
    knots.append((_ms(start + step * (steps + 1)), 0.0))
    return tuple(knots)


def choke(p: Patch, ms: float, release: float = DEFAULT_RELEASE_MS) -> Patch:
    """Replace the damper with a choke at `ms` that falls 60 dB over `release` (plan §4)."""
    return dataclasses.replace(p, damper=Damper(choke_knots(ms, release)))


def rises(values) -> list[tuple[int, int]]:
    """(first, last) knot indices of each rise: a maximal run of strictly increasing knots.

    Both ends of a rise have zero slope, the first because nothing before it is
    lower and the last because nothing after it is higher, so a rise can be
    lengthened without changing the rest of the curve.
    """
    out, i, n = [], 0, len(values)
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


def _lengthen_rises(knots: tuple, factor: Decimal) -> tuple:
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
    return tuple((float(t), v) for t, (_, v) in zip(new, knots))


def softer(p: Patch, db: float, contact: float = 0.2) -> Patch:
    """The same cymbal struck `db` softer, by two rules of plan §2.11.

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

    def level(x):
        return x if x == -math.inf else float(round(_dec(x) - Decimal(repr(float(db))), 6))

    def soften(s):
        amp, band = s.amp, s.band_hz
        if contact:
            amp = _lengthen_rises(amp, Decimal(repr(v**-contact)))
            band = (band[0], float(round(_dec(band[1]) * Decimal(repr(v**contact)), 1)))
        return dataclasses.replace(s, level_db=level(s.level_db), band_hz=band, amp=amp)

    return dataclasses.replace(p, noise=tuple(map(soften, p.noise)), tone=tuple(map(soften, p.tone)))
