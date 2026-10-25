# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line (plan §7.6): brightwork init | render | verify | extract | audition | pack.

The editing commands, pitch | stretch | choke | softer, print a complete, canonical patch.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, wav
from .audition import Settings, audition
from .pack import Options, pack
from .patch import DEFAULT_RELEASE_MS, Patch, PatchError, choke, dumps, load, loads, pitch, softer, stretch
from .reference import verify
from .render import FORMATS, RenderError, db, export, render_takes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

REFERENCES = (
    "closed",
    "open",
    "pedal",
    "crash",
    "ride",
    "bell",
    "analog",
    "shaker",
    "tambourine",
    "offbeat",
    "tight",
    "dead",
    "wash",
)
SHORTEST_BEATS, LONGEST_BEATS = 1 / 8, 16  # the grid's note values, 1/32 to 4/1, in beats of a 4/4 bar
MAX_KEY = 127  # the highest MIDI key


def reference_text(name: str) -> str:
    """Return a reference patch's text, as it ships."""
    return resources.files("brightwork_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


class CliError(Exception):
    """A message for stderr, and the exit status: 2 for bad input, 1 for a refusal or a failure."""

    def __init__(self, message: str, status: int = 2) -> None:
        """Keep the message, and the status to exit with."""
        super().__init__(message)
        self.status = status


def _listing(head: str, problems: Sequence[str]) -> CliError:
    """Make an error of `head`, then every problem, indented."""
    return CliError("\n".join([head, *(f"  {m}" for m in problems)]))


def _patch(path: str) -> Patch:
    """Read a patch file."""
    try:
        return load(path)
    except OSError as e:
        msg = f"{path}: {e.strerror or e}"
        raise CliError(msg) from None
    except PatchError as e:
        msg = f"{path}: invalid patch"
        raise _listing(msg, e.problems) from None


def _edit(path: str, what: str, edit: Callable[..., Patch], *args: object) -> int:
    """Apply an editing command and print the patch."""
    try:
        q = edit(_patch(path), *args)
    except PatchError as e:
        msg = f"{path}: cannot {what}"
        raise _listing(msg, e.problems) from None
    sys.stdout.write(dumps(q))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Print a reference patch, as a complete starting point."""
    sys.stdout.write(dumps(loads(reference_text(args.reference))))
    return 0


def cmd_pitch(args: argparse.Namespace) -> int:
    """Print the patch with every frequency moved by semitones, or to a note."""
    return _edit(args.patch, f"pitch it {args.amount}", pitch, args.amount)


def cmd_stretch(args: argparse.Namespace) -> int:
    """Print the patch with curves' knot times rescaled."""
    what = f"stretch {' '.join(args.curves)} by {args.factor:g}"
    return _edit(args.patch, what, stretch, args.curves, args.factor, args.after)


def cmd_choke(args: argparse.Namespace) -> int:
    """Print the patch with a damper that chokes it."""
    return _edit(args.patch, f"choke it at {args.ms:g} ms", choke, args.ms, args.release)


def cmd_softer(args: argparse.Namespace) -> int:
    """Print the same cymbal struck softer."""
    return _edit(args.patch, f"make it {args.db:g} dB softer", softer, args.db, args.contact)


def file_paths(out: str, files: int) -> list[Path]:
    """Name a set's files: `crash.wav` for one file; `crash_rr1.wav` … for several."""
    path = Path(out)
    if files == 1:
        return [path]
    return [path.with_name(f"{path.stem}_rr{j}{path.suffix or '.wav'}") for j in range(1, files + 1)]


def cmd_render(args: argparse.Namespace) -> int:
    """Render a patch, its takes or its stereo pairs, and write the files unless they would clip."""
    p = _patch(args.patch)
    pcm = FORMATS[p.output.format] is not None
    if args.allow_over and pcm:
        msg = "--allow-over writes float32 only: PCM cannot hold samples beyond full scale"
        raise CliError(msg)
    try:
        audio, report = render_takes(p, args.takes, level_match=args.level_match, stereo=args.stereo)
    except RenderError as e:
        msg = f"{args.patch}: {e}"
        raise CliError(msg) from None
    if not report.over or args.allow_over:
        # Every file first, then the report: a closed output never leaves a set half written.
        paths = file_paths(args.out, len(report.files))
        for path, a, f in zip(paths, audio, report.files, strict=True):
            export(path, a, p, report, f)
        sys.stdout.write(f"{report}\n" + "".join(f"wrote {path}\n" for path in paths))
        return 0
    sys.stdout.write(f"{report}\n")
    if pcm:
        msg = (
            f"not written: the peak is {db(report.peak):+.4f} dBFS, above {p.output.format}'s ceiling of "
            f"{db(report.ceiling):.4f} dBFS, which keeps 2 LSB for dither. Lower output.db by at least "
            f"{db(report.peak / report.ceiling):.4f} dB."
        )
    else:
        msg = (
            f"not written: the peak is {db(report.peak):+.2f} dBFS. Lower output.db by at least "
            f"{db(report.peak):.2f} dB, or pass --allow-over to write a float file with overs."
        )
    raise CliError(msg, 1)


def cmd_verify(args: argparse.Namespace) -> int:
    """Compare a patch, or its takes, with the reference render."""
    v = verify(_patch(args.patch), takes=args.takes)
    sys.stdout.write(f"{v}\n")
    return 0 if v.passed else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Print the patch embedded in an exported WAV."""
    try:
        found = wav.chunks(args.wav)
    except OSError as e:
        msg = f"{args.wav}: {e.strerror or e}"
        raise CliError(msg) from None
    except ValueError as e:
        raise CliError(str(e)) from None
    if wav.PRIVATE not in found:
        msg = f"{args.wav}: no embedded Brightwork patch"
        raise CliError(msg)
    sys.stdout.write(json.loads(found[wav.PRIVATE].decode("utf-8"))["patch"])
    return 0


def cmd_audition(args: argparse.Namespace) -> int:
    """Play patches in a loop or a pattern."""
    settings = Settings(
        watch=args.watch,
        bpm=args.bpm,
        every=args.every,
        pattern=args.pattern,
        chokes=tuple(args.choke),
        release_ms=args.release,
        bed=args.bed,
        against=args.against,
        blind=args.blind,
        takes=args.takes,
        stereo=args.stereo,
    )
    return audition(args.patches, settings)


def cmd_pack(args: argparse.Namespace) -> int:
    """Render a kit and write it, with its SFZ mapping."""
    options = Options(args.takes, args.stereo, args.level_match, tuple(args.choke), args.release)
    return pack(args.groups, args.out, options)


def _finite(text: str) -> float:
    """Parse a finite number."""
    try:
        x = float(text)
    except ValueError:
        msg = f"expected a number, found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(x):
        msg = f"expected a finite number, found {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return x


def _note_value(text: str) -> float:
    """Parse a note value such as 1/16 or 4/1, in beats of a 4/4 bar (a quarter note is one beat)."""
    try:
        num, den = (int(part) for part in text.split("/"))
    except ValueError:
        msg = f"expected a note value such as 1/8 or 4/1, found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    beats = 4 * num / den if den else 0
    if not (SHORTEST_BEATS <= beats <= LONGEST_BEATS):
        msg = "note values run from 1/32 to 4/1"
        raise argparse.ArgumentTypeError(msg)
    return beats


def _pattern(text: str) -> tuple[int, ...]:
    """Parse a pattern: patch numbers per step, 0 for a rest."""
    try:
        steps = tuple(int(s) for s in text.split(","))
    except ValueError:
        msg = f"expected patch numbers per step, such as 1,1,2,1 or 1,0,1,0, found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not steps or min(steps) < 0:
        msg = "a pattern lists patch numbers from 1, with 0 for a rest"
        raise argparse.ArgumentTypeError(msg)
    return steps


def _choke(text: str) -> tuple[int, tuple[int, ...]]:
    """Parse a choke rule A:B,C, in which A is choked by B and C."""
    try:
        target, by = text.split(":")
        return int(target), tuple(int(b) for b in by.split(","))
    except ValueError:
        msg = f"expected A:B,C (A is choked by B and C), found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None


def _group(text: str) -> tuple[int, tuple[str, ...]]:
    """Parse a kit's key and its layers' patches, KEY=PATCH[,PATCH…]."""
    try:
        name, patches = text.split("=", 1)
        key = int(name)
    except ValueError:
        msg = f"expected KEY=PATCH[,PATCH…], such as 42=closed.toml, found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not 0 <= key <= MAX_KEY:
        msg = f"MIDI keys run from 0 to 127, found {key}"
        raise argparse.ArgumentTypeError(msg)
    paths = tuple(p for p in patches.split(",") if p)
    if not paths:
        msg = f"key {key} needs at least one patch"
        raise argparse.ArgumentTypeError(msg)
    return key, paths


def _release(p: argparse.ArgumentParser, what: str) -> None:
    """Add a --release option: a choke's fade, or its fall."""
    p.add_argument("--release", type=_finite, default=DEFAULT_RELEASE_MS, metavar="MS", help=f"{what} (default 40)")


def _render_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the commands that start, render, verify and extract patches."""
    p = sub.add_parser("init", help="print a complete starting patch")
    p.add_argument("reference", choices=REFERENCES)
    p.set_defaults(run=cmd_init)

    p = sub.add_parser("render", help="render a patch, a set of round-robin takes, or stereo pairs, to WAV")
    p.add_argument("patch")
    p.add_argument("--out", required=True, help="the file; several files are written as <stem>_rr<n>.wav")
    p.add_argument("--takes", type=int, default=1, help="round-robin files: takes take, take + 1, … (default 1)")
    p.add_argument("--stereo", action="store_true", help="two consecutive takes per file, as left and right")
    p.add_argument("--level-match", action="store_true", help="scale each file to the loudness of the first")
    p.add_argument(
        "--allow-over",
        action="store_true",
        help="write a gain-mode render that exceeds full scale (float32)",
    )
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("verify", help="compare a patch, or its takes, with the reference render")
    p.add_argument("patch")
    p.add_argument("--takes", type=int, default=1)
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("extract", help="print the patch embedded in an exported WAV")
    p.add_argument("wav")
    p.set_defaults(run=cmd_extract)


def _edit_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the editing commands, which print a complete patch."""
    p = sub.add_parser("pitch", help="print the patch with every frequency moved by semitones, or to a note")
    p.add_argument("patch")
    p.add_argument("amount", help="semitones such as +2 or -3.5, or a note such as D5 for tone[1]'s lowest partial")
    p.set_defaults(run=cmd_pitch)

    p = sub.add_parser("stretch", help="print the patch with curves' knot times rescaled after their peaks")
    p.add_argument("patch")
    p.add_argument("curves", nargs="+", metavar="CURVE", help="noise[N].amp, tone[N].amp, damper.amp, or all")
    p.add_argument("factor", type=_finite, help="below 1 shortens, above 1 lengthens")
    p.add_argument(
        "--after",
        type=_finite,
        metavar="MS",
        help="keep knots up to this time (default: each shape's peak, and 0 ms for the damper)",
    )
    p.set_defaults(run=cmd_stretch)

    p = sub.add_parser("choke", help="print the patch with a damper that chokes it at a time")
    p.add_argument("patch")
    p.add_argument("ms", type=_finite, metavar="MS", help="when the damper starts")
    _release(p, "its 60 dB fall")
    p.set_defaults(run=cmd_choke)

    p = sub.add_parser("softer", help="print the same cymbal struck DB softer")
    p.add_argument("patch")
    p.add_argument("db", type=_finite, metavar="DB")
    p.add_argument("--contact", type=_finite, default=0.2, help="the contact exponent (default 0.2; 0 turns it off)")
    p.set_defaults(run=cmd_softer)


def _play_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add audition and pack, which play patches together."""
    p = sub.add_parser("audition", help="play patches in a loop or a pattern, with chokes, over a bed or a sample")
    p.add_argument("patches", nargs="+", metavar="PATCH")
    p.add_argument("--watch", action="store_true", help="re-render whenever a file changes")
    p.add_argument("--bpm", type=_finite, help="hit on a grid of 4/4 bars at this tempo")
    p.add_argument("--every", type=_note_value, default=1.0, metavar="NOTE", help="the grid: 1/32 to 4/1 (default 1/4)")
    p.add_argument("--pattern", type=_pattern, help="patch numbers per step, 0 for a rest, such as 1,1,2,1")
    p.add_argument("--choke", type=_choke, action="append", default=[], metavar="A:B,…", help="A is choked by B, …")
    _release(p, "a choke's fade")
    p.add_argument(
        "--with",
        dest="bed",
        metavar="WAV",
        help="loop a bed under the hits: a whole number of bars at --bpm",
    )
    p.add_argument("--against", metavar="WAV", help="alternate with a sample, hit by hit, matched in loudness")
    p.add_argument("--blind", action="store_true", help="with --against, play each pair in random order")
    p.add_argument("--takes", type=int, default=1, help="cycle this many takes, hit by hit")
    p.add_argument("--stereo", action="store_true", help="play take pairs as stereo, and a stereo sample in stereo")
    p.set_defaults(run=cmd_audition)

    p = sub.add_parser("pack", help="render a kit: velocity layers with takes on keys, chokes, and an SFZ mapping")
    p.add_argument("groups", nargs="+", type=_group, metavar="KEY=PATCH[,PATCH…]", help="loudest layer first")
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--takes", type=int, default=1)
    p.add_argument("--stereo", action="store_true")
    p.add_argument("--level-match", action="store_true")
    p.add_argument("--choke", type=_choke, action="append", default=[], metavar="KEY:KEY,…", help="e.g. 46:42,44")
    _release(p, "a choke's fade")
    p.set_defaults(run=cmd_pack)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the brightwork command, and return its exit status."""
    what = "A one-voice synthesizer of hi-hats, cymbals, shakers and tambourines."
    ap = argparse.ArgumentParser(prog="brightwork", description=what)
    ap.add_argument("--version", action="version", version=f"brightwork {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    _render_commands(sub)
    _edit_commands(sub)
    _play_commands(sub)
    for stream in (sys.stdout, sys.stderr):  # not the ANSI code page and CRLF of a Windows redirect
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", newline="\n")
    args = ap.parse_args(argv)
    if getattr(args, "bpm", None) is not None and not args.bpm > 0:
        ap.error("--bpm must be positive")
    try:
        code: int = args.run(args)
    except CliError as e:
        sys.stderr.write(f"{e}\n")
        return e.status
    except BrokenPipeError:  # the reader went away, as `brightwork render … | head` does: the files are written
        sys.stderr.close()
        return 0
    except KeyboardInterrupt:  # Ctrl+C: stop quietly, with the status a shell expects
        return 130
    return code
