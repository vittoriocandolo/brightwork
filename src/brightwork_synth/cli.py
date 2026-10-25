# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line (plan §7.6): brightwork init | render | verify | extract | audition | pack, and the editing commands
pitch | stretch | choke | softer, which print a complete, canonical patch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import resources
from pathlib import Path

from . import patch as P
from .patch import PatchError, dumps, loads

REFERENCES = ("closed", "open", "pedal", "crash", "ride", "bell", "analog")


def reference_text(name: str) -> str:
    return resources.files("brightwork_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise SystemExit(_fail(f"{path}: {e.strerror}"))


def _fail(message: str, code: int = 2) -> int:
    print(message, file=sys.stderr)
    return code


def _patch(path: str, text: str | None = None) -> P.Patch:
    try:
        return loads(_read(path) if text is None else text)
    except PatchError as e:
        raise SystemExit(_fail(f"{path}: invalid patch\n" + "\n".join(f"  {m}" for m in e.problems)))


def _edit(path: str, what: str, fn, *args, **kwargs) -> int:
    """Apply an editing command and print the patch, or say why it cannot be done."""
    p = _patch(path)
    try:
        q = fn(p, *args, **kwargs)
    except (ValueError, PatchError) as e:
        problems = e.problems if isinstance(e, PatchError) else [str(e)]
        return _fail(f"{path}: cannot {what}\n" + "\n".join(f"  {m}" for m in problems))
    sys.stdout.write(dumps(q))
    return 0


def cmd_init(args) -> int:
    sys.stdout.write(dumps(loads(reference_text(args.reference))))
    return 0


def cmd_pitch(args) -> int:
    return _edit(args.patch, f"pitch it {args.amount}", P.pitch, args.amount)


def cmd_stretch(args) -> int:
    return _edit(
        args.patch,
        f"stretch {' '.join(args.curves)} by {args.factor:g}",
        P.stretch,
        args.curves,
        args.factor,
        args.after,
    )


def cmd_choke(args) -> int:
    return _edit(args.patch, f"choke it at {args.ms:g} ms", P.choke, args.ms, args.release)


def cmd_softer(args) -> int:
    return _edit(args.patch, f"make it {args.db:g} dB softer", P.softer, args.db, args.contact)


def file_paths(out: str, files: int) -> list[Path]:
    """`crash.wav` for one file; `crash_rr1.wav` … for several."""
    path = Path(out)
    if files == 1:
        return [path]
    return [path.with_name(f"{path.stem}_rr{j}{path.suffix or '.wav'}") for j in range(1, files + 1)]


def cmd_render(args) -> int:
    from .render import FORMATS, RenderError, db, export, render_takes

    p = _patch(args.patch)
    if args.allow_over and FORMATS[p.output.format] is not None:
        return _fail("--allow-over writes float32 only: PCM cannot hold samples beyond full scale")
    try:
        audio, report = render_takes(p, args.takes, args.level_match, args.stereo)
    except RenderError as e:
        return _fail(f"{args.patch}: {e}")
    if not report.over or (args.allow_over and FORMATS[p.output.format] is None):
        # Every file first, then the report: a closed output never leaves a set half written.
        paths = file_paths(args.out, len(report.files))
        for path, a, f in zip(paths, audio, report.files):
            export(path, a, p, report, f)
        print(report)
        for path in paths:
            print(f"wrote {path}")
        return 0
    print(report)
    if FORMATS[p.output.format] is not None:
        return _fail(
            f"not written: the peak is {db(report.peak):+.4f} dBFS, above {p.output.format}'s ceiling of "
            f"{db(report.ceiling):.4f} dBFS, which keeps 2 LSB for dither. Lower output.db by at least "
            f"{db(report.peak / report.ceiling):.4f} dB.",
            1,
        )
    return _fail(
        f"not written: the peak is {db(report.peak):+.2f} dBFS. Lower output.db by at least "
        f"{db(report.peak):.2f} dB, or pass --allow-over to write a float file with overs.",
        1,
    )


def cmd_verify(args) -> int:
    from .reference import verify

    p = _patch(args.patch)
    v = verify(p, takes=args.takes)
    print(v)
    return 0 if v.passed else 1


def cmd_extract(args) -> int:
    from . import wav

    try:
        found = wav.chunks(args.wav)
    except (OSError, ValueError) as e:
        return _fail(f"{args.wav}: {e}")
    if wav.PRIVATE not in found:
        return _fail(f"{args.wav}: no embedded Brightwork patch")
    sys.stdout.write(json.loads(found[wav.PRIVATE].decode("utf-8"))["patch"])
    return 0


def cmd_audition(args) -> int:
    from .audition import audition

    return audition(
        args.patches,
        watch=args.watch,
        bpm=args.bpm,
        every=args.every,
        pattern=args.pattern,
        chokes=args.choke,
        release_ms=args.release,
        bed=args.bed,
        against=args.against,
        blind=args.blind,
        takes=args.takes,
        stereo=args.stereo,
    )


def cmd_pack(args) -> int:
    from .pack import pack

    return pack(
        args.groups,
        args.out,
        takes=args.takes,
        stereo=args.stereo,
        level_match=args.level_match,
        chokes=args.choke,
        release_ms=args.release,
    )


def _finite(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, found {text!r}") from None
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError(f"expected a finite number, found {text!r}")
    return x


def _note_value(text: str) -> float:
    """A note value such as 1/16 or 4/1, in beats of a 4/4 bar (a quarter note is one beat)."""
    try:
        num, den = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a note value such as 1/8 or 4/1, found {text!r}") from None
    beats = 4 * num / den if den else 0
    if not (1 / 8 <= beats <= 16):
        raise argparse.ArgumentTypeError("note values run from 1/32 to 4/1")
    return beats


def _pattern(text: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(s) for s in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected patch numbers per step, such as 1,1,2,1 or 1,0,1,0, found {text!r}")
    if not steps or min(steps) < 0:
        raise argparse.ArgumentTypeError("a pattern lists patch numbers from 1, with 0 for a rest")
    return steps


def _choke(text: str) -> tuple[int, tuple[int, ...]]:
    try:
        target, by = text.split(":")
        return int(target), tuple(int(b) for b in by.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected A:B,C (A is choked by B and C), found {text!r}") from None


def _group(text: str) -> tuple[int, tuple[str, ...]]:
    try:
        key, patches = text.split("=", 1)
        key = int(key)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected KEY=PATCH[,PATCH…], such as 42=closed.toml, found {text!r}"
        ) from None
    if not 0 <= key <= 127:
        raise argparse.ArgumentTypeError(f"MIDI keys run from 0 to 127, found {key}")
    paths = tuple(p for p in patches.split(",") if p)
    if not paths:
        raise argparse.ArgumentTypeError(f"key {key} needs at least one patch")
    return key, paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="brightwork", description="A one-voice hi-hat and cymbal synthesizer.")
    sub = ap.add_subparsers(dest="command", required=True)

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
        "--allow-over", action="store_true", help="write a gain-mode render that exceeds full scale (float32)"
    )
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("verify", help="compare a patch, or its takes, with the reference render")
    p.add_argument("patch")
    p.add_argument("--takes", type=int, default=1)
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("extract", help="print the patch embedded in an exported WAV")
    p.add_argument("wav")
    p.set_defaults(run=cmd_extract)

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
    p.add_argument(
        "--release", type=_finite, default=P.DEFAULT_RELEASE_MS, metavar="MS", help="its 60 dB fall (default 40)"
    )
    p.set_defaults(run=cmd_choke)

    p = sub.add_parser("softer", help="print the same cymbal struck DB softer")
    p.add_argument("patch")
    p.add_argument("db", type=_finite, metavar="DB")
    p.add_argument("--contact", type=_finite, default=0.2, help="the contact exponent (default 0.2; 0 turns it off)")
    p.set_defaults(run=cmd_softer)

    p = sub.add_parser("audition", help="play patches in a loop or a pattern, with chokes, over a bed or a sample")
    p.add_argument("patches", nargs="+", metavar="PATCH")
    p.add_argument("--watch", action="store_true", help="re-render whenever a file changes")
    p.add_argument("--bpm", type=_finite, help="hit on a grid of 4/4 bars at this tempo")
    p.add_argument("--every", type=_note_value, default=1.0, metavar="NOTE", help="the grid: 1/32 to 4/1 (default 1/4)")
    p.add_argument("--pattern", type=_pattern, help="patch numbers per step, 0 for a rest, such as 1,1,2,1")
    p.add_argument("--choke", type=_choke, action="append", default=[], metavar="A:B,…", help="A is choked by B, …")
    p.add_argument(
        "--release", type=_finite, default=P.DEFAULT_RELEASE_MS, metavar="MS", help="a choke's fade (default 40)"
    )
    p.add_argument(
        "--with", dest="bed", metavar="WAV", help="loop a bed under the hits: a whole number of bars at --bpm"
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
    p.add_argument(
        "--release", type=_finite, default=P.DEFAULT_RELEASE_MS, metavar="MS", help="a choke's fade (default 40)"
    )
    p.set_defaults(run=cmd_pack)

    args = ap.parse_args(argv)
    if getattr(args, "bpm", None) is not None and not args.bpm > 0:
        ap.error("--bpm must be positive")
    try:
        return args.run(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except BrokenPipeError:  # the reader went away, as `brightwork render … | head` does: the files are written
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
