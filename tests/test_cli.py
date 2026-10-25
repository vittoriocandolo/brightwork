# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The command line: every command, its files, its report and its errors (plan §7.6)."""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile
from numpy.typing import NDArray

from brightwork_synth import audition, cli, wav
from brightwork_synth.patch import dumps, loads
from brightwork_synth.render import render, render_takes

from .conftest import EXAMPLE

BAD_INPUT = 2  # the exit status for input the command cannot use
REFUSED = 1  # the exit status for a render refused, or a verification failed
RATE = 48000
CHOKED_MS = 95.0  # where a choke at 60 ms with a 30 ms release ends: one step after its 60 dB fall
COMPOSED_HZ = 0.2  # how far 6 then 6 dB softer may round an upper band edge from 12 dB softer


def run(capsys: pytest.CaptureFixture[str], *args: object) -> tuple[int, str, str]:
    """Run `brightwork ARGS…`, and return its exit status, standard output and standard error."""
    code = cli.main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Write the plan's example patch to a file."""
    path = tmp_path / "hat.toml"
    path.write_text(EXAMPLE)
    return path


@pytest.mark.parametrize("name", cli.REFERENCES)
def test_init_writes_complete_patches(capsys: pytest.CaptureFixture[str], name: str) -> None:
    """Print each reference as a complete, canonical patch."""
    code, out, _ = run(capsys, "init", name)
    assert code == 0
    assert out == dumps(loads(out))
    assert loads(out) == loads(cli.reference_text(name))


def test_render_and_extract(capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Path) -> None:
    """Render the example to the very samples `render` makes, report it, and extract its exact patch."""
    out = tmp_path / "hat.wav"
    code, text, _ = run(capsys, "render", example, "--out", out)
    assert code == 0
    assert f"wrote {out}" in text
    assert "714 modes in 6000–16000 Hz" in text
    data, rate = soundfile.read(out, dtype="float32")
    assert rate == RATE
    assert np.array_equal(data, render(loads(EXAMPLE))[0])
    code, text, _ = run(capsys, "extract", out)
    assert code == 0
    assert text == dumps(loads(EXAMPLE))
    # The report comes after the files are written, and names every one.
    assert text.count("\n") > 1
    assert out.exists()


def test_render_writes_round_robin_takes(capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Path) -> None:
    """Render three level-matched takes as three files, each holding its own take's patch; refuse no takes."""
    out = tmp_path / "hat.wav"
    code, text, _ = run(capsys, "render", example, "--out", out, "--takes", 3, "--level-match")
    assert code == 0
    assert "3 files sharing one gain" in text
    assert "level-matched" in text
    audio, _ = render_takes(loads(EXAMPLE), 3, level_match=True)
    for j in (1, 2, 3):
        path = tmp_path / f"hat_rr{j}.wav"
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], audio[j - 1])
        # Each file holds its own patch: the patch with its take.
        assert loads(run(capsys, "extract", path)[1]) == dataclasses.replace(loads(EXAMPLE), take=j)
    assert not out.exists()
    code, _, err = run(capsys, "render", example, "--out", out, "--takes", 0)
    assert code == BAD_INPUT
    assert "--takes" in err


def test_render_writes_stereo_pairs(capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Path) -> None:
    """Render take pairs as stereo files, each holding its left channel's patch; refuse too many."""
    out = tmp_path / "pair.wav"
    code, text, _ = run(capsys, "render", example, "--out", out, "--stereo", "--takes", 2)
    assert code == 0
    assert "stereo, takes as left and right" in text
    assert "takes 1–4" in text
    audio, _ = render_takes(loads(EXAMPLE), 2, stereo=True)
    for j in (1, 2):
        path = tmp_path / f"pair_rr{j}.wav"
        data, _ = soundfile.read(path, dtype="float32")
        assert data.shape == audio[j - 1].shape
        assert np.array_equal(data, audio[j - 1])
        assert loads(run(capsys, "extract", path)[1]).take == 2 * j - 1  # the left channel's patch
    code, _, err = run(capsys, "render", example, "--out", out, "--stereo", "--takes", 65)
    assert code == BAD_INPUT
    assert "with --stereo" in err


def test_render_refuses_overs_unless_allowed(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Refuse to write a render over full scale, saying what to lower, unless --allow-over writes float32."""
    patch = tmp_path / "loud.toml"
    patch.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"))
    out = tmp_path / "loud.wav"
    code, _, err = run(capsys, "render", patch, "--out", out)
    assert code == REFUSED
    assert not out.exists()
    assert "not written" in err
    assert "Lower output.db by at least" in err
    code, _, _ = run(capsys, "render", patch, "--out", out, "--allow-over")
    assert code == 0
    assert np.max(np.abs(wav.read(out).samples)) > 1.0


def test_render_writes_pcm(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Render 24- and 16-bit PCM, refuse --allow-over for it, and refuse a render over its ceiling."""
    for fmt, subtype in (("pcm24", "PCM_24"), ("pcm16", "PCM_16")):
        patch = tmp_path / f"{fmt}.toml"
        patch.write_text(EXAMPLE.replace('"float32"     #', f'"{fmt}"       #'))
        assert loads(patch.read_text()).output.format == fmt
        out = tmp_path / f"{fmt}.wav"
        code, text, _ = run(capsys, "render", patch, "--out", out)
        assert code == 0
        assert f"{fmt[3:]}-bit PCM" in text
        assert soundfile.info(out).subtype == subtype
        assert loads(run(capsys, "extract", out)[1]) == loads(patch.read_text())
        code, _, err = run(capsys, "render", patch, "--out", tmp_path / "x.wav", "--allow-over")
        assert code == BAD_INPUT
        assert "float32 only" in err
    loud = tmp_path / "loud.toml"
    loud.write_text(
        EXAMPLE.replace('"float32"     #', '"pcm24"       #').replace("db          = -6.0", "db          = 12.0"),
    )
    code, _, err = run(capsys, "render", loud, "--out", tmp_path / "loud.wav")
    assert code == REFUSED
    assert "not written" in err
    assert "ceiling" in err
    assert not (tmp_path / "loud.wav").exists()


def test_errors_are_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Report an invalid patch's every problem, a missing file, and a WAV without a patch."""
    bad = tmp_path / "bad.toml"
    bad.write_text(EXAMPLE.replace("damping    = 0.5", "damping    = 4.0").replace("seed = 1 ", "seed = -1 "))
    code, _, err = run(capsys, "render", bad, "--out", tmp_path / "x.wav")
    assert code == BAD_INPUT
    assert "noise[1].damping: 4 is outside 0 … 3" in err
    assert "seed:" in err
    code, _, err = run(capsys, "render", tmp_path / "missing.toml", "--out", "x.wav")
    assert code == BAD_INPUT
    assert "missing.toml" in err
    other = tmp_path / "plain.wav"
    soundfile.write(other, np.zeros(10), 48000)
    code, _, err = run(capsys, "extract", other)
    assert code == BAD_INPUT
    assert "no embedded Brightwork patch" in err
    code, _, err = run(capsys, "extract", tmp_path / "nothing.wav")
    assert code == BAD_INPUT
    assert "nothing.wav" in err


def test_pitch(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Pitch a patch by semitones or to a note, and refuse a word or a pitch that leaves the valid domain."""
    analog = tmp_path / "analog.toml"
    analog.write_text(cli.reference_text("analog"))
    p = loads(analog.read_text())
    code, out, _ = run(capsys, "pitch", analog, "-2")
    assert code == 0
    assert out == dumps(loads(out))
    r = 2 ** (-2 / 12)
    assert loads(out).tone[0].partials[0][0] == round(p.tone[0].partials[0][0] * r, 3)
    code, out, _ = run(capsys, "pitch", analog, "A3")
    assert code == 0
    assert loads(out).tone[0].partials[0][0] == pytest.approx(220.0)
    code, _, err = run(capsys, "pitch", analog, "up")
    assert code == BAD_INPUT
    assert "cannot pitch it up" in err
    assert "expected semitones" in err
    code, _, err = run(capsys, "pitch", analog, "+24")  # the bands would pass 20 kHz
    assert code == BAD_INPUT
    assert "cannot pitch it +24" in err
    assert "band_hz" in err


def test_stretch(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Stretch every curve after its peak, or one curve after a time, and refuse a curve the patch lacks."""
    code, out, _ = run(capsys, "stretch", example, "all", "0.8")
    assert code == 0
    before, after = loads(EXAMPLE), loads(out)
    assert after.noise[0].amp[:2] == before.noise[0].amp[:2]  # the rise, up to the peak, is kept
    assert after.noise[0].amp[2][0] == pytest.approx(0.3 + 0.8 * 7.7)
    code, out, _ = run(capsys, "stretch", example, "noise[1].amp", "2", "--after", "8")
    assert code == 0
    assert loads(out).noise[0].amp[-1][0] == 8 + 2 * 112
    assert loads(out).noise[1] == before.noise[1]
    code, _, err = run(capsys, "stretch", example, "noise[3].amp", "2")
    assert code == BAD_INPUT
    assert "not a curve of this patch" in err


def test_choke(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Choke a patch at a time with a release, and refuse a negative time."""
    code, out, _ = run(capsys, "choke", example, "60", "--release", "30")
    assert code == 0
    q = loads(out)
    assert q.damper.amp[0] == (60.0, 1.0)
    assert q.damper.amp[-1] == (CHOKED_MS, 0.0)
    assert q.length_ms == CHOKED_MS
    code, _, err = run(capsys, "choke", example, "-1")
    assert code == BAD_INPUT
    assert "cannot choke it at -1 ms" in err


def test_softer(capsys: pytest.CaptureFixture[str], example: Path, tmp_path: Path) -> None:
    """Strike a patch softer, with contact or without, refuse too soft, and compose 6 and 6 dB into 12."""
    code, out, _ = run(capsys, "softer", example, 12)
    assert code == 0
    q = loads(out)
    v = 10 ** (-12 / 20)
    assert [s.level_db for _, s in q.sources] == [-14.0, -22.0]
    assert q.noise[0].band_hz == (6000.0, round(16000 * v**0.2, 1))
    code, out, _ = run(capsys, "softer", example, 12, "--contact", 0)
    assert code == 0
    assert loads(out).noise[0].amp == loads(EXAMPLE).noise[0].amp
    code, _, err = run(capsys, "softer", example, 90)
    assert code == BAD_INPUT
    assert "cannot make it 90 dB softer" in err
    assert "is outside" in err
    # Softer layers compose: 6 then 6 is 12, up to the decimal rounding it writes.
    six = tmp_path / "six.toml"
    six.write_text(run(capsys, "softer", example, 6)[1])
    twice = loads(run(capsys, "softer", six, 6)[1])
    assert [s.level_db for _, s in twice.sources] == [s.level_db for _, s in q.sources]
    assert abs(twice.noise[0].band_hz[1] - q.noise[0].band_hz[1]) <= COMPOSED_HZ


def test_the_report_explains_the_domain(capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Path) -> None:
    """Report what the predicted aliasing comes from, and the drive headroom."""
    code, text, _ = run(capsys, "render", example, "--out", tmp_path / "d.wav")
    assert code == 0
    assert "LUFS" in text
    assert "mostly kinks" in text
    assert "drive headroom +" in text


def test_arguments_are_checked_before_anything_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse non-finite or non-numeric arguments before running anything."""
    for args in (
        ("stretch", "x.toml", "all", "inf"),
        ("choke", "x.toml", "nan"),
        ("softer", "x.toml", "loud"),
    ):
        with pytest.raises(SystemExit) as e:
            cli.main(list(args))
        assert e.value.code == BAD_INPUT
        assert "expected a" in capsys.readouterr().err


def test_verify(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Verify a patch and its takes against the reference, and fail one outside the domain."""
    code, out, _ = run(capsys, "verify", example)
    assert code == 0
    assert "PASS" in out
    assert "E_rel" in out
    assert "6.144 and 12.288 MHz" in out
    code, out, _ = run(capsys, "verify", example, "--takes", 2)
    assert code == 0
    assert "take 2" in out
    assert "2 of 2 takes pass" in out
    hot = example.with_name("hot.toml")
    hot.write_text(EXAMPLE.replace("level_db   = -2.0", "level_db   = 6.0").replace("db = 0.0 ", "db = 24.0 "))
    code, out, _ = run(capsys, "verify", hot)
    assert code == REFUSED
    assert "FAIL" in out
    assert "OUTSIDE" in out


def test_audition_arguments(capsys: pytest.CaptureFixture[str], example: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audition with a grid, a pattern, chokes, a release, stereo and takes; refuse bad arguments."""
    seen: dict[str, object] = {}

    class Stream:
        def __enter__(self) -> None:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            return None

    def stream(rate: int, channels: int, fill: Callable[[int], NDArray[np.float32]]) -> Stream:
        seen.update(rate=rate, channels=channels, block=fill(4800))
        return Stream()

    monkeypatch.setattr(audition, "_sounddevice_stream", stream)
    other = example.with_name("open.toml")
    other.write_text(cli.reference_text("open"))
    args = ["audition", example, other, "--bpm", "120", "--every", "1/8", "--pattern", "1,1,2,0", "--choke", "2:1"]
    code, _, _ = run(capsys, *args, "--release", "30", "--stereo", "--takes", "2")
    assert code == 0
    assert seen["rate"] == RATE
    assert seen["channels"] == len(("left", "right"))
    block = seen["block"]
    assert isinstance(block, np.ndarray)
    assert block.shape == (4800, 2)
    for bad, message in (
        (["--every", "1/64"], "note values run from 1/32 to 4/1"),
        (["--pattern", "1,x"], "expected patch numbers per step"),
        (["--choke", "2-1"], "expected A:B,C"),
        (["--bpm", "0"], "--bpm must be positive"),
    ):
        with pytest.raises(SystemExit):
            cli.main(["audition", str(example), *bad])
        assert message in capsys.readouterr().err
    code, _, err = run(capsys, "audition", example, "--pattern", "1,0")
    assert code == BAD_INPUT
    assert "--pattern needs --bpm" in err
