# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""WAV files: float and PCM, mono and stereo, read back by independent readers, with their provenance (plan §7.2)."""

import dataclasses
import json
import math
import struct
import warnings
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile
from numpy.typing import NDArray
from scipy.io import wavfile

from brightwork_synth import __version__, render, spectrum, wav
from brightwork_synth.patch import SAMPLE_RATES, Patch, dumps, loads

from .conftest import EXAMPLE, make, make_tone, meta_of

type Chunk = tuple[bytes, int, bytes, int]  # id, size, payload, offset
type Exported = tuple[Path, NDArray[np.float32], render.Report]

RATE = 48000
FLOAT_FMT = 18  # bytes in a float fmt chunk: 16 and an empty extension
SMPL = 36  # bytes in a smpl chunk with no loops
PERIOD_NS = 20833  # the sample period at 48 kHz
G3 = 55  # the MIDI note under 205.25 Hz
INTERNAL_RATE, FACTOR = 1536000, 32  # the 48 kHz family's
DITHERED_LSB = 1.5  # dither of ±1 LSB and rounding of ½: the error stays below
MOMENTS = 0.006  # how close the dithered error's mean is to 0, and its power to 1/4 LSB²


def walk(path: Path) -> list[Chunk]:
    """Walk a RIFF file's chunks independently: (id, size, payload, offset) in file order."""
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    pos, out = 12, list[Chunk]()
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        out.append((cid, size, data[pos + 8 : pos + 8 + size], pos))
        pos += 8 + size + (size & 1)
    assert pos == len(data)
    return out


def export(path: Path, p: Patch, *, stereo: bool = False) -> tuple[NDArray[np.float32], render.Report]:
    """Render a patch's takes, and export the first file."""
    audio, report = render.render_takes(p, 1, stereo=stereo)
    render.export(path, audio[0], p, report, report.files[0])
    return audio[0], report


@pytest.fixture
def exported(tmp_path: Path) -> Exported:
    """Export the example with the plan's tone, and return the path, its audio and its report."""
    p = make_tone()
    path = tmp_path / "hat.wav"
    audio, report = export(path, p)
    return path, audio, report


def test_independent_readers_get_the_exact_samples(exported: Exported) -> None:
    """Read the export with soundfile, SciPy and our reader, and find the exact samples."""
    path, audio, _ = exported
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == RATE
    assert np.array_equal(data, audio)
    info = soundfile.info(path)
    assert info.subtype == "FLOAT"
    assert info.channels == 1
    assert info.frames == audio.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # it does not know smpl or cymb
        rate, data = wavfile.read(path)
    assert rate == RATE
    assert data.dtype == np.float32
    assert np.array_equal(data, audio)
    ours = wav.read(path)
    assert ours.rate == RATE
    assert np.array_equal(ours.samples, audio)
    assert ours.channels == 1


def test_headers(exported: Exported) -> None:
    """Write fmt, fact, smpl, cymb and data in that order, with float32's header."""
    path, audio, _ = exported
    chunks = walk(path)
    assert [c[0] for c in chunks] == [b"fmt ", b"fact", b"smpl", b"cymb", b"data"]
    fmt = chunks[0]
    assert fmt[1] == FLOAT_FMT
    assert struct.unpack("<HHIIHHH", fmt[2]) == (3, 1, 48000, 192000, 4, 32, 0)
    assert struct.unpack("<I", chunks[1][2]) == (audio.size,)
    assert chunks[4][1] == 4 * audio.size


def test_smpl_chunk_maps_tone_1s_lowest_partial(exported: Exported) -> None:
    """Write tone[1]'s lowest partial, as the grid plays it, as the smpl chunk's root note and fraction."""
    path, _, _ = exported
    smpl = {c[0]: c[2] for c in walk(path)}[b"smpl"]
    fields = struct.unpack("<9I", smpl)
    assert len(smpl) == SMPL
    assert fields[2] == PERIOD_NS
    note, fraction = fields[3], fields[4]
    assert note == G3  # 205.25 Hz, as the grid plays 205.3 Hz: G#3 −20.1 cents, so G3 and 0.8 of a semitone up
    assert note + fraction / 2**32 == pytest.approx(69 + 12 * np.log2(205.25 / 440), abs=1e-9)
    assert fields[7] == 0
    assert fields[8] == 0


def test_an_unpitched_patch_has_no_root_key(tmp_path: Path) -> None:
    """Leave out the smpl chunk for noise alone, or a silent tone."""
    for p in (loads(EXAMPLE), make_tone(**{"tone[1].level_db": -math.inf})):
        path = tmp_path / "noise.wav"
        export(path, p)
        assert [c[0] for c in walk(path)] == [b"fmt ", b"fact", b"cymb", b"data"]
        assert wav.read(path).root is None


@pytest.mark.parametrize(
    ("hz", "note", "cents"),
    [
        (440.0, 69, 0.0),
        (146.832, 50, 0.0),
        (185.0, 54, 0.03),
        (1800.0, 93, 38.9),
        (41.19, 27, 99.4),
        (4000.0, 107, 21.3),
    ],
)
def test_root_note(hz: float, note: int, cents: float) -> None:
    """Name a frequency's root note, a pitch less than half a cent flat counting as the note above."""
    n, fraction = wav.root_note(hz)
    assert n == note
    assert 100 * fraction / 2**32 == pytest.approx(cents, abs=0.06)
    assert 0 <= fraction < 2**32


def test_provenance(exported: Exported) -> None:
    """Record the patch, the renderer, seed, takes, gains, filters and library versions in the cymb chunk."""
    path, _, _ = exported
    meta = meta_of(path)
    p = make_tone()
    assert meta["patch"] == dumps(p)
    assert loads(meta["patch"]) == p
    assert meta["renderer"] == f"brightwork {__version__}"
    assert (meta["seed"], meta["takes"], meta["files"]) == (1, [1], 1)
    assert meta["set_gain_db"] == pytest.approx(-6.0)
    assert meta["file_gain_db"] == 0.0
    assert meta["decimator"]["sha256"] == render.decimator(48000).sha256
    assert meta["decimator"]["internal_rate"] == INTERNAL_RATE
    assert meta["decimator"]["factor"] == FACTOR
    assert meta["interpolator"] == {"sha256": render.interpolator_for(48000).sha256, "table_rate": 96000}
    assert meta["interpolator"]["sha256"] is not None
    assert {"python", "numpy", "scipy"} <= set(meta)
    raw = {c[0]: c[2] for c in walk(path)}[b"cymb"]
    assert json.loads(raw.decode("utf-8")) == meta


def test_each_file_records_its_own_patch_and_gains(tmp_path: Path) -> None:
    """Record each file's own take and gains, from which its patch renders its samples again."""
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = render.render_takes(p, 3, level_match=True)
    for a, f in zip(audio, report.files, strict=True):
        path = tmp_path / f"t{f.number}.wav"
        render.export(path, a, p, report, f)
        meta = meta_of(path)
        take = report.takes[f.takes[0]].take
        assert loads(meta["patch"]) == dataclasses.replace(p, take=take)
        assert take == p.take + f.number - 1
        assert (meta["takes"], meta["files"]) == ([take], len(report.files))
        assert meta["set_gain_db"] == report.set_gain_db
        assert meta["file_gain_db"] == f.gain_db
        # The file holds its gains: its patch rendered alone and scaled by them gives the same samples.
        s = render.synthesize(loads(meta["patch"]))
        gain = 10 ** ((meta["set_gain_db"] + meta["file_gain_db"]) / 20)
        np.testing.assert_allclose(wav.read(path).samples, (s.out * gain).astype(np.float32), rtol=1e-6, atol=1e-9)


def test_stereo_files_interleave_their_takes(tmp_path: Path) -> None:
    """Interleave a pair's takes as left and right, its patch the left channel's, its provenance both takes."""
    p = make_tone(take=4)
    path = tmp_path / "pair.wav"
    audio, report = export(path, p, stereo=True)
    assert audio.shape == (report.samples, 2)
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == RATE
    assert data.shape == audio.shape
    assert np.array_equal(data, audio)
    info = soundfile.info(path)
    assert info.channels == len(("left", "right"))
    assert info.frames == report.samples
    chunks = {c[0]: c[2] for c in walk(path)}
    assert struct.unpack("<HHIIHHH", chunks[b"fmt "]) == (3, 2, 48000, 384000, 8, 32, 0)
    assert struct.unpack("<I", chunks[b"fact"]) == (report.samples,)
    ours = wav.read(path)
    assert ours.channels == info.channels
    assert np.array_equal(ours.samples, audio)
    # The embedded patch renders the left channel; the provenance names both takes.
    meta = meta_of(path)
    assert loads(meta["patch"]) == p
    assert meta["takes"] == [4, 5]
    assert np.array_equal(render.render(loads(meta["patch"]))[0], audio[:, 0])


def test_riff_padding(tmp_path: Path) -> None:
    """Pad an odd-length chunk so the data stays word-aligned."""
    # An odd-length private chunk is padded so the data chunk stays word-aligned.
    for text in ("x", "xy", "π"):
        path = tmp_path / "pad.wav"
        samples = np.linspace(-1, 1, 7, dtype=np.float32)
        wav.write(path, samples, wav.Format(44100), meta={"patch": text})
        chunks = walk(path)
        private = next(c for c in chunks if c[0] == b"cymb")
        data = next(c for c in chunks if c[0] == b"data")
        assert data[3] % 2 == 0
        if private[1] % 2:
            raw = Path(path).read_bytes()
            assert raw[private[3] + 8 + private[1]] == 0
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], samples)
        assert wav.read(path).meta == {"patch": text}


def test_reads_beds_and_samples_but_rejects_other_formats(tmp_path: Path) -> None:
    """Read stereo PCM and float files, plain or extensible, and refuse every other format."""
    stereo = np.stack([np.linspace(-0.5, 0.5, 10), np.linspace(0.5, -0.5, 10)], axis=1)
    for subtype, bits in (("PCM_16", 16), ("PCM_24", 24), ("FLOAT", None)):
        for fmt in ("WAV", "WAVEX"):
            path = tmp_path / f"stereo-{subtype}-{fmt}.wav"
            soundfile.write(path, stereo, 44100, subtype=subtype, format=fmt)
            w = wav.read(path)
            assert w.channels == stereo.shape[1]
            assert w.samples.shape == (10, 2)
            assert w.bits == bits
            np.testing.assert_allclose(w.as_float(), soundfile.read(path)[0], atol=1e-9)
    for name, data, subtype in (
        ("pcm32", np.zeros(10), "PCM_32"),
        ("double", np.zeros(10), "DOUBLE"),
        ("u8", np.zeros(10), "PCM_U8"),
    ):
        path = tmp_path / f"{name}.wav"
        soundfile.write(path, data, 48000, subtype=subtype)
        with pytest.raises(ValueError, match="expected 32-bit float or 16/24-bit PCM"):
            wav.read(path)
        assert b"fmt " in wav.chunks(path)
    with pytest.raises(ValueError, match="expected a float32 buffer"):
        wav.write(tmp_path / "x.wav", np.zeros(4), wav.Format(48000))
    with pytest.raises(ValueError, match="16 or 24 bits"):
        wav.write(tmp_path / "x.wav", np.zeros(4, dtype=np.int32), wav.Format(48000, 20))
    with pytest.raises(ValueError, match="out of range for 16-bit PCM"):
        wav.write(tmp_path / "x.wav", np.array([2**15], dtype=np.int32), wav.Format(48000, 16))
    with pytest.raises(ValueError, match="mono buffer or a"):
        wav.write(tmp_path / "x.wav", np.zeros((4, 3), dtype=np.float32), wav.Format(48000))


def test_refuses_what_is_not_a_readable_wave(tmp_path: Path) -> None:
    """Refuse a file that is not RIFF WAVE, one without its format or samples, and an impossible frame."""
    path = tmp_path / "x.wav"
    path.write_bytes(b"RIFX\x04\x00\x00\x00WAVE")
    with pytest.raises(ValueError, match="not a RIFF WAVE file"):
        wav.read(path)
    path.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
    with pytest.raises(ValueError, match="missing fmt or data chunk"):
        wav.read(path)
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 0, 0, 12)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 0)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(ValueError, match=r"unsupported format \(12-bit samples, 1 channel\)"):
        wav.read(path)


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        (((b"fmt ", b"\1\0"), (b"data", b"\0\0")), "a fmt or smpl chunk is cut short"),
        (((b"smpl", b"\0"),), "a fmt or smpl chunk is cut short"),
        (((wav.PRIVATE, b"{"),), "the embedded patch is damaged"),
        (((wav.PRIVATE, b"[" * 100_000),), "the embedded patch is damaged"),
        (((wav.PRIVATE, b"[1]"),), "the embedded patch is damaged"),
    ],
)
def test_refuses_a_damaged_chunk(tmp_path: Path, chunks: tuple[tuple[bytes, bytes], ...], message: str) -> None:
    """Refuse a fmt or smpl chunk cut short, and an embedded patch garbled, nested too deep or not a table."""
    fmt = struct.pack("<HHIIHH", wav.PCM, 1, 48000, 96000, 2, 16)
    body = b"".join(c + struct.pack("<I", len(d)) + d + b"\0" * (len(d) % 2) for c, d in (*chunks, (b"fmt ", fmt)))
    body += b"data" + struct.pack("<I", 2) + b"\0\0"
    path = tmp_path / "x.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    with pytest.raises(ValueError, match=message):
        wav.read(path)


def test_every_rate_writes_a_valid_file(tmp_path: Path) -> None:
    """Write a valid file at every export rate."""
    for rate in SAMPLE_RATES:
        path = tmp_path / f"{rate}.wav"
        audio, _ = export(path, make(**{"output.sample_rate": rate}))
        data, sr = soundfile.read(path, dtype="float32")
        assert sr == rate
        assert np.array_equal(data, audio)


@pytest.mark.parametrize(("fmt", "bits", "subtype"), [("pcm24", 24, "PCM_24"), ("pcm16", 16, "PCM_16")])
def test_pcm_exports_read_back_exactly(tmp_path: Path, fmt: str, bits: int, subtype: str) -> None:
    """Write dithered PCM that three readers read back exactly, with its dither recorded, the same every time."""
    p = make_tone(**{"output.format": fmt})
    path = tmp_path / f"{fmt}.wav"
    audio, _ = export(path, p)
    expected = render.quantize(audio, bits, p.seed, p.take)
    info = soundfile.info(path)
    assert info.subtype == subtype
    assert info.channels == 1
    assert info.frames == audio.size
    data, rate = soundfile.read(path, dtype="int32")  # left-justified in 32 bits
    assert rate == RATE
    assert np.array_equal(data >> (32 - bits), expected)
    with wave.open(str(path)) as w:
        assert (w.getsampwidth(), w.getnchannels(), w.getframerate()) == (bits // 8, 1, 48000)
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, bits // 8).astype(np.int32)
    ints = sum(raw[:, i] << (8 * i) for i in range(bits // 8))
    assert np.array_equal((ints << (32 - bits)) >> (32 - bits), expected)
    ours = wav.read(path)
    assert ours.bits == bits
    assert np.array_equal(ours.samples, expected)
    # The error is the dither plus the rounding, and the silent first sample stays zero.
    x = audio.astype(float) * 2 ** (bits - 1)
    assert np.max(np.abs(expected - x)) < DITHERED_LSB
    assert expected[0] == 0
    assert [c[0] for c in walk(path)] == [b"fmt ", b"smpl", b"cymb", b"data"]  # no fact chunk for PCM
    meta = meta_of(path)
    assert loads(meta["patch"]) == p
    assert meta["pcm"] == {
        "bits": bits,
        "dither": "TPDF, ±1 LSB; exact zeros kept",
        "dither_key": "cymbal/dither/v1",
        "seed": 1,
        "takes": [1],
    }
    # The same patch writes the same bytes again.
    export(tmp_path / "again.wav", p)
    assert Path(path).read_bytes() == (tmp_path / "again.wav").read_bytes()


def test_stereo_pcm_dithers_each_channel_with_its_own_take(tmp_path: Path) -> None:
    """Dither each channel of a PCM pair with its own take."""
    p = make(**{"output.format": "pcm24"})
    path = tmp_path / "pair24.wav"
    audio, _ = export(path, p, stereo=True)
    ours = wav.read(path)
    assert ours.bits == wav.PCM_BITS[1]
    assert ours.channels == len(("left", "right"))
    for c, take in enumerate((1, 2)):
        assert np.array_equal(ours.samples[:, c], render.quantize(audio[:, c], 24, p.seed, take))
    assert meta_of(path)["pcm"]["takes"] == [1, 2]
    assert soundfile.info(path).subtype == "PCM_24"
    assert soundfile.info(path).channels == ours.channels


def test_exact_zeros_stay_zero() -> None:
    """Leave exact zeros at zero through the dither."""
    x = np.zeros(1000, dtype=np.float32)
    x[500:510] = 0.25
    q = render.quantize(x, 16, 1, 1)
    assert np.all(q[:500] == 0)
    assert np.all(q[510:] == 0)
    assert np.all(np.abs(q[500:510] - 8192) <= 1)


def test_24_bit_packing_round_trips(tmp_path: Path) -> None:
    """Pack 24-bit samples, extremes included, and read them back."""
    values = np.array([-(2**23), 2**23 - 1, -1, 0, 1, 12345, -54321, 2**22], dtype=np.int32)
    path = tmp_path / "pack.wav"
    wav.write(path, values, wav.Format(48000, 24))
    assert np.array_equal(wav.read(path).samples, values)
    assert np.array_equal(soundfile.read(path, dtype="int32")[0] >> 8, values)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75])
def test_dither_leaves_no_noise_modulation(fraction: float) -> None:
    """Keep the error's mean at 0 and its power at 1/4 LSB², wherever the signal lies between steps."""
    # With TPDF dither the total error has mean 0 and power 1/4 LSB² wherever the signal lies between
    # two steps: the first two moments of the error do not depend on the signal.
    x = np.full(200_000, (1000 + fraction) / 2**15, dtype=np.float32)
    e = render.quantize(x, 16, 3, 1) - x.astype(float) * 2**15
    assert abs(e.mean()) < MOMENTS
    assert e.var() == pytest.approx(0.25, abs=MOMENTS)


def test_dither_is_bit_identical_per_seed_and_take() -> None:
    """Dither the same way for one seed and take, and differently for another."""
    x = (0.3 * np.sin(np.arange(10000) / 7)).astype(np.float32)
    assert np.array_equal(render.quantize(x, 16, 5, 1), render.quantize(x, 16, 5, 1))
    assert not np.array_equal(render.quantize(x, 16, 5, 1), render.quantize(x, 16, 6, 1))
    assert not np.array_equal(render.quantize(x, 16, 5, 1), render.quantize(x, 16, 5, 2))
    assert spectrum.DITHER_KEY == b"cymbal/dither/v1"


def test_pcm_keeps_room_for_dither() -> None:
    """Keep a PCM render 2 LSB below full scale in peak mode, so its dithered samples always fit."""
    for fmt, bits in (("pcm16", 16), ("pcm24", 24)):
        p = make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0, "output.format": fmt})
        audio, report = render.render(p)
        assert np.max(np.abs(audio)) <= render.ceiling(fmt) * (1 + 2**-24)
        assert not report.over
        q = render.quantize(audio, bits, p.seed, p.take)
        assert q.max() <= 2 ** (bits - 1) - 1
        assert q.min() >= -(2 ** (bits - 1))
    # Float keeps its full-scale target; a gain-mode render between the ceiling and full scale is over for PCM.
    assert np.max(
        np.abs(render.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[0]),
    ) == pytest.approx(1.0)
    unity = np.max(np.abs(render.synthesize(make()).out))
    gain_db = 20 * math.log10((1 - 2**-15) / unity)  # half an LSB below full scale: fine as float, over as pcm16
    assert not render.render(make(**{"output.db": round(gain_db, 6)}))[1].over
    over = render.render(make(**{"output.db": round(gain_db, 6), "output.format": "pcm16"}))[1]
    assert over.over
    assert "16-bit ceiling" in str(over)
