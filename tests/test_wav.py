# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import json
import math
import struct
import warnings

import numpy as np
import pytest
import soundfile
from conftest import EXAMPLE, make
from scipy.io import wavfile

from brightwork_synth import __version__, spectrum, wav
from brightwork_synth import render as R
from brightwork_synth.patch import dumps, loads


def walk(path):
    """An independent chunk walker: (id, size, payload, offset) in file order."""
    data = open(path, "rb").read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    pos, out = 12, []
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        out.append((cid, size, data[pos + 8 : pos + 8 + size], pos))
        pos += 8 + size + (size & 1)
    assert pos == len(data)
    return out


def export(path, p, takes=1, stereo=False):
    audio, report = R.render_takes(p, takes, stereo=stereo)
    R.export(path, audio[0], p, report, report.files[0])
    return audio[0], report


@pytest.fixture
def exported(tmp_path):
    p = make(tone=True)
    path = tmp_path / "hat.wav"
    audio, report = export(path, p)
    return path, audio, report


def test_independent_readers_get_the_exact_samples(exported):
    path, audio, _ = exported
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == 48000 and np.array_equal(data, audio)
    info = soundfile.info(path)
    assert info.subtype == "FLOAT" and info.channels == 1 and info.frames == audio.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # it does not know smpl or cymb
        rate, data = wavfile.read(path)
    assert rate == 48000 and data.dtype == np.float32 and np.array_equal(data, audio)
    ours = wav.read(path)
    assert ours.rate == 48000 and np.array_equal(ours.samples, audio) and ours.channels == 1


def test_headers(exported):
    path, audio, _ = exported
    chunks = walk(path)
    assert [c[0] for c in chunks] == [b"fmt ", b"fact", b"smpl", b"cymb", b"data"]
    fmt = chunks[0]
    assert fmt[1] == 18
    assert struct.unpack("<HHIIHHH", fmt[2]) == (3, 1, 48000, 192000, 4, 32, 0)
    assert struct.unpack("<I", chunks[1][2]) == (audio.size,)
    assert chunks[4][1] == 4 * audio.size


def test_smpl_chunk_maps_tone_1s_lowest_partial(exported):
    path, _, _ = exported
    smpl = dict((c[0], c[2]) for c in walk(path))[b"smpl"]
    fields = struct.unpack("<9I", smpl)
    assert len(smpl) == 36
    assert fields[2] == 20833  # the sample period in ns at 48 kHz
    note, fraction = fields[3], fields[4]
    assert note == 55  # 205.25 Hz, as the grid plays 205.3 Hz: G#3 −20.1 cents, so G3 and 0.8 of a semitone up
    assert note + fraction / 2**32 == pytest.approx(69 + 12 * np.log2(205.25 / 440), abs=1e-9)
    assert fields[7] == 0 and fields[8] == 0  # no loops, no sampler data


def test_an_unpitched_patch_has_no_root_key(tmp_path):
    for p in (loads(EXAMPLE), make(tone=True, **{"tone[1].level_db": -math.inf})):
        path = tmp_path / "noise.wav"
        export(path, p)
        assert [c[0] for c in walk(path)] == [b"fmt ", b"fact", b"cymb", b"data"]
        assert wav.read(path).root is None


@pytest.mark.parametrize(
    "hz, note, cents",
    [
        (440.0, 69, 0.0),
        (146.832, 50, 0.0),
        (185.0, 54, 0.03),
        (1800.0, 93, 38.9),
        (41.19, 27, 99.4),
        (4000.0, 107, 21.3),
    ],
)
def test_root_note(hz, note, cents):
    n, fraction = wav.root_note(hz)
    assert n == note
    assert 100 * fraction / 2**32 == pytest.approx(cents, abs=0.06)
    assert 0 <= fraction < 2**32


def test_provenance(exported):
    path, _, report = exported
    meta = wav.read(path).meta
    p = make(tone=True)
    assert meta["patch"] == dumps(p) and loads(meta["patch"]) == p
    assert meta["renderer"] == f"brightwork {__version__}"
    assert (meta["seed"], meta["takes"], meta["files"]) == (1, [1], 1)
    assert meta["set_gain_db"] == pytest.approx(-6.0) and meta["file_gain_db"] == 0.0
    assert meta["decimator"]["sha256"] == R.decimator(48000).sha256
    assert meta["decimator"]["internal_rate"] == 1536000 and meta["decimator"]["factor"] == 32
    assert meta["interpolator"] == {"sha256": R.interpolator_for(48000).sha256, "table_rate": 96000}
    assert meta["interpolator"]["sha256"] is not None
    assert {"python", "numpy", "scipy"} <= set(meta)
    raw = dict((c[0], c[2]) for c in walk(path))[b"cymb"]
    assert json.loads(raw.decode("utf-8")) == meta


def test_each_file_records_its_own_patch_and_gains(tmp_path):
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = R.render_takes(p, 3, level_match=True)
    for a, f in zip(audio, report.files):
        path = tmp_path / f"t{f.number}.wav"
        R.export(path, a, p, report, f)
        meta = wav.read(path).meta
        take = report.takes[f.takes[0]].take
        assert loads(meta["patch"]) == dataclasses.replace(p, take=take) and take == p.take + f.number - 1
        assert (meta["takes"], meta["files"]) == ([take], 3)
        assert meta["set_gain_db"] == report.set_gain_db and meta["file_gain_db"] == f.gain_db
        # The file holds its gains: its patch rendered alone and scaled by them gives the same samples.
        s = R.synthesize(loads(meta["patch"]))
        gain = 10 ** ((meta["set_gain_db"] + meta["file_gain_db"]) / 20)
        np.testing.assert_allclose(wav.read(path).samples, (s.out * gain).astype(np.float32), rtol=1e-6, atol=1e-9)


def test_stereo_files_interleave_their_takes(tmp_path):
    p = make(tone=True, take=4)
    path = tmp_path / "pair.wav"
    audio, report = export(path, p, stereo=True)
    assert audio.shape == (report.samples, 2)
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == 48000 and data.shape == audio.shape and np.array_equal(data, audio)
    info = soundfile.info(path)
    assert info.channels == 2 and info.frames == report.samples
    chunks = dict((c[0], c[2]) for c in walk(path))
    assert struct.unpack("<HHIIHHH", chunks[b"fmt "]) == (3, 2, 48000, 384000, 8, 32, 0)
    assert struct.unpack("<I", chunks[b"fact"]) == (report.samples,)
    ours = wav.read(path)
    assert ours.channels == 2 and np.array_equal(ours.samples, audio)
    # The embedded patch renders the left channel; the provenance names both takes.
    meta = ours.meta
    assert loads(meta["patch"]) == p and meta["takes"] == [4, 5]
    assert np.array_equal(R.render(loads(meta["patch"]))[0], audio[:, 0])


def test_riff_padding(tmp_path):
    # An odd-length private chunk is padded so the data chunk stays word-aligned.
    for text in ("x", "xy", "π"):
        path = tmp_path / "pad.wav"
        samples = np.linspace(-1, 1, 7, dtype=np.float32)
        wav.write(path, samples, 44100, meta={"patch": text})
        chunks = walk(path)
        private = next(c for c in chunks if c[0] == b"cymb")
        data = next(c for c in chunks if c[0] == b"data")
        assert data[3] % 2 == 0
        if private[1] % 2:
            raw = open(path, "rb").read()
            assert raw[private[3] + 8 + private[1]] == 0
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], samples)
        assert wav.read(path).meta == {"patch": text}


def test_reads_beds_and_samples_but_rejects_other_formats(tmp_path):
    stereo = np.stack([np.linspace(-0.5, 0.5, 10), np.linspace(0.5, -0.5, 10)], axis=1)
    for subtype, bits in (("PCM_16", 16), ("PCM_24", 24), ("FLOAT", None)):
        for fmt in ("WAV", "WAVEX"):
            path = tmp_path / f"stereo-{subtype}-{fmt}.wav"
            soundfile.write(path, stereo, 44100, subtype=subtype, format=fmt)
            w = wav.read(path)
            assert w.channels == 2 and w.samples.shape == (10, 2) and w.bits == bits
            np.testing.assert_allclose(w.as_float(), soundfile.read(path)[0], atol=1e-9)
    for name, data, subtype in (
        ("pcm32", np.zeros(10), "PCM_32"),
        ("double", np.zeros(10), "DOUBLE"),
        ("u8", np.zeros(10), "PCM_U8"),
    ):
        path = tmp_path / f"{name}.wav"
        soundfile.write(path, data, 48000, subtype=subtype)
        with pytest.raises(ValueError):
            wav.read(path)
        assert b"fmt " in wav.chunks(path)
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.zeros(4), 48000)  # float64 is not a format
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.zeros(4, dtype=np.int32), 48000, bits=20)
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.array([2**15], dtype=np.int32), 48000, bits=16)  # out of range
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.zeros((4, 3), dtype=np.float32), 48000)  # mono or stereo only


def test_every_rate_writes_a_valid_file(tmp_path):
    for rate in (44100, 48000, 88200, 96000):
        path = tmp_path / f"{rate}.wav"
        audio, _ = export(path, make(**{"output.sample_rate": rate}))
        data, sr = soundfile.read(path, dtype="float32")
        assert sr == rate and np.array_equal(data, audio)


@pytest.mark.parametrize("fmt, bits, subtype", [("pcm24", 24, "PCM_24"), ("pcm16", 16, "PCM_16")])
def test_pcm_exports_read_back_exactly(tmp_path, fmt, bits, subtype):
    import wave

    p = make(tone=True, **{"output.format": fmt})
    path = tmp_path / f"{fmt}.wav"
    audio, report = export(path, p)
    expected = R.quantize(audio, bits, p.seed, p.take)
    info = soundfile.info(path)
    assert info.subtype == subtype and info.channels == 1 and info.frames == audio.size
    data, rate = soundfile.read(path, dtype="int32")  # left-justified in 32 bits
    assert rate == 48000 and np.array_equal(data >> (32 - bits), expected)
    with wave.open(str(path)) as w:
        assert (w.getsampwidth(), w.getnchannels(), w.getframerate()) == (bits // 8, 1, 48000)
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, bits // 8).astype(np.int32)
    ints = sum(raw[:, i] << (8 * i) for i in range(bits // 8))
    assert np.array_equal((ints << (32 - bits)) >> (32 - bits), expected)
    ours = wav.read(path)
    assert ours.bits == bits and np.array_equal(ours.samples, expected)
    # The error is the dither plus the rounding, and the silent first sample stays zero.
    x = audio.astype(float) * 2 ** (bits - 1)
    assert np.max(np.abs(expected - x)) < 1.5 and expected[0] == 0
    assert [c[0] for c in walk(path)] == [b"fmt ", b"smpl", b"cymb", b"data"]  # no fact chunk for PCM
    meta = ours.meta
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
    assert open(path, "rb").read() == open(tmp_path / "again.wav", "rb").read()


def test_stereo_pcm_dithers_each_channel_with_its_own_take(tmp_path):
    p = make(**{"output.format": "pcm24"})
    path = tmp_path / "pair24.wav"
    audio, report = export(path, p, stereo=True)
    ours = wav.read(path)
    assert ours.bits == 24 and ours.channels == 2
    for c, take in enumerate((1, 2)):
        assert np.array_equal(ours.samples[:, c], R.quantize(audio[:, c], 24, p.seed, take))
    assert ours.meta["pcm"]["takes"] == [1, 2]
    assert soundfile.info(path).subtype == "PCM_24" and soundfile.info(path).channels == 2


def test_exact_zeros_stay_zero():
    x = np.zeros(1000, dtype=np.float32)
    x[500:510] = 0.25
    q = R.quantize(x, 16, 1, 1)
    assert np.all(q[:500] == 0) and np.all(q[510:] == 0) and np.all(np.abs(q[500:510] - 8192) <= 1)


def test_24_bit_packing_round_trips(tmp_path):
    values = np.array([-(2**23), 2**23 - 1, -1, 0, 1, 12345, -54321, 2**22], dtype=np.int32)
    path = tmp_path / "pack.wav"
    wav.write(path, values, 48000, bits=24)
    assert np.array_equal(wav.read(path).samples, values)
    assert np.array_equal(soundfile.read(path, dtype="int32")[0] >> 8, values)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75])
def test_dither_leaves_no_noise_modulation(fraction):
    # With TPDF dither the total error has mean 0 and power 1/4 LSB² wherever the signal lies between
    # two steps: the first two moments of the error do not depend on the signal.
    x = np.full(200_000, (1000 + fraction) / 2**15, dtype=np.float32)
    e = R.quantize(x, 16, 3, 1) - x.astype(float) * 2**15
    assert abs(e.mean()) < 0.006 and e.var() == pytest.approx(0.25, abs=0.006)


def test_dither_is_bit_identical_per_seed_and_take():
    x = (0.3 * np.sin(np.arange(10000) / 7)).astype(np.float32)
    assert np.array_equal(R.quantize(x, 16, 5, 1), R.quantize(x, 16, 5, 1))
    assert not np.array_equal(R.quantize(x, 16, 5, 1), R.quantize(x, 16, 6, 1))
    assert not np.array_equal(R.quantize(x, 16, 5, 1), R.quantize(x, 16, 5, 2))
    assert spectrum.DITHER_KEY == b"cymbal/dither/v1"


def test_pcm_keeps_room_for_dither():
    for fmt, bits in (("pcm16", 16), ("pcm24", 24)):
        p = make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0, "output.format": fmt})
        audio, report = R.render(p)
        assert np.max(np.abs(audio)) <= R.ceiling(fmt) * (1 + 2**-24) and not report.over
        q = R.quantize(audio, bits, p.seed, p.take)
        assert q.max() <= 2 ** (bits - 1) - 1 and q.min() >= -(2 ** (bits - 1))
    # Float keeps its full-scale target; a gain-mode render between the ceiling and full scale is over for PCM.
    assert np.max(
        np.abs(R.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[0])
    ) == pytest.approx(1.0)
    unity = np.max(np.abs(R.synthesize(make()).out))
    gain_db = 20 * math.log10((1 - 2**-15) / unity)  # half an LSB below full scale: fine as float, over as pcm16
    assert not R.render(make(**{"output.db": round(gain_db, 6)}))[1].over
    over = R.render(make(**{"output.db": round(gain_db, 6), "output.format": "pcm16"}))[1]
    assert over.over and "16-bit ceiling" in str(over)
