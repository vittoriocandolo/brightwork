"""A kit played by real SFZ players (plan §10, milestone 3): sfizz and liquidsfz.

The test writes a kit, plays a list of notes through each player offline,
and checks what the player did: which file each note played (keys, velocity
layers and round robins), at what level (the square law across layers), and
how a choke faded the open hat. sfizz renders a MIDI file with its own
`sfizz_render`, writes 16-bit files and repeats one sample at the first block
boundary after a note starts; liquidsfz ships no offline renderer, so
tools/liquidsfz_render.cc drives its library. Each note is therefore
identified by correlation over a window after its first block, with the lag
searched, and levels are measured by energy, not sample by sample.

A player is used when its renderer is on the PATH or named by SFIZZ_RENDER or
LIQUIDSFZ_RENDER, and skipped otherwise.
"""

import dataclasses
import math
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest
import soundfile

from cymbal import cli
from cymbal import pack as K
from cymbal.loudness import momentary_max
from cymbal.patch import loads, softer

RATE = 48000


def vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def smf(notes, ppq=480, bpm=120.0) -> bytes:
    """A format-0 MIDI file: (time in s, key, velocity, length in s) per note, at one tempo."""
    events = []
    for t, key, velocity, length in notes:
        events += [(t, 1, 0x90, key, velocity), (t + length, 0, 0x80, key, 0)]
    events.sort()
    track, last = bytearray(b"\x00\xff\x51\x03" + round(60e6 / bpm).to_bytes(3, "big")), 0
    for t, _, status, key, velocity in events:
        tick = round(t * ppq * bpm / 60)
        track += vlq(tick - last) + bytes((status, key, velocity))
        last = tick
    track += b"\x00\xff\x2f\x00"
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq) + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def onsets(y: np.ndarray, threshold: float, gap: int) -> list[int]:
    above = np.flatnonzero(np.abs(y) > threshold)
    return [int(above[0])] + [int(above[i]) for i in range(1, above.size) if above[i] - above[i - 1] > gap]


def identify(y: np.ndarray, start: int, files: dict) -> tuple[str, float, int]:
    """The file a note plays: the best normalized correlation over 256 samples from 300 samples in, within ±12
    samples of `start`. Returns the file, the correlation and where the file starts in y."""
    best = (-1.0, "", 0)
    for name, x in files.items():
        b = x[300:556]
        for lag in range(-12, 13):
            a = y[start + lag + 300 : start + lag + 556]
            c = float(np.dot(a, b) / math.sqrt(np.dot(a, a) * np.dot(b, b)))
            best = max(best, (c, name, start + lag))
    return best[1], best[0], best[2]


def band_energy(a: np.ndarray, lo: float = 4000.0, hi: float = 8000.0) -> float:
    """Energy between lo and hi Hz, through a Hann window."""
    spectrum = np.fft.rfft(a * np.hanning(a.size))
    f = np.fft.rfftfreq(a.size, 1 / RATE)
    return float(np.sum(np.abs(spectrum[(f >= lo) & (f < hi)]) ** 2))


def level_db(a: np.ndarray, b: np.ndarray) -> float:
    """The gain of a over b, from their energies in one band: a player's own frequency response (liquidsfz's
    interpolator rolls off the top octave by about 2 dB) then cancels between notes."""
    return 10 * math.log10(max(band_energy(a), 1e-300) / band_energy(b))


def play_sfizz(binary, sfz, notes, out) -> np.ndarray:
    (out / "notes.mid").write_bytes(smf(notes))
    subprocess.run(
        [binary, "--sfz", str(sfz), "--midi", str(out / "notes.mid"), "--wav", str(out / "played.wav")]
        + ["-s", str(RATE), "-b", "256"],
        check=True,
        capture_output=True,
    )
    y, rate = soundfile.read(out / "played.wav", dtype="float64")
    assert rate == RATE
    return y


def play_liquidsfz(binary, sfz, notes, out) -> np.ndarray:
    lines = []
    for t, key, velocity, length in notes:
        lines += [f"{round(t * RATE)} {key} {velocity}", f"{round((t + length) * RATE)} {key} 0"]
    (out / "notes.txt").write_text("\n".join(lines) + "\n")
    frames = round((max(t for t, *_ in notes) + 2.0) * RATE)
    subprocess.run(
        [binary, str(sfz), str(out / "notes.txt"), str(frames), str(RATE), str(out / "played.f32")], check=True
    )
    return np.fromfile(out / "played.f32", dtype="<f4").astype(float).reshape(-1, 2)


PLAYERS = {
    "sfizz": (os.environ.get("SFIZZ_RENDER") or shutil.which("sfizz_render"), play_sfizz),
    "liquidsfz": (os.environ.get("LIQUIDSFZ_RENDER") or shutil.which("liquidsfz_render"), play_liquidsfz),
}


@pytest.fixture(scope="module", params=list(PLAYERS))
def played(request, tmp_path_factory):
    binary, play = PLAYERS[request.param]
    if not binary:
        pytest.skip(f"{request.param} is not installed")
    out = tmp_path_factory.mktemp(request.param)

    def loud(name):  # at unity gain, so the open hat's ring stands well above a player's 16-bit floor
        p = loads(cli.reference_text(name))
        return dataclasses.replace(p, output=dataclasses.replace(p.output, db=0.0))

    closed = loud("closed")
    # The softer layer takes its own strikes, takes 11 and 12, so that every file is a realization of its own and
    # each note can be told by correlation even through a player's interpolation.
    soft = dataclasses.replace(softer(closed, 12), take=11)
    groups = [(42, [("closed", closed), ("closed-12", soft)]), (46, [("open", loud("open"))])]
    kit = K.build(groups, takes=2, chokes=[(46, (42,))], stem="kit")
    K.write(kit, out / "kit")
    notes = [
        (0.5, 42, 127, 0.05),
        (1.0, 42, 127, 0.05),
        (1.5, 42, 127, 0.05),
        (2.0, 42, 40, 0.05),
        (2.5, 42, 40, 0.05),
        (3.0, 46, 127, 0.05),
        (3.4, 42, 1, 0.05),  # chokes the open hat, itself 84 dB down
        (4.5, 46, 127, 0.05),
    ]
    y = play(binary, out / "kit" / "kit.sfz", notes, out)
    # A mono kit plays in the centre: one channel is the other times a constant within 0.2 dB of 1, the
    # player's pan law (exactly 1 in sfizz, 1.0125 in liquidsfz).
    left, right = y[:, 0], y[:, 1]
    c = float(np.dot(left, right) / np.dot(left, left))
    assert abs(20 * math.log10(c)) < 0.2 and np.max(np.abs(right - c * left)) <= 1e-6 * np.max(np.abs(right))
    files = {p.name: soundfile.read(p, dtype="float64")[0] for p in (out / "kit").glob("*.wav")}
    return kit, notes, y[:, 0], files


def test_each_note_plays_its_key_layer_and_round_robin(played):
    kit, notes, y, files = played
    (loud, soft), (open_hat,) = kit.keys[0].layers, kit.keys[1].layers
    heard = []
    for t, key, velocity, _ in notes:
        if velocity > 1:
            name, corr, _ = identify(y, round(t * RATE), files)
            assert corr > 0.99, (t, name, corr)  # any other file, an independent strike, correlates near 0
            heard.append(name)
    assert soft.lovel == 1 and soft.hivel < 127 and loud.lovel == soft.hivel + 1
    # Velocity 127 plays the loudest layer and 40 the softer one, on the key played.
    rr = loud.files(42)
    assert heard[:3] == [rr[0], rr[1], rr[0]]
    # Each layer cycles through its takes. sfizz keeps one sequence counter per key across its layers, so where
    # a layer's cycle starts depends on the player.
    assert sorted(heard[3:5]) == sorted(soft.files(42))
    assert heard[5:] == open_hat.files(46)


def test_velocity_follows_the_square_law_across_layers(played):
    # The player scales each file by its group's volume and curve point. Each layer's curve is set from its first
    # take's loudness, so a first take sounds at exactly 40·log10(v/127) dB from the loudest layer's first take at
    # 127, and every take of a layer gets the same gain at the same velocity.
    kit, notes, y, files = played
    (loud, soft), _ = kit.keys[0].layers, kit.keys[1].layers
    heard = {}
    for t in (0.5, 2.0, 2.5):
        name, _, start = identify(y, round(t * RATE), files)
        gain = level_db(y[start + 300 : start + 2300], files[name][300:2300])
        heard[name] = (gain, momentary_max(files[name], RATE) + gain)
    top_gain, top = heard[loud.files(42)[0]]
    gain1, quiet = heard[soft.files(42)[0]]
    gain2, _ = heard[soft.files(42)[1]]
    assert quiet - top == pytest.approx(40 * math.log10(40 / 127), abs=0.05)
    assert gain2 == pytest.approx(gain1, abs=0.05)


def test_the_closed_hat_chokes_the_open_one_over_the_release(played):
    kit, notes, y, files = played
    x = files[kit.keys[1].layers[0].files(46)[0]]
    _, _, at = identify(y, round(3.0 * RATE), files)
    gain = level_db(y[at + 300 : at + 2300], x[300:2300])
    cut = round(0.4 * RATE)  # the choke, 0.4 s after the open hit
    window = round(0.01 * RATE)

    def drop(offset):
        """The played level over the file's own, in dB, over 10 ms from `offset` after the open hit."""
        return level_db(y[at + offset : at + offset + window], x[offset : offset + window]) - gain

    assert abs(drop(cut - window)) < 0.1  # the hit itself until the choke
    assert -40 < drop(cut + round(0.010 * RATE)) < -1  # fading over the 40 ms release, by each player's own curve
    # A release later the open hat is gone, down to the player's 16-bit floor, where it would still ring.
    later = slice(at + cut + round(0.1 * RATE), at + cut + round(0.11 * RATE))
    assert np.max(np.abs(y[later])) <= 2 * 2.0**-15
    ringing = 10 ** (gain / 20) * x[cut + round(0.1 * RATE) : cut + round(0.11 * RATE)]
    assert np.sqrt(np.mean(ringing**2)) > 8 * 2.0**-15
