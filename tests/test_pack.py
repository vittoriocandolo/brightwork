# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Kits: keys, velocity layers on the square law, takes, chokes, one gain, and the SFZ mapping (plan §7.3)."""

import dataclasses
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile

from brightwork_synth import cli, pack
from brightwork_synth.loudness import momentary_max
from brightwork_synth.patch import Patch, dumps, loads, softer

from .conftest import meta_of

type Opcodes = dict[str, str]
type Written = tuple[Path, list[Path]]

TOP_VELOCITY = 127
RELEASE_S = 0.04  # a choke's fade, by default
SQUARE_LAW_DB = 0.01  # how close every velocity's level comes to 40·log₁₀(v/127)
MATCHED_DB = 0.01  # how close a level-matched layer's files come to its first
BAD_INPUT = 2  # the exit status for input the command cannot use


def parse_sfz(text: str) -> list[tuple[str, Opcodes]]:
    """Read the mapping independently: its headers with their opcodes, in order."""
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    for header, body in re.findall(r"<(\w+)>([^<]*)", text):
        opcodes = dict(re.findall(r"(\w+)=(\S+)", body))
        out.append((header, opcodes))
    return out


def regions(text: str) -> list[Opcodes]:
    """Return every region with the opcodes it inherits from its global and group."""
    glob: Opcodes = {}
    group: Opcodes = {}
    found: list[Opcodes] = []
    for header, ops in parse_sfz(text):
        if header == "global":
            glob = ops
        elif header == "group":
            group = ops
        elif header == "region":
            found.append({**glob, **group, **ops})
    return found


def sfz(directory: Path) -> Path:
    """Return the kit's one mapping file."""
    (found,) = directory.glob("*.sfz")
    return found


def ladder(name: str, steps: tuple[float, ...]) -> list[tuple[str, Patch]]:
    """Return a reference struck softer by each step in dB, named after it, loudest first."""
    base = loads(cli.reference_text(name))
    return [(name if db == 0 else f"{name}-{db}", softer(base, db)) for db in steps]


@pytest.fixture(scope="module")
def hats() -> pack.Kit:
    """Build a hi-hat kit: three closed layers, a pedal, and two open layers choked by both."""
    groups = [(42, ladder("closed", (0, 8, 16))), (44, ladder("pedal", (0,))), (46, ladder("open", (0, 12)))]
    return pack.build(groups, pack.Options(takes=2, chokes=((46, (42, 44)),)))


@pytest.fixture(scope="module")
def written(hats: pack.Kit, tmp_path_factory: pytest.TempPathFactory) -> Written:
    """Write the hi-hat kit once, and return its folder and files."""
    out = tmp_path_factory.mktemp("hats")
    return out, pack.write(hats, out)


def test_velocity_ranges_follow_the_square_law() -> None:
    """Split velocities 1–127 among layers where the square law reaches each, meeting halfway in dB."""
    ranges = pack.velocity_ranges([0.0, -6.0, -12.0, -24.0])
    assert ranges[0][1] == TOP_VELOCITY
    assert ranges[-1][0] == 1
    for (lo, _), (_, hi) in itertools.pairwise(ranges):
        assert hi == lo - 1  # contiguous, no gaps and no overlaps
    # A layer 6 dB down sits where 40·log10(v/127) = −6: v ≈ 90; neighbours meet halfway in dB.
    natural = 127 * 10 ** (-6 / 40)
    assert ranges[1][0] <= natural <= ranges[1][1]
    assert ranges[0][0] == math.floor(math.sqrt(127 * natural)) + 1
    with pytest.raises(pack.PackError, match="too close in loudness"):
        pack.velocity_ranges([0.0, -0.001, -0.002])


def test_the_curve_puts_each_layer_on_the_square_law() -> None:
    """Give each layer a volume and curve that put every velocity on 40·log₁₀(v/127) dB."""
    for relative in (0.0, -5.0, -17.5):
        lo, hi = 40, 90
        volume, curve = pack.curve_for(relative, lo, hi)
        assert max(curve) == 1.0
        assert min(curve) > 0
        for v, c in zip(range(lo, hi + 1), curve, strict=True):
            assert relative + volume + 20 * math.log10(c) == pytest.approx(40 * math.log10(v / 127), abs=1e-9)


def test_every_key_velocity_and_take_maps_to_exactly_one_file(written: Written) -> None:
    """Map every key, velocity and take to exactly one file of the kit, named so it maps by hand too."""
    out, files = written
    text = sfz(out).read_text()
    rs = regions(text)
    assert len(rs) == (3 + 1 + 2) * 2
    assert all(r["pitch_keytrack"] == "0" and r["loop_mode"] == "one_shot" for r in rs)
    for key in (42, 44, 46):
        for v in range(1, 128):
            for position in (1, 2):
                hits = [
                    r
                    for r in rs
                    if r["key"] == str(key)
                    and int(r["lovel"]) <= v <= int(r["hivel"])
                    and int(r["seq_position"]) == position
                ]
                assert len(hits) == 1
                assert hits[0]["seq_length"] == "2"
                assert (out / hits[0]["sample"]).exists()
    assert {p.name for p in files} == {r["sample"] for r in rs} | {sfz(out).name}
    assert all(re.fullmatch(r"(42|44|46)_[a-z]+(-\d+)?_v\d+_rr[12]\.wav", r["sample"]) for r in rs)


def test_chokes_as_declared(written: Written) -> None:
    """Put the closed and pedal hats in one choke group, which turns the open hat off over 40 ms."""
    out, _ = written
    rs = regions(sfz(out).read_text())
    by_key = {k: [r for r in rs if r["key"] == str(k)] for k in (42, 44, 46)}
    groups = {r["group"] for k in (42, 44) for r in by_key[k]}
    assert len(groups) == 1  # the closed and pedal hats are one choke group
    (group,) = groups
    for r in by_key[46]:
        assert r["off_by"] == group
        assert r["off_mode"] == "normal"
        assert float(r["ampeg_release"]) == RELEASE_S
        assert "group" not in r  # the open hat chokes nothing
    assert all("off_by" not in r for k in (42, 44) for r in by_key[k])


def test_choke_rules_are_checked() -> None:
    """Number choke groups from the rules, and refuse a key outside the kit, one choking itself, or in two groups."""
    assert pack.choke_groups([(46, (42, 44))], [42, 44, 46]) == {42: (1, None), 44: (1, None), 46: (None, 1)}
    # Rules for one target add up, and targets choked by the same keys share their group.
    roles = pack.choke_groups([(46, (42,)), (46, (44,)), (49, (42, 44))], [42, 44, 46, 49])
    assert roles == {42: (1, None), 44: (1, None), 46: (None, 1), 49: (None, 1)}
    # Two keys can choke each other.
    assert pack.choke_groups([(46, (42,)), (42, (46,))], [42, 46]) == {42: (1, 2), 46: (2, 1)}
    for rules, message in (
        ([(46, (40,))], "key 40 is not in the kit"),
        ([(46, (46,))], "cannot choke itself"),
        ([(46, (42, 44)), (49, (42,))], "key 42 would need two choke groups"),
    ):
        with pytest.raises(pack.PackError, match=message):
            pack.choke_groups(rules, [42, 44, 46, 49])


def test_the_level_follows_the_square_law_on_every_key(hats: pack.Kit, written: Written) -> None:
    """Measure every velocity's level from the files: on the square law, and within a step across boundaries."""
    out, _ = written
    text = sfz(out).read_text()
    rate = hats.keys[0].layers[0].patch.output.sample_rate
    for key in (42, 46):
        groups = [ops for header, ops in parse_sfz(text) if header == "group" and ops["key"] == str(key)]
        top: float | None = None
        levels: list[tuple[int, float]] = []
        for group in groups:
            name = next(
                r["sample"]
                for r in regions(text)
                if r["key"] == str(key) and r["lovel"] == group["lovel"] and r["seq_position"] == "1"
            )
            data, _ = soundfile.read(out / name, dtype="float64")
            loud = momentary_max(data, rate)
            top = loud if top is None else top
            for v in range(int(group["lovel"]), int(group["hivel"]) + 1):
                c = float(group[f"amp_velcurve_{v}"])
                levels.append((v, loud + float(group["volume"]) + 20 * math.log10(c) - top))
        levels.sort()
        assert [v for v, _ in levels] == list(range(1, 128))
        for v, level in levels:
            assert level == pytest.approx(40 * math.log10(v / 127), abs=SQUARE_LAW_DB)
        # Within one velocity step across every boundary.
        step = {v: 40 * math.log10((v + 1) / v) for v in range(1, 127)}
        for (v0, l0), (v1, l1) in itertools.pairwise(levels):
            assert v1 == v0 + 1
            assert 0 < l1 - l0 <= step[v0] + 0.02


def test_files_carry_their_key_layer_and_take(hats: pack.Kit, written: Written) -> None:
    """Write each file's patch, takes, place in the kit and the kit's gain into it."""
    out, _ = written
    for k in hats.keys:
        for i, layer in enumerate(k.layers, 1):
            for j, name in enumerate(layer.files(k.key), 1):
                assert name == f"{k.key}_{layer.name}_v{layer.hivel}_rr{j}.wav"
                meta = meta_of(out / name)
                assert loads(meta["patch"]) == dataclasses.replace(layer.patch, take=layer.patch.take + j - 1)
                assert meta["takes"] == [layer.patch.take + j - 1]
                assert meta["kit"] == {
                    "key": k.key,
                    "layer": i,
                    "layers": len(k.layers),
                    "lovel": layer.lovel,
                    "hivel": layer.hivel,
                    "group": k.group,
                    "off_by": k.off_by,
                }
                assert meta["set_gain_db"] == hats.set_gain_db


def test_the_report_lists_every_key_and_its_chokes(hats: pack.Kit) -> None:
    """Report the kit: its keys, layers, chokes, and every layer inside the validated domain."""
    text = "\n".join(hats.lines("hats"))
    assert "3 keys, 12 files in hats" in text
    assert "key 42" in text
    assert "layer 3 closed-16" in text
    assert "choked by keys 42, 44, over 40 ms" in text
    assert text.count("inside, worst margin") == sum(len(k.layers) for k in hats.keys)


def test_a_kit_shares_one_gain() -> None:
    """Bring a peak-mode kit's loudest file to the target, every other at its natural level."""
    peak = dataclasses.replace(loads(cli.reference_text("closed")).output, mode="peak", db=-1.0)
    groups = [
        (key, [(n, dataclasses.replace(p, output=peak)) for n, p in layers])
        for key, layers in ((42, ladder("closed", (0, 10))), (46, ladder("open", (0,))))
    ]
    kit = pack.build(groups, pack.Options(takes=2))
    peaks = [float(np.max(np.abs(a))) for k in kit.keys for row in k.audio for a in row]
    assert 20 * math.log10(max(peaks)) == pytest.approx(-1.0, abs=1e-6)
    assert len({round(p, 9) for p in peaks}) == len(peaks)  # the others keep their natural levels


def test_level_matching_within_each_layer() -> None:
    """Level-match each layer's files to its first."""
    kit = pack.build([(42, ladder("closed", (0, 8)))], pack.Options(takes=4, level_match=True))
    for layer, files in zip(kit.keys[0].layers, kit.keys[0].audio, strict=True):
        rate = layer.patch.output.sample_rate
        first = momentary_max(files[0], rate)
        assert all(momentary_max(a, rate) == pytest.approx(first, abs=0.01) for a in files)
        assert layer.spread < MATCHED_DB


def test_a_stereo_kit_pairs_consecutive_takes(tmp_path: Path) -> None:
    """Pair consecutive takes as the left and right of each stereo file."""
    kit = pack.build([(42, ladder("closed", (0, 12)))], pack.Options(takes=2, stereo=True))
    files = pack.write(kit, tmp_path)
    layer = kit.keys[0].layers[0]
    assert layer.takes == ((1, 2), (3, 4))
    data, _ = soundfile.read(tmp_path / layer.files(42)[1], dtype="float32")
    assert data.shape[1] == len(("left", "right"))
    assert meta_of(tmp_path / layer.files(42)[1])["takes"] == [3, 4]
    assert "seq_length=2" in sfz(tmp_path).read_text()
    assert len(files) == 2 * 2 + 1  # two layers of two files, and the mapping


def test_a_kit_re_renders_bit_identically(tmp_path: Path) -> None:
    """Build and write a kit twice, and find every file byte for byte the same."""
    groups = [(42, ladder("closed", (0, 12))), (46, ladder("open", (0,)))]
    for d in ("a", "b"):
        pack.write(pack.build(groups, pack.Options(takes=2, chokes=((46, (42,)),)), stem="kit"), tmp_path / d)
    names = sorted(f.name for f in (tmp_path / "a").iterdir())
    assert len(names) == (2 + 1) * 2 + 1  # three layers of two files, and the mapping
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_kits_refuse_what_they_cannot_map() -> None:
    """Refuse layers out of order, different outputs, a silent layer, a key given twice and too many takes."""
    (_, loud), (_, soft) = ladder("closed", (0, 8))
    with pytest.raises(pack.PackError, match="not softer"):
        pack.build([(42, [("soft", soft), ("loud", loud)])])
    other = dataclasses.replace(soft, output=dataclasses.replace(soft.output, sample_rate=44100))
    with pytest.raises(pack.PackError, match=r"output\.sample_rate"):
        pack.build([(42, [("loud", loud)]), (44, [("other", other)])])
    silent = dataclasses.replace(soft, noise=tuple(dataclasses.replace(n, level_db=-math.inf) for n in soft.noise))
    with pytest.raises(pack.PackError, match="silent"):
        pack.build([(42, [("loud", loud), ("silent", silent)])])
    with pytest.raises(pack.PackError, match="given twice"):
        pack.build([(42, [("loud", loud)]), (42, [("soft", soft)])])
    with pytest.raises(pack.PackError, match="--takes"):
        pack.build([(42, [("loud", loud)])], pack.Options(takes=65, stereo=True))


def test_the_command_writes_a_kit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Run `brightwork pack`: it writes the kit and its report, or says what is wrong."""
    paths = {}
    for name, p in ladder("closed", (0, 8)) + ladder("open", (0,)):
        paths[name] = tmp_path / f"{name}.toml"
        paths[name].write_text(dumps(p))
    out = tmp_path / "hats"
    args = ["pack", f"42={paths['closed']},{paths['closed-8']}", f"46={paths['open']}", "--choke", "46:42"]
    code = cli.main([*args, "--takes", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "2 keys, 6 files in" in text
    assert "velocities" in text
    assert "choked by key 42" in text
    assert (out / "hats.sfz").exists()
    assert len(list(out.glob("*.wav"))) == (2 + 1) * 2
    code = cli.main(["pack", f"42={paths['closed-8']},{paths['closed']}", "--out", str(tmp_path / "bad")])
    assert code == BAD_INPUT
    assert "not softer" in capsys.readouterr().err
    code = cli.main(["pack", f"42={paths['closed']}", "--choke", "46:42", "--out", str(tmp_path / "bad")])
    assert code == BAD_INPUT
    assert "key 46 is not in the kit" in capsys.readouterr().err
