import dataclasses
import math

import numpy as np
import pytest
from conftest import EXAMPLE, TONE, example, make

from brightwork_synth import patch as P
from brightwork_synth.patch import PatchError, dumps, from_dict, loads


def problems(fn, *args, **kwargs):
    with pytest.raises(PatchError) as e:
        fn(*args, **kwargs)
    return e.value.problems


def reject(tone=False, **edits):
    return problems(make, tone, **edits)


def amp(*points):
    return [{"ms": m, "gain": g} for m, g in points]


def partials(*pairs):
    return [{"hz": f, "level_db": v} for f, v in pairs]


# Parsing.


def test_example_parses():
    p = loads(EXAMPLE)
    assert (p.seed, p.take) == (1, 1)
    assert len(p.noise) == 2 and p.tone == ()
    assert p.noise[0] == P.Noise(
        -2.0,
        (6000.0, 16000.0),
        14.0,
        0.5,
        0.0,
        0.0,
        220.0,
        ((0.0, 0.0), (0.3, 1.0), (8.0, 0.45), (30.0, 0.14), (70.0, 0.04), (120.0, 0.0)),
    )
    assert p.damper == P.Damper(())
    assert p.drive == P.Drive(0.0)
    assert p.output == P.Output(48000, "float32", "gain", -6.0, 1)
    assert p.length_ms == 120.0
    q = make(tone=True)
    assert q.tone[0].partials[0] == (205.3, 0.0) and q.tone[0].f_ref == 7000.0
    assert [prefix for prefix, _ in q.sources] == ["noise[1]", "noise[2]", "tone[1]"]


def random_shape(rng, n, first, limit):
    times = [first, *(first + np.cumsum(rng.uniform(0.01, limit / n, n - 1)))]
    g = list(rng.uniform(0, 1, n))
    g[0] = g[-1] = 0.0
    g[int(rng.integers(1, n - 1))] = 1.0
    return tuple(zip(times, g))


def random_patch(rng) -> P.Patch:
    def common(lo, f_ref):
        # Damping and bloom within the 1000-fold limit at 20 kHz.
        top = math.log(1000) / math.log(max(20000 / f_ref, 1.0001))
        return dict(
            damping=float(rng.uniform(0, min(3, top))),
            bloom=float(rng.choice([0.0, rng.uniform(0, min(0.5, top))])),
            chatter=float(rng.uniform(0, 1)),
            chatter_hz=float(rng.uniform(10, 4000)),
            amp=random_shape(rng, int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 1500),
        )

    noise = []
    for _ in range(int(rng.integers(0, 5))):
        lo = float(rng.uniform(20, 19000))
        noise.append(
            P.Noise(
                level_db=float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
                band_hz=(lo, float(rng.uniform(lo + 1, 20000))),
                spacing_hz=float(rng.choice([0.0, rng.uniform(1, 200)])),
                **common(lo, lo),
            )
        )
    tones = []
    for _ in range(int(rng.integers(0 if noise else 1, 3))):
        n = int(rng.integers(1, 17))
        hz = np.sort(rng.uniform(20, 19000, n))
        hz = [float(f) for f in hz if True]
        hz = [f for i, f in enumerate(hz) if i == 0 or round(f * 8) > round(hz[i - 1] * 8)]
        levels = list(rng.uniform(-60, 0, len(hz)))
        levels[int(rng.integers(len(hz)))] = 0.0
        lo = float(rng.uniform(20, 19000))
        tones.append(
            P.Tone(
                level_db=float(rng.uniform(-90, 24)),
                wave=str(rng.choice(P.WAVES)),
                phase=str(rng.choice(P.PHASES)),
                partials=tuple(zip(hz, levels)),
                band_hz=(lo, float(rng.uniform(lo + 1, 20000))),
                **common(lo, max(lo, hz[0])),
            )
        )
    damper = ()
    if rng.random() < 0.5:
        n = int(rng.integers(2, 33))
        t = 100 + np.cumsum(rng.uniform(0.01, 50, n))
        g = np.sort(rng.uniform(0, 1, n))[::-1]
        g[0], g[-1] = 1.0, 0.0
        damper = tuple(zip(map(float, t), map(float, g)))
    mode = str(rng.choice(["gain", "peak"]))
    return P.Patch(
        seed=int(rng.integers(0, 2**63 - 1)),
        take=int(rng.integers(1, 2**31)),
        noise=tuple(noise),
        tone=tuple(tones),
        damper=P.Damper(damper),
        drive=P.Drive(float(rng.uniform(0, 48))),
        output=P.Output(
            int(rng.choice(P.SAMPLE_RATES)),
            str(rng.choice(P.FORMATS)),
            mode,
            float(rng.uniform(-60, 0 if mode == "peak" else 24)),
            int(rng.choice([1, -1])),
        ),
    )


def test_dump_round_trips_exactly():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = random_patch(rng)
        text = dumps(p)
        assert loads(text) == p
        assert dumps(loads(text)) == text


def test_dump_of_example_keeps_the_plan_layout():
    ours, plan = dumps(loads(EXAMPLE)).splitlines(), EXAMPLE.splitlines()
    assert len(ours) == len(plan)
    for a, b in zip(ours, plan):
        # The plan annotates its sources; the canonical comments name them by path.
        assert a.split("#")[0].rstrip() == b.split("#")[0].rstrip()
    assert ours[:4] == plan[:4]


def test_instrument_comes_first(example_dict):
    d = {k: v for k, v in example_dict.items() if k != "instrument"}
    assert problems(from_dict, d) == ['instrument: missing; a cymbal patch starts with instrument = "cymbal"']
    d = dict(example_dict, instrument="snare", colour="red")
    assert problems(from_dict, d) == ["instrument: this is a 'snare' patch, not a \"cymbal\" one"]


def test_schema_versions(example_dict):
    for bad, message in ((2, "unsupported version 2"), ("1", "expected an integer"), (True, "expected an integer")):
        assert message in problems(from_dict, dict(example_dict, schema=bad))[0]
    d = {k: v for k, v in example_dict.items() if k != "schema"}
    assert problems(from_dict, d) == ["schema: missing"]
    assert len(problems(from_dict, dict(example_dict, schema=2, colour="red"))) == 1


def test_integers_accepted_for_floats_but_not_booleans():
    p = make(
        **{"noise[1].band_hz": [6000, 16000], "drive.db": 0, "noise[1].spacing_hz": 14, "noise[1].chatter_hz": 220}
    )
    assert all(type(x) is float for x in p.noise[0].band_hz)
    assert type(p.drive.db) is float and type(p.noise[0].spacing_hz) is float
    assert "noise[1].damping: expected a number, found a boolean" in reject(**{"noise[1].damping": False})
    assert any("output.sample_rate: expected an integer" in m for m in reject(**{"output.sample_rate": 48000.0}))
    assert any("seed: expected an integer" in m for m in reject(seed=1.0))
    assert any("take: expected an integer" in m for m in reject(take=1.0))
    assert any(
        "noise[2].chatter: expected a number, found a string" in m for m in reject(**{"noise[2].chatter": "0.5"})
    )


def test_unknown_and_missing_fields(example_dict):
    d = example_dict
    d["colour"] = "red"
    d["noise"][0]["decay"] = 1.0
    del d["noise"][1]["bloom"]
    d["noise"][0]["amp"][1]["curve"] = "exp"
    d["damper"]["release"] = 3
    del d["drive"]
    found = problems(from_dict, d)
    assert "colour: unknown field" in found
    assert "noise[1].decay: unknown field" in found
    assert "noise[2].bloom: missing" in found
    assert "noise[1].amp[2].curve: unknown field" in found
    assert "damper.release: unknown field" in found
    assert "drive: missing" in found


def test_every_field_is_required_but_the_source_arrays(example_dict):
    for section in ("damper", "drive", "output"):
        for key in list(example_dict[section]):
            d = dict(example_dict)
            d[section] = {k: v for k, v in example_dict[section].items() if k != key}
            assert f"{section}.{key}: missing" in problems(from_dict, d)
    for key in list(example_dict["noise"][0]):
        d = dict(example_dict)
        d["noise"] = [{k: v for k, v in example_dict["noise"][0].items() if k != key}, example_dict["noise"][1]]
        assert f"noise[1].{key}: missing" in problems(from_dict, d)
    for key in TONE:
        d = example(tone=True)
        d["tone"][0] = {k: v for k, v in TONE.items() if k != key}
        assert f"tone[1].{key}: missing" in problems(from_dict, d)
    for key in ("seed", "take", "damper", "drive", "output"):
        d = {k: v for k, v in example_dict.items() if k != key}
        assert f"{key}: missing" in problems(from_dict, d)
    # A patch of tones alone, and one of noise alone, are both complete.
    d = example(tone=True)
    del d["noise"]
    assert from_dict(d).noise == () and len(from_dict(d).tone) == 1
    d = {k: v for k, v in example_dict.items() if k != "noise"}
    assert "noise: a patch needs at least one source, [[noise]] or [[tone]]" in problems(from_dict, d)


def test_non_finite_numbers():
    assert make(**{"noise[2].level_db": -math.inf}).noise[1].level_db == -math.inf
    assert make(tone=True, **{"tone[1].level_db": -math.inf}).tone[0].level_db == -math.inf
    for path in ("noise[1].level_db",):
        assert any("must be finite" in m for m in reject(**{path: math.inf}))
        assert any("must be finite" in m for m in reject(**{path: math.nan}))
    for path in ("noise[1].damping", "noise[1].spacing_hz", "drive.db", "output.db", "noise[1].chatter"):
        assert any("must be finite" in m for m in reject(**{path: -math.inf}))
        assert any("must be finite" in m for m in reject(**{path: math.nan}))
    assert any("must be finite" in m for m in reject(tone=True, **{"tone[1].partials[2].hz": math.inf}))


def test_knot_times():
    found = reject(**{"noise[1].amp": amp((0, 0), (1, 1), (1, 0.5), (2, 0))})
    assert any("repeats the previous knot time" in m for m in found)
    found = reject(**{"noise[1].amp": amp((0, 0), (2, 1), (1, 0.5), (3, 0))})
    assert any("is earlier than the previous knot" in m for m in found)
    found = reject(**{"noise[2].amp": amp((0, 0), (0.005, 1), (1, 0))})
    assert any(m.startswith("noise[2].amp[2].ms: is 0.005 ms after the previous knot") for m in found)
    # Exactly 10 µs is allowed, even where binary rounding makes the float gap smaller.
    make(**{"noise[2].amp": amp((0, 0), (0.02, 0.5), (0.03, 1), (1, 0))})
    assert any("noise[1].amp[1].ms" in m for m in reject(**{"noise[1].amp": amp((-1, 0), (1, 1), (2, 0))}))


def test_amplitude_shapes():
    assert "noise[1].amp[1].gain: an amplitude shape must start at 0" in reject(
        **{"noise[1].amp": amp((0, 0.1), (1, 1), (2, 0))}
    )
    assert "noise[1].amp[3].gain: an amplitude shape must end at 0" in reject(
        **{"noise[1].amp": amp((0, 0), (1, 1), (2, 0.1))}
    )
    found = reject(**{"noise[2].amp": amp((0, 0), (1, 0.9999999), (2, 0))})
    assert "noise[2].amp: the highest gain must be exactly 1, found 0.9999999" in found
    assert "noise[1].amp[2].gain: 1.5 is outside 0 … 1" in reject(
        **{"noise[1].amp": amp((0, 0), (1, 1.5), (2, 1), (3, 0))}
    )
    # Delayed shapes are allowed: the first knot need not be at 0 ms.
    assert make(**{"noise[2].amp": amp((5, 0), (6, 1), (20, 0))}).noise[1].amp[0] == (5.0, 0.0)


def test_dampers():
    knots = amp((100, 1), (110, 0.3), (120, 0.3), (140, 0))
    assert make(**{"damper.amp": knots}).damper.amp[0] == (100.0, 1.0)
    assert make(**{"damper.amp": amp((100, 1), (140, 0))}).length_ms == 120.0
    assert "damper.amp[1].gain: a damper must start at exactly 1" in reject(**{"damper.amp": amp((100, 0.9), (140, 0))})
    assert "damper.amp[2].gain: a damper must end at exactly 0" in reject(**{"damper.amp": amp((100, 1), (140, 0.1))})
    found = reject(**{"damper.amp": amp((100, 1), (110, 0.2), (120, 0.4), (140, 0))})
    assert "damper.amp[3].gain: a damper never increases" in found
    assert any("needs 2 to 32 knots, found 1" in m for m in reject(**{"damper.amp": amp((100, 1))}))
    assert any("damper.amp[2].ms" in m for m in reject(**{"damper.amp": amp((100, 1), (8000.5, 0))}))


def test_noise_sources(example_dict):
    four = dict(example_dict, noise=example_dict["noise"] * 2)
    assert len(from_dict(four).noise) == 4
    assert "noise: at most 4 noise sources, found 5" in problems(
        from_dict, dict(example_dict, noise=example_dict["noise"] * 2 + example_dict["noise"][:1])
    )
    assert any(
        m.startswith("noise: expected an array of tables ([[noise]])")
        for m in problems(from_dict, dict(example_dict, noise=example_dict["noise"][0]))
    )
    make(**{"noise[1].band_hz": [20, 20000]})
    for bad, path in (
        ([19, 1000], "noise[1].band_hz[1]"),
        ([100, 20001], "noise[1].band_hz[2]"),
        ([500, 500], "noise[1].band_hz"),
        ([900, 800], "noise[1].band_hz"),
    ):
        assert any(m.startswith(path + ":") for m in reject(**{"noise[1].band_hz": bad}))
    assert "noise[2].band_hz: expected [low, high] in Hz" in reject(**{"noise[2].band_hz": [100.0]})
    for good in (0.0, 1.0, 200.0):
        make(**{"noise[1].spacing_hz": good})
    for bad in (0.5, -1.0, 200.5):
        assert any(m.startswith("noise[1].spacing_hz") for m in reject(**{"noise[1].spacing_hz": bad}))


def test_tones():
    d = example(tone=True)
    d["tone"] = [TONE, TONE]
    assert len(from_dict(d).tone) == 2
    d["tone"] = [TONE] * 3
    assert "tone: at most 2 tone sources, found 3" in problems(from_dict, d)
    assert any(m.startswith("tone[1].wave") for m in reject(tone=True, **{"tone[1].wave": "saw"}))
    assert any(m.startswith("tone[1].phase") for m in reject(tone=True, **{"tone[1].phase": "random"}))
    assert "tone[1].partials: needs 1 to 16 partials, found 0" in reject(tone=True, **{"tone[1].partials": []})
    many = partials(*[(100.0 + i, 0.0) for i in range(17)])
    assert "tone[1].partials: needs 1 to 16 partials, found 17" in reject(tone=True, **{"tone[1].partials": many})
    found = reject(tone=True, **{"tone[1].partials": partials((500.0, 0.0), (500.06, -3.0))})
    assert found == ["tone[1].partials[2].hz: partials must increase by at least 1/8 Hz, the grid's step"]
    found = reject(tone=True, **{"tone[1].partials": partials((500.0, -1.0), (600.0, -3.0))})
    assert found == ["tone[1].partials: the loudest partial must be exactly 0 dB, found -1.0"]
    for bad in ((19.0, 0.0), (20000.5, 0.0)):
        assert any(
            m.startswith("tone[1].partials[1].hz") for m in reject(tone=True, **{"tone[1].partials": partials(bad)})
        )
    assert any(
        m.startswith("tone[1].partials[2].level_db")
        for m in reject(tone=True, **{"tone[1].partials": partials((500.0, 0.0), (600.0, -61.0))})
    )
    # The reference frequency is the higher of the band's lower edge and the lowest partial.
    assert make(tone=True, **{"tone[1].band_hz": [20.0, 20000.0]}).tone[0].f_ref == 205.3


def test_damping_and_bloom_limits():
    # (20 kHz / 300 Hz)^damping must stay within 1000: damping 1.64 passes, 1.65 does not.
    make(**{"noise[1].band_hz": [300.0, 18000.0], "noise[1].damping": 1.64})
    found = reject(**{"noise[1].band_hz": [300.0, 18000.0], "noise[1].damping": 1.65})
    assert len(found) == 1 and found[0].startswith("noise[1]: damping 1.65 and bloom 0 compress or stretch")
    # Bloom stretches the rise of the high sub-bands; none may end after 8 s.
    slow = amp((0, 0), (500, 1), (7000, 0))
    make(**{"noise[1].amp": slow, "noise[1].bloom": 0.1, "noise[1].damping": 0.5})
    found = reject(**{"noise[1].amp": slow, "noise[1].bloom": 1.0, "noise[1].damping": 0.0})
    assert len(found) == 1 and found[0].startswith("noise[1].bloom: its slowest sub-band would end at")


@pytest.mark.parametrize(
    "path, good, bad",
    [
        ("noise[1].level_db", [-90, 24], [-90.1, 24.1]),
        ("noise[1].damping", [0, 3], [-0.001, 3.001]),
        ("noise[2].bloom", [0, 3], [-0.01, 3.01]),
        ("noise[2].chatter", [0, 1], [-0.01, 1.01]),
        ("noise[2].chatter_hz", [10, 4000], [9.9, 4000.1]),
        ("drive.db", [0, 48], [-0.1, 48.1]),
        ("output.db", [-60, 24], [-60.5, 24.5]),
        ("output.sample_rate", [44100, 48000, 88200, 96000], [22050, 192000]),
        ("output.format", ["float32", "pcm24", "pcm16"], ["pcm32", "float64", 24]),
        ("output.mode", ["gain", "peak"], ["loudness"]),
        ("output.polarity", [1, -1], [0, 2]),
        ("seed", [0, 2**63 - 1], [-1, 2**63]),
        ("take", [1, 2**31], [0, 2**31 + 1]),
    ],
)
def test_domain_edges(path, good, bad):
    for v in good:
        make(**{path: v, "output.db": -6.0} if path == "output.mode" else {path: v})
    for v in bad:
        assert any(m.startswith(path) for m in reject(**{path: v}))


def test_peak_target_is_at_most_full_scale():
    make(**{"output.mode": "peak", "output.db": 0.0})
    assert any(m.startswith("output.db") for m in reject(**{"output.mode": "peak", "output.db": 0.1}))


def test_all_problems_are_reported_together():
    found = reject(
        **{
            "noise[1].damping": 4.0,
            "drive.db": 60.0,
            "output.mode": "loud",
            "noise[2].chatter": 2.0,
            "noise[1].amp": amp((0, 0), (1, 0.5), (2, 0)),
        }
    )
    assert len(found) == 5
    # Problems of structure and of value come together too: an unknown field, a seed and a damping.
    d = example()
    d["noise"][0]["dampning"] = 1.0
    d["seed"], d["take"] = -1, 0
    d["noise"][1]["damping"] = 4.0
    found = problems(from_dict, d)
    assert sorted(m.split(":")[0] for m in found) == ["noise[1].dampning", "noise[2].damping", "seed", "take"]


def test_toml_syntax_errors_are_patch_errors():
    assert problems(loads, 'instrument = "cymbal"\nschema = 1\n[drive\n')[0].startswith("TOML:")


def test_programmatic_edits_are_validated():
    p = loads(EXAMPLE)
    with pytest.raises(PatchError):
        dataclasses.replace(p, noise=(dataclasses.replace(p.noise[0], damping=3.5),))
    with pytest.raises(PatchError):
        dataclasses.replace(p, seed=-5)
    with pytest.raises(PatchError):
        dataclasses.replace(p, noise=p.noise * 3)
    assert dataclasses.replace(p, take=7).take == 7


def test_source_length_counts_only_audible_sources():
    assert make().length_ms == 120.0
    assert make(**{"noise[1].level_db": -math.inf}).length_ms == 5.0
    silent = {"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}
    assert make(**silent).length_ms == 0.0
    # Damping shortens only the high sub-bands, so the shape's end still sets the length.
    assert make(**{"noise[1].damping": 2.0}).length_ms == 120.0
    # Bloom delays the high sub-bands' peaks and with no damping lengthens the sound.
    longer = make(**{"noise[1].damping": 0.0, "noise[1].bloom": 1.0})
    ends = P.sub_band_ends(longer.noise[0])
    assert longer.length_ms == max(ends) > 120.0
    assert max(ends) == pytest.approx(0.3 * 2 ** (math.ceil(2 * math.log2(20000 / 6000)) / 2) + 119.7)


def test_notes():
    assert P.parse_note("A4") == 440.0
    assert P.parse_note("C#4") == P.parse_note("Db4")
    assert P.parse_note("D5") == pytest.approx(587.3295, rel=1e-6)
    assert P.parse_note("+2") is None and P.parse_note("H2") is None
    assert P.describe_pitch(440.0) == "A4 +0.0 cents"
    assert P.describe_pitch(205.25) == "G#3 -20.1 cents"
    assert P.note_name(50) == "D3"


# Editing commands.


def test_pitch_scales_every_frequency_and_keeps_the_times():
    p = make(tone=True)
    lower = {
        "noise[1].band_hz": [3000.0, 9000.0],
        "noise[2].band_hz": [1500.0, 9000.0],
        "tone[1].band_hz": [3000.0, 8000.0],
    }
    up = P.pitch(make(tone=True, **lower), "+12")
    assert up.noise[0].band_hz == (6000.0, 18000.0) and up.noise[0].spacing_hz == 28.0
    assert up.noise[0].chatter_hz == 440.0 and up.tone[0].partials[0] == (410.6, 0.0)
    assert up.tone[0].band_hz == (6000.0, 16000.0)
    assert up.noise[0].amp == p.noise[0].amp and up.damper == p.damper
    down = P.pitch(p, "-2")
    r = 2 ** (-2 / 12)
    assert down.noise[0].band_hz[0] == round(6000 * r, 3) and down.tone[0].partials[5][0] == round(800 * r, 3)
    # A note puts tone[1]'s lowest partial on it.
    on = P.pitch(p, "A3")
    assert on.tone[0].partials[0][0] == 220.0
    assert on.tone[0].partials[1][0] == round(304.4 * 220 / 205.3, 3)
    # Leaving the valid domain fails rather than clamping.
    assert any(m.startswith("noise[1].band_hz[2]") for m in problems(P.pitch, p, "+12"))
    assert problems(P.pitch, make(), "D5")[0].startswith("pitch D5: a note sets the lowest partial of tone[1]")
    assert problems(P.pitch, p, "up")[0].startswith("pitch: expected semitones")


def test_stretch_rescales_after_each_peak_and_the_damper_from_zero():
    p = make(**{"damper.amp": amp((100, 1), (110, 0.1), (140, 0))})
    q = P.stretch(p, "all", 0.8)
    assert q.noise[0].amp == ((0.0, 0.0), (0.3, 1.0), (6.46, 0.45), (24.06, 0.14), (56.06, 0.04), (96.06, 0.0))
    assert q.noise[1].amp == ((0.0, 0.0), (0.03, 1.0), (0.11, 0.6), (1.206, 0.1), (4.006, 0.0))
    assert q.damper.amp == ((80.0, 1.0), (88.0, 0.1), (112.0, 0.0))
    r = P.stretch(p, ["noise[1].amp", "damper.amp"], 1.5, after_ms=10.0)
    assert r.noise[0].amp[2] == (8.0, 0.45) and r.noise[0].amp[3] == (40.0, 0.14)
    assert r.damper.amp == ((145.0, 1.0), (160.0, 0.1), (205.0, 0.0))
    assert r.noise[1] == p.noise[1]
    assert P.stretch(make(), "all", 2.0).damper.amp == ()  # no damper, nothing to stretch


def test_stretch_failures():
    p = loads(EXAMPLE)
    assert problems(P.stretch, p, "noise[3].amp", 2.0)[0].startswith("noise[3].amp: not a curve of this patch")
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(PatchError):
            P.stretch(p, "all", bad)
    with pytest.raises(PatchError):
        P.stretch(p, "all", 2.0, after_ms=-1.0)
    # Stretching past 8 s fails rather than clamping.
    assert any(m.startswith("noise[1].amp[6].ms") for m in problems(P.stretch, p, "noise[1].amp", 70.0))


def test_choke_writes_a_60_db_release():
    p = P.choke(loads(EXAMPLE), 80.0, release=30.0)
    knots = p.damper.amp
    assert knots[0] == (80.0, 1.0) and knots[-1] == (115.0, 0.0)
    assert [ms for ms, _ in knots] == [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    assert [g for _, g in knots[1:-1]] == [0.3162, 0.1, 0.03162, 0.01, 0.003162, 0.001]
    assert p.length_ms == 115.0 and p.noise == loads(EXAMPLE).noise
    assert P.choke(p, 50.0).damper.amp[0] == (50.0, 1.0)  # replaces the damper
    assert P.choke(p, 50.0).damper.amp[-1] == (round(50 + 40 * 7 / 6, 3), 0.0)
    with pytest.raises(PatchError):
        P.choke(p, -1.0)
    with pytest.raises(PatchError):
        P.choke(p, 10.0, release=0.05)
    assert any(m.startswith("damper.amp") for m in problems(P.choke, p, 7990.0, 40.0))


def test_softer_applies_level_and_contact():
    p = make(tone=True)
    q = P.softer(p, 12.0)
    v = 10 ** (-12 / 20)
    assert [s.level_db for _, s in q.sources] == [-14.0, -22.0, -15.0]
    assert q.noise[0].band_hz == (6000.0, round(16000 * v**0.2, 1))
    assert q.tone[0].band_hz == (7000.0, round(16000 * v**0.2, 1))
    rise = q.noise[0].amp[1][0]
    assert rise == pytest.approx(0.3 * v**-0.2, abs=1e-3)
    # Later knots shift by the added time.
    assert q.noise[0].amp[2][0] == pytest.approx(8.0 + rise - 0.3, abs=1e-9)
    assert q.damper == p.damper and q.drive == p.drive
    assert P.softer(p, 0.0) is p
    off = P.softer(p, 12.0, contact=0.0)
    assert off.noise[0].amp == p.noise[0].amp and off.noise[0].band_hz == p.noise[0].band_hz
    silent = make(**{"noise[2].level_db": -math.inf})
    assert P.softer(silent, 6.0).noise[1].level_db == -math.inf
    # Composition, up to the decimals written.
    a, b = P.softer(P.softer(p, 6.0), 6.0), P.softer(p, 12.0)
    for (_, x), (_, y) in zip(a.sources, b.sources):
        assert x.level_db == y.level_db
        assert x.band_hz[1] == pytest.approx(y.band_hz[1], abs=0.11)
        assert all(abs(m - n) <= 0.002 for (m, _), (n, _) in zip(x.amp, y.amp))
