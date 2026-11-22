# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tuning: MIDI note numbers and names, and the frequencies a note name or a number gives.

Shared by the patch's editing commands, the render report and the WAV root note.
"""

from __future__ import annotations

import math
import re

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
