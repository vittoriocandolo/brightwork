# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""A small RIFF writer and reader for WAV, mono or stereo: IEEE float32, or PCM at 24 or 16 bits (plan §7.2).

Files carry fmt, fact (float only), smpl (so samplers map the root key; left
out for unpitched patches), a private 'cymb' chunk with the patch text and
provenance, and data. PCM uses the plain PCM format tag, which hardware
samplers read most widely. RIFF readers skip chunks they do not know.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .patch import midi_pitch

if TYPE_CHECKING:
    import os

    from numpy.typing import ArrayLike, NDArray

FLOAT = 3  # WAVE_FORMAT_IEEE_FLOAT
PCM = 1  # WAVE_FORMAT_PCM
EXTENSIBLE = 0xFFFE  # WAVE_FORMAT_EXTENSIBLE, whose real tag opens the subformat GUID
EXTENSIBLE_FMT = 40  # the length of an extensible fmt chunk, which holds that GUID
PCM_BITS = (16, 24)
FLOAT_BITS = 32
HEADER = 12  # "RIFF", the size and "WAVE"
PRIVATE = b"cymb"


class Format(NamedTuple):
    """A file's sample rate, and its PCM bit depth, or None for float32."""

    rate: int
    bits: int | None = None


def root_note(hz: float) -> tuple[int, int]:
    """Smpl unity note and pitch fraction (units of 2⁻³² semitone, always upward).

    The unity note is the note at or below the pitch, except that a pitch less
    than half a cent flat of a note counts as that note, so samplers that
    ignore the fraction are never a semitone off.
    """
    p = midi_pitch(hz)
    note = math.floor(p + 0.005)
    return note, round(max(0.0, p - note) * 2**32)


def _chunk(cid: bytes, data: bytes) -> bytes:
    """Make a RIFF chunk: its ID, its size and its data, padded to an even length."""
    return cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def _pcm_bytes(samples: NDArray[np.integer[Any]], bits: int) -> bytes:
    """Pack integer samples as little-endian PCM of 16 or 24 bits."""
    if bits == PCM_BITS[0]:
        return samples.astype("<i2").tobytes()
    # 24 bits: the low three bytes of each little-endian 32-bit integer.
    return samples.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def write(
    path: str | os.PathLike[str],
    buffer: ArrayLike,
    fmt: Format,
    *,
    root: tuple[int, int] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write float32 samples, or with `fmt.bits` (16 or 24) integer PCM samples already quantized.

    The buffer is mono, or (frames, 2) for stereo, whose channels are interleaved.
    """
    rate, bits = fmt
    samples = np.asarray(buffer)
    stereo = (samples.ndim, samples.shape[-1]) == (2, 2)
    if samples.ndim != 1 and not stereo:
        msg = "expected a mono buffer or a (frames, 2) stereo buffer"
        raise ValueError(msg)
    channels = 2 if stereo else 1
    frames = samples.shape[0]
    flat = samples.reshape(-1)
    if bits is None:
        if samples.dtype != np.float32:
            msg = "expected a float32 buffer"
            raise ValueError(msg)
        align = 4 * channels
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHHH", FLOAT, channels, rate, align * rate, align, FLOAT_BITS, 0))
        chunks += _chunk(b"fact", struct.pack("<I", frames))
        data = flat.astype("<f4").tobytes()
    else:
        if bits not in PCM_BITS or samples.dtype.kind != "i":
            msg = "expected integer samples and 16 or 24 bits"
            raise ValueError(msg)
        top = 2 ** (bits - 1)
        if flat.size and not (-top <= flat.min() and flat.max() < top):
            msg = f"samples out of range for {bits}-bit PCM"
            raise ValueError(msg)
        align = bits // 8 * channels
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHH", PCM, channels, rate, align * rate, align, bits))
        data = _pcm_bytes(flat, bits)
    if root is not None:
        note, fraction = root
        chunks += _chunk(b"smpl", struct.pack("<9I", 0, 0, round(1e9 / rate), note, fraction, 0, 0, 0, 0))
    if meta is not None:
        chunks += _chunk(PRIVATE, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    chunks += _chunk(b"data", data)
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def chunks(path: str | os.PathLike[str]) -> dict[bytes, bytes]:
    """Return every top-level chunk of a RIFF WAVE file, by ID; the first occurrence wins."""
    data = Path(path).read_bytes()
    if len(data) < HEADER or data[:4] != b"RIFF" or data[8:HEADER] != b"WAVE":
        msg = f"{path}: not a RIFF WAVE file"
        raise ValueError(msg)
    end = min(8 + struct.unpack_from("<I", data, 4)[0], len(data))
    found: dict[bytes, bytes] = {}
    pos = HEADER
    while pos + 8 <= end:
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        found.setdefault(cid, data[pos + 8 : pos + 8 + size])
        pos += 8 + size + size % 2
    return found


@dataclass(frozen=True)
class Wav:
    """A WAV file as read: its rate, samples, root note, Brightwork's data, and format."""

    rate: int
    samples: NDArray[np.float32] | NDArray[np.int32]  # PCM as integers; several channels in columns
    root: tuple[int, int] | None
    meta: dict[str, Any] | None
    bits: int | None = None  # 16 or 24 for PCM, None for float
    channels: int = 1

    def as_float(self) -> NDArray[np.float64]:
        """Return the samples as float64 in [−1, 1)."""
        scale = 1.0 if self.bits is None else float(2 ** (self.bits - 1))
        return self.samples.astype(np.float64) / scale


def read(path: str | os.PathLike[str]) -> Wav:
    """Read a WAV of 32-bit float or 16- or 24-bit PCM, mono or with several channels.

    This reads what `write` writes, and the beds and samples that audition plays
    against; several channels come back as columns.
    """
    found = chunks(path)
    if b"fmt " not in found or b"data" not in found:
        msg = f"{path}: missing fmt or data chunk"
        raise ValueError(msg)
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", found[b"fmt "])
    if tag == EXTENSIBLE and len(found[b"fmt "]) >= EXTENSIBLE_FMT:
        tag = struct.unpack_from("<H", found[b"fmt "], 24)[0]
    raw = found[b"data"]
    frame = channels * bits // 8 if bits % 8 == 0 and channels > 0 else 0
    if not frame:
        msg = f"{path}: unsupported format ({bits}-bit samples, {channels} channel{'' if channels == 1 else 's'})"
        raise ValueError(msg)
    raw = raw[: len(raw) // frame * frame]
    samples: NDArray[np.float32] | NDArray[np.int32]
    if (tag, bits) == (FLOAT, FLOAT_BITS):
        samples, pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32), None
    elif (tag, bits) == (PCM, PCM_BITS[0]):
        samples, pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32), bits
    elif (tag, bits) == (PCM, PCM_BITS[1]):
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples, pcm = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8, bits  # sign-extend from bit 23
    else:
        msg = f"{path}: expected 32-bit float or 16/24-bit PCM, found format {tag} with {bits} bits"
        raise ValueError(msg)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    root = struct.unpack_from("<9I", found[b"smpl"])[3:5] if b"smpl" in found else None
    meta = json.loads(found[PRIVATE].decode("utf-8")) if PRIVATE in found else None
    return Wav(rate, samples, root, meta, pcm, channels)
