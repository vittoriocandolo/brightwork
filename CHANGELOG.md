# Changelog

All notable changes to Brightwork are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Within a version, a patch renders
the same samples bit for bit on one platform with the same NumPy and SciPy, and
the same sound on any other, the same samples to within a millionth of full
scale; a change to the sound of an existing patch is a new version.

## [0.1.0] - Unreleased

The first public release.

### Added

- A one-voice hi-hat, cymbal and percussion synthesizer: noise sources of seeded modes or
  continuous noise in a band, tones of square or sine partials struck at zero
  phase or free, damping and bloom that make high frequencies decay faster
  and rise later, chatter that gates a source as plates that slap, a damper
  over every source, and a tanh drive, all computed at 32 times the sample
  rate. A render depends only on its patch: the cymbal is its seed, the
  strike its take.
- The validated domain: every render reports whether it lies where its
  aliasing is demonstrated below −100 dB, from a rule refit on an adversarial
  grid of 949 fixtures, with the cause of any predicted aliasing and the
  drive headroom left before the status changes. `brightwork verify` checks a
  render against a reference rendered at 128 times the base rate, itself
  checked against one at 256.
- Rattles for the tops of a groove, shakers, tambourines, maracas and
  cabasas: any source can sound at a seeded stream of collisions,
  `grains_hz` of them a second at the shape's peak, one to a cell of time,
  thinning out with the shape, each ringing for `grain_ms`, as loud at the
  peak as the plain source. Sources with the same rate share their
  collisions. Every collision is smooth and counted by the validated-domain
  rule.
- Frozen minimum-phase decimators and linear-phase interpolators for the
  44.1 and 48 kHz families, verified by checksum.
- Thirteen references to start from, matched in loudness and sharing one
  output setting: `closed`, `open`, `pedal`, `crash`, `ride`, `bell`,
  `analog`, `shaker` and `tambourine`, and four voiced for trance, tech house
  and the two ends of the axis from machine to band: `offbeat`, `tight`,
  `dead` and `wash`.
- Patches in TOML, read as UTF-8 or UTF-16, with every field required, every
  problem reported with its field path, and a canonical form.
- Editing commands that print a complete patch: `pitch` by semitones or to a
  note, `stretch` of chosen curves, `choke` with a 60 dB release, and
  `softer`, the same cymbal struck softer.
- Exports at 44.1, 48, 88.2 and 96 kHz, as float32 or as 24- or 16-bit PCM
  with seeded TPDF dither, at a gain, a sample peak or a true peak, measured
  exactly between the samples, with the root key in a
  `smpl` chunk and the patch and its provenance in a `cymb` chunk, which
  `brightwork extract` gives back. Overs are reported, never clipped.
- Round-robin sets of takes sharing one gain, with optional level matching,
  and stereo files of consecutive takes.
- `brightwork audition`: live playback that re-renders on every save, on a
  tempo grid with patterns, chokes and a bed, or alternating with a sample
  matched in loudness, in a blind order revealed at the end.
- `brightwork pack`: kits of keys with velocity layers on a square law, takes
  as round robins and choke groups, mapped by an SFZ file and checked in
  sfizz and liquidsfz.
