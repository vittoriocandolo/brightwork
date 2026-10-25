# The sample-replacement benchmark (plan §8.5)

This is where "no more cymbal or hi-hat samples" becomes a measurement. It
needs your own samples and your ears, so it is a protocol to run by hand.

## Result

Not yet run.

## Protocol

1. **Collect.** Gather the 20–30 hi-hat and cymbal samples you actually use
   that fall inside the family: closed, open, half-open and pedal hi-hats;
   crashes, splashes and chinas; rides and ride bells; and drum-machine hats
   and cymbals. Keep their articulations and velocity layers where they have
   them. Gongs, tam-tams, bells and cowbells, rolls, swells, bowing and
   brushes, rivet and sizzle cymbals, and sounds defined by processing or
   storage (rooms, filtering after the fact, the 6-bit cymbals of a TR-909)
   are out of family. List them anyway, marked out of family.
2. **Fix the time limit before starting.** 15 minutes per sample is a
   reasonable choice. Start each from the nearest reference (`brightwork init`),
   and edit with audition watching, in a pattern over your groove:
   `brightwork audition hat.toml --watch --bpm 120 --every 1/8 --with groove.wav --against sample.wav`.
3. **Compare blind and level-matched, in context,** over several takes:
   `brightwork audition hat.toml --watch --bpm 120 --every 1/8 --with groove.wav --against sample.wav --blind --takes 8`.
   Audition matches the sample to the render's largest 400 ms momentary
   loudness (BS.1770) and prints each pair's order when stopped. Record the
   matching method if you use another.
   - Compare hats in patterns with their chokes: voice the closed, pedal and
     open hats together with `--pattern` and `--choke`, and compare the kit
     with the sampled kit in your sampler.
   - Compare stereo cymbals with `--stereo` against stereo samples, and check
     the mono sum.
   - Where the source has velocity layers, derive yours with `brightwork softer`,
     pack them, and compare kits in a groove with dynamics.
4. **Judge and sort.** A pass means you would use the render in place of the
   sample. Sort each failure as a missing control, a voicing-skill issue, or
   out of family.

The claim holds for the fraction that passes.

## Pre-registered additions

Nine additions are pre-registered. Add one only if it fixes recorded failures:

- impulsive sizzle with a density control, for rivets and loose hats;
- chatter at a jittered rate, if the periodic gate buzzes;
- a cascade rule in `softer`, if softer hits need to bloom less and reach less high than contact alone gives;
- take variation that keeps part of each mode's amplitude, if takes of one cymbal sound like different cymbals;
- seeded scatter of modal decay times, if tails sound too even;
- pedal position (MIDI CC4) regions in kits, for electronic hi-hats;
- an extended band at 88.2 and 96 kHz, if repitched samples sound dull;
- ring modulation and pulse width for tones, for drum-machine metallics beyond square banks;
- a post-drive low-pass for the fizz of heavy drive.

Anything else needs its own evidence.

## Record

| # | Sample | Family | Articulation | Start | Minutes | Matching | Result | Failure category | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |

Pass rate: _ / _ in family. Out of family: _.
