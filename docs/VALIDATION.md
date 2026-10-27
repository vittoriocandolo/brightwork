# Validation record

This records the numerical criteria of the plan (§8.1), the reference method
(§8.2), the adversarial grid (§8.3), the frozen rendering policy, the refit of
the validated-domain rule, the references by measurement and their
neighbourhoods, two SFZ players, and performance. The listening half of the plan (§8.4 by ear, and §8.5) needs a
person; its state is recorded at the end.

**Machine:** Intel Xeon @ 2.80 GHz, 4 cores, a virtual machine on which OpenBLAS selects its SkylakeX (AVX-512) kernels; Linux 6.18.
**Software:** Python 3.11.15, NumPy 2.4.6 with OpenBLAS 0.3.31, SciPy 1.17.1, threadpoolctl 3.7.0, soundfile 0.14.0; sfizz 1.2.3 and liquidsfz 0.4.1, built from source.

**Rattles.** When rattles joined the model (plan §2.12), the 32× grid, the drive-headroom brute force and the references by measurement were run again, on a Qualcomm Oryon, 12 cores, under WSL 2 (Linux 6.18), with Python 3.15.0, NumPy 2.5.4 and SciPy 1.18.1. Those sections give the new figures; the 16× grid and every other section are from the machine above.

**Storage.** When storage joined the model (plan §2.13), with fifteen more references, the 32× grid, the drive-headroom brute force, the references by measurement and their neighbourhoods were run again on the Oryon, and the rule's constant refit; those sections give the figures.

## Numerical criteria (§8.1)

Every criterion is a test in `tests/`. The *measured* column gives the value on
this machine.

| Area | Pass criterion | Measured |
|---|---|---|
| Curves | 10⁻¹² relative; mapped shapes keep their peak at exactly 1 and their segments monotone; jumps scale by r⁻², r⁻³, s², s³ | Interior slopes agree with SciPy's PCHIP to 1.1·10⁻¹⁵ of the largest secant over 2,000 random knot sets, and evaluation with an independent Hermite spline to 10⁻¹². A mapped shape is the shape at the mapped time, its peak exactly 1 and its segments monotone; its jumps scale as stated (random shapes, r and s up to 1000). |
| Tables | Line spectra against direct summation at common instants, ≤ 10⁻¹²; period exactly 8 s | 1.0·10⁻¹⁴ of the carrier's RMS at 200 random instants of a full-band noise; the carrier repeats after 8 s to the sample. |
| Interpolator | ≤ 10⁻¹³ of the carrier's RMS against direct summation with integer-reduced phases, both families; bit-identical across partitions | 6.6·10⁻¹⁴ to 7.1·10⁻¹⁴ for every shipped factor; values at arbitrary instants within 10⁻¹³ of the RMS and slopes within 10⁻¹² of the RMS slope; identical samples for any block partition, render length and thread count. |
| Noise | 200-take mean square 0.5 ± 0.5%; identical bands bit-identical; two bands of one noise correlate as their skirts overlap, noise and modes not at all | 0.50017 (+0.03%). Identical bands give identical lines. Over 200 takes, 2–4 and 3–9 kHz correlate at 0.832, their skirts' cosine similarity within 4σ, and noise against modes at 0.000 ± 0.003. |
| Modes | One mode per cell, uniform within cells (KS over 2,000 seeds), identical in every take; Rayleigh amplitudes and uniform phases across takes; within one bin of s·(i + u_i), which `pitch` scales | KS p = 0.89 for the places, 0.20 for the amplitudes and 0.77 for the phases. Every mode within one bin of its place, before and after `pitch`, and the same in every take. |
| Tones | Squares: every odd harmonic up to 20 kHz at 1/n and nothing else; partials within 1/16 Hz; strike at zero phase in every take; free phases uniform | As stated; the analog reference's partials sound within 0.05 Hz of the values written. |
| Damping and bloom | Windows sum to 1 within 10⁻¹⁵; at 0 one table, bit-identical; each centre's fitted decay rate s_b times the shape's within 0.1%, its peak at r_b·t_pk; the mix within the computed excess | Windows sum to 1 within 10⁻¹⁵. At sub-band centres the decay rate is 2.00005 and 4.0002 times the shape's at s = 2 and 4 (≤ 0.005%), and each centre's envelope is the shape on its time map to 10⁻¹¹ of the peak, rise included. Between centres the mix is at most 0.89, 2.38 or 4.47 dB louder and 0.61, 0.81 or 0.78 dB quieter than the power law at 20, 40 or 60 dB down. |
| Chatter | Modulation at f_c with depth d within 1%; none when d = 0 or the shape is 0 | The gated source equals the plain one times 1 − d(1 − cos 2πf_c t)/2 to 10⁻⁹; exactly the plain source at d = 0. |
| Damper | Multiplies every source exactly; the file ends at its last knot | Exact to 10⁻¹³; earlier samples unchanged; the source length is the last knot. |
| Drive | Identities | 0 dB is the identity; S(±1) = ±1 exactly; odd; small-signal gain 10^(drive/20) to 10⁻⁶; S′ and S″ match numerical derivatives. |
| Dynamics | `softer` on every reference, 0–36 dB | Levels fall by exactly DB, rises lengthen by v^(−1/5), upper edges scale by v^(1/5), and `softer` composes within its written rounding, on all twenty-eight references; every reference takes the whole ladder. |
| Decimators | Frozen coefficients | The snare's designs, bit for bit (table below). |
| Kink bound | Never exceeded by the measured error on an undriven grid fixture | Never exceeded, with or without drive, rattles and storage included. On the 533 undriven fixtures that measure an error above rounding it lies 6.7–60.2 dB above it (median 20.6 dB), and on the 208 driven ones where it is at least twice the saturation term it is never exceeded either. |
| Rendering accuracy | E_max and E_rel ≤ −100 dB for every fixture inside | All 887 fixtures inside at 32× pass, and all converged: the worst E_max is −113.8 dB, the worst rattle's −126.2 dB and the worst stored fixture's −136.8 dB. |
| Drive headroom | The report's value against full renders on every fixture; dense scans | Never optimistic beyond its 0.00001 dB resolution on the 1,206 fixtures whose status changes within 0–48 dB, and exact to 0.001 dB on 944; at most 0.419 dB conservative within 3 dB of the edge; right on all 163 that never change; no change of status between candidates in 377 dense scans. |
| Rattles | Collisions per take, thinned by the shape, at unit mean square at the peak; exact and fast evaluation | Over 200 takes the collisions in each stretch of a shake are its rate times the shape's area there, within 4σ; R's mean square is 1 within 0.1 where the shape is 1, and the thinned stream's where it is ½; a dense shake's energy varies by 0.2 dB (standard deviation) between takes, against 0.8 dB for a Poisson stream; the run-summed evaluation equals the collisions summed one by one within 10⁻¹³, however the samples are split; every collision's jumps in x″ and x‴ match the model's one-sided derivatives. |
| Storage | Values every D-th internal sample, rounded to their bits against the peak; the playback the closed form, smooth and causal; renders validated driven and choked | Every stored value lies on its grid of 2^(bits−1) − 1 steps, the peak exactly on it; the filter is 3 dB down at its frequency to 10⁻¹⁴; a constant plays back as itself to 10⁻¹², nothing before its first value and nothing 2L after its last; the playback is the same at F and 4F, however it is split, and as summed value by value, to 10⁻¹²; stored closed and open hats under 12–18 dB of drive, the open one choked, are inside and agree with the reference below −200 dB, their only kinks the damper's. |
| Loudness | EBU Tech 3341, mono and stereo, every rate; K-weighting | The 48 kHz coefficients match BS.1770's to 10⁻¹⁴, the other rates match its response within 0.03 dB. The −23 dBFS 1 kHz sine in both channels reads −22.99 to −23.01 LUFS at all four rates; one channel alone reads 3.01 dB lower, and a second, different channel adds its energy. |
| Export | Independent readers, mono and stereo; each file carries its patch and gains | soundfile, `scipy.io.wavfile`, Python's `wave` and an independent chunk walker read identical samples and headers, float and PCM, mono and stereo. A stereo file's channels equal its takes' mono renders; each channel of a PCM pair is dithered with its own take. Each file's `cymb` chunk holds the first take's patch and the file's gains, and its samples follow from them. |
| Kits | The SFZ mapping, parsed back and played | Every key, velocity 1–127 and take maps to exactly one existing file; the level, measured from the files, follows 40·log₁₀(v/127) within 0.01 dB, never stepping more than one velocity step across a boundary; chokes as declared; a kit re-renders byte-identically. Two players load it (below). |
| Takes | Seeds 1–4 × takes 1–8 of each reference: one gain per set; spread ≤ 1 dB, with level matching where flagged | One gain per set. Spreads 0.07–0.66 dB for the open hat, crash, ride and bell; 1.4, 1.9 and 1.5 dB for the closed, pedal and analog hats, whose sets the report flags and level matching brings within 0.0001 dB (below). |
| Performance | The budget of §3 | Within budget in every case on this machine: the 5 s crash in at most 1.11 s and the 8 s ride in at most 0.87 s at any rate, the 96-file kit in 8.1 s, and at most 253 MiB above the baseline. |

### Decimators

Designed once by `tools/design_decimator.py`, as the snare's, and shipped with
SHA-256 checksums that a test compares with the snare's: a Kaiser-window
linear-phase prototype at 116 dB, converted with
`minimum_phase(method='homomorphic', half=False)` at 2²⁰ points and
renormalized so the exactly rounded sum of the taps is 1.

| Family and rate | Taps | Length | 99% of energy within | Passband deviation | Stopband |
|---|---|---|---|---|---|
| 44.1 kHz, 16× | 2,593 | 3.67 ms | 0.85 ms | 1.7·10⁻⁵ dB | 115.7 dB |
| 48 kHz, 16× | 1,447 | 1.88 ms | 0.52 ms | 1.5·10⁻⁵ dB | 115.9 dB |
| **44.1 kHz, 32×** | **5,183** | 3.67 ms | 0.85 ms | 3.7·10⁻⁵ dB | 115.6 dB |
| **48 kHz, 32×** | **2,891** | 1.88 ms | 0.52 ms | 1.8·10⁻⁵ dB | 115.1 dB |
| 44.1 kHz, 64× | 10,363 | 3.67 ms | 0.85 ms | 1.9·10⁻⁵ dB | 115.4 dB |
| 48 kHz, 64× | 5,781 | 1.88 ms | 0.52 ms | 1.6·10⁻⁵ dB | 115.2 dB |

Decimation is one matrix product per block (plan §3). For every shipped
design at its base and double rate it agrees with `upfirdn` within 9·10⁻¹⁷ of
Σ|h|·max|y| on Gaussian inputs of 1 to 600,000 samples; it gives identical
outputs for any split of the work and any render length; and changing one
input changes exactly the outputs whose window holds it. The 96 and 88.2 kHz
exports equal the 48 and 44.1 kHz ones with samples filled in, bit for bit.

### Interpolators

`tools/design_interpolator.py` designs the carriers' interpolators from the
table rate R = 2 × base to F: Kaiser-windowed sincs at 280 dB, cut off at R/2,
flat to 20 kHz and stopped from R − 20 kHz, with 34 taps per phase at 96 kHz
and 36 at 88.2 kHz, shipped with checksums. The reference rates use the same
design, made on the fly.

## Reference method (§8.2)

`cymbal/reference.py` renders the take at an absolute reference rate, 128 ×
base (6.144 MHz for the 48 kHz family), with carriers from the same tables
through interpolators of the same design at that rate. It applies a
zero-phase mask flat to 0.4F and a raised cosine to zero at 0.5F, in segments
with guard bands, keeps every (128/K)-th sample, and runs the result through
the same frozen decimator, at unity gain, with no time shift or gain matching.
A reference counts only if it agrees with the 256 × base render to −120 dB
(10⁻⁸ absolute for near-silent fixtures), 20 dB below the threshold. On its
own synthesis path at the shipping rate, the reference equals the shipping
internal signal to 10⁻¹⁴. It decimates with `upfirdn`, apart from the
shipping path's matrix products; the two agree within 9·10⁻¹⁷ of full scale,
far below every error measured here.

- 1,233 of the 1,369 grid references at 32× converged, with a median
  agreement of −205 dB.
- The 512 × base check (24.6 MHz) on 136 fixtures agreed at least as well as
  the 128/256 pair in every case.
- The 136 unconverged references all have at least 10.6 dB of drive and are
  extreme corners: noise under 36–48 dB of drive or at +24 dB under 24 dB,
  four layers at +24 dB and the louder crossfades, every export rate under
  36 dB, one 50 µs tick at +12 dB under 24 dB, 26 rattles under 12–24 dB and
  32 random draws. Their saturated signal still aliases at 12.3 MHz. The rule
  puts none of them inside: the closest is 76.3 dB outside. Every stored
  fixture converged.

## The adversarial grid (§8.3)

`tools/grid.py` builds 1,369 fixtures from combined corners, not one knob at a
time, and runs each at 32× (`docs/grid/k32.csv`). The 16× run
(`docs/grid/k16.csv`) is the record of the 949 fixtures before rattles, which
chose K.

| Family | Fixtures | Covers |
|---|---|---|
| noise | 240 | the hat, upper hat, crash, dark, top 1 kHz and narrowest (1 Hz) bands × spacing 0, 1, 16 and 200 Hz × level and drive: 0 dB under 0–48 dB of drive, +12 dB under 0 and 12, +24 dB under 0 and 24; over the closed hat's plates |
| damping | 120 | damping 0–3 × bloom 0–3 from 20 Hz, 300 Hz and 6 kHz, on short and 8 s shapes, × drive 0 and 24 dB, where the valid domain admits them |
| chatter | 36 | depth 0.25–1 × rate 10–4,000 Hz × drive 0–24 dB |
| ticks | 108 | 10, 20 and 50 µs rises under full-scale bright, hat and upper-hat noise, continuous or as 1 Hz modes, × level 0 and +12 dB × drive 0–24 dB |
| tones | 33 | square banks at the analog frequencies and eight times higher, through three bands, × drive 0–36 dB; struck sine partials to 19.5 kHz with 10–50 µs rises × drive 0–24 dB |
| chokes | 16 | `choke` at 50 and 300 ms with releases of 1–100 ms over a ringing source × drive 0 and 24 dB |
| layers | 13 | four coherent overlapping sources at +24 dB × drive 0–48 dB; crossfades between spacings and bands at 0 and +12 dB |
| dynamics | 195 | `softer` ladders of 0–36 dB from every reference, stored ones stored |
| rattles | 198 | collision rates 20–4,000 a second × rings of 1–200 ms over a shaker's shell, a tambourine's jingles and the whole band × level and drive, where at most 64 collisions ring at once; a tambourine of jingles and a head sharing their collisions, under chatter, × drive 0–24 dB |
| storage | 76 | 4, 6, 8 and 12 bits × storage rates of 12, 32 and 64 kHz × playback filters at 4, 12 and 20 kHz over the hat's plates, × drive 0 and 24 dB; a long ring choked, and the analog squares, stored at 6 bits and 32 kHz, × drive 0 and 24 dB |
| seeds | 90 | six representative fixtures across seeds 1–4 × takes 1–4 |
| rates | 18 | 44.1, 88.2 and 96 kHz exports × drive 0–36 dB, plain and +6 dB |
| random | 220 | the whole valid domain; three sources in ten rattle, and one fixture in five is stored |
| longest | 6 | 8 s rides and crashes of three sources at 96 kHz × drive 0–24 dB |

### Results

Each column applies its own refit constant: 0.85 at 16× and 0.76 at 32×. The
16× column is the 949 fixtures before rattles; the 32× column, all 1,369.

| | 16× | 32× |
|---|---|---|
| Pass (E ≤ −100 dB) | 529 | 982 |
| Converged references | 840 | 1,233 |
| Inside the validated domain | 410 | 887 |
| Inside and failing or unconverged | **0** | **0** |
| Worst error inside | −110.4 dB | −113.8 dB |
| Undriven fixtures passing | 358 of 368 | 629 of 630 |
| Kink bound exceeded | never | never |
| Saturation constant for 6 dB of margin | 0.849 | 0.756 |
| Outside but passing (the rule's caution) | 119 | 95 |

Rattles left the rule as it was: the constant that covered every converged
fixture with 6 dB of margin stayed 0.356, and no rattle comes closer to its
kink bound than 11.7 dB. Storage did not: three random draws, stored at 3–12
bits and 5.6–19 kHz under 4–41 dB of drive, alias more than 0.36 predicts,
though at −181 to −225 dB, 80 dB and more under the limit. The constant was
refit by the same criterion to 0.756, set by the one stored at 3 bits and
5.6 kHz (−225 dB), and frozen as 0.76. It costs 14 fixtures that were inside
at 0.36 and about 0.6 dB of every reference's drive headroom, the analog
references' and the bell's excepted, which the kinks or nothing bound.
Errors under 10⁻¹² of the peak (−240 dB) are rounding: neither term need
cover them, and a stored sound's playback rounds near 10⁻¹³.

Per family at 32×:

| Family | Fixtures | Inside | Pass | Unconverged | Worst error inside |
|---|---|---|---|---|---|
| noise | 240 | 127 | 138 | 64 | −115.8 dB |
| damping | 120 | 63 | 68 | 0 | −121.0 dB |
| chatter | 36 | 24 | 25 | 0 | −173.0 dB |
| ticks | 108 | 55 | 90 | 1 | −114.0 dB |
| tones | 33 | 24 | 28 | 0 | −113.8 dB |
| chokes | 16 | 8 | 8 | 0 | −203.8 dB |
| layers | 13 | 4 | 4 | 7 | −154.5 dB |
| dynamics | 195 | 195 | 195 | 0 | −136.3 dB |
| storage | 76 | 55 | 63 | 0 | −136.8 dB |
| rattles | 198 | 101 | 115 | 26 | −126.2 dB |
| seeds | 90 | 56 | 70 | 0 | −117.7 dB |
| rates | 18 | 6 | 6 | 6 | −152.5 dB |
| random | 220 | 164 | 167 | 32 | −115.8 dB |
| longest | 6 | 5 | 5 | 0 | −134.2 dB |

The worst error inside is a struck sine with a 10 µs rise, undriven. The
32× grid took 2.7 CPU-hours, a median of 0.8 s per fixture and at most 256 s,
for the wash 6 dB softer.

### The kink bound

No grid fixture, undriven or driven, measures more than its kink bound, so
the rule gains no term, and a rattle's collisions, every knot of every one a
kink, need none either. On the 533 undriven fixtures that measure an error
above rounding the bound lies 6.7–60.2 dB above it, median 20.6 dB. It adds every knot's
share, while the error of collisions far apart in time never adds up, so on
dense rattles it is cautious: a shaker's lies 35 dB above its error. The one undriven fixture that
fails at 32×, a random draw at −92.1 dB, is outside by the kink bound alone.
Inside the domain, the prediction is never closer than 3.8 dB to the measured
error, on a struck sine with a 10 µs rise under 24 dB of drive. At 16× ten
undriven fixtures fail, nine of them 10 µs rises, all outside by the bound.

### Drive headroom

`tools/headroom.py` compares the headroom each report computes from its one
render with brute force on every grid fixture (`docs/grid/headroom.csv`). The
saturator's input is synthesized once per fixture, and every candidate the
report considers, every 0.5 dB and both sides of every drop of the knee slew,
is a full render from it: saturator, decimator, output peak, the knee slew
measured directly and the kink bound. The search starts at the fixture's own
drive and stops at the first change of status, whose bracket is bisected with
full renders to 0.00001 dB. On fixtures whose edge is within 6 dB, it also
scans every 0.05 dB up to the edge. In all, 1,369 fixtures and 100,210 full
renders:

- **Never optimistic.** On the 1,206 fixtures whose status changes within
  0–48 dB, the report is at most 0.00001 dB beyond the brute force, its
  resolution. It is exact to 0.001 dB on 944, conservative by 0.001–0.1 dB on
  199, by 0.1–1 dB on 44 and by 1–3.4 dB on 19.
- **Exact where it matters.** Within 3 dB of the edge (197 fixtures) it is at
  most 0.419 dB conservative.
- **No change of status left.** Where full renders never change status within
  0–48 dB (163 fixtures), the report says so on 161 and reports nothing for
  the 2 that are silent.
- **No islands missed.** The 377 dense scans found no change of status
  between the candidates.

The largest gaps are on the densest rattle and on square banks, fixtures
where the kink bound sets the edge. On the square banks the output peak at the edge's drive lies 1.4–2.2 dB above
what the probes see. The limit scales with the peak and the kink term does
not, so a peak seen too low can only lower the estimate.

## The frozen policy

**K = 32.** The plan's criterion is the smallest K within the budget for which
every reference is inside the validated domain with at least 3 dB of drive
headroom, over seeds 1–4 × takes 1–4. At 16× the analog reference, whose
square oscillators start in 20 µs, is inside in only 7 of its 16 seeds and
takes. At 32× every reference is inside with at least 10 dB of drive headroom,
so 64× was not needed.

**The rule's constant.** C started at the snare's 0.5 and was refit as the
smallest value that covers every converged fixture with 6 dB of margin: 0.356
at 32×, set by a damping fixture under 24 dB of drive, and frozen as 0.36.
Storage's grid raised it to 0.756, set by a random draw stored at 3 bits and
5.6 kHz under 35 dB of drive, frozen as 0.76 (above). The
same form would need 0.85 at 16×, set by 4 kHz chatter under 12 dB of drive,
so the constant belongs to the K it was fitted at. No mechanism was found that
the rule misses at 32×: the kink bound is never exceeded, with or without
drive.

## The references by measurement (§8.4)

`tools/references.py` measures every reference over seeds 1–4 × takes 1–8 at
its own output settings.

![Every reference's drive headroom](figures/headroom.svg)

| Reference | LUFS, seed 1 take 1 | Mean of 32 | Largest take spread | Largest peak | Inside | Smallest margin | Smallest drive headroom |
|---|---|---|---|---|---|---|---|
| closed | −32.8 | −32.5 | 1.40 dB | −5.64 dBFS | all | 26.5 dB | 23.7 dB |
| open | −32.6 | −32.5 | 0.66 dB | −14.19 dBFS | all | 30.3 dB | 33.5 dB |
| pedal | −32.1 | −32.5 | 1.90 dB | −4.10 dBFS | all | 69.0 dB | 25.5 dB |
| crash | −32.5 | −32.5 | 0.63 dB | −11.57 dBFS | all | 36.8 dB | 31.5 dB |
| ride | −32.5 | −32.5 | 0.63 dB | −13.17 dBFS | all | 26.3 dB | 34.7 dB |
| bell | −32.5 | −32.5 | 0.07 dB | −18.87 dBFS | all | 34.1 dB | 47.1 dB |
| analog | −33.3 | −33.1 | 1.49 dB | −1.43 dBFS | all | 6.2 dB | 11.3 dB |
| shaker | −32.6 | −32.6 | 1.29 dB | −6.60 dBFS | all | 12.2 dB | 23.3 dB |
| tambourine | −32.4 | −32.5 | 1.94 dB | −11.49 dBFS | all | 34.1 dB | 32.8 dB |
| offbeat | −32.5 | −32.4 | 0.76 dB | −11.45 dBFS | all | 26.7 dB | 29.2 dB |
| tight | −33.1 | −32.6 | 1.95 dB | −3.12 dBFS | all | 27.8 dB | 21.3 dB |
| dead | −33.0 | −32.5 | 2.81 dB | −1.15 dBFS | all | 42.4 dB | 21.0 dB |
| wash | −32.6 | −32.5 | 0.51 dB | −18.24 dBFS | all | 135.3 dB | 40.6 dB |
| house-open | −32.6 | −32.5 | 0.73 dB | −12.33 dBFS | all | 26.2 dB | 31.8 dB |
| house-ride | −32.5 | −32.4 | 0.71 dB | −11.94 dBFS | all | 24.8 dB | 32.7 dB |
| house-tamb | −32.1 | −32.3 | 2.54 dB | −8.64 dBFS | all | 34.8 dB | 28.9 dB |
| shaker-16 | −32.9 | −32.4 | 2.36 dB | −1.09 dBFS | all | 14.9 dB | 20.7 dB |
| top | −32.7 | −32.5 | 1.38 dB | −9.68 dBFS | all | 14.0 dB | 27.7 dB |
| trance-crash | −32.4 | −32.4 | 0.57 dB | −13.05 dBFS | all | 37.1 dB | 33.0 dB |
| trance-ride | −32.5 | −32.4 | 0.51 dB | −14.57 dBFS | all | 25.6 dB | 35.2 dB |
| analog-open | −32.6 | −32.5 | 0.47 dB | −9.63 dBFS | all | 6.8 dB | 19.6 dB |
| analog-cymbal | −32.6 | −32.5 | 0.22 dB | −12.64 dBFS | all | 6.6 dB | 23.2 dB |
| machine-closed | −32.6 | −32.5 | 1.27 dB | −8.72 dBFS | all | 153.9 dB | 26.5 dB |
| machine-open | −32.5 | −32.4 | 0.73 dB | −14.21 dBFS | all | 186.9 dB | 33.7 dB |
| half-open | −32.6 | −32.4 | 0.88 dB | −10.48 dBFS | all | 29.0 dB | 30.0 dB |
| sizzle | −32.8 | −32.5 | 0.68 dB | −15.67 dBFS | all | 13.6 dB | 35.8 dB |
| china | −32.6 | −32.5 | 0.87 dB | −8.43 dBFS | all | 30.8 dB | 27.9 dB |
| splash | −32.4 | −32.4 | 0.97 dB | −11.06 dBFS | all | 35.0 dB | 30.4 dB |


The spread is the largest over the four seeds of the loudness range across a
seed's eight takes. The means span 0.70 dB. Every take peaks below −1 dBFS
and is inside with at least 3 dB of drive headroom. The closed hat's seeds 1
and 4, every seed of the pedal, analog, tight and dead hats, of the shaker,
the house tambourine and the 16th shaker, three of the tambourine's and the
machine's closed hat's, and one of the top's spread over 1 dB (up to 2.81 dB,
the dead hat's, whose damped plates leave the shortest effective length); the
report flags each of those sets, and `--level-match` brings each to one
loudness within 0.0001 dB. The thirteen references from before storage
measure as they did, to the digit, but for their drive headroom, 0.6 dB less
under the refit constant (the analog hat's, bound by its kinks, unchanged). The two machine references keep their margins far from
the rule's limit: storage plays back through a smooth filter, so only their
dampers' knots are kinks.

Voicing by measurement set the levels and left the timbres as drafted,
pending listening:

- **Loudness.** The references are matched at −32.5 LUFS (the analog hat
  −33.1), through their source levels, so they share one output setting
  (−6 dB gain) and combine into any kit. The analog hat binds the target: a
  60 ms hit of square partials has the highest crest of the references.
- **Pitch room.** The bell's tone and tick bands now stop at 100 Hz and
  16 kHz, and the analog hat's band is 6–16.5 kHz, so `pitch` moves them from
  −12 to +3 semitones and `softer` takes their whole 36 dB ladder; a band edge
  at 20 Hz or 20 kHz cannot move, and `pitch` refuses rather than clamps.
- **Take spread.** A short hit's loudness depends on the time–bandwidth
  product of its effective length, not on its number of modes: the closed
  hat's plates vary by 0.3–0.5 dB (standard deviation) over takes whether made
  of modes or of continuous noise. The plan had estimated a small spread for
  hats; the closed, pedal and analog hats spread by 1.4–1.9 dB, and the
  report flags their sets for level matching.

Each control that §10 asks to earn its place moves its measurement on its
reference:

| Control | Measurement on its reference | With | Without |
|---|---|---|---|
| Spacing | Crash tail, 1–1.5 s: spectral flatness over 300–2,000 Hz (lower is more tonal) | 0.023 (modes, 18 Hz) | 0.042 (continuous noise) |
| Damping | Crash: centroid at 1–2 s over the strike's | 0.06 (0.9) | 1.07 (0) |
| Bloom | Crash: 8–18 kHz energy at 20–60 ms over 0–4 ms | 22.3 (0.4) | 1.1 (0) |
| Chatter | Open hat: envelope modulation over 20–300 Hz | 0.74 (0.25) | 0.62 (0) |
| The damper | Open hat closed at 250 ms by `choke`: silent 47 ms later | 297 ms | 1.3 s |
| Square tones | Analog hat: energy in its band | +27.7 dB | sines |
| Stereo pairs | Crash: correlation between the channels | −0.03 | 1 (mono) |
| Rattle | Shaker, 10–100 ms, takes 1–4: standard deviation of the 2 ms envelope over the 20 ms one (higher is more granular) | 0.44 (collisions) | 0.12 (the plain sources) |
| Storage's playback | Machine open hat, 0–100 ms: energy at 12–20 kHz over 2–8 kHz | 0.32 (6 bits at 32 kHz, through 11 kHz) | 0.98 (not stored) |
| Storage's bits | Machine open hat, its last 100 ms, takes 1–4: the rattle's granularity | 0.38 (6 bits) | 0.24 (16 bits) |

Whether each control earns its place is a listening judgement (§8.4). The
usability tasks below show that each is reachable in a few edits.

### Usability (§8.4)

`tests/test_usability.py` runs every task of §8.4 from its reference, each in
at most three field edits or one command, and checks that its measurement
moves the expected way: tighten, loosen, darken and brighten the closed hat,
add stick and make a ghost note; shorten, half-open, close on a beat and
darken the open hat; shorten and darken the pedal; lengthen, darken the tail
of, add bloom to and choke the crash, and make a splash and a china from it;
add ping and wash to the ride and darken it; retune and shorten the bell; open
the analog hat, turn it into its cymbal, retune and darken it; loosen,
smooth, lengthen and brighten the shaker; shake the tambourine longer, choke
it and thin its shiver; shorten the off-beat hat for a faster tempo and close
it on the next 16th; loosen and brighten the tight hat; take the towel off the
dead hat; take the swell out of the wash and brighten it; close the house
open hat on the next 16th and darken the house ride; shake the house
tambourine and the 16th shaker longer; drop the top's rattle; shorten the
trance crash and darken the trance ride; retune the analog open hat and
lengthen the analog cymbal; store the machine's closed hat in fewer bits and
open its open hat longer by its damper; stop the half-open hat's plates
slapping and still the sizzle's rivets; darken the china and shorten the
splash. All pass. A rattle's grain is measured as the
fluctuation of its 2 ms envelope about its 20 ms one, over four takes.

Three recipes taught something. A darker band alone lengthens a cymbal's
measured decay, because what remains is its slow low end, so the splash and
the china shorten with a damper. More damping darkens a strike's first
milliseconds as well as its tail, so a splash brightens through its band. And
the closed hat's tick shows below 6 kHz, where it reaches lower than the
plates.

### Neighbourhoods (§5, §8.4)

`tools/neighbourhoods.py` moves one control at a time on each reference's
main source (the hats' plates, the crash's modes, the ride's wash, the bell's
partials, the analog hat's squares, the shaker's grains, the tambourine's
shiver, the analog references' squares, the house tambourine's shiver, and
the plates or wash of the others), everything else as the reference has it, and renders every value over seeds 1–4 × takes 1–4. Spacing, damping,
bloom and chatter are scanned over their whole valid ranges, damping and
bloom in steps of 0.25 and chatter in steps of 0.1. The low band edge runs
from three octaves below to three above in half octaves, the high edge from
two below to one above in quarter octaves, and the tail length, by `stretch`,
from 0.25 to 4 times in steps of √2. On a source that rattles, the collision
rate runs from 23 to 2,900 a second and the ring from 1 to 256 ms, both in
steps of √2. On a stored reference, the bits run from 2 to 16, the storage
rate from 4 to 91 kHz and the playback filter from 1 to 16 kHz, both in
steps of √2. A control's extent is the run of values
around the reference's own at which all 16 renders are inside. An end in
italics is where the patch refuses the next value, an end in bold is where
the next value is outside the validated domain, and any other end is the
scan's. Drive is each reference's smallest drive headroom over seeds 1–4 ×
takes 1–8 (above).

| Reference | Low band edge | High band edge | Spacing | Damping | Bloom | Chatter | Collisions a second | Ring | Tail length | Drive |
|---|---|---|---|---|---|---|---|---|---|---|
| closed | 750 Hz – *12 kHz* | *6.7* – *19 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 23.7 dB |
| open | 625 Hz – *14.1 kHz* | *5.7* – *19 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 33.5 dB |
| pedal | 312 Hz – *10 kHz* | 3 – *17 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 25.5 dB |
| crash | 50 Hz – 3.2 kHz | 4.5 – *18 kHz* | 0 – 200 Hz | 0 – *1.75* | 0 – *1.75* | 0 – 1 | | | 0.25 – *1.41×* | 0 – 31.5 dB |
| ride | 62.5 Hz – 4 kHz | 4 – *19 kHz* | 0 – 200 Hz | 0 – *1.75* | 0 – *1.75* | 0 – 1 | | | 0.25 – *1×* | 0 – 34.7 dB |
| bell | *25* – 800 Hz | 4 – *19 kHz* | | 0 – *2* | 0 – *2* | 0 – 1 | | | 0.25 – *2×* | 0 – 47.1 dB |
| analog | 750 Hz – *12 kHz* | *6.9* – *19.6 kHz* | | 0 – **2.5** | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 11.3 dB |
| shaker | 375 Hz – *8.5 kHz* | *3.6* – *17 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | 23 – 3,000 | 1 – *16 ms* | 0.25 – 4× | 0 – 23.3 dB |
| tambourine | 312 Hz – *10 kHz* | 3 – *17 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | 23 – *1,024* | 1 – *128 ms* | 0.25 – 4× | 0 – 32.8 dB |
| offbeat | 1 kHz – *16 kHz* | *9* – *18 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 29.2 dB |
| tight | 688 Hz – *11 kHz* | *5.9* – *19.8 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 21.3 dB |
| dead | *884 Hz* – *7.1 kHz* | *3* – 20 kHz | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 21.0 dB |
| wash | 50 Hz – 3.2 kHz | 3.5 – *19.8 kHz* | 0 – 200 Hz | 0 – *1.75* | 0 – *1.5* | 0 – 1 | | | 0.25 – *1×* | 0 – 40.6 dB |
| house-open | 562 Hz – *12.7 kHz* | *5.3* – *17.8 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 31.8 dB |
| house-ride | 75 Hz – 4.8 kHz | 4 – *19 kHz* | 0 – 200 Hz | 0 – *1.75* | 0 – *1.75* | 0 – 1 | | | 0.25 – *2.83×* | 0 – 32.7 dB |
| house-tamb | 375 Hz – *12 kHz* | 3.25 – *18.4 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | 23 – *1,448* | 1 – *90.5 ms* | 0.25 – 4× | 0 – 28.9 dB |
| shaker-16 | 438 Hz – *9.9 kHz* | *3.86* – *18.4 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | 23 – 3,000 | 1 – *16 ms* | 0.25 – 4× | 0 – 20.7 dB |
| top | 750 Hz – *12 kHz* | *6.73* – *19 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 27.7 dB |
| trance-crash | 62.5 Hz – 4 kHz | 4.75 – *19 kHz* | 0 – 200 Hz | 0 – *1.75* | 0 – *1.75* | 0 – 1 | | | 0.25 – *1×* | 0 – 33.0 dB |
| trance-ride | 112 Hz – 7.2 kHz | 4.5 – *18 kHz* | 0 – 200 Hz | 0 – *2* | 0 – *2* | 0 – 1 | | | 0.25 – *2×* | 0 – 35.2 dB |
| analog-open | 750 Hz – *12 kHz* | *6.94* – *19.6 kHz* | | 0 – **2.5** | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 19.6 dB |
| analog-cymbal | 438 Hz – *14 kHz* | 4.5 – *18 kHz* | | 0 – **1.75** | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 23.2 dB |
| machine-closed | 625 Hz – *14.1 kHz* | *5.3* – *17.8 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 26.5 dB |
| machine-open | 625 Hz – *14.1 kHz* | *5.3* – *17.8 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 33.7 dB |
| half-open | 500 Hz – *11.3 kHz* | *4.16* – *19.8 kHz* | 0 – 200 Hz | 0 – 3 | 0 – 3 | 0 – 1 | | | 0.25 – 4× | 0 – 30.0 dB |
| sizzle | 62.5 Hz – 4 kHz | 4 – *19 kHz* | 0 – 200 Hz | 0 – *1.75* | 0 – *1.75* | 0 – 1 | | | 0.25 – *1×* | 0 – 35.8 dB |
| china | 87.5 Hz – 5.6 kHz | 3.5 – *19.8 kHz* | 0 – 200 Hz | 0 – *2* | 0 – *2* | 0 – 1 | | | 0.25 – *2.83×* | 0 – 27.9 dB |
| splash | 188 Hz – 12 kHz | 4.5 – *18 kHz* | 0 – 200 Hz | 0 – *2.5* | 0 – *2.5* | 0 – 1 | | | 0.25 – 4× | 0 – 30.4 dB |

The patch refuses a low edge at or above the high edge, an edge beyond
20 Hz–20 kHz, damping or bloom that compresses or stretches 20 kHz more than
1000 times, more than 64 collisions ringing at once, and a sound that would
end after 8 s. Tones have no spacing.

On the two stored references, the machine's closed and open hats, every
value of the storage controls the scan reaches is inside on all 16 seeds and
takes: bits 2 – 16, the storage rate 4 – 90.5 kHz and the playback filter
1 – 16 kHz, every end the scan's. Storage plays back through a smooth filter,
so none of them adds a kink: the two references themselves keep margins of
153.9 and 186.9 dB.

In all twenty-eight neighbourhoods the validated domain bounds a single
control: the damping of the three references made of square tones. The analog
hat and its open hat stop at 2.5: at 2.5 the analog hat's smallest margin is
1.3 dB, and from 2.75 the kinks alone exceed the limit, because the upper
sub-bands of its 20 µs square onsets then decay up to 16 times faster after
the peak, which sharpens the kink there. The analog cymbal stops at 1.75: at 2,
one of its 16 seeds and takes is outside by 2.2 dB, undriven, so by its kinks
alone. Everywhere else, without drive, every value the patch accepts
within the scan is validated on all 16 seeds and takes; with drive, the drive
headroom bounds it. Which parts of these extents are musically useful, the
voiced regions of §5, is for listening to decide.

## Two SFZ players (§7.3, milestone 3)

`tests/test_sfz_player.py` writes a kit (a closed hat in two layers with two
takes each, and an open hat choked by it) and plays a list of notes through
sfizz (`sfizz_render`, built from source) and liquidsfz (its library, through
`tools/liquidsfz_render.cc`). In both:

- every note plays its key's layer for its velocity and its round robin,
  identified by correlation above 0.99 against every other file;
- the softer layer's first take at velocity 40 sounds 40·log₁₀(40/127) dB
  under the loud layer's first take at 127, within 0.05 dB, and both takes of a
  layer get the same gain;
- the closed hat fades the open one over the release and silences it to the
  player's 16-bit floor within 100 ms, where it would still ring.

What the players do differently, measured: sfizz keeps one round-robin
counter per key across its layers, repeats one sample at the first block
boundary after a note starts, and its release falls about 2 dB per ms before
cutting the voice some 50 ms after a 40 ms release. liquidsfz plays its right
channel 1.0125 times its left, and its default interpolation rolls off
16–20 kHz by about 2.4 dB relative to 1–4 kHz. Neither affects the mapping.

After rattles, the true-peak mode and the kit's shared cores, the test ran
again in a throwaway Debian 13 container on the Oryon, with sfizz 1.2.3 and
liquidsfz 0.4.1 built from source: all six checks pass.

## Performance (§3)

`tools/bench.py`, in fresh interpreters, with tables computed for every timed
render (a new take each time) and the report included:

| Sound | Sub-bands | 44.1 kHz | 48 kHz | 88.2 kHz | 96 kHz | Budget |
|---|---|---|---|---|---|---|
| Closed hat, 0.25 s, two sources | 4 | 135 ms | 161 ms | 182 ms | 170 ms | 250 ms |
| Open hat, 1.5 s, two sources | 5 | 252 ms | 227 ms | 277 ms | 232 ms | 500 ms |
| Crash, 5 s, three sources | 27 | 836 ms | 842 ms | 1,042 ms | 1,112 ms | 1.5 s |
| Ride, 8 s, three sources | 18 | 683 ms | 745 ms | 869 ms | 797 ms | 2 s |
| Kit: 3 keys × 4 layers × 8 takes, 96 files | | | 8.1 s | | | 30 s |

Each sound is its reference stretched to the stated length, and the crash
gains a dark wash as its third source. A repeated take,
as audition re-renders it, finds its tables cached: 55–82 ms for the closed
hat, 146–180 ms for the open hat, 656–872 ms for the crash and 506–656 ms for
the ride. Memory above the baseline is 91–130 MiB for the hats and 226–253
MiB for the crash and the ride. Between runs on this machine the times vary
by about 10%.

Where the time goes, for the crash at 88.2 kHz with a new take: tables
321 ms, synthesis 538 ms, the knee-slew record 57 ms, decimation 76 ms, the
true peak 58 ms, the drive-headroom search 28 ms, the probes 22 ms, loudness
14 ms and the kink terms 11 ms.

On the Oryon (12 cores, WSL 2, Python 3.15, NumPy 2.5.4, SciPy 1.18.1), after
rattles, the exact true peak and the kit's shared cores, a new take of the
closed hat renders in 67–77 ms, the open hat in 162–185 ms, the crash in
662–732 ms and the ride in 513–596 ms over the four rates, the 96-file kit in
3.6 s, and memory stays within 318 MiB above the baseline. The exact true peak
costs about 0.13 s of the crash; before the kit's workers shared the cores,
twelve of them each started a thread per core and the kit took 13–48 s,
missing its 30 s budget in the worst runs. On a busier day the same machine
measured up to 1.2 s for the crash and 6.9 s for the kit.

Decimation first ran as parallel `upfirdn` chunks, bit-identical to one call.
Their scalar inner loop took 760 ms of this render, which then took 1,675 ms
and missed its budget. As matrix products in fixed blocks (plan §3) it takes
76 ms, and the headroom brute force on the grid fell from 17,800 to 3,800
CPU-seconds. The grid and the headroom check were rerun on the final
decimator: every recorded figure is unchanged. Errors above −210 dB moved by
at most 0.0001 dB, the largest change was 0.0014 dB on an error near −240 dB,
and the headroom check's results are identical.

## Listening (§8.4, §8.5)

Not yet done; it needs a person. The references are voiced by measurement
only. Recorded so far: the references, their kits from `softer`, their
usability tasks and their validated neighbourhoods, all by measurement. Open:
approval of the twenty-eight references by ear, across seeds 1–4 × takes 1–8; the
four-layer kits in patterns with accents; 16th-note hats with eight takes
against one; closed–open patterns with chokes; the stereo crash and its mono
sum; blind, level-matched comparisons; and the sample-replacement benchmark
of [BENCHMARK.md](BENCHMARK.md).
The open decisions of the plan's §11 that listening settles stay open: the
sub-band step, whether each control earns its place, the voiced ranges,
`softer`'s exponent and whether contact should darken square tones, and the
default choke release.
