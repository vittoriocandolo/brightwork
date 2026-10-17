# Cymbal Synth

A one-voice hi-hat and cymbal synthesizer, rendered offline to WAV. Every
stage is a pure function of time, every render states whether it lies where
its aliasing is demonstrated below −100 dB, and a render depends only on its
patch: the cymbal, which is its seed, and the strike, which is its take.

It renders one-shots, sets of round-robin takes, stereo pairs, and kits:
articulations on keys with velocity layers and choke groups, mapped by SFZ.

```
pip install -e .[audition,test]
cymbal init closed > closed.toml
cymbal audition closed.toml --watch --bpm 120 --every 1/8
cymbal render closed.toml --out closed.wav --takes 8
cymbal pack 42=closed.toml 46=open.toml --choke 46:42 --takes 8 --out hats/
```

- [docs/GUIDE.md](docs/GUIDE.md): the workflow, every control, recipes, kits and chokes, and reading the report.
- [docs/PLAN.md](docs/PLAN.md): the design and the reasoning behind it.
- [docs/VALIDATION.md](docs/VALIDATION.md): the evidence, from the numerical criteria to the adversarial grid and the performance budget.
- [docs/BENCHMARK.md](docs/BENCHMARK.md): the sample-replacement benchmark, a protocol to run by hand.

This is an independent codebase. It carries over the designs and findings of
the Kick Synth and the Snare/Clap Synth, which live on their own branches,
not their code.
