# Brightwork

A one-voice synthesizer of hi-hats, cymbals and the tops beside them, shakers
and tambourines, rendered offline to WAV. Every
stage is a pure function of time, every render states whether it lies where
its aliasing is demonstrated below −100 dB, and a render depends only on its
patch: the cymbal, which is its seed, and the strike, which is its take.

It renders one-shots, sets of round-robin takes, stereo pairs, and kits:
articulations on keys with velocity layers and choke groups, mapped by SFZ.

```
uv tool install ".[audition]"    # the brightwork command, with live audition
brightwork init closed > closed.toml
brightwork audition closed.toml --watch --bpm 120 --every 1/8
brightwork render closed.toml --out closed.wav --takes 8
brightwork pack 42=closed.toml 46=open.toml --choke 46:42 --takes 8 --out hats/
```

- [docs/GUIDE.md](docs/GUIDE.md): the workflow, every control, recipes, kits and chokes, and reading the report.
- [docs/PLAN.md](docs/PLAN.md): the design and the reasoning behind it.
- [docs/VALIDATION.md](docs/VALIDATION.md): the evidence, from the numerical criteria to the adversarial grid and the performance budget.
- [docs/BENCHMARK.md](docs/BENCHMARK.md): the sample-replacement benchmark, a protocol to run by hand.
- [CHANGELOG.md](CHANGELOG.md): what each version changed.
- [CONTRIBUTING.md](CONTRIBUTING.md): setting up, the checks a change passes, and the rules it keeps.

This is an independent codebase. It carries over the designs and findings of
the Kick Synth and the Snare/Clap Synth, which live on their own branches,
not their code.

## License

Brightwork is free software, under the [GNU General Public License](LICENSE),
version 3 or later.

The sounds you make with it are yours. An additional permission in
[NOTICE](NOTICE) lets you use the sounds Brightwork renders, the patches and
SFZ maps it writes, and its reference patches under any terms you choose, in a
track or in a sample pack you sell.
