# Contributing

Thank you for helping with Brightwork.

## Set up

Brightwork uses [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras
```

## Check a change

```
uv run coverage run -m pytest
uv run coverage report
uv run ruff check
uv run ruff format --check
uv run mypy
uv run typos
uv run zizmor --persona auditor .
```

Coverage runs the tests and requires every line and every branch of
`brightwork_synth` to run, ruff runs every stable rule, mypy checks all the
code strictly, as Python 3.12 would run it, pytest turns any warning into an
error, typos checks the spelling of code and documentation, and zizmor audits
the GitHub workflows. `uv run pytest` alone runs the tests faster.

`uv run ruff check --fix`, then `uv run ruff format`, fix most of what ruff
finds; in that order, the formatter keeps the trailing commas the linter
adds. The formatter warns that the trailing comma rule, COM812, may conflict
with it: on this code they agree, so the rule stays on.

CI runs mypy and the tests on every platform and Python that Brightwork
supports, and the tests again with the oldest supported NumPy, SciPy and
threadpoolctl.

The kit test plays a kit through two SFZ players, sfizz and liquidsfz, when
their renderers are on the `PATH` or named by `SFIZZ_RENDER` and
`LIQUIDSFZ_RENDER`, and is skipped otherwise; `tools/liquidsfz_render.cc` is
the offline renderer it drives liquidsfz with.

## Rules

- **Correct, simple, minimal, elegant, in that order.** A change that makes
  the code shorter but less obviously correct is not an improvement.
- **Renders stay bit for bit the same.** On one platform with the same NumPy
  and SciPy, a change to any rendered sample changes the sound of existing
  patches. It needs a reason, a changelog entry, a new version and new golden
  renders (`uv run python -m tools.golden`). Across platforms and library
  versions, samples may differ by rounding, which is no change: the golden
  renders allow a millionth of full scale.
- **Patches have no defaults.** A new field means a new schema version, and
  older schemas keep rendering exactly as before.
- **Seeds and names are forever.** The keys that seed the modes, the strike,
  the noise, the phases and the dither, `cymbal/modes/v1`, `cymbal/strike/v1`,
  `cymbal/noise/v1`, `cymbal/phase/v1` and `cymbal/dither/v1`, the patch's
  `instrument = "cymbal"` and the WAV chunk `cymb` never change: a new
  behavior takes a new key. They keep the name the instrument was designed
  under, so that the evidence in `docs/` stays the evidence for this code.
- **Every render is judged.** A change to the synthesis keeps the validated
  domain's rule honest: `docs/VALIDATION.md` records the evidence, and the
  adversarial grid in `docs/grid` is checked by the tests.
- **Every behavior has a test,** written as a property that holds on every
  platform, such as "rendering in blocks never changes a sample", rather than
  as a stored hash of a render, which rounding would break. Coverage checks
  that the tests run every line and every branch; code no test can reach is
  removed, not excluded.
- **Errors say where and what.** They name the file or the field path, and the
  problem.
- **NumPy, SciPy and threadpoolctl are the only dependencies,** and
  `sounddevice` only for audition.
- **Every check stays on.** Fix what a linter finds instead of silencing it.
  The exceptions are `assert` in tests, which is how pytest checks, and the
  kit test's calls of the SFZ players' renderers, which are programs it runs.

## Frozen filters

The decimators and interpolators ship as data in `src/brightwork_synth/data`,
checked by SHA-256. `uv run python -m tools.design_decimator --check` and
`uv run python -m tools.design_interpolator --check` design them again with
the current NumPy and SciPy and compare the designs with what ships. Without
`--check` they replace the shipped data, which changes every render.

## Releasing

1. Set the version in `pyproject.toml` and `src/brightwork_synth/__init__.py`;
   a test checks that they agree. Before the first release, replace `OWNER`
   in `[project.urls]` with the repository's account: the release workflow
   refuses the placeholder.
2. Move the changelog's entries under the new version, with the date.
3. Tag the commit `vX.Y.Z` and push the tag.

The release workflow then checks that the tag matches the version, runs the
tests, builds the package and publishes it to PyPI as `brightwork-synth`,
through PyPI's trusted publishing: configure this repository and its `pypi`
environment as a trusted publisher on PyPI once, and no token is needed.
The package is `brightwork_synth`; the command is `brightwork`.

## Commits and pull requests

Write commit subjects in the imperative, such as "Add a 24-bit export", with a
body that explains why. Keep a pull request to one change, with its tests and
documentation.

## License

Brightwork is licensed under the GNU GPL version 3 or later, with the
additional permission in [NOTICE](NOTICE) for the files it writes. By
contributing, you agree to license your contribution under the same terms, so
that the sounds people make with Brightwork stay theirs.
