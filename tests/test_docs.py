# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The documentation's links lead somewhere, every figure and sound is shown, and what it shows is the code's."""

import re
import runpy
import shlex
from pathlib import Path

import pytest

from brightwork_synth import cli, wav
from brightwork_synth.patch import loads
from brightwork_synth.render import render
from docs import build
from tools import balance, readme

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")])
LINK = re.compile(r"\]\(([^)\s]+)\)")  # the target of a Markdown link or image


def prose(page: Path) -> str:
    """Read a page without its code blocks, where brackets and hashes are neither links nor headings."""
    return re.sub(r"^```.*?^```", "", page.read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)


def anchors(page: Path) -> set[str]:
    """Return the anchors that GitHub gives a page's headings."""
    headings = re.findall(r"^#+ (.+)$", prose(page), flags=re.MULTILINE)
    return {re.sub(r"[^\w\- ]", "", heading.strip()).lower().replace(" ", "-") for heading in headings}


def targets(page: Path) -> list[str]:
    """List the targets of a page's links and images, other than web addresses."""
    return [target for target in LINK.findall(prose(page)) if not re.match(r"[a-z]+:", target)]


def test_the_guides_balance_is_what_the_tool_measures() -> None:
    """Measure every preset against Ballast's kicks, and find the guide's table as the tool writes it."""
    guide = balance.GUIDE.read_text(encoding="utf-8")
    start, end = balance.in_guide(guide)
    assert guide[start:end] == balance.table(balance.rows()), "run `uv run python -m tools.balance --write`"
    plain = balance.KICKS.parent / "balance.py"  # a file with no patch from Ballast in it
    with pytest.raises(ValueError, match="no patch from Ballast"):
        balance.ballast_patch(plain)


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_every_link_leads_somewhere(page: Path) -> None:
    """Find every file a page links to, and every heading it links to in it."""
    for target in targets(page):
        path, _, anchor = target.partition("#")
        linked = (page.parent / path).resolve() if path else page
        assert linked.exists(), f"{page.name}: {target}"
        if anchor:
            assert anchor in anchors(linked), f"{page.name}: {target}"


def test_every_figure_and_sound_is_shown() -> None:
    """Find every figure and sound in docs/ linked from some page, so none is left behind."""
    linked = {(page.parent / target.partition("#")[0]).resolve() for page in PAGES for target in targets(page)}
    for folder in ("figures", "audio"):
        for path in sorted((ROOT / "docs" / folder).iterdir()):
            assert path.resolve() in linked, path.name


def test_every_sound_is_its_preset_as_it_ships() -> None:
    """Find in every preset's sound the patch it ships with, and a sound for every preset."""
    for name in cli.PRESETS:
        meta = wav.read(ROOT / "docs" / "audio" / f"{name}.wav").meta
        assert meta is not None
        assert loads(meta["patch"]) == loads(cli.preset_text(name)), name


def test_the_gallery_is_the_presets_as_they_are() -> None:
    """Write the gallery from the presets, and find docs/presets.md the same."""
    measured = [build.measure(name) for name in cli.PRESETS]
    assert build.gallery(measured) == (ROOT / "docs" / "presets.md").read_text(encoding="utf-8")


def test_the_guides_report_is_the_closed_hats() -> None:
    """Render the closed hat, and find its report in the guide, but for the time the render took."""
    guide = (ROOT / "docs" / "guide.md").read_text(encoding="utf-8")
    shown = re.search(r"## Reading the report\n\n```\n(.*?)```", guide, flags=re.DOTALL)
    assert shown is not None
    report = str(render(loads(cli.preset_text("closed")))[1])

    def timeless(text: str) -> list[str]:
        return [line for line in text.splitlines() if not line.startswith("render")]

    assert timeless(shown.group(1)) == timeless(report)


def test_the_readme_names_every_preset() -> None:
    """Find every preset's sound linked from the README."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in cli.PRESETS:
        assert f"(docs/audio/{name}.wav)" in text, name


def test_the_readme_on_pypi_links_into_the_repository_at_the_release() -> None:
    """Point the README's relative links at the repository at the tag, pages as pages, files as themselves."""
    text = "See [the guide](docs/guide.md), ![a figure](docs/figures/flow.svg) and [a sound](docs/audio/open.wav)."
    text += " Leave [the web](https://example.org/x.md) and [this page](#install) alone."
    out = readme.absolute(text, "https://github.com/someone/brightwork/", "v1.2.3")
    assert "[the guide](https://github.com/someone/brightwork/blob/v1.2.3/docs/guide.md)" in out
    assert "![a figure](https://github.com/someone/brightwork/raw/v1.2.3/docs/figures/flow.svg)" in out
    assert "[a sound](https://github.com/someone/brightwork/raw/v1.2.3/docs/audio/open.wav)" in out
    assert "[the web](https://example.org/x.md)" in out
    assert "[this page](#install)" in out
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "](docs/" not in readme.absolute(readme_text, "https://github.com/someone/brightwork", "v0")


def test_the_guides_python_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run every Python example in the guide, in an empty folder, and find the files they say they write."""
    guide = (ROOT / "docs" / "guide.md").read_text(encoding="utf-8")
    examples = re.findall(r"```python\n(.*?)```", guide, flags=re.DOTALL)
    assert examples
    monkeypatch.chdir(tmp_path)
    for i, code in enumerate(examples):
        script = tmp_path / f"example_{i}.py"
        script.write_text(code, encoding="utf-8")
        runpy.run_path(str(script))
    capsys.readouterr()
    assert sorted(p.name for p in tmp_path.glob("hat_rr*.wav")) == [f"hat_rr{j}.wav" for j in range(1, 5)]
    assert (tmp_path / "hats" / "hats.sfz").exists()


def test_the_readmes_quick_start_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run every command of the README's quick start but audition, in an empty folder, as a shell would."""
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"## Quick start\n\n```\n(.*?)```", readme_text, flags=re.DOTALL)
    assert block is not None
    monkeypatch.chdir(tmp_path)
    for line in block.group(1).splitlines():
        command = shlex.split(line.split("#")[0])
        assert command[0] == "brightwork"
        if command[1] == "audition":  # it plays until stopped
            continue
        args, out = (command[1:-2], command[-1]) if ">" in command else (command[1:], None)
        assert cli.main(args) == 0, line
        printed = capsys.readouterr().out
        if out is not None:
            Path(out).write_text(printed, encoding="utf-8")
    assert len(list((tmp_path / "hats").glob("*.wav"))) == (2 + 1) * 8
    assert (tmp_path / "hats" / "hats.sfz").exists()
