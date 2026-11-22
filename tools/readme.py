# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The README as PyPI shows it: its relative links made absolute, into the repository at a release's tag.

PyPI renders the README alone, so a link to docs/guide.md or an image of docs/figures/presets.svg would
lead nowhere. Pages link to the repository's page for them, and images and sounds to the file itself.
"""

from __future__ import annotations

import re

LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
FILES = (".svg", ".png", ".wav")  # served as themselves, not as a page about them


def absolute(text: str, repository: str, ref: str) -> str:
    """Return the text with every relative link pointing into `repository` at `ref`; others stay as they are."""
    base = repository.rstrip("/")

    def rewrite(m: re.Match[str]) -> str:
        bang, label, target = m.groups()
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        kind = "raw" if bang or target.lower().endswith(FILES) else "blob"
        return f"{bang}[{label}]({base}/{kind}/{ref}/{target})"

    return LINK.sub(rewrite, text)
