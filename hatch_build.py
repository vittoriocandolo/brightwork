# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build hook: the README that PyPI shows, its links made absolute by tools/readme.py, at the version's tag."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, override

from hatchling.metadata.plugin.interface import MetadataHookInterface


class ReadmeHook(MetadataHookInterface):
    """Fill the dynamic readme: README.md, its links pointing into the repository at tag v<version>."""

    @override
    def update(self, metadata: dict[str, Any]) -> None:
        """Set the readme from README.md and [project.urls] Repository."""
        root = Path(self.root)
        spec = importlib.util.spec_from_file_location("readme", root / "tools" / "readme.py")
        if spec is None or spec.loader is None:
            msg = "tools/readme.py is missing"
            raise RuntimeError(msg)
        readme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(readme)
        text = (root / "README.md").read_text(encoding="utf-8")
        repository, ref = metadata["urls"]["Repository"], f"v{metadata['version']}"
        metadata["readme"] = {"content-type": "text/markdown", "text": readme.absolute(text, repository, ref)}
