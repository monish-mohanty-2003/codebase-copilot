"""Shared pytest fixtures."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def tmp_repo() -> Iterator[Path]:
    """Create a tiny temp Python repo for indexing tests."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "mod.py").write_text(
            "def add(a, b):\n"
            "    \"\"\"Add two numbers.\"\"\"\n"
            "    return a + b\n"
            "\n"
            "class Greeter:\n"
            "    def hello(self, name):\n"
            "        return f'hi {name}'\n"
        )
        (root / "unused.py").write_text(
            "def never_called():\n"
            "    return 42\n"
        )
        yield root


@pytest.fixture
def test_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointed at a tmp data dir."""
    monkeypatch.setenv("COPILOT_DATA_DIR", str(tmp_path / ".copilot"))
    monkeypatch.setenv("COPILOT_CHROMA_DIR", str(tmp_path / ".copilot" / "chroma"))
    monkeypatch.setenv("COPILOT_CACHE_DIR", str(tmp_path / ".copilot" / "cache"))
    # Reset singleton
    import app.config
    app.config._settings = None
    s = Settings()
    s.ensure_dirs()
    return s
 