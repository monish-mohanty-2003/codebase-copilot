"""open_file: read a slice of a source file from the repo workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import Tool


class OpenFileTool(Tool):
    name = "open_file"
    description = (
        "Read a slice of a source file from the indexed repository. "
        "Use after search_code to inspect full context around a result. "
        "Default returns first 200 lines; pass start_line/end_line for a slice."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository-relative path, e.g. 'src/auth/middleware.py'.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-indexed inclusive start (default 1).",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "1-indexed inclusive end (default start+200).",
            },
        },
        "required": ["path"],
    }

    MAX_LINES = 400

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()

    def run(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
        # Path traversal guard — must remain inside repo_root.
        target = (self.repo_root / path).resolve()
        if not str(target).startswith(str(self.repo_root)):
            raise ValueError("path escapes repo root")
        if not target.is_file():
            raise FileNotFoundError(path)

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start_line))
        end = int(end_line) if end_line else min(start + 200 - 1, total)
        end = min(end, total, start + self.MAX_LINES - 1)
        slice_lines = lines[start - 1:end]
        return {
            "path": path,
            "start_line": start,
            "end_line": end,
            "total_lines": total,
            "content": "\n".join(slice_lines),
        }