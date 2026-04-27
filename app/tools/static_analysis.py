"""static_analysis: surface lint, complexity, and type errors via local tools."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.tools.base import Tool


class StaticAnalysisTool(Tool):
    name = "static_analysis"
    description = (
        "Run static analysis on a file or the whole repo. Returns lint findings "
        "(ruff for Python). Use to check code health before suggesting changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory to analyze (default: whole repo).",
            },
        },
    }

    TIMEOUT_SECONDS = 30

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()

    async def run(self, path: str | None = None) -> dict[str, Any]:
        target = self.repo_root if not path else (self.repo_root / path).resolve()
        if not str(target).startswith(str(self.repo_root)):
            raise ValueError("path escapes repo root")

        cmd = ["ruff", "check", "--output-format=json", str(target)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, asyncio.TimeoutError) as e:
            return {"error": f"ruff unavailable or timed out: {e}", "findings": []}

        stdout = stdout_b.decode(errors="replace").strip()
        try:
            findings_raw = json.loads(stdout) if stdout else []
        except json.JSONDecodeError:
            return {"error": "ruff produced invalid JSON", "raw": stdout[-1000:]}

        findings = [
            {
                "file": str(Path(f["filename"]).relative_to(self.repo_root)),
                "line": f["location"]["row"],
                "code": f["code"],
                "message": f["message"],
            }
            for f in findings_raw[:200]  # cap output size
        ]
        return {
            "tool": "ruff",
            "target": str(target.relative_to(self.repo_root)) if target != self.repo_root else ".",
            "total": len(findings_raw),
            "findings": findings,
        }