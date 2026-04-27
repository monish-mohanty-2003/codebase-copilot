"""run_tests: invoke the project's test runner.

For Python repos we invoke pytest. For multi-language repos this can be
extended (jest, go test, etc.). We hard-cap runtime and capture output.

SECURITY NOTE: This runs subprocesses against user-supplied paths. We
constrain to the repo root and never execute arbitrary user input.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.tools.base import Tool


class RunTestsTool(Tool):
    name = "run_tests"
    description = (
        "Run the project's test suite (pytest by default) and return a summary. "
        "Optionally filter by test path/expression. Capped at 60 seconds."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "test_filter": {
                "type": "string",
                "description": "Optional pytest -k expression or file path.",
            },
            "framework": {
                "type": "string",
                "enum": ["pytest", "unittest", "auto"],
                "description": "Test framework. 'auto' detects from repo.",
                "default": "auto",
            },
        },
    }

    TIMEOUT_SECONDS = 60

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()

    async def run(self, test_filter: str | None = None, framework: str = "auto") -> dict[str, Any]:
        framework = self._detect_framework() if framework == "auto" else framework
        if framework == "pytest":
            cmd = ["pytest", "--tb=short", "-q"]
            if test_filter:
                cmd += ["-k", test_filter] if not test_filter.endswith(".py") else [test_filter]
        elif framework == "unittest":
            cmd = ["python", "-m", "unittest", "discover", "-v"]
        else:
            return {"framework": framework, "error": "Unsupported framework"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"framework": framework, "error": "Timed out"}
        except FileNotFoundError as e:
            return {"framework": framework, "error": f"Runner not found: {e}"}

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        return {
            "framework": framework,
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            # Truncate to keep token usage bounded.
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

    def _detect_framework(self) -> str:
        if (self.repo_root / "pytest.ini").exists() or (self.repo_root / "pyproject.toml").exists():
            return "pytest"
        if any(self.repo_root.rglob("test_*.py")):
            return "pytest"
        return "unittest"