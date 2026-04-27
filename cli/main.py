"""Codebase Copilot CLI.

Examples:

    copilot index ~/code/myrepo
    copilot search "where is authentication handled?"
    copilot chat "explain the indexing pipeline"
    copilot serve
    copilot eval ~/code/myrepo cases.json --out report.json
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.config import get_settings

app = typer.Typer(help="Local AI copilot for your repository.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to repository to index"),
    name: str | None = typer.Option(None, help="Repo name in the index"),
    full: bool = typer.Option(False, "--full", help="Force full reindex"),
) -> None:
    """Index a repository into the vector store."""
    from app.api.state import AppState

    async def _run() -> None:
        state = AppState.build(get_settings())
        repo_name = name or repo_path.name
        stats = await state.pipeline.index_repo(repo_path.resolve(), repo_name=repo_name, full=full)
        console.print(f"[green]✓[/green] Indexed [bold]{repo_name}[/bold]")
        table = Table(show_header=False)
        table.add_row("Files scanned", str(stats.files_scanned))
        table.add_row("Files indexed", str(stats.files_indexed))
        table.add_row("Files skipped (unchanged)", str(stats.files_skipped_unchanged))
        table.add_row("Chunks embedded", str(stats.chunks_embedded))
        if stats.errors:
            table.add_row("Errors", f"[red]{len(stats.errors)}[/red]")
        console.print(table)

    asyncio.run(_run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Query string"),
    repo: str | None = typer.Option(None, help="Repo name (defaults to last indexed)"),
    k: int = typer.Option(8, help="Number of results"),
    language: str | None = typer.Option(None, help="Filter by language"),
) -> None:
    """Run a one-off semantic search."""
    from app.api.state import AppState

    async def _run() -> None:
        state = AppState.build(get_settings())
        if repo is None and state.store.count() == 0:
            console.print("[red]No indexed repos. Run `copilot index` first.[/red]")
            raise typer.Exit(1)
        hits = await state.retriever.search(query=query, k=k, repo=repo, language=language)
        if not hits:
            console.print("[yellow]No results.[/yellow]")
            return
        table = Table(title=f"Top {len(hits)} hits for: {query!r}")
        table.add_column("Score", justify="right")
        table.add_column("Path")
        table.add_column("Symbol")
        table.add_column("Lines")
        for h in hits:
            table.add_row(
                f"{h.score:.3f}",
                h.chunk.path,
                h.chunk.symbol,
                f"{h.chunk.start_line}-{h.chunk.end_line}",
            )
        console.print(table)

    asyncio.run(_run())


@app.command()
def chat(
    query: str = typer.Argument(..., help="Question for the agent"),
    repo_path: Path = typer.Option(Path("."), "--repo", help="Repo root for tools"),
    name: str | None = typer.Option(None, help="Repo name in index"),
    show_steps: bool = typer.Option(False, "--steps", help="Show reasoning trace"),
) -> None:
    """Ask the agent a question."""
    from app.api.state import AppState

    async def _run() -> None:
        state = AppState.build(get_settings())
        repo_root = repo_path.resolve()
        repo_name = name or repo_root.name
        state.attach_repo(repo_root, repo_name)
        assert state.agent is not None
        trace = await state.agent.run(query)

        if show_steps:
            for step in trace.steps:
                if step.tool_call:
                    console.print(
                        f"[blue]→ tool[/blue] [bold]{step.tool_call.name}[/bold]"
                        f"({json.dumps(step.tool_call.arguments)[:120]})"
                    )
        console.rule("[bold green]Answer[/bold green]")
        console.print(trace.answer)
        console.print(f"[dim]({trace.elapsed_seconds}s, {len(trace.steps)} steps, {trace.model})[/dim]")

    asyncio.run(_run())


@app.command()
def serve() -> None:
    """Start the FastAPI server."""
    from app.api.server import main as serve_main
    serve_main()


@app.command()
def eval_cmd(
    repo_path: Path = typer.Argument(..., help="Repo to evaluate against"),
    cases: Path = typer.Argument(..., help="Path to eval cases JSON"),
    out: Path = typer.Option(Path("eval_report.json"), help="Where to write the report"),
) -> None:
    """Run the model/prompt/hyperparameter eval grid."""
    from app.eval.harness import DEFAULT_CONFIGS, run_eval
    asyncio.run(run_eval(repo_path.resolve(), cases.resolve(), DEFAULT_CONFIGS, out.resolve()))


# Typer doesn't allow naming a command `eval` directly because it shadows builtin
app.command(name="eval")(eval_cmd)


if __name__ == "__main__":
    app()