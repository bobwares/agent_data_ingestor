# Data Ingestor
# Package: data_ingestor
# File: main.py
# Version: 1.1.0
# Author: Bobwares
# Date: 2025-06-03 16:10
# Description: CLI – now has 'ingest' and 'send' commands.
#

from __future__ import annotations

from pathlib import Path

import typer

from data_ingestor.connectors import ConnectorFactory
from data_ingestor.llm_bridge import inline_prompt, retrieval_qa, upload_and_prompt
from data_ingestor.writers import WriterFactory

app = typer.Typer(add_completion=False)

# ──────────────────────────────────────────────────────────────────────────────
#  Ingest (existing)
# ──────────────────────────────────────────────────────────────────────────────
@app.command()
def ingest(
        source: str = typer.Argument(...),
        target: str = typer.Option("blob", "--target", "-t"),
        raw: bool = typer.Option(False, "--raw"),
) -> None:
    blob = ConnectorFactory.from_uri(source).read_bytes()
    writer = WriterFactory.from_uri(target)

    if hasattr(writer, "write_bytes"):
        writer.write_bytes(blob, raw=raw)  # type: ignore[attr-defined]
    else:
        writer.write(blob)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
#  NEW: Send file to LLM
# ──────────────────────────────────────────────────────────────────────────────
@app.command()
def send(  # noqa: D401
        file_path: Path = typer.Argument(..., exists=True, readable=True),
        mode: str = typer.Option(
            "inline",
            "--mode",
            "-m",
            help="inline | retrieval | upload",
        ),
        task: str = typer.Option(
            None,
            "--task",
            "-t",
            help="Instruction for inline/upload modes.",
        ),
        query: str = typer.Option(
            None,
            "--query",
            "-q",
            help="Question for retrieval mode.",
        ),
) -> None:
    if mode == "inline":
        if not task:
            typer.echo("`--task` is required for inline mode", err=True)
            raise typer.Exit(1)
        result = inline_prompt(file_path, task)
    elif mode == "retrieval":
        if not query:
            typer.echo("`--query` is required for retrieval mode", err=True)
            raise typer.Exit(1)
        result = retrieval_qa(file_path, query)
    elif mode == "upload":
        if not task:
            typer.echo("`--task` is required for upload mode", err=True)
            raise typer.Exit(1)
        result = upload_and_prompt(file_path, task)
    else:
        typer.echo(f"Unknown mode: {mode}", err=True)
        raise typer.Exit(1)

    typer.echo(result)


def cli() -> None:  # noqa: D401
    app()


if __name__ == "__main__":
    cli()
