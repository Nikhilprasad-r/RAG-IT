import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import click
import typer
from rich.console import Console
from rich.table import Table

from ragkit.agents import (
    AgentManifest,
    AgentRuntime,
    ApprovalDecision,
    PendingAction,
    add_agent,
    discover_agents,
    list_templates,
)
from ragkit.agents.manifest import MANIFEST_FILENAME
from ragkit.agents.runtime import new_thread_id
from ragkit.agents.templates import is_builtin
from ragkit.core.config import CONFIG_FILENAME, DEFAULT_MODEL, ProjectConfig
from ragkit.core.errors import ConfigError, RagkitError
from ragkit.core.types import ScoredChunk
from ragkit.ingest import ingest as run_ingest
from ragkit.retrieval import CollectionStats, KnowledgeBase, create_store

app = typer.Typer(help="Build scalable RAG and agent applications.", no_args_is_help=True)
add_app = typer.Typer(help="Copy a template into the current project.", no_args_is_help=True)
app.add_typer(add_app, name="add")

console = Console()
err_console = Console(stderr=True)

PROJECT_CONFIG = """\
name: {name}
mode: local  # local | cluster
default_model: {model}
agents_dir: agents

# Needed for the postgres store and checkpointer.
# database_url: ${{DATABASE_URL}}

store:
  backend: sqlite          # sqlite (single file, no server) | postgres (pgvector)
  path: .ragkit/index.db

embeddings:
  model: google_genai:gemini-embedding-2
  dimensions: 768

# rerank:                  # optional; applied to every search unless an agent overrides it
#   provider: cohere       # cohere | voyage | cross-encoder
#   top_n: 8

collections:
  docs:
    description: Product documentation
    sources:
      - type: files
        path: data
        include: ["**/*.md", "**/*.pdf", "**/*.docx", "**/*.html", "**/*.txt"]
      # - type: web
      #   urls: [https://docs.example.com/]
      #   max_pages: 200
      # - type: github
      #   repo: owner/name
    chunking:
      max_tokens: 400
      overlap_tokens: 40

server:
  host: 127.0.0.1
  port: 8000
  cors_origins: []
"""

ENV_EXAMPLE = """\
# Copy to .env and fill in the keys for the model providers you use.
GOOGLE_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# COHERE_API_KEY=
# GITHUB_TOKEN=
# DATABASE_URL=postgresql://user:pass@localhost:5432/ragkit
"""

GITIGNORE = ".env\n.venv/\n__pycache__/\n.ragkit/\n"


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory to create the project in.")] = Path("."),
    name: Annotated[str | None, typer.Option(help="Project name (default: folder name).")] = None,
    model: Annotated[str, typer.Option(help="Default model as provider:model.")] = DEFAULT_MODEL,
) -> None:
    """Create a new ragkit project."""
    with _errors():
        path = path.resolve()
        if (path / CONFIG_FILENAME).exists():
            raise ConfigError(f"{path / CONFIG_FILENAME} already exists.")
        (path / "agents").mkdir(parents=True, exist_ok=True)
        (path / "data").mkdir(exist_ok=True)
        (path / CONFIG_FILENAME).write_text(
            PROJECT_CONFIG.format(name=name or path.name, model=model)
        )
        (path / ".env.example").write_text(ENV_EXAMPLE)
        if not (path / ".gitignore").exists():
            (path / ".gitignore").write_text(GITIGNORE)
        console.print(f"[green]Created project in {path}[/green]")
        console.print(
            "Next: copy .env.example to .env, put documents in data/, run `ragkit ingest`, "
            "then `ragkit add agent rag-chat`."
        )


@app.command()
def templates() -> None:
    """List the agent templates you can add."""
    table = Table("Template", "Pattern", "Description")
    for template in list_templates():
        table.add_row(template.name, template.pattern, template.description)
    console.print(table)


@add_app.command("agent")
def add_agent_command(
    template: Annotated[
        str,
        typer.Argument(
            help="Built-in name (see `ragkit templates`), a local folder, or "
            "gh:owner/repo/path[@ref]."
        ),
    ],
    name: Annotated[str | None, typer.Option(help="Name for the new agent.")] = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing agent folder.")] = False,
) -> None:
    """Copy an agent template into this project's agents folder."""
    with _errors():
        project = ProjectConfig.load()
        dest = add_agent(template, project.agents_path, name=name, force=force)
        console.print(f"[green]Added {dest.relative_to(project.root)}[/green]")
        code = sorted(str(p.relative_to(dest)) for p in dest.rglob("*.py"))
        if not is_builtin(template) and code:
            console.print(
                f"[yellow]Review {', '.join(code)} before running this agent: "
                "the code comes from outside ragkit and runs with your permissions.[/yellow]"
            )


@app.command()
def agents() -> None:
    """List the agents in this project."""
    with _errors():
        project = ProjectConfig.load()
        table = Table("Agent", "Pattern", "Model", "Folder")
        for path in discover_agents(project.agents_path):
            manifest = AgentManifest.from_file(path / MANIFEST_FILENAME)
            model = manifest.model or f"{project.default_model} (default)"
            table.add_row(
                manifest.name, manifest.pattern, model, str(path.relative_to(project.root))
            )
        console.print(table)


@app.command()
def chat(
    agent: Annotated[str, typer.Argument(help="Agent name.")],
    message: Annotated[str, typer.Argument(help="Message to send.")],
    thread: Annotated[str | None, typer.Option(help="Thread id to continue.")] = None,
) -> None:
    """Send one message to an agent and stream its reply."""
    with _errors():
        asyncio.run(_chat(ProjectConfig.load(), agent, message, thread))


async def _chat(project: ProjectConfig, name: str, message: str, thread_id: str | None) -> None:
    async with AgentRuntime(project, only=[name]) as runtime:
        agent = runtime.get(name)
        thread_id = thread_id or new_thread_id()
        if agent.manifest.output_schema:
            result = await agent.ainvoke(message, thread_id)
            console.print_json(data=result.structured) if result.structured else console.print(
                result.reply, markup=False
            )
        else:
            await _print_stream(agent.astream(message, thread_id))
        while pending := await agent.pending(thread_id):
            decisions = [_ask_decision(action) for action in pending]
            await _print_stream(await agent.astream_resume(thread_id, decisions))
        console.print(f"[dim]thread: {thread_id}[/dim]")


async def _print_stream(tokens: AsyncIterator[str]) -> None:
    async for token in tokens:
        console.print(token, end="", markup=False, highlight=False)
    console.print()


def _ask_decision(action: PendingAction) -> ApprovalDecision:
    console.print(f"[yellow]Approval needed:[/yellow] {action.tool}")
    console.print_json(data=action.args)
    keys = {d[0]: d for d in action.allowed_decisions}
    choice = typer.prompt(
        " / ".join(f"[{key}]{name[1:]}" for key, name in keys.items()),
        type=click.Choice(list(keys)),
        show_choices=False,
    )
    decision = keys[choice]
    if decision == "edit":
        raw = typer.prompt("New arguments as JSON", default=json.dumps(action.args))
        try:
            return ApprovalDecision(type="edit", args=json.loads(raw))
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Not valid JSON: {exc}") from exc
    if decision == "reject":
        reason = typer.prompt("Reason (optional)", default="", show_default=False)
        return ApprovalDecision(type="reject", message=reason or None)
    return ApprovalDecision(type="approve")


@app.command()
def ingest(
    collections: Annotated[
        list[str] | None, typer.Argument(help="Collections to ingest (default: all).")
    ] = None,
    full: Annotated[
        bool, typer.Option(help="Re-process every item, not just changed ones.")
    ] = False,
) -> None:
    """Fetch, parse, chunk, embed and index the configured sources."""
    with _errors():
        project = ProjectConfig.load()
        if not project.collections:
            raise ConfigError("No collections are defined in ragkit.yaml.")
        with console.status("Ingesting..."):
            reports = asyncio.run(run_ingest(project, collections, full=full))
    table = Table(
        "Collection", "Added", "Updated", "Unchanged", "Deleted", "Chunks", "Skipped", "Failed"
    )
    for r in reports:
        table.add_row(
            r.collection,
            *map(
                str,
                (
                    r.added,
                    r.updated,
                    r.unchanged,
                    r.deleted,
                    r.chunks,
                    len(r.skipped),
                    len(r.failed),
                ),
            ),
        )
    console.print(table)
    for r in reports:
        for uri, reason in r.failed:
            err_console.print(f"[red]failed[/red] {r.collection}: {uri}: {reason}")
        for uri, reason in r.skipped:
            console.print(f"[dim]skipped {r.collection}: {uri}: {reason}[/dim]")
    if any(not r.ok for r in reports):
        raise typer.Exit(1)


@app.command()
def search(
    collection: Annotated[str, typer.Argument(help="Collection to search.")],
    query: Annotated[str, typer.Argument(help="Search text.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
    mode: Annotated[str, typer.Option(help="dense, sparse or hybrid.")] = "hybrid",
    rerank: Annotated[bool, typer.Option(help="Apply the project's reranker.")] = True,
) -> None:
    """Search a collection and print the best passages."""
    if mode not in ("dense", "sparse", "hybrid"):
        raise typer.BadParameter("mode must be dense, sparse or hybrid")
    with _errors():
        project = ProjectConfig.load()
        results = asyncio.run(_search(project, collection, query, top_k, mode, rerank))
    if not results:
        console.print("No results.")
    for number, result in enumerate(results, start=1):
        path = " > ".join(result.chunk.metadata.get("heading_path") or [])
        console.print(
            f"[bold]{number}. {result.title or result.uri}[/bold]  "
            f"[dim]{result.retriever} {result.score:.3f}  {result.uri}[/dim]"
        )
        if path:
            console.print(f"   [cyan]{path}[/cyan]", highlight=False)
        snippet = " ".join(result.chunk.text.split())
        console.print(f"   {snippet[:300]}{'…' if len(snippet) > 300 else ''}", markup=False)


async def _search(
    project: ProjectConfig, collection: str, query: str, top_k: int, mode: Any, rerank: bool
) -> list[ScoredChunk]:
    async with KnowledgeBase(project) as knowledge:
        return await knowledge.search(collection, query, top_k=top_k, mode=mode, use_rerank=rerank)


@app.command()
def collections() -> None:
    """Show each collection's size and embedding model."""
    with _errors():
        project = ProjectConfig.load()
        stats = {s.collection: s for s in asyncio.run(_stats(project))}
    table = Table("Collection", "Sources", "Documents", "Chunks", "Embedding model")
    for name, config in project.collections.items():
        s = stats.get(name)
        model = f"{s.embedding_model} ({s.dimensions}d)" if s and s.embedding_model else "-"
        table.add_row(
            name,
            str(len(config.sources)),
            str(s.documents if s else 0),
            str(s.chunks if s else 0),
            model,
        )
    console.print(table)


async def _stats(project: ProjectConfig) -> list[CollectionStats]:
    async with create_store(project) as store:
        return await store.stats()


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Defaults to server.host.")] = None,
    port: Annotated[int | None, typer.Option(help="Defaults to server.port.")] = None,
    reload: Annotated[bool, typer.Option(help="Restart on code changes.")] = False,
) -> None:
    """Serve this project's agents over HTTP."""
    import uvicorn

    with _errors():
        project = ProjectConfig.load()
    uvicorn.run(
        "ragkit.server.app:create_app",
        factory=True,
        host=host or project.server.host,
        port=port or project.server.port,
        reload=reload,
    )


@app.command("version")
def show_version() -> None:
    """Show the installed ragkit version."""
    console.print(version("ragkit-cli"))


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except RagkitError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
