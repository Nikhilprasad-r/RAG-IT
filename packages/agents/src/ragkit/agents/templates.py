"""Copy-in agent templates.

A template is any folder with an `agent.yaml`. `add_agent` copies one into a
project, where it becomes ordinary code the project owns and edits. Templates
can come from:

- the built-in set shipped in this package (`rag-chat`, `support-agent`, ...),
- a local folder (`./shared/agents/triage`),
- GitHub: `gh:owner/repo/path/to/agent@ref` or a github.com/.../tree/<ref>/<path> URL.

Templates from outside this package can contain Python (tools.py, schemas) that
runs with the project's permissions; the CLI asks people to review them.
"""

import io
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from ragkit.agents.manifest import MANIFEST_FILENAME, NAME_PATTERN, AgentManifest
from ragkit.core.errors import ConfigError, RagkitError

TEMPLATES_DIR = Path(__file__).parent / "templates"

_NAME_LINE = re.compile(r"^name:.*$", re.MULTILINE)
_GH_SHORT = re.compile(
    r"^gh:(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)(?P<path>/[^@]*)?(?:@(?P<ref>.+))?$"
)
_GH_URL = re.compile(
    r"^https://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[^/]+)(?P<path>/.*)?)?/?$"
)


class TemplateNotFoundError(RagkitError):
    pass


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    pattern: str
    path: Path


@dataclass(frozen=True)
class GitHubTemplate:
    owner: str
    repo: str
    path: str
    ref: str | None


def list_templates() -> list[Template]:
    templates = []
    for path in sorted(TEMPLATES_DIR.iterdir()):
        if (path / MANIFEST_FILENAME).is_file():
            manifest = AgentManifest.from_file(path / MANIFEST_FILENAME)
            templates.append(Template(path.name, manifest.description, manifest.pattern, path))
    return templates


def is_builtin(template: str) -> bool:
    return "/" not in template and (TEMPLATES_DIR / template / MANIFEST_FILENAME).is_file()


def parse_github(template: str) -> GitHubTemplate | None:
    match = _GH_SHORT.match(template) or _GH_URL.match(template)
    if not match:
        return None
    path = (match.group("path") or "").strip("/")
    return GitHubTemplate(match["owner"], match["repo"], path, match.group("ref"))


def add_agent(
    template: str,
    agents_dir: Path,
    *,
    name: str | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
) -> Path:
    """Copy `template` into `agents_dir`, optionally renaming it. Returns the new folder."""
    if is_builtin(template):
        return _install(TEMPLATES_DIR / template, agents_dir, name or template, force)
    if github := parse_github(template):
        with tempfile.TemporaryDirectory(prefix="ragkit-template-") as tmp:
            source = download_github_template(github, Path(tmp), client=client)
            return _install(source, agents_dir, name, force)
    local = Path(template).expanduser()
    if (local / MANIFEST_FILENAME).is_file():
        return _install(local.resolve(), agents_dir, name, force)
    available = ", ".join(t.name for t in list_templates())
    raise TemplateNotFoundError(
        f"No template {template!r}. Use a built-in name ({available}), a folder containing "
        f"{MANIFEST_FILENAME}, or gh:owner/repo/path[@ref]."
    )


def download_github_template(
    template: GitHubTemplate, workdir: Path, *, client: httpx.Client | None = None
) -> Path:
    """Download the repository tarball and extract only the template's folder."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ragkit"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{template.owner}/{template.repo}/tarball"
    if template.ref:
        url += f"/{template.ref}"
    http = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        response = http.get(url, headers=headers)
    finally:
        if client is None:
            http.close()
    if response.status_code == 404:
        raise TemplateNotFoundError(
            f"GitHub has no {template.owner}/{template.repo}"
            f"{'@' + template.ref if template.ref else ''} (or it is private: set GITHUB_TOKEN)."
        )
    response.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise TemplateNotFoundError("The downloaded repository archive is empty.")
        top = PurePosixPath(members[0].name).parts[0]
        prefix = PurePosixPath(top, template.path) if template.path else PurePosixPath(top)
        wanted = [
            m
            for m in members
            if PurePosixPath(m.name) == prefix or PurePosixPath(m.name).is_relative_to(prefix)
        ]
        # The "data" filter refuses absolute paths, links leaving the folder and devices.
        archive.extractall(workdir, members=wanted, filter="data")
    source = workdir / prefix
    if not (source / MANIFEST_FILENAME).is_file():
        raise TemplateNotFoundError(
            f"{template.owner}/{template.repo}/{template.path or ''} has no {MANIFEST_FILENAME}."
        )
    return source


def _install(source: Path, agents_dir: Path, name: str | None, force: bool) -> Path:
    manifest = AgentManifest.from_file(source / MANIFEST_FILENAME)
    agent_name = name or manifest.name
    if not re.fullmatch(NAME_PATTERN, agent_name):
        raise ConfigError(
            f"Invalid agent name {agent_name!r}: use lowercase letters, digits and hyphens."
        )

    dest = agents_dir / agent_name
    if dest.exists():
        if not force:
            raise ConfigError(f"{dest} already exists. Pass force=True to overwrite it.")
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))

    manifest_path = dest / MANIFEST_FILENAME
    manifest_path.write_text(
        _NAME_LINE.sub(f"name: {agent_name}", manifest_path.read_text(), count=1)
    )
    return dest
