import io
import tarfile
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from conftest import write_project
from ragkit.agents import AgentManifest, TemplateNotFoundError, add_agent
from ragkit.agents.templates import GitHubTemplate, parse_github
from ragkit.cli.main import app as cli


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("gh:acme/agents", GitHubTemplate("acme", "agents", "", None)),
        ("gh:acme/agents/triage@v2", GitHubTemplate("acme", "agents", "triage", "v2")),
        (
            "https://github.com/acme/agents/tree/main/teams/triage",
            GitHubTemplate("acme", "agents", "teams/triage", "main"),
        ),
        ("https://github.com/acme/agents.git", GitHubTemplate("acme", "agents", "", None)),
        ("rag-chat", None),
    ],
)
def test_parse_github(source: str, expected: GitHubTemplate | None) -> None:
    assert parse_github(source) == expected


def _tarball(files: dict[str, str], top: str = "acme-agents-1a2b3c") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


TRIAGE = {
    "teams/triage/agent.yaml": "name: triage\ndescription: Sorts tickets.\n",
    "teams/triage/prompts/system.md": "Sort tickets.",
    "teams/triage/tools.py": "def label(x: str) -> str:\n    return x\n",
    "teams/other/agent.yaml": "name: other\n",
    "README.md": "repo readme",
}


def _github(tarball: bytes, seen: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/repos/acme/agents/tarball/v2":
            return httpx.Response(200, content=tarball)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_install_from_github_copies_only_the_template(tmp_path: Path) -> None:
    seen: list[str] = []

    dest = add_agent(
        "gh:acme/agents/teams/triage@v2", tmp_path, client=_github(_tarball(TRIAGE), seen)
    )

    assert seen == ["https://api.github.com/repos/acme/agents/tarball/v2"]
    assert dest == tmp_path / "triage"
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
        "agent.yaml",
        "prompts/system.md",
        "tools.py",
    ]
    assert not (tmp_path / "other").exists()


def test_install_from_github_errors(tmp_path: Path) -> None:
    client = _github(_tarball(TRIAGE), [])
    with pytest.raises(TemplateNotFoundError, match="has no agent.yaml"):
        add_agent("gh:acme/agents/teams@v2", tmp_path, client=client)
    with pytest.raises(TemplateNotFoundError, match="GitHub has no acme/agents@v9"):
        add_agent("gh:acme/agents/teams/triage@v9", tmp_path, client=client)


def test_archive_paths_cannot_escape_the_target(tmp_path: Path) -> None:
    evil = {**TRIAGE, "teams/triage/../../../../escaped.txt": "nope"}
    client = _github(_tarball(evil), [])

    with pytest.raises(tarfile.OutsideDestinationError):
        add_agent("gh:acme/agents/teams/triage@v2", tmp_path / "agents", client=client)
    assert not list(tmp_path.rglob("escaped.txt"))


def test_install_from_local_folder_uses_its_manifest_name(tmp_path: Path) -> None:
    source = tmp_path / "shared" / "triage-v2"
    (source / "prompts").mkdir(parents=True)
    (source / "agent.yaml").write_text("name: triage\n")
    (source / "prompts" / "system.md").write_text("Sort.")

    dest = add_agent(str(source), tmp_path / "agents")
    renamed = add_agent(str(source), tmp_path / "agents", name="triage-eu")

    assert dest.name == "triage"
    assert AgentManifest.from_file(renamed / "agent.yaml").name == "triage-eu"


def test_cli_warns_about_third_party_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = write_project(tmp_path / "app")
    source = tmp_path / "template"
    (source / "prompts").mkdir(parents=True)
    (source / "agent.yaml").write_text("name: helper\ntools:\n  local: [ping]\n")
    (source / "prompts" / "system.md").write_text("Help.")
    (source / "tools.py").write_text("def ping() -> str:\n    return 'pong'\n")
    monkeypatch.chdir(project.root)

    external = CliRunner().invoke(cli, ["add", "agent", str(source)])
    builtin = CliRunner().invoke(cli, ["add", "agent", "support-agent"])

    assert external.exit_code == 0, external.output
    assert "Review tools.py" in external.output
    assert "Review" not in builtin.output
