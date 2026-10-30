from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragkit.cli.main import app

runner = CliRunner()


def test_init_add_and_list_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    init = runner.invoke(app, ["init", "--name", "demo"])
    add = runner.invoke(app, ["add", "agent", "support-agent", "--name", "helpdesk"])
    listing = runner.invoke(app, ["agents"])

    assert init.exit_code == 0, init.output
    assert add.exit_code == 0, add.output
    assert listing.exit_code == 0, listing.output
    assert "helpdesk" in listing.output
    assert "name: demo" in (tmp_path / "ragkit.yaml").read_text()
    assert (tmp_path / "agents" / "helpdesk" / "agent.yaml").is_file()


def test_init_refuses_existing_project(tmp_path: Path) -> None:
    assert runner.invoke(app, ["init", str(tmp_path)]).exit_code == 0

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_commands_outside_a_project_fail_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["agents"])

    assert result.exit_code == 1
    assert "ragkit init" in result.output


def test_templates_lists_support_agent() -> None:
    result = runner.invoke(app, ["templates"])

    assert result.exit_code == 0
    assert "support-agent" in result.output
