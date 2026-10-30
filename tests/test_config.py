from pathlib import Path

import pytest

from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError, ProjectNotFoundError


def test_load_finds_project_from_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "ragkit.yaml").write_text("name: demo\nserver:\n  port: 9000\n")
    nested = tmp_path / "agents" / "deep"
    nested.mkdir(parents=True)

    config = ProjectConfig.load(nested)

    assert config.name == "demo"
    assert config.root == tmp_path.resolve()
    assert config.server.port == 9000
    assert config.agents_path == tmp_path.resolve() / "agents"


def test_env_references_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGKIT_TEST_DB", "postgresql://db")
    (tmp_path / "ragkit.yaml").write_text(
        "name: demo\n"
        "database_url: ${RAGKIT_TEST_DB}\n"
        "default_model: ${RAGKIT_TEST_UNSET:-openai:gpt-5.5}\n"
        "# commented: ${NOT_SET_ANYWHERE}\n"
    )

    config = ProjectConfig.load(tmp_path)

    assert config.database_url == "postgresql://db"
    assert config.default_model == "openai:gpt-5.5"


def test_dotenv_in_project_root_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGKIT_TEST_FROM_DOTENV", raising=False)
    (tmp_path / ".env").write_text("RAGKIT_TEST_FROM_DOTENV=sqlite://x\n")
    (tmp_path / "ragkit.yaml").write_text("name: demo\ndatabase_url: ${RAGKIT_TEST_FROM_DOTENV}\n")

    assert ProjectConfig.load(tmp_path).database_url == "sqlite://x"
    monkeypatch.delenv("RAGKIT_TEST_FROM_DOTENV")


def test_missing_env_reference_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "ragkit.yaml").write_text("name: demo\ndatabase_url: ${RAGKIT_TEST_MISSING}\n")

    with pytest.raises(ConfigError, match="RAGKIT_TEST_MISSING"):
        ProjectConfig.load(tmp_path)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "ragkit.yaml").write_text("name: demo\nagent_dir: typo\n")

    with pytest.raises(ConfigError, match="agent_dir"):
        ProjectConfig.load(tmp_path)


def test_missing_project_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ProjectNotFoundError):
        ProjectConfig.load(tmp_path)
