from pathlib import Path

import pytest

from ragkit.agents import AgentManifest, TemplateNotFoundError, add_agent, list_templates
from ragkit.core.errors import ConfigError


def test_every_shipped_template_has_a_valid_manifest() -> None:
    templates = list_templates()

    assert "support-agent" in {t.name for t in templates}
    for template in templates:
        manifest = AgentManifest.from_file(template.path / "agent.yaml")
        assert manifest.name == template.name
        assert (template.path / manifest.system_prompt).is_file()


def test_add_agent_copies_and_renames(tmp_path: Path) -> None:
    dest = add_agent("support-agent", tmp_path, name="billing-help")

    assert dest == tmp_path / "billing-help"
    assert AgentManifest.from_file(dest / "agent.yaml").name == "billing-help"
    assert (dest / "tools.py").is_file()
    assert "# Support agent" in (dest / "agent.yaml").read_text(), "comments are kept"


def test_add_agent_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    dest = add_agent("support-agent", tmp_path)
    (dest / "tools.py").write_text("# edited\n")

    with pytest.raises(ConfigError, match="already exists"):
        add_agent("support-agent", tmp_path)
    assert (dest / "tools.py").read_text() == "# edited\n"

    add_agent("support-agent", tmp_path, force=True)
    assert "lookup_order" in (dest / "tools.py").read_text()


def test_add_agent_rejects_bad_names_and_templates(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Invalid agent name"):
        add_agent("support-agent", tmp_path, name="Bad Name")
    with pytest.raises(TemplateNotFoundError, match="support-agent"):
        add_agent("no-such-template", tmp_path)
    assert not any(tmp_path.iterdir())
