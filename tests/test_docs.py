"""Keep the documentation honest.

- The generated reference pages must match the code.
- YAML examples headed `# ragkit.yaml` or `# agent.yaml` must validate against the
  real models (partial examples get a `name` filled in).
- Python examples must compile, and JSON examples must parse (as one document or
  one object per line).
"""

import importlib.util
import json
import re
import textwrap
from pathlib import Path

import pytest
import yaml

from ragkit.agents.manifest import AgentManifest
from ragkit.core.config import ProjectConfig, _expand_env

DOCS = Path(__file__).parent.parent / "docs"
FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>\w+)[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.M | re.S
)


def _blocks(lang: str) -> list[tuple[str, str]]:
    found = []
    for page in sorted(DOCS.rglob("*.md")):
        for match in FENCE.finditer(page.read_text()):
            if match["lang"] == lang:
                body = textwrap.dedent(
                    "\n".join(
                        line[len(match["indent"]) :] if line.startswith(match["indent"]) else line
                        for line in match["body"].splitlines()
                    )
                )
                found.append((f"{page.relative_to(DOCS)}:{match.start()}", body))
    return found


def test_reference_pages_are_current() -> None:
    spec = importlib.util.spec_from_file_location("gen_reference", DOCS / "gen_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, text in module.generate().items():
        current = (DOCS / "reference" / name).read_text()
        assert current == text, f"docs/reference/{name} is stale: run python docs/gen_reference.py"


YAML_BLOCKS = [(where, body) for where, body in _blocks("yaml") if body.startswith("# ")]


@pytest.mark.parametrize(("where", "body"), YAML_BLOCKS, ids=[w for w, _ in YAML_BLOCKS])
def test_yaml_examples_validate(where: str, body: str, monkeypatch: pytest.MonkeyPatch) -> None:
    kind = body.splitlines()[0].removeprefix("# ").strip()
    if kind not in ("ragkit.yaml", "agent.yaml"):
        pytest.skip(f"not a checked example ({kind})")
    for variable in re.findall(r"\$\{(\w+)\}", body):
        monkeypatch.setenv(variable, "postgresql://example")
    data = _expand_env(yaml.safe_load(body)) or {}
    data.setdefault("name", "example")
    if kind == "ragkit.yaml":
        ProjectConfig.model_validate({**data, "root": Path(".")})
    else:
        AgentManifest.model_validate(data)


def test_there_are_checked_yaml_examples() -> None:
    kinds = {body.splitlines()[0] for _, body in YAML_BLOCKS}
    assert {"# ragkit.yaml", "# agent.yaml"} <= kinds
    assert len(YAML_BLOCKS) >= 30


@pytest.mark.parametrize(
    ("where", "body"), _blocks("python"), ids=[w for w, _ in _blocks("python")]
)
def test_python_examples_compile(where: str, body: str) -> None:
    # Snippets may use top-level await, as in an async context or notebook.
    wrapped = "async def _snippet():\n" + textwrap.indent(body, "    ") + "\n    pass\n"
    compile(wrapped if "await " in body else body, where, "exec")


@pytest.mark.parametrize(("where", "body"), _blocks("json"), ids=[w for w, _ in _blocks("json")])
def test_json_examples_parse(where: str, body: str) -> None:
    try:
        json.loads(body)
    except json.JSONDecodeError:
        for line in body.splitlines():
            if line.strip():
                json.loads(line)
