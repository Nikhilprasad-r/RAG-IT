"""Eval datasets: one JSON object per line.

Agent cases:
    {"input": "...", "reference": "expected answer" | {"field": value},
     "expected_tools": ["lookup_order"], "expected_sources": ["password.md"]}

Search cases:
    {"query": "...", "expected_sources": ["password.md"]}

`expected_sources` match when the text appears in a retrieved passage's URI.
Every field except the input or query is optional; metrics are computed only
for the fields a case provides.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragkit.core.errors import ConfigError


class AgentCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    input: str = Field(min_length=1)
    reference: str | dict[str, Any] | None = None
    expected_tools: list[str] | None = None
    expected_sources: list[str] | None = None


class SearchCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    query: str = Field(min_length=1)
    expected_sources: list[str] = Field(min_length=1)


def load_cases[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        raise ConfigError(f"Eval dataset not found: {path}")
    cases = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = model.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigError(f"{path}:{number}: invalid eval case: {exc}") from exc
        cases.append(
            case if getattr(case, "id", None) else case.model_copy(update={"id": str(number)})
        )
    if not cases:
        raise ConfigError(f"Eval dataset {path} has no cases.")
    return cases
