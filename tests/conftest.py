import json
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from ragkit.agents import AgentManifest, add_agent
from ragkit.core.config import ProjectConfig


class FakeChatModel(GenericFakeChatModel):
    """Fake model that accepts tools and records the messages it was sent."""

    seen: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def _stream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        # The base fake cannot stream tool calls; emit them as a single chunk.
        message = self._generate(messages, *args, **kwargs).generations[0].message
        if isinstance(message, AIMessage) and message.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(message.tool_calls)
            ]
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=message.content, tool_call_chunks=chunks)
            )
            return
        for piece in re.findall(r"\S+\s*|\s+", str(message.content)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def fake_models(*replies: AIMessage | str) -> Callable[[AgentManifest], FakeChatModel]:
    """Model factory whose model answers with `replies` in order."""
    messages = [AIMessage(r) if isinstance(r, str) else r for r in replies]
    return lambda _manifest: FakeChatModel(messages=iter(messages))


def fake_model_from(messages: Iterable[AIMessage]) -> Callable[[AgentManifest], FakeChatModel]:
    return lambda _manifest: FakeChatModel(messages=iter(messages))


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def write_project(root: Path, config: str = "name: test\n") -> ProjectConfig:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ragkit.yaml").write_text(config)
    return ProjectConfig.load(root)


@pytest.fixture
def project(tmp_path: Path) -> ProjectConfig:
    """A project containing a copy of the support-agent template."""
    config = write_project(tmp_path)
    add_agent("support-agent", config.agents_path)
    return config


RAG_CONFIG = """\
name: rag-test
store:
  backend: sqlite
embeddings:
  model: hash:256
collections:
  docs:
    description: Nimbus product docs
    sources:
      - type: files
        path: data
    chunking:
      max_tokens: 120
      overlap_tokens: 10
"""

SAMPLE_DOCS = {
    "plans.md": """# Nimbus plans

## Free plan
The Free plan includes 5 GB of storage and one workspace.

## Pro plan
The Pro plan costs $12 per user per month and includes 2 TB of storage.

## Team plan
The Team plan adds single sign-on (SSO) and audit logs.
""",
    "account/password.md": """# Resetting your password

Open Settings, choose Security and click "Reset password".
A reset link is emailed to you and expires after 30 minutes.
""",
    "api.md": """# API limits

The REST API allows 600 requests per minute per token.
Requests over the limit receive HTTP 429 with a Retry-After header.
""",
}


def write_docs(root: Path, docs: dict[str, str] = SAMPLE_DOCS) -> None:
    for name, text in docs.items():
        path = root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def rag_project(tmp_path: Path) -> ProjectConfig:
    """A project with a `docs` collection of three Markdown files and an offline embedder."""
    write_docs(tmp_path)
    return write_project(tmp_path, RAG_CONFIG)
