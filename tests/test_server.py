import json

from fastapi.testclient import TestClient

from conftest import fake_models
from ragkit.core.config import ProjectConfig
from ragkit.server import create_app


def _client(project: ProjectConfig, *replies: str) -> TestClient:
    return TestClient(create_app(project, model_factory=fake_models(*replies)))


def test_health_and_agent_listing(project: ProjectConfig) -> None:
    with _client(project) as client:
        assert client.get("/health").json() == {"status": "ok"}
        agents = client.get("/agents").json()

    assert [a["name"] for a in agents] == ["support-agent"]
    assert agents[0]["pattern"] == "react"


def test_chat_returns_reply_and_reuses_thread(project: ProjectConfig) -> None:
    with _client(project, "Hello there.", "Welcome back.") as client:
        first = client.post("/agents/support-agent/chat", json={"message": "hi"}).json()
        second = client.post(
            "/agents/support-agent/chat",
            json={"message": "again", "thread_id": first["thread_id"]},
        ).json()

    assert first["reply"] == "Hello there."
    assert second == {
        "reply": "Welcome back.",
        "thread_id": first["thread_id"],
        "status": "completed",
        "pending": [],
        "structured": None,
    }


def test_unknown_agent_returns_404(project: ProjectConfig) -> None:
    with _client(project) as client:
        response = client.post("/agents/nope/chat", json={"message": "hi"})

    assert response.status_code == 404
    assert "support-agent" in response.json()["detail"]


def test_empty_message_is_rejected(project: ProjectConfig) -> None:
    with _client(project) as client:
        response = client.post("/agents/support-agent/chat", json={"message": ""})

    assert response.status_code == 422


def test_stream_sends_tokens_as_server_sent_events(project: ProjectConfig) -> None:
    with _client(project, "Your order has shipped.") as client:
        response = client.post(
            "/agents/support-agent/stream", json={"message": "status?", "thread_id": "t1"}
        )

    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert events[0] == ("start", {"thread_id": "t1"})
    assert events[-1] == ("end", {"thread_id": "t1"})
    tokens = [data["text"] for name, data in events if name == "token"]
    assert len(tokens) > 1
    assert "".join(tokens) == "Your order has shipped."


def _parse_sse(body: str) -> list[tuple[str, dict[str, str]]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events
