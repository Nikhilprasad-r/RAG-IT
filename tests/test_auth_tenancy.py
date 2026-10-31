import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from conftest import FakeChatModel, fake_models, tool_call, write_docs, write_project
from ragkit.agents import add_agent
from ragkit.core.config import ProjectConfig
from ragkit.core.errors import ConfigError
from ragkit.core.identity import Identity, acting_as
from ragkit.ingest import ingest
from ragkit.retrieval import KnowledgeBase
from ragkit.server import create_app

TENANTS = """\
name: saas
embeddings: {model: "hash:256"}
auth:
  mode: api_key
  api_keys:
    - {key_env: KEY_ACME, name: alice, tenant: acme, scopes: [chat, search, ingest]}
    - {key_env: KEY_ACME_HR, name: hank, tenant: acme, principals: [hr], scopes: [search]}
    - {key_env: KEY_GLOBEX, name: gina, tenant: globex, scopes: [chat, search]}
collections:
  acme-docs:
    tenant: acme
    sources: [{type: files, path: acme}]
  acme-hr:
    tenant: acme
    acl: [hr]
    sources: [{type: files, path: hr}]
  globex-docs:
    tenant: globex
    sources: [{type: files, path: globex}]
  public:
    tenant: platform
    shared: true
    sources: [{type: files, path: public}]
"""


@pytest.fixture
def saas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectConfig:
    for env, key in [("KEY_ACME", "k-acme"), ("KEY_ACME_HR", "k-hr"), ("KEY_GLOBEX", "k-globex")]:
        monkeypatch.setenv(env, key)
    for folder, text in [
        ("acme", "# Acme refunds\n\nAcme refunds within 14 days."),
        ("hr", "# Salaries\n\nAcme salary bands are reviewed in March."),
        ("globex", "# Globex refunds\n\nGlobex refunds within 60 days."),
        ("public", "# Status page\n\nThe status page is status.example.com."),
    ]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "doc.md").write_text(text)
    return write_project(tmp_path, TENANTS)


def _client(project: ProjectConfig, factory=None) -> TestClient:
    return TestClient(create_app(project, model_factory=factory or fake_models()))


def _search(client: TestClient, key: str, collection: str, query: str):
    return client.post(
        f"/collections/{collection}/search",
        json={"query": query, "mode": "sparse"},
        headers={"Authorization": f"Bearer {key}"},
    )


async def test_keys_scopes_and_tenant_isolation(saas: ProjectConfig) -> None:
    await ingest(saas)

    with _client(saas) as client:
        assert client.get("/collections").status_code == 401
        assert client.get("/collections", headers={"X-API-Key": "nope"}).status_code == 401
        acme = client.get("/collections", headers={"X-API-Key": "k-acme"}).json()
        globex_view = client.get("/collections", headers={"X-API-Key": "k-globex"}).json()
        own = _search(client, "k-acme", "acme-docs", "refunds")
        foreign = _search(client, "k-acme", "globex-docs", "refunds")
        shared = _search(client, "k-globex", "public", "status page")
        hr_hidden = _search(client, "k-acme", "acme-hr", "salary bands")
        hr_visible = _search(client, "k-hr", "acme-hr", "salary bands")
        no_scope = client.post(
            "/agents/x/chat", json={"message": "hi"}, headers={"X-API-Key": "k-hr"}
        )

    assert sorted(c["name"] for c in acme) == ["acme-docs", "acme-hr", "public"]
    assert sorted(c["name"] for c in globex_view) == ["globex-docs", "public"]
    assert "14 days" in own.json()[0]["text"]
    assert foreign.status_code == 404
    assert "status.example.com" in shared.json()[0]["text"]
    assert hr_hidden.json() == []  # ACL: alice lacks the hr principal
    assert "March" in hr_visible.json()[0]["text"]
    assert no_scope.status_code == 403


async def test_agents_search_as_the_caller(saas: ProjectConfig) -> None:
    await ingest(saas)
    add_agent("rag-chat", saas.agents_path)
    manifest = saas.agents_path / "rag-chat" / "agent.yaml"
    manifest.write_text(manifest.read_text().replace("[docs]", "[acme-docs, globex-docs]"))
    models: list[FakeChatModel] = []

    def factory(_manifest):
        models.append(FakeChatModel(messages=iter([AIMessage("a"), AIMessage("b")])))
        return models[-1]

    with _client(saas, factory) as client:
        for key in ("k-acme", "k-globex"):
            response = client.post(
                "/agents/rag-chat/chat",
                json={"message": "refunds policy"},
                headers={"X-API-Key": key},
            )
            assert response.status_code == 200, response.text

    acme_prompt, globex_prompt = (seen[0].text for seen in models[0].seen)
    assert "14 days" in acme_prompt and "60 days" not in acme_prompt
    assert "60 days" in globex_prompt and "14 days" not in globex_prompt


def test_threads_are_private_to_each_caller(saas: ProjectConfig) -> None:
    add_agent("support-agent", saas.agents_path)
    refund = tool_call("issue_refund", order_id="1", amount=5.0, reason="x")

    with _client(saas, fake_models(refund, "Hello Gina.", "Refunded.")) as client:
        alice = {"X-API-Key": "k-acme"}
        gina = {"X-API-Key": "k-globex"}
        waiting = client.post(
            "/agents/support-agent/chat",
            json={"message": "refund", "thread_id": "t1"},
            headers=alice,
        ).json()
        other = client.post(
            "/agents/support-agent/chat", json={"message": "hi", "thread_id": "t1"}, headers=gina
        ).json()
        hijack = client.post(
            "/agents/support-agent/resume",
            json={"thread_id": "t1", "decisions": [{"type": "approve"}]},
            headers=gina,
        )
        owner = client.post(
            "/agents/support-agent/resume",
            json={"thread_id": "t1", "decisions": [{"type": "approve"}]},
            headers=alice,
        ).json()

    assert waiting["status"] == "waiting_for_approval" and waiting["thread_id"] == "t1"
    assert other == other | {"status": "completed", "reply": "Hello Gina.", "thread_id": "t1"}
    assert hijack.status_code == 409
    assert owner["reply"] == "Refunded."


def test_jwt_claims_become_the_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "s" * 32)
    project = write_project(
        tmp_path,
        "name: t\nauth:\n  mode: jwt\n  jwt:\n    secret_env: JWT_SECRET\n"
        "    algorithms: [HS256]\n    audience: ragkit\n",
    )

    def token(**claims) -> dict[str, str]:
        body = {"sub": "u1", "tenant": "acme", "aud": "ragkit", "exp": time.time() + 60} | claims
        return {"Authorization": "Bearer " + jwt.encode(body, "s" * 32, algorithm="HS256")}

    with _client(project) as client:
        ok = client.get("/agents", headers=token(scope="chat search"))
        expired = client.get("/agents", headers=token(exp=time.time() - 5))
        wrong_audience = client.get("/agents", headers=token(aud="other"))
        search_only = client.get("/agents", headers=token(scope="search"))
        forged = client.get(
            "/agents",
            headers={
                "Authorization": "Bearer "
                + jwt.encode(
                    {"sub": "u", "aud": "ragkit", "exp": time.time() + 60},
                    "x" * 32,
                    algorithm="HS256",
                )
            },
        )

    assert ok.status_code == 200
    assert expired.status_code == wrong_audience.status_code == forged.status_code == 401
    assert search_only.status_code == 403


def test_rate_limit_per_caller(saas: ProjectConfig) -> None:
    text = (
        (saas.root / "ragkit.yaml")
        .read_text()
        .replace("mode: api_key", "mode: api_key\n  rate_limit_per_minute: 2")
    )
    project = write_project(saas.root, text)

    with _client(project) as client:
        codes = [client.get("/agents", headers={"X-API-Key": "k-acme"}) for _ in range(3)]
        other = client.get("/agents", headers={"X-API-Key": "k-globex"})

    assert [c.status_code for c in codes] == [200, 200, 429]
    assert int(codes[2].headers["Retry-After"]) >= 1
    assert other.status_code == 200


def test_missing_key_variable_fails_at_startup(
    saas: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KEY_GLOBEX")
    with pytest.raises(ConfigError, match="KEY_GLOBEX"):
        create_app(saas)


async def test_knowledge_base_uses_the_current_identity(saas: ProjectConfig) -> None:
    await ingest(saas)

    async with KnowledgeBase(saas) as kb:
        anonymous = await kb.search("acme-docs", "refunds", mode="sparse")
        with acting_as(Identity(tenant_id="acme")):
            acme = await kb.search("acme-docs", "refunds", mode="sparse")
            public = await kb.search("public", "status", mode="sparse")

    assert anonymous == []  # default tenant owns nothing in acme-docs
    assert len(acme) == 1 and len(public) == 1


def test_auth_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="needs api_keys"):
        write_project(tmp_path, "name: t\nauth: {mode: api_key}\n")
    with pytest.raises(ConfigError, match="exactly one of jwks_url or secret_env"):
        write_project(tmp_path, "name: t\nauth: {mode: jwt, jwt: {}}\n")
    write_docs(tmp_path)
