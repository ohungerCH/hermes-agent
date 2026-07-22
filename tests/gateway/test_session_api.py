"""Focused tests for API server session-control endpoints."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import jwt
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/api/trusted-surface/sessions", adapter._handle_trusted_surface_create_session)
    app.router.add_post(
        "/api/trusted-surface/sessions/{session_id}/chat",
        adapter._handle_trusted_surface_session_chat,
    )
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }


@pytest.fixture
def trusted_surface_adapter(session_db):
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "key": "sk-untrusted",
                "trusted_surface": {
                    "enabled": True,
                    "signing_key": "trusted-signing-key",
                },
            },
        )
    )
    adapter._session_db = session_db
    return adapter


def _trusted_surface_token(
    *,
    allowed_capabilities=None,
    allowed_toolsets=None,
    device_id: str = "devA",
    principal_id: str = "owner:owner1",
    workspace_id: str = "private",
    tenant_id: str = "1a7530bd-3ae8-46b4-96a6-86a510debdab",
    user_id: str = "user:owner1",
    owner_id: str = "owner1",
    role: str = "owner",
    auth_strength: str = "biometric_step_up",
) -> str:
    return jwt.encode(
        {
            "surface": "trusted_surface",
            "principal_id": principal_id,
            "role": role,
            "workspace_id": workspace_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "owner_id": owner_id,
            "device_id": device_id,
            "auth_strength": auth_strength,
            "allowed_toolsets": list(allowed_toolsets or []),
            "allowed_capabilities": list(
                allowed_capabilities
                or ["session.describe", "session.open", "session.chat"]
            ),
        },
        "trusted-signing-key",
        algorithm="HS256",
    )


@pytest.mark.asyncio
async def test_capabilities_advertise_trusted_surface_session_routes_when_live(
    trusted_surface_adapter,
):
    app = _create_session_app(trusted_surface_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities", headers={"Authorization": "Bearer sk-untrusted"})
        assert resp.status == 200
        data = await resp.json()

    assert data["endpoints"]["trusted_surface_session_describe"] == {
        "method": "GET",
        "path": "/api/trusted-surface/session/describe",
    }
    assert data["endpoints"]["trusted_surface_sessions"] == {
        "method": "POST",
        "path": "/api/trusted-surface/sessions",
    }
    assert data["endpoints"]["trusted_surface_session_chat"] == {
        "method": "POST",
        "path": "/api/trusted-surface/sessions/{session_id}/chat",
    }


@pytest.mark.asyncio
async def test_trusted_surface_create_session_requires_trusted_surface_bearer(
    trusted_surface_adapter,
):
    app = _create_session_app(trusted_surface_adapter)
    trusted_token = _trusted_surface_token()
    async with TestClient(TestServer(app)) as cli:
        rejected = await cli.post(
            "/api/trusted-surface/sessions",
            headers={"Authorization": "Bearer sk-untrusted"},
            json={},
        )
        assert rejected.status == 401

        ok = await cli.post(
            "/api/trusted-surface/sessions",
            headers={"Authorization": f"Bearer {trusted_token}"},
            json={},
        )
        assert ok.status == 201
        payload = await ok.json()

    assert payload["object"] == "hermes.trusted_surface.session"
    assert payload["surface"] == "trusted_surface"
    assert payload["identity"]["user_id"] == "user:owner1"
    assert payload["session"]["source"] == "trusted_surface"
    assert payload["session"]["user_id"] == "user:owner1"


@pytest.mark.asyncio
async def test_trusted_surface_chat_loads_history_and_uses_server_session_key(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-chat-session",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    session_db.append_message(session_id, "user", "trusted earlier")
    session_db.append_message(session_id, "assistant", "trusted prior answer")

    mock_run = AsyncMock(
        return_value=(
            {"final_response": "trusted fresh answer", "session_id": session_id},
            {"total_tokens": 7},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": "next trusted turn"},
                headers={
                    "Authorization": f"Bearer {_trusted_surface_token(allowed_toolsets=[])}",
                },
            )
            assert resp.status == 200
            payload = await resp.json()

    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert payload["object"] == "hermes.trusted_surface.chat.completion"
    assert payload["surface"] == "trusted_surface"
    assert payload["session_id"] == session_id
    assert payload["message"]["content"] == "trusted fresh answer"
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"].startswith("trusted_surface:")
    assert kwargs["enabled_toolsets_override"] == []
    assert kwargs["vault_tenant_id"] == "1a7530bd-3ae8-46b4-96a6-86a510debdab"
    assert kwargs["vault_owner_id"] == "owner1"
    history = kwargs["conversation_history"]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "trusted earlier"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "trusted prior answer"
    prompt = kwargs["ephemeral_system_prompt"]
    assert isinstance(prompt, str)
    assert "trusted surface live-slice contract" in prompt.lower()
    assert "persistent memory changes" in prompt.lower()
    assert "must never claim" in prompt.lower()
    assert "scratchpad/docs note writes may be prepared here" in prompt.lower()
    assert "explicit confirmation flow" in prompt.lower()
    assert "jarvis_action_intent" in prompt.lower()
    assert "docs.read" in prompt
    assert "docs.search" in prompt
    assert "docs.write" in prompt
    assert "m365.read" in prompt
    assert "research.request" in prompt
    assert "sms.send" in prompt
    assert "media.play" in prompt
    assert "memory.manage" in prompt


@pytest.mark.asyncio
async def test_trusted_surface_chat_passes_direct_kanban_toolset_override(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-kanban-session",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": "Ich habe die Aufgabe im Board angelegt.",
                "session_id": session_id,
            },
            {"total_tokens": 6},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": "Leg eine Kanban-Aufgabe an."},
                headers={
                    "Authorization": (
                        f"Bearer {_trusted_surface_token(allowed_toolsets=['kanban'])}"
                    )
                },
            )
            assert resp.status == 200
            payload = await resp.json()

    assert payload["message"]["content"] == "Ich habe die Aufgabe im Board angelegt."
    _, kwargs = mock_run.call_args
    assert kwargs["enabled_toolsets_override"] == ["kanban"]
    prompt = kwargs["ephemeral_system_prompt"]
    assert "direct hermes toolsets live today on this path: kanban" in prompt.lower()
    assert "use the direct kanban tools" in prompt.lower()
    assert "memory.manage" in prompt.lower()
    assert "never use a direct memory tool" in prompt.lower()
    assert "skill creation or edits are not live on this path yet" in prompt.lower()


@pytest.mark.asyncio
async def test_trusted_surface_memory_toolset_uses_direct_tool_and_disables_honesty_guard(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-memory-live-session",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": "Ich habe mir das gemerkt.",
                "session_id": session_id,
            },
            {"total_tokens": 8},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": "Merke dir bitte, dass Martin am Donnerstag telefonieren will."},
                headers={
                    "Authorization": (
                        "Bearer "
                        + _trusted_surface_token(
                            allowed_toolsets=["kanban", "memory", "skills"]
                        )
                    )
                },
            )
            assert resp.status == 200
            payload = await resp.json()

    mock_run.assert_awaited_once()
    assert "action_intent" not in payload
    assert payload["message"]["content"] == "Ich habe mir das gemerkt."
    _, kwargs = mock_run.call_args
    assert kwargs["enabled_toolsets_override"] == ["kanban", "memory", "skills"]
    assert kwargs["vault_tenant_id"] == "1a7530bd-3ae8-46b4-96a6-86a510debdab"
    assert kwargs["vault_owner_id"] == "owner1"
    prompt = kwargs["ephemeral_system_prompt"].lower()
    assert "use the direct memory tool" in prompt
    assert "confirmation card" not in prompt


@pytest.mark.parametrize(
    ("skill_name", "content"),
    [
        ("hermes-agent", "Mein Memory-Canary lautet HERMES-M6A-20260719-A."),
        (
            "research-paper-writing",
            "Mein Research-Memory-Canary lautet HERMES-M6A-20260719-B.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_trusted_surface_explicit_memory_command_is_normalized_without_model_tools(
    trusted_surface_adapter,
    session_db,
    skill_name,
    content,
):
    session_id = session_db.create_session(
        f"trusted-memory-deterministic-{skill_name}",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    user_message = f"Merke dir mit {skill_name} dauerhaft: {content}"
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": (
                    "Anweisung notiert und Auftrag in Aufgabenliste übernommen."
                ),
                "session_id": session_id,
            },
            {"total_tokens": 5},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": user_message},
                headers={
                    "Authorization": (
                        "Bearer "
                        + _trusted_surface_token(
                            allowed_toolsets=["kanban", "skills"]
                        )
                    )
                },
            )
            assert resp.status == 200
            payload = await resp.json()

    mock_run.assert_not_awaited()
    assert payload["message"]["content"] == (
        "Ich bereite die Memory-Aktion zur Bestätigung vor."
    )
    assert payload["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert len(payload["action_intent"]) == 1
    intent = payload["action_intent"][0]
    assert intent["type"] == "jarvis.action.intent"
    assert intent["intent_id"].startswith("memory-add-")
    assert len(intent["intent_id"]) <= 64
    assert intent["action"] == "memory.manage"
    assert intent["params"] == {
        "skill_name": skill_name,
        "operation": "add",
        "target": "memory",
        "content": content,
    }
    assert intent["provenance"] == {
        "skill_name": "owner_spoken",
        "operation": "owner_spoken",
        "target": "owner_spoken",
        "content": "owner_spoken",
    }
    messages = session_db.get_messages(session_id)
    assert [(message["role"], message["content"]) for message in messages] == [
        ("user", user_message),
        (
            "assistant",
            "Ich bereite die Memory-Aktion zur Bestätigung vor.",
        ),
    ]


@pytest.mark.asyncio
async def test_trusted_surface_identical_memory_add_is_idempotent_per_session(
    trusted_surface_adapter,
    session_db,
):
    first_session = session_db.create_session(
        "trusted-memory-idempotent-a",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    second_session = session_db.create_session(
        "trusted-memory-idempotent-b",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    message = (
        "Merke dir mit hermes-agent dauerhaft: "
        "Mein Memory-Canary lautet HERMES-M6A-IDEMPOTENT."
    )
    mock_run = AsyncMock()
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/trusted-surface/sessions/{first_session}/chat",
                json={"message": message},
                headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
            )
            retry = await cli.post(
                f"/api/trusted-surface/sessions/{first_session}/chat",
                json={"message": message},
                headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
            )
            separate_session = await cli.post(
                f"/api/trusted-surface/sessions/{second_session}/chat",
                json={"message": message},
                headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
            )
            assert first.status == retry.status == separate_session.status == 200
            first_payload = await first.json()
            retry_payload = await retry.json()
            separate_payload = await separate_session.json()

    mock_run.assert_not_awaited()
    first_id = first_payload["action_intent"][0]["intent_id"]
    assert retry_payload["action_intent"][0]["intent_id"] == first_id
    assert separate_payload["action_intent"][0]["intent_id"] != first_id


@pytest.mark.asyncio
async def test_trusted_surface_memory_turn_does_not_leave_partial_transcript(
    trusted_surface_adapter,
    session_db,
    monkeypatch,
):
    session_id = session_db.create_session(
        "trusted-memory-atomic-failure",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    real_insert = session_db._insert_message_rows

    def fail_after_first_row(conn, stored_session_id, messages):
        real_insert(conn, stored_session_id, messages[:1])
        raise RuntimeError("injected paired-turn failure")

    monkeypatch.setattr(
        session_db,
        "_insert_message_rows",
        fail_after_first_row,
    )
    mock_run = AsyncMock()
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={
                    "message": (
                        "Merke dir mit hermes-agent dauerhaft: "
                        "Dieser Turn muss atomar bleiben."
                    )
                },
                headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
            )
            assert resp.status == 503
            payload = await resp.json()

    mock_run.assert_not_awaited()
    assert payload["error"]["code"] == (
        "trusted_surface_memory_turn_persist_failed"
    )
    assert session_db.get_messages(session_id) == []


@pytest.mark.parametrize(
    "message",
    [
        "Merke dir mit unknown-skill dauerhaft: Inhalt.",
        "Merke dir mit hermes-agent dauerhaft:",
        "Merke dir mit hermes-agent dauerhaft: " + ("x" * 2201),
        "Merke dir mit hermes-agent dauerhaft: vor\u0001nach",
    ],
)
@pytest.mark.asyncio
async def test_trusted_surface_invalid_closed_memory_add_fails_without_model_tools(
    trusted_surface_adapter,
    session_db,
    message,
):
    session_id = session_db.create_session(
        f"trusted-memory-invalid-{len(message)}",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock()
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": message},
                headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
            )
            assert resp.status == 200
            payload = await resp.json()

    mock_run.assert_not_awaited()
    assert "action_intent" not in payload
    assert payload["message"]["content"] == (
        "Ich konnte keine bestätigbare Memory-Aktion vorbereiten. "
        "Es wurde nichts dauerhaft gespeichert oder geändert."
    )


@pytest.mark.asyncio
async def test_trusted_surface_memory_request_without_intent_cannot_claim_success(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-memory-missing-intent",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": (
                    "Anweisung notiert und Auftrag in Aufgabenliste übernommen."
                ),
                "session_id": session_id,
            },
            {"total_tokens": 5},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={
                    "message": "Erinnere dich bitte an Martin."
                },
                headers={
                    "Authorization": (
                        "Bearer "
                        + _trusted_surface_token(
                            allowed_toolsets=["kanban", "skills"]
                        )
                    )
                },
            )
            assert resp.status == 200
            payload = await resp.json()

    mock_run.assert_not_awaited()
    assert "action_intent" not in payload
    assert payload["message"]["content"] == (
        "Ich konnte keine bestätigbare Memory-Aktion vorbereiten. "
        "Es wurde nichts dauerhaft gespeichert oder geändert."
    )


@pytest.mark.asyncio
async def test_trusted_surface_chat_returns_additive_action_intent_without_marker_text(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-action-intent-session",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": (
                    "Ich bereite die Notiz vor.\n"
                    "<<<JARVIS_ACTION_INTENT>>>"
                    '{"intent_id":"docs-write-1","action":"docs.write","params":{"title":"Martin","content":"Martin will am Donnerstag telefonieren."}}'
                    "<<<END_JARVIS_ACTION_INTENT>>>"
                ),
                "session_id": session_id,
            },
            {"total_tokens": 11},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": "Leg eine Notiz an."},
                headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
            )
            assert resp.status == 200
            payload = await resp.json()

    assert payload["message"]["content"] == "Ich bereite die Notiz vor."
    assert payload["action_intent"] == [
        {
            "intent_id": "docs-write-1",
            "action": "docs.write",
            "params": {
                "title": "Martin",
                "content": "Martin will am Donnerstag telefonieren.",
            },
        }
    ]


@pytest.mark.asyncio
async def test_trusted_surface_chat_rejects_foreign_session_and_client_session_key(
    trusted_surface_adapter,
    session_db,
):
    api_server_session = session_db.create_session(
        "plain-api-session",
        "api_server",
        user_id="user:owner1",
    )
    trusted_session = session_db.create_session(
        "trusted-owned-session",
        "trusted_surface",
        user_id="user:owner1",
    )
    app = _create_session_app(trusted_surface_adapter)
    async with TestClient(TestServer(app)) as cli:
        wrong_source = await cli.post(
            f"/api/trusted-surface/sessions/{api_server_session}/chat",
            json={"message": "should fail"},
            headers={"Authorization": f"Bearer {_trusted_surface_token()}"},
        )
        assert wrong_source.status == 404

        injected_key = await cli.post(
            f"/api/trusted-surface/sessions/{trusted_session}/chat",
            json={"message": "should also fail"},
            headers={
                "Authorization": f"Bearer {_trusted_surface_token()}",
                "X-Hermes-Session-Key": "client-controlled-scope",
            },
        )
        assert injected_key.status == 400
        payload = await injected_key.json()

    assert payload["error"]["code"] == "trusted_surface_session_key_not_allowed"


@pytest.mark.asyncio
async def test_trusted_surface_memory_request_uses_confirmed_action_intent_not_direct_tool(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-memory-session",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": (
                    "Ich bereite die Erinnerung zur Bestätigung vor."
                    "<<<JARVIS_ACTION_INTENT>>>"
                    '{"type":"jarvis.action.intent","intent_id":"memory-add-2",'
                    '"action":"memory.manage","params":{"skill_name":"hermes-agent",'
                    '"operation":"add","target":"memory","content":'
                    '"Martin will am Donnerstag telefonieren."},'
                    '"provenance":{"content":"owner_spoken"}}'
                    "<<<END_JARVIS_ACTION_INTENT>>>"
                ),
                "session_id": session_id,
            },
            {"total_tokens": 9},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={
                    "message": (
                        "Bereite eine dauerhafte Memory-Aktion für Martin vor."
                    )
                },
                headers={"Authorization": f"Bearer {_trusted_surface_token(allowed_toolsets=[])}"},
            )
            assert resp.status == 200
            payload = await resp.json()

    mock_run.assert_awaited_once()
    assert payload["message"]["content"] == (
        "Ich bereite die Erinnerung zur Bestätigung vor."
    )
    assert payload["action_intent"][0]["action"] == "memory.manage"
    _, kwargs = mock_run.call_args
    assert kwargs["enabled_toolsets_override"] == []
    assert kwargs["vault_tenant_id"] == "1a7530bd-3ae8-46b4-96a6-86a510debdab"
    assert kwargs["vault_owner_id"] == "owner1"


@pytest.mark.asyncio
async def test_trusted_surface_memory_meta_question_still_reaches_model(
    trusted_surface_adapter,
    session_db,
):
    session_id = session_db.create_session(
        "trusted-memory-meta-session",
        "trusted_surface",
        user_id="user:owner1",
        model="test-model",
    )
    mock_run = AsyncMock(
        return_value=(
            {"final_response": "Das ist eine Meta-Erklärung.", "session_id": session_id},
            {"total_tokens": 5},
        )
    )
    app = _create_session_app(trusted_surface_adapter)
    with patch.object(trusted_surface_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/trusted-surface/sessions/{session_id}/chat",
                json={"message": "Wie funktioniert deine Erinnerung in diesem Modus?"},
                headers={"Authorization": f"Bearer {_trusted_surface_token(allowed_toolsets=[])}"},
            )
            assert resp.status == 200
            payload = await resp.json()

    mock_run.assert_awaited_once()
    assert payload["message"]["content"] == "Das ist eine Meta-Erklärung."


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert result["last_tool_outcome"] is None
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_run_agent_returns_last_memory_tool_outcome(adapter):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "scribe-session"

        def run_conversation(self, user_message, conversation_history, task_id):
            from tools.memory_tool import record_memory_tool_outcome

            record_memory_tool_outcome(
                "remove",
                {
                    "success": False,
                    "outcome": "class_not_removable",
                    "message": "Diese Erinnerungsklasse ist nicht löschbar",
                },
            )
            return {"final_response": "Ich habe gelöscht."}

    adapter._create_agent = Mock(return_value=FakeAgent())
    result, _ = await adapter._run_agent(
        user_message="Vergiss die Erinnerung.",
        conversation_history=[],
        session_id="scribe-session",
    )

    assert result["last_tool_outcome"] == {
        "action": "remove",
        "outcome": "class_not_removable",
        "message": "Diese Erinnerungsklasse ist nicht löschbar",
    }


def test_last_memory_tool_outcome_does_not_expose_legacy_error_details():
    from tools.memory_tool import (
        begin_memory_tool_turn,
        end_memory_tool_turn,
        get_last_memory_tool_outcome,
        record_memory_tool_outcome,
    )

    turn_context = begin_memory_tool_turn()
    try:
        record_memory_tool_outcome(
            "add",
            {"success": False, "error": "internal/path: validation detail"},
        )
        assert get_last_memory_tool_outcome() == {
            "action": "add",
            "outcome": "store_unavailable",
            "message": "Die Änderung wurde nicht bestätigt",
        }
    finally:
        end_memory_tool_turn(turn_context)


def test_recall_memory_tool_outcome_is_read_action_not_write_failure():
    """R6/N2: Recall bleibt als Leseaktion erkennbar und wird nie zum Schreibfehler."""
    from tools.memory_tool import (
        begin_memory_tool_turn,
        end_memory_tool_turn,
        get_last_memory_tool_outcome,
        record_memory_tool_outcome,
    )

    turn_context = begin_memory_tool_turn()
    try:
        record_memory_tool_outcome(
            "recall",
            {
                "action": "recall",
                "available": True,
                "matches": [],
                "note": "Kein Treffer zu dieser Anfrage im gemerkten Gedächtnis.",
            },
        )
        assert get_last_memory_tool_outcome() == {
            "action": "recall",
            "outcome": "recalled",
            "message": "Kein Treffer zu dieser Anfrage im gemerkten Gedächtnis.",
        }
    finally:
        end_memory_tool_turn(turn_context)


@pytest.mark.asyncio
async def test_run_agent_sets_nonempty_vault_write_identity(adapter):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "ok"}

    adapter._create_agent = Mock(return_value=FakeAgent())
    identity_token = object()
    set_identity = Mock(return_value=identity_token)
    reset_identity = Mock()

    with (
        patch(
            "tools.vault.vault_wiring.set_vault_write_identity",
            set_identity,
        ),
        patch(
            "tools.vault.vault_wiring.reset_vault_write_identity",
            reset_identity,
        ),
    ):
        await adapter._run_agent(
            user_message="remember this",
            conversation_history=[],
            session_id="owner-chat",
            enabled_toolsets_override=["memory"],
            vault_tenant_id="tenant-1",
            vault_owner_id="owner-primary",
        )

    set_identity.assert_called_once_with("tenant-1", "owner-primary")
    reset_identity.assert_called_once_with(identity_token)


@pytest.mark.asyncio
async def test_session_crud_and_message_history(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        create_resp = await cli.post("/api/sessions", json={"title": "Mobile chat", "model": "test-model"})
        assert create_resp.status == 201
        created = await create_resp.json()
        session_id = created["session"]["id"]
        assert created["object"] == "hermes.session"
        assert created["session"]["title"] == "Mobile chat"

        session_db.append_message(session_id, "user", "hello from phone")
        session_db.append_message(session_id, "assistant", "hello from hermes")

        list_resp = await cli.get("/api/sessions?limit=10&offset=0")
        assert list_resp.status == 200
        listed = await list_resp.json()
        assert listed["object"] == "list"
        assert [s["id"] for s in listed["data"]] == [session_id]
        assert listed["data"][0]["message_count"] == 2

        get_resp = await cli.get(f"/api/sessions/{session_id}")
        assert get_resp.status == 200
        got = await get_resp.json()
        assert got["session"]["id"] == session_id
        assert got["session"]["message_count"] == 2

        messages_resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()
        assert messages["object"] == "list"
        assert [m["role"] for m in messages["data"]] == ["user", "assistant"]
        assert messages["data"][0]["content"] == "hello from phone"

        patch_resp = await cli.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})
        assert patch_resp.status == 200
        patched = await patch_resp.json()
        assert patched["session"]["title"] == "Renamed"

        delete_resp = await cli.delete(f"/api/sessions/{session_id}")
        assert delete_resp.status == 200
        deleted = await delete_resp.json()
        assert deleted == {"object": "hermes.session.deleted", "id": session_id, "deleted": True}
        assert session_db.get_session(session_id) is None


@pytest.mark.asyncio
async def test_session_messages_follow_compression_tip(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server")
    session_db.append_message(source_id, "user", "before compression")
    session_db.end_session(source_id, "compression")
    session_db.create_session("tip-session", "api_server", parent_session_id=source_id)
    session_db.replace_messages(source_id, [])
    session_db.append_message("tip-session", "user", "after compression")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        messages_resp = await cli.get(f"/api/sessions/{source_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()

    assert messages["object"] == "list"
    assert messages["session_id"] == "tip-session"
    assert [m["content"] for m in messages["data"]] == ["after compression"]


@pytest.mark.asyncio
async def test_session_fork_uses_current_sessiondb_branch_primitives(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server", model="test-model")
    session_db.set_session_title(source_id, "Original")
    session_db.append_message(source_id, "user", "first path")
    session_db.append_message(source_id, "assistant", "answer")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(f"/api/sessions/{source_id}/fork", json={"title": "Alternative"})
        assert resp.status == 201
        payload = await resp.json()

    fork = payload["session"]
    assert payload["object"] == "hermes.session"
    assert fork["id"] != source_id
    assert fork["parent_session_id"] == source_id
    assert fork["title"] == "Alternative"
    assert [m["content"] for m in session_db.get_messages(fork["id"])] == ["first path", "answer"]
    assert session_db.get_session(source_id)["end_reason"] == "branched"


@pytest.mark.asyncio
async def test_session_chat_loads_history_and_preserves_session_headers(auth_adapter, session_db):
    session_id = session_db.create_session("chat-session", "api_server")
    session_db.set_session_title(session_id, "Chat")
    session_db.append_message(session_id, "user", "earlier")
    session_db.append_message(session_id, "assistant", "prior answer")

    tool_outcome = {
        "action": "remove",
        "outcome": "removed",
        "message": "Die Erinnerung wurde entfernt",
    }
    mock_run = AsyncMock(return_value=({
        "final_response": "fresh answer",
        "session_id": session_id,
        "last_tool_outcome": tool_outcome,
    }, {"total_tokens": 3}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "next", "system_message": "stay focused"},
                headers={"Authorization": "Bearer sk-test", "X-Hermes-Session-Key": "client-42"},
            )
            assert resp.status == 200
            payload = await resp.json()

    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert resp.headers["X-Hermes-Session-Key"] == "client-42"
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "fresh answer"
    assert payload["last_tool_outcome"] == tool_outcome
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"] == "client-42"
    assert kwargs["ephemeral_system_prompt"] == "stay focused"
    history = kwargs["conversation_history"]
    assert len(history) == 2
    assert isinstance(history[0].pop("timestamp"), (int, float))
    assert isinstance(history[1].pop("timestamp"), (int, float))
    assert history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.asyncio
async def test_session_chat_accepts_multimodal_message(auth_adapter, session_db):
    session_id = session_db.create_session("image-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    mock_run = AsyncMock(return_value=({"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": image_payload},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert resp.status == 200, await resp.text()

    _, kwargs = mock_run.call_args
    assert kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_accepts_zero_toolset_override_only(
    auth_adapter, session_db
):
    session_id = session_db.create_session("zero-tool-session", "api_server")
    mock_run = AsyncMock(
        return_value=(
            {"final_response": "safe answer", "session_id": session_id},
            {"total_tokens": 1},
        )
    )
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            ok = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "explain this", "enabled_toolsets": []},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert ok.status == 200, await ok.text()

            bad = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "explain this", "enabled_toolsets": ["terminal"]},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert bad.status == 400
            payload = await bad.json()

    _, kwargs = mock_run.call_args
    assert kwargs["enabled_toolsets_override"] == []
    assert payload["error"]["code"] == "enabled_toolsets_not_allowed"


@pytest.mark.asyncio
async def test_owner_chat_session_gets_server_memory_and_explicit_empty_override_wins(
    auth_adapter, session_db, monkeypatch
):
    monkeypatch.setenv(
        "JARVIS_VAULT_TENANT_ID",
        "1a7530bd-3ae8-46b4-96a6-86a510debdab",
    )
    monkeypatch.setenv("JARVIS_VAULT_OWNER_ID", "owner-primary")

    async def fake_run(**kwargs):
        return (
            {
                "final_response": "safe answer",
                "session_id": kwargs["session_id"],
            },
            {"total_tokens": 1},
        )

    mock_run = AsyncMock(side_effect=fake_run)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            owner_create = await cli.post(
                "/api/sessions",
                json={"id": "owner-chat", "owner_chat": True},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert owner_create.status == 201, await owner_create.text()
            plain_create = await cli.post(
                "/api/sessions",
                json={"id": "plain-chat"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert plain_create.status == 201, await plain_create.text()

            owner = await cli.post(
                "/api/sessions/owner-chat/chat",
                json={"message": "remember this"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert owner.status == 200, await owner.text()
            owner_kwargs = mock_run.await_args_list[-1].kwargs

            narrowed = await cli.post(
                "/api/sessions/owner-chat/chat",
                json={"message": "read only", "enabled_toolsets": []},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert narrowed.status == 200, await narrowed.text()
            narrowed_kwargs = mock_run.await_args_list[-1].kwargs

            monkeypatch.delenv("JARVIS_VAULT_OWNER_ID")
            missing_identity = await cli.post(
                "/api/sessions/owner-chat/chat",
                json={"message": "remember without complete server identity"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert missing_identity.status == 200, await missing_identity.text()
            missing_identity_kwargs = mock_run.await_args_list[-1].kwargs

            plain = await cli.post(
                "/api/sessions/plain-chat/chat",
                json={"message": "read only"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert plain.status == 200, await plain.text()
            plain_kwargs = mock_run.await_args_list[-1].kwargs

    assert owner_kwargs["enabled_toolsets_override"] == ["memory"]
    assert owner_kwargs["vault_tenant_id"] == (
        "1a7530bd-3ae8-46b4-96a6-86a510debdab"
    )
    assert owner_kwargs["vault_owner_id"] == "owner-primary"
    assert narrowed_kwargs["enabled_toolsets_override"] == []
    assert narrowed_kwargs["vault_tenant_id"] is None
    assert narrowed_kwargs["vault_owner_id"] is None
    assert missing_identity_kwargs["enabled_toolsets_override"] == ["memory"]
    assert missing_identity_kwargs["vault_tenant_id"] is None
    assert missing_identity_kwargs["vault_owner_id"] is None
    assert plain_kwargs["enabled_toolsets_override"] is None
    assert plain_kwargs["vault_tenant_id"] is None
    assert plain_kwargs["vault_owner_id"] is None


@pytest.mark.asyncio
async def test_session_chat_stream_accepts_multimodal_message(adapter, session_db):
    session_id = session_db.create_session("image-stream-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("A cat.")
        return {"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": image_payload},
            )
            assert resp.status == 200, await resp.text()
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: assistant.completed" in body
    assert captured_kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_emits_lifecycle_events_and_keepalive_safe_shape(adapter, session_db):
    session_id = session_db.create_session("stream-session", "api_server")
    session_db.set_session_title(session_id, "Stream")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Hello")
        kwargs["stream_delta_callback"](" world")
        kwargs["tool_progress_callback"]("reasoning.available", tool_name="_thinking", preview="thinking")
        return {"final_response": "Hello world", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "stream please"})
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: run.started" in body
    assert "event: message.started" in body
    assert "event: assistant.delta" in body
    assert "Hello world" in body
    assert "event: tool.progress" in body
    assert "event: assistant.completed" in body
    assert "event: run.completed" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)



@pytest.mark.asyncio
async def test_session_endpoints_require_auth_when_key_configured(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/sessions")
        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["code"] == "invalid_api_key"

        ok = await cli.get("/api/sessions", headers={"Authorization": "Bearer sk-test"})
        assert ok.status == 200
        data = await ok.json()
        assert data["object"] == "list"
        assert data["data"] == []


@pytest.mark.asyncio
async def test_session_header_rejected_without_api_key(adapter, session_db):
    session_id = session_db.create_session("unsafe-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello"},
            headers={"X-Hermes-Session-Key": "client-42"},
        )
        assert resp.status == 403
        data = await resp.json()
        assert "X-Hermes-Session-Key requires API key" in data["error"]["message"]
