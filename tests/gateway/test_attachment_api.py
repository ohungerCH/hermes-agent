from unittest.mock import AsyncMock, Mock, patch
import inspect

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_ATTACHMENT_BYTES,
    MAX_REQUEST_BYTES,
)
from tools.vault.vault_store import SOURCE_TABLE_OBJECT


class FakeAttachmentStore:

    @staticmethod
    def exif_context(data, content_type):
        return data, ""
    def __init__(self):
        self.put_calls = []
        self.loaded = type(
            "Loaded", (), {"data": b"image", "content_type": "image/png"}
        )()

    def put_transient(self, data, **kwargs):
        self.put_calls.append((data, kwargs))
        return type(
            "Record",
            (),
            {
                "object_key": "att_0123456789abcdef",  # gitleaks:allow -- test fixture, not a secret
                "expires_at": "2026-07-29T12:00:00+00:00",
                "content_type": kwargs["content_type"],
                "byte_size": len(data),
            },
        )()

    def load(self, *args, **kwargs):
        return self.loaded


def attachment_app(adapter):
    # Mirrors connect(): the attachment handler streams multipart and enforces
    # its own 25 MiB cap; unrelated routes keep the existing 10 MB app cap.
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app.router.add_post(
        "/api/jarvis/attachments/upload", adapter._handle_attachment_upload
    )
    app.router.add_post(
        "/api/jarvis/attachments/{attachment_ref}/memory",
        adapter._handle_attachment_memory,
    )
    return app


def test_api_server_general_body_cap_is_not_widened_for_attachment_route():
    source = inspect.getsource(APIServerAdapter.connect)
    assert "client_max_size=MAX_REQUEST_BYTES" in source


@pytest.mark.asyncio
async def test_upload_requires_edge_asserted_owner_identity():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._attachment_store = FakeAttachmentStore()
    async with TestClient(TestServer(attachment_app(adapter))) as cli:
        form = FormData()
        form.add_field("file", b"image", filename="photo.png", content_type="image/png")
        response = await cli.post("/api/jarvis/attachments/upload", data=form)
    assert response.status == 401


@pytest.mark.asyncio
async def test_upload_accepts_owner_image_and_returns_opaque_reference():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    store = FakeAttachmentStore()
    adapter._attachment_store = store
    headers = {
        "X-Jarvis-Device-Id": "dev-a",
        "X-Jarvis-Tenant-Id": "tenant-a",
        "X-Jarvis-Owner-Id": "owner-a",
    }
    async with TestClient(TestServer(attachment_app(adapter))) as cli:
        form = FormData()
        form.add_field("file", b"image", filename="photo.png", content_type="image/png")
        response = await cli.post(
            "/api/jarvis/attachments/upload", data=form, headers=headers
        )
        payload = await response.json()
    assert response.status == 201
    assert payload["attachment_ref"] == "att_0123456789abcdef"
    assert store.put_calls[0][0] == b"image"
    assert store.put_calls[0][1]["owner_id"] == "owner-a"


@pytest.mark.asyncio
async def test_upload_rejects_body_over_25_mib_without_persisting():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    store = FakeAttachmentStore()
    adapter._attachment_store = store
    headers = {
        "X-Jarvis-Device-Id": "dev-a",
        "X-Jarvis-Tenant-Id": "tenant-a",
        "X-Jarvis-Owner-Id": "owner-a",
    }
    async with TestClient(TestServer(attachment_app(adapter))) as cli:
        form = FormData()
        form.add_field(
            "file",
            b"x" * (MAX_ATTACHMENT_BYTES + 1),
            filename="large.png",
            content_type="image/png",
        )
        response = await cli.post(
            "/api/jarvis/attachments/upload", data=form, headers=headers
        )
    assert response.status == 413
    assert store.put_calls == []


@pytest.mark.asyncio
async def test_streamed_upload_can_exceed_general_api_cap_but_not_attachment_cap():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    store = FakeAttachmentStore()
    adapter._attachment_store = store
    headers = {
        "X-Jarvis-Device-Id": "dev-a",
        "X-Jarvis-Tenant-Id": "tenant-a",
        "X-Jarvis-Owner-Id": "owner-a",
    }
    payload = b"x" * (MAX_REQUEST_BYTES + 1)
    async with TestClient(TestServer(attachment_app(adapter))) as cli:
        form = FormData()
        form.add_field(
            "file", payload, filename="large-photo.jpg", content_type="image/jpeg"
        )
        response = await cli.post(
            "/api/jarvis/attachments/upload", data=form, headers=headers
        )
    assert response.status == 201
    assert store.put_calls[0][0] == payload


@pytest.mark.asyncio
async def test_image_memory_extraction_is_zero_tool_redacted_before_persist():
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "bridge-key"})
    )
    adapter._attachment_store = FakeAttachmentStore()
    adapter._persist_attachment_extraction = Mock(
        return_value={"status": "written", "object_key": None}
    )
    adapter._run_agent = AsyncMock(
        return_value=(
            {
                "final_response": (
                    "<<<JARVIS_IMAGE_EXTRACTION>>>"
                    '{"status":"ready","full_text":"sk_live_abcdef0123456789abcdef",'  # gitleaks:allow -- test fixture, not a secret
                    '"summary":"Mietvertrag","meta_context":"Meine Wohnung"}'
                    "<<<END_JARVIS_IMAGE_EXTRACTION>>>"
                )
            },
            {"total_tokens": 7},
        )
    )
    headers = {"Authorization": "Bearer bridge-key"}
    with patch.dict(
        "os.environ",
        {"JARVIS_VAULT_TENANT_ID": "tenant-a", "JARVIS_VAULT_OWNER_ID": "owner-a"},
    ):
        async with TestClient(TestServer(attachment_app(adapter))) as cli:
            response = await cli.post(
                "/api/jarvis/attachments/att_0123456789abcdef/memory",
                json={"operation": "remember", "owner_text": "Merk dir den Vertrag"},
                headers=headers,
            )
    assert response.status == 200
    kwargs = adapter._run_agent.await_args.kwargs
    assert kwargs["enabled_toolsets_override"] == []
    assert kwargs["user_message"][1]["type"] == "image_url"
    persisted = adapter._persist_attachment_extraction.call_args.kwargs["content"]
    assert persisted.startswith("Aus Foto/Dokument übernommen: ")
    assert "sk_live_abcdef0123456789abcdef" not in persisted  # gitleaks:allow -- test fixture, not a secret


@pytest.mark.asyncio
async def test_answered_meta_context_forbids_a_second_question():
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "bridge-key"})
    )
    adapter._attachment_store = FakeAttachmentStore()
    adapter._persist_attachment_extraction = Mock(
        return_value={"status": "written", "object_key": None}
    )
    adapter._run_agent = AsyncMock(
        return_value=(
            {
                "final_response": (
                    "<<<JARVIS_IMAGE_EXTRACTION>>>"
                    '{"status":"ready","full_text":"Vertragstext",'
                    '"summary":"Mietvertrag","meta_context":"Meine Wohnung"}'
                    "<<<END_JARVIS_IMAGE_EXTRACTION>>>"
                )
            },
            {"total_tokens": 7},
        )
    )
    headers = {"Authorization": "Bearer bridge-key"}
    with patch.dict(
        "os.environ",
        {"JARVIS_VAULT_TENANT_ID": "tenant-a", "JARVIS_VAULT_OWNER_ID": "owner-a"},
    ):
        async with TestClient(TestServer(attachment_app(adapter))) as cli:
            response = await cli.post(
                "/api/jarvis/attachments/att_0123456789abcdef/memory",
                json={
                    "operation": "remember",
                    "owner_text": "Merk dir den Vertrag. Kontextantwort: meine Wohnung.",
                    "context_answered": True,
                },
                headers=headers,
            )
    assert response.status == 200
    kwargs = adapter._run_agent.await_args.kwargs
    assert kwargs["enabled_toolsets_override"] == []
    assert "keine weitere Rueckfrage" in kwargs["ephemeral_system_prompt"]


def test_memory_write_uses_object_key_as_stable_source_even_without_keep():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    attachment_store = FakeAttachmentStore()
    attachment_store._crypto = object()
    adapter._attachment_store = attachment_store
    captured = []

    class FakeVaultStore:
        def __init__(self, **kwargs):
            pass

        def write(self, request):
            captured.append(request)
            return type("WriteResult", (), {"persisted": True, "status": "written"})()

    pool = Mock()
    pool.getconn.return_value = object()
    with (
        patch("tools.vault.db_runtime.get_vault_pool", return_value=pool),
        patch("tools.vault.vault_store.VaultStore", FakeVaultStore),
    ):
        result = adapter._persist_attachment_extraction(
            content="Aus Foto/Dokument übernommen: Mietvertrag",
            operation="remember",
            attachment_ref="att_0123456789abcdef",
            loaded=attachment_store.loaded,
            tenant_id="tenant-a",
            owner_id="owner-a",
        )

    assert result["status"] == "written"
    assert captured[0].source_table == SOURCE_TABLE_OBJECT
    assert captured[0].source_id == "att_0123456789abcdef"
    assert captured[0].raw_bytes is None
