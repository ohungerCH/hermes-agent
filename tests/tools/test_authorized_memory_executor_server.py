"""Geschlossene, replay-feste UDS-Grenze fuer v2-bestaetigte Memory-Actions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import stat
import threading

import pytest

from tools.authorized_memory_executor_server import (
    AuthorizedMemoryExecutor,
    AuthorizedMemoryExecutorError,
    SqliteAuthorizedMemoryExecutionStore,
    canonical_json_bytes,
    serve_memory_executor_connection,
)
from tools.memory_tool import MemoryStore


NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def _sha256(value) -> str:
    return "sha256-v1:" + hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def _request(**changes):
    params = {
        "skill_name": "hermes-agent",
        "operation": "add",
        "target": "memory",
        "content": "Owner prefers concise release notes.",
    }
    value = {
        "schema_version": "jarvis.memory_executor.request.v1",
        "execution_claim_id": "m6a.memory.executor-claim.123",
        "execution_claim_hash": "sha256-v1:" + "1" * 64,
        "execution_package_hash": "sha256-v1:" + "2" * 64,
        "materializer_resume_hash": "sha256-v1:" + "3" * 64,
        "idempotency_key": "m6a.memory.idempotency.123",
        "expires_at": (NOW + timedelta(seconds=30)).isoformat(),
        "request_id": "request.memory.123",
        "principal_id": "principal.owner.1",
        "tenant_id": "00000000-0000-4000-8000-000000000001",
        "owner_id": "owner.single.1",
        "workspace_id": "workspace.owner.1",
        "product_action_id": "memory.manage",
        "skill_id": "skill.hermes-agent",
        "capability_id": "tool.memory",
        "action_id": "invoke",
        "reserved_spool_sequence": 25,
        "params_hash": _sha256({
            "action": "invoke",
            "params": params,
        }),
        "params": params,
    }
    value.update(changes)
    return value


@pytest.fixture
def memory_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    value = MemoryStore(memory_char_limit=500, user_char_limit=300)
    value.load_from_disk()
    return value


@pytest.fixture
def journal(tmp_path: Path) -> SqliteAuthorizedMemoryExecutionStore:
    value = SqliteAuthorizedMemoryExecutionStore(
        tmp_path / "memory-executor.sqlite3"
    )
    yield value
    value.close()


def test_exact_request_executes_with_bound_identity_and_origin(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def shadow(action, target, content, **_kwargs):
        from tools.skill_provenance import get_current_write_origin
        from tools.vault.vault_wiring import get_vault_write_identity

        seen["action"] = action
        seen["origin"] = get_current_write_origin()
        seen["identity"] = get_vault_write_identity()

    monkeypatch.setattr(
        "tools.vault.vault_wiring.vault_shadow_write",
        shadow,
    )
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )

    response = executor.execute(_request())

    assert response["schema_version"] == (
        "jarvis.memory_executor.response.v1"
    )
    assert response["status"] == "executed"
    assert response["execution_claim_hash"] == _request()[
        "execution_claim_hash"
    ]
    assert response["result"]["status"] == "completed"
    assert response["result"]["changed"] is True
    assert "content" not in response["result"]
    assert seen == {
        "action": "add",
        "origin": "assistant_tool",
        "identity": (
            "00000000-0000-4000-8000-000000000001",
            "owner.single.1",
        ),
    }


def test_completed_request_replays_receipt_without_reexecution(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )
    first = executor.execute(_request())
    monkeypatch.setattr(
        "tools.authorized_memory_executor_server.apply_authorized_memory_action",
        lambda *_args, **_kwargs: pytest.fail("must not execute twice"),
    )

    assert executor.execute(_request()) == first

    journal.close()
    reopened = SqliteAuthorizedMemoryExecutionStore(journal.path)
    try:
        executor = AuthorizedMemoryExecutor(
            reopened,
            store_loader=lambda: memory_store,
            clock=lambda: NOW,
        )
        assert executor.execute(_request()) == first
    finally:
        reopened.close()


def test_completed_receipt_remains_replayable_after_request_expiry(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )
    first = executor.execute(_request())
    monkeypatch.setattr(
        "tools.authorized_memory_executor_server.apply_authorized_memory_action",
        lambda *_args, **_kwargs: pytest.fail("must not execute after expiry"),
    )
    expired_executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW + timedelta(minutes=5),
    )

    assert expired_executor.execute(_request()) == first


def test_journal_is_private_and_detects_response_corruption(
    memory_store: MemoryStore,
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "memory-executor.sqlite3"
    journal = SqliteAuthorizedMemoryExecutionStore(path)
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )
    executor.execute(_request())
    journal.close()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE memory_executor_runs_v1 "
        "SET response_json='{}' WHERE idempotency_key=?",
        (_request()["idempotency_key"],),
    )
    connection.commit()
    connection.close()

    with pytest.raises(
        AuthorizedMemoryExecutorError,
        match="memory_executor_store_row_invalid",
    ):
        SqliteAuthorizedMemoryExecutionStore(path)


def test_idempotency_collision_and_started_run_fail_closed(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )
    executor.execute(_request())
    other = _request(request_id="request.memory.other")
    with pytest.raises(
        AuthorizedMemoryExecutorError,
        match="memory_executor_idempotency_conflict",
    ):
        executor.execute(other)

    pending = _request(
        execution_claim_id="m6a.memory.executor-claim.pending",
        execution_claim_hash="sha256-v1:" + "4" * 64,
        idempotency_key="m6a.memory.idempotency.pending",
        request_id="request.memory.pending",
    )
    request_hash = journal.begin(pending)
    assert isinstance(request_hash, str)
    monkeypatch.setattr(
        "tools.authorized_memory_executor_server.apply_authorized_memory_action",
        lambda *_args, **_kwargs: pytest.fail(
            "indeterminate run must never execute automatically"
        ),
    )
    with pytest.raises(
        AuthorizedMemoryExecutorError,
        match="memory_executor_execution_indeterminate",
    ):
        executor.execute(pending)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(extra="open"),
        lambda value: value.update(product_action_id="terminal.run"),
        lambda value: value.update(capability_id="tool.terminal"),
        lambda value: value.update(action_id="write"),
        lambda value: value.update(reserved_spool_sequence=True),
        lambda value: value.update(params_hash="sha256-v1:" + "f" * 64),
        lambda value: value.update(expires_at=NOW.isoformat()),
        lambda value: value["params"].update(path="/root/.hermes"),
    ),
)
def test_open_stale_or_cross_bound_request_never_starts(
    mutation,
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
) -> None:
    raw = _request()
    mutation(raw)
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )

    with pytest.raises(AuthorizedMemoryExecutorError):
        executor.execute(raw)

    assert journal.count() == 0
    assert memory_store.memory_entries == []


def test_malformed_action_result_is_never_persisted_or_returned(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _request()
    raw["params"] = {
        "skill_name": "hermes-agent",
        "operation": "recall",
        "target": "memory",
        "query": "owner",
        "limit": 1,
    }
    raw["params_hash"] = _sha256({
        "action": "invoke",
        "params": raw["params"],
    })
    monkeypatch.setattr(
        "tools.authorized_memory_executor_server.apply_authorized_memory_action",
        lambda *_args, **_kwargs: {
            "schema_version": "jarvis.memory_executor.result.v1",
            "operation": "recall",
            "status": "completed",
            "target": "memory",
            "available": True,
            "matches": [{"source": "x", "content": "y", "raw": "leak"}],
        },
    )
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )

    with pytest.raises(
        AuthorizedMemoryExecutorError,
        match="memory_executor_result_invalid",
    ):
        executor.execute(raw)

    with pytest.raises(
        AuthorizedMemoryExecutorError,
        match="memory_executor_execution_indeterminate",
    ):
        executor.execute(raw)


def test_socket_peer_and_framing_are_fail_closed(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
) -> None:
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )
    server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    thread = threading.Thread(
        target=serve_memory_executor_connection,
        kwargs={
            "connection": server,
            "allowed_peer_uid": os.getuid(),
            "executor": executor,
        },
    )
    thread.start()
    client.sendall(canonical_json_bytes(_request()) + b"\n")
    client.shutdown(socket.SHUT_WR)
    response = json.loads(client.makefile("rb").read())
    thread.join(timeout=2)
    client.close()

    assert response["status"] == "executed"
    assert not thread.is_alive()

    server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    thread = threading.Thread(
        target=serve_memory_executor_connection,
        kwargs={
            "connection": server,
            "allowed_peer_uid": os.getuid() + 1,
            "executor": executor,
        },
    )
    thread.start()
    client.sendall(canonical_json_bytes(_request()) + b"\n")
    client.shutdown(socket.SHUT_WR)
    denied = json.loads(client.makefile("rb").read())
    thread.join(timeout=2)
    client.close()

    assert denied == {
        "schema_version": "jarvis.memory_executor.error.v1",
        "status": "deny",
        "reason": "memory_executor_request_failed",
    }


def test_socket_oversize_and_trailing_frame_never_execute(
    memory_store: MemoryStore,
    journal: SqliteAuthorizedMemoryExecutionStore,
) -> None:
    executor = AuthorizedMemoryExecutor(
        journal,
        store_loader=lambda: memory_store,
        clock=lambda: NOW,
    )
    for payload in (
        b"{" + b"x" * 40_000 + b"}\n",
        canonical_json_bytes(_request()) + b"\n{}\n",
    ):
        server, client = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        thread = threading.Thread(
            target=serve_memory_executor_connection,
            kwargs={
                "connection": server,
                "allowed_peer_uid": os.getuid(),
                "executor": executor,
            },
        )
        thread.start()
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        denied = json.loads(client.makefile("rb").read())
        thread.join(timeout=2)
        client.close()
        assert denied["status"] == "deny"

    assert journal.count() == 0
    assert memory_store.memory_entries == []
