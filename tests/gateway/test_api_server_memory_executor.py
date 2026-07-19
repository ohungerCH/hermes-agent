"""Der Jarvis-Memory-Executor lebt nur explizit im API-Prozess."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _config(tmp_path: Path):
    return PlatformConfig(
        enabled=True,
        extra={
            "jarvis_memory_executor": {
                "enabled": True,
                "socket_path": str(tmp_path / "run" / "memory.sock"),
                "journal_path": str(
                    tmp_path / "state" / "memory-executor.sqlite3"
                ),
                "allowed_peer_uid": 1234,
            },
        },
    )


def test_disabled_by_default_and_no_runtime_is_created() -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    assert adapter._jarvis_memory_executor_status() == {
        "enabled": False,
        "running": False,
    }
    assert adapter._jarvis_memory_executor is None


def test_closed_deployment_env_configures_executor_without_volume_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARVIS_MEMORY_EXECUTOR_ENABLED", "true")
    monkeypatch.setenv(
        "JARVIS_MEMORY_EXECUTOR_SOCKET_PATH",
        str(tmp_path / "run" / "memory.sock"),
    )
    monkeypatch.setenv(
        "JARVIS_MEMORY_EXECUTOR_JOURNAL_PATH",
        str(tmp_path / "state" / "memory-executor.sqlite3"),
    )
    monkeypatch.setenv("JARVIS_MEMORY_EXECUTOR_ALLOWED_PEER_UID", "10027")

    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    assert adapter._jarvis_memory_executor_config == {
        "socket_path": tmp_path / "run" / "memory.sock",
        "journal_path":
            tmp_path / "state" / "memory-executor.sqlite3",
        "allowed_peer_uid": 10027,
    }


@pytest.mark.parametrize(
    "environment",
    (
        {"JARVIS_MEMORY_EXECUTOR_ENABLED": "yes"},
        {"JARVIS_MEMORY_EXECUTOR_ENABLED": "true"},
        {
            "JARVIS_MEMORY_EXECUTOR_ENABLED": "true",
            "JARVIS_MEMORY_EXECUTOR_SOCKET_PATH": "/run/memory.sock",
            "JARVIS_MEMORY_EXECUTOR_JOURNAL_PATH": "/state/memory.sqlite3",
            "JARVIS_MEMORY_EXECUTOR_ALLOWED_PEER_UID": "root",
        },
        {
            "JARVIS_MEMORY_EXECUTOR_ENABLED": "true",
            "JARVIS_MEMORY_EXECUTOR_SOCKET_PATH": "memory.sock",
            "JARVIS_MEMORY_EXECUTOR_JOURNAL_PATH": "/state/memory.sqlite3",
            "JARVIS_MEMORY_EXECUTOR_ALLOWED_PEER_UID": "10027",
        },
    ),
)
def test_open_or_partial_deployment_env_is_rejected(
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(
        ValueError,
        match="jarvis_memory_executor_config_invalid",
    ):
        APIServerAdapter(PlatformConfig(enabled=True))


def test_explicit_closed_config_starts_and_stops_owned_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            created["started"] = False

        def start(self):
            created["started"] = True

        def stop(self):
            created["stopped"] = True

        def status(self):
            return {
                "enabled": True,
                "running": created["started"],
                "socket_path": str(created["kwargs"]["socket_path"]),
            }

    monkeypatch.setattr(
        "tools.authorized_memory_executor_server."
        "AuthorizedMemoryExecutorUnixServer",
        FakeServer,
    )
    adapter = APIServerAdapter(_config(tmp_path))

    adapter._start_jarvis_memory_executor()

    assert created["started"] is True
    assert created["kwargs"] == {
        "socket_path": tmp_path / "run" / "memory.sock",
        "journal_path":
            tmp_path / "state" / "memory-executor.sqlite3",
        "allowed_peer_uid": 1234,
    }
    assert adapter._jarvis_memory_executor_status()["running"] is True

    adapter._stop_jarvis_memory_executor()
    assert created["stopped"] is True
    assert adapter._jarvis_memory_executor is None


@pytest.mark.asyncio
async def test_api_connect_and_disconnect_own_executor_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class FakeRunner:
        def __init__(self, _app):
            pass

        async def setup(self):
            events.append("http_setup")

        async def cleanup(self):
            events.append("http_cleanup")

    class FakeSite:
        def __init__(self, _runner, _host, _port):
            pass

        async def start(self):
            events.append("http_start")

        async def stop(self):
            events.append("http_stop")

    monkeypatch.setattr(
        "gateway.platforms.api_server.web.AppRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        "gateway.platforms.api_server.web.TCPSite",
        FakeSite,
    )
    monkeypatch.setattr(
        APIServerAdapter,
        "_port_is_available",
        lambda _self: True,
    )
    monkeypatch.setattr(
        APIServerAdapter,
        "_start_jarvis_memory_executor",
        lambda _self: events.append("memory_start"),
    )
    monkeypatch.setattr(
        APIServerAdapter,
        "_stop_jarvis_memory_executor",
        lambda _self: events.append("memory_stop"),
    )
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": "a" * 32},
        )
    )

    assert await adapter.connect() is True
    await adapter.disconnect()

    assert events == [
        "http_setup",
        "http_start",
        "memory_start",
        "memory_stop",
        "http_stop",
        "http_cleanup",
    ]


@pytest.mark.asyncio
async def test_memory_executor_start_failure_tears_down_http_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class FakeRunner:
        def __init__(self, _app):
            pass

        async def setup(self):
            events.append("http_setup")

        async def cleanup(self):
            events.append("http_cleanup")

    class FakeSite:
        def __init__(self, _runner, _host, _port):
            pass

        async def start(self):
            events.append("http_start")

        async def stop(self):
            events.append("http_stop")

    monkeypatch.setattr(
        "gateway.platforms.api_server.web.AppRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        "gateway.platforms.api_server.web.TCPSite",
        FakeSite,
    )
    monkeypatch.setattr(
        APIServerAdapter,
        "_port_is_available",
        lambda _self: True,
    )

    def fail_start(_self):
        raise ValueError("memory_executor_boot_failed")

    monkeypatch.setattr(
        APIServerAdapter,
        "_start_jarvis_memory_executor",
        fail_start,
    )
    monkeypatch.setattr(
        APIServerAdapter,
        "_stop_jarvis_memory_executor",
        lambda _self: events.append("memory_stop"),
    )
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"key": "a" * 32},
        )
    )

    assert await adapter.connect() is False
    assert events == [
        "http_setup",
        "http_start",
        "memory_stop",
        "http_stop",
        "http_cleanup",
    ]
    assert adapter._app is None
    assert adapter._runner is None
    assert adapter._site is None


@pytest.mark.parametrize(
    "value",
    (
        True,
        {},
        {"enabled": "yes"},
        {"enabled": False, "socket_path": "/tmp/open"},
        {
            "enabled": True,
            "socket_path": "relative.sock",
            "journal_path": "/tmp/state.sqlite3",
            "allowed_peer_uid": 1234,
        },
        {
            "enabled": True,
            "socket_path": "/tmp/memory.sock",
            "journal_path": "/tmp/state.sqlite3",
            "allowed_peer_uid": True,
        },
        {
            "enabled": True,
            "socket_path": "/tmp/memory.sock",
            "journal_path": "/tmp/state.sqlite3",
            "allowed_peer_uid": 1234,
            "open": True,
        },
    ),
)
def test_open_or_partial_config_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="jarvis_memory_executor_config_invalid"):
        APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"jarvis_memory_executor": value},
            )
        )
