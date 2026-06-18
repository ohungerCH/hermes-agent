"""R1 worker-deprivilege boundary tests for `_default_spawn`.

These cover Fix 1.1a + 1.1b of the R1 hardening: tasks created from the
gateway voice/chat background path (``created_by == "gateway-voice-background"``)
are an UNTRUSTED origin, so the dispatched worker must be deprivileged.

  1.1a (env gate): ``HERMES_EXEC_ASK=1`` is forced on (so approval.py's
        ``check_all_command_guards`` cannot fall into the auto-approve path
        for shell / execute_code), and any inherited ``HERMES_YOLO_MODE``
        bypass is stripped.
  1.1b (toolset narrowing): the worker argv carries ``-t web,vision`` AFTER
        the ``chat`` token, structurally removing terminal/process/
        execute_code/delegate_task. The kanban lifecycle toolset is still
        auto-appended at runtime via HERMES_KANBAN_TASK, so this does not
        strand the worker's card lifecycle.

Trusted-origin tasks (any other ``created_by``) must keep their full
capability surface and auto-approve env — that is the regression guard.

We intercept ``subprocess.Popen`` to capture the worker argv + env without
actually spawning a hermes subprocess (which would hang calling an LLM),
mirroring the established pattern in
tests/hermes_cli/test_kanban_core_functionality.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


VOICE_ORIGIN = "gateway-voice-background"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _spawn_and_capture(monkeypatch, *, created_by):
    """Create a task with the given origin, run _default_spawn under a fake
    Popen, and return (cmd, env) captured from the spawn call."""
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env", {}))
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # The bundled kanban-worker skill is absent from the isolated home; pin
    # availability False so argv shape doesn't depend on the test machine.
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="r1 depriv probe",
            assignee="some-profile",
            created_by=created_by,
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        pid = kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    assert pid == 4242
    assert "cmd" in captured and "env" in captured, "Popen was not invoked"
    return captured["cmd"], captured["env"]


# ---------------------------------------------------------------------------
# 1.1a — env approval gate for untrusted origin
# ---------------------------------------------------------------------------


def test_voice_origin_forces_exec_ask_and_strips_yolo(kanban_home, monkeypatch):
    """Untrusted voice origin => HERMES_EXEC_ASK=1 and no HERMES_YOLO_MODE."""
    # Simulate an inherited yolo bypass that must be stripped for this worker.
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    _cmd, env = _spawn_and_capture(monkeypatch, created_by=VOICE_ORIGIN)

    assert env.get("HERMES_EXEC_ASK") == "1", (
        "voice-origin worker must force the approval gate on"
    )
    assert "HERMES_YOLO_MODE" not in env, (
        "inherited yolo bypass must be stripped for the untrusted worker"
    )


def test_normal_origin_does_not_force_exec_ask(kanban_home, monkeypatch):
    """Trusted origin keeps its env: EXEC_ASK is NOT injected by deprivilege."""
    # Ensure no ambient EXEC_ASK so we can attribute its presence to the fix.
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

    _cmd, env = _spawn_and_capture(monkeypatch, created_by="some-user")

    assert env.get("HERMES_EXEC_ASK") is None, (
        "trusted-origin worker must not be force-gated by the R1 deprivilege"
    )


def test_normal_origin_preserves_inherited_yolo(kanban_home, monkeypatch):
    """Trusted origin must not have an inherited yolo bypass stripped."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    _cmd, env = _spawn_and_capture(monkeypatch, created_by="some-user")

    assert env.get("HERMES_YOLO_MODE") == "1", (
        "trusted-origin worker env must be left intact"
    )


# ---------------------------------------------------------------------------
# 1.1b — toolset narrowing for untrusted origin
# ---------------------------------------------------------------------------


def test_voice_origin_narrows_toolsets_after_chat(kanban_home, monkeypatch):
    """Untrusted voice origin => argv carries `-t web,vision` after `chat`."""
    cmd, _env = _spawn_and_capture(monkeypatch, created_by=VOICE_ORIGIN)

    assert "-t" in cmd, f"voice-origin argv must narrow toolsets: {cmd}"
    idx = cmd.index("-t")
    assert cmd[idx + 1] == "web,vision", (
        f"expected research-only toolsets, got {cmd[idx + 1]!r}"
    )
    # `-t` is owned by the `chat` subparser, so it must appear after `chat`.
    assert "chat" in cmd, f"argv missing chat subcommand: {cmd}"
    assert cmd.index("-t") > cmd.index("chat"), (
        f"-t must come after the chat token: {cmd}"
    )
    # The excluded capabilities must not be re-enabled via this flag.
    for forbidden in ("terminal", "process", "execute_code", "delegation"):
        assert forbidden not in cmd[idx + 1].split(","), (
            f"{forbidden} must not be in the narrowed toolset"
        )


def test_normal_origin_keeps_full_toolset(kanban_home, monkeypatch):
    """Trusted origin must NOT get the `-t` toolset narrowing flag."""
    cmd, _env = _spawn_and_capture(monkeypatch, created_by="some-user")

    assert "-t" not in cmd, (
        f"trusted-origin worker must keep its full toolset (no -t): {cmd}"
    )
