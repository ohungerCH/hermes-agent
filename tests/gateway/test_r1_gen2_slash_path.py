"""R1 slash-path readback tests for the voice-background-control hook.

Round 1 added a readback + second-turn confirm to the *natural-language*
background-start path (``message:pre_agent``). Round 2 extends the SAME
untrusted-origin safety to the ``/background`` (and ``/bg``/``/btw``) slash
path handled via the ``command:background`` hook event.

The gateway origin is untrusted whether the request arrives as speech or as a
slash command — a chat gateway can carry an attacker-influenced
``/background ...``. So the slash path must no longer dispatch on the first
turn; instead it records a pending readback (keyed by source identity) and asks
for an explicit confirm. The confirm turn is handled by the existing
``message:pre_agent`` branch, because ``_take_pending_confirm`` is keyed by
source identity (not by slash-vs-NL), so "ja, starten" dispatches the stored
original prompt.

The hook module lives outside the package tree (profile-local hook); it is
loaded dynamically by path, mirroring tests/gateway/test_r1_voice_dispatch_boundary.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


_HANDLER_PATH = Path(
    "/srv/services/Hermes/runtime/hooks/voice-background-control/handler.py"
)


def _load_handler_module():
    mod_name = "r1_gen2_slash_path_handler_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if not _HANDLER_PATH.exists():
        pytest.skip(
            f"hook handler not deployed at {_HANDLER_PATH} (externes Jarvis-Deployment-"
            "Artefakt, im repo-only Test-Env abwesend) -- skip statt error"
        )
    spec = importlib.util.spec_from_file_location(mod_name, _HANDLER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def handler():
    mod = _load_handler_module()
    mod._PENDING_CONFIRMS.clear()
    yield mod
    mod._PENDING_CONFIRMS.clear()


def _make_source(chat_id="67890", user_id="12345", thread_id=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        user_id=user_id,
        thread_id=thread_id,
        user_name="testuser",
    )


def _command_context(prompt, source):
    # `handle` for command:background reads raw_args/args, plus gateway+source.
    return {
        "gateway": MagicMock(),
        "source": source,
        "raw_args": prompt,
        "args": prompt,
    }


def _message_context(message, source):
    return {
        "gateway": MagicMock(),
        "source": source,
        "message": message,
    }


def _install_spy(handler, monkeypatch):
    receipt = MagicMock()
    receipt.message = "dispatched"
    receipt.task_id = "t_deadbeef"
    spy = AsyncMock(return_value=receipt)
    monkeypatch.setattr(handler, "create_voice_background_task", spy)
    return spy


# ---------------------------------------------------------------------------
# (c) slash path: first turn -> readback (no dispatch); confirm -> dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_first_turn_does_not_dispatch(handler, monkeypatch):
    """`/background X` first turn issues a readback, never dispatches."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    result = await handler.handle(
        "command:background",
        _command_context("Recherchiere die Marktlage", source),
    )

    assert spy.call_count == 0, "slash first turn must NOT dispatch"
    assert result is not None and result.get("decision") == "handled"
    msg = result.get("message", "")
    assert "Bestätige" in msg or "bestätige" in msg.lower(), (
        "first turn must return a confirmation prompt, not a started receipt"
    )
    assert handler._PENDING_CONFIRMS, "a pending confirm must be stored"


@pytest.mark.asyncio
async def test_slash_then_confirm_dispatches_stored_prompt(handler, monkeypatch):
    """A confirm turn after a `/background` readback dispatches the stored
    original prompt exactly once — via the message:pre_agent confirm branch."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    start_prompt = "Plane die Reise nach Rom"
    await handler.handle(
        "command:background", _command_context(start_prompt, source)
    )
    assert spy.call_count == 0

    # Confirm arrives as a normal message turn (no slash).
    result = await handler.handle(
        "message:pre_agent", _message_context("ja, starten", source)
    )

    assert spy.call_count == 1, "confirm after slash readback must dispatch once"
    dispatched_prompt = spy.call_args.args[0]
    assert dispatched_prompt == start_prompt, (
        f"dispatch must use the slash readback's prompt, got {dispatched_prompt!r}"
    )
    assert result.get("decision") == "handled"
    assert not handler._PENDING_CONFIRMS, "pending confirm must be cleared"


@pytest.mark.asyncio
async def test_slash_empty_prompt_falls_through(handler, monkeypatch):
    """`/background` with no args falls through (built-in usage hint), no
    pending state, no dispatch."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    result = await handler.handle(
        "command:background", _command_context("   ", source)
    )

    assert result is None, "empty slash prompt must fall through to built-in"
    assert spy.call_count == 0
    assert not handler._PENDING_CONFIRMS


@pytest.mark.asyncio
async def test_slash_confirm_is_source_keyed(handler, monkeypatch):
    """A confirm from a different source must not dispatch another source's
    pending slash request (source-identity keying, not slash-vs-NL)."""
    spy = _install_spy(handler, monkeypatch)
    source_a = _make_source(chat_id="aaa", user_id="ua")
    source_b = _make_source(chat_id="bbb", user_id="ub")

    await handler.handle(
        "command:background", _command_context("Analysiere das", source_a)
    )
    assert spy.call_count == 0

    result = await handler.handle(
        "message:pre_agent", _message_context("ja, starten", source_b)
    )
    assert spy.call_count == 0, "cross-source confirm must not dispatch"
    assert result is None
    assert handler._PENDING_CONFIRMS, "source_a pending must remain"
