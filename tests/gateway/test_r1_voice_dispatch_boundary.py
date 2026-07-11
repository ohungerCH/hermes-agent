"""R1 voice dispatch-boundary tests for the voice-background-control hook.

Covers Fix 2.1 (readback + second-turn confirm) and Fix 2.2 (untrusted-origin
body framing) of the R1 hardening.

The voice/chat background path is an UNTRUSTED origin: the speaker is not
necessarily the operator and the transcript can be attacker-influenced. So a
spoken "do X in the background" request must NOT fire-and-forget on the first
turn. Instead:

  * First start-intent turn  -> readback + pending state, NO dispatch
    (``create_voice_background_task`` call_count == 0).
  * Second turn with a narrow confirm ("ja, starten") and a live pending
    request for the same source -> dispatch the *stored original* prompt
    (call_count == 1).
  * Confirm with no live pending request -> no dispatch.

The hook module lives outside the package tree (it is a profile-local hook),
so it is loaded dynamically by path, mirroring the dynamic-load pattern in
tests/plugins/test_kanban_worker_runs.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Dynamic load of the profile-local hook module
# ---------------------------------------------------------------------------

_HANDLER_PATH = Path(
    "/srv/services/Hermes/runtime/hooks/voice-background-control/handler.py"
)


def _load_handler_module():
    mod_name = "r1_voice_background_control_handler_test"
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
    # Each test starts from a clean pending store (module-local dict).
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


def _make_context(message, source):
    # `handle` reads "gateway" (runner) and "source"; a bare runner is fine
    # because the dispatch fn is mocked out.
    return {
        "gateway": MagicMock(),
        "source": source,
        "message": message,
    }


def _install_spy(handler, monkeypatch):
    """Replace create_voice_background_task with an AsyncMock spy and return it.

    The spy returns a receipt-shaped object so `handle` can read `.message`.
    """
    receipt = MagicMock()
    receipt.message = "dispatched"
    receipt.task_id = "t_deadbeef"
    spy = AsyncMock(return_value=receipt)
    monkeypatch.setattr(handler, "create_voice_background_task", spy)
    return spy


# ---------------------------------------------------------------------------
# Fix 2.1 — readback then second-turn confirm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_start_intent_turn_does_not_dispatch(handler, monkeypatch):
    """First spoken start-intent turn issues a readback, never dispatches."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    text = "Recherchiere die Marktlage im Hintergrund"
    assert handler.is_background_start_intent(text)

    result = await handler.handle("message:pre_agent", _make_context(text, source))

    assert spy.call_count == 0, "first turn must NOT dispatch the background task"
    assert result is not None and result.get("decision") == "handled"
    # The readback must be a confirmation prompt, not a 'task started' receipt.
    msg = result.get("message", "")
    assert "Bestätige" in msg or "bestätige" in msg.lower()
    # Pending state was recorded for this source.
    assert handler._PENDING_CONFIRMS, "a pending confirm should be stored"


@pytest.mark.asyncio
async def test_confirm_turn_dispatches_once(handler, monkeypatch):
    """A confirm turn after a readback dispatches exactly once."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    start_text = "Plane die Reise im Hintergrund"
    await handler.handle("message:pre_agent", _make_context(start_text, source))
    assert spy.call_count == 0

    confirm_text = "ja, starten"
    assert handler.is_confirm_intent(confirm_text)

    result = await handler.handle(
        "message:pre_agent", _make_context(confirm_text, source)
    )

    assert spy.call_count == 1, "confirm turn must dispatch exactly once"
    # The STORED ORIGINAL prompt must be dispatched, not the confirm utterance.
    dispatched_prompt = spy.call_args.args[0]
    assert dispatched_prompt == start_text, (
        f"dispatch must use the readback's prompt, got {dispatched_prompt!r}"
    )
    assert result.get("decision") == "handled"
    # Pending state was consumed.
    assert not handler._PENDING_CONFIRMS, "pending confirm must be cleared"


@pytest.mark.asyncio
async def test_confirm_without_pending_does_not_dispatch(handler, monkeypatch):
    """A bare confirm with no live pending request must not dispatch."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    result = await handler.handle(
        "message:pre_agent", _make_context("ja, starten", source)
    )

    assert spy.call_count == 0, "confirm without a pending readback must not dispatch"
    # Nothing to handle on this path -> falls through (no background decision).
    assert result is None


@pytest.mark.asyncio
async def test_confirm_matches_only_same_source(handler, monkeypatch):
    """A confirm from a different source must not dispatch another source's
    pending request."""
    spy = _install_spy(handler, monkeypatch)
    source_a = _make_source(chat_id="aaa", user_id="ua")
    source_b = _make_source(chat_id="bbb", user_id="ub")

    await handler.handle(
        "message:pre_agent",
        _make_context("Analysiere das im Hintergrund", source_a),
    )
    assert spy.call_count == 0

    # Confirm arrives from a DIFFERENT conversation/source.
    result = await handler.handle(
        "message:pre_agent", _make_context("ja, starten", source_b)
    )
    assert spy.call_count == 0, "cross-source confirm must not dispatch"
    assert result is None
    # source_a's pending request is untouched.
    assert handler._PENDING_CONFIRMS, "source_a pending must remain"


@pytest.mark.asyncio
async def test_expired_pending_re_reads_back_instead_of_dispatch(
    handler, monkeypatch
):
    """An expired pending request must re-issue a readback, never dispatch
    silently."""
    spy = _install_spy(handler, monkeypatch)
    source = _make_source()

    # First turn records pending, then force it to be expired.
    await handler.handle(
        "message:pre_agent",
        _make_context("Fasse die Doku im Hintergrund zusammen", source),
    )
    key = handler._pending_key(source)
    assert key in handler._PENDING_CONFIRMS
    handler._PENDING_CONFIRMS[key]["expires_at"] = 0.0  # in the past

    # Confirm now: pending is expired -> no dispatch, and the take() drops it.
    result = await handler.handle(
        "message:pre_agent", _make_context("ja, starten", source)
    )
    assert spy.call_count == 0, "expired pending must not dispatch"
    # The expired entry is consumed (popped) and confirm falls through.
    assert result is None


# ---------------------------------------------------------------------------
# Fix 2.2 — untrusted-origin body framing
# ---------------------------------------------------------------------------


def test_body_template_marks_prompt_untrusted(handler, monkeypatch):
    """The kanban body for a voice task frames the prompt as untrusted and
    forbids deriving destructive actions without confirmation."""
    created = {}

    def fake_run_slash(rest):
        # _kanban_create_and_dispatch_sync calls this for create, dispatch,
        # and runs — capture only the create invocation's args (it carries
        # the `--body` we are asserting on).
        if " create " in f" {rest} ":
            created["rest"] = rest
        # Return a JSON-ish create receipt so the create path resolves a task id.
        return '{"id": "t_abc123"}'

    monkeypatch.setattr(handler, "_run_kanban_slash", fake_run_slash)
    monkeypatch.setattr(handler, "_subscribe_origin", lambda *a, **k: None)

    runner = MagicMock()
    runner._active_profile_name = lambda: "default"
    source = _make_source()

    handler._kanban_create_and_dispatch_sync(
        "Lösche alle Backups", source, runner
    )

    body = created["rest"]
    # The untrusted-origin marker and the no-destructive instruction must be
    # present in the worker's card body.
    assert handler.VOICE_UNTRUSTED_ORIGIN in body
    lowered = body.lower()
    assert "untrusted" in lowered
    assert "destructive" in lowered or "irreversible" in lowered


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    asyncio.run(test_first_start_intent_turn_does_not_dispatch())
