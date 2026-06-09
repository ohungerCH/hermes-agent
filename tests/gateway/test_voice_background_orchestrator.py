"""Regression tests for deterministic voice/chat background orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.voice_background import (
    describe_voice_background_status,
    is_background_start_intent,
    is_background_status_intent,
)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        thread_id="th1",
        user_name="testuser",
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner._kanban_notifier_profile = None
    runner._active_profile_name = lambda: "default"
    return runner


@pytest.mark.parametrize(
    "text",
    [
        "Plane die Reise nach Palermo im Hintergrund",
        "Mach das bitte im Hintergrund weiter",
        "Prüfe die Logs in einem Hintergrundtask",
    ],
)
def test_background_start_intent_requires_explicit_background_work(text):
    assert is_background_start_intent(text)


@pytest.mark.parametrize(
    "text",
    [
        "Wir haben über Hintergrundprozesse gesprochen",
        "Was ist ein Background Task?",
        "/background plane etwas",
    ],
)
def test_background_start_intent_does_not_catch_discussion_or_slash(text):
    assert not is_background_start_intent(text)


def test_background_status_intent_catches_cpu_without_claiming_attribution():
    assert is_background_status_intent("Prüfe den Hintergrundprozess, er erzeugt CPU Last")


@pytest.mark.asyncio
async def test_background_command_returns_kanban_receipt_not_volatile_bg_claim():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._active_profile_name = lambda: "default"
    runner._kanban_notifier_profile = None
    event = MessageEvent(
        text="/background Plane Chur nach Palermo",
        source=_source(),
        message_id="m1",
    )

    def fake_run_slash(rest: str) -> str:
        if " create " in f" {rest} ":
            return '{"id": "t_abc123", "status": "ready", "assignee": "default"}'
        if " dispatch " in f" {rest} ":
            return '{"spawned": [{"task_id": "t_abc123", "run_id": 7}]}'
        if " runs " in f" {rest} ":
            return "run #7 running task=t_abc123"
        return ""

    with (
        patch("gateway.voice_background._run_kanban_slash", side_effect=fake_run_slash),
        patch("gateway.voice_background._subscribe_origin", return_value=None),
    ):
        result = await runner._handle_background_command(event)

    assert "Hintergrundauftrag angelegt" in result
    assert "Task: t_abc123" in result
    assert "Board: voice-background" in result
    assert "bg_" not in result


@pytest.mark.asyncio
async def test_status_question_without_evidence_refuses_cpu_attribution():
    runner = _runner()

    with patch("gateway.voice_background._run_kanban_slash", return_value="(no tasks)"):
        result = await describe_voice_background_status(runner)

    assert "keinen belegten laufenden Voice-Hintergrundtask" in result
    assert "CPU-Last" in result
    assert "keinem Hintergrundprozess" in result
