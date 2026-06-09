"""Deterministic voice/chat background-work orchestration.

This module keeps background-work claims out of the model's free-form
conversation path.  A chat/voice background request is only acknowledged as
started after an auditable Kanban card exists on the ``voice-background`` board.
Status-like background questions are answered from evidence instead of model
inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any, Optional


VOICE_BACKGROUND_BOARD = "voice-background"
BACKGROUND_TRUTH_SKILL = "voice-background-truth-protocol"

_BACKGROUND_START_RE = re.compile(
    r"\b("
    r"mach(?:e|st)?|plane|plan|recherchier(?:e|st)?|prüf(?:e|st)?|pruef(?:e|st)?|"
    r"lies|fass(?:e)?|analysier(?:e|st)?|arbeite|führ(?:e)?|fuehr(?:e)?"
    r")\b.*(?:\b(im|in den|in einem)\s+hintergrund\b|\bhintergrund(?:task|auftrag|arbeit)\b)",
    re.IGNORECASE | re.DOTALL,
)
_BACKGROUND_STATUS_RE = re.compile(
    r"\b(hintergrund(?:prozess|task|auftrag|arbeit)?|background(?:[-_ ]?(?:process|task|job))?)\b",
    re.IGNORECASE,
)
_BACKGROUND_STATUS_WORD_RE = re.compile(
    r"\b(läuft|laeuft|fertig|status|cpu|last|beenden|stoppen|stoppe|kill|kanban|wo ist|prüf|pruef)\b",
    re.IGNORECASE,
)


@dataclass
class VoiceBackgroundReceipt:
    task_id: Optional[str]
    status: str
    message: str
    create_output: str = ""
    dispatch_output: str = ""
    runs_output: str = ""


def is_background_start_intent(text: str) -> bool:
    """Return True for explicit natural-language background start requests."""
    s = (text or "").strip()
    if not s or s.startswith("/"):
        return False
    return bool(_BACKGROUND_START_RE.search(s))


def is_background_status_intent(text: str) -> bool:
    """Return True for natural-language background status/stop/CPU questions."""
    s = (text or "").strip()
    if not s or s.startswith("/"):
        return False
    return bool(_BACKGROUND_STATUS_RE.search(s) and _BACKGROUND_STATUS_WORD_RE.search(s))


def _title_from_prompt(prompt: str) -> str:
    title = re.sub(r"\s+", " ", (prompt or "")).strip()
    title = re.sub(r"^(/background|/bg|/btw)\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\b(im|in den|in einem)\s+hintergrund\b", "", title, flags=re.IGNORECASE).strip(" -:;,.\n\t")
    if not title:
        title = "Voice background task"
    if len(title) > 96:
        title = title[:93].rstrip() + "..."
    return f"[voice-background] {title}" if not title.lower().startswith("[voice-background]") else title


def _source_field(source: Any, name: str) -> str:
    return str(getattr(source, name, "") or "")


def _platform_name(source: Any) -> str:
    platform = getattr(source, "platform", "")
    value = getattr(platform, "value", None)
    return (str(value) if value is not None else str(platform or "")).lower()


def _idempotency_key(prompt: str, source: Any) -> str:
    raw = "|".join([
        _platform_name(source),
        _source_field(source, "chat_id"),
        _source_field(source, "thread_id"),
        _source_field(source, "user_id"),
        re.sub(r"\s+", " ", prompt or "").strip().lower(),
    ])
    return "voice-bg-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _active_profile_name(runner: Any) -> str:
    fn = getattr(runner, "_active_profile_name", None)
    if callable(fn):
        try:
            name = fn()
            if name:
                return str(name)
        except Exception:
            pass
    return "default"


def _run_kanban_slash(rest: str) -> str:
    from hermes_cli.kanban import run_slash
    return run_slash(rest)


def _extract_task_id_from_create_output(output: str) -> Optional[str]:
    text = (output or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("id"):
            return str(data["id"])
    except Exception:
        pass
    m = re.search(r"\b(t_[0-9a-f]+)\b", text)
    return m.group(1) if m else None


def _subscribe_origin(task_id: str, runner: Any, source: Any) -> None:
    platform = _platform_name(source)
    chat_id = _source_field(source, "chat_id")
    if not platform or not chat_id:
        return
    from hermes_cli import kanban_db as kb

    conn = kb.connect(board=VOICE_BACKGROUND_BOARD)
    try:
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            thread_id=_source_field(source, "thread_id") or None,
            user_id=_source_field(source, "user_id") or None,
            notifier_profile=getattr(runner, "_kanban_notifier_profile", None) or _active_profile_name(runner),
        )
    finally:
        conn.close()


def _kanban_create_and_dispatch_sync(prompt: str, source: Any, runner: Any) -> VoiceBackgroundReceipt:
    assignee = _active_profile_name(runner)
    title = _title_from_prompt(prompt)
    body = (
        "Origin: gateway voice/chat background request\n"
        f"Platform: {_platform_name(source) or '-'}\n"
        f"Chat: {_source_field(source, 'chat_id') or '-'}\n"
        f"Thread: {_source_field(source, 'thread_id') or '-'}\n"
        f"User: {_source_field(source, 'user_id') or '-'}\n\n"
        "User request:\n"
        f"{prompt.strip()}\n\n"
        "Truth rule: report progress only from Kanban/run/tool evidence. Do not claim CPU, running, done, or stopped without a concrete handle."
    )
    create_cmd = " ".join([
        "--board", shlex.quote(VOICE_BACKGROUND_BOARD),
        "create", shlex.quote(title),
        "--body", shlex.quote(body),
        "--assignee", shlex.quote(assignee),
        "--created-by", "gateway-voice-background",
        "--priority", "1000",
        "--idempotency-key", shlex.quote(_idempotency_key(prompt, source)),
        "--max-runtime", "2h",
        "--skill", shlex.quote(BACKGROUND_TRUTH_SKILL),
        "--json",
    ])
    create_output = _run_kanban_slash(create_cmd)
    task_id = _extract_task_id_from_create_output(create_output)
    if not task_id:
        return VoiceBackgroundReceipt(
            task_id=None,
            status="failed",
            create_output=create_output,
            message=(
                "Ich habe keinen verifizierten Hintergrundtask gestartet.\n"
                "Grund: Kanban konnte keinen Task-Handle erzeugen."
            ),
        )

    try:
        _subscribe_origin(task_id, runner, source)
    except Exception:
        # Subscription improves UX but is not the task evidence itself.
        pass

    dispatch_output = _run_kanban_slash(
        f"--board {shlex.quote(VOICE_BACKGROUND_BOARD)} dispatch --max 1 --json"
    )
    runs_output = _run_kanban_slash(
        f"--board {shlex.quote(VOICE_BACKGROUND_BOARD)} runs {shlex.quote(task_id)}"
    )

    lower = f"{dispatch_output}\n{runs_output}".lower()
    if "spawn" in lower or "claimed" in lower or "running" in lower or "completed" in lower:
        status = "running_or_dispatched"
        status_line = "Status: Dispatch angestossen; Run-Evidence liegt vor."
    else:
        status = "created_ready"
        status_line = "Status: Task angelegt; noch keine Run-Evidence im Dispatch-Output."

    message = (
        "Hintergrundauftrag angelegt.\n"
        "Evidence:\n"
        f"- Mechanismus: Kanban\n"
        f"- Board: {VOICE_BACKGROUND_BOARD}\n"
        f"- Task: {task_id}\n"
        f"- {status_line}\n"
        f"- Prüfen: /kanban --board {VOICE_BACKGROUND_BOARD} show {task_id}\n"
        f"- Runs: /kanban --board {VOICE_BACKGROUND_BOARD} runs {task_id}"
    )
    return VoiceBackgroundReceipt(
        task_id=task_id,
        status=status,
        message=message,
        create_output=create_output,
        dispatch_output=dispatch_output,
        runs_output=runs_output,
    )


async def create_voice_background_task(prompt: str, source: Any, runner: Any) -> VoiceBackgroundReceipt:
    return await asyncio.to_thread(_kanban_create_and_dispatch_sync, prompt, source, runner)


def _summarize_status_sync(runner: Any) -> str:
    list_output = _run_kanban_slash(
        f"--board {shlex.quote(VOICE_BACKGROUND_BOARD)} list --status running"
    )
    ready_output = _run_kanban_slash(
        f"--board {shlex.quote(VOICE_BACKGROUND_BOARD)} list --status ready"
    )
    async_count = len([
        t for t in (getattr(runner, "_background_tasks", set()) or set())
        if hasattr(t, "done") and not t.done()
    ])
    combined = f"{list_output}\n{ready_output}"
    task_ids = sorted(set(re.findall(r"\b(t_[0-9a-f]+)\b", combined)))
    if not task_ids and async_count == 0:
        return (
            "Ich finde keinen belegten laufenden Voice-Hintergrundtask.\n"
            "Ich schreibe CPU-Last deshalb keinem Hintergrundprozess zu."
        )
    lines = ["Belegte Hintergrund-Evidence:"]
    if task_ids:
        lines.append(f"- Kanban-Board: {VOICE_BACKGROUND_BOARD}")
        lines.append(f"- Aktive/ready Task-Handles: {', '.join(task_ids[:8])}")
    if async_count:
        lines.append(f"- Gateway-/background Async-Jobs: {async_count}")
    lines.append("CPU-Zuordnung ist damit noch nicht bewiesen; dafür braucht es PID/Run-Log-Evidence.")
    return "\n".join(lines)


async def describe_voice_background_status(runner: Any) -> str:
    return await asyncio.to_thread(_summarize_status_sync, runner)
