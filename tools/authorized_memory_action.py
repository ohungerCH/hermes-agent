"""Closed executor edge for a Jarvis-v2-confirmed memory action.

This is not a model tool.  It accepts only the canonical Memory action shape
that the Jarvis gate already bound to a verified execution claim, then reuses
``MemoryStore`` directly.  The legacy local approval adapter is the only
layer intentionally skipped; content scanning, limits, file locking, drift
detection, atomic replacement, and Vault shadowing remain in place.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from tools.memory_tool import (
    MemoryStore,
    _handle_recall,
    _shadow_batch_changes,
    load_on_disk_store,
)


_SKILLS = frozenset({"hermes-agent", "research-paper-writing"})
_TARGETS = frozenset({"memory", "user"})
_OPERATIONS = frozenset({"add", "replace", "remove", "batch", "recall"})
_MAX_CONTENT_CHARS = 2200
_MAX_SELECTOR_CHARS = 512
_MAX_BATCH_OPERATIONS = 16
_BASE_KEYS = frozenset({"skill_name", "operation", "target"})
_OPERATION_KEYS = {
    "add": (_BASE_KEYS | {"content"},),
    "replace": (_BASE_KEYS | {"old_text", "content"},),
    "remove": (_BASE_KEYS | {"old_text"},),
    "batch": (_BASE_KEYS | {"operations"},),
    "recall": (
        _BASE_KEYS | {"query"},
        _BASE_KEYS | {"query", "limit"},
    ),
}
_BATCH_KEYS = {
    "add": frozenset({"action", "content"}),
    "replace": frozenset({"action", "old_text", "content"}),
    "remove": frozenset({"action", "old_text"}),
}


class AuthorizedMemoryActionError(ValueError):
    """The action is open, ambiguous, or outside the v2 parameter ceiling."""


def _fail(code: str) -> None:
    raise AuthorizedMemoryActionError(code)


def _text(
    value: Any,
    *,
    field: str,
    maximum: int,
    multiline: bool,
) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or len(value.encode("utf-8")) > maximum * 4
    ):
        _fail(f"authorized_memory_{field}_invalid")
    for character in value:
        number = ord(character)
        if (
            number == 0x7F
            or number < 0x20
            and (not multiline or character not in {"\n", "\t"})
        ):
            _fail(f"authorized_memory_{field}_invalid")
    return value


def _batch_operation(raw: Any) -> Dict[str, str]:
    if type(raw) is not dict:
        _fail("authorized_memory_batch_invalid")
    action = raw.get("action")
    if (
        type(action) is not str
        or action not in _BATCH_KEYS
        or frozenset(raw) != _BATCH_KEYS[action]
        or any(type(key) is not str for key in raw)
    ):
        _fail("authorized_memory_batch_invalid")
    result = {"action": action}
    if "old_text" in raw:
        result["old_text"] = _text(
            raw["old_text"],
            field="old_text",
            maximum=_MAX_SELECTOR_CHARS,
            multiline=False,
        )
    if "content" in raw:
        result["content"] = _text(
            raw["content"],
            field="content",
            maximum=_MAX_CONTENT_CHARS,
            multiline=True,
        )
    return result


def parse_authorized_memory_action(raw: Any) -> Dict[str, Any]:
    """Copy and validate exactly the Jarvis ``memory.manage`` v1 shape."""

    if type(raw) is not dict:
        _fail("authorized_memory_params_invalid")
    skill = raw.get("skill_name")
    operation = raw.get("operation")
    target = raw.get("target")
    if (
        type(skill) is not str
        or skill not in _SKILLS
        or type(operation) is not str
        or operation not in _OPERATIONS
        or type(target) is not str
        or target not in _TARGETS
        or frozenset(raw) not in _OPERATION_KEYS[operation]
        or any(type(key) is not str for key in raw)
    ):
        _fail("authorized_memory_scope_invalid")
    result: Dict[str, Any] = {
        "skill_name": skill,
        "operation": operation,
        "target": target,
    }
    if "old_text" in raw:
        result["old_text"] = _text(
            raw["old_text"],
            field="old_text",
            maximum=_MAX_SELECTOR_CHARS,
            multiline=False,
        )
    if "content" in raw:
        result["content"] = _text(
            raw["content"],
            field="content",
            maximum=_MAX_CONTENT_CHARS,
            multiline=True,
        )
    if operation == "batch":
        operations = raw["operations"]
        if (
            type(operations) is not list
            or not 1 <= len(operations) <= _MAX_BATCH_OPERATIONS
        ):
            _fail("authorized_memory_batch_invalid")
        result["operations"] = [
            _batch_operation(item) for item in operations
        ]
    if operation == "recall":
        result["query"] = _text(
            raw["query"],
            field="query",
            maximum=_MAX_SELECTOR_CHARS,
            multiline=False,
        )
        if "limit" in raw:
            limit = raw["limit"]
            if type(limit) is not int or not 1 <= limit <= 8:
                _fail("authorized_memory_limit_invalid")
            result["limit"] = limit
    encoded = json.dumps(
        result,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 32_768:
        _fail("authorized_memory_params_oversized")
    return result


def _reason(result: Mapping[str, Any]) -> str:
    if result.get("drift_backup") is not None:
        return "external_drift"
    message = str(result.get("error", "")).lower()
    if "threat pattern" in message or "injection" in message:
        return "content_rejected"
    if "exceed" in message or "over the limit" in message:
        return "capacity_exceeded"
    if "no entry matched" in message or "no entry" in message:
        return "selector_not_found"
    if "multiple" in message and "matched" in message:
        return "selector_ambiguous"
    return "operation_rejected"


def _bounded_write_result(
    operation: str,
    target: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    if result.get("success") is not True:
        return {
            "schema_version": "jarvis.memory_executor.result.v1",
            "operation": operation,
            "status": "rejected",
            "target": target,
            "changed": False,
            "reason": _reason(result),
        }
    message = result.get("message")
    changed = not (
        isinstance(message, str)
        and "already exists" in message.lower()
    )
    return {
        "schema_version": "jarvis.memory_executor.result.v1",
        "operation": operation,
        "status": "completed",
        "target": target,
        "changed": changed,
        "usage": str(result.get("usage", "")),
        "entry_count": int(result.get("entry_count", 0)),
    }


def _bounded_recall_result(
    parsed: Mapping[str, Any],
    raw: str,
) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AuthorizedMemoryActionError(
            "authorized_memory_recall_result_invalid"
        ) from exc
    maximum = int(parsed.get("limit", 8))
    matches = value.get("matches")
    if (
        type(value) is not dict
        or type(value.get("available")) is not bool
        or type(matches) is not list
        or len(matches) > maximum
    ):
        _fail("authorized_memory_recall_result_invalid")
    bounded: List[Dict[str, str]] = []
    for item in matches:
        if (
            type(item) is not dict
            or frozenset(item) != {"source", "content"}
            or type(item.get("source")) is not str
            or type(item.get("content")) is not str
            or not item["source"]
            or len(item["source"]) > 64
            or len(item["content"]) > 4096
        ):
            _fail("authorized_memory_recall_result_invalid")
        bounded.append({
            "source": item["source"],
            "content": item["content"],
        })
    if not value["available"] and bounded:
        _fail("authorized_memory_recall_result_invalid")
    return {
        "schema_version": "jarvis.memory_executor.result.v1",
        "operation": "recall",
        "status": "completed",
        "target": parsed["target"],
        "available": value["available"],
        "matches": bounded,
    }


def _shadow_single(
    *,
    action: str,
    target: str,
    content: Optional[str],
    result: Dict[str, Any],
) -> None:
    old_entry = result.pop("_vault_old_entry", None)
    try:
        from tools.vault.vault_wiring import vault_shadow_write

        vault_shadow_write(
            action,
            target,
            content,
            store_result=result,
            old_entry=old_entry,
        )
    except Exception:
        pass


def apply_authorized_memory_action(
    raw: Any,
    *,
    store: Optional[MemoryStore] = None,
) -> Dict[str, Any]:
    """Execute one already-v2-confirmed action without legacy re-approval."""

    parsed = parse_authorized_memory_action(raw)
    memory = store if store is not None else load_on_disk_store()
    if type(memory) is not MemoryStore:
        _fail("authorized_memory_store_invalid")
    operation = parsed["operation"]
    target = parsed["target"]
    if operation == "recall":
        return _bounded_recall_result(
            parsed,
            _handle_recall(
                target,
                parsed["query"],
                parsed.get("limit"),
            ),
        )
    if operation == "add":
        result = memory.add(target, parsed["content"])
        _shadow_single(
            action="add",
            target=target,
            content=parsed["content"],
            result=result,
        )
    elif operation == "replace":
        result = memory.replace(
            target,
            parsed["old_text"],
            parsed["content"],
        )
        _shadow_single(
            action="replace",
            target=target,
            content=parsed["content"],
            result=result,
        )
    elif operation == "remove":
        result = memory.remove(target, parsed["old_text"])
        _shadow_single(
            action="remove",
            target=target,
            content=None,
            result=result,
        )
    else:
        result = memory.apply_batch(target, parsed["operations"])
        _shadow_batch_changes(result, target)
    return _bounded_write_result(operation, target, result)


__all__ = [
    "AuthorizedMemoryActionError",
    "apply_authorized_memory_action",
    "parse_authorized_memory_action",
]
