"""Engine-Fork-Tests fuer die Action-Intent-Emission (Task #35, sichere Aktuation).

Beweist die NEUE Naht (Konzept §0.1): das tool-lose Brain (no_mcp) emittiert einen
getypten Action-Intent als marker-abgegrenzten Block im Antworttext; der api-server
trennt ihn vom TTS-Text und haengt ihn als ADDITIVES Antwortfeld `action_intent`
(LISTE der Kandidaten) an. KEINE Validierung/Zaehlung in der Engine -- die fail-closed
Logik (0/1/>1, Schema, Lane, Registry) macht das Aktuations-Gate.

Mechanik wie tests/gateway/test_session_api.py: aiohttp TestClient, _run_agent gemockt,
keine echte LLM-Runde. Zusaetzlich Unit-Tests fuer extract_action_intent.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    extract_action_intent,
    _ACTION_INTENT_OPEN,
    _ACTION_INTENT_CLOSE,
)
from hermes_state import SessionDB


# --- Fixtures (1:1 wie test_session_api.py) ----------------------------------------

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


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


def _wrap(obj) -> str:
    payload = json.dumps(obj) if not isinstance(obj, str) else obj
    return f"{_ACTION_INTENT_OPEN}\n{payload}\n{_ACTION_INTENT_CLOSE}"


# --- Unit-Tests: extract_action_intent (dumb extract, kein Validieren) -------------

def test_no_marker_returns_text_and_none():
    text, candidates = extract_action_intent("Klar, ich habe verstanden.")
    assert text == "Klar, ich habe verstanden."
    assert candidates is None  # additive Feld bleibt aus


def test_non_string_returns_empty_and_none():
    text, candidates = extract_action_intent(None)
    assert text == ""
    assert candidates is None


def test_single_block_extracted_and_stripped_from_tts():
    intent = {
        "type": "jarvis.action.intent",
        "intent_id": "abc-123",
        "action": "sms.send",
        "params": {"recipient_ref": "kontakt:anna", "body": "Komme 10 Min spaeter."},
        "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"},
    }
    raw = f"Soll ich Anna schreiben? {_wrap(intent)}"
    text, candidates = extract_action_intent(raw)
    # Marker-Block vollstaendig aus dem TTS-Text entfernt.
    assert _ACTION_INTENT_OPEN not in text
    assert _ACTION_INTENT_CLOSE not in text
    assert "Soll ich Anna schreiben?" in text
    assert "kontakt:anna" not in text  # die rohen params landen NICHT im gesprochenen Text
    # Genau ein Kandidat, unveraendert geparst (Engine zaehlt/validiert nicht).
    assert candidates == [intent]


def test_multiple_blocks_all_forwarded_as_list():
    # Die >1->DROP-Verteidigung lebt im Gate -- die Engine MUSS beide Bloecke liefern,
    # sonst koennte ein "injiziere zwei Intents" still auf einen kollabieren.
    a = {"type": "jarvis.action.intent", "intent_id": "a", "action": "sms.send",
         "params": {"recipient_ref": "x", "body": "1"},
         "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"}}
    b = {"type": "jarvis.action.intent", "intent_id": "b", "action": "sms.send",
         "params": {"recipient_ref": "y", "body": "2"},
         "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"}}
    raw = f"Antwort {_wrap(a)} und {_wrap(b)}"
    text, candidates = extract_action_intent(raw)
    assert candidates == [a, b]          # beide weitergereicht, NICHT vorausgewaehlt
    assert _ACTION_INTENT_OPEN not in text


def test_non_json_block_skipped_but_stripped():
    raw = f"Text {_ACTION_INTENT_OPEN}das ist kein json{_ACTION_INTENT_CLOSE} Ende"
    text, candidates = extract_action_intent(raw)
    # Block war da -> Liste (nicht None), aber leer (nichts parsebar). Block dennoch entfernt.
    assert candidates == []
    assert _ACTION_INTENT_OPEN not in text
    assert "Text" in text and "Ende" in text


def test_two_blocks_do_not_collapse_into_one():
    # Non-greedy: zwei aufeinanderfolgende Bloecke duerfen nicht in einen verschmelzen.
    a = {"type": "jarvis.action.intent", "intent_id": "a", "action": "sms.send",
         "params": {"recipient_ref": "x", "body": "1"},
         "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"}}
    raw = f"{_wrap(a)}{_wrap(a)}"
    _text, candidates = extract_action_intent(raw)
    assert len(candidates) == 2


def test_block_only_output_strips_content_to_empty_but_keeps_candidate():
    # Brain emittiert NUR den Block, keine Prosa -> content wird leer (""), Kandidat bleibt.
    # FAIL-SOFT-Folge ist BRIDGE-seitig (sie ersetzt den Leer-Text durch einen Platzhalter,
    # damit der Turn nicht bricht) -- hier nur die Engine-Eigenschaft festgehalten.
    intent = {"type": "jarvis.action.intent", "intent_id": "only", "action": "sms.send",
              "params": {"recipient_ref": "x", "body": "1"},
              "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"}}
    text, candidates = extract_action_intent(_wrap(intent))
    assert text == ""               # nichts ausser dem Block -> leerer TTS-Text
    assert candidates == [intent]   # der Intent geht dennoch ans Gate


def test_block_cap_limits_parsing():
    a = {"type": "jarvis.action.intent", "intent_id": "a", "action": "sms.send",
         "params": {"recipient_ref": "x", "body": "1"},
         "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"}}
    raw = "".join(_wrap(a) for _ in range(50))
    _text, candidates = extract_action_intent(raw)
    # Cap greift: hoechstens _ACTION_INTENT_MAX_BLOCKS Kandidaten (>1 -> Gate DROP ohnehin).
    assert 0 < len(candidates) <= 8


# --- Route-Tests: additives Feld am /chat-Body ------------------------------------

@pytest.mark.asyncio
async def test_chat_attaches_action_intent_field(adapter, session_db):
    session_id = session_db.create_session("ai-session", "api_server")
    intent = {
        "type": "jarvis.action.intent",
        "intent_id": "turn-1",
        "action": "sms.send",
        "params": {"recipient_ref": "kontakt:anna", "body": "Bin gleich da."},
        "provenance": {"recipient_ref": "owner_spoken", "body": "owner_spoken"},
    }
    final = f"Soll ich Anna schreiben, dass du gleich da bist? {_wrap(intent)}"
    mock_run = AsyncMock(return_value=({"final_response": final, "session_id": session_id},
                                       {"total_tokens": 3}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat", json={"message": "schreib anna"})
            assert resp.status == 200
            payload = await resp.json()

    assert payload["object"] == "hermes.session.chat.completion"
    # TTS-Text ist bereinigt (kein Marker, keine rohen params).
    assert _ACTION_INTENT_OPEN not in payload["message"]["content"]
    assert "kontakt:anna" not in payload["message"]["content"]
    assert "Soll ich Anna schreiben" in payload["message"]["content"]
    # additives Feld = LISTE der Kandidaten (unvalidiert).
    assert payload["action_intent"] == [intent]


@pytest.mark.asyncio
async def test_chat_without_intent_omits_field(adapter, session_db):
    session_id = session_db.create_session("plain-session", "api_server")
    mock_run = AsyncMock(return_value=({"final_response": "Alles klar.", "session_id": session_id},
                                       {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat", json={"message": "hallo"})
            assert resp.status == 200
            payload = await resp.json()

    assert payload["message"]["content"] == "Alles klar."
    # Kein Marker -> Feld fehlt komplett (nicht null, nicht []).
    assert "action_intent" not in payload


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
