"""German TTS normalization for ordered Markdown lists."""

import json

import tools.tts_tool as tts_tool
from tools.tts_tool import text_to_speech_tool


def test_german_ordered_list_markers_are_spelled_for_speech():
    text = "Intro\n1. Route prüfen\n2. Prozess kontrollieren\n3) Ergebnis melden\nVersion 1.2 bleibt gleich"

    normalized = tts_tool._normalize_german_ordered_list_markers(text)

    assert "1. Route" not in normalized
    assert "2. Prozess" not in normalized
    assert "3) Ergebnis" not in normalized
    assert "erstens: Route prüfen" in normalized
    assert "zweitens: Prozess kontrollieren" in normalized
    assert "drittens: Ergebnis melden" in normalized
    assert "Version 1.2 bleibt gleich" in normalized


def test_tts_tool_normalizes_before_provider_for_german_edge_voice(tmp_path, monkeypatch):
    captured = {}

    async def fake_generate_edge_tts(text, output_path, config):
        captured["text"] = text
        with open(output_path, "wb") as fh:
            fh.write(b"fake mp3")
        return output_path

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "edge", "edge": {"voice": "de-DE-KatjaNeural"}},
    )
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_generate_edge_tts)

    output_path = tmp_path / "speech.mp3"
    result = json.loads(text_to_speech_tool(
        text="1. Route prüfen\n2. Prozess kontrollieren\n3. Ergebnis melden",
        output_path=str(output_path),
    ))

    assert result["success"] is True
    assert output_path.exists()
    assert captured["text"] == (
        "erstens: Route prüfen\n"
        "zweitens: Prozess kontrollieren\n"
        "drittens: Ergebnis melden"
    )


def test_tts_tool_leaves_english_voice_numbered_lists_unchanged(tmp_path, monkeypatch):
    captured = {}

    async def fake_generate_edge_tts(text, output_path, config):
        captured["text"] = text
        with open(output_path, "wb") as fh:
            fh.write(b"fake mp3")
        return output_path

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "edge", "edge": {"voice": "en-US-AriaNeural"}},
    )
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_generate_edge_tts)

    output_path = tmp_path / "speech.mp3"
    result = json.loads(text_to_speech_tool(
        text="1. Check route\n2. Check process",
        output_path=str(output_path),
    ))

    assert result["success"] is True
    assert captured["text"] == "1. Check route\n2. Check process"


def test_tts_tool_pronounces_chur_as_khur_without_changing_canonical_text(tmp_path, monkeypatch):
    captured = {}

    async def fake_generate_edge_tts(text, output_path, config):
        captured["text"] = text
        with open(output_path, "wb") as fh:
            fh.write(b"fake mp3")
        return output_path

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "edge", "edge": {"voice": "de-DE-KatjaNeural"}},
    )
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_generate_edge_tts)

    output_path = tmp_path / "speech.mp3"
    original = "Die Route startet in Chur. Chur bleibt intern die korrekte Schreibweise."
    result = json.loads(text_to_speech_tool(text=original, output_path=str(output_path)))

    assert result["success"] is True
    assert "Khur" in captured["text"]
    assert "Schur" not in captured["text"]
    assert original == "Die Route startet in Chur. Chur bleibt intern die korrekte Schreibweise."
