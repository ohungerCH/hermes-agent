"""User-specific voice alias normalization for Chur."""

from tools import transcription_tools
from tools import tts_tool


def test_stt_normalizes_spoken_city_variants_to_canonical_chur():
    text = "Plane die Route von Khur nach Zürich. Danach zurück nach Schur. Route ab Kur."

    normalized = transcription_tools._normalize_chur_aliases_in_transcript(text)

    assert normalized == "Plane die Route von Chur nach Zürich. Danach zurück nach Chur. Route ab Chur."


def test_stt_does_not_rewrite_plain_german_kur_without_location_context():
    text = "Die Kur dauert drei Wochen."

    normalized = transcription_tools._normalize_chur_aliases_in_transcript(text)

    assert normalized == text


def test_stt_result_keeps_raw_transcript_when_alias_changed():
    result = {"success": True, "transcript": "Ich starte in Kur und fahre weiter."}

    normalized = transcription_tools._normalize_stt_result_transcript(result, {"normalize_chur_aliases": True})

    assert normalized["transcript"] == "Ich starte in Chur und fahre weiter."
    assert normalized["raw_transcript"] == "Ich starte in Kur und fahre weiter."


def test_stt_alias_normalization_can_be_disabled():
    result = {"success": True, "transcript": "Ich starte in Khur."}

    normalized = transcription_tools._normalize_stt_result_transcript(result, {"normalize_chur_aliases": False})

    assert normalized is result
    assert "raw_transcript" not in normalized


def test_tts_pronounces_chur_as_khur_not_schur():
    text = "Die Route startet in Chur und endet wieder in Chur."

    normalized = tts_tool._normalize_tts_input_text(
        text,
        "edge",
        {"provider": "edge", "edge": {"voice": "de-DE-KatjaNeural"}},
    )

    assert normalized == "Die Route startet in Khur und endet wieder in Khur."
    assert "Schur" not in normalized
