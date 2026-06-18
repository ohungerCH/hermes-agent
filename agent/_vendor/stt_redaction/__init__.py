"""stt_redaction -- symmetrischer, fail-closed Egress-Redaktor fuer den
Jarvis-Audio-Pfad (STT-Transkript / TTS-Input / Logging).

Oeffentliche API:
    redact(text, policy=None) -> (redigierter_text, RedactionReport)
    RedactionPolicy(...)        -- Klassen-Auswahl + fail-closed-Modus
    RedactionReport             -- leak-sicherer Bericht (Klasse + Anzahl, KEIN Wert)
    RedactionError              -- fail-closed-Signal an den Caller (Egress blocken)

Drei SYMMETRISCHE Einbindungspunkte (alle rufen dieselbe redact()-Funktion;
Symmetrie ist load-bearing, sonst leckt PII ueber TTS- oder Log-Pfad zurueck):
    redact_for_llm_input(text, policy=None)   -- VOR STT->LLM-Dispatch
    redact_for_tts_output(text, policy=None)  -- VOR TTS-Synthese/Egress
    redact_for_log(text, policy=None)         -- VOR Logging/Task-Registry-Append
"""

from .redactor import (
    RedactionError,
    RedactionPolicy,
    RedactionReport,
    redact,
)

__all__ = [
    "redact",
    "RedactionPolicy",
    "RedactionReport",
    "RedactionError",
    "redact_for_llm_input",
    "redact_for_tts_output",
    "redact_for_log",
]

__version__ = "0.1.0"


def redact_for_llm_input(text, policy: RedactionPolicy | None = None):
    """Hook (a): VOR dem STT->LLM-Dispatch. Untrusted Transkript-String."""
    return redact(text, policy)


def redact_for_tts_output(text, policy: RedactionPolicy | None = None):
    """Hook (b): VOR der TTS-Synthese/dem Egress. Verhindert PII-Rueckleak
    ueber den Sprach-Ausgabepfad (Symmetrie zu Hook a)."""
    return redact(text, policy)


def redact_for_log(text, policy: RedactionPolicy | None = None):
    """Hook (c): VOR Logging/Task-Registry-Append. Nur den redigierten Text
    UND/ODER report.as_log_dict() persistieren, NIE den Rohtext."""
    return redact(text, policy)
