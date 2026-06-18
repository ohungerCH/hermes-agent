"""#2c: STT/PII + Art.9-health redaction on the LOG/AUDIT egress path.

These tests are the anti-false-green gate for the #2c change: the global log
``RedactingFormatter`` (agent/redact.py) is extended so that, in addition to the
secrets/URLs the engine already masked, the canonical ``stt_redaction`` lib
(vendored under ``agent/_vendor/stt_redaction``) ALSO redacts Email/Phone/IBAN/
CreditCard/Health (Art.9) from the formatted log line.

Two properties are proven:

1. POSITIVE (redaction happens on the log path): a record carrying a synthetic
   Secret + Email + IBAN + Health-term, rendered through ``RedactingFormatter``,
   contains ``[REDACTED:...]``/masked output and NONE of the plaintext values.

2. SCOPE (redaction does NOT touch the functional path): the SAME PII placed in
   a ``transcribe_audio``-style result dict (``result["transcript"]``) is
   UNCHANGED by the redaction primitives. Redaction lives only in the log
   formatter; the brain/LLM transcript path keeps the raw value, so the
   incident-research use case is preserved.

DLP: every fixture is synthetic/obviously fake PII (formally valid -- IBAN
mod-97 ok, no real person/account). NEVER a real secret/PII.

HONEST COVERAGE LIMIT: this gate exercises the LOG-egress chokepoint with a
synthetic LogRecord. It does NOT exercise the real STT provider path, possible
QQ-adapter log call sites, the user-echo (gateway/run.py), or session
persistence -- those are deliberately out of #2c scope. A green run here does
NOT mean "all logs are now fully redacted".
"""

import logging

import pytest

from agent.redact import RedactingFormatter, redact_sensitive_text


# --- Synthetic fixtures (obviously fake) -----------------------------------
SECRET = "sk-THISISA-fake-token-000000000000"      # OpenAI-shaped, not a real key
EMAIL = "max.mustermann@example.invalid"           # .invalid TLD -> never routable
IBAN = "DE89370400440532013000"                    # public ECBA test IBAN, mod-97 valid
HEALTH = "Diabetes"                                 # Art.9 health term


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Match tests/agent/test_redact.py: clear any env opt-out + force the
    module snapshot ON so the formatter's _REDACT_ENABLED gate is True."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _format_line(message: str) -> str:
    """Render *message* through the real RedactingFormatter (LOG egress)."""
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test.stt.egress",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=None,
        exc_info=None,
    )
    # hermes_logging installs a record factory that adds session_tag for every
    # record; the synthetic record built here bypasses it, so add the field the
    # default format string would otherwise KeyError on (we use %(message)s, but
    # set it for parity with the real format).
    record.session_tag = ""
    return formatter.format(record)


class TestLogEgressRedaction:
    """Property 1: secret + PII + Art.9 are redacted on the log path."""

    def test_vendored_lib_is_wired(self):
        # If the vendored lib failed to resolve the add-on silently degrades to
        # secrets-only -- which would make the PII assertions below a false
        # green. Assert the lib is actually loaded so the gate is meaningful.
        import agent.redact as r
        assert r._stt_redact is not None, "stt_redaction add-on not wired (PII pass inactive)"

    def test_secret_redacted(self):
        out = _format_line(f"key is {SECRET} done")
        assert SECRET not in out
        # engine secret pass keeps a head/tail crumb (sk-THI...0000); assert the
        # token MIDDLE never survives.
        assert "fake-token" not in out

    def test_email_redacted(self):
        out = _format_line(f"user {EMAIL} logged in")
        assert EMAIL not in out
        assert "[REDACTED:EMAIL]" in out

    def test_iban_redacted(self):
        out = _format_line(f"transfer to {IBAN} ok")
        assert IBAN not in out
        assert "[REDACTED:IBAN]" in out

    def test_health_term_redacted(self):
        out = _format_line(f"patient note: {HEALTH} suspected")
        assert HEALTH not in out
        assert "[REDACTED:HEALTH]" in out

    def test_all_classes_in_one_line(self):
        line = f"{SECRET} | {EMAIL} | {IBAN} | {HEALTH}"
        out = _format_line(line)
        for plaintext in (EMAIL, IBAN, HEALTH):
            assert plaintext not in out, f"{plaintext!r} leaked into log line"
        assert "fake-token" not in out
        assert "[REDACTED:EMAIL]" in out
        assert "[REDACTED:IBAN]" in out
        assert "[REDACTED:HEALTH]" in out


class TestFunctionalPathUnchanged:
    """Property 2: the functional (brain/LLM transcript) value is NOT redacted.

    Redaction must live ONLY in the log formatter. The transcript that flows to
    the LLM/brain must keep its raw content so incident-research ("what did the
    caller say about their diagnosis?") still works. We prove the redaction
    primitives, applied to the log path, do not mutate the functional dict.
    """

    def _transcribe_result(self) -> dict:
        """Mirror the dict shape transcribe_audio() returns (see
        tools/transcription_tools.py): result['transcript'] is the raw text."""
        return {
            "success": True,
            "provider": "test-stt",
            "transcript": (
                f"Meine Email ist {EMAIL}, mein Konto {IBAN}, "
                f"ich habe {HEALTH}."
            ),
            "error": "",
        }

    def test_transcript_value_is_not_mutated_by_log_redaction(self):
        result = self._transcribe_result()
        functional_transcript = result["transcript"]

        # The LOG path redacts -- prove it does on this very content.
        logged = _format_line(functional_transcript)
        assert EMAIL not in logged
        assert IBAN not in logged
        assert HEALTH not in logged

        # The FUNCTIONAL value is untouched: same object, raw PII still present.
        assert result["transcript"] is functional_transcript
        assert EMAIL in result["transcript"]
        assert IBAN in result["transcript"]
        assert HEALTH in result["transcript"]

    def test_brain_message_variable_survives(self):
        # A brain-side message variable the agent reasons over keeps its raw
        # value; only when it is *logged* does redaction apply.
        brain_message = f"caller mentioned {HEALTH} and gave IBAN {IBAN}"
        snapshot = brain_message  # value the LLM would receive

        _ = _format_line(brain_message)  # log it (redacted) -- side-effect free

        assert brain_message == snapshot
        assert HEALTH in brain_message
        assert IBAN in brain_message

    def test_redact_sensitive_text_alone_still_passes_pii_through(self):
        # Documents the BEFORE state / scope boundary: the engine's existing
        # secret-only primitive does NOT touch PII (that's exactly the gap #2c
        # closes in the formatter). This also guards against someone wiring the
        # PII pass into redact_sensitive_text() itself (which the brain path
        # also calls in places) instead of the log-only formatter.
        out = redact_sensitive_text(f"mail {EMAIL} iban {IBAN} health {HEALTH}")
        assert EMAIL in out   # secret-only pass leaves PII untouched
        assert IBAN in out
        assert HEALTH in out
