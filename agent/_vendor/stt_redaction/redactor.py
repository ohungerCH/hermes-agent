"""Symmetrischer STT/TTS-Egress-Redaktor (fail-closed, leak-sicher).

Designziel (audio_realtime_container_design.md Paragraf 5.3, ADR-0016
Paragraf 7.1, threat_model R5): EIN symmetrischer Redaktor, der an drei
Punkten greift -- VOR llm_input, VOR tts_output, VOR logging/task_registry --
und Secrets + PII (Email, Telefon, IBAN, Kreditkarte, Health/Art.9) aus dem
Transkript-String entfernt, BEVOR er die Container-/Egress-Grenze ueberquert.

Harte Eigenschaften:
- SELBST-ENTHALTEN: stdlib + re. Kein Import aus runtime/ (agent.redact).
- KEIN env-Opt-out. Anders als agent/redact.py (_REDACT_ENABLED) gibt es
  hier KEINEN Schalter, der die Redaction zur Laufzeit abschaltet -- genau
  diese Abschalt-Hintertuer ist das False-Green-Muster (bridge:170), das
  dieses Modul beseitigt. Klassen sind per RedactionPolicy waehlbar, aber
  "alles aus" ist nicht das stille Default-Verhalten und kein env-Kippschalter.
- FAIL-CLOSED: Jeder interne Fehler fuehrt NIE zur Rueckgabe von Rohtext.
  Entweder voll maskiert ODER Exception (Caller blockt den Egress).
- LEAK-SICHER: redact() liefert (text, report). Der Report enthaelt NIE den
  Klartext-Treffer und NIE den redigierten Wert -- nur Klasse + Anzahl + Bool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import patterns as P

# Voll-Ersetzung statt head/tail-Maskierung: ein Egress-Redaktor darf keine
# Klartext-Fragmente (z.B. "sk-p...7890") durchlassen. Der Klassen-Tag hilft
# dem nachgelagerten LLM beim Verstehen ("hier stand eine Email"), ohne den
# Wert preiszugeben (siehe Spec e: Over- vs Under-Redaction).
def _tag(cls: str) -> str:
    return f"[REDACTED:{cls}]"


class RedactionError(Exception):
    """Interner Redaktor-Fehler. Wird vom fail-closed-Pfad genutzt, damit der
    Caller den Egress hart blockiert, statt potenziell Rohtext durchzulassen."""


@dataclass(frozen=True)
class RedactionPolicy:
    """Welche Klassen sind aktiv. Default = sichere, hochpraezise Auswahl.

    `secrets`, `email`, `phone_e164`, `iban`, `credit_card` sind hochpraezise
    (Praefix-/Format-/Pruefsummen-gestuetzt) und default AN.
    `health` (Art.9) ist AN (DSGVO-Pflicht), aber lexikon-basiert -> hoeheres
    FP-Risiko, separat abschaltbar.
    `phone_national`, `address` sind FP-anfaellig und default AUS.
    """

    secrets: bool = True
    email: bool = True
    phone_e164: bool = True
    iban: bool = True
    credit_card: bool = True
    health: bool = True
    phone_national: bool = False
    address: bool = False

    # Bei Unsicherheit fail-closed: wirft eine interne Exception sich auf,
    # entscheidet dieser Schalter, ob redact() voll maskiert zurueckgibt
    # (raise_on_error=False) oder die Exception propagiert (=True). In beiden
    # Faellen wird NIE Rohtext zurueckgegeben. Default True = der Caller
    # (Bridge) trifft die Block-Entscheidung bewusst.
    raise_on_error: bool = True


@dataclass
class RedactionReport:
    """Leak-SICHERER Bericht. Enthaelt KEINE Klartext-Werte, KEINE Treffer-
    Inhalte. Nur Klasse, Anzahl je Klasse, Offsets, Gesamt-Bool. Spiegelt die
    ADR-0016-Paragraf-7.1-Logging-Policy (`redaction_patterns_matched` als Bool,
    `keine Werte`).

    Die `offsets`-Liste (vom Task explizit gefordert: "Klassen/Counts/Offsets")
    traegt je Treffer ein `(class, start, end)`-Tripel. Das sind INTEGER-
    POSITIONEN im REDIGIERTEN Endtext (siehe `_index_tags`), KEINE Klartext-
    Werte -- damit leak-sicher und konsistent zu ADR-0016 (`keine Werte`).
    Frame = Endtext (nicht Originaltext), weil die Passes den Text sequentiell
    mutieren und Original-Offsets unter spaeteren Passes verschieben wuerden;
    Positionen im stabilen Endtext sind eindeutig und reproduzierbar.
    """

    counts: dict[str, int] = field(default_factory=dict)
    offsets: list[tuple[str, int, int]] = field(default_factory=list)
    redaction_patterns_matched: bool = False
    fail_closed_masked: bool = False  # True falls fail-closed voll maskiert wurde

    def _bump(self, cls: str, n: int) -> None:
        if n:
            self.counts[cls] = self.counts.get(cls, 0) + n
            self.redaction_patterns_matched = True

    def as_log_dict(self) -> dict:
        """Form fuer Logging/Task-Registry. Nur Klassen + Zaehler + Offsets +
        Bool. Offsets sind Integer-Positionen, KEINE Werte."""
        return {
            "redaction_patterns_matched": self.redaction_patterns_matched,
            "redacted_classes": sorted(self.counts.keys()),
            "redacted_counts": dict(self.counts),
            "redacted_offsets": [list(o) for o in self.offsets],
            "fail_closed_masked": self.fail_closed_masked,
        }


# ---------------------------------------------------------------------------
# Pruefsummen-Validierung (schneidet False Positives hart)
# ---------------------------------------------------------------------------

def _luhn_ok(digits: str) -> bool:
    """Luhn-Pruefsumme fuer Kreditkartennummern."""
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    """ISO 13616 mod-97-Validierung. Uppercased intern -> lowercase-tolerant
    (Befund 1). FIX (Befund 2): Space, Punkt UND Bindestrich werden gestrippt,
    damit dotted/dashed IBANs (DE89.3704... / DE89-3704...) korrekt validieren."""
    s = re.sub(r"[ .\-]", "", candidate).upper()
    if not (15 <= len(s) <= 34):
        return False
    if not (s[:2].isalpha() and s[2:4].isdigit()):
        return False
    rearranged = s[4:] + s[:4]
    # A->10 ... Z->35
    digits = []
    for ch in rearranged:
        if ch.isdigit():
            digits.append(ch)
        elif "A" <= ch <= "Z":
            digits.append(str(ord(ch) - 55))
        else:
            return False
    try:
        return int("".join(digits)) % 97 == 1
    except ValueError:
        return False


def _e164_ok(candidate: str) -> bool:
    """E.164-Plausibilitaet (Befund 4): nach Entfernen aller Nicht-Ziffern
    7..15 Ziffern. Spiegelt die CC-Lane (Separatoren in der Regex tolerieren,
    Ziffern im Code zaehlen). '+' wird beim Strippen mit entfernt -> reine
    Ziffernzahl. '+49' allein (2 Ziffern) faellt damit korrekt raus."""
    digits = re.sub(r"\D", "", candidate)
    return 7 <= len(digits) <= 15


# ---------------------------------------------------------------------------
# Einzel-Passes. Jeder Pass nimmt (text, report) und gibt redigierten text
# zurueck; er aktualisiert report->_bump. Jeder Pass ist idempotent gegenueber
# bereits eingesetzten [REDACTED:...]-Tags (sie matchen keines der Muster).
# ---------------------------------------------------------------------------

def _count_sub(pattern: re.Pattern, repl, text: str) -> tuple[str, int]:
    new_text, n = pattern.subn(repl, text)
    return new_text, n


def _pass_secrets(text: str, rep: RedactionReport) -> str:
    n_total = 0
    if "BEGIN" in text and "-----" in text:
        text, n = _count_sub(P.PRIVATE_KEY_RE, _tag("PRIVATE_KEY"), text)
        n_total += n
    if any(s in text for s in P._PREFIX_SUBSTRINGS):
        text, n = _count_sub(P.PREFIX_RE, _tag("SECRET"), text)
        n_total += n
    if "eyJ" in text:
        text, n = _count_sub(P.JWT_RE, _tag("JWT"), text)
        n_total += n
    if "://" in text:
        text, n = _count_sub(
            P.DB_CONNSTR_RE, lambda m: f"{m.group(1)}{_tag('SECRET')}{m.group(3)}", text
        )
        n_total += n
    if "earer" in text:  # Bearer / bearer
        text, n = _count_sub(
            P.AUTH_HEADER_RE, lambda m: f"{m.group(1)}{_tag('SECRET')}", text
        )
        n_total += n
        text, n = _count_sub(
            P.BEARER_TOKEN_RE, lambda m: f"Bearer {_tag('SECRET')}", text
        )
        n_total += n
    if "=" in text:
        text, n = _count_sub(
            P.ENV_ASSIGN_RE,
            lambda m: f"{m.group(1)}={m.group(2)}{_tag('SECRET')}{m.group(2)}",
            text,
        )
        n_total += n
    # Passwort-/PIN-Keyword + Wert (DE/EN). Nur den WERT ersetzen, Keyword bleibt.
    text, n = _count_sub(
        P.PASSWORD_KEYWORD_RE, lambda m: f"{m.group(1)} {_tag('SECRET')}", text
    )
    n_total += n
    rep._bump("SECRET", n_total)
    return text


def _pass_email(text: str, rep: RedactionReport) -> str:
    if "@" not in text:
        return text
    text, n = _count_sub(P.EMAIL_RE, _tag("EMAIL"), text)
    rep._bump("EMAIL", n)
    return text


def _pass_phone_e164(text: str, rep: RedactionReport) -> str:
    if "+" not in text:
        return text
    before = text.count(_tag("PHONE"))
    def _sub(m: re.Match) -> str:
        # Befund 4: Separatoren in der Regex, Ziffern-Plausibilitaet im Code.
        # subn-n zaehlt auch verworfene Kandidaten (Ersatz==Original); deshalb
        # zaehlen wir tatsaechlich eingesetzte Tags (wie die IBAN-/CC-Lane).
        return _tag("PHONE") if _e164_ok(m.group(1)) else m.group(0)
    text = P.E164_PHONE_RE.sub(_sub, text)
    rep._bump("PHONE", text.count(_tag("PHONE")) - before)
    return text


def _pass_phone_national(text: str, rep: RedactionReport) -> str:
    text, n = _count_sub(P.NATIONAL_PHONE_RE, _tag("PHONE"), text)
    rep._bump("PHONE", n)
    return text


def _pass_iban(text: str, rep: RedactionReport) -> str:
    before = text.count(_tag("IBAN"))
    def _sub(m: re.Match) -> str:
        cand = m.group(1)
        if _iban_ok(cand):
            return _tag("IBAN")
        # FIX (Befund 2, Restleak): Form 2 der IBAN_CANDIDATE_RE kann gierig EINE
        # finale <Trenner><Kurzwort>-Gruppe ueber die IBAN hinaus konsumieren
        # (z.B. " ok"/" an"/" ja"/" ist"). Der ueberlange Kandidat faellt dann
        # mod-97 und der frueher returnte m.group(0) liess die VOLLE IBAN im
        # Klartext passieren (silent egress leak, kein fail-closed-Signal).
        # 'ok'/'an'/... sind strukturell gueltige 1..4-Zeichen-Gruppen -> keine
        # Regex kann sie von einer echten Schlussgruppe trennen, NUR mod-97.
        # Darum hier validator-seitig: von hinten Gruppen abschneiden und die
        # LAENGSTE eingebettete gueltige IBAN redigieren, den Nicht-IBAN-Schwanz
        # unveraendert wieder anhaengen. Laenderagnostisch (auch even-grouped
        # BE/NL...), separator-agnostisch, mehrwortrobust (' ja ja' via Schleife).
        # Form 1 (kontiguierlich, keine Trenner) hat keine Gruppen -> unveraendert.
        parts = re.split(r"([ .\-])", cand)   # [grp, sep, grp, sep, ...]
        idx = len(parts)
        while idx > 1:
            idx -= 2                          # je eine (sep, grp)-Endgruppe droppen
            if _iban_ok("".join(parts[:idx])):
                return _tag("IBAN") + "".join(parts[idx:])
        return m.group(0)
    # subn-n zaehlt auch Nicht-mod97-Kandidaten (Ersatz==Original); deshalb
    # zaehlen wir tatsaechlich eingesetzte Tags, nicht die subn-Trefferzahl.
    text = P.IBAN_CANDIDATE_RE.sub(_sub, text)
    rep._bump("IBAN", text.count(_tag("IBAN")) - before)
    return text


def _pass_credit_card(text: str, rep: RedactionReport) -> str:
    before = text.count(_tag("CREDIT_CARD"))
    def _sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(1))
        return _tag("CREDIT_CARD") if _luhn_ok(digits) else m.group(0)
    text = P.CREDIT_CARD_CANDIDATE_RE.sub(_sub, text)
    rep._bump("CREDIT_CARD", text.count(_tag("CREDIT_CARD")) - before)
    return text


def _pass_health(text: str, rep: RedactionReport) -> str:
    text, n = _count_sub(P.HEALTH_TERM_RE, _tag("HEALTH"), text)
    rep._bump("HEALTH", n)
    return text


def _pass_address(text: str, rep: RedactionReport) -> str:
    n_total = 0
    text, n = _count_sub(P.STREET_RE, _tag("ADDRESS"), text)
    n_total += n
    text, n = _count_sub(P.GERMAN_POSTCODE_CITY_RE, _tag("ADDRESS"), text)
    n_total += n
    rep._bump("ADDRESS", n_total)
    return text


# Reihenfolge ist load-bearing: IBAN/Kreditkarte VOR der nationalen Telefon-
# Heuristik (sonst frisst die Ziffern-Heuristik Karten-/IBAN-Stellen). Secrets
# zuerst (greifen URL-/Connstr-eingebettete Tokens, bevor andere Passes laufen).
def _ordered_passes(pol: RedactionPolicy) -> list[Callable[[str, RedactionReport], str]]:
    passes: list[Callable[[str, RedactionReport], str]] = []
    if pol.secrets:
        passes.append(_pass_secrets)
    if pol.email:
        passes.append(_pass_email)
    if pol.iban:
        passes.append(_pass_iban)
    if pol.credit_card:
        passes.append(_pass_credit_card)
    if pol.phone_e164:
        passes.append(_pass_phone_e164)
    if pol.phone_national:
        passes.append(_pass_phone_national)
    if pol.health:
        passes.append(_pass_health)
    if pol.address:
        passes.append(_pass_address)
    return passes


# Erkennt eingesetzte [REDACTED:CLASS]-Tags im Endtext, um leak-sichere
# Offsets (Integer-Positionen, KEINE Werte) abzuleiten. CLASS ist [A-Z_]+.
_TAG_SCAN_RE = re.compile(r"\[REDACTED:([A-Z_]+)\]")


def _index_tags(
    text: str,
    exclude: list[tuple[str, int, int]] | None = None,
) -> list[tuple[str, int, int]]:
    """Liefert je eingesetztem Tag (class, start, end) im REDIGIERTEN Endtext.

    Leak-sicher: nur Klasse + Integer-Positionen, nie ein Klartext-Wert. Wird
    nur aufgerufen, wenn in DIESEM redact()-Lauf tatsaechlich etwas redigiert
    wurde (rep.redaction_patterns_matched) -- so bleibt die Idempotenz erhalten:
    ein No-op-Zweitlauf ueber bereits redigierten Text setzt KEINE Offsets.

    FIX (Befund 8, offsets/counts-desync): Tags, die bereits im EINGABE-String
    standen (`exclude`-Spans, gematcht VOR den Passes), werden NICHT als Offsets
    gemeldet. So gilt Summe(counts) == len(offsets) auch dann, wenn der Eingabe-
    String schon einen [REDACTED:...]-Tag enthielt. Der Abgleich erfolgt
    KLASSENWEISE (gleiche Klasse, je 1x abgezogen), nicht ueber Positionen --
    die verschieben sich durch die Passes.

    BEKANNTE GRENZE (nicht-blockierend, leak-sicher): liegt ein vorbestehender
    Tag NACH einem in DIESEM Lauf gesetzten Tag DERSELBEN Klasse, zieht der
    klassenweise Abgleich irgendeinen Tag dieser Klasse ab -- der behaltene
    Offset kann dann auf den vorbestehenden statt den neuen Tag zeigen. Die
    ANZAHL bleibt korrekt (Summe(counts) == len(offsets), genau das geforderte
    Befund-8-Kriterium) und es leakt KEIN Wert (Offsets sind reine Integer).
    """
    all_tags = [(m.group(1), m.start(), m.end()) for m in _TAG_SCAN_RE.finditer(text)]
    if not exclude:
        return all_tags
    # Pre-existing Tags klassenweise abziehen (gleiche Klasse, je 1x).
    from collections import Counter
    pre = Counter(cls for cls, _, _ in exclude)
    result: list[tuple[str, int, int]] = []
    for cls, start, end in all_tags:
        if pre.get(cls, 0) > 0:
            pre[cls] -= 1
            continue
        result.append((cls, start, end))
    return result


# Bei einem internen Fehler: was bleibt vom Text uebrig, das ein Caller
# faelschlich als "sauber" durchlassen koennte? -> alles ersetzen.
_FULL_MASK = _tag("REDACTION_FAILED")


def redact(
    text: str | None,
    policy: RedactionPolicy | None = None,
) -> tuple[str, RedactionReport]:
    """Redigiere `text` und liefere (redigierter_text, leak-sicherer Report).

    FAIL-CLOSED: Bei JEDEM internen Fehler wird NIE Rohtext zurueckgegeben.
    Je nach policy.raise_on_error wird entweder eine RedactionError geworfen
    (Caller blockt Egress) oder voll maskierter Text zurueckgegeben.

    IDEMPOTENT: redact(redact(x)) liefert denselben Text -- bereits gesetzte
    [REDACTED:...]-Tags matchen keines der PII-/Secret-Muster.
    """
    pol = policy or RedactionPolicy()
    rep = RedactionReport()

    # None / Nicht-String defensiv behandeln; KEINE Klartext-Rueckgabe.
    if text is None:
        return "", rep
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as exc:  # pragma: no cover - defensiv
            if pol.raise_on_error:
                raise RedactionError("non-coercible input") from exc
            rep.fail_closed_masked = True
            return _FULL_MASK, rep
    if text == "":
        return "", rep

    # Tags, die bereits im Eingabe-String standen -- VOR den Passes erfassen,
    # damit sie nicht faelschlich als "in diesem Lauf eingesetzt" gezaehlt
    # werden (Befund 8, offsets/counts-Konsistenz).
    pre_existing = [
        (m.group(1), m.start(), m.end()) for m in _TAG_SCAN_RE.finditer(text)
    ]

    try:
        for fn in _ordered_passes(pol):
            text = fn(text, rep)
        # Leak-sichere Offsets NUR ableiten, wenn dieser Lauf wirklich etwas
        # redigiert hat. Ein idempotenter Zweitlauf ueber schon redigierten
        # Text matcht keine PII-Muster -> redaction_patterns_matched bleibt
        # False -> keine Offsets (konsistent zu counts == leer). Vorbestands-
        # Tags werden subtrahiert (Befund 8).
        if rep.redaction_patterns_matched:
            rep.offsets = _index_tags(text, exclude=pre_existing)
        return text, rep
    except Exception as exc:
        # Fail-closed: niemals den (potenziell noch teil-rohen) text zurueck.
        rep.fail_closed_masked = True
        if pol.raise_on_error:
            raise RedactionError("redaction pass failed; egress blocked") from exc
        # Voll maskieren: Laenge approximieren, KEIN Rohinhalt.
        return _FULL_MASK, rep
