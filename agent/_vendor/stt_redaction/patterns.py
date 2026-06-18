"""Kompilierte Muster fuer den Redaktor (stdlib/regex only).

Diese Datei ist SELBST-ENTHALTEN: kein Import aus dem runtime/-Baum
(`agent/redact.py`). Die Secret-Muster sind aus `agent/redact.py`
PORTIERT (nicht importiert), damit dieses Modul in den Bridge-Container
verlagert werden kann, ohne in den hermes-agent-Prozess hineinzugreifen
(audio_realtime_container_design.md, Paragraf 5.3: CONTAINER-FIRST).

Konvention: KEINE echten Secret-Werte in dieser Datei. Alle Muster sind
Form-Beschreibungen, keine Klartext-Geheimnisse.
"""

import re

# ---------------------------------------------------------------------------
# Secret-/Credential-Klasse (portiert aus agent/redact.py, gekuerzt + erweitert)
# ---------------------------------------------------------------------------

# Bekannte API-Key-Praefixe -- Praefix + zusammenhaengende Token-Zeichen.
# Aus agent/redact.py portiert; die Liste ist eine Form-Allowlist, keine Werte.
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",            # OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"sk_live_[A-Za-z0-9]{10,}",         # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",         # Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",         # Stripe restricted key
    r"sk_[A-Za-z0-9_]{10,}",             # ElevenLabs TTS key (sk_ underscore)
    r"ghp_[A-Za-z0-9]{10,}",             # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",     # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",             # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",             # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",             # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",             # GitHub refresh token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",     # Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",           # Google API keys
    r"AKIA[A-Z0-9]{16}",                 # AWS Access Key ID
    r"SG\.[A-Za-z0-9_-]{10,}",           # SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",              # HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",              # Replicate API token
    r"npm_[A-Za-z0-9]{10,}",             # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",          # PyPI API token
    r"gsk_[A-Za-z0-9]{10,}",             # Groq Cloud API key (Jarvis-STT-Egress)
    r"pplx-[A-Za-z0-9]{10,}",            # Perplexity
    r"tvly-[A-Za-z0-9]{10,}",            # Tavily
    r"xai-[A-Za-z0-9]{30,}",             # xAI (Grok)
    r"gAAAA[A-Za-z0-9_=-]{20,}",         # Fernet / Codex encrypted tokens
]

# (?<!...) / (?!...) verhindern Treffer mitten in laengeren Identifiern.
PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

# Bekannte Praefix-Substrings -- billiger Pre-Check vor dem teuren PREFIX_RE.
# Form-Beschreibung, keine Werte.
_PREFIX_SUBSTRINGS = (
    "sk-", "sk_", "rk_live_", "ghp_", "github_pat_", "gho_", "ghu_", "ghs_",
    "ghr_", "xox", "AIza", "AKIA", "SG.", "hf_", "r8_", "npm_", "pypi-",
    "gsk_", "pplx-", "tvly-", "xai-", "gAAAA",
)

# Authorization: Bearer <token>
# FIX (Idempotenz, Befund 7): '[' / ']' aus der Wert-Erfassung ausgeschlossen,
# damit ein bereits eingesetzter [REDACTED:...]-Tag bei einem Re-Run NICHT als
# Token re-erfasst wird (sonst re-feuert die Lane + desynct offsets/counts).
AUTH_HEADER_RE = re.compile(r"(Authorization:\s*Bearer\s+)([^\s\[\]]+)", re.IGNORECASE)
# Frei stehendes "Bearer <token>" (z.B. wenn ein Sprecher es diktiert)
BEARER_TOKEN_RE = re.compile(r"\bBearer\s+([A-Za-z0-9._~+/=-]{16,})", re.IGNORECASE)

# Passwort-/Secret-Schluesselwort gefolgt von einem Wert.
# Deckt DE + EN: "passwort ist X", "password: X", "API key = X", "PIN 1234".
# FIX (Befund 9): bare 'schluessel'/'schlüssel' + 'passcode'/'zugangscode' als
#   eigene Alternativen ergaenzt (vorher nur 'api_schluessel'); 'Mein Schluessel
#   ist X' leakte komplett.
# FIX (Befund 7, Idempotenz): '[' und ']' aus der Wert-Zeichenklasse
#   ausgeschlossen, damit ein bereits eingesetzter [REDACTED:...]-Tag NIE als
#   Wert re-gewrapped werden kann (sonst waechst der Tag je Re-Run).
PASSWORD_KEYWORD_RE = re.compile(
    r"\b(passwor[dt]|kennwort|secret|api[\s_-]?key|api[\s_-]?schluessel|"
    r"schl(?:ue|ü)ssel|passcode|zugangscode|"
    r"client[\s_-]?secret|token|pin|tan|access[\s_-]?token|refresh[\s_-]?token)\b"
    r"\s*(?:ist|lautet|=|:|->| )\s*"
    r"([^\s,.;:!?\[\]]{4,})",
    re.IGNORECASE,
)

# KEY=value mit secret-aehnlichem KEY-Namen (ENV-Stil).
# FIX (Befund 7, Idempotenz + run1-Doppelzaehlung): Wert-Klasse schliesst
#   '[' / ']' aus. Sonst re-erfasst die ENV-Lane einen bereits von PREFIX_RE
#   gesetzten [REDACTED:SECRET]-Tag ('OPENAI_API_KEY=[REDACTED:SECRET]') ->
#   doppelter SECRET-Count im SELBEN Lauf (counts=2, offsets=1, desync) UND
#   re-feuert beim Re-Run. Der Wert kann nie auf einem Tag ('[') beginnen.
ENV_ASSIGN_RE = re.compile(
    r"\b([A-Z0-9_]{0,40}(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|"
    r"CREDENTIAL|AUTH)[A-Z0-9_]{0,40})\s*=\s*(['\"]?)([^\s\[\]]+)\2"
)

# JWT (eyJ...header[.payload[.signature]])
JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}")

# Private-Key-Bloecke
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# DB-Connection-String mit Passwort: scheme://user:PASS@host
# FIX (Befund 7, Idempotenz): Passwort-Erfassung schliesst '[' / ']' aus, damit
#   ein bereits eingesetzter Tag bei einem Re-Run nicht als Passwort re-erfasst
#   wird.
DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:@/\s]+:)"
    r"([^@/\s\[\]]+)(@)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# PII-Klassen (die in agent/redact.py FEHLENDE Abdeckung)
# ---------------------------------------------------------------------------

# E.164-Telefon -- FIX (Befund 4): interne Trennzeichen (Space/Dash/Dot/
#   Klammern) zwischen den Ziffern-Gruppen erlaubt; die genaue Ziffern-Zahl
#   (7..15, E.164) wird im Code (_e164_ok) geprueft, spiegelt die CC-Lane.
#   Kandidaten-Regex ist linear (kein Inner-Backtracking). Beginnt mit '+',
#   endet auf einer Ziffer; '+49' allein (2 Ziffern) faellt im Code raus.
E164_PHONE_RE = re.compile(r"(?<![\w.])(\+\d[\d /.()-]{5,17}\d)(?![\w])")

# Nationale DE/CH-Telefonnummer (heuristisch, separat schaltbar wg. FP-Risiko):
# 0xx... oder 00xx..., 7..14 Ziffern, optionale Trenner.
NATIONAL_PHONE_RE = re.compile(
    r"(?<![\d+])(0\d[\d /.()-]{6,14}\d)(?![\d])"
)

# E-Mail -- FIX (Befund 5, ReDoS): local-part auf {1,64} (RFC 5321) begrenzt und
#   Domain als verankerte DNS-Labels. Das eliminiert das quadratische
#   Backtracking des alten `[A-Za-z0-9.-]+\.` und der unbegrenzten local-part-
#   Wiederholung. Linear: ~10ms @ 64KB, ~17ms @ 128KB adversarial (alt: ~3.8s
#   @ 64KB). IDN/punycode (xn--...) ist abgedeckt.
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]{1,64}@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}\b"
)

# IBAN-Kandidat: 2 Buchstaben Laendercode + 2 Pruefziffern + BBAN.
# Zwei Formen, damit ein einzelner Trenner die Gruppe nicht in das Folgewort
# weiterlaufen laesst (sonst frisst die Gruppe " ueberweisen"):
#   1. kontiguierlich: DE89370400440532013000  (11..30 alnum, kein Trenner)
#   2. Gruppen:        DE89 3704 0044 ...  /  DE89.3704.0044...  /  DE89-3704-...
# mod-97-Validierung erfolgt im Code (schneidet Format-FP hart).
# FIX (Befund 1): re.IGNORECASE, damit lowercase-IBANs ('de89...') nicht
#   komplett bypassed werden; mod-97 (_iban_ok) uppercased ohnehin -> Praezision
#   bleibt (lowercase-mit-falscher-Pruefziffer wird weiterhin NICHT redigiert).
# FIX (Befund 2): Separator-Klasse Form 2 von '[ ]' auf '[ .\-]' erweitert, damit
#   dotted/dashed IBANs (DE89.3704... / DE89-3704...) erfasst werden. Weil
#   _pass_iban VOR _pass_phone_national laeuft, verschwindet damit zugleich der
#   Teil-Leak + die PHONE-Fehlklassifikation des IBAN-Fragments.
IBAN_CANDIDATE_RE = re.compile(
    r"\b("
    r"[A-Z]{2}\d{2}"
    r"(?:"
    r"[A-Za-z0-9]{11,30}"                       # Form 1: kontiguierlich
    r"|"
    r"(?:[ .\-][A-Za-z0-9]{1,4}){2,8}"          # Form 2: Space/Dot/Dash-Gruppen
    r")"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Kreditkarten-Kandidat: 13..19 Ziffern, optional in 4er-Gruppen mit
# Space/Dash/Dot. Luhn-Validierung erfolgt im Code.
# FIX (Befund 3): Separator-Klasse '[ .\-]' (Punkt ergaenzt) + Lookbehind/-ahead
#   von '(?<![\d.])'/'(?![\d.])' auf '(?<!\d)'/'(?!\d)' geaendert. Der '.' muss
#   in der Nachbarschaft erlaubt sein, sonst greift '4111.1111.1111.1111' nicht.
#   IPs wie '192.168.0.1' fallen weiter durch Luhn raus.
CREDIT_CARD_CANDIDATE_RE = re.compile(
    r"(?<!\d)(\d(?:[ .\-]?\d){12,18})(?!\d)"
)

# ---------------------------------------------------------------------------
# Health / Art.9 (besondere Kategorien) -- kuratiertes DE-Lexikon.
# WICHTIG: separat schaltbar, hoechstes Over-Redaction-Risiko (siehe Spec e).
# Dies adressiert Art.9-INHALTSBEGRIFFE im Transkript, NICHT Sprecher-ID/
# Wake-Word (jene bleiben per Threat-Model bis Consent + DPIA komplett AUS).
# Bewusst KONSERVATIV/kurz gehalten: lieber wenige eindeutige Begriffe als
# breite Alltagswoerter ("Druck", "Stress"), die alles ueber-redigieren.
# ---------------------------------------------------------------------------
_HEALTH_TERMS = [
    # Diagnosen / Zustaende (eindeutig medizinisch)
    "diagnose", "diagnostiziert", "krebs", "tumor", "karzinom", "metastase",
    "hiv", "aids", "diabetes", "depression", "depressiv", "schizophren",
    "schizophrenie", "epilepsie", "epileptisch", "demenz", "alzheimer",
    "parkinson", "multiple sklerose", "hepatitis", "tuberkulose",
    "schwangerschaft", "schwanger", "abtreibung", "fehlgeburt",
    "psychotherapie", "psychiatrie", "psychiatrisch", "suizid", "suizidal",
    "chemotherapie", "bestrahlung", "dialyse", "transplantation",
    # Ergaenzte eindeutige Diagnosen/Zustaende (Art.9-Abdeckung; safe-direction
    # = Over-Redaction, FP-clean gegen Alltagstext verifiziert).
    "herzinfarkt", "schlaganfall", "asthma", "rheuma", "burnout",
    "bandscheibenvorfall", "querschnittslaehmung", "blindheit", "taubheit",
    # Versorgungs-/Verwaltungs-Kontext, der Gesundheitsdaten markiert
    "krankschreibung", "arbeitsunfaehig", "au-bescheinigung", "befund",
    "rezept", "medikament", "verschrieben", "behandlung", "operation",
    # Andere Art.9-Sonderkategorien (Religion, Gewerkschaft, Sexualleben,
    # politische/weltanschauliche Ueberzeugung, ethnische Herkunft) -- nur
    # eindeutige Marker, bewusst minimal.
    "gewerkschaft", "gewerkschaftsmitglied",
    # Religion / Weltanschauung (eindeutige Bekenntnis-Marker)
    "katholisch", "evangelisch", "protestantisch", "muslimisch", "moslem",
    "muslim", "juedisch", "buddhistisch", "hinduistisch", "atheist",
    "konfessionslos",
    # Sexualorientierung (eindeutige Marker)
    "homosexuell", "heterosexuell", "bisexuell", "transgender", "intersexuell",
    "lesbisch", "schwul",
    # Ethnische Herkunft (eindeutige Marker; multi-word toleriert " "->"[ -]?")
    "kurdischer herkunft", "tuerkischer herkunft", "afrikanischer herkunft",
    "arabischer herkunft", "asiatischer herkunft", "ethnische herkunft",
]
# \b-umrandet, case-insensitiv; multi-word Terme erlauben optionalen Bindestrich.
HEALTH_TERM_RE = re.compile(
    r"\b(" + "|".join(re.escape(t).replace(r"\ ", r"[ -]?") for t in _HEALTH_TERMS) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Adressen (OPTIONAL, default AUS -- hoechstes FP-Risiko, siehe Spec b/e).
# Heuristik: Strassenname + Hausnummer ODER deutsche PLZ + Ort.
# ---------------------------------------------------------------------------
# FIX (Befund 6, ReDoS): der geschachtelte Quantor
#   `[a-zäöüß]+(?:[ -][A-ZÄÖÜ][a-zäöüß]+)*` machte STREET_RE quadratisch
#   (~1.6s @ 16KB, 4x je Verdopplung). Possessive/atomic auf der Wortstruktur
#   bricht die Semantik (frisst die Suffix-Buchstaben mit). Korrekter Rewrite:
#   ein lazy, GEKAPPTER Single-Char-Body `{0,40}?` (kein geschachtelter Quantor
#   -> beschraenktes Backtracking-Fenster -> linear; 256KB adversarial = ~135ms).
#   Semantisch >= Original: faengt zusaetzlich hyphenierte/mehrwortige Strassen
#   ('Heinrich-Heine-Allee', 'Berliner Strasse', 'Lange Gasse', 'Unter den
#   Linden Allee'), die das alte Muster verfehlte; 0 neue False Positives.
STREET_RE = re.compile(
    r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß .\-]{0,40}?"
    r"(?:strasse|straße|str\.|gasse|weg|allee|platz|ring|damm))"
    r"\s+(\d{1,4}[a-z]?)\b",
    re.IGNORECASE,
)
GERMAN_POSTCODE_CITY_RE = re.compile(
    r"\b(\d{5})\s+([A-ZÄÖÜ][a-zäöüß.-]+(?:[ ][A-ZÄÖÜ][a-zäöüß.-]+){0,3})\b"
)
