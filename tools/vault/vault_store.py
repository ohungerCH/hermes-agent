"""VaultStore -- der EINE Schreiber des Personal-Context-Vault (INV-3, Stufe 5).

Load-bearing Spine: JEDER durable Vault-Write läuft durch VaultStore.write(). Es gibt
keinen zweiten, gate-losen Pfad in memory_items/object_metadata. Der Store sitzt beim
Schreib-Gate (tools/write_approval.py) und verbindet die bereits gebauten + bewiesenen
Nähte:

  * Gate (GAP-C): vault_scan -> vault_gate_posture (Kritikalität, nicht Trust; fail-closed)
  * Sanitizer (GAP-D): via vault_scan (normalize + classify_threats, EN+DE)
  * RLS-Konsument (Stufe 3): vault_transaction (composite anchor tenant_id AND owner_id)
  * Krypto (OD-3): ObjectStoreCrypto.encrypt (per-owner Envelope, Crypto-Shred-fähig)
  * Tabelle (Stufe 5): migrations/0001_memory_items.sql (memory_items + object_metadata)

INTEGRATIONSGRENZE (no-false-green): DIESES Modul ist gebaut + getestet, aber NICHT an
einen Live-Aufrufer verdrahtet und NICHT deployt. embedding ist beim Write IMMER NULL
(candidate/pre-embed; der Vektor kommt aus einem separaten Reindex-Lauf des bge-m3-Servers,
der Embed-Gate hält). Offene Write-Path-Fragen (source_id-Factory, Background-Promote-Naht,
Feature-Flags) sind in VAULTSTORE_WRITE_PATH_SPEC.md §5 benannt und bleiben offen.

TRUSTED-SQL-ONLY (Bedrohungsmodell, s. vault_context.py): die App-Rolle darf AUSSCHLIESSLICH
getrustetes, parametrisiertes SQL mit server-gesetztem Kontext ausführen. Alle Werte hier
gehen als Query-Parameter (%s), NIE interpoliert. Der Store schreibt KEIN vom Modell verfasstes
SQL.

DB-TREIBER absichtlich NICHT importiert: write() nimmt eine duck-typed Connection (wie
vault_context.py) -> testbar ohne psycopg/Netz, drop-in für eine gepoolte psycopg-Connection
in Stufe 5. Die Krypto-Instanz + der Object-Sink werden injiziert (kein Keystore-Pfad im Modul).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from tools.write_approval import (
    VAULT_CANDIDATE,
    GateDecision,
    vault_gate_posture,
    vault_scan,
)
from tools.vault.vault_context import (
    VaultContextError,
    normalize_context_value,
    vault_transaction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vokabular-Registry (Write-Path-SSOT, VAULTSTORE_WRITE_PATH_SPEC.md §4)
# ---------------------------------------------------------------------------
# Die DDL liess source / trust_level / retention_class bewusst als Freitext (ein
# geratenes DB-CHECK wäre fail-WRONG und bräche legitime künftige Writes). Der
# VaultStore ist der Ort, wo das Vokabular VOR dem ersten Insert gepinnt wird -- ein
# App-seitiges fail-closed, KEIN Rival-Enum in der DB. Werte ausserhalb -> Ablehnung
# VOR jedem Gate-/DB-Kontakt.

SOURCE_FOREGROUND_OWNER = "foreground_owner"   # Owner hat es im Vordergrund selbst kuratiert
SOURCE_AUTO_CAPTURE = "auto_capture"           # Hintergrund-Retrieval-abgeleitet (retrieval_derived-Träger)
SOURCE_INGEST = "ingest"                        # Inbound-Konnektor (M365 etc.) -- untrusted-by-default
SOURCE_SKILL = "skill"                          # Modell-vorgeschlagen (memory.propose-Klasse)
VALID_SOURCES = frozenset({
    SOURCE_FOREGROUND_OWNER, SOURCE_AUTO_CAPTURE, SOURCE_INGEST, SOURCE_SKILL,
})

TRUST_UNTRUSTED = "untrusted"
TRUST_TRUSTED = "trusted"
VALID_TRUST_LEVELS = frozenset({TRUST_UNTRUSTED, TRUST_TRUSTED})

# retention_class: SSOT ist die privacy_lifecycle_matrix (ops/services/vault-db/
# privacy_lifecycle_matrix.yaml). Hier gespiegelt als das erlaubte Vokabular (meaning-Ebene
# + raw-Ebenen-Ceilings), NICHT als geratenes neues Enum. Die per-Domäne-Validierung
# (welches Ceiling für welche source_table gilt) ist §5-offen; hier nur die Wert-Zugehörigkeit.
RETENTION_PERMANENT_MEANING = "permanent_meaning"
VALID_RETENTION_CLASSES = frozenset({
    RETENTION_PERMANENT_MEANING, "short", "medium", "long", "manual_keep",
})

# sensitivity: ADR-0041-Kanon, deckungsgleich mit mem_sensitivity_ck in der DDL. Das
# DB-CHECK bleibt der letzte Wächter; die App-Validierung liefert nur den früheren,
# klaren Fehler (KEIN Rival-Enum -- exakt dieselbe Menge).
VALID_SENSITIVITY = frozenset({
    "public_project", "personal_low", "personal_sensitive", "personal_high", "secret",
})

# lifecycle_status-Kanon (ADR-0041:605; deckungsgleich mem_lifecycle_ck).
LIFECYCLE_CANDIDATE = "candidate"    # NICHT recall-fähig bis Owner-Confirm
LIFECYCLE_CONFIRMED = "confirmed"    # recall-fähig (Recall surft nur confirmed, ADR-0044:207)

# Invalidierungs-Modi (§5b Edit/Delete-Propagation, WIRING_PLAN §5b). Ein foreground_owner-Write
# leitet CONFIRMED ab -> ein Edit/Delete heilt sich NICHT durch Unrecall-Bleiben. Damit Vault +
# MEMORY.md nicht divergieren, invalidiert der EINE Schreiber die alte Zeile über ihren Natural-Key:
#   delete    -> deleted_at gesetzt (Owner löschte den Eintrag; Soft-Delete)
#   supersede -> superseded_at + reindex_state='superseded' (Owner ersetzte "X"->"Y"; die alte Zeile
#                wird abgelöst, die neue via write() separat eingefügt).
# Beide Zustände sind vom Recall-Filter ausgeschlossen (s. _derive_provenance-Kontrakt). KEIN
# superseded_by-Link (Provenienz-Kür, nicht recall-korrektheits-nötig -- Advisor 2026-07-09).
INVALIDATE_DELETE = "delete"
INVALIDATE_SUPERSEDE = "supersede"
VALID_INVALIDATE_MODES = frozenset({INVALIDATE_DELETE, INVALIDATE_SUPERSEDE})

# Traversierungs-Ziel der Roh-Schicht: eine memory_items-Zeile mit raw_bytes verweist via
# source_table='object_metadata' + source_id (== object_key) auf ihre object_metadata-Zeile
# (kohärente Naht, Codex-Cross-Review 2026-07-08). Ein raw-Write mit anderem source_table
# erzeugte einen Waisen (Objekt ohne traversierbare Meaning-Zeile) -> fail-closed (write()).
SOURCE_TABLE_OBJECT = "object_metadata"

# Embedding-Achse: bge-m3-Pins (OD-1, 2026-07-07). embedding selbst bleibt beim Write NULL.
EMBED_PROVIDER_DEFAULT = "local-bge-m3"
EMBED_MODEL_DEFAULT = "BAAI/bge-m3"
EMBED_VERSION_DEFAULT = "v1"          # echte Version aus HF-revision-Bytes -> §deferred
EMBED_DIMENSIONS_DEFAULT = 1024       # MUSS = vector(1024) (mem_embdim_ck)


class VaultStoreError(RuntimeError):
    """Fail-closed: ungültige Anfrage / inkonsistente Provenienz / Persistenz-Fehler.
    Trägt NIE Roh-Inhalt oder einen Key-/Klartext-Wert (DLP)."""


# ---------------------------------------------------------------------------
# Ergebnis
# ---------------------------------------------------------------------------

# Status-Vokabular des Write-Ergebnisses:
STATUS_WRITTEN = "written"     # allow -> confirmed-Zeile committet (recall-fähig)
STATUS_STAGED = "staged"       # stage -> candidate-Zeile committet (durabel, NICHT recall-fähig)
STATUS_BLOCKED = "blocked"     # Gate-Ablehnung (Vordergrund-Injektion) -- owner-facing message
STATUS_DROPPED = "dropped"     # Gate-Drop (Hintergrund-Injektion) -- Audit im Gate gefeuert
STATUS_REFUSED = "refused"     # VaultStore-fail-closed (z.B. special-category ohne OD-11-Naht)
STATUS_ERROR = "error"         # Persistenz fehlgeschlagen -> NIE als Erfolg melden (never-lost)
STATUS_INVALIDATED = "invalidated"  # invalidate() committet (deleted_at/superseded_at gesetzt bzw. 0-Zeilen-No-op)
STATUS_RECALLED = "recalled"        # recall() lief sauber, >=1 Treffer
STATUS_RECALL_EMPTY = "recall_empty"  # recall() lief sauber, 0 Treffer (KEIN Fehler, NICHT "kein Gedächtnis")

# Recall-Grenzen: bounded, damit ein breiter tsvector-Treffer den Turn-Kontext nicht flutet.
RECALL_LIMIT_DEFAULT = 8
RECALL_LIMIT_MAX = 25


@dataclass
class WriteResult:
    """Ergebnis eines write()-Versuchs.

    ``persisted`` ist der load-bearing never-lost-Anker (ADR-0044:228-241, DB-Zwilling des
    stage_write-_persisted-Flags): True NUR nach erfolgreichem commit(). Ein Aufrufer darf
    Erfolg NUR bei ``persisted`` melden. blocked/dropped/refused sind bewusste Nicht-Writes
    (kein Fehler); ``error`` ist ein still-verlorener Write -> sichtbarer Fehler, nie Erfolg.
    """
    status: str
    persisted: bool = False
    message: str = ""
    lifecycle_status: str = ""
    memory_item_written: bool = False
    object_metadata_written: bool = False


# ---------------------------------------------------------------------------
# Anfrage
# ---------------------------------------------------------------------------

@dataclass
class MemoryWrite:
    """Eine durable Vault-Schreibanfrage.

    ``content`` ist der Text, den das Gate scannt (Injektions-/Kritikalitäts-Prüfung).
    ``summary_redacted`` ist die DLP-redigierte, recall-/tsvector-fähige Zusammenfassung
    (plaintext, Owner-Entscheid 2026-07-07). ``source_id`` ist die STABILE, reindex-invariante
    Identität (Natural-Key-Teil); bei ``raw_bytes`` wird sie zugleich zum object_key
    (kohärente Traversierung object_metadata.object_key == memory_items.source_id).

    KONTRAKT (Scan-vs-Recall-Fläche, load-bearing): das Gate scannt ``content``; durable
    gespeichert + über tsvector('german') recall-fähig ist ``summary_redacted``. Der
    Injektions-Scan schützt die Recall-Fläche NUR, wenn ``summary_redacted`` eine
    Redaktions-Teilmenge des gescannten ``content`` ist (kein Text im Summary, der nicht im
    gescannten content stand). Aktuell ANGENOMMEN, nicht erzwungen (Recall ist deferred,
    nichts liest den tsvector) -> Enforcement/Scan von summary_redacted = §5-offen.
    """
    content: str
    owner_id: str
    tenant_id: str
    origin: str                 # "foreground" | "background" | "background_review"
    source: str                 # VALID_SOURCES
    source_table: str
    source_id: str
    source_hash: str
    sensitivity: str            # VALID_SENSITIVITY
    trust_level: str            # VALID_TRUST_LEVELS
    retention_class: str        # VALID_RETENTION_CLASSES

    summary_redacted: Optional[str] = None
    redaction_state: str = "pending"      # Caller-DLP-Zustand; 'applied' erst nach Redaktion
    redaction_version: str = "v0"
    device_id: Optional[str] = None       # NULLABLE (Timeline-Artefakt, nicht die Identität)
    local_id: Optional[str] = None        # NULLABLE

    taint: Dict[str, Any] = field(default_factory=dict)
    raw_bytes: Optional[bytes] = None     # Roh-Schicht (nur wenn zu sichern)

    embedding_provider: str = EMBED_PROVIDER_DEFAULT
    embedding_model: str = EMBED_MODEL_DEFAULT
    embedding_version: str = EMBED_VERSION_DEFAULT
    embedding_dimensions: int = EMBED_DIMENSIONS_DEFAULT


@dataclass
class MemoryInvalidate:
    """Eine Invalidierungs-Anfrage (§5b): markiert EINE bestehende memory_items-Zeile über ihren
    Natural-Key als gelöscht bzw. abgelöst. Trägt KEINEN Content (kein Injektions-Scan/Gate nötig --
    es wird nichts Neues geschrieben, nur ein Lifecycle-Zustand auf einer schon owner-eigenen Zeile
    gesetzt; die RLS-Isolation trägt die Sicherheit). ``source_id`` ist die STABILE Identität, mit
    der die Zeile geschrieben wurde (bei owner_memory Phase-1 = Content-Hash des vollen Eintrags)."""
    owner_id: str
    tenant_id: str
    source_table: str
    source_id: str
    mode: str                   # VALID_INVALIDATE_MODES


@dataclass
class MemoryRecall:
    """Eine Recall-/Lese-Anfrage (Stufe 6, tsvector-Fläche). Trägt KEINEN neuen Content -- sie liest
    nur owner-eigene, confirmed Zeilen (RLS + WHERE-Kontrakt). ``tenant_id``/``owner_id`` kommen
    server-autoritativ vom Aufrufer (ContextVar-Identität, NIE client-geliefert). ``query`` ist der
    Volltext-Suchbegriff (websearch_to_tsquery, deutsch)."""
    owner_id: str
    tenant_id: str
    query: str
    limit: int = RECALL_LIMIT_DEFAULT


@dataclass
class RecallItem:
    """Eine zurückgeholte Zeile. ``summary`` ist der ROHE summary_redacted-Plaintext -- das
    untrusted-Wrapping passiert am Modell-Rand (vault_wiring._wrap_recalled), NICHT hier (Trennung
    Datenzugriff vs. Präsentation; der Store liefert Daten, die Naht neutralisiert sie)."""
    source_table: str
    source_id: str
    summary: str
    created_at: Any
    sensitivity: str
    from_untrusted_inbound: bool


@dataclass
class RecallResult:
    """Ergebnis eines recall()-Versuchs. ``available`` ist load-bearing für die Ehrlichkeits-Klausel:
    True = der Lesepfad lief (auch bei 0 Treffern -> begründete Abwesenheit); False = Fehler / nicht
    erreichbar (KEIN Rückschluss auf An-/Abwesenheit möglich)."""
    status: str
    items: list = field(default_factory=list)
    available: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# SQL (parametrisiert, Spalten-SSOT)
# ---------------------------------------------------------------------------
# Spalten-Reihenfolge = die im migrations/0001_memory_items.sql + memory_items_seed.sql
# bewiesene Menge. Ausgelassen (Default/GENERATED): id, created_at, sensitivity_rank,
# deleted_at, quarantined_at, embedding_job_id, reindex_state, superseded_at, embedding
# (embedding bleibt NULL = candidate/pre-embed).
_MEMORY_ITEMS_COLUMNS = (
    "tenant_id", "owner_id", "device_id", "local_id",
    "source", "sensitivity", "trust_level", "retention_class",
    "source_table", "source_id", "source_hash",
    "redaction_state", "redaction_version", "sanitization_state",
    "embedding_provider", "embedding_model", "embedding_version", "embedding_dimensions",
    "summary_redacted", "from_untrusted_inbound", "lifecycle_status",
)

# Upsert (Poisoning-Guard, spec §3.5): ALLE DREI Invalidierungs-Spalten (lifecycle_status,
# embedding, reindex_state) folgen EINER kohärenten Bedingung -- der "keep"-Bedingung: gleicher
# Hash UND die neuen Redaktions-/Sanitisierungs-Zustände weiter beide 'applied'. NUR dann bleibt
# die Zeile unverändert recall-fähig (idempotenter Re-Write). Sonst (Hash geändert = Content-
# Änderung ODER State-Downgrade) wird die Zeile invalidiert: lifecycle -> candidate (Re-Confirm),
# embedding -> NULL, reindex -> stale.
#
# WARUM lifecycle an dieselbe Bedingung wie embedding gekoppelt ist (Review-Befund HIGH, sql-Linse
# 2026-07-08): wäre lifecycle NUR an die Hash-Änderung gekoppelt, könnte ein same-hash-Re-Write
# mit State-Downgrade (z.B. sanitization 'applied'->'pending', Scanner transient tot) den Vektor
# nullen, aber lifecycle='confirmed' stehen lassen -> eine confirmed-Zeile OHNE Vektor. Das verletzt
# die Invariante "confirmed => embed-eligible/sanitisiert" (ADR-0044:207 Recall surft confirmed).
# Fail-safe-Richtung: nicht-mehr-sanitisiert -> candidate (nicht recall-fähig bis Re-Confirm).
#
# Die Anker-Spalten (tenant/owner/source_table/source_id) + created_at werden NIE überschrieben.
_EMBED_KEEP_COND = (
    "public.memory_items.source_hash IS NOT DISTINCT FROM EXCLUDED.source_hash "
    "AND EXCLUDED.redaction_state = 'applied' AND EXCLUDED.sanitization_state = 'applied'"
)

# §5b Resurrection-Guard (Review-Befund 2026-07-10): ein Upsert-Konflikt auf einen zuvor
# invalidierten Natural-Key (Owner: add "X" -> remove "X" -> add "X") MUSS die Tombstones lösen,
# sonst behält der Vault die alte Löschung -> MEMORY.md hat "X", der Vault sagt gelöscht (Under-
# Recall = dieselbe §5b-Divergenz). ABER owner-only: nur ein foreground_owner-Re-Write darf
# wiederbeleben. Ein späterer Background-/ingest-Write (source != foreground_owner) auf einen
# owner-gelöschten Key darf NICHT resurrekten (sonst spült Auto-Capture Gelöschtes hoch). Der
# VaultStore ist INV-3 auch für jene künftigen Aufrufer -> Guard JETZT setzen, nicht später.
_OWNER_RESURRECT = "EXCLUDED.source = '" + SOURCE_FOREGROUND_OWNER + "'"
_MEMORY_ITEMS_INSERT = (
    "INSERT INTO public.memory_items (" + ", ".join(_MEMORY_ITEMS_COLUMNS) + ") "
    "VALUES (" + ", ".join(["%s"] * len(_MEMORY_ITEMS_COLUMNS)) + ") "
    "ON CONFLICT ON CONSTRAINT mem_natural_uq DO UPDATE SET "
    "  source = EXCLUDED.source,"
    "  sensitivity = EXCLUDED.sensitivity,"
    "  trust_level = EXCLUDED.trust_level,"
    "  retention_class = EXCLUDED.retention_class,"
    "  source_hash = EXCLUDED.source_hash,"
    "  redaction_state = EXCLUDED.redaction_state,"
    "  redaction_version = EXCLUDED.redaction_version,"
    "  sanitization_state = EXCLUDED.sanitization_state,"
    "  summary_redacted = EXCLUDED.summary_redacted,"
    "  from_untrusted_inbound = EXCLUDED.from_untrusted_inbound,"
    "  device_id = EXCLUDED.device_id,"
    "  local_id = EXCLUDED.local_id,"
    # §5b Resurrection (owner-only): ein foreground_owner-Re-Write auf einen invalidierten Key löst
    # die Tombstones (Owner belebt seinen eigenen Eintrag wieder); jede andere source lässt sie stehen.
    "  deleted_at = CASE WHEN " + _OWNER_RESURRECT +
    "    THEN NULL ELSE public.memory_items.deleted_at END,"
    "  superseded_at = CASE WHEN " + _OWNER_RESURRECT +
    "    THEN NULL ELSE public.memory_items.superseded_at END,"
    # keep-Fall behält den BESTEHENDEN lifecycle -> promoviert NIE candidate->confirmed (bewusst:
    # Promotion ist der separate Confirm-Flow, nicht ein Re-Write). Nur Abwärts (-> candidate).
    "  lifecycle_status = CASE WHEN " + _EMBED_KEEP_COND +
    "    THEN public.memory_items.lifecycle_status ELSE 'candidate' END,"
    "  embedding = CASE WHEN " + _EMBED_KEEP_COND +
    "    THEN public.memory_items.embedding ELSE NULL END,"
    "  reindex_state = CASE WHEN " + _EMBED_KEEP_COND +
    "    THEN public.memory_items.reindex_state ELSE 'stale' END"
)

_OBJECT_METADATA_COLUMNS = (
    "tenant_id", "owner_id", "source", "trust_level", "object_key", "key_ref",
)
# Idempotent auf (tenant, owner, object_key): der Ciphertext wurde am Sink unter demselben
# object_key abgelegt; die Metadaten-Zeile ändert sich nicht (key_ref = per-owner-stabil).
_OBJECT_METADATA_INSERT = (
    "INSERT INTO public.object_metadata (" + ", ".join(_OBJECT_METADATA_COLUMNS) + ") "
    "VALUES (" + ", ".join(["%s"] * len(_OBJECT_METADATA_COLUMNS)) + ") "
    "ON CONFLICT ON CONSTRAINT objmeta_key_uq DO NOTHING"
)

# §5b Invalidierung -- zielgerichteter UPDATE über den Natural-Key (kein Insert, kein Upsert:
# ein Upsert könnte bei Cold-Start eine Geister-Tombstone-Zeile INSERTen; ein UPDATE mit 0
# betroffenen Zeilen ist der saubere No-op). Läuft unter der Composite-Anker-Policy (FOR ALL,
# USING+WITH CHECK): USING filtert auf owner-eigene Zeilen -> cross-owner-Invalidierung unmöglich;
# WITH CHECK verhindert Owner-Verschiebung (tenant/owner unverändert). Die WHERE-Zusatzklausel
# (deleted_at/superseded_at IS NULL) macht Re-Invalidierung idempotent (0 Zeilen).
_MEMORY_ITEMS_DELETE = (
    "UPDATE public.memory_items SET deleted_at = now() "
    "WHERE tenant_id = %s AND owner_id = %s AND source_table = %s AND source_id = %s "
    "AND deleted_at IS NULL"
)
_MEMORY_ITEMS_SUPERSEDE = (
    "UPDATE public.memory_items SET superseded_at = now(), reindex_state = 'superseded' "
    "WHERE tenant_id = %s AND owner_id = %s AND source_table = %s AND source_id = %s "
    "AND superseded_at IS NULL AND deleted_at IS NULL"
)

# Recall (Stufe 6, tsvector-Fläche). Der WHERE-Kontrakt ist die verlustfreie INVERSE des
# _derive_provenance-Schreibpfads: confirmed UND nicht gelöscht/quarantänet/abgelöst. tenant/owner
# stehen NICHT im Query-Text -- die Composite-Anker-Policy (RLS-GUCs, via vault_transaction) filtert
# sie als ECHTER Pre-Filter (kein ANN-Index -> GAP-B). websearch_to_tsquery als FROM-Item `q` (EIN
# Aufruf, wiederverwendet in @@ + ts_rank -> ein einziger Query-Parameter). coalesce(...) MATCHT die
# GIN-Index-Expression (memory_items_summary_fts) byte-genau, damit der Index nutzbar bleibt (kein
# toter Index). from_untrusted_inbound wird PROJIZIERT (retrieval_derived-Marker am Modell-Rand),
# NICHT gefiltert; sanitization_state wird NICHT gefiltert -- eine scanner-tote Zeile ist ohnehin
# candidate (nicht confirmed), und ein Filter darauf würde Owners legitimes Security-Vokabular
# kastrieren. Die Verteidigung gegen zitierte Injektion im Treffer ist der untrusted-Wrap am Rand,
# NICHT der Ausschluss (Advisor 2026-07-10, #75 warn-vs-block).
_MEMORY_ITEMS_RECALL = (
    "SELECT source_table, source_id, summary_redacted, created_at, sensitivity, from_untrusted_inbound "
    "FROM public.memory_items, websearch_to_tsquery('german', %s) AS q "
    "WHERE lifecycle_status = 'confirmed' "
    "AND deleted_at IS NULL AND quarantined_at IS NULL AND superseded_at IS NULL "
    "AND to_tsvector('german', coalesce(summary_redacted, '')) @@ q "
    "ORDER BY ts_rank(to_tsvector('german', coalesce(summary_redacted, '')), q) DESC, created_at DESC "
    "LIMIT %s"
)


# ---------------------------------------------------------------------------
# Taint-Helfer (deckungsgleich mit der Gate-Logik in write_approval.py)
# ---------------------------------------------------------------------------

def _explicitly_trusted(taint: Dict[str, Any]) -> bool:
    """True NUR bei einem expliziten False-artigen from_untrusted_inbound (untrusted-by-default,
    ADR-0042:35). Spiegelt write_approval._vault_untrusted_capture invertiert."""
    v = (taint or {}).get("from_untrusted_inbound", True)
    if v is False:
        return True
    if isinstance(v, str) and v.strip().lower() in {"false", "no", "0", "off", "disabled"}:
        return True
    return False


def _is_special_category(taint: Dict[str, Any]) -> bool:
    """Health/biometrisch = besondere Kategorie (Art. 9). Deckungsgleich mit
    write_approval._vault_is_special_category."""
    taint = taint or {}
    v = taint.get("special_category")
    if v is True:
        return True
    if isinstance(v, str) and v.strip().lower() in {"on", "true", "yes", "1", "enabled"}:
        return True
    return str(taint.get("sensitivity", "")).strip().lower() == "special_category"


_MSG_SPECIAL_CATEGORY = (
    "Nicht gespeichert: besondere Kategorie personenbezogener Daten (Gesundheit/Biometrie). "
    "Durable Ablage braucht die bewusste Opt-in-Freischaltung (noch nicht verfügbar)."
)


# ---------------------------------------------------------------------------
# VaultStore
# ---------------------------------------------------------------------------

# Typ eines Object-Sinks: legt den Ciphertext-Envelope durable ab (MinIO, später). Bekommt
# den opaken object_key + den Envelope-String; owner/tenant nur für Scoping/Logging. Wirft
# bei Fehlschlag (fail-closed -> kein Metadaten-Waise auf fehlendem Objekt).
ObjectSink = Callable[..., None]


class VaultStore:
    """INV-3 single-writer. Erzwingt das Gate bei JEDEM durable Write.

    Args:
        connect: zero-arg Callable -> duck-typed DB-API-Connection (cursor/commit/rollback).
            In Stufe 5 = pool.getconn (reset_on_return='rollback', POOL_RESET_ON_RETURN); im
            Test = Fake. Der Store committet/rollbackt, schliesst die Connection NICHT (Pool-Job).
        crypto: eine ObjectStoreCrypto-Instanz (per-owner Envelope). NUR für raw_bytes-Writes.
        object_sink: legt den Envelope durable ab. raw_bytes ohne Sink -> fail-closed.
    """

    def __init__(self, connect: Callable[[], Any], *,
                 crypto: Any = None, object_sink: Optional[ObjectSink] = None) -> None:
        self._connect = connect
        self._crypto = crypto
        self._object_sink = object_sink

    # -- öffentlich --------------------------------------------------------
    def write(self, req: MemoryWrite) -> WriteResult:
        """Der EINE Schreibpfad. Reihenfolge: validieren -> scannen -> Gate -> verzweigen ->
        (Krypto) -> RLS-Insert -> commit. Wirft NIE bei erwarteten Ablehnungen (blocked/drop/
        refused) -- die kommen als WriteResult zurück. Ein echter Persistenz-Fehler -> status
        error (persisted=False), NIE stiller Erfolg."""
        # (0a) Anker-Kontext fail-closed validieren (non-empty + Regex; Header-Safety). Wirft
        #      VaultContextError -> als Refuse zurückgeben (kein Insert mit halbem Anker).
        try:
            tenant = normalize_context_value(req.tenant_id, "tenant_id")
            owner = normalize_context_value(req.owner_id, "owner_id")
        except VaultContextError as e:
            return WriteResult(status=STATUS_REFUSED, persisted=False, message=str(e))

        # (0b) Vokabular fail-closed VOR Gate/DB (kein Rival-Enum, aber App-seitiger Pin).
        vocab_err = self._validate_vocab(req)
        if vocab_err:
            return WriteResult(status=STATUS_REFUSED, persisted=False, message=vocab_err)

        # (0c) special-category (Art. 9) fail-closed: die DDL hat KEINE consent_ref/dsfa_ref/
        #      special_category-Spalten -> durable Vektor-Persistenz ist ohne die OD-11-Opt-in-Naht
        #      verboten (ADR-0041:674 "fehlt eines davon, ist der Schreibpfad fail-closed und
        #      persistiert nicht"). Der Gate würde STAGE liefern; der Store geht weiter (Refuse),
        #      weil die sichere-Ablage-Spalten fehlen. Kein candidate-Row für special-category.
        if _is_special_category(req.taint):
            return WriteResult(status=STATUS_REFUSED, persisted=False, message=_MSG_SPECIAL_CATEGORY)

        # (1) Scan -- vault_scan normalisiert INTERN (normalize + classify_threats, strict) und
        #     meldet die Scanner-Gesundheit atomar. NICHT doppelt normalisieren.
        block_ids, warn_ids, scanner_ok = vault_scan(req.content or "")

        # sanitization_state ist die EIGENE Aussage des Stores: der Sanitizer (vault_scan) lief
        # hier. Scanner gesund -> 'applied'; Scanner tot -> 'pending' (der Vektor bleibt so ohnehin
        # NULL, Embed-Gate). redaction_state kommt vom Caller (dessen DLP). Einmal berechnet aus dem
        # bereits vorliegenden scanner_ok -- KEIN zweiter Scan.
        sanitization_state = "applied" if scanner_ok else "pending"

        # (2) Gate -- Kritikalität, nie Trust; liest keine Config; verzweigt origin-konditional.
        d = vault_gate_posture(
            VAULT_CANDIDATE, origin=req.origin, taint=req.taint or {},
            block_ids=block_ids, warn_ids=warn_ids, scanner_ok=scanner_ok,
        )

        # (3) Nicht-schreibende Zweige.
        if d.blocked:
            return WriteResult(status=STATUS_BLOCKED, persisted=False, message=d.message)
        if d.drop:
            # Content-freies Audit ist bereits IM Gate gefeuert (vault_audit_drop) -- nicht erneut.
            return WriteResult(status=STATUS_DROPPED, persisted=False)

        # (4) Durable Provenienz ableiten (lifecycle_status, resolved source, from_untrusted_inbound).
        try:
            lifecycle, resolved_source, untrusted = self._derive_provenance(d, req)
        except VaultStoreError as e:
            return WriteResult(status=STATUS_REFUSED, persisted=False, message=str(e))

        # (5) Persistieren.
        return self._persist(req, tenant, owner, lifecycle, resolved_source, untrusted,
                             sanitization_state=sanitization_state)

    def invalidate(self, inv: MemoryInvalidate) -> WriteResult:
        """§5b Edit/Delete-Propagation: markiert die bestehende Zeile (Natural-Key) als gelöscht
        (deleted_at) bzw. abgelöst (superseded_at + reindex_state='superseded'). KEIN Scan/Gate:
        es wird kein neuer Content geschrieben, nur ein Lifecycle-Zustand auf einer schon
        owner-eigenen Zeile gesetzt (die RLS-Policy trägt die Isolation). Idempotent + graceful:
        eine nie geschriebene / bereits invalidierte Zeile -> 0 Zeilen betroffen -> persisted=True,
        memory_item_written=False (kein Fehler). persisted=True NUR nach erfolgreichem commit."""
        # (0a) Anker fail-closed (wie write()): kein UPDATE mit halbem Anker.
        try:
            tenant = normalize_context_value(inv.tenant_id, "tenant_id")
            owner = normalize_context_value(inv.owner_id, "owner_id")
        except VaultContextError as e:
            return WriteResult(status=STATUS_REFUSED, persisted=False, message=str(e))

        # (0b) Modus + Identitäts-Felder fail-closed.
        if inv.mode not in VALID_INVALIDATE_MODES:
            return WriteResult(status=STATUS_REFUSED, persisted=False,
                               message=f"unbekannter invalidate-mode '{inv.mode}'")
        if not (isinstance(inv.source_table, str) and inv.source_table):
            return WriteResult(status=STATUS_REFUSED, persisted=False, message="source_table fehlt")
        if not (isinstance(inv.source_id, str) and inv.source_id):
            return WriteResult(status=STATUS_REFUSED, persisted=False, message="source_id fehlt")

        sql = _MEMORY_ITEMS_DELETE if inv.mode == INVALIDATE_DELETE else _MEMORY_ITEMS_SUPERSEDE
        conn = self._connect()
        affected = 0
        try:
            with vault_transaction(conn, tenant, owner) as cur:
                cur.execute(sql, (tenant, owner, inv.source_table, inv.source_id))
                # rowcount kann bei manchen Treibern -1 sein (unbekannt) -> defensiv auf 0.
                rc = getattr(cur, "rowcount", 0)
                affected = rc if isinstance(rc, int) and rc >= 0 else 0
            conn.commit()
        except BaseException as e:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error("vault invalidate failed (rolled back): %s", type(e).__name__)
            return WriteResult(status=STATUS_ERROR, persisted=False,
                               message="Vault-Invalidierung nicht committet -- bitte erneut versuchen")

        return WriteResult(status=STATUS_INVALIDATED, persisted=True,
                           memory_item_written=(affected > 0))

    def recall(self, req: MemoryRecall) -> RecallResult:
        """Stufe-6 Lesepfad (tsvector-Fläche). SELECT über den vollständigen Recall-Filter unter der
        Composite-Anker-Policy (RLS-GUCs via vault_transaction) -- tenant/owner NIE im Query-Text.
        Read-only: KEIN commit (der Pool rollbackt den Read-Scope beim Checkin). Fetchall INNERHALB
        des Transaktions-Fensters (reset-on-return leert die Connection danach). Wirft NIE -- ein
        Fehler kommt als RecallResult(status=error, available=False) zurück (die Naht ist zusätzlich
        fail-soft). ``available`` unterscheidet 'sauber 0 Treffer' (True) von 'nicht nachschaubar'
        (False) -- die Ehrlichkeits-Klausel darf leeren Recall NIE als bewiesene Abwesenheit lesen."""
        # (0a) Anker fail-closed (wie write()/invalidate()): kein SELECT mit halbem Anker.
        try:
            tenant = normalize_context_value(req.tenant_id, "tenant_id")
            owner = normalize_context_value(req.owner_id, "owner_id")
        except VaultContextError as e:
            return RecallResult(status=STATUS_REFUSED, available=False, message=str(e))

        # (0b) Query fail-safe + Limit bounded. Leerer Suchbegriff -> sauberer 0-Treffer-Lauf
        #      (available=True: der Pfad ist erreichbar, es gibt nur nichts zu matchen), KEIN Fehler.
        query = req.query.strip() if isinstance(req.query, str) else ""
        if not query:
            return RecallResult(status=STATUS_RECALL_EMPTY, items=[], available=True)
        limit = req.limit if isinstance(req.limit, int) else RECALL_LIMIT_DEFAULT
        limit = max(1, min(limit, RECALL_LIMIT_MAX))

        conn = self._connect()
        try:
            with vault_transaction(conn, tenant, owner) as cur:
                cur.execute(_MEMORY_ITEMS_RECALL, (query, limit))
                rows = cur.fetchall()   # INNERHALB der Txn materialisieren (reset-on-return leert danach)
        except BaseException as e:  # noqa: BLE001 -- fail-closed: kein stiller Teil-Read
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error("vault recall failed (rolled back): %s", type(e).__name__)
            return RecallResult(status=STATUS_ERROR, available=False,
                                message="Vault-Recall nicht ausführbar -- kein Gedächtnis-Zugriff")

        items = [
            RecallItem(
                source_table=r[0], source_id=r[1], summary=(r[2] or ""),
                created_at=r[3], sensitivity=r[4], from_untrusted_inbound=bool(r[5]),
            )
            for r in (rows or [])
        ]
        return RecallResult(
            status=(STATUS_RECALLED if items else STATUS_RECALL_EMPTY),
            items=items, available=True)

    # -- Validierung --------------------------------------------------------
    def _validate_vocab(self, req: MemoryWrite) -> str:
        """Gibt "" bei OK, sonst eine owner-freundliche Fehlermeldung (fail-closed)."""
        if req.source not in VALID_SOURCES:
            return f"unbekannte source '{req.source}'"
        if req.trust_level not in VALID_TRUST_LEVELS:
            return f"unbekanntes trust_level '{req.trust_level}'"
        if req.retention_class not in VALID_RETENTION_CLASSES:
            return f"unbekannte retention_class '{req.retention_class}'"
        if req.sensitivity not in VALID_SENSITIVITY:
            return f"unbekannte sensitivity '{req.sensitivity}'"
        if not isinstance(req.source_table, str) or req.source_table == "":
            return "source_table fehlt"
        if not isinstance(req.source_id, str) or req.source_id == "":
            return "source_id fehlt"
        if not isinstance(req.source_hash, str) or req.source_hash == "":
            return "source_hash fehlt"
        if req.raw_bytes is not None and req.source_table != SOURCE_TABLE_OBJECT:
            # Kohärenz-Guard: ein raw-Write MUSS via source_table='object_metadata' traversieren,
            # sonst hängt die object_metadata-Zeile als Waise (kein Meaning-Row findet sie).
            return f"raw_bytes verlangt source_table='{SOURCE_TABLE_OBJECT}', nicht '{req.source_table}'"
        if req.embedding_dimensions != EMBED_DIMENSIONS_DEFAULT:
            # Muss zur gepinnten vector(1024)-Spalte passen (mem_embdim_ck); früher, klarer Fehler.
            return f"embedding_dimensions {req.embedding_dimensions} != {EMBED_DIMENSIONS_DEFAULT}"
        return ""

    # -- Provenienz-Ableitung (die verlustfreie Inverse des Recall-Filters) --
    def _derive_provenance(self, d: GateDecision, req: MemoryWrite):
        """Bildet die Gate-Entscheidung auf durable Spalten ab. Der (deferred) Recall-Filter ist
        WHERE lifecycle_status='confirmed' AND deleted_at IS NULL AND quarantined_at IS NULL
        AND superseded_at IS NULL; dieser Write ist dessen exakte Inverse. Die §5b-Invalidierung
        (invalidate(): deleted_at bzw. superseded_at) setzt genau die hier ausgeschlossenen
        Zustände -> eine editierte/gelöschte Zeile wird nicht mehr recall-fähig.

        Rückgabe: (lifecycle_status, resolved_source, from_untrusted_inbound).
        """
        if d.stage:
            # candidate: NICHT recall-fähig bis Owner-Confirm. from_untrusted_inbound spiegelt
            # den Taint (untrusted-by-default). Kollabiert NICHT mit allow-bg (das ist confirmed).
            return LIFECYCLE_CANDIDATE, req.source, (not _explicitly_trusted(req.taint))

        # d.allow ab hier.
        if d.taint_marker == "retrieval_derived":
            # Sauberer Hintergrund-Auto-Capture (ADR-0044:202/210): COMMIT recall-fähig, getaintet.
            # retrieval_derived hat KEINE Spalte -> verlustfrei kodiert als source='auto_capture'
            # (∧ confirmed). Der künftige forbid-Gate keyt darauf. Fail-closed, falls der Caller
            # eine andere source deklariert (Provenienz-Widerspruch).
            if req.source != SOURCE_AUTO_CAPTURE:
                raise VaultStoreError(
                    "retrieval_derived (bg-clean) verlangt source='auto_capture', "
                    f"nicht '{req.source}'")
            return LIFECYCLE_CONFIRMED, SOURCE_AUTO_CAPTURE, False

        # allow ohne Marker = sauberer Vordergrund.
        if req.source == SOURCE_AUTO_CAPTURE:
            # auto_capture NUR über den bg-retrieval_derived-Zweig -- nie im Vordergrund.
            raise VaultStoreError("source='auto_capture' nur über den Hintergrund-Pfad zulässig")
        if req.source == SOURCE_FOREGROUND_OWNER:
            # Owner kuratiert bewusst im Vordergrund -> confirmed, recall-fähig.
            return LIFECYCLE_CONFIRMED, SOURCE_FOREGROUND_OWNER, False
        # Modell-vorgeschlagen (skill) oder sonstige saubere Vordergrund-source -> candidate
        # (ADR-0041:718 memory.propose = "Candidate only, kein Persist ohne Confirm").
        return LIFECYCLE_CANDIDATE, req.source, (not _explicitly_trusted(req.taint))

    # -- Persistenz ---------------------------------------------------------
    def _persist(self, req: MemoryWrite, tenant: str, owner: str,
                 lifecycle: str, source: str, untrusted: bool, *,
                 sanitization_state: str) -> WriteResult:
        """Krypto (nur raw) -> Object-Sink -> EIN RLS-Transaktions-Fenster (object_metadata +
        memory_items) -> commit. persisted=True NUR nach erfolgreichem commit (never-lost)."""
        object_key: Optional[str] = None
        key_ref: Optional[str] = None

        # (a) Roh-Schicht: verschlüsseln + Ciphertext durable ablegen VOR den DB-Zeilen. Ein
        #     Waise-Ciphertext (commit scheitert danach) ist harmlos + GC-bar; eine object_metadata-
        #     Zeile ohne Objekt wäre ein toter Verweis -> darum Sink zuerst.
        if req.raw_bytes is not None:
            if self._crypto is None or self._object_sink is None:
                return WriteResult(
                    status=STATUS_ERROR, persisted=False,
                    message="raw_bytes ohne crypto/object_sink verdrahtet -> nicht sicher ablegbar")
            try:
                enc = self._crypto.encrypt(req.raw_bytes, owner_id=owner)
                key_ref = enc["key_ref"]
                object_key = req.source_id  # kohärente Traversierung: object_key == source_id
                self._object_sink(object_key, enc["envelope"], owner_id=owner, tenant_id=tenant)
            except Exception as e:  # noqa: BLE001 -- DLP: nur Grund, nie Wert
                logger.error("vault raw-layer persist failed: %s", type(e).__name__)
                return WriteResult(status=STATUS_ERROR, persisted=False,
                                   message="Roh-Schicht konnte nicht sicher abgelegt werden")

        # (b) sanitization_state wurde in write() aus scanner_ok abgeleitet und hereingereicht.
        params = (
            tenant, owner, req.device_id, req.local_id,
            source, req.sensitivity, req.trust_level, req.retention_class,
            req.source_table, req.source_id, req.source_hash,
            req.redaction_state, req.redaction_version, sanitization_state,
            req.embedding_provider, req.embedding_model, req.embedding_version, req.embedding_dimensions,
            req.summary_redacted, untrusted, lifecycle,
        )

        conn = self._connect()
        object_written = False
        try:
            with vault_transaction(conn, tenant, owner) as cur:
                if object_key is not None:
                    cur.execute(_OBJECT_METADATA_INSERT,
                                (tenant, owner, source, req.trust_level, object_key, key_ref))
                    object_written = True
                cur.execute(_MEMORY_ITEMS_INSERT, params)
            # commit NACH den Inserts (beendet die Txn + löscht die transaction-local GUCs;
            # danach läuft KEIN RLS-scoped Statement mehr). vault_transaction committet nicht.
            conn.commit()
        except BaseException as e:  # noqa: BLE001
            # vault_transaction hat bei Fehler IM with schon rollback gemacht; hier fängt es den
            # Fehler des commit() selbst + macht rollback idempotent (belt-and-suspenders).
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error("vault persist failed (rolled back): %s", type(e).__name__)
            return WriteResult(status=STATUS_ERROR, persisted=False,
                               message="Vault-Write nicht committet -- bitte erneut versuchen")

        # status ≡ Recall-Fähigkeit (nicht das rohe Gate-Flag): confirmed = written (recall-fähig),
        # candidate = staged (durabel, NICHT recall-fähig). Ein propose (allow-fg, aber candidate)
        # ist damit korrekt 'staged', nicht 'written'.
        status = STATUS_WRITTEN if lifecycle == LIFECYCLE_CONFIRMED else STATUS_STAGED
        return WriteResult(status=status, persisted=True, lifecycle_status=lifecycle,
                           memory_item_written=True, object_metadata_written=object_written)
