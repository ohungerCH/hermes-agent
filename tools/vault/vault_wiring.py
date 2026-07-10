"""Vault dark-wire: der caller-seitige Andock des VaultStore an den Foreground-Owner-Memory-Write.

Stufe-5 Live-Scheibe Teil 3 (VAULTSTORE_WIRING_PLAN.md). Diese Datei ist die EINZIGE Naht
zwischen dem bestehenden file-backed Memory-Pfad (tools/memory_tool.py) und dem VaultStore
(tools/vault/vault_store.py). memory_tool ruft NUR vault_shadow_write() -- alle Bedingungen,
Flags und der fail-soft-Vertrag leben hier.

LOAD-BEARING INVARIANTEN (Advisor-Review 2026-07-09, WIRING_PLAN §4):
  * SHADOW, nicht REPLACE: der Vault-Write passiert ZUSAETZLICH zum MEMORY.md-Write; MEMORY.md
    bleibt autoritativ. Wir beeinflussen das Aufrufer-Ergebnis NIE.
  * FAIL-SOFT: läuft im Live-Turn -> NIE raise, NIE blockieren/verzögern, Fehler nur geloggt
    (PII-frei). vault-db langsam/tot darf "merk dir X" nicht anfassen.
  * FG-ONLY + RESOLVED-OWNER: nur origin==foreground UND eine server-autoritative Session-
    Identität (tenant_id+owner_id, via ContextVar) -> sonst no-op. Schliesst den geteilten
    background_review-Pfad + den §5-deferrten Background-Promote-Hazard aus.

Flag-Leiter (INV-6, default alles AUS = heutiges Verhalten exakt):
  vault.plumbing_enabled -> Pfad läuft als Dry-Run (bauen + verwerfen, kein durabler Write).
  vault.write_enabled    -> durabler Shadow-Write (impliziert plumbing).
  vault.recall_enabled   -> Lesen (separat, hier nicht genutzt).

INTEGRATIONSGRENZE: der ContextVar-Setter wird vom api-server-Turn-Eingang aufgerufen (dort ist
die TrustedSurfaceSessionIdentity aufgelöst) -- diese EINE Zeile ist Deploy-Schritt, nicht hier
(api_server.py ist fremdes WIP). Ohne Setter liefert get_vault_write_identity() None -> der
dark-wire ist inert (korrekt: nichts gesetzt, Flags aus).
"""
from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# Foreground-Owner-Origins (skill_provenance-Taxonomie): der dark-wire feuert NUR für diese --
# "foreground" (roher ContextVar-Default) ODER "assistant_tool" (der Normalfall: das Modell ruft
# im OWNER-Turn das memory-Werkzeug). "background_review" (Hintergrund-Selbstverbesserungs-Fork)
# ist bewusst AUSGESCHLOSSEN (§5-Background-Promote-Hazard: fehl-gestempelte owner_id passiert RLS).
# Fail-closed: eine unbekannte Herkunft -> kein Vault-Write.
_FOREGROUND_ORIGINS = frozenset({"foreground", "assistant_tool"})


# ---------------------------------------------------------------------------
# Flag-Leiter (INV-6)
# ---------------------------------------------------------------------------

def _vault_flag(name: str) -> bool:
    """Liest vault.<name> aus config.yaml. Default False; jeder Fehler -> False (fail-closed)."""
    try:
        from hermes_cli.config import load_config, cfg_get
        return bool(cfg_get(load_config(), "vault", name, default=False))
    except Exception:
        return False


def vault_write_enabled() -> bool:
    return _vault_flag("write_enabled")


def vault_plumbing_enabled() -> bool:
    return _vault_flag("plumbing_enabled")


def vault_recall_enabled() -> bool:
    return _vault_flag("recall_enabled")


def vault_path_active() -> bool:
    """Läuft der dark-wire überhaupt? (plumbing ODER write). write impliziert plumbing."""
    return vault_plumbing_enabled() or vault_write_enabled()


# ---------------------------------------------------------------------------
# Server-autoritative Session-Identität (ContextVar-Träger)
# ---------------------------------------------------------------------------
# Gesetzt vom api-server-Turn-Eingang aus TrustedSurfaceSessionIdentity (tenant_id+owner_id,
# client-unfälschbar). Muster wie skill_provenance/current_origin. Default None = keine
# aufgelöste Identität -> kein Vault-Write (resolved-owner-Invariante).

_identity: ContextVar[Optional[Tuple[str, str]]] = ContextVar("vault_write_identity", default=None)


def set_vault_write_identity(tenant_id: str, owner_id: str) -> Token:
    """Setzt die server-autoritative (tenant_id, owner_id) für diesen Kontext. Nur non-empty
    Strings; sonst wird KEINE Identität gesetzt (Token löscht dann auf None)."""
    if not (isinstance(tenant_id, str) and tenant_id and isinstance(owner_id, str) and owner_id):
        return _identity.set(None)
    return _identity.set((tenant_id, owner_id))


def reset_vault_write_identity(token: Token) -> None:
    _identity.reset(token)


def get_vault_write_identity() -> Optional[Tuple[str, str]]:
    return _identity.get()


@contextmanager
def vault_write_identity(tenant_id: str, owner_id: str) -> Iterator[None]:
    """Scoped-Setter für den Turn (der api-server umschliesst den Agent-Turn hiermit)."""
    tok = set_vault_write_identity(tenant_id, owner_id)
    try:
        yield
    finally:
        reset_vault_write_identity(tok)


# ---------------------------------------------------------------------------
# Shadow-Write (die einzige von memory_tool gerufene Funktion)
# ---------------------------------------------------------------------------

def _store_result_ok(store_result: Any) -> bool:
    """Nur einen ERFOLGREICHEN file-backed Write shadowen. Strikt `is True` (fail-safe: bei
    fehlendem/mehrdeutigem success-Feld NICHT shadowen). MemoryStore.add/replace/remove liefern
    bei Erfolg _success_response mit success:True (memory_tool.py:472)."""
    return isinstance(store_result, dict) and store_result.get("success") is True


def _source_table(target: str) -> str:
    return "user_profile" if target == "user" else "owner_memory"


def _normalize_entry(text: Optional[str]) -> Optional[str]:
    """MEMORY.md speichert Einträge GESTRIPPT (MemoryStore.add/replace strippen intern). Die
    Natural-Key-Identität (Content-Hash) MUSS dieselbe Normalform hashen, sonst lokalisiert ein
    späteres remove/replace die alte Zeile nicht (hash(un-gestrippt) != hash(gestrippt)). Deshalb
    strippt der Vault-Rand JEDEN Eintrag EINMAL -- neuer Content wie Alt-Eintrag (letzterer kommt
    schon gestrippt aus dem Store, strip ist dort idempotent)."""
    return text.strip() if isinstance(text, str) else text


def _phase1_source_id(content: str) -> str:
    """PHASE-1 (WIRING_PLAN §5, NICHT enshrined): Content-Hash der GESTRIPPTEN Normalform als
    source_id. Für owner_memory/user_profile IST der Eintrag seine Identität (kein externer Key) --
    laut Advisor 2026-07-09 wahrscheinlich die permanente, nicht bloss Phase-1-Identität; die echte
    Factory-Frage betrifft die ANDEREN source_tables (object_metadata/ingest). §5b (Edit/Delete-
    Propagation) ist jetzt gebaut: replace supersediert hash(Alt-Eintrag) + fügt hash(Neu) ein,
    remove soft-deletet hash(Alt-Eintrag). Lokalisierung beweisbar konsistent: jeder Write hasht
    hash(voller gestrippter Eintrag), jede Invalidierung hasht dieselbe Form des Alt-Eintrags."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def vault_shadow_write(action: str, target: str, content: Optional[str],
                       *, store_result: Any = None, old_entry: Optional[str] = None) -> Optional[str]:
    """Best-effort Shadow-Write in den Vault. Rückgabe rein informativ (Status-String oder None);
    memory_tool IGNORIERT sie. NIE raise. Deckt alle drei Mutationen (§5b):
      add     -> neue Zeile schreiben (hash(neuer Content)).
      replace -> alte Zeile supersedieren (hash(Alt-Eintrag)) + neue schreiben (hash(neuer Content)).
      remove  -> alte Zeile soft-deleten (hash(Alt-Eintrag)); KEIN Insert.

    No-op wenn: Pfad aus / unbekannte Aktion / file-Write nicht erfolgreich / origin!=foreground /
    keine Session-Identität / fehlender Content (add|replace) bzw. fehlender Alt-Eintrag
    (replace|remove). Content + Alt-Eintrag werden EINMAL gestrippt (Hash-Konsistenz zur
    MEMORY.md-Normalform, s. _normalize_entry)."""
    try:
        if not vault_path_active():
            return None
        if action not in ("add", "replace", "remove"):
            return None
        if not _store_result_ok(store_result):
            return None
        content = _normalize_entry(content)
        old_entry = _normalize_entry(old_entry)
        # Content nur für add|replace nötig (der NEUE Eintrag); Alt-Eintrag nur für replace|remove
        # (die zu invalidierende Zeile). Fehlt das Jeweilige -> fail-safe No-op.
        if action in ("add", "replace") and not content:
            return None
        if action in ("replace", "remove") and not old_entry:
            return None
        try:
            from tools.write_approval import current_origin
            if current_origin() not in _FOREGROUND_ORIGINS:
                return None
        except Exception:
            return None  # Herkunft unklar -> fail-closed kein Vault-Write
        ident = get_vault_write_identity()
        if ident is None:
            return None
        tenant_id, owner_id = ident
        return _do_vault_op(action, target, content, old_entry, tenant_id, owner_id)
    except Exception as e:  # noqa: BLE001 -- fail-soft: der Live-Turn darf NIE hierdran hängen
        logger.warning("vault shadow-write übersprungen (fail-soft): %s", type(e).__name__)
        return None


def _build_request(target: str, content: str, tenant_id: str, owner_id: str):
    """Baut den MemoryWrite. PHASE-1-Defaults (WIRING_PLAN §5): sensitivity personal_low,
    trust_level trusted (Owner-authored), summary_redacted=content (keine DLP-Pipeline ->
    trivial scan⊇recall, WIRING_PLAN §5a-Invariante erfüllt), redaction pending -> embedding
    bleibt NULL bis ein späterer Redaktions-/Embed-Lauf."""
    from tools.vault.vault_store import (
        MemoryWrite, SOURCE_FOREGROUND_OWNER, TRUST_TRUSTED, RETENTION_PERMANENT_MEANING,
    )
    return MemoryWrite(
        content=content, owner_id=owner_id, tenant_id=tenant_id,
        origin="foreground", source=SOURCE_FOREGROUND_OWNER,
        source_table=_source_table(target),
        source_id=_phase1_source_id(content), source_hash=_phase1_source_id(content),
        sensitivity="personal_low", trust_level=TRUST_TRUSTED,
        retention_class=RETENTION_PERMANENT_MEANING,
        summary_redacted=content, redaction_state="pending", redaction_version="v0",
        taint={"from_untrusted_inbound": False},
    )


def _build_invalidate(target: str, old_entry: str, tenant_id: str, owner_id: str, mode: str):
    """Baut die MemoryInvalidate für §5b: Natural-Key der abzulösenden/zu löschenden Zeile aus der
    STABILEN Identität des vollen Alt-Eintrags (source_id == source_table-lokaler Content-Hash,
    identisch zu dem, mit dem die Zeile via _build_request geschrieben wurde)."""
    from tools.vault.vault_store import MemoryInvalidate
    return MemoryInvalidate(
        owner_id=owner_id, tenant_id=tenant_id,
        source_table=_source_table(target),
        source_id=_phase1_source_id(old_entry), mode=mode,
    )


def _do_vault_op(action: str, target: str, content: Optional[str], old_entry: Optional[str],
                 tenant_id: str, owner_id: str) -> str:
    """PLUMBING = bauen + verwerfen (Dry-Run). WRITE = borrow conn -> VaultStore-Op(s) -> return conn.
    Der Owner-Memory-Write ist reine Bedeutungs-Schicht (raw_bytes=None) -> kein crypto/object_sink.

    replace = supersede-ALT ZUERST, dann insert-NEU (privacy-safe: ein Crash dazwischen lässt die
    alte Zeile abgelöst + die neue fehlend = Under-Recall; die umgekehrte Reihenfolge liesse bei
    Crash BEIDE recall-fähig = die §5b-Divergenz). Scheitert der Supersede, wird NICHT eingefügt."""
    from tools.vault.vault_store import INVALIDATE_DELETE, INVALIDATE_SUPERSEDE

    if not vault_write_enabled():
        # tenant/owner MIT loggen: der Override-Smoke (§7) beweist in PLUMBING, dass der am
        # Schreibpunkt aufgelöste Anker IMMER die JWT-Session-Identität ist (NICHT client-
        # beeinflussbar) -> der resolved owner MUSS beobachtbar sein. owner_id = stabiler
        # Identifier (owner-primary), kein DLP-Inhalt; nur im flag-gegateten Dry-Run.
        logger.info("vault PLUMBING dry-run: action=%s source_table=%s tenant=%s owner=%s "
                    "(kein durabler Write)", action, _source_table(target), tenant_id, owner_id)
        return "plumbing_dry_run"

    from tools.vault.vault_store import VaultStore
    from tools.vault import db_runtime
    pool = db_runtime.get_vault_pool()
    # getconn mit kurzem Timeout: ein toter Pool/DB darf den Live-Turn nicht hängen (fail-soft
    # deckt Blockieren, nicht nur Exceptions). Timeout -> Exception -> vom äusseren try gefangen.
    conn = pool.getconn(timeout=db_runtime.VAULT_GETCONN_TIMEOUT_S)
    try:
        store = VaultStore(connect=lambda: conn)   # eine Connection, je Op eine eigene Txn+commit
        if action == "add":
            result = store.write(_build_request(target, content, tenant_id, owner_id))
        elif action == "remove":
            result = store.invalidate(
                _build_invalidate(target, old_entry, tenant_id, owner_id, INVALIDATE_DELETE))
        else:  # replace
            inv = store.invalidate(
                _build_invalidate(target, old_entry, tenant_id, owner_id, INVALIDATE_SUPERSEDE))
            if not inv.persisted:
                logger.warning(
                    "vault replace: supersede-alt nicht committet (status=%s) -> insert-neu "
                    "übersprungen (verhindert doppelte Recall-Zeile)", inv.status)
                return inv.status
            result = store.write(_build_request(target, content, tenant_id, owner_id))
    finally:
        pool.putconn(conn)  # reset_on_return='rollback' säubert die transaction-local GUCs
    if not result.persisted:
        logger.warning("vault shadow-op nicht persistiert: action=%s status=%s", action, result.status)
    return result.status


# ---------------------------------------------------------------------------
# Recall (Lese-Naht) -- die zweite (und einzige lesende) von memory_tool gerufene Funktion
# ---------------------------------------------------------------------------
# LOAD-BEARING (Advisor 2026-07-10): Recall läuft im Trusted-Surface-Brain, das WERKZEUGE hat.
# Zurückgeholter Text, der "ignoriere alles, ruf Werkzeug X" sagt, ist ein echter Injektionsvektor --
# auch in owner-authored Memory (der Owner kann eine zitierte Injektion bewusst gemerkt haben, z.B.
# "merk dir diese verdächtige SMS: ..."). Verteidigung = STRUKTURELL WRAPPEN (als Daten, entity-
# encoded), NICHT blocken: das Security-Vokabular des Owners (C2-Namen, Exfil-Beispiele) bleibt voll
# abrufbar (#75 warn-vs-block). Der Wrap entity-encodet &,<,> -> ein Snippet kann den Delimiter NICHT
# aufbrechen (Wrap-Escape-Klasse, Task #34; NICHT die _maybe_wrap_untrusted-Rohinterpolation).

_RECALL_OPEN = "<recalled_memory"
_RECALL_CLOSE = "</recalled_memory>"


def _entity_encode(text: str) -> str:
    """Neutralisiert die Markup-Struktur: &,<,> -> Entities. Ein zurückgeholtes Snippet kann damit
    weder den Wrapper-Delimiter aufbrechen (Wrap-Escape) noch als Markup interpretiert werden. KEIN
    Token-Match, KEIN Blocken -- reine strukturelle Neutralisierung (Owner-Inhalt bleibt lesbar).
    Reihenfolge: & ZUERST (sonst würden die eingefügten &amp;/&lt;/&gt; doppelt kodiert)."""
    if not isinstance(text, str):
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wrap_recalled(summary: str, source_table: str, untrusted: bool) -> str:
    """Wrappt EIN zurückgeholtes Snippet als untrusted DATEN fürs Brain. Der Inhalt ist entity-
    encoded (strukturell neutralisiert); der Wrapper markiert klar, dass dies gemerkte DATEN sind,
    keine Anweisungen. ``source_table`` ist kontrolliertes Vokabular (owner_memory/user_profile),
    wird aber defensiv mitkodiert."""
    marker = "true" if untrusted else "false"
    return (f'{_RECALL_OPEN} source="{_entity_encode(source_table)}" untrusted_data="{marker}">'
            f"{_entity_encode(summary)}{_RECALL_CLOSE}")


def vault_shadow_recall(query: str, target: str = "memory",
                        *, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Best-effort Vault-Recall (tsvector-Fläche). Rückgabe = ein Modell-sicheres Dict oder None.
    NIE raise. No-op (None) wenn: recall_enabled aus / leerer Query / origin!=foreground / keine
    Session-Identität. Bei aktiver Naht + Fehler/toter DB: {available: False, ...} (fail-soft) --
    NIE als "kein Gedächtnis": eine leere Trefferliste bei available=True heisst "nichts gemerkt
    zu X"; available=False heisst "konnte nicht nachsehen". Jeder Treffer ist als untrusted DATEN
    gewrappt (s. _wrap_recalled)."""
    try:
        if not vault_recall_enabled():
            return None
        if not isinstance(query, str) or not query.strip():
            return None
        try:
            from tools.write_approval import current_origin
            if current_origin() not in _FOREGROUND_ORIGINS:
                return None
        except Exception:
            return None  # Herkunft unklar -> fail-closed kein Vault-Read
        ident = get_vault_write_identity()
        if ident is None:
            return None
        tenant_id, owner_id = ident
        return _do_vault_recall(query, target, tenant_id, owner_id, limit)
    except Exception as e:  # noqa: BLE001 -- fail-soft: der Live-Turn darf NIE hierdran hängen
        logger.warning("vault shadow-recall übersprungen (fail-soft): %s", type(e).__name__)
        return None


def _do_vault_recall(query: str, target: str, tenant_id: str, owner_id: str,
                     limit: Optional[int]) -> Dict[str, Any]:
    """Führt den Recall aus: borrow conn -> VaultStore.recall -> return conn. Wrappt jeden Treffer
    als untrusted Daten. ``available`` spiegelt RecallResult.available (Ehrlichkeits-Klausel)."""
    from tools.vault.vault_store import VaultStore, MemoryRecall, RECALL_LIMIT_DEFAULT
    from tools.vault import db_runtime
    lim = limit if isinstance(limit, int) and limit > 0 else RECALL_LIMIT_DEFAULT
    pool = db_runtime.get_vault_pool()
    # getconn mit kurzem Timeout: ein toter Pool/DB darf den Live-Turn nicht hängen (fail-soft
    # deckt Blockieren, nicht nur Exceptions). Timeout -> Exception -> vom äusseren try gefangen.
    conn = pool.getconn(timeout=db_runtime.VAULT_GETCONN_TIMEOUT_S)
    try:
        store = VaultStore(connect=lambda: conn)
        res = store.recall(MemoryRecall(
            owner_id=owner_id, tenant_id=tenant_id, query=query, limit=lim))
    finally:
        pool.putconn(conn)  # reset_on_return='rollback' säubert die Read-Txn + transaction-local GUCs
    if not res.available:
        return {"available": False, "matches": [], "reason": res.status}
    matches = [
        {
            "source": it.source_table,
            "content": _wrap_recalled(it.summary, it.source_table, it.from_untrusted_inbound),
        }
        for it in res.items
    ]
    return {"available": True, "matches": matches, "count": len(matches)}
