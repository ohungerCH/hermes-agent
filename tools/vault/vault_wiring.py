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


def _phase1_source_id(content: str) -> str:
    """PHASE-1 (WIRING_PLAN §5, NICHT enshrined): Content-Hash als source_id. ACHTUNG (Advisor
    2026-07-09): dieser Aufrufer ist foreground_owner -> der VaultStore leitet CONFIRMED ab (NICHT
    candidate). Folge: ein Memory-Edit "X"->"Y" erzeugt eine NEUE confirmed-Zeile hash(Y) und lässt
    die ALTE confirmed-Zeile hash(X) verwaist (kein Supersede-Link); ein remove wird gar nicht
    geshadowt -> Vault + MEMORY.md divergieren. Deshalb ist RECALL BLOCKIERT, bis Edit/Delete-
    Propagation existiert (remove->soft-delete, replace->supersede-alt). Identischer Re-Write ist
    idempotent (gleicher Natural-Key). Echte source_id-Factory bleibt Write-Path §5-offen."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def vault_shadow_write(action: str, target: str, content: Optional[str],
                       *, store_result: Any = None) -> Optional[str]:
    """Best-effort Shadow-Write in den Vault. Rückgabe rein informativ (Status-String oder None);
    memory_tool IGNORIERT sie. NIE raise. No-op wenn: Pfad aus / kein add|replace / kein content /
    file-Write nicht erfolgreich / origin!=foreground / keine Session-Identität.

    remove wird bewusst NICHT geshadowt (Löschung = eigene Soft-Delete/Quarantäne-Semantik im
    Vault, spätere Scheibe)."""
    try:
        if not vault_path_active():
            return None
        if action not in ("add", "replace"):
            return None
        if not content:
            return None
        if not _store_result_ok(store_result):
            return None
        try:
            from tools.write_approval import current_origin
            if current_origin() != "foreground":
                return None
        except Exception:
            return None  # Herkunft unklar -> fail-closed kein Vault-Write
        ident = get_vault_write_identity()
        if ident is None:
            return None
        tenant_id, owner_id = ident
        return _do_vault_write(target, content, tenant_id, owner_id)
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


def _do_vault_write(target: str, content: str, tenant_id: str, owner_id: str) -> str:
    """PLUMBING = bauen + verwerfen (Dry-Run). WRITE = borrow conn -> VaultStore.write -> return conn.
    Der Owner-Memory-Write ist reine Bedeutungs-Schicht (raw_bytes=None) -> kein crypto/object_sink."""
    req = _build_request(target, content, tenant_id, owner_id)

    if not vault_write_enabled():
        logger.info("vault PLUMBING dry-run: source_table=%s (kein durabler Write)", req.source_table)
        return "plumbing_dry_run"

    from tools.vault.vault_store import VaultStore
    from tools.vault import db_runtime
    pool = db_runtime.get_vault_pool()
    # getconn mit kurzem Timeout: ein toter Pool/DB darf den Live-Turn nicht hängen (fail-soft
    # deckt Blockieren, nicht nur Exceptions). Timeout -> Exception -> vom äusseren try gefangen.
    conn = pool.getconn(timeout=db_runtime.VAULT_GETCONN_TIMEOUT_S)
    try:
        store = VaultStore(connect=lambda: conn)
        result = store.write(req)
    finally:
        pool.putconn(conn)  # reset_on_return='rollback' säubert die transaction-local GUCs
    if not result.persisted:
        logger.warning("vault shadow-write nicht persistiert: status=%s", result.status)
    return result.status
