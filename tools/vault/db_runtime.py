"""Vault-DB Connection-Pool (psycopg) für den WRITE-Modus des dark-wire.

Lazy + Modul-Singleton: psycopg wird ERST beim ersten WRITE-Write importiert. Heute ist psycopg
NICHT im Engine-Trunk (bewusst, Isolation Stufe 3) -> get_vault_pool() wirft dann, und der
Aufrufer (vault_wiring._do_vault_write) fällt fail-soft. psycopg + die DSN kommen mit dem Deploy
(api-server dual-home Zone 51 + Image-Rebuild, WIRING_PLAN §6). Diese Datei ist die Naht, KEINE
Laufzeit-Abhängigkeit an der Integrationsgrenze.

TRAGENDE POOL-INVARIANTEN (WIRING_PLAN §6 / Write-Path-Spec §5a):
  * autocommit=False -- die transaction-local GUCs (set_config(...,is_local=true)) halten NUR in
    einer offenen Transaktion; eine Autocommit-Connection liesse den RLS-Kontext verdampfen ->
    INSERT trifft WITH-CHECK mit leeren GUCs = fail-closed 0 Zeilen.
  * reset_on_return='rollback' (vault_context.POOL_RESET_ON_RETURN) -- beim putconn wird jede offene
    Transaktion (und damit der transaction-local Kontext) verworfen; der nächste Borrower sieht 0.
Die exakte psycopg-Pool-Verifikation (API-Details) passiert am Deploy, wenn psycopg verfügbar ist.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from tools.vault.vault_context import POOL_RESET_ON_RETURN

logger = logging.getLogger(__name__)

# Env-Namen (am Deploy provisioniert; NICHT hier hartkodiert). Die DSN zeigt auf den in-zone-FQDN
# vault-db.51.jarvis.internal:5432/jarvis_vault, Rolle jarvis_vault_app (NOSUPERUSER NOBYPASSRLS).
# Die DSN MUSS einen server-seitigen statement_timeout mitgeben (options=-c statement_timeout=2000),
# damit eine langsame Query den Live-Turn nicht hängt (fail-soft-Vertrag deckt Blockieren, nicht
# nur Exceptions -- Review 2026-07-09). Der connect-/pool-Warte-Timeout kommt aus getconn() (unten).
_DSN_ENV = "VAULT_DB_DSN"
# _FILE-Konvention (deckungsgleich mit POSTGRES_PASSWORD_FILE der vault-db): die DSN trägt das
# DB-Passwort -> bevorzugt aus einer read-only gemounteten Secret-Datei lesen, NICHT als env-Literal
# (env landet in docker-inspect/Crash-Dumps/Diag-Telemetrie). _FILE gewinnt, wenn gesetzt; sonst
# Fallback auf die env-Variable (v.a. Tests). "Passwort NIE als Literal" (vault-db-compose-Kanon).
_DSN_FILE_ENV = "VAULT_DB_DSN_FILE"

# Kurzer getconn-Timeout: der Shadow-Write ist best-effort; kann der Pool nicht rasch eine
# Connection liefern (vault-db tot/Pool erschöpft), wirft getconn(timeout=) statt bis 30s zu
# blockieren -> der Aufrufer fängt es fail-soft und überspringt. Bounded worst-case statt Hänger.
VAULT_GETCONN_TIMEOUT_S = 1.0

_pool: Optional[Any] = None


class VaultPoolUnavailable(RuntimeError):
    """psycopg fehlt ODER keine DSN gesetzt -> kein WRITE-Modus (Aufrufer fällt fail-soft)."""


def _configure(conn: Any) -> None:
    # Transaktionen erzwingen (kein Autocommit) -> transaction-local GUCs halten. Siehe Kopf.
    conn.autocommit = False


def _read_dsn() -> str:
    """DSN aus dem _FILE-Secret (bevorzugt) ODER der env-Variable (Fallback). Ist _FILE gesetzt aber
    unlesbar -> VaultPoolUnavailable (fail-closed: NICHT still auf eine evtl. veraltete env fallen)."""
    path = os.environ.get(_DSN_FILE_ENV, "").strip()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError as exc:
            raise VaultPoolUnavailable(f"{_DSN_FILE_ENV} nicht lesbar: {type(exc).__name__}") from exc
    return os.environ.get(_DSN_ENV, "")


def get_vault_pool() -> Any:
    """Lazy Modul-Singleton-Pool. Wirft VaultPoolUnavailable, wenn psycopg fehlt oder keine DSN
    gesetzt ist (Integrationsgrenze / kein Deploy)."""
    global _pool
    if _pool is not None:
        return _pool
    dsn = _read_dsn()
    if not dsn:
        raise VaultPoolUnavailable(f"{_DSN_ENV}/{_DSN_FILE_ENV} nicht gesetzt")
    try:
        from psycopg_pool import ConnectionPool  # lazy: heute nicht im Engine-Trunk
    except Exception as exc:  # noqa: BLE001
        raise VaultPoolUnavailable("psycopg_pool nicht verfügbar") from exc
    # reset (rollback-on-return): psycopg3 rollt beim Return eine offene Txn zurück -> Kontext weg.
    # POOL_RESET_ON_RETURN ('rollback') ist der byte-fixierte Vertrag, gegen den das assertet.
    _pool = ConnectionPool(
        conninfo=dsn, min_size=1, max_size=8, open=True,
        configure=_configure, reset=_reset,
    )
    logger.info("vault pool erstellt (reset_on_return=%s)", POOL_RESET_ON_RETURN)
    return _pool


def _reset(conn: Any) -> None:
    """Return-Reset: offene Transaktion verwerfen (transaction-local Kontext löschen)."""
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        pass


def close_vault_pool() -> None:
    """Für Tests/Shutdown: den Singleton schliessen + zurücksetzen."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:  # noqa: BLE001
            pass
        _pool = None
