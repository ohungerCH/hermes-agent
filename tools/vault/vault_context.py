"""vault_context.py - GUC-Set-Konsument für den Jarvis-Vault-Composite-Anker (Stufe 3).

Setzt pro Transaktion die BEIDEN RLS-GUCs jarvis.tenant_id UND jarvis.owner_id
transaction-local (set_config(..., is_local=True)) aus SERVER-ASSERTED Werten und
garantiert fail-closed:
  - beide GUCs oder keiner (Validierung VOR jedem set_config; kein Teilkontext),
  - set_config-Fehler mitten in der Txn -> Transaktion MUSS abbrechen (kein Weiterlaufen),
  - Pool-Rückgabe -> Kontext gelöscht (rollback-on-return), nächster Borrower sieht 0.

Muster: finanzdashboard db_runtime.py (im Jarvis-Repo NICHT vorhanden -> hier frisch nach
der dokumentierten gate0-/ADR-0041-§3-Spezifikation gebaut).

STUFE-3-GRENZE (no-false-green):
  - Dieses Modul ist GEBAUT + UNIT-GETESTET, aber NOCH NICHT in api-server/engine verdrahtet
    (Live-Flow api-server/engine -> vault-db:5432 = Stufe 5).
  - Es erwartet die Header X-Jarvis-Tenant-Id / X-Jarvis-Owner-Id als bereits EDGE-GESTEMPELT
    (client-nicht-überschreibbar). Das Edge-Stamping BEIDER Header (proxy_set_header aus der
    Introspektion) ist GAP-G / Stufe 4. Der Issuer emittiert heute X-Jarvis-Tenant-Id +
    X-Jarvis-Device-Id; die owner_id-Header-Herkunft ist die Schwester-Naht (Stufe 4).
  - Der DB-Treiber (psycopg) ist absichtlich NICHT importiert: die Funktionen nehmen eine
    DB-API-kompatible cursor/connection (duck-typed) -> testbar ohne Treiber/Netz, drop-in
    für psycopg in Stufe 5. Das schützt zugleich die internal:true-Isolation der Zone 51
    (kein Host-Port aufreissen, nur um den Konsumenten zu testen).

SQL-Injection: der GUC-NAME ist immer ein fixes Literal (nie Eingabe); der WERT wird als
Query-Parameter übergeben (%s), zusätzlich durch die Regex normalisiert (defense-in-depth).

KRITISCHE INVARIANTE - TRUSTED-SQL-ONLY (Bedrohungsmodell, Red-Team-Befund 2026-07-06):
Die transaction-local GUCs sind von JEDEM Statement, das unter der App-Rolle in DERSELBEN
Transaktion läuft, neu setzbar (RLS wertet current_setting pro Statement aus). Ein einziges
angreifer-beeinflusstes set_config ODER ein zweites gestapeltes Statement kollabiert den Anker
(bewiesen: unter Kontext T1/O1 sichtbar 'a,d'; ein SPAETERES set_config('jarvis.tenant_id','T2')
in derselben Txn -> 'c' sichtbar). Folge: RLS-per-transaction-local-GUC gibt NULL Isolation,
sobald untrusted-/LLM-autoriertes rohes SQL unter der App-Rolle läuft. Die Isolation lebt daher
VOLLSTAENDIG upstream und dieses Modul kann eine Fehl-Stempelung oder ein mid-txn-GUC-Reset NICHT
abwehren:
  (1) die App-Rolle darf AUSSCHLIESSLICH getrustetes, parametrisiertes SQL mit server-gesetztem
      Kontext ausführen - NIEMALS vom Modell verfasstes rohes SQL (ADR-0029 untrusted-inbound-Gate);
  (2) das Edge stempelt X-Jarvis-Tenant-Id UND X-Jarvis-Owner-Id client-nicht-überschreibbar
      (GAP-G / Stufe 4).
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator, Mapping, Tuple

# --- Kanon-Konstanten (byte-fixiert, gate0 / ADR-0041 §3) ---------------------------------

TENANT_GUC = "jarvis.tenant_id"
OWNER_GUC = "jarvis.owner_id"

TENANT_HEADER = "X-Jarvis-Tenant-Id"
OWNER_HEADER = "X-Jarvis-Owner-Id"

# Normalisierungs-Regex (gate0 §Anker-Mechanik). Erlaubt UUIDs, owner-primary, device-ids;
# blockt leere Werte, führenden Bindestrich, Steuerzeichen, Whitespace, Quotes, Semikola,
# CR/LF (Header-Injection) und Überlänge (>128).
CONTEXT_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

# Pool-Rückgabe-Vertrag: eine Connection-Pool MUSS reset_on_return='rollback' fahren, damit
# transaction-local GUCs beim Checkin verworfen werden. Als Konstante exportiert, damit die
# Stufe-5-Pool-Konfiguration byte-genau dagegen assertet.
POOL_RESET_ON_RETURN = "rollback"


class VaultContextError(ValueError):
    """Fail-closed: ungültiger/fehlender Tenant-/Owner-Kontext. NIE mit Teilkontext weiter."""


# --- Validierung / Herkunft ---------------------------------------------------------------

def normalize_context_value(raw: object, field: str) -> str:
    """Validiere einen Kontext-Wert fail-closed. Wirft VaultContextError bei allem Unreinen.

    Deckt T9 (Regex-Guard) + T11 (Header-Safety: Steuerzeichen/CR/LF/NUL -> reject).
    """
    if raw is None:
        raise VaultContextError(f"{field}: fehlt (None)")
    if not isinstance(raw, str):
        raise VaultContextError(f"{field}: kein String ({type(raw).__name__})")
    # Explizit CR/LF/NUL/Steuerzeichen fangen (klare Fehlermeldung; die Regex würde sie
    # ohnehin ablehnen, aber Header-Injection soll benannt scheitern).
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in raw):
        raise VaultContextError(f"{field}: Steuerzeichen/Control-Char verboten (Header-Safety)")
    if not CONTEXT_VALUE_RE.match(raw):
        raise VaultContextError(f"{field}: verletzt Kontext-Regex {CONTEXT_VALUE_RE.pattern!r}")
    return raw


def context_from_introspection_headers(headers: Mapping[str, str]) -> Tuple[str, str]:
    """Ziehe (tenant_id, owner_id) AUSSCHLIESSLICH aus den server-asserted Introspektions-Headern.

    Deckt T10 (Minting-Spoof): liest NUR X-Jarvis-Tenant-Id / X-Jarvis-Owner-Id und NICHTS aus
    Request-Body oder anderen (client-kontrollierten) Feldern; kein Default-Bucket, kein
    Fallback. Fehlt ein Header -> fail-closed. Header-Namen case-insensitive.

    HINWEIS (Stufe-4-Grenze): dass diese Header vertrauenswürdig (edge-gestempelt, client-nicht-
    überschreibbar) sind, stellt das Edge sicher (GAP-G). Der Konsument erfindet keine Herkunft.
    """
    lower = {str(k).lower(): v for k, v in headers.items()}
    raw_tenant = lower.get(TENANT_HEADER.lower())
    raw_owner = lower.get(OWNER_HEADER.lower())
    if raw_tenant is None:
        raise VaultContextError(f"{TENANT_HEADER}: fehlt (kein server-asserted Tenant -> fail-closed)")
    if raw_owner is None:
        raise VaultContextError(f"{OWNER_HEADER}: fehlt (kein server-asserted Owner -> fail-closed)")
    return (
        normalize_context_value(raw_tenant, TENANT_HEADER),
        normalize_context_value(raw_owner, OWNER_HEADER),
    )


# --- GUC-Set (der eigentliche Konsument) --------------------------------------------------

def set_vault_context(cursor, tenant_id: str, owner_id: str) -> None:
    """Setze BEIDE RLS-GUCs transaction-local. Beide-oder-keiner.

    Validiert zuerst BEIDE Werte (wirft VOR jedem set_config -> nie Teilkontext), dann exakt
    zwei parametrisierte set_config-Statements. Wirft ein execute -> propagiert (der Aufrufer
    MUSS die Transaktion abbrechen; vault_transaction erledigt das).
    """
    t = normalize_context_value(tenant_id, "tenant_id")
    o = normalize_context_value(owner_id, "owner_id")
    # GUC-Name = fixes Literal; Wert = Parameter (%s). is_local=true -> transaction-local.
    cursor.execute("SELECT set_config(%s, %s, true)", (TENANT_GUC, t))
    cursor.execute("SELECT set_config(%s, %s, true)", (OWNER_GUC, o))


@contextmanager
def vault_transaction(conn, tenant_id: str, owner_id: str) -> Iterator[object]:
    """Transaktions-Scope mit gesetztem Vault-Kontext. Fail-closed bei jedem Fehler.

    Setzt beide GUCs, yieldet den Cursor. Bei JEDER BaseException (Validierung ODER set_config
    ODER Aufrufer-Fehler) -> conn.rollback() (Kontext + Aenderungen verworfen) und re-raise.
    Bei normalem Verlauf committet der Aufrufer NICHT hier - das Commit-/Checkin-Regime liegt
    beim Pool (reset_on_return='rollback'); dieser Scope garantiert nur, dass ein Fehler NIE
    mit halbem Kontext weiterläuft.
    """
    cursor = conn.cursor()
    try:
        set_vault_context(cursor, tenant_id, owner_id)
        yield cursor
    except BaseException:
        conn.rollback()
        raise


def checkin(conn) -> None:
    """Pool-Rückgabe-Vertrag: transaction-local Kontext beim Checkin verwerfen (rollback).

    Deckt die 'Connection zurück in den Pool -> Kontext gelöscht'-Invariante (ADR-0041
    Grenze-2). Ein realer Pool konfiguriert reset_on_return='rollback' (POOL_RESET_ON_RETURN);
    dieser Helper macht den Vertrag explizit für Pfade ohne solchen Pool.
    """
    conn.rollback()
