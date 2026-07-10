"""Owner-scoped Embed-Reindex (Fläche-B Phase 3a). Füllt den ``embedding``-Vektor confirmed,
embed-fähiger memory_items-Zeilen -- SELECT embed-eligible -> POST an den Embedding-Server -> UPDATE.

OWNER-SCOPED, NICHT Cross-Owner (Advisor 2026-07-11): der Reindex läuft pro (tenant_id, owner_id)
unter vault_transaction (beide RLS-GUCs gesetzt), NIE als BYPASSRLS-Batch über alle Owner. Bei einem
Single-User-Vault ist das owner-primary; die Cross-Owner-Variante (BYPASSRLS/SET-DISTINCT-owner) wäre
genau die RLS-Kollaps-/Trust-Konzentrations-Fläche, die die TRUSTED-SQL-ONLY-Doktrin verbietet ->
deferred bis Multi-User real ist. Korrektheit lebt im Aufrufer (der die aufgelöste Owner-Identität
liefert), nicht in einer abstrakten Cross-Owner-Abstraktion.

Embed-Gate (0001_memory_items.sql:115): ein Vektor darf NUR existieren, wenn redaction_state='applied'
AND sanitization_state='applied'. Der SELECT filtert exakt darauf (+ confirmed + nicht invalidiert +
embedding IS NULL + reindex_state IN current/stale); der UPDATE re-assertet die Gate-Bedingung
(defense-in-depth gegen eine Nebenläufigkeits-Änderung zwischen SELECT und UPDATE).

Das Embedding passiert ASYNCHRON zum Choke (STUFE5_BUILD_SPEC:374): dieser Lauf ist ein
Idle/Backfill-Job, NICHT im Live-Write-Turn. Fail-soft: toter/langsamer Embedding-Server -> der Lauf
bricht sauber ab (available=False), NIE eine Exception nach aussen, NIE ein Garbage-Vektor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools.vault.embed_client import EMBED_SERVER_MAX_BATCH, embed_texts, to_pgvector_literal
from tools.vault.vault_context import VaultContextError, normalize_context_value, vault_transaction

logger = logging.getLogger(__name__)

REINDEX_MAX_ROWS_DEFAULT = 500   # Backfill-Deckel je Aufruf (bounded; ein Idle-Job, kein Endlos-Loop)

# Embed-eligible: die exakte Menge, die einen Vektor bekommen DARF (embed-gate + recall-fähig).
# reindex_state IN ('current','stale') -- NICHT 'superseded' (abgelöste Zeile bekommt keinen Vektor).
_SELECT_ELIGIBLE = (
    "SELECT id, summary_redacted FROM public.memory_items "
    "WHERE lifecycle_status = 'confirmed' "
    "AND redaction_state = 'applied' AND sanitization_state = 'applied' "
    "AND embedding IS NULL "
    "AND reindex_state IN ('current', 'stale') "
    "AND deleted_at IS NULL AND quarantined_at IS NULL AND superseded_at IS NULL "
    "AND summary_redacted IS NOT NULL AND summary_redacted <> '' "
    "ORDER BY created_at ASC LIMIT %s"
)

# UPDATE re-assertet den embed-gate + embedding IS NULL (idempotent/race-safe). embedding_version
# vom Server (KNN vergleicht NUR gleiche Version). id im WHERE; tenant/owner trägt die RLS-Policy.
_UPDATE_EMBEDDING = (
    "UPDATE public.memory_items "
    "SET embedding = %s::vector, embedding_provider = %s, embedding_model = %s, "
    "    embedding_version = %s, reindex_state = 'current' "
    "WHERE id = %s AND embedding IS NULL "
    "AND lifecycle_status = 'confirmed' "
    "AND redaction_state = 'applied' AND sanitization_state = 'applied'"
)


@dataclass
class ReindexResult:
    status: str          # reindexed | nothing_eligible | embed_unavailable | error
    embedded: int = 0    # Zeilen, die einen Vektor bekamen
    scanned: int = 0     # Zeilen, die als eligible gelesen wurden
    available: bool = True
    message: str = ""


def reindex_owner(tenant_id: str, owner_id: str, connect: Callable[[], Any], *,
                  batch: int = EMBED_SERVER_MAX_BATCH, max_rows: int = REINDEX_MAX_ROWS_DEFAULT) -> ReindexResult:
    """Backfillt Vektoren für EINEN (tenant, owner). ``connect`` = zero-arg -> DB-API-Connection
    (wie VaultStore). Schleife: SELECT eligible (eigene Txn) -> embed (Netz, KEINE offene Txn) ->
    UPDATE (eigene Txn, commit) -- bis nichts mehr eligible ODER max_rows. Wirft NIE."""
    try:
        tenant = normalize_context_value(tenant_id, "tenant_id")
        owner = normalize_context_value(owner_id, "owner_id")
    except VaultContextError as e:
        return ReindexResult(status="error", available=False, message=str(e))

    batch = max(1, min(int(batch) if isinstance(batch, int) else EMBED_SERVER_MAX_BATCH, EMBED_SERVER_MAX_BATCH))
    embedded = 0
    scanned = 0
    conn = connect()
    try:
        while embedded < max_rows:
            # (1) eligible lesen -- eigene Txn, KEIN Halten während des Netz-Embeds.
            with vault_transaction(conn, tenant, owner) as cur:
                cur.execute(_SELECT_ELIGIBLE, (batch,))
                rows = cur.fetchall()
            if not rows:
                break
            scanned += len(rows)
            ids = [r[0] for r in rows]
            summaries = [r[1] for r in rows]

            # (2) einbetten -- Netz, fail-soft. None -> sauberer Abbruch (kein Endlos-Loop).
            res = embed_texts(summaries)
            if res is None:
                logger.warning("reindex: Embedding-Server nicht verfügbar/ungültig -> Abbruch (fail-soft)")
                return ReindexResult(status="embed_unavailable", embedded=embedded,
                                     scanned=scanned, available=False)

            # (3) UPDATE je Zeile -- eigene Txn, commit. Der UPDATE-Guard (embedding IS NULL + gate)
            #     macht es idempotent + race-safe.
            with vault_transaction(conn, tenant, owner) as cur:
                for row_id, vec in zip(ids, res.vectors):
                    try:
                        lit = to_pgvector_literal(vec)
                    except ValueError:
                        continue  # nicht-endlicher Vektor -> Zeile überspringen (kein Garbage)
                    cur.execute(_UPDATE_EMBEDDING,
                                (lit, res.provider, res.model, res.version, row_id))
                    rc = getattr(cur, "rowcount", 0)
                    if isinstance(rc, int) and rc > 0:
                        embedded += 1
            conn.commit()

            if len(rows) < batch:
                break  # letzte Teilcharge -> fertig
    except BaseException as e:  # noqa: BLE001 -- fail-soft
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error("reindex fehlgeschlagen (rolled back): %s", type(e).__name__)
        return ReindexResult(status="error", embedded=embedded, scanned=scanned,
                             available=False, message="Reindex nicht abgeschlossen")

    status = "reindexed" if embedded > 0 else "nothing_eligible"
    return ReindexResult(status=status, embedded=embedded, scanned=scanned, available=True)
