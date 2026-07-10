"""Embedding-Server-Client (bge-m3, Zone 52). Text -> 1024-dim Vektor via POST /embed.

Fail-soft + validiert: jeder HTTP-/Parse-/Shape-Fehler -> None (der Aufrufer fällt zurück bzw.
überspringt). NIE die Texte loggen (können Owner-Inhalt tragen). Bounded Timeout: der Read-Pfad
(KNN-Query-Vektor) läuft im Live-Turn -> darf NICHT hängen (fail-soft-Vertrag deckt Blockieren).

Der Server ist ZUSTANDSLOS (Design-Doc): er kennt keine tenant/owner, nur Text->Vektor. Die
RLS-Ablage macht der Vault-Write-/Reindex-Pfad, NICHT dieser Client. Der Server ist in Zone 52
(internal, egress-none) und NUR in-zone erreichbar (embedding-server.52.jarvis.internal:5001) ->
der Konsument (api-server/engine) muss auf net_22_52 dual-homed sein (Phase-3-Membership).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Endpunkt: am Deploy provisioniert (nicht hartkodiert), Default = der in-zone-FQDN. KEIN Secret ->
# reine env (kein _FILE). rstrip('/') gegen doppelten Slash.
_URL_ENV = "VAULT_EMBED_URL"
_DEFAULT_URL = "http://embedding-server.52.jarvis.internal:5001"

# Bounded: der Modell-Load ist gebacken (kein Runtime-Pull) -> ein Embed ist schnell; der Read-Pfad
# darf trotzdem nicht am toten Server hängen.
EMBED_TIMEOUT_S = 8.0
# MUSS zur gepinnten Spalte vector(1024) passen (mem_embdim_ck) -> Mismatch = fail-closed (kein
# Garbage-Vektor in die DDL). Deckungsgleich EMBED_DIMENSIONS_DEFAULT im vault_store.
EMBED_DIM = 1024
# = EMBED_MAX_BATCH des Servers (413 bei Überschreitung). Der Reindex chunked darunter; der
# Query-Vektor-Pfad schickt genau 1 Text.
EMBED_SERVER_MAX_BATCH = 16


@dataclass
class EmbedResult:
    """Antwort des Embedding-Servers. ``version`` pinnt embedding_version (KNN darf NUR gleiche
    Version vergleichen; ein Modell-/Version-Wechsel = Reindex-Trigger, ADR-0042 §C)."""
    provider: str
    model: str
    version: str
    dim: int
    vectors: List[List[float]]


def _endpoint() -> str:
    base = (os.environ.get(_URL_ENV, "").strip() or _DEFAULT_URL).rstrip("/")
    return base + "/embed"


def embed_texts(texts: List[str]) -> Optional[EmbedResult]:
    """Bettet bis EMBED_SERVER_MAX_BATCH Texte ein. Rückgabe None bei JEDEM Fehler (fail-soft) --
    NIE raise, NIE die Texte loggen. Grössere Batches muss der Aufrufer (reindex) chunken; hier
    hart begrenzt, weil der Server >16 mit 413 ablehnt.

    Shape-Validierung fail-closed: dim==1024, #Vektoren==#Texte, jeder Vektor len==1024 -- sonst
    None (kein halb-valider/garbage Vektor darf in die vector(1024)-Spalte)."""
    if not texts or not isinstance(texts, list):
        return None
    if len(texts) > EMBED_SERVER_MAX_BATCH:
        logger.warning("embed_texts: batch %d > %d -- Aufrufer muss chunken", len(texts), EMBED_SERVER_MAX_BATCH)
        return None
    try:
        body = json.dumps({"texts": texts}).encode("utf-8")
        req = urllib.request.Request(
            _endpoint(), data=body, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT_S) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 -- fail-soft; KEINE Texte/Antwort-Bodies loggen (Owner-Inhalt)
        logger.warning("embed_texts fehlgeschlagen (fail-soft): %s", type(e).__name__)
        return None
    try:
        vecs = out["embeddings"]
        dim = int(out["dim"])
        if dim != EMBED_DIM:
            raise ValueError(f"dim {dim} != {EMBED_DIM}")
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            raise ValueError("Vektor-Anzahl != Text-Anzahl")
        for v in vecs:
            if not isinstance(v, list) or len(v) != EMBED_DIM:
                raise ValueError("Vektor-Länge != 1024")
        return EmbedResult(
            provider=str(out["provider"]), model=str(out["model"]),
            version=str(out["version"]), dim=dim, vectors=vecs)
    except Exception as e:  # noqa: BLE001 -- fail-closed bei ungültiger Antwort-Form
        logger.warning("embed_texts Antwort ungültig (fail-closed): %s", type(e).__name__)
        return None


def to_pgvector_literal(vec: List[float]) -> str:
    """Ein Vektor als pgvector-Text-Literal '[0.1,0.2,...]' für den Query-Parameter (%s::vector).
    KEIN pgvector-python-Adapter nötig -> keine neue Dependency. Nur endliche Floats (fail-closed:
    NaN/Inf würden die Distanz vergiften)."""
    import math
    parts = []
    for x in vec:
        f = float(x)
        if not math.isfinite(f):
            raise ValueError("nicht-endlicher Vektor-Wert")
        parts.append(repr(f))
    return "[" + ",".join(parts) + "]"
