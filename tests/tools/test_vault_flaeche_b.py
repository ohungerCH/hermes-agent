"""Tests für Fläche B (semantische KNN-Recall): embed_client, reindex, KNN-Zweig in recall().

psycopg + der Embedding-Server sind NICHT im Test verfügbar -> Fakes/Monkeypatch. Die echte KNN-SQL
gegen die DDL + der echte /embed-Kontrakt werden am Deploy (psql-Selbsttest Szenario J + Live-E2E)
bewiesen -- hier: die Routing-/fail-soft-/owner-scoped-Invarianten.
"""
import json

import pytest

from tools.vault import embed_client as ec
from tools.vault import reindex as rx
from tools.vault import vault_store as vs
from tools.vault.vault_store import MemoryRecall, VaultStore
from tools.vault.vault_context import TENANT_GUC, OWNER_GUC


# ---------------------------------------------------------------------------
# embed_client
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _ok_payload(n, dim=1024):
    return {"provider": "local-bge-m3", "model": "BAAI/bge-m3", "version": "v1",
            "dim": dim, "embeddings": [[0.01] * dim for _ in range(n)]}


def test_embed_success(monkeypatch):
    monkeypatch.setattr(ec.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(_ok_payload(2)))
    res = ec.embed_texts(["a", "b"])
    assert res is not None and res.dim == 1024 and res.version == "v1"
    assert len(res.vectors) == 2 and len(res.vectors[0]) == 1024


def test_embed_http_error_is_failsoft(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")
    monkeypatch.setattr(ec.urllib.request, "urlopen", boom)
    assert ec.embed_texts(["a"]) is None   # fail-soft, kein raise


def test_embed_wrong_dim_is_failclosed(monkeypatch):
    monkeypatch.setattr(ec.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(_ok_payload(1, dim=512)))
    assert ec.embed_texts(["a"]) is None   # dim != 1024 -> kein Garbage in vector(1024)


def test_embed_count_mismatch_is_failclosed(monkeypatch):
    p = _ok_payload(1)          # nur 1 Vektor
    monkeypatch.setattr(ec.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(p))
    assert ec.embed_texts(["a", "b"]) is None   # 2 Texte, 1 Vektor -> None


def test_embed_batch_over_server_cap_refused():
    assert ec.embed_texts(["x"] * (ec.EMBED_SERVER_MAX_BATCH + 1)) is None


def test_pgvector_literal_finite_and_nan():
    assert ec.to_pgvector_literal([0.5, -0.25]) == "[0.5,-0.25]"
    with pytest.raises(ValueError):
        ec.to_pgvector_literal([0.1, float("nan")])   # NaN vergiftet die Distanz -> fail-closed


# ---------------------------------------------------------------------------
# reindex (owner-scoped) -- Fake-Conn mit Multi-Batch-fetchall + rowcount
# ---------------------------------------------------------------------------

class _RxCursor:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = -1
    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
        if "UPDATE public.memory_items" in sql:
            self.rowcount = self._conn.update_rowcount
    def fetchall(self):
        return self._conn.select_batches.pop(0) if self._conn.select_batches else []


class _RxConn:
    def __init__(self, *, select_batches, update_rowcount=1):
        self.executed = []
        self.select_batches = list(select_batches)
        self.update_rowcount = update_rowcount
        self.committed = 0
    def cursor(self):
        return _RxCursor(self)
    def commit(self):
        self.committed += 1
    def rollback(self):
        pass


def test_reindex_embeds_eligible(monkeypatch):
    # 1. Batch: 2 eligible Zeilen; 2. Batch: leer -> Schleife endet.
    conn = _RxConn(select_batches=[[("id1", "Notiz A"), ("id2", "Notiz B")], []])
    monkeypatch.setattr(rx, "embed_texts", lambda texts: ec.EmbedResult(
        "local-bge-m3", "BAAI/bge-m3", "v1", 1024, [[0.01] * 1024 for _ in texts]))
    res = rx.reindex_owner("t1", "o1", connect=lambda: conn)
    assert res.status == "reindexed" and res.embedded == 2 and res.scanned == 2 and res.available
    # owner-scoped: die GUCs wurden auf t1/o1 gesetzt (NICHT cross-owner)
    gucs = [p for (s, p) in conn.executed if "set_config" in s]
    assert (TENANT_GUC, "t1") in gucs and (OWNER_GUC, "o1") in gucs
    # UPDATE mit ::vector-Literal + version
    ups = [(s, p) for (s, p) in conn.executed if "UPDATE public.memory_items" in s]
    assert len(ups) == 2 and ups[0][1][3] == "v1"   # embedding_version-Param


def test_reindex_nothing_eligible(monkeypatch):
    conn = _RxConn(select_batches=[[]])
    monkeypatch.setattr(rx, "embed_texts", lambda texts: None)  # darf nie gerufen werden
    res = rx.reindex_owner("t1", "o1", connect=lambda: conn)
    assert res.status == "nothing_eligible" and res.embedded == 0 and res.available


def test_reindex_embed_unavailable_is_failsoft(monkeypatch):
    conn = _RxConn(select_batches=[[("id1", "Notiz")]])
    monkeypatch.setattr(rx, "embed_texts", lambda texts: None)   # Server weg
    res = rx.reindex_owner("t1", "o1", connect=lambda: conn)
    assert res.status == "embed_unavailable" and res.available is False and res.embedded == 0


def test_reindex_bad_anchor_is_error():
    res = rx.reindex_owner("", "o1", connect=lambda: None)
    assert res.status == "error" and res.available is False


# ---------------------------------------------------------------------------
# KNN-Zweig in recall() (Mode-Dispatch + fail-soft-Fallback)
# ---------------------------------------------------------------------------

class _RecallConn:
    def __init__(self, rows=()):
        self.executed = []
        self.rows = list(rows)
    def cursor(self):
        return _RecallCursor(self)
    def rollback(self):
        pass
    def commit(self):
        pass


class _RecallCursor:
    def __init__(self, conn):
        self._conn = conn
    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
    def fetchall(self):
        return list(self._conn.rows)


def _knn_row():
    return ("item-h1", "owner_memory", "h1", "Owner mag Kaffee", "2026-07-11T00:00:00Z", "personal_low", False)


def _select_sql(conn):
    return [(s, p) for (s, p) in conn.executed if "FROM public.memory_items" in s][0]


def test_recall_knn_uses_knn_sql(monkeypatch):
    """mode='knn' + Server liefert Vektor -> KNN-SQL (embedding <=> + embedding_version), mode_used='knn'."""
    conn = _RecallConn(rows=[_knn_row()])
    monkeypatch.setattr("tools.vault.embed_client.embed_texts", lambda texts: ec.EmbedResult(
        "local-bge-m3", "BAAI/bge-m3", "v1", 1024, [[0.02] * 1024]))
    res = VaultStore(connect=lambda: conn).recall(
        MemoryRecall(owner_id="o1", tenant_id="t1", query="kaffee", mode="knn"))
    sql, params = _select_sql(conn)
    assert "embedding <=> %s::vector" in sql and "embedding_version = %s" in sql
    assert "embedding IS NOT NULL" in sql
    assert params[0] == "v1" and params[1].startswith("[") and params[2] == vs.RECALL_LIMIT_DEFAULT
    assert res.mode_used == "knn" and res.available is True and res.status == vs.STATUS_RECALLED


def test_recall_knn_fallback_when_embed_down(monkeypatch):
    """mode='knn' + Server weg (embed None) -> fail-soft FALLBACK auf tsvector, beobachtbar."""
    conn = _RecallConn(rows=[])
    monkeypatch.setattr("tools.vault.embed_client.embed_texts", lambda texts: None)
    res = VaultStore(connect=lambda: conn).recall(
        MemoryRecall(owner_id="o1", tenant_id="t1", query="kaffee", mode="knn"))
    sql, _ = _select_sql(conn)
    assert "websearch_to_tsquery" in sql and "embedding <=>" not in sql   # tsvector-SQL lief
    assert res.mode_used == "knn_fallback_tsvector" and res.available is True


def test_recall_tsvector_default_mode(monkeypatch):
    """Default (kein mode) = tsvector; der Embedding-Server wird NICHT gerufen."""
    called = {"n": 0}
    monkeypatch.setattr("tools.vault.embed_client.embed_texts",
                        lambda texts: called.__setitem__("n", called["n"] + 1))
    conn = _RecallConn(rows=[_knn_row()])
    res = VaultStore(connect=lambda: conn).recall(MemoryRecall(owner_id="o1", tenant_id="t1", query="kaffee"))
    assert res.mode_used == "tsvector" and called["n"] == 0


# ---------------------------------------------------------------------------
# Verdrahtung: config-Mode-Default + Reindex-Trigger
# ---------------------------------------------------------------------------

def test_shadow_recall_mode_from_config(monkeypatch):
    """vault_shadow_recall(mode=None) nimmt den server-seitigen config-Default (kein model-facing
    Param -> kein Tool-Call-Site-Drift). recall_mode='knn' -> die Anfrage geht als knn an den Store."""
    from tools.vault import vault_wiring as vw
    from tools.vault import db_runtime
    monkeypatch.setattr(vw, "vault_recall_enabled", lambda: True)
    monkeypatch.setattr(vw, "vault_recall_mode", lambda: "knn")
    monkeypatch.setattr(vw, "get_vault_write_identity", lambda: ("t1", "o1"))
    import tools.write_approval as wa
    monkeypatch.setattr(wa, "current_origin", lambda: "assistant_tool")
    cap = {}
    class _Store:
        def __init__(self, connect): pass
        def recall(self, req):
            cap["mode"] = req.mode
            from tools.vault.vault_store import RecallResult
            return RecallResult(status="recall_empty", items=[], available=True, mode_used=req.mode)
    class _Pool:
        def getconn(self, timeout=None): return object()
        def putconn(self, c): pass
    monkeypatch.setattr("tools.vault.vault_store.VaultStore", _Store)
    monkeypatch.setattr(db_runtime, "get_vault_pool", lambda: _Pool())
    out = vw.vault_shadow_recall("kaffee")
    assert cap["mode"] == "knn" and out["available"] is True and out["mode_used"] == "knn"


def test_vault_reindex_owner_trigger(monkeypatch):
    """vault_reindex_owner: Admin-Trigger holt den Pool + ruft reindex_owner owner-scoped, gibt ein
    fail-soft Dict zurück (nie raise)."""
    from tools.vault import vault_wiring as vw
    from tools.vault import db_runtime, reindex as rx
    class _Pool:
        def getconn(self, timeout=None): return object()
        def putconn(self, c): pass
    monkeypatch.setattr(db_runtime, "get_vault_pool", lambda: _Pool())
    monkeypatch.setattr(rx, "reindex_owner",
                        lambda t, o, connect, max_rows=500: rx.ReindexResult(status="reindexed", embedded=3, scanned=3))
    out = vw.vault_reindex_owner("t1", "o1")
    assert out["status"] == "reindexed" and out["embedded"] == 3 and out["available"] is True


# ---------------------------------------------------------------------------
# Hybrid-Recall (tsvector-Boden + knn-Ranking, union+dedup)
# ---------------------------------------------------------------------------

class _HybridCursor:
    def __init__(self, conn):
        self._conn = conn; self._last = None
    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params)); self._last = sql
    def fetchall(self):
        if "embedding <=>" in (self._last or ""):
            return list(self._conn.knn_rows)
        if "websearch_to_tsquery" in (self._last or ""):
            return list(self._conn.ts_rows)
        return []


class _HybridConn:
    def __init__(self, knn_rows=(), ts_rows=()):
        self.executed = []; self.knn_rows = list(knn_rows); self.ts_rows = list(ts_rows)
    def cursor(self):
        return _HybridCursor(self)
    def rollback(self): pass
    def commit(self): pass


def _row(sid, summary="x"):
    return (f"item-{sid}", "owner_memory", sid, summary, "2026-07-11T00:00:00Z", "personal_low", False)


def test_recall_hybrid_merges_knn_and_tsvector(monkeypatch):
    """Hybrid: knn zuerst (semantisch), dann tsvector-only; dedup per stabiler item_id."""
    conn = _HybridConn(knn_rows=[_row("a"), _row("b")], ts_rows=[_row("b"), _row("c")])
    monkeypatch.setattr("tools.vault.embed_client.embed_texts", lambda texts: ec.EmbedResult(
        "local-bge-m3", "BAAI/bge-m3", "v1", 1024, [[0.02] * 1024]))
    res = VaultStore(connect=lambda: conn).recall(
        MemoryRecall(owner_id="o1", tenant_id="t1", query="kaffee", mode="hybrid"))
    ids = [it.source_id for it in res.items]
    assert ids == ["a", "b", "c"]          # knn a,b zuerst; b dedupt; c (tsvector-only) gefangen
    assert res.mode_used == "hybrid" and res.available is True
    # beide SELECTs liefen (KNN + tsvector)
    sqls = " ".join(s for (s, p) in conn.executed)
    assert "embedding <=>" in sqls and "websearch_to_tsquery" in sqls


def test_recall_hybrid_tsvector_only_when_embed_down(monkeypatch):
    """Server weg (embed None) -> nur tsvector-Boden, KEIN KNN-SELECT, mode_used='hybrid_tsvector_only'."""
    conn = _HybridConn(knn_rows=[_row("a")], ts_rows=[_row("c")])
    monkeypatch.setattr("tools.vault.embed_client.embed_texts", lambda texts: None)
    res = VaultStore(connect=lambda: conn).recall(
        MemoryRecall(owner_id="o1", tenant_id="t1", query="kaffee", mode="hybrid"))
    ids = [it.source_id for it in res.items]
    assert ids == ["c"]                    # NUR tsvector (neue Zeile sichtbar), knn nicht gelaufen
    assert res.mode_used == "hybrid_tsvector_only"
    sqls = " ".join(s for (s, p) in conn.executed)
    assert "embedding <=>" not in sqls and "websearch_to_tsquery" in sqls


def test_recall_hybrid_new_row_never_invisible(monkeypatch):
    """Kern-Garantie: eine NEUE, noch nicht embeddete Zeile (nur im tsvector-Ergebnis, NICHT im knn)
    wird von hybrid gefunden -- der KNN-`embedding IS NOT NULL`-Filter kann sie nicht verstecken."""
    conn = _HybridConn(knn_rows=[], ts_rows=[_row("neu")])   # knn findet nichts (kein Vektor), tsvector schon
    monkeypatch.setattr("tools.vault.embed_client.embed_texts", lambda texts: ec.EmbedResult(
        "local-bge-m3", "BAAI/bge-m3", "v1", 1024, [[0.02] * 1024]))
    res = VaultStore(connect=lambda: conn).recall(
        MemoryRecall(owner_id="o1", tenant_id="t1", query="kaffee", mode="hybrid"))
    assert [it.source_id for it in res.items] == ["neu"] and res.mode_used == "hybrid"


def test_trigger_async_embed_gated(monkeypatch):
    """_trigger_async_embed feuert NUR bei recall_mode knn/hybrid (kein Embed wenn nie semantisch
    gelesen wird) -- Daemon-Thread, fail-soft. Via Event robust getestet."""
    from tools.vault import vault_wiring as vw
    import threading, time
    ev = threading.Event(); calls = []
    def fake_reindex(t, o, **kw):
        calls.append((t, o)); ev.set()
        return {"status": "reindexed", "embedded": 1, "available": True}
    monkeypatch.setattr(vw, "vault_reindex_owner", fake_reindex)
    # OFF (tsvector): kein Thread, kein Embed
    monkeypatch.setattr(vw, "vault_recall_mode", lambda: "tsvector")
    vw._trigger_async_embed("t1", "o1"); time.sleep(0.15)
    assert calls == []
    # ON (hybrid): Thread feuert
    monkeypatch.setattr(vw, "vault_recall_mode", lambda: "hybrid")
    vw._trigger_async_embed("t1", "o1")
    assert ev.wait(timeout=3) and ("t1", "o1") in calls


def test_shadow_recall_hybrid_not_downgraded(monkeypatch):
    """Regressions-Guard (Live-E2E-Befund 2026-07-11): _do_vault_recall/vault_shadow_recall dürfen
    'hybrid' NICHT auf tsvector downgraden. Die Unit-Tests riefen store.recall DIREKT (umgingen die
    Wiring-Mode-Validierung) -- nur der Live-Turn fing das fehlende 'hybrid' in der Allowlist."""
    from tools.vault import vault_wiring as vw
    from tools.vault import db_runtime
    monkeypatch.setattr(vw, "vault_recall_enabled", lambda: True)
    monkeypatch.setattr(vw, "vault_recall_mode", lambda: "hybrid")
    monkeypatch.setattr(vw, "get_vault_write_identity", lambda: ("t1", "o1"))
    import tools.write_approval as wa
    monkeypatch.setattr(wa, "current_origin", lambda: "assistant_tool")
    cap = {}
    class _Store:
        def __init__(self, connect): pass
        def recall(self, req):
            cap["mode"] = req.mode
            from tools.vault.vault_store import RecallResult
            return RecallResult(status="recall_empty", items=[], available=True, mode_used=req.mode)
    class _Pool:
        def getconn(self, timeout=None): return object()
        def putconn(self, c): pass
    monkeypatch.setattr("tools.vault.vault_store.VaultStore", _Store)
    monkeypatch.setattr(db_runtime, "get_vault_pool", lambda: _Pool())
    assert vw.vault_shadow_recall("kaffee").get("mode_used") == "hybrid"   # config-Default
    assert cap["mode"] == "hybrid"
    cap.clear()
    assert vw.vault_shadow_recall("kaffee", mode="hybrid").get("mode_used") == "hybrid"  # explizit
    assert cap["mode"] == "hybrid"
