"""Tests für den Vault dark-wire (tools/vault/vault_wiring.py) -- Live-Scheibe Teil 3.

Integrationsgrenze: kein psycopg, kein Deploy. Alles mit Fakes (ContextVar-Identität,
Fake-Pool, Fake-VaultStore, Flag-Monkeypatch). Prüft die load-bearing Invarianten:
shadow-not-replace, fail-soft, fg-only+resolved-owner, Flag-Leiter (PLUMBING vs WRITE).
"""

import json

import pytest

from tools.vault import vault_wiring as vw


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

def _flags(monkeypatch, *, plumbing=False, write=False):
    table = {"plumbing_enabled": plumbing, "write_enabled": write}
    monkeypatch.setattr(vw, "_vault_flag", lambda name: table.get(name, False))


def _foreground(monkeypatch, origin="foreground"):
    monkeypatch.setattr("tools.write_approval.current_origin", lambda: origin)


class FakePool:
    def __init__(self):
        self.borrowed = 0
        self.returned = 0
        self._conn = object()

    def getconn(self, timeout=None):
        self.borrowed += 1
        return self._conn

    def putconn(self, conn):
        self.returned += 1


class FakeStore:
    last_req = None
    raise_on_write = False

    def __init__(self, connect):
        self.connect = connect

    def write(self, req):
        FakeStore.last_req = req
        if FakeStore.raise_on_write:
            raise RuntimeError("write boom")
        from tools.vault.vault_store import WriteResult, STATUS_WRITTEN
        return WriteResult(status=STATUS_WRITTEN, persisted=True, lifecycle_status="confirmed")


@pytest.fixture(autouse=True)
def _reset_fakestore():
    FakeStore.last_req = None
    FakeStore.raise_on_write = False
    yield


def _patch_write_backend(monkeypatch, pool=None):
    pool = pool or FakePool()
    monkeypatch.setattr("tools.vault.db_runtime.get_vault_pool", lambda: pool)
    monkeypatch.setattr("tools.vault.vault_store.VaultStore", FakeStore)
    return pool


OK = {"success": True}


# ---------------------------------------------------------------------------
# Flag-Leiter + Identität
# ---------------------------------------------------------------------------

def test_flags_default_off(monkeypatch):
    # _vault_flag fällt bei fehlender Config auf False -> Pfad inaktiv.
    monkeypatch.setattr(vw, "_vault_flag", lambda name: False)
    assert vw.vault_path_active() is False
    assert vw.vault_write_enabled() is False


def test_write_implies_path_active(monkeypatch):
    _flags(monkeypatch, write=True)
    assert vw.vault_path_active() is True


def test_identity_set_get_reset():
    assert vw.get_vault_write_identity() is None
    tok = vw.set_vault_write_identity("t1", "o1")
    assert vw.get_vault_write_identity() == ("t1", "o1")
    vw.reset_vault_write_identity(tok)
    assert vw.get_vault_write_identity() is None


@pytest.mark.parametrize("t,o", [("", "o"), ("t", ""), (None, "o"), ("t", 5)])
def test_identity_rejects_bad_values(t, o):
    tok = vw.set_vault_write_identity(t, o)
    assert vw.get_vault_write_identity() is None
    vw.reset_vault_write_identity(tok)


def test_identity_contextmanager_scopes():
    with vw.vault_write_identity("t1", "o1"):
        assert vw.get_vault_write_identity() == ("t1", "o1")
    assert vw.get_vault_write_identity() is None


# ---------------------------------------------------------------------------
# no-op-Bedingungen (kein VaultStore-Kontakt)
# ---------------------------------------------------------------------------

def test_noop_when_path_inactive(monkeypatch):
    _flags(monkeypatch, plumbing=False, write=False)
    _foreground(monkeypatch)
    pool = _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK) is None
    assert pool.borrowed == 0 and FakeStore.last_req is None


def test_noop_on_remove(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("remove", "memory", "x", store_result=OK) is None
    assert FakeStore.last_req is None


def test_noop_without_content(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("add", "memory", "", store_result=OK) is None
    assert FakeStore.last_req is None


def test_noop_when_store_write_failed(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("add", "memory", "Notiz",
                                     store_result={"success": False}) is None
    assert FakeStore.last_req is None


def test_noop_when_not_foreground(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch, origin="background_review")
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK) is None
    assert FakeStore.last_req is None


def test_noop_without_identity(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    # KEINE Identität gesetzt.
    assert vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK) is None
    assert FakeStore.last_req is None


# ---------------------------------------------------------------------------
# PLUMBING (Dry-Run) vs WRITE
# ---------------------------------------------------------------------------

def test_plumbing_dry_run_no_backend(monkeypatch):
    _flags(monkeypatch, plumbing=True, write=False)
    _foreground(monkeypatch)

    def _boom():
        raise AssertionError("Pool im PLUMBING-Dry-Run NICHT anfassen")
    monkeypatch.setattr("tools.vault.db_runtime.get_vault_pool", _boom)
    with vw.vault_write_identity("t1", "o1"):
        out = vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK)
    assert out == "plumbing_dry_run"


def test_write_mode_calls_vaultstore_with_correct_request(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    pool = _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("tenant-a", "owner-a"):
        out = vw.vault_shadow_write("add", "memory", "Termin mit Anna", store_result=OK)
    assert out == "written"
    # borrow + return (auch Reihenfolge egal, beide genau einmal).
    assert pool.borrowed == 1 and pool.returned == 1
    req = FakeStore.last_req
    assert req is not None
    assert (req.tenant_id, req.owner_id) == ("tenant-a", "owner-a")
    assert req.origin == "foreground"
    from tools.vault.vault_store import SOURCE_FOREGROUND_OWNER, TRUST_TRUSTED
    assert req.source == SOURCE_FOREGROUND_OWNER
    assert req.source_table == "owner_memory"
    assert req.trust_level == TRUST_TRUSTED
    assert req.summary_redacted == "Termin mit Anna"
    assert req.taint == {"from_untrusted_inbound": False}
    assert req.raw_bytes is None
    # PHASE-1 source_id = source_hash = Content-Hash.
    assert req.source_id == req.source_hash and len(req.source_id) == 64


def test_write_mode_user_target_maps_to_user_profile(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t", "o"):
        vw.vault_shadow_write("add", "user", "mag JSON", store_result=OK)
    assert FakeStore.last_req.source_table == "user_profile"


# ---------------------------------------------------------------------------
# fail-soft: Backend-Fehler darf NIE hochblubbern
# ---------------------------------------------------------------------------

def test_failsoft_on_write_raise(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    pool = _patch_write_backend(monkeypatch)
    FakeStore.raise_on_write = True
    with vw.vault_write_identity("t", "o"):
        # kein raise -> None; conn wurde trotzdem zurückgegeben (finally).
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK) is None
    assert pool.returned == 1


def test_failsoft_on_pool_unavailable(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)

    def _boom():
        from tools.vault.db_runtime import VaultPoolUnavailable
        raise VaultPoolUnavailable("keine DSN")
    monkeypatch.setattr("tools.vault.db_runtime.get_vault_pool", _boom)
    with vw.vault_write_identity("t", "o"):
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK) is None


def test_failsoft_on_getconn_timeout(monkeypatch):
    """getconn-Timeout (toter Pool) darf den Turn nicht hängen -> Exception -> fail-soft skip."""
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)

    class TimeoutPool:
        borrowed = 0
        returned = 0

        def getconn(self, timeout=None):
            TimeoutPool.borrowed += 1
            raise TimeoutError("pool timeout")

        def putconn(self, conn):
            TimeoutPool.returned += 1

    pool = TimeoutPool()
    monkeypatch.setattr("tools.vault.db_runtime.get_vault_pool", lambda: pool)
    monkeypatch.setattr("tools.vault.vault_store.VaultStore", FakeStore)
    with vw.vault_write_identity("t", "o"):
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK) is None
    # nichts geborgt-gehalten: getconn warf VOR dem try -> putconn NICHT gerufen, VaultStore nie.
    assert pool.returned == 0 and FakeStore.last_req is None


def test_noop_when_success_missing_or_ambiguous(monkeypatch):
    """`is True`-Strenge: fehlendes/mehrdeutiges success-Feld -> KEIN Shadow (fail-safe)."""
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t", "o"):
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result={}) is None
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result={"success": 1}) is None
        assert vw.vault_shadow_write("add", "memory", "Notiz", store_result="ok") is None
    assert FakeStore.last_req is None


# ---------------------------------------------------------------------------
# memory_tool-Hook: der Vault-Write beeinflusst den file-backed Turn NIE
# ---------------------------------------------------------------------------

def test_memory_tool_hook_is_fail_soft(monkeypatch):
    import tools.memory_tool as mt

    called = {}

    def _boom(action, target, content, *, store_result=None):
        called["args"] = (action, target, content, store_result)
        raise RuntimeError("vault kaputt")

    monkeypatch.setattr("tools.vault.vault_wiring.vault_shadow_write", _boom)

    class FakeStore2:
        def add(self, target, content):
            return {"success": True, "target": target, "content": content}

    out = mt.memory_tool("add", "memory", content="hallo", store=FakeStore2())
    body = json.loads(out)
    # Der file-backed Write ist autoritativ + unbeeinflusst, trotz Vault-Exception.
    assert body.get("success") is True
    assert called["args"][0] == "add" and called["args"][3] == {"success": True, "target": "memory", "content": "hallo"}
