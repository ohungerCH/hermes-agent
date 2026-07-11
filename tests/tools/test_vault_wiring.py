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

def _flags(monkeypatch, *, plumbing=False, write=False, recall_mode="tsvector"):
    table = {"plumbing_enabled": plumbing, "write_enabled": write}
    monkeypatch.setattr(vw, "_vault_flag", lambda name: table.get(name, False))
    # recall_mode explizit pinnen: sonst erbt der Test den DEFAULT_CONFIG-Seed
    # ('hybrid', §8b-Härtung) und der async-Embed-Daemon feuert nach dem Write
    # -> borgt eine zweite Connection und bricht die borrowed==1-Invariante.
    monkeypatch.setattr(vw, "vault_recall_mode", lambda: recall_mode)


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
    last_inv = None
    raise_on_write = False
    invalidate_persisted = True   # False -> Supersede-fail-Pfad (replace darf dann NICHT einfügen)

    def __init__(self, connect):
        self.connect = connect

    def write(self, req):
        FakeStore.last_req = req
        if FakeStore.raise_on_write:
            raise RuntimeError("write boom")
        from tools.vault.vault_store import WriteResult, STATUS_WRITTEN
        return WriteResult(status=STATUS_WRITTEN, persisted=True, lifecycle_status="confirmed")

    def invalidate(self, inv):
        FakeStore.last_inv = inv
        from tools.vault.vault_store import WriteResult, STATUS_INVALIDATED, STATUS_ERROR
        if not FakeStore.invalidate_persisted:
            return WriteResult(status=STATUS_ERROR, persisted=False)
        return WriteResult(status=STATUS_INVALIDATED, persisted=True, memory_item_written=True)


@pytest.fixture(autouse=True)
def _reset_fakestore():
    FakeStore.last_req = None
    FakeStore.last_inv = None
    FakeStore.raise_on_write = False
    FakeStore.invalidate_persisted = True
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


def test_default_config_seeds_vault_live_state():
    """§8b-Härtung: der versionierte DEFAULT_CONFIG trägt den Live-Zustand des
    Vault (all-on + hybrid). Verhindert, dass ein Volume-Reseed den Vault still
    dunkel schaltet -- und dass jemand den Seed versehentlich entfernt.
    """
    from hermes_cli.config import DEFAULT_CONFIG, _deep_merge, cfg_get

    seed = DEFAULT_CONFIG.get("vault")
    assert seed == {
        "plumbing_enabled": True,
        "write_enabled": True,
        "recall_enabled": True,
        "recall_mode": "hybrid",
    }

    # Self-Heal-Beweis: eine User-config.yaml OHNE vault-Block (wie nach einem
    # Reseed) erbt die Seed-Werte über den Deep-Merge, den load_config() fährt.
    merged = _deep_merge(DEFAULT_CONFIG, {"model": {"default": "x"}})
    assert cfg_get(merged, "vault", "recall_mode") == "hybrid"
    assert cfg_get(merged, "vault", "write_enabled") is True

    # Reversibilität: ein User-Override gewinnt weiterhin pro Key (Live-Flip).
    off = _deep_merge(DEFAULT_CONFIG, {"vault": {"recall_mode": "tsvector"}})
    assert cfg_get(off, "vault", "recall_mode") == "tsvector"
    assert cfg_get(off, "vault", "write_enabled") is True  # unberührter Key bleibt


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


def test_noop_on_remove_without_old_entry(monkeypatch):
    """remove OHNE Alt-Eintrag (die zu löschende Identität fehlt) -> fail-safe No-op."""
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("remove", "memory", None, store_result=OK) is None
    assert FakeStore.last_req is None and FakeStore.last_inv is None


def test_noop_on_replace_without_old_entry(monkeypatch):
    """replace OHNE Alt-Eintrag -> kann nicht supersedieren -> No-op (statt Waise zu erzeugen)."""
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("replace", "memory", "neu", store_result=OK) is None
    assert FakeStore.last_req is None and FakeStore.last_inv is None


def test_noop_on_unknown_action(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t1", "o1"):
        assert vw.vault_shadow_write("purge", "memory", "x", store_result=OK, old_entry="x") is None
    assert FakeStore.last_req is None and FakeStore.last_inv is None


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


def test_fires_for_assistant_tool_origin(monkeypatch):
    # assistant_tool = das Modell ruft im Owner-Turn das memory-Werkzeug (Normalfall) -> MUSS als
    # Foreground-Owner-Origin feuern (Live-Befund 2026-07-10; current_origin() ist im echten Turn
    # 'assistant_tool', nicht 'foreground').
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch, origin="assistant_tool")
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t", "o"):
        out = vw.vault_shadow_write("add", "memory", "Notiz", store_result=OK)
    assert out == "written"
    assert FakeStore.last_req is not None


def test_write_mode_user_target_maps_to_user_profile(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t", "o"):
        vw.vault_shadow_write("add", "user", "mag JSON", store_result=OK)
    assert FakeStore.last_req.source_table == "user_profile"


# ---------------------------------------------------------------------------
# §5b Edit/Delete-Propagation: remove -> invalidate-delete, replace -> supersede + insert
# ---------------------------------------------------------------------------

def test_remove_routes_to_invalidate_delete(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    pool = _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("tenant-a", "owner-a"):
        out = vw.vault_shadow_write("remove", "memory", None, store_result=OK, old_entry="Notiz X")
    assert out == "invalidated"
    assert FakeStore.last_req is None            # KEIN Insert bei remove
    inv = FakeStore.last_inv
    assert inv is not None
    from tools.vault.vault_store import INVALIDATE_DELETE
    assert inv.mode == INVALIDATE_DELETE
    assert (inv.tenant_id, inv.owner_id, inv.source_table) == ("tenant-a", "owner-a", "owner_memory")
    assert len(inv.source_id) == 64             # Content-Hash des Alt-Eintrags
    assert pool.borrowed == 1 and pool.returned == 1


def test_replace_supersedes_old_then_inserts_new(monkeypatch):
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("tenant-a", "owner-a"):
        out = vw.vault_shadow_write("replace", "memory", "Neuer Text",
                                    store_result=OK, old_entry="Alter Text")
    assert out == "written"
    from tools.vault.vault_store import INVALIDATE_SUPERSEDE
    # Supersede-alt (auf Alt-Eintrag) UND Insert-neu (auf Neu-Content) liefen beide.
    assert FakeStore.last_inv.mode == INVALIDATE_SUPERSEDE
    assert FakeStore.last_req is not None
    # verschiedene Natural-Keys (alte vs neue Zeile).
    assert FakeStore.last_inv.source_id != FakeStore.last_req.source_id
    assert FakeStore.last_req.source_id == vw._phase1_source_id("Neuer Text")
    assert FakeStore.last_inv.source_id == vw._phase1_source_id("Alter Text")


def test_replace_supersede_fail_skips_insert(monkeypatch):
    """Scheitert der Supersede-alt, darf die neue Zeile NICHT eingefügt werden (sonst beide
    recall-fähig = die §5b-Divergenz). Fail-soft: alte Zeile bleibt, MEMORY.md autoritativ."""
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    FakeStore.invalidate_persisted = False
    with vw.vault_write_identity("tenant-a", "owner-a"):
        out = vw.vault_shadow_write("replace", "memory", "Neuer Text",
                                    store_result=OK, old_entry="Alter Text")
    assert out == "error"                        # Supersede-Status durchgereicht
    assert FakeStore.last_inv is not None and FakeStore.last_req is None   # kein Insert


def test_remove_plumbing_dry_run_no_backend(monkeypatch):
    _flags(monkeypatch, plumbing=True, write=False)
    _foreground(monkeypatch)

    def _boom():
        raise AssertionError("Pool im PLUMBING-Dry-Run NICHT anfassen")
    monkeypatch.setattr("tools.vault.db_runtime.get_vault_pool", _boom)
    with vw.vault_write_identity("t", "o"):
        assert vw.vault_shadow_write("remove", "memory", None,
                                     store_result=OK, old_entry="X") == "plumbing_dry_run"


def test_replace_plumbing_dry_run_no_backend(monkeypatch):
    _flags(monkeypatch, plumbing=True, write=False)
    _foreground(monkeypatch)

    def _boom():
        raise AssertionError("Pool im PLUMBING-Dry-Run NICHT anfassen")
    monkeypatch.setattr("tools.vault.db_runtime.get_vault_pool", _boom)
    with vw.vault_write_identity("t", "o"):
        assert vw.vault_shadow_write("replace", "memory", "neu",
                                     store_result=OK, old_entry="alt") == "plumbing_dry_run"


def test_source_id_hash_consistency_across_whitespace(monkeypatch):
    """Hash-Konsistenz-Beweis (MEMORY.md speichert gestrippt): der bei add gehashte Content und
    der bei remove/replace gehashte Alt-Eintrag ergeben DENSELBEN Natural-Key trotz Whitespace ->
    remove/replace lokalisieren die add-Zeile. Sonst wäre §5b wirkungslos."""
    _flags(monkeypatch, write=True)
    _foreground(monkeypatch)
    _patch_write_backend(monkeypatch)
    with vw.vault_write_identity("t", "o"):
        vw.vault_shadow_write("add", "memory", "   Notiz Y   ", store_result=OK)
        add_sid = FakeStore.last_req.source_id
        vw.vault_shadow_write("remove", "memory", None, store_result=OK, old_entry="Notiz Y")
        rem_sid = FakeStore.last_inv.source_id
    assert add_sid == rem_sid == vw._phase1_source_id("Notiz Y")


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

    def _boom(action, target, content, *, store_result=None, old_entry=None):
        called["args"] = (action, target, content, store_result, old_entry)
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


def test_memory_tool_hook_pops_vault_old_entry(monkeypatch):
    """§5b: replace/remove reichen `_vault_old_entry` als interne Naht durch -> das Modell darf es
    NIE im Tool-Ergebnis sehen (memory_tool poppt es vor json.dumps) + der Hook bekommt es."""
    import tools.memory_tool as mt

    seen = {}

    def _capture(action, target, content, *, store_result=None, old_entry=None):
        seen["old_entry"] = old_entry
        seen["result_has_key"] = isinstance(store_result, dict) and "_vault_old_entry" in store_result

    monkeypatch.setattr("tools.vault.vault_wiring.vault_shadow_write", _capture)

    class FakeStore3:
        def remove(self, target, old_text):
            return {"success": True, "target": target, "entries": [], "_vault_old_entry": "Alter Eintrag"}

    out = mt.memory_tool("remove", "memory", old_text="Alt", store=FakeStore3())
    body = json.loads(out)
    assert body.get("success") is True
    assert "_vault_old_entry" not in body          # dem Modell NIE gezeigt
    assert seen["old_entry"] == "Alter Eintrag"    # aber an den Shadow-Write gereicht
    assert seen["result_has_key"] is False         # schon vor dem Hook-Call gepoppt


# ---------------------------------------------------------------------------
# DLP-Redaktion owner_memory (Owner-Ratifikation 2026-07-11)
# ---------------------------------------------------------------------------

def test_build_request_owner_memory_is_embed_eligible():
    """Owner-ratifizierter Guard: owner_memory ist owner-authored Klartext -> redaction_state MUSS
    'applied' sein (nichts zu redigieren), sonst wird die Zeile NIE embed-fähig (embed-gate) und die
    semantische Suche bleibt tot. Dreht ein Refactor das still auf 'pending', bricht dieser Test.
    sanitization_state bleibt Store-Sache (Injektions-Scan) -> hier nicht geprüft."""
    req = vw._build_request("memory", "Owner-Notiz", "t1", "o1")
    assert req.redaction_state == "applied", "owner_memory muss redaktions-fertig sein (embed-eligible)"
    assert req.taint == {"from_untrusted_inbound": False}
    from tools.vault.vault_store import SOURCE_FOREGROUND_OWNER
    assert req.source == SOURCE_FOREGROUND_OWNER
    # user_profile (target='user') ist ebenfalls owner-authored -> gleiche Regel.
    assert vw._build_request("user", "Profil", "t1", "o1").redaction_state == "applied"
