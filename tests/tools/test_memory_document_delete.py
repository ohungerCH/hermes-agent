"""Task #16/Stufe 2: ID-Löschpfad und deterministische Scribe-Schranken."""

from contextlib import contextmanager
import json

import pytest

from tools.memory_tool import (
    MemoryStore,
    begin_memory_tool_turn,
    end_memory_tool_turn,
    memory_tool,
    record_memory_tool_outcome,
)
from tools.vault import vault_wiring as vw
from tools.vault.vault_store import VaultStore


class _Result:
    def __init__(self, *, persisted=True, written=True):
        self.persisted = persisted
        self.memory_item_written = written


class _Metadata:
    def __init__(self, ref, *, promotion_fails=False):
        self.ref = ref
        self.promotion_fails = promotion_fails
        self.tombstoned = []
        self.forgotten = []
        self.reads = []

    def read_memory_item_by_id(self, **kwargs):
        self.reads.append(kwargs)
        return type("Lookup", (), {"available": True, "item": self.ref})()

    def tombstone_memory_item_by_id(self, **kwargs):
        self.tombstoned.append(kwargs)
        return _Result()

    def forget_object(self, **kwargs):
        self.forgotten.append(kwargs)
        return _Result()


class _AttachmentStore:
    def __init__(self, metadata, *, archived=False, promotion_fails=False):
        self._metadata = metadata
        self.archived = archived
        self.promotion_fails = promotion_fails
        self.promotions = []
        self.deleted_ciphertexts = []
        self.splits = []

    def promote_to_archive(self, object_key, **kwargs):
        self.promotions.append((object_key, kwargs))
        if self.promotion_fails:
            raise RuntimeError("promotion failed")
        return not self.archived

    def delete_ciphertext(self, object_key):
        self.deleted_ciphertexts.append(object_key)

    def forget_content_keep_object(self, object_key, **kwargs):
        self.splits.append((object_key, kwargs))
        return _Result()


def _arm_identity(monkeypatch):
    monkeypatch.setattr(vw, "get_vault_write_identity", lambda: ("tenant-a", "owner-a"))


def _document_ref(item_id="11111111-1111-1111-1111-111111111111"):
    return type(
        "Ref",
        (),
        {
            "item_id": item_id,
            "source_table": "object_metadata",
            "source_id": "att_0123456789abcdef",
        },
    )()


def test_remove_per_item_id_refuses_to_orphan_server_local_document(monkeypatch):
    """R7: Der Sprach-Split lässt kein server-lokales Objekt ohne Meaning zurück."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref())
    attachment_store = _AttachmentStore(metadata, archived=True)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    out = json.loads(memory_tool(
        action="remove",
        target="memory",
        item_id="11111111-1111-1111-1111-111111111111",
        old_text="Dieser absichtlich falsche Text darf nicht verwendet werden.",
        store=MemoryStore(),
    ))

    assert out["outcome"] == "keep_would_orphan"
    assert "Vergiss beides" in out["message"]
    assert metadata.tombstoned == []
    assert attachment_store.promotions == []


def test_forget_memory_keep_object_never_promotes_or_tombstones_object_metadata(monkeypatch):
    """R7: object_metadata wird unabhängig vom Archivstatus ohne Teilzustand verweigert."""
    _arm_identity(monkeypatch)

    transient_metadata = _Metadata(_document_ref("22222222-2222-2222-2222-222222222222"))
    transient_store = _AttachmentStore(transient_metadata)
    transient = vw.vault_remove_by_item_id(
        "22222222-2222-2222-2222-222222222222",
        attachment_store=transient_store,
    )
    assert transient["outcome"] == "keep_would_orphan"
    assert transient_store.promotions == []
    assert transient_metadata.tombstoned == []

    archived_metadata = _Metadata(_document_ref("33333333-3333-3333-3333-333333333333"))
    archived_store = _AttachmentStore(archived_metadata, archived=True)
    archived = vw.vault_remove_by_item_id(
        "33333333-3333-3333-3333-333333333333",
        attachment_store=archived_store,
    )
    assert archived["outcome"] == "keep_would_orphan"
    assert archived_store.promotions == []
    assert archived_metadata.tombstoned == []

    failed_metadata = _Metadata(_document_ref("44444444-4444-4444-4444-444444444444"))
    failed_store = _AttachmentStore(failed_metadata, promotion_fails=True)
    failed = vw.vault_remove_by_item_id(
        "44444444-4444-4444-4444-444444444444",
        attachment_store=failed_store,
    )
    assert failed["outcome"] == "keep_would_orphan"
    assert failed_store.promotions == []
    assert failed_metadata.tombstoned == []


def test_invalid_item_id_is_not_found_without_store_access(monkeypatch):
    """R8: Eine erfundene Nicht-UUID erreicht weder Lookup noch Promotion."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref())
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    out = json.loads(memory_tool(
        action="remove",
        target="memory",
        item_id="kein-uuid",
        store=MemoryStore(),
    ))

    assert out["outcome"] == "not_found"
    assert metadata.reads == []
    assert attachment_store.promotions == []


def test_item_id_for_native_class_is_class_not_removable(monkeypatch):
    """R3: bekannte, aber über ID nicht löschbare Klasse ist nie ein not_found."""
    _arm_identity(monkeypatch)
    native_ref = type(
        "Ref",
        (),
        {
            "item_id": "55555555-5555-5555-5555-555555555555",
            "source_table": "owner_memory",
            "source_id": "native-hash",
        },
    )()
    metadata = _Metadata(native_ref)
    out = vw.vault_remove_by_item_id(
        native_ref.item_id,
        attachment_store=_AttachmentStore(metadata),
    )

    assert out["outcome"] == "class_not_removable"
    assert metadata.tombstoned == []


def test_forget_full_keeps_ciphertext_first_compound_order(monkeypatch):
    """R2: Voll-Löschung nutzt Ciphertext-zuerst und danach den Verbund-Tombstone."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("66666666-6666-6666-6666-666666666666"))
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    out = json.loads(memory_tool(
        action="remove",
        target="memory",
        item_id="66666666-6666-6666-6666-666666666666",
        forget_mode="forget_full",
        store=MemoryStore(),
    ))

    assert out["outcome"] == "removed"
    assert attachment_store.deleted_ciphertexts == ["att_0123456789abcdef"]
    assert metadata.forgotten == [{
        "tenant_id": "tenant-a",
        "owner_id": "owner-a",
        "object_key": "att_0123456789abcdef",
    }]


def test_native_store_remove_old_text_remains_supported(tmp_path, monkeypatch):
    """R6: der bestehende file-backed old_text-Pfad bleibt funktional."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    store = MemoryStore()
    assert store.add("memory", "Die native Erinnerung bleibt adressierbar.")["success"] is True

    out = json.loads(memory_tool(
        action="remove",
        target="memory",
        old_text="native Erinnerung",
        store=store,
    ))

    assert out["success"] is True
    assert out["outcome"] == "removed"
    assert store.memory_entries == []

    missing = json.loads(memory_tool(
        action="remove",
        target="memory",
        old_text="nicht vorhanden",
        store=store,
    ))
    assert missing["outcome"] == "not_found"
    assert "No entry matched" not in json.dumps(missing)


@pytest.mark.parametrize("request_class", ["forget", "split"])
@pytest.mark.parametrize(
    ("action", "kwargs"),
    [
        ("add", {"content": "Darf nicht gespeichert werden."}),
        (
            "replace",
            {
                "old_text": "Bestehender Eintrag",
                "content": "Darf nicht ersetzt werden.",
            },
        ),
    ],
)
def test_forget_and_split_turns_reject_add_replace_in_tool_layer(
    tmp_path,
    monkeypatch,
    request_class,
    action,
    kwargs,
):
    """§6b: Die Auftragsklasse sperrt klassenfremde Writes vor dem Store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    store = MemoryStore()
    store.add("memory", "Bestehender Eintrag")
    before = list(store.memory_entries)

    turn = begin_memory_tool_turn(memory_request_class=request_class)
    try:
        out = json.loads(memory_tool(
            action=action,
            target="memory",
            store=store,
            **kwargs,
        ))
    finally:
        end_memory_tool_turn(turn)

    assert out == {
        "success": False,
        "action": action,
        "outcome": "write_forbidden",
        "message": (
            "Dieser Vergiss-Auftrag darf keine Erinnerung speichern oder ersetzen"
        ),
    }
    assert store.memory_entries == before


@pytest.mark.parametrize("request_class", ["forget", "split"])
def test_forget_and_split_turns_reject_batch_with_add_or_replace(
    tmp_path,
    monkeypatch,
    request_class,
):
    """§6b: Auch die Batch-Naht darf das Schreibverbot nicht umgehen."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    store = MemoryStore()
    before = list(store.memory_entries)

    turn = begin_memory_tool_turn(memory_request_class=request_class)
    try:
        out = json.loads(memory_tool(
            target="memory",
            operations=[{"action": "add", "content": "Verbotener Batch-Write"}],
            store=store,
        ))
    finally:
        end_memory_tool_turn(turn)

    assert out["outcome"] == "write_forbidden"
    assert store.memory_entries == before


@pytest.mark.parametrize(
    "forget_mode",
    [None, "forget_memory_keep_object", "forget_full"],
)
def test_split_turn_rejects_every_non_split_remove_mode(monkeypatch, forget_mode):
    """Review-P1 23.07.: split darf forget_full nie erreichen -- die Datei
    bleibt erhalten; nur der inhalts-erhaltende Archiv-Split ist zugelassen.
    Die Sperre sitzt in der Tool-Schicht, nicht in der Instruktion."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("79797979-7979-7979-7979-797979797979"))
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    turn = begin_memory_tool_turn(memory_request_class="split")
    try:
        out = json.loads(memory_tool(
            action="remove",
            target="memory",
            item_id="79797979-7979-7979-7979-797979797979",
            forget_mode=forget_mode,
            store=MemoryStore(),
        ))
    finally:
        end_memory_tool_turn(turn)

    assert out["outcome"] == "remove_mode_forbidden"
    assert out["success"] is False
    # Kein Store-Zugriff: weder gelesen noch geloescht noch promoted.
    assert metadata.reads == []
    assert metadata.forgotten == []
    assert attachment_store.deleted_ciphertexts == []
    assert attachment_store.splits == []


def test_remember_turn_rejects_remove_in_tool_layer(monkeypatch):
    """Symmetrie zu §6b: ein Merk-Auftrag darf nie löschen."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("79797979-7979-7979-7979-797979797979"))
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    turn = begin_memory_tool_turn(memory_request_class="remember")
    try:
        out = json.loads(memory_tool(
            action="remove",
            target="memory",
            item_id="79797979-7979-7979-7979-797979797979",
            forget_mode="forget_full",
            store=MemoryStore(),
        ))
    finally:
        end_memory_tool_turn(turn)

    assert out["outcome"] == "remove_forbidden"
    assert metadata.reads == []
    assert metadata.forgotten == []
    assert attachment_store.deleted_ciphertexts == []


@pytest.mark.parametrize("request_class", ["split", "remember"])
def test_split_and_remember_turns_reject_batch_remove(
    tmp_path,
    monkeypatch,
    request_class,
):
    """Batch-Ops tragen kein forget_mode -- remove darin ist klassenfremd."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    store = MemoryStore()
    store.add("memory", "Bestehender Eintrag")
    before = list(store.memory_entries)

    turn = begin_memory_tool_turn(memory_request_class=request_class)
    try:
        out = json.loads(memory_tool(
            target="memory",
            operations=[{"action": "remove", "old_text": "Bestehender Eintrag"}],
            store=store,
        ))
    finally:
        end_memory_tool_turn(turn)

    assert out["outcome"] in {"remove_mode_forbidden", "remove_forbidden"}
    assert store.memory_entries == before


def test_forget_turn_keeps_full_delete_available(monkeypatch):
    """Gegenprobe: die forget-Klasse behält forget_full ("Vergiss beides")."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("79797979-7979-7979-7979-797979797979"))
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    turn = begin_memory_tool_turn(memory_request_class="forget")
    try:
        out = json.loads(memory_tool(
            action="remove",
            target="memory",
            item_id="79797979-7979-7979-7979-797979797979",
            forget_mode="forget_full",
            store=MemoryStore(),
        ))
    finally:
        end_memory_tool_turn(turn)

    assert out["outcome"] == "removed"
    assert attachment_store.deleted_ciphertexts == ["att_0123456789abcdef"]


@pytest.mark.parametrize(
    ("request_class", "forget_mode"),
    [
        ("forget", "forget_full"),
        ("split", "forget_content_keep_object"),
    ],
)
def test_ambiguous_recall_blocks_remove_until_one_context_answer(
    monkeypatch,
    request_class,
    forget_mode,
):
    """§5d: Mehrere Treffer können vor der Rückfrage nichts löschen."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("79797979-7979-7979-7979-797979797979"))
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    turn = begin_memory_tool_turn(memory_request_class=request_class)
    try:
        record_memory_tool_outcome(
            "recall",
            {
                "action": "recall",
                "available": True,
                "count": 2,
                "matches": [{"item_id": "one"}, {"item_id": "two"}],
            },
        )
        blocked = json.loads(memory_tool(
            action="remove",
            target="memory",
            item_id="79797979-7979-7979-7979-797979797979",
            forget_mode=forget_mode,
            store=MemoryStore(),
        ))
    finally:
        end_memory_tool_turn(turn)

    assert blocked["outcome"] == "needs_disambiguation"
    assert metadata.reads == []
    assert metadata.forgotten == []
    assert attachment_store.splits == []

    answered_turn = begin_memory_tool_turn(
        memory_request_class=request_class,
        memory_disambiguated=True,
    )
    try:
        record_memory_tool_outcome(
            "recall",
            {
                "action": "recall",
                "available": True,
                "count": 2,
                "matches": [{"item_id": "one"}, {"item_id": "two"}],
            },
        )
        allowed = json.loads(memory_tool(
            action="remove",
            target="memory",
            item_id="79797979-7979-7979-7979-797979797979",
            forget_mode=forget_mode,
            store=MemoryStore(),
        ))
    finally:
        end_memory_tool_turn(answered_turn)

    assert allowed["outcome"] in {"removed", "split_done"}


def test_forget_content_keep_object_returns_split_done(monkeypatch):
    """§6c: Die neue Operation ist separat vom weiter gesperrten Plain-Split."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("77777777-7777-7777-7777-777777777777"))
    attachment_store = _AttachmentStore(metadata)

    out = vw.vault_remove_by_item_id(
        "77777777-7777-7777-7777-777777777777",
        operation="forget_content_keep_object",
        attachment_store=attachment_store,
    )

    assert out["success"] is True
    assert out["outcome"] == "split_done"
    assert "Inhalt" in out["message"]
    assert len(attachment_store.splits) == 1
    object_key, split_kwargs = attachment_store.splits[0]
    assert object_key == "att_0123456789abcdef"
    assert split_kwargs["tenant_id"] == "tenant-a"
    assert split_kwargs["owner_id"] == "owner-a"
    assert split_kwargs["item_id"] == "77777777-7777-7777-7777-777777777777"
    assert split_kwargs["stub_summary"].startswith(
        "Dokument att_01234567, Inhalt auf Owner-Wunsch vergessen, Datei archiviert "
    )


def test_memory_tool_dispatches_forget_content_keep_object(monkeypatch):
    """§6c: Der Scribe erreicht die neue Operation über den öffentlichen Tool-Vertrag."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(_document_ref("abababab-abab-abab-abab-abababababab"))
    attachment_store = _AttachmentStore(metadata)
    monkeypatch.setattr(
        "tools.vault.attachment_store.create_attachment_store",
        lambda: attachment_store,
    )

    out = json.loads(memory_tool(
        action="remove",
        target="memory",
        item_id="abababab-abab-abab-abab-abababababab",
        forget_mode="forget_content_keep_object",
        store=MemoryStore(),
    ))

    assert out["outcome"] == "split_done"
    assert len(attachment_store.splits) == 1


def test_forget_content_keep_object_missing_item_is_not_found(monkeypatch):
    """§7: Ein nicht existentes Dokument erzeugt keinen Split-Teilerfolg."""
    _arm_identity(monkeypatch)
    metadata = _Metadata(None)
    attachment_store = _AttachmentStore(metadata)

    out = vw.vault_remove_by_item_id(
        "88888888-8888-8888-8888-888888888888",
        operation="forget_content_keep_object",
        attachment_store=attachment_store,
    )

    assert out["outcome"] == "not_found"
    assert attachment_store.splits == []


def test_archive_stub_and_object_promotion_share_one_transaction(monkeypatch):
    """§6c: Beide DB-Autoritäten werden in einem Transaktionsfenster geändert."""
    class _Cursor:
        rowcount = 1

        def __init__(self):
            self.calls = []

        def execute(self, statement, params):
            self.calls.append((statement, params))

    class _Connection:
        def __init__(self):
            self.cursor = _Cursor()
            self.commits = 0
            self.rollbacks = 0

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    connection = _Connection()
    transaction_entries = []

    @contextmanager
    def _fake_transaction(conn, tenant_id, owner_id):
        transaction_entries.append((tenant_id, owner_id))
        yield conn.cursor

    monkeypatch.setattr(
        "tools.vault.vault_store.vault_transaction",
        _fake_transaction,
    )
    store = VaultStore(connect=lambda: connection)

    out = store.promote_object_and_replace_memory_with_stub(
        tenant_id="tenant-a",
        owner_id="owner-a",
        item_id="99999999-9999-9999-9999-999999999999",
        object_key="att_0123456789abcdef",
        stub_summary="Dokument att_01234567, Inhalt vergessen, Datei archiviert 2026-07-23",
        key_ref="key-a",
        content_type="image/jpeg",
        byte_size=42,
    )

    assert out.persisted is True
    assert transaction_entries == [("tenant-a", "owner-a")]
    assert len(connection.cursor.calls) == 2
    assert "UPDATE public.object_metadata" in connection.cursor.calls[0][0]
    assert "UPDATE public.memory_items" in connection.cursor.calls[1][0]
    assert connection.commits == 1
    assert connection.rollbacks == 0
