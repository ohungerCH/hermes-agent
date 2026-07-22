"""Task #16: ID-basierter Löschpfad für Dokument-Erinnerungen."""

from datetime import datetime, timezone
import json

from tools.memory_tool import MemoryStore, memory_tool
from tools.vault import vault_wiring as vw


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

    def read_memory_item_by_id(self, **kwargs):
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

    def promote_to_archive(self, object_key, **kwargs):
        self.promotions.append((object_key, kwargs))
        if self.promotion_fails:
            raise RuntimeError("promotion failed")
        return not self.archived

    def delete_ciphertext(self, object_key):
        self.deleted_ciphertexts.append(object_key)


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


def test_remove_per_item_id_tombstones_owner_scoped_object_memory(monkeypatch):
    """R1/R5: item_id ist primär; der DB-Zugriff bleibt über die Owner-Identität gebunden."""
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

    assert out["outcome"] == "removed"
    assert metadata.tombstoned == [{
        "tenant_id": "tenant-a",
        "owner_id": "owner-a",
        "item_id": "11111111-1111-1111-1111-111111111111",
    }]
    assert attachment_store.promotions[0][0] == "att_0123456789abcdef"


def test_forget_memory_keep_object_promotes_then_tombstones_or_aborts(monkeypatch):
    """R2: transient wird zuerst promoviert; archiviert wird nur tombstoned; Fehler löscht nichts."""
    _arm_identity(monkeypatch)

    transient_metadata = _Metadata(_document_ref("22222222-2222-2222-2222-222222222222"))
    transient_store = _AttachmentStore(transient_metadata)
    transient = vw.vault_remove_by_item_id(
        "22222222-2222-2222-2222-222222222222",
        attachment_store=transient_store,
    )
    assert transient["outcome"] == "removed"
    assert len(transient_store.promotions) == 1
    assert len(transient_metadata.tombstoned) == 1

    archived_metadata = _Metadata(_document_ref("33333333-3333-3333-3333-333333333333"))
    archived_store = _AttachmentStore(archived_metadata, archived=True)
    archived = vw.vault_remove_by_item_id(
        "33333333-3333-3333-3333-333333333333",
        attachment_store=archived_store,
    )
    assert archived["outcome"] == "removed"
    assert len(archived_metadata.tombstoned) == 1

    failed_metadata = _Metadata(_document_ref("44444444-4444-4444-4444-444444444444"))
    failed_store = _AttachmentStore(failed_metadata, promotion_fails=True)
    failed = vw.vault_remove_by_item_id(
        "44444444-4444-4444-4444-444444444444",
        attachment_store=failed_store,
    )
    assert failed["outcome"] == "promotion_failed"
    assert failed_metadata.tombstoned == []


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
