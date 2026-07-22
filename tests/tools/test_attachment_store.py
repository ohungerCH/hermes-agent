from datetime import datetime, timedelta, timezone
import json
import stat

from tools.vault.attachment_store import AttachmentStore
from tools.vault.object_store_crypto import ObjectStoreCrypto


class MetadataRepo:
    def __init__(self):
        self.rows = {}

    def write_object_metadata(self, req):
        self.rows[req.object_key] = {
            "tenant_id": req.tenant_id,
            "owner_id": req.owner_id,
            "object_key": req.object_key,
            "expires_at": req.expires_at,
            "content_type": req.content_type,
            "byte_size": req.byte_size,
            "deleted_at": None,
        }
        return type("Result", (), {"persisted": True, "status": "written"})()

    def read_object_metadata(self, *, tenant_id, owner_id, object_key):
        row = self.rows.get(object_key)
        if row and row["tenant_id"] == tenant_id and row["owner_id"] == owner_id:
            return row
        return None

    def list_expired_objects(self, *, now):
        return [row for row in self.rows.values() if row["expires_at"] and row["expires_at"] <= now]

    def delete_transient_object(self, *, tenant_id, owner_id, object_key):
        self.rows.pop(object_key, None)
        return type("Result", (), {"persisted": True, "status": "invalidated"})()


def test_transient_crypto_roundtrip_and_permissions(tmp_path):
    root = tmp_path / "attachments"
    repo = MetadataRepo()
    store = AttachmentStore(root=root, crypto=ObjectStoreCrypto(root / "keys"), metadata_store=repo)
    now = datetime(2026, 7, 22, 12, tzinfo=timezone.utc)

    record = store.put_transient(
        b"original-image-bytes",
        tenant_id="tenant-a",
        owner_id="owner-a",
        content_type="image/jpeg",
        now=now,
    )
    loaded = store.load(record.object_key, tenant_id="tenant-a", owner_id="owner-a", now=now)

    assert loaded.data == b"original-image-bytes"
    assert loaded.content_type == "image/jpeg"
    assert record.expires_at == now + timedelta(days=7)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "transient").stat().st_mode) == 0o700
    envelope = json.loads((root / "transient" / f"{record.object_key}.json").read_text())
    assert "original-image-bytes" not in json.dumps(envelope)


def test_cleanup_removes_expired_ciphertext_and_metadata(tmp_path):
    root = tmp_path / "attachments"
    repo = MetadataRepo()
    store = AttachmentStore(root=root, crypto=ObjectStoreCrypto(root / "keys"), metadata_store=repo)
    created = datetime(2026, 7, 1, tzinfo=timezone.utc)
    record = store.put_transient(
        b"old",
        tenant_id="tenant-a",
        owner_id="owner-a",
        content_type="image/png",
        now=created,
    )

    assert store.cleanup_expired(now=created + timedelta(days=8)) == 1
    assert record.object_key not in repo.rows
    assert not (root / "transient" / f"{record.object_key}.json").exists()
