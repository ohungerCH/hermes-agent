from datetime import datetime, timedelta, timezone
import json
import stat

import pytest

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
            "key_ref": req.key_ref,
            "source": req.source,
            "trust_level": req.trust_level,
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


def test_promote_to_archive_is_idempotent_and_unlinks_transient_last(tmp_path):
    root = tmp_path / "attachments"
    repo = MetadataRepo()
    store = AttachmentStore(root=root, crypto=ObjectStoreCrypto(root / "keys"), metadata_store=repo)
    now = datetime.now(timezone.utc)
    record = store.put_transient(
        b"durable-document",
        tenant_id="tenant-a",
        owner_id="owner-a",
        content_type="application/pdf",
        now=now,
    )

    assert store.promote_to_archive(
        record.object_key,
        tenant_id="tenant-a",
        owner_id="owner-a",
    ) is True
    assert repo.rows[record.object_key]["expires_at"] is None
    # Review 22.07./W1: Keep-Promotion schreibt Owner-Kuratierung, nie
    # ingest/untrusted (_OWNER_RESURRECT haengt an source).
    assert repo.rows[record.object_key]["source"] == "foreground_owner"
    assert repo.rows[record.object_key]["trust_level"] == "trusted"
    assert repo.rows[record.object_key]["key_ref"].startswith("per_owner_domain:")
    assert (root / "archive" / f"{record.object_key}.json").exists()
    assert not (root / "transient" / f"{record.object_key}.json").exists()
    assert store.load(
        record.object_key,
        tenant_id="tenant-a",
        owner_id="owner-a",
    ).data == b"durable-document"

    assert store.promote_to_archive(
        record.object_key,
        tenant_id="tenant-a",
        owner_id="owner-a",
    ) is False


def test_promotion_failure_preserves_transient_object(tmp_path):
    class FailingPromotionRepo(MetadataRepo):
        def write_object_metadata(self, req):
            if req.expires_at is None:
                return type("Result", (), {"persisted": False, "status": "error"})()
            return super().write_object_metadata(req)

    root = tmp_path / "attachments"
    repo = FailingPromotionRepo()
    store = AttachmentStore(root=root, crypto=ObjectStoreCrypto(root / "keys"), metadata_store=repo)
    record = store.put_transient(
        b"must-survive",
        tenant_id="tenant-a",
        owner_id="owner-a",
        content_type="application/pdf",
        now=datetime.now(timezone.utc),
    )

    with pytest.raises(Exception, match="promotion"):
        store.promote_to_archive(
            record.object_key,
            tenant_id="tenant-a",
            owner_id="owner-a",
        )

    assert (root / "transient" / f"{record.object_key}.json").exists()
    assert repo.rows[record.object_key]["expires_at"] is not None


class TestExifContext:
    """Slice-2-Verfeinerung: EXIF lokal lesen (Zeit/GPS) + Bild strippen."""

    @staticmethod
    def _jpeg_with_exif() -> bytes:
        from io import BytesIO
        from PIL import Image
        image = Image.new("RGB", (32, 32), "blue")
        exif = image.getexif()
        exif[306] = "2026:07:22 14:03:00"  # DateTime
        gps = exif.get_ifd(34853)
        gps[1] = "N"; gps[2] = (47.0, 22.0, 12.0)
        gps[3] = "E"; gps[4] = (8.0, 32.0, 24.0)
        out = BytesIO()
        image.save(out, format="JPEG", exif=exif)
        return out.getvalue()

    def test_reads_time_and_gps_and_strips(self):
        from tools.vault.attachment_store import AttachmentStore
        data = self._jpeg_with_exif()
        stripped, ctx = AttachmentStore.exif_context(data, "image/jpeg")
        assert "aufgenommen 2026:07:22 14:03:00" in ctx
        assert "GPS 47.37" in ctx and "8.54" in ctx
        # Das gestrippte Bild darf weder Zeit noch GPS mehr tragen.
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(stripped)) as reloaded:
            exif2 = reloaded.getexif()
            assert not exif2.get(306)
            assert not dict(exif2.get_ifd(34853) or {})

    def test_failsoft_on_non_image_and_garbage(self):
        from tools.vault.attachment_store import AttachmentStore
        raw = b"not an image"
        assert AttachmentStore.exif_context(raw, "application/pdf") == (raw, "")
        data, ctx = AttachmentStore.exif_context(raw, "image/jpeg")
        assert data == raw and ctx == ""
