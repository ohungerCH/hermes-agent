"""Encrypted filesystem store for Jarvis owner-chat attachments.

Only ciphertext envelopes are written below HERMES_HOME.  PostgreSQL receives
metadata, never payload bytes.  All object references are opaque and owner-
scoped through the existing Vault RLS context.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Optional

from tools.vault.vault_store import (
    ObjectMetadataWrite,
    SOURCE_INGEST,
    TRUST_UNTRUSTED,
)


ATTACHMENT_TTL = timedelta(days=7)
ATTACHMENT_REF_RE = re.compile(r"^att_[0-9a-f]{16,64}$")


class AttachmentStoreError(RuntimeError):
    """Value-free attachment failure safe for an HTTP boundary."""


@dataclass(frozen=True)
class AttachmentRecord:
    object_key: str
    expires_at: datetime
    content_type: str
    byte_size: int


@dataclass(frozen=True)
class LoadedAttachment:
    data: bytes
    content_type: str
    object_key: str


def _utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


class AttachmentStore:
    def __init__(self, *, root: str | os.PathLike[str], crypto: Any,
                 metadata_store: Any) -> None:
        self.root = Path(root)
        self.transient_dir = self.root / "transient"
        self.archive_dir = self.root / "archive"
        self.key_dir = self.root / "keys"
        self._crypto = crypto
        self._metadata = metadata_store
        for directory in (self.root, self.transient_dir, self.archive_dir, self.key_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)

    @staticmethod
    def _validate_key(object_key: str) -> str:
        if not isinstance(object_key, str) or ATTACHMENT_REF_RE.fullmatch(object_key) is None:
            raise AttachmentStoreError("invalid attachment reference")
        return object_key

    def _path(self, object_key: str, *, archive: bool = False) -> Path:
        key = self._validate_key(object_key)
        return (self.archive_dir if archive else self.transient_dir) / f"{key}.json"

    @staticmethod
    def _write_atomic(path: Path, envelope: str) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".attachment-", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp)
        try:
            os.write(fd, envelope.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            path.chmod(0o600)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    def put_transient(self, data: bytes, *, tenant_id: str, owner_id: str,
                      content_type: str, now: Optional[datetime] = None) -> AttachmentRecord:
        if not isinstance(data, bytes) or not data:
            raise AttachmentStoreError("attachment is empty")
        current = now or datetime.now(timezone.utc)
        expires = current + ATTACHMENT_TTL
        object_key = f"att_{os.urandom(16).hex()}"
        encrypted = self._crypto.encrypt(data, owner_id=owner_id)
        path = self._path(object_key)
        try:
            self._write_atomic(path, encrypted["envelope"])
            result = self._metadata.write_object_metadata(ObjectMetadataWrite(
                tenant_id=tenant_id,
                owner_id=owner_id,
                source=SOURCE_INGEST,
                trust_level=TRUST_UNTRUSTED,
                object_key=object_key,
                key_ref=encrypted["key_ref"],
                expires_at=expires,
                content_type=content_type,
                byte_size=len(data),
            ))
            if not getattr(result, "persisted", False):
                raise AttachmentStoreError("attachment metadata was not persisted")
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        return AttachmentRecord(object_key, expires, content_type, len(data))

    def load(self, object_key: str, *, tenant_id: str, owner_id: str,
             now: Optional[datetime] = None) -> LoadedAttachment:
        key = self._validate_key(object_key)
        row = self._metadata.read_object_metadata(
            tenant_id=tenant_id, owner_id=owner_id, object_key=key,
        )
        if not row:
            raise AttachmentStoreError("attachment not found")
        expires = _utc(row.get("expires_at"))
        if expires is not None and expires <= (now or datetime.now(timezone.utc)):
            raise AttachmentStoreError("attachment expired")
        path = self._path(key)
        if not path.exists():
            path = self._path(key, archive=True)
        try:
            envelope = path.read_text(encoding="utf-8")
            data = self._crypto.decrypt(envelope, owner_id=owner_id)
        except Exception as exc:
            raise AttachmentStoreError("attachment is unavailable") from exc
        return LoadedAttachment(
            data=data,
            content_type=str(row.get("content_type") or "application/octet-stream"),
            object_key=key,
        )

    def cleanup_expired(self, *, now: Optional[datetime] = None) -> int:
        current = now or datetime.now(timezone.utc)
        removed = 0
        for row in self._metadata.list_expired_objects(now=current):
            key = str(row.get("object_key") or "")
            try:
                path = self._path(key)
            except AttachmentStoreError:
                continue
            try:
                path.unlink(missing_ok=True)
                result = self._metadata.delete_transient_object(
                    tenant_id=row["tenant_id"], owner_id=row["owner_id"], object_key=key,
                )
                if getattr(result, "persisted", False):
                    removed += 1
            except Exception:
                continue
        return removed

    def delete_ciphertext(self, object_key: str) -> None:
        for archive in (False, True):
            try:
                self._path(object_key, archive=archive).unlink(missing_ok=True)
            except OSError as exc:
                raise AttachmentStoreError("attachment deletion failed") from exc

    def promote_to_archive(self, object_key: str, *, tenant_id: str,
                           owner_id: str, original: bool = False) -> bool:
        """Sichert ein transientes Objekt dauerhaft, bevor Meaning gelöscht wird.

        Rückgabe ``True`` bedeutet neu promoviert, ``False`` bereits archiviert.
        Reihenfolge: lesen/konvertieren -> Archiv-Ciphertext -> expires_at=NULL ->
        Transient-Unlink. Jeder Fehler vor dem letzten Schritt lässt das transiente
        Original und damit die Wiederholbarkeit erhalten.
        """
        row = self._metadata.read_object_metadata(
            tenant_id=tenant_id,
            owner_id=owner_id,
            object_key=object_key,
        )
        if not row:
            raise AttachmentStoreError("attachment metadata unavailable")
        if row.get("expires_at") is None:
            return False
        loaded = self.load(
            object_key,
            tenant_id=tenant_id,
            owner_id=owner_id,
        )
        raw_bytes, content_type = self.archive_copy(
            loaded.data,
            loaded.content_type,
            original=original,
        )
        encrypted = self._crypto.encrypt(raw_bytes, owner_id=owner_id)
        self.write_archive_envelope(object_key, encrypted["envelope"])
        self.finalize_archive_promotion(
            object_key,
            tenant_id=tenant_id,
            owner_id=owner_id,
            content_type=content_type,
            byte_size=len(raw_bytes),
            key_ref=encrypted["key_ref"],
        )
        return True

    def finalize_archive_promotion(self, object_key: str, *, tenant_id: str,
                                   owner_id: str, content_type: str,
                                   byte_size: int, key_ref: Optional[str] = None) -> None:
        """Finalisiert die gemeinsame Promotion nach durablem Archiv-Ciphertext."""
        row = self._metadata.read_object_metadata(
            tenant_id=tenant_id,
            owner_id=owner_id,
            object_key=object_key,
        )
        if not row:
            raise AttachmentStoreError("attachment metadata unavailable")
        promoted = self._metadata.write_object_metadata(ObjectMetadataWrite(
            tenant_id=tenant_id,
            owner_id=owner_id,
            source=SOURCE_INGEST,
            trust_level=TRUST_UNTRUSTED,
            object_key=object_key,
            key_ref=key_ref or row.get("key_ref") or "",
            expires_at=None,
            content_type=content_type,
            byte_size=byte_size,
        ))
        if not getattr(promoted, "persisted", False):
            raise AttachmentStoreError("attachment promotion was not persisted")
        try:
            self._path(object_key).unlink(missing_ok=True)
        except OSError as exc:
            raise AttachmentStoreError("transient attachment cleanup failed") from exc

    @staticmethod
    def exif_context(data: bytes, content_type: str) -> "tuple[bytes, str]":
        """Liest Aufnahmezeit + GPS LOKAL aus (Pillow, kein Egress) und liefert
        (bild_ohne_exif, kontext_text). Das gestrippte Bild geht zum Provider
        (Datenminimierung: eingebettetes GPS verlaesst den Server nie); der
        Kontext-Text fliesst in den Meta-Kontext des Gedaechtnis-Eintrags.
        Fail-soft: bei jedem Fehler Original-Bytes + leerer Kontext."""
        if not content_type.lower().startswith("image/"):
            return data, ""
        try:
            from PIL import Image, ImageOps

            parts = []
            with Image.open(BytesIO(data)) as image:
                exif = image.getexif()
                taken = exif.get(36867) or exif.get(306)  # DateTimeOriginal | DateTime
                if isinstance(taken, str) and taken.strip():
                    parts.append("aufgenommen " + taken.strip())
                try:
                    gps = exif.get_ifd(34853)  # GPSInfo
                except Exception:
                    gps = None
                if gps:
                    def _deg(v, ref):
                        try:
                            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
                            return -d if str(ref) in ("S", "W") else d
                        except Exception:
                            return None
                    lat = _deg(gps.get(2), gps.get(1))
                    lon = _deg(gps.get(4), gps.get(3))
                    if lat is not None and lon is not None:
                        parts.append(f"GPS {lat:.5f},{lon:.5f}")
                image = ImageOps.exif_transpose(image)
                output = BytesIO()
                fmt = "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG"
                if fmt == "JPEG" and image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(output, format=fmt, quality=92)
                stripped = output.getvalue()
            return stripped, "; ".join(parts)
        except Exception:
            return data, ""

    @staticmethod
    def archive_copy(data: bytes, content_type: str, *, original: bool) -> tuple[bytes, str]:
        """Return original bytes or a ~2000px JPEG archive copy for images."""
        if original or not content_type.lower().startswith("image/"):
            return data, content_type
        try:
            from PIL import Image, ImageOps

            with Image.open(BytesIO(data)) as image:
                image = ImageOps.exif_transpose(image)
                image.thumbnail((2000, 2000))
                if image.mode not in ("RGB", "L"):
                    background = Image.new("RGB", image.size, "white")
                    if "A" in image.getbands():
                        background.paste(image, mask=image.getchannel("A"))
                    else:
                        background.paste(image)
                    image = background
                elif image.mode == "L":
                    image = image.convert("RGB")
                output = BytesIO()
                image.save(output, format="JPEG", quality=80, optimize=True)
                return output.getvalue(), "image/jpeg"
        except Exception as exc:
            raise AttachmentStoreError("image archive conversion failed") from exc

    def write_archive_envelope(self, object_key: str, envelope: str) -> None:
        self._write_atomic(self._path(object_key, archive=True), envelope)


class PooledVaultMetadataStore:
    """Small pool adapter that keeps VaultStore's per-operation ownership intact."""

    @staticmethod
    def _call(method: str, *args: Any, **kwargs: Any) -> Any:
        from tools.vault import db_runtime
        from tools.vault.vault_store import VaultStore

        pool = db_runtime.get_vault_pool()
        conn = pool.getconn(timeout=db_runtime.VAULT_GETCONN_TIMEOUT_S)
        try:
            return getattr(VaultStore(connect=lambda: conn), method)(*args, **kwargs)
        finally:
            pool.putconn(conn)

    def write_object_metadata(self, req: ObjectMetadataWrite) -> Any:
        return self._call("write_object_metadata", req)

    def read_object_metadata(self, **kwargs: Any) -> Any:
        return self._call("read_object_metadata", **kwargs)

    def list_expired_objects(self, **kwargs: Any) -> Any:
        return self._call("list_expired_objects", **kwargs)

    def delete_transient_object(self, **kwargs: Any) -> Any:
        return self._call("delete_transient_object", **kwargs)

    def forget_object(self, **kwargs: Any) -> Any:
        return self._call("forget_object", **kwargs)

    def read_memory_item_by_id(self, **kwargs: Any) -> Any:
        return self._call("read_memory_item_by_id", **kwargs)

    def tombstone_memory_item_by_id(self, **kwargs: Any) -> Any:
        return self._call("tombstone_memory_item_by_id", **kwargs)


def create_attachment_store() -> AttachmentStore:
    """Baut den profilgebundenen AttachmentStore für API- und Memory-Pfade."""
    from hermes_constants import get_hermes_home
    from tools.vault.object_store_crypto import ObjectStoreCrypto

    root = get_hermes_home() / "jarvis-attachments"
    return AttachmentStore(
        root=root,
        crypto=ObjectStoreCrypto(root / "keys"),
        metadata_store=PooledVaultMetadataStore(),
    )
