"""OD-3: per-owner Crypto-Shred-Key für Object-Store-Rohbytes (AES-256-GCM Envelope).

SPIEGELT die Krypto-Disziplin von ops/services/m365-connector/app/token_store.py (bewusst
byte-nah gehalten, damit der geprüfte Kern nicht divergiert), mit EINEM tragenden Unterschied:
m365 nutzt EINEN gemounteten KEK für alle Owner (owner nur in der AAD gebunden). OD-3 hält
einen KEK PRO OWNER -> das Löschen eines Owner-Keys macht NUR dessen Rohbytes unlesbar
(Crypto-Shred = Erasure pro Owner). Ein anderer Owner ist unberührt.

GRENZE DES PRIMITIVS (ehrlich, kein false-green):
  - Dies ist NUR das LIVE-Crypto-Shred-Primitiv. Es ist NICHT die ganze Löschung:
      * PHYSISCHES Löschen der Zeilen/Objekte (Platz freigeben in der lebenden DB) = separate
        Purge-Kaskade (App-Job, STUFE5 §5). Der KEK arbeitet auf OWNER-Ebene, NIE pro Objekt;
        per-Objekt-Retention ist physischer Purge, kein Crypto-Shred.
      * "Erasure wirkt gegen BACKUPS" ist NUR wahr, wenn die Keys getrennt von den Daten liegen
        (tun sie: Keystore ausserhalb des pgdata-Volumes) UND ein Keystore-Backup einen
        geschredderten Key NICHT wiederbelebt. Letzteres ist NICHT hier gelöst, sondern Sache
        des Backup-Runbooks (Shred-Journal nach Restore / Keystore-Retention < Daten-Retention /
        Keystore vom Restore ausschliessen). keys-separat ist NOTWENDIG, nicht HINREICHEND.

AT-REST-MODELL (Advisor-entschieden): der per-owner KEK liegt als 32-Byte-Datei im Keystore
(Plaintext hinter Mount + LUKS-at-rest des Host-Volumes = dieselbe Haltung wie m365s gemounteter
KEK, keine Regression). KEIN Master-KEK-Wrap / KEIN KDF (brachte gegenüber LUKS fast nichts und
führte den bewusst vermiedenen Single-Point wieder ein). Crypto-Shred = unlink der Key-Datei;
KEIN Block-Overwrite (auf CoW/SSD ohnehin Theater, und der Running-Host-Angreifer hat den Live-
Keystore ohnehin) -> die Garantie ruht auf LUKS + unlink, nicht auf Byte-Überschreiben.

DLP: NIE einen Key- oder Klartext-Wert loggen/zurückgeben (ausser die Bytes, die der Caller
braucht). Fehlermeldungen tragen Grund + owner-slug (PII-freier Hash), NIE einen Wert.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENVELOPE_V = 1
_ALG = "A256GCM-envelope"
_RING_V = 1

# Domänen-separierte, einweg-abgeleitete Identifier (leaken NICHTS über Key/owner_id).
_KID_PREFIX = b"vault-objstore-kid-v1:"
_OWNER_FILE_PREFIX = b"vault-objstore-owner-v1:"


class ObjectStoreCryptoError(RuntimeError):
    """Krypto-/Persistenz-/Parsing-Fehler. Trägt NIE einen Key-/Klartext-Wert (DLP)."""


class CryptoShreddedError(ObjectStoreCryptoError):
    """Der per-owner Key ist weg (unlink = Crypto-Shred) ODER die konkrete kid ist nicht mehr im
    Ring -> die Rohbytes sind kryptografisch unlesbar = gelöscht. KEIN stilles None."""


def _canonical_json(obj: dict) -> bytes:
    """Kanonische Serialisierung (sortierte Keys, kompakt). EINE Helper-Funktion für Envelope-Dump
    UND AAD in encrypt UND decrypt -- divergiert das, schlägt JEDER decrypt mit InvalidTag fehl."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ObjectStoreCryptoError("envelope b64 field malformed") from exc


def _kid(key: bytes) -> str:
    return hashlib.sha256(_KID_PREFIX + key).hexdigest()[:16]


def _owner_file_id(owner_id: str) -> str:
    """Pfad-sicherer, PII-FREIER Dateiname-/Slug-Bestandteil = reiner domänen-separierter Hash.
    Aus dem Dateinamen ist die owner_id (z.B. eine E-Mail) nicht rekonstruierbar; nur [0-9a-f]
    (kein Path-Traversal); eindeutig je Owner (voller owner_id geht in den Hash ein).

    Zentraler Typ-Guard (alle öffentlichen Methoden laufen hier durch): ein Nicht-String owner_id
    (None/int/bytes) -> ObjectStoreCryptoError statt AttributeError (fail-closed + DLP-konforme
    Fehlermeldung), review-gehärtet."""
    if not isinstance(owner_id, str) or owner_id == "":
        raise ObjectStoreCryptoError(f"owner_id must be a non-empty str, not {type(owner_id).__name__}")
    return hashlib.sha256(_OWNER_FILE_PREFIX + owner_id.encode("utf-8")).hexdigest()[:32]


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class ObjectStoreCrypto:
    """Per-owner Envelope-Krypto + Crypto-Shred. Der Keystore-Ordner MUSS ausserhalb des
    Daten-(pgdata-)Volumes liegen (physische Trennung = Grundlage für 'Daten frei sichern')."""

    def __init__(self, keystore_dir: str | os.PathLike[str]) -> None:
        self._keystore = Path(keystore_dir)

    # --- per-owner Key-Ring (Keystore) -------------------------------------
    def _ring_path(self, owner_id: str) -> Path:
        return self._keystore / f"{_owner_file_id(owner_id)}.keyring.json"

    def _load_ring(self, owner_id: str) -> Optional[dict]:
        """Liest den per-owner Ring. None = nicht vorhanden (nie erzeugt ODER geschreddert).
        Vorhanden-aber-kaputt -> fail-closed (ObjectStoreCryptoError), NICHT stilles None."""
        path = self._ring_path(owner_id)
        if not path.exists():
            return None
        try:
            ring = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ObjectStoreCryptoError(
                f"keyring unreadable/corrupt for owner '{_owner_file_id(owner_id)}'"
            ) from exc
        if not isinstance(ring, dict) or "active_kid" not in ring or not isinstance(ring.get("keys"), dict):
            raise ObjectStoreCryptoError(f"keyring malformed for owner '{_owner_file_id(owner_id)}'")
        return ring

    def _get_or_create_ring(self, owner_id: str) -> dict:
        """Idempotent + RACE-SICHER: zwei gleichzeitige Erst-Writes dürfen NICHT zwei verschiedene
        KEKs prägen. Ein bereits vorhandener Ring wird gelesen; sonst wird EINER via O_EXCL-artigem
        os.link (atomar, EEXIST wenn ein anderer gewann) geprägt -> der Verlierer nutzt den Gewinner."""
        existing = self._load_ring(owner_id)
        if existing is not None:
            return existing
        key = os.urandom(32)
        kid = _kid(key)
        ring = {"v": _RING_V, "active_kid": kid, "keys": {kid: _b64(key)}}
        final = self._ring_path(owner_id)
        fd, tmp = tempfile.mkstemp(dir=str(self._keystore), prefix=".ring-", suffix=".tmp")
        tmp_path = Path(tmp)
        try:
            os.write(fd, _canonical_json(ring))
            os.fsync(fd)
            os.close(fd)
            os.chmod(tmp, 0o600)
            # os.link ist atomar + wirft FileExistsError, wenn final schon da ist (Race verloren).
            try:
                os.link(tmp, final)
            except FileExistsError:
                won = self._load_ring(owner_id)
                if won is None:
                    raise ObjectStoreCryptoError("keyring vanished mid-create race")
                return won
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        _fsync_dir(self._keystore)
        return ring

    def _write_ring_atomic(self, owner_id: str, ring: dict) -> None:
        """Update eines EXISTIERENDEN Rings (Rotation) -- os.replace (overwrite) ist hier korrekt:
        kein Präge-Race (Ring existiert schon), und Rotation behält alte Keys -> kein Datenverlust."""
        final = self._ring_path(owner_id)
        fd, tmp = tempfile.mkstemp(dir=str(self._keystore), prefix=".ring-", suffix=".tmp")
        try:
            os.write(fd, _canonical_json(ring))
            os.fsync(fd)
            os.close(fd)
            os.chmod(tmp, 0o600)
            os.replace(tmp, final)
        except Exception:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
            raise
        _fsync_dir(self._keystore)

    # --- Envelope-Krypto ----------------------------------------------------
    def encrypt(self, plaintext: bytes, *, owner_id: str) -> dict:
        """Envelope-verschlüsselt plaintext für owner_id mit dem ACTIVE per-owner Key.
        Liefert {'envelope': <json>, 'key_ref': 'per_owner_domain:<owner_file_id>'}. key_ref ist die
        Crypto-Shred-Scope-Kennung für object_metadata.key_ref (kanonisches Vokabular); die konkrete
        kid steckt im Envelope."""
        if not isinstance(plaintext, (bytes, bytearray)):
            raise ObjectStoreCryptoError("plaintext must be bytes")
        ring = self._get_or_create_ring(owner_id)
        active_kid = ring["active_kid"]
        kek = _unb64(ring["keys"][active_kid])
        dek = os.urandom(32)
        data_nonce = os.urandom(12)
        aad = _canonical_json({"owner_id": owner_id, "kid": active_kid, "v": _ENVELOPE_V})
        ct = AESGCM(dek).encrypt(data_nonce, bytes(plaintext), aad)
        dek_nonce = os.urandom(12)
        wdk = AESGCM(kek).encrypt(dek_nonce, dek, aad)  # DEK-Wrap mit DERSELBEN aad
        envelope = {
            "v": _ENVELOPE_V, "alg": _ALG, "kid": active_kid, "owner_id": owner_id,
            "dn": _b64(data_nonce), "ct": _b64(ct), "kn": _b64(dek_nonce), "wdk": _b64(wdk),
        }
        return {
            "envelope": json.dumps(envelope, sort_keys=True, separators=(",", ":")),
            "key_ref": f"per_owner_domain:{_owner_file_id(owner_id)}",
        }

    def decrypt(self, envelope_json: str, *, owner_id: str) -> bytes:
        """Entschlüsselt einen Envelope für owner_id. Key/kid weg -> CryptoShreddedError
        (= gelöscht). owner-Mismatch/kaputt -> ObjectStoreCryptoError (fail-closed VOR Krypto).

        AKZEPTIERT-LOW (Review): shredded UND never-created liefern DENSELBEN CryptoShreddedError
        (keine Unterscheidung). Eine Ring-EXISTENZ-Enumeration (owner-mismatch vs no-keyring) bleibt
        theoretisch möglich, ist aber server-seitig (Aufrufer = trusted Vault-Pfad, KEIN externer
        Angreifer) + die Fehlermeldung trägt nur den gehashten owner-slug, NIE Key/Klartext/rohe owner_id."""
        ring = self._load_ring(owner_id)
        if ring is None:
            raise CryptoShreddedError(f"no keyring for owner '{_owner_file_id(owner_id)}' (shredded/never-created)")
        try:
            env = json.loads(envelope_json)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ObjectStoreCryptoError("envelope is not valid JSON") from exc
        if not isinstance(env, dict):
            raise ObjectStoreCryptoError("envelope is not a JSON object")
        if env.get("owner_id") != owner_id:
            raise ObjectStoreCryptoError("envelope owner_id mismatch")

        kid = env.get("kid")
        key_b64 = ring["keys"].get(kid) if isinstance(kid, str) else None
        if key_b64 is None:
            raise CryptoShreddedError(f"kid not in keyring for owner '{_owner_file_id(owner_id)}' (rotated-out/shredded)")
        kek = _unb64(key_b64)
        try:
            dn = _unb64(env["dn"]); ct = _unb64(env["ct"]); kn = _unb64(env["kn"]); wdk = _unb64(env["wdk"])
        except KeyError as exc:
            raise ObjectStoreCryptoError("envelope is missing a required field") from exc
        aad = _canonical_json({"owner_id": owner_id, "kid": kid, "v": env.get("v")})
        dek = AESGCM(kek).decrypt(kn, wdk, aad)  # InvalidTag (Tamper) propagiert roh
        return AESGCM(dek).decrypt(dn, ct, aad)

    # --- Rotation + Crypto-Shred -------------------------------------------
    def rotate(self, owner_id: str) -> str:
        """Prägt einen neuen active Key; alte Keys BLEIBEN im Ring (alte Chiffrate weiter lesbar).
        Neue Writes nutzen den neuen Key. Rückgabe: neue active_kid. Fehlt der Ring -> Fehler
        (ein geschredderter/nicht-existenter Owner wird NICHT still wiederbelebt)."""
        ring = self._load_ring(owner_id)
        if ring is None:
            raise ObjectStoreCryptoError(f"cannot rotate: no keyring for owner '{_owner_file_id(owner_id)}'")
        new_key = os.urandom(32)
        new_kid = _kid(new_key)
        ring["keys"][new_kid] = _b64(new_key)
        ring["active_kid"] = new_kid
        self._write_ring_atomic(owner_id, ring)
        return new_kid

    def crypto_shred(self, owner_id: str) -> bool:
        """CRYPTO-SHRED: löscht den GANZEN per-owner Ring (active + ALLE prev) via unlink -> jedes
        Chiffrat dieses Owners (live UND in bereits geschriebenen Kopien) wird unlesbar. Idempotent:
        schon geschreddert -> no-op Erfolg (False). True = eine Key-Datei wurde gelöscht.
        NB: das ist NICHT die physische Purge-Kaskade (Zeilen/Objekte/Platz) - die ist §5-App-Job."""
        path = self._ring_path(owner_id)
        try:
            os.remove(path)
        except FileNotFoundError:
            return False
        _fsync_dir(self._keystore)
        return True
