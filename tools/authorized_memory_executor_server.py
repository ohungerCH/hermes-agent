"""Replay-feste lokale Executor-Grenze fuer v2-bestaetigte Memory-Actions.

Das Modul ist kein Model-Tool und besitzt keine M6a-Schluessel. Authority
kommt ausschliesslich vom per ``SO_PEERCRED`` gebundenen Gate-Prozess. Der
Request bleibt trotzdem geschlossen und an den verifizierten Effect-Permit,
Execution-Claim, Scope, Ablauf und die exakten Params gebunden.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import stat
import struct
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from tools.authorized_memory_action import (
    AuthorizedMemoryActionError,
    apply_authorized_memory_action,
    parse_authorized_memory_action,
)
from tools.memory_tool import MemoryStore, load_on_disk_store


_REQUEST_KEYS = frozenset({
    "schema_version",
    "effect_permit_hash",
    "execution_claim_id",
    "execution_claim_hash",
    "idempotency_key",
    "expires_at",
    "request_id",
    "principal_id",
    "tenant_id",
    "owner_id",
    "workspace_id",
    "product_action_id",
    "skill_id",
    "capability_id",
    "action_id",
    "params_hash",
    "params",
})
_RESPONSE_KEYS = frozenset({
    "schema_version",
    "status",
    "request_hash",
    "effect_permit_hash",
    "execution_claim_id",
    "execution_claim_hash",
    "idempotency_key",
    "result",
})
_RESULT_COMMON_KEYS = frozenset({
    "schema_version",
    "operation",
    "status",
    "target",
})
_WRITE_RESULT_KEYS = (
    _RESULT_COMMON_KEYS
    | {"changed", "usage", "entry_count"}
)
_REJECTED_RESULT_KEYS = (
    _RESULT_COMMON_KEYS
    | {"changed", "reason"}
)
_RECALL_RESULT_KEYS = (
    _RESULT_COMMON_KEYS
    | {"available", "matches"}
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_HASH = re.compile(r"sha256-v1:[0-9a-f]{64}")
_MAX_REQUEST_BYTES = 36_864
_MAX_RESPONSE_BYTES = 40_960
_MAX_EXPIRY_SECONDS = 120
_BUSY_TIMEOUT_MS = 5000
_SCHEMA_VERSION = "2"


class AuthorizedMemoryExecutorError(ValueError):
    """Transport, Request, Journal oder Execution ist nicht exakt."""


def _fail(code: str) -> None:
    raise AuthorizedMemoryExecutorError(code)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuthorizedMemoryExecutorError(
            "memory_executor_json_invalid"
        ) from exc


def _sha256(value: Any) -> str:
    return "sha256-v1:" + hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        _fail(f"memory_executor_{field}_invalid")
    return value


def _hash(value: Any, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        _fail(f"memory_executor_{field}_invalid")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if type(value) is not str or len(value) > 64:
        _fail(f"memory_executor_{field}_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail(f"memory_executor_{field}_invalid")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"memory_executor_{field}_invalid")
    return parsed


def _parse_request(raw: Any, *, now: datetime) -> Dict[str, Any]:
    if (
        type(raw) is not dict
        or frozenset(raw) != _REQUEST_KEYS
        or any(type(key) is not str for key in raw)
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        _fail("memory_executor_request_invalid")
    expires_at = _timestamp(raw["expires_at"], "expires_at")
    lifetime = (expires_at - now).total_seconds()
    if lifetime > _MAX_EXPIRY_SECONDS:
        _fail("memory_executor_request_stale")
    parsed_params = parse_authorized_memory_action(raw["params"])
    skill_id = _identifier(raw["skill_id"], "skill_id")
    if (
        raw["schema_version"] != "jarvis.memory_executor.request.v1"
        or raw["product_action_id"] != "memory.manage"
        or raw["capability_id"] != "tool.memory"
        or raw["action_id"] != "invoke"
        or skill_id != f"skill.{parsed_params['skill_name']}"
    ):
        _fail("memory_executor_scope_invalid")
    result: Dict[str, Any] = {
        "schema_version": "jarvis.memory_executor.request.v1",
        "effect_permit_hash": _hash(
            raw["effect_permit_hash"],
            "effect_permit_hash",
        ),
        "execution_claim_id": _identifier(
            raw["execution_claim_id"],
            "execution_claim_id",
        ),
        "execution_claim_hash": _hash(
            raw["execution_claim_hash"],
            "execution_claim_hash",
        ),
        "idempotency_key": _identifier(
            raw["idempotency_key"],
            "idempotency_key",
        ),
        "expires_at": expires_at.isoformat(),
        "request_id": _identifier(raw["request_id"], "request_id"),
        "principal_id": _identifier(
            raw["principal_id"],
            "principal_id",
        ),
        "tenant_id": _identifier(raw["tenant_id"], "tenant_id"),
        "owner_id": _identifier(raw["owner_id"], "owner_id"),
        "workspace_id": _identifier(
            raw["workspace_id"],
            "workspace_id",
        ),
        "product_action_id": "memory.manage",
        "skill_id": skill_id,
        "capability_id": "tool.memory",
        "action_id": "invoke",
        "params_hash": _hash(raw["params_hash"], "params_hash"),
        "params": parsed_params,
    }
    if result["params_hash"] != _sha256({
        "action": "invoke",
        "params": parsed_params,
    }):
        _fail("memory_executor_params_hash_invalid")
    if len(canonical_json_bytes(result)) > _MAX_REQUEST_BYTES:
        _fail("memory_executor_request_oversized")
    return result


def _validate_result(value: Any) -> Dict[str, Any]:
    if (
        type(value) is not dict
        or value.get("schema_version")
        != "jarvis.memory_executor.result.v1"
        or value.get("operation")
        not in {"add", "replace", "remove", "batch", "recall"}
        or value.get("target") not in {"memory", "user"}
    ):
        _fail("memory_executor_result_invalid")
    status = value.get("status")
    keys = frozenset(value)
    if status == "rejected":
        if (
            keys != _REJECTED_RESULT_KEYS
            or value.get("changed") is not False
            or value.get("reason") not in {
                "external_drift",
                "content_rejected",
                "capacity_exceeded",
                "selector_not_found",
                "selector_ambiguous",
                "operation_rejected",
            }
        ):
            _fail("memory_executor_result_invalid")
    elif value["operation"] == "recall":
        if (
            status != "completed"
            or keys != _RECALL_RESULT_KEYS
            or type(value.get("available")) is not bool
            or type(value.get("matches")) is not list
            or len(value["matches"]) > 8
            or any(
                type(item) is not dict
                or frozenset(item) != {"source", "content"}
                or type(item.get("source")) is not str
                or not 1 <= len(item["source"]) <= 64
                or type(item.get("content")) is not str
                or not item["content"].startswith(
                    '<recalled_memory untrusted_data="true">'
                )
                or not item["content"].endswith("</recalled_memory>")
                or len(item["content"]) > 4096
                for item in value["matches"]
            )
            or (
                value["available"] is False
                and bool(value["matches"])
            )
        ):
            _fail("memory_executor_result_invalid")
    elif (
        status != "completed"
        or keys != _WRITE_RESULT_KEYS
        or type(value.get("changed")) is not bool
        or type(value.get("usage")) is not str
        or len(value["usage"]) > 64
        or type(value.get("entry_count")) is not int
        or not 0 <= value["entry_count"] <= 10_000
    ):
        _fail("memory_executor_result_invalid")
    if len(canonical_json_bytes(value)) > _MAX_RESPONSE_BYTES:
        _fail("memory_executor_result_oversized")
    return dict(value)


def _validate_response(value: Any) -> Dict[str, Any]:
    if (
        type(value) is not dict
        or frozenset(value) != _RESPONSE_KEYS
        or value.get("schema_version")
        != "jarvis.memory_executor.response.v1"
        or value.get("status") != "executed"
    ):
        _fail("memory_executor_response_invalid")
    for field in (
        "request_hash",
        "effect_permit_hash",
        "execution_claim_hash",
    ):
        _hash(value[field], field)
    for field in ("execution_claim_id", "idempotency_key"):
        _identifier(value[field], field)
    result = dict(value)
    result["result"] = _validate_result(value["result"])
    if len(canonical_json_bytes(result)) > _MAX_RESPONSE_BYTES:
        _fail("memory_executor_response_oversized")
    return result


class SqliteAuthorizedMemoryExecutionStore:
    """FULL/WAL-Journal: ein begonnenes unbekanntes Resultat wird nie reexecuted."""

    def __init__(self, path: Path) -> None:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or (
                path.exists()
                and (path.is_symlink() or not path.is_file())
            )
        ):
            _fail("memory_executor_store_path_invalid")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                _fail("memory_executor_store_mode_invalid")
        else:
            try:
                descriptor = os.open(
                    path,
                    os.O_CREAT
                    | os.O_EXCL
                    | os.O_RDWR
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o600,
                )
                os.close(descriptor)
            except OSError as exc:
                raise AuthorizedMemoryExecutorError(
                    "memory_executor_store_path_invalid"
                ) from exc
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            str(path),
            isolation_level=None,
            check_same_thread=False,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        self._db.row_factory = sqlite3.Row
        try:
            self._configure()
        except Exception:
            self._db.close()
            raise

    def _configure(self) -> None:
        with self._lock:
            journal = self._db.execute(
                "PRAGMA journal_mode=WAL"
            ).fetchone()[0]
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if (
                str(journal).lower() != "wal"
                or self._db.execute(
                    "PRAGMA synchronous"
                ).fetchone()[0] != 2
            ):
                _fail("memory_executor_store_profile_invalid")
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_executor_meta_v1 (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS memory_executor_runs_v1 (
                    idempotency_key TEXT PRIMARY KEY,
                    request_hash TEXT UNIQUE NOT NULL,
                    effect_permit_hash TEXT UNIQUE NOT NULL,
                    execution_claim_id TEXT UNIQUE NOT NULL,
                    execution_claim_hash TEXT UNIQUE NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('started','completed')),
                    response_json TEXT,
                    response_hash TEXT,
                    CHECK(
                        (state='started' AND response_json IS NULL
                         AND response_hash IS NULL)
                        OR
                        (state='completed' AND response_json IS NOT NULL
                         AND response_hash IS NOT NULL)
                    )
                ) WITHOUT ROWID;
                """
            )
            row = self._db.execute(
                "SELECT value FROM memory_executor_meta_v1 "
                "WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                if self._db.execute(
                    "SELECT COUNT(*) FROM memory_executor_runs_v1"
                ).fetchone()[0]:
                    _fail("memory_executor_store_schema_invalid")
                self._db.execute(
                    "INSERT INTO memory_executor_meta_v1(key,value) "
                    "VALUES('schema_version',?)",
                    (_SCHEMA_VERSION,),
                )
            elif row[0] != _SCHEMA_VERSION:
                _fail("memory_executor_store_schema_invalid")
            for stored in self._db.execute(
                "SELECT * FROM memory_executor_runs_v1"
            ).fetchall():
                self._validate_row(stored)

    @staticmethod
    def _validate_row(row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        _identifier(row["idempotency_key"], "idempotency_key")
        _hash(row["request_hash"], "request_hash")
        _hash(row["effect_permit_hash"], "effect_permit_hash")
        _identifier(row["execution_claim_id"], "execution_claim_id")
        _hash(row["execution_claim_hash"], "execution_claim_hash")
        if row["state"] == "started":
            if (
                row["response_json"] is not None
                or row["response_hash"] is not None
            ):
                _fail("memory_executor_store_row_invalid")
            return None
        if row["state"] != "completed":
            _fail("memory_executor_store_row_invalid")
        try:
            response = json.loads(row["response_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AuthorizedMemoryExecutorError(
                "memory_executor_store_row_invalid"
            ) from exc
        if (
            canonical_json_bytes(response).decode("utf-8")
            != row["response_json"]
            or _sha256(response) != row["response_hash"]
            or response.get("request_hash") != row["request_hash"]
            or response.get("idempotency_key")
            != row["idempotency_key"]
            or response.get("effect_permit_hash")
            != row["effect_permit_hash"]
            or response.get("execution_claim_id")
            != row["execution_claim_id"]
            or response.get("execution_claim_hash")
            != row["execution_claim_hash"]
        ):
            _fail("memory_executor_store_row_invalid")
        return _validate_response(response)

    def _find_locked(
        self,
        request: Mapping[str, Any],
        request_hash: str,
    ) -> Optional[sqlite3.Row]:
        row = self._db.execute(
            "SELECT * FROM memory_executor_runs_v1 "
            "WHERE idempotency_key=? OR request_hash=? "
            "OR effect_permit_hash=? OR execution_claim_id=? "
            "OR execution_claim_hash=?",
            (
                request["idempotency_key"],
                request_hash,
                request["effect_permit_hash"],
                request["execution_claim_id"],
                request["execution_claim_hash"],
            ),
        ).fetchone()
        if row is not None and (
            row["idempotency_key"] != request["idempotency_key"]
            or row["request_hash"] != request_hash
            or row["effect_permit_hash"]
            != request["effect_permit_hash"]
            or row["execution_claim_id"] != request["execution_claim_id"]
            or row["execution_claim_hash"]
            != request["execution_claim_hash"]
        ):
            _fail("memory_executor_idempotency_conflict")
        return row

    def acquire(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        request_hash = _sha256(request)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._find_locked(request, request_hash)
                if row is None:
                    self._db.execute(
                        "INSERT INTO memory_executor_runs_v1("
                        "idempotency_key,request_hash,effect_permit_hash,"
                        "execution_claim_id,execution_claim_hash,state"
                        ") VALUES(?,?,?,?,?, 'started')",
                        (
                            request["idempotency_key"],
                            request_hash,
                            request["effect_permit_hash"],
                            request["execution_claim_id"],
                            request["execution_claim_hash"],
                        ),
                    )
                    state = ("new", request_hash, None)
                else:
                    response = self._validate_row(row)
                    state = (row["state"], request_hash, response)
                self._db.execute("COMMIT")
                return state
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def lookup(
        self,
        request: Mapping[str, Any],
    ) -> Optional[tuple[str, str, Optional[Dict[str, Any]]]]:
        request_hash = _sha256(request)
        with self._lock:
            row = self._find_locked(request, request_hash)
            if row is None:
                return None
            return (
                row["state"],
                request_hash,
                self._validate_row(row),
            )

    def begin(self, request: Mapping[str, Any]) -> str:
        """Test-/Recovery-Naht zum durable Markieren vor dem Effect."""
        state, request_hash, _response = self.acquire(request)
        if state == "completed":
            _fail("memory_executor_execution_completed")
        return request_hash

    def finish(
        self,
        request: Mapping[str, Any],
        request_hash: str,
        response: Mapping[str, Any],
    ) -> None:
        parsed = _validate_response(response)
        if (
            parsed["request_hash"] != request_hash
            or parsed["idempotency_key"] != request["idempotency_key"]
        ):
            _fail("memory_executor_response_binding_invalid")
        encoded = canonical_json_bytes(parsed).decode("utf-8")
        response_hash = _sha256(parsed)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._find_locked(request, request_hash)
                if row is None or row["state"] != "started":
                    _fail("memory_executor_store_transition_invalid")
                self._db.execute(
                    "UPDATE memory_executor_runs_v1 "
                    "SET state='completed',response_json=?,response_hash=? "
                    "WHERE idempotency_key=? AND state='started'",
                    (
                        encoded,
                        response_hash,
                        request["idempotency_key"],
                    ),
                )
                if self._db.execute(
                    "SELECT changes()"
                ).fetchone()[0] != 1:
                    _fail("memory_executor_store_transition_invalid")
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute(
                "SELECT COUNT(*) FROM memory_executor_runs_v1"
            ).fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


class AuthorizedMemoryExecutor:
    """Fuehrt genau eine Gate-gebundene Action mit durable Idempotenz aus."""

    def __init__(
        self,
        journal: SqliteAuthorizedMemoryExecutionStore,
        *,
        store_loader: Callable[[], MemoryStore] = load_on_disk_store,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        if (
            type(journal) is not SqliteAuthorizedMemoryExecutionStore
            or not callable(store_loader)
            or not callable(clock)
        ):
            _fail("memory_executor_config_invalid")
        self._journal = journal
        self._store_loader = store_loader
        self._clock = clock

    def execute(self, raw: Any) -> Dict[str, Any]:
        now = self._clock()
        try:
            request = _parse_request(raw, now=now)
        except AuthorizedMemoryActionError as exc:
            raise AuthorizedMemoryExecutorError(
                "memory_executor_params_invalid"
            ) from exc
        existing = self._journal.lookup(request)
        if existing is not None:
            state, _request_hash, response = existing
            if state == "completed" and response is not None:
                return response
            _fail("memory_executor_execution_indeterminate")
        if _timestamp(request["expires_at"], "expires_at") <= now:
            _fail("memory_executor_request_stale")
        state, request_hash, response = self._journal.acquire(request)
        if state == "completed" and response is not None:
            return response
        if state != "new":
            _fail("memory_executor_execution_indeterminate")

        memory = self._store_loader()
        if type(memory) is not MemoryStore:
            _fail("memory_executor_memory_store_invalid")
        from tools.skill_provenance import (
            reset_current_write_origin,
            set_current_write_origin,
        )
        from tools.vault.vault_wiring import (
            reset_vault_write_identity,
            set_vault_write_identity,
        )

        origin_context = set_current_write_origin("assistant_tool")
        identity_context = set_vault_write_identity(
            request["tenant_id"],
            request["owner_id"],
        )
        try:
            result = apply_authorized_memory_action(
                request["params"],
                store=memory,
            )
        finally:
            reset_vault_write_identity(identity_context)
            reset_current_write_origin(origin_context)
        response = {
            "schema_version": "jarvis.memory_executor.response.v1",
            "status": "executed",
            "request_hash": request_hash,
            "effect_permit_hash": request["effect_permit_hash"],
            "execution_claim_id": request["execution_claim_id"],
            "execution_claim_hash": request["execution_claim_hash"],
            "idempotency_key": request["idempotency_key"],
            "result": _validate_result(result),
        }
        self._journal.finish(request, request_hash, response)
        return response


def _read_request(connection: socket.socket) -> Any:
    chunks = bytearray()
    while len(chunks) <= _MAX_REQUEST_BYTES:
        piece = connection.recv(4096)
        if not piece:
            break
        chunks.extend(piece)
    if (
        not chunks
        or len(chunks) > _MAX_REQUEST_BYTES
        or chunks[-1:] != b"\n"
        or b"\n" in chunks[:-1]
    ):
        _fail("memory_executor_request_invalid")
    try:
        return json.loads(bytes(chunks[:-1]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizedMemoryExecutorError(
            "memory_executor_request_invalid"
        ) from exc


def _peer_uid(connection: socket.socket) -> int:
    try:
        credentials = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        _pid, uid, _gid = struct.unpack("3i", credentials)
        return uid
    except (AttributeError, OSError, struct.error) as exc:
        raise AuthorizedMemoryExecutorError(
            "memory_executor_peer_invalid"
        ) from exc


def serve_memory_executor_connection(
    *,
    connection: socket.socket,
    allowed_peer_uid: int,
    executor: AuthorizedMemoryExecutor,
) -> None:
    """Bedient genau einen EOF-terminierten Request und schliesst die Verbindung."""

    with connection:
        try:
            if (
                type(allowed_peer_uid) is not int
                or allowed_peer_uid < 0
                or _peer_uid(connection) != allowed_peer_uid
                or type(executor) is not AuthorizedMemoryExecutor
            ):
                _fail("memory_executor_peer_invalid")
            response = executor.execute(_read_request(connection))
        except (
            AuthorizedMemoryActionError,
            AuthorizedMemoryExecutorError,
            OSError,
            TypeError,
            ValueError,
        ):
            response = {
                "schema_version": "jarvis.memory_executor.error.v1",
                "status": "deny",
                "reason": "memory_executor_request_failed",
            }
        try:
            connection.sendall(canonical_json_bytes(response))
            connection.shutdown(socket.SHUT_WR)
            # Bei einem vor dem Read abgewiesenen Peer liegen noch Request-
            # Bytes im Empfangspuffer. Begrenzt bis zum bereits verlangten
            # EOF leeren, damit die generische Deny-Antwort nicht durch ein
            # TCP-artiges RST der Unix-Stream-Verbindung verloren geht.
            connection.settimeout(0.1)
            remaining = _MAX_REQUEST_BYTES + 1
            while remaining > 0:
                chunk = connection.recv(min(4096, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass


class AuthorizedMemoryExecutorUnixServer:
    """Kleiner API-prozesseigener Listener fuer den lokalen Gate-Peer."""

    def __init__(
        self,
        *,
        socket_path: Path,
        journal_path: Path,
        allowed_peer_uid: int,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        if (
            not isinstance(socket_path, Path)
            or not socket_path.is_absolute()
            or len(str(socket_path).encode("utf-8")) > 100
            or not isinstance(journal_path, Path)
            or not journal_path.is_absolute()
            or socket_path == journal_path
            or type(allowed_peer_uid) is not int
            or allowed_peer_uid < 0
            or not callable(clock)
        ):
            _fail("memory_executor_server_config_invalid")
        self._socket_path = socket_path
        self._journal_path = journal_path
        self._allowed_peer_uid = allowed_peer_uid
        self._clock = clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._listener: Optional[socket.socket] = None
        self._journal: Optional[
            SqliteAuthorizedMemoryExecutionStore
        ] = None
        self._executor: Optional[AuthorizedMemoryExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._socket_identity: Optional[tuple[int, int]] = None

    def _prepare_socket_path(self) -> None:
        parent = self._socket_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            _fail("memory_executor_socket_path_busy")
        try:
            metadata = self._socket_path.lstat()
        except FileNotFoundError:
            return
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            _fail("memory_executor_socket_path_busy")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.1)
            probe.connect(str(self._socket_path))
        except (ConnectionRefusedError, FileNotFoundError):
            self._socket_path.unlink()
            return
        except OSError:
            _fail("memory_executor_socket_path_busy")
        finally:
            probe.close()
        _fail("memory_executor_socket_path_busy")

    def _run(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            executor = self._executor
            if listener is None or executor is None:
                return
            try:
                connection, _address = listener.accept()
                connection.settimeout(5)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            serve_memory_executor_connection(
                connection=connection,
                allowed_peer_uid=self._allowed_peer_uid,
                executor=executor,
            )

    def start(self) -> None:
        with self._lock:
            if self._listener is not None:
                _fail("memory_executor_server_already_running")
            self._prepare_socket_path()
            journal: Optional[
                SqliteAuthorizedMemoryExecutionStore
            ] = None
            listener: Optional[socket.socket] = None
            try:
                journal = SqliteAuthorizedMemoryExecutionStore(
                    self._journal_path
                )
                executor = AuthorizedMemoryExecutor(
                    journal,
                    clock=self._clock,
                )
                listener = socket.socket(
                    socket.AF_UNIX,
                    socket.SOCK_STREAM,
                )
                listener.bind(str(self._socket_path))
                os.chmod(self._socket_path, 0o660)
                listener.listen(4)
                listener.settimeout(0.2)
                metadata = self._socket_path.lstat()
                if (
                    not stat.S_ISSOCK(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o660
                ):
                    _fail("memory_executor_socket_invalid")
                self._socket_identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                )
                self._stop.clear()
                self._journal = journal
                self._executor = executor
                self._listener = listener
                self._thread = threading.Thread(
                    target=self._run,
                    name="jarvis-memory-executor",
                    daemon=True,
                )
                self._thread.start()
            except Exception:
                if listener is not None:
                    listener.close()
                if journal is not None:
                    journal.close()
                try:
                    metadata = self._socket_path.lstat()
                    if (
                        stat.S_ISSOCK(metadata.st_mode)
                        and metadata.st_uid == os.geteuid()
                    ):
                        self._socket_path.unlink()
                except FileNotFoundError:
                    pass
                raise

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            listener = self._listener
            thread = self._thread
            journal = self._journal
            identity = self._socket_identity
            self._listener = None
            self._thread = None
            self._executor = None
            self._journal = None
            self._socket_identity = None
            if listener is not None:
                listener.close()
        if thread is not None:
            thread.join(timeout=2)
        if journal is not None:
            journal.close()
        if identity is not None:
            try:
                metadata = self._socket_path.lstat()
                if (
                    stat.S_ISSOCK(metadata.st_mode)
                    and (metadata.st_dev, metadata.st_ino) == identity
                ):
                    self._socket_path.unlink()
            except FileNotFoundError:
                pass

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "running": (
                    self._listener is not None
                    and self._thread is not None
                    and self._thread.is_alive()
                ),
                "socket_path": str(self._socket_path),
            }


__all__ = [
    "AuthorizedMemoryExecutor",
    "AuthorizedMemoryExecutorError",
    "AuthorizedMemoryExecutorUnixServer",
    "SqliteAuthorizedMemoryExecutionStore",
    "canonical_json_bytes",
    "serve_memory_executor_connection",
]
